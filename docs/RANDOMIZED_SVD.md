# Randomized SVD for SSA Decomposition

The fastest decomposition method. Based on Halko, Martinsson & Tropp (2011).

## The Problem

SSA needs the top-k singular triplets of an L×K Hankel matrix H:

```
H ≈ U·Σ·Vᵀ = Σᵢ σᵢ·uᵢ·vᵢᵀ    (i = 0, ..., k-1)
```

Full SVD is O(min(L,K)·L·K) ≈ O(N³). We need O(k·N log N).

## Algorithm Overview

```
1. Random projection:  Y = H·Ω           Ω is K×(k+p) Gaussian random
2. Orthonormalize:     Q = qr(Y)         Q is L×(k+p) orthonormal
3. Project to small:   B = Qᵀ·H          B is (k+p)×K
4. SVD of small:       B = Uᵦ·Σ·Vᵀ       Standard LAPACK
5. Recover U:          U = Q·Uᵦ          Rotate back to original space
```

**Key insight**: Random projection Y captures the column space of H with high probability. The approximation error depends on σₖ₊₁ (the first neglected singular value).

## Step-by-Step Implementation

### Step 1: Random Projection

```c
// Generate random K×(k+p) matrix Ω
vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, ssa->rng, K * kp, Omega, 0.0, 1.0);

// Y = H·Ω  (L×kp matrix)
// Each column of Y is a Hankel matvec
for (int j = 0; j < kp; j++) {
    ssa_opt_hankel_matvec(ssa, &Omega[j * K], &Y[j * L]);
}
```

Why Gaussian? Any random matrix with independent entries works. Gaussian has nice theoretical guarantees and MKL generates it fast.

**Oversampling p**: We compute k+p columns, then truncate to k. Typical p=5-10 improves accuracy at negligible cost. Default p=8.

### Step 2: Orthonormalize via QR

```c
// Q, tau = qr(Y)
// Q is stored in-place in Y, tau holds Householder reflectors
LAPACKE_dgeqrf(LAPACK_COL_MAJOR, L, kp, Y, L, tau);

// Extract explicit Q matrix (first kp columns of the implicit Q)
LAPACKE_dorgqr(LAPACK_COL_MAJOR, L, kp, kp, Y, L, tau);
// Now Y contains Q
double *Q = Y;
```

Why QR? Gram-Schmidt is numerically unstable. Householder QR is O(L·kp²) and backward stable.

### Step 3: Project to Small Matrix

```c
// B = Qᵀ·H  where B is kp×K
// Equivalent to: B[i,:] = Hᵀ·Q[:,i] for each row i

// Use adjoint Hankel matvec
for (int i = 0; i < kp; i++) {
    ssa_opt_hankel_matvec_T(ssa, &Q[i * L], &B[i * K]);
}
```

Now B is a small (k+p)×K matrix that captures the essential structure of H.

### Step 4: SVD of Small Matrix

```c
// B = Uᵦ·Σ·Vᵀ via divide-and-conquer SVD
LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S',  // 'S' = economy SVD
               kp, K,                   // dimensions
               B, kp,                   // input matrix
               S,                       // singular values (output)
               U_small, kp,             // left singular vectors
               Vt, kp,                  // right singular vectors (transposed)
               ...);
```

GESDD (divide-and-conquer) is faster than GESVD for small-to-medium matrices.

### Step 5: Recover Final U

```c
// U_final = Q · U_small
// Q is L×kp, U_small is kp×k, result is L×k
cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            L, k, kp,           // dimensions
            1.0, Q, L,          // Q
            U_small, kp,        // U_small
            0.0, U_final, L);   // output
```

The right singular vectors V come directly from step 4 (first k columns of Vᵀ transposed).

## Malloc-Free Hot Path

For streaming applications, we pre-allocate all workspace:

```c
int ssa_opt_prepare(SSA_Opt *ssa, int max_k, int oversampling) {
    int kp = max_k + oversampling;
    
    // Pre-allocate all matrices
    ssa->decomp_Omega = mkl_malloc(K * kp * sizeof(double), 64);      // Random matrix
    ssa->decomp_Y = mkl_malloc(L * kp * sizeof(double), 64);          // Y = H·Ω
    ssa->decomp_Q = mkl_malloc(L * kp * sizeof(double), 64);          // Q from QR
    ssa->decomp_B = mkl_malloc(kp * K * sizeof(double), 64);          // B = Qᵀ·H
    ssa->decomp_tau = mkl_malloc(kp * sizeof(double), 64);            // QR reflectors
    ssa->decomp_S = mkl_malloc(kp * sizeof(double), 64);              // Singular values
    ssa->decomp_work = mkl_malloc(lwork * sizeof(double), 64);        // LAPACK workspace
    ssa->decomp_iwork = mkl_malloc(8 * kp * sizeof(int), 64);         // GESDD iwork
    
    ssa->prepared = true;
}
```

After `prepare()`, decomposition does **zero allocations**:

```c
int ssa_opt_decompose_randomized(SSA_Opt *ssa, int k, int oversampling) {
    // All workspace comes from pre-allocated buffers
    double *Omega = ssa->decomp_Omega;
    double *Y = ssa->decomp_Y;
    // ... etc
    
    // No malloc() calls in the entire function
}
```

## Sign and Ordering Conventions

SSA convention requires:
1. Singular values in **descending** order
2. First element of each left singular vector is **positive** (for reproducibility)

```c
// Sort by descending singular value
for (int i = 0; i < k - 1; i++) {
    for (int j = i + 1; j < k; j++) {
        if (sigma[j] > sigma[i]) {
            // Swap sigma[i] ↔ sigma[j]
            // Swap U[:,i] ↔ U[:,j]
            // Swap V[:,i] ↔ V[:,j]
        }
    }
}

// Fix sign: ensure U[0,i] > 0
for (int i = 0; i < k; i++) {
    double sum = 0;
    for (int t = 0; t < L; t++) sum += U[i * L + t];
    if (sum < 0) {
        cblas_dscal(L, -1.0, &U[i * L], 1);  // Flip U[:,i]
        cblas_dscal(K, -1.0, &V[i * K], 1);  // Flip V[:,i] to maintain U·Vᵀ
    }
}
```

## Complexity Analysis

| Step | Operations | Complexity |
|------|------------|------------|
| 1. Random projection | kp Hankel matvecs | O(kp · N log N) |
| 2. QR factorization | Householder | O(L · kp²) |
| 3. Project to B | kp adjoint matvecs | O(kp · N log N) |
| 4. Small SVD | GESDD | O(kp² · K) |
| 5. Recover U | GEMM | O(L · kp · k) |

**Total: O(kp · N log N)** dominated by Hankel matvecs.

Compare to power iteration: O(k · iterations · N log N) where iterations ≈ 50-100.

Randomized SVD is **5-10× faster** for typical k=20-50.

## When to Use

| Condition | Recommended Method |
|-----------|-------------------|
| k < min(L,K)/4 | **Randomized** (fastest) |
| k > min(L,K)/4 | Block (more stable) |
| Need exact comparison with Rssa | Sequential |
| Streaming/real-time | Randomized + prepare() |

## Error Bounds

With oversampling p and power iterations q=0:

```
E[||H - Q·Qᵀ·H||] ≤ (1 + √(k/(p-1))) · σₖ₊₁ + (e√(k+p)/p) · Σⱼ₌ₖ₊₁ σⱼ²
```

For SSA signals with fast singular value decay, this error is tiny.

## References

- Halko, N., Martinsson, P.G., & Tropp, J.A. (2011). "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." SIAM Review, 53(2), 217-288.
