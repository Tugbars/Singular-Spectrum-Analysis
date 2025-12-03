# Hankel Matrix-Vector Product via FFT

The core primitive of SSA-Opt. All decomposition methods build on this.

## Role in SSA

SSA works by embedding a 1D signal into a 2D trajectory matrix, then computing its SVD. The trajectory matrix H is a **Hankel matrix** - constant along anti-diagonals:

```
Signal: x = [x₀, x₁, x₂, x₃, x₄, x₅, x₆]    (N=7)
Window: L = 4
Columns: K = N - L + 1 = 4

Trajectory Matrix H (L×K = 4×4):

        j=0   j=1   j=2   j=3
       ┌─────────────────────┐
  i=0  │ x₀    x₁    x₂    x₃ │
  i=1  │ x₁    x₂    x₃    x₄ │
  i=2  │ x₂    x₃    x₄    x₅ │
  i=3  │ x₃    x₄    x₅    x₆ │
       └─────────────────────┘

H[i,j] = x[i+j]
```

We never form H explicitly (O(N²) memory). Instead, we compute matrix-vector products H·v and Hᵀ·u implicitly.

## The Problem

SSA decomposition requires many matrix-vector products:

```
y = H·v      (forward)   - used in power iteration
z = Hᵀ·u     (adjoint)   - used in power iteration
```

Direct multiplication is O(L·K) ≈ O(N²) per product. With hundreds of iterations, this dominates runtime.

## Key Insight: Hankel = Convolution

The forward product y = H·v expands to:

```
y[i] = Σⱼ H[i,j]·v[j] = Σⱼ x[i+j]·v[j]
```

This is a **convolution** of x with v, sampled at offset positions:

```
y = (x ∗ v)[i : i+L]    where i ∈ [0, L-1]
```

Similarly, the adjoint Hᵀ·u is convolution with a **reversed** vector:

```
z[j] = Σᵢ x[i+j]·u[i] = (x ∗ flip(u))[L-1+j : L-1+j+K]
```

## FFT Convolution

Convolution theorem: `conv(a,b) = IFFT(FFT(a) ⊙ FFT(b))`

Since the signal x is fixed, we **pre-compute FFT(x)** once at initialization.

Each matvec then costs:
- 1 FFT of the input vector
- 1 pointwise complex multiply
- 1 IFFT
- 1 memcpy to extract the result slice

**Complexity: O(N²) → O(N log N)**

## Implementation

```c
// Forward: y = H·v, where y is length L, v is length K
static void ssa_opt_hankel_matvec(SSA_Opt *ssa, const double *v, double *y) {
    // 1. Zero-pad and reverse v into workspace
    ssa_opt_zero(ssa->ws_real, fft_len);
    ssa_opt_reverse_copy(v, ssa->ws_real, K);  // ws_real = [v[K-1], ..., v[0], 0, ...]
    
    // 2. FFT of reversed v
    DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);
    
    // 3. Pointwise multiply with pre-computed FFT(x)
    //    ws_complex = FFT(x) ⊙ FFT(flip(v))
    ssa_opt_complex_mul_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len);
    
    // 4. IFFT to get convolution result
    DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);
    
    // 5. Extract y from position [K-1, K-1+L)
    //    This is where the valid convolution output lives
    memcpy(y, ssa->ws_real + (K - 1), L * sizeof(double));
}
```

### Why Reverse?

The convolution `x ∗ v` computes `Σⱼ x[k-j]·v[j]`. But we need `Σⱼ x[i+j]·v[j]`.

Reversing v converts convolution to **correlation**, giving us the Hankel product.

### Adjoint (Transpose)

```c
// Adjoint: z = Hᵀ·u, where z is length K, u is length L
static void ssa_opt_hankel_matvec_T(SSA_Opt *ssa, const double *u, double *z) {
    ssa_opt_zero(ssa->ws_real, fft_len);
    ssa_opt_reverse_copy(u, ssa->ws_real, L);  // Reverse u
    
    DftiComputeForward(ssa->fft_r2c, ssa->ws_real, ssa->ws_complex);
    ssa_opt_complex_mul_r2c(ssa->fft_x, ssa->ws_complex, ssa->ws_complex, r2c_len);
    DftiComputeBackward(ssa->fft_c2r, ssa->ws_complex, ssa->ws_real);
    
    // Extract from different offset for adjoint
    memcpy(z, ssa->ws_real + (L - 1), K * sizeof(double));
}
```

## Batched Version

For block methods, we process b vectors simultaneously:

```c
static void ssa_opt_hankel_matvec_block(SSA_Opt *ssa, const double *V_block, 
                                         double *Y_block, int b) {
    // Process in batches of SSA_BATCH_SIZE (default 32)
    while (col < b) {
        int batch_count = min(SSA_BATCH_SIZE, b - col);
        
        // 1. Pack reversed vectors into contiguous memory
        for (int j = 0; j < batch_count; j++) {
            ssa_opt_reverse_copy(&V_block[(col + j) * K], &ws_batch_real[j * fft_len], K);
        }
        
        // 2. Batched FFT (single MKL call for all 32 vectors)
        DftiComputeForward(fft_r2c_batch, ws_batch_real, ws_batch_complex);
        
        // 3. Pointwise multiply each with FFT(x)
        for (int j = 0; j < batch_count; j++) {
            ssa_opt_complex_mul_r2c(fft_x, &ws_batch_complex[j * 2 * r2c_len], 
                                    &ws_batch_complex[j * 2 * r2c_len], r2c_len);
        }
        
        // 4. Batched IFFT
        DftiComputeBackward(fft_c2r_batch, ws_batch_complex, ws_batch_real);
        
        // 5. Extract results
        for (int j = 0; j < batch_count; j++) {
            memcpy(&Y_block[(col + j) * L], &ws_batch_real[j * fft_len + K - 1], 
                   L * sizeof(double));
        }
        col += batch_count;
    }
}
```

### Why Batch?

| Approach | FFT Calls | MKL Overhead |
|----------|-----------|--------------|
| Sequential (b vectors) | 2b | High (per-call setup) |
| Batched (b/32 batches) | 2·⌈b/32⌉ | Low (amortized) |

MKL batched FFT also exploits SIMD better across multiple transforms.

## R2C Optimization

Real-to-Complex FFT exploits Hermitian symmetry:

```
FFT of real signal: X[k] = conj(X[N-k])
```

Only need to store N/2+1 complex values instead of N.

- **50% less memory** for FFT buffers
- **~17% faster** than C2C FFT

## Memory Layout

```
ws_real:     [fft_len doubles]     - Real workspace (zero-padded)
ws_complex:  [2 * r2c_len doubles] - Complex workspace (interleaved re/im)
fft_x:       [2 * r2c_len doubles] - Pre-computed FFT of signal (persistent)
```

Pre-computing `fft_x` at init saves one FFT per matvec.

## Complexity Summary

| Operation | Direct | FFT-based |
|-----------|--------|-----------|
| Single matvec | O(L·K) ≈ O(N²) | O(N log N) |
| b matvecs | O(b·N²) | O(b·N log N) |
| With batching | - | Lower constant factor |

For N=10000, L=2500: Direct ≈ 6.25M ops, FFT ≈ 130K ops. **~50× speedup**.
