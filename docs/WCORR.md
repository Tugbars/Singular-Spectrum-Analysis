# W-Correlation Matrix: Fast Computation

W-correlation measures separability between SSA components. Essential for grouping.

## Definition

The weighted correlation between components i and j:

```
W(i,j) = ⟨Xᵢ, Xⱼ⟩_w / (||Xᵢ||_w · ||Xⱼ||_w)
```

where:
- Xᵢ = σᵢ · uᵢ · vᵢᵀ is the rank-1 elementary matrix for component i
- ⟨·,·⟩_w is the weighted inner product with weights w[t] = min(t+1, L, K, N-t)
- ||·||_w is the corresponding weighted norm

The weights w[t] count how many times position t appears in the trajectory matrix (diagonal averaging weights).

## Naive Approach: O(n² · N)

```python
# Pseudocode - DO NOT USE
for i in range(n):
    h_i = reconstruct(component=i)  # O(N log N) FFT
    for j in range(i, n):
        h_j = reconstruct(component=j)  # O(N log N) FFT
        W[i,j] = weighted_correlation(h_i, h_j, weights)  # O(N)
```

For n=50 components: 1275 pairs × 2 reconstructions × O(N log N) = very slow.

## Key Insight: Factor Out Common Structure

Each component reconstruction hᵢ = IFFT(FFT(σᵢ·uᵢ) ⊙ FFT(vᵢ)).

If we **cache the FFTs** of U and V at decomposition time:

```c
// At decomposition (once):
for (int i = 0; i < n; i++) {
    U_fft[i] = FFT(sigma[i] * U[:,i])
    V_fft[i] = FFT(V[:,i])
}
```

Then reconstruction becomes just:
```c
h[i] = IFFT(U_fft[i] ⊙ V_fft[i])  // No forward FFT needed
```

## Better Insight: DSYRK

The W-correlation matrix W is symmetric. We can compute it as:

```
W = G · Gᵀ    (approximately)
```

where G is a normalized version of all reconstructions stacked.

**BLAS3 DSYRK** computes symmetric rank-k update in one highly optimized call.

## Implementation: ssa_opt_wcorr_matrix_fast

### Step 1: Batch All Complex Multiplies

```c
// Compute FFT(u_i) ⊙ FFT(v_i) for all components in parallel
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    double *u_fft = &U_fft[i * 2 * r2c_len];
    double *v_fft = &V_fft[i * 2 * r2c_len];
    double *ws = &all_ws_complex[i * 2 * r2c_len];
    
    // Complex multiply (MKL vzMul under the hood)
    for (int k = 0; k < r2c_len; k++) {
        double ar = u_fft[2*k], ai = u_fft[2*k+1];
        double br = v_fft[2*k], bi = v_fft[2*k+1];
        ws[2*k] = ar*br - ai*bi;
        ws[2*k+1] = ar*bi + ai*br;
    }
}
```

### Step 2: Batched IFFT

```c
// Single MKL call: IFFT all n frequency-domain products simultaneously
DftiComputeBackward(ssa->fft_c2r_wcorr, all_ws_complex, all_h);
```

This uses a special FFT descriptor configured for n simultaneous transforms:

```c
// At prepare() time:
MKL_LONG n_transforms = n;
DftiCreateDescriptor(&fft_c2r_wcorr, DFTI_DOUBLE, DFTI_REAL, 1, fft_len);
DftiSetValue(fft_c2r_wcorr, DFTI_NUMBER_OF_TRANSFORMS, n_transforms);
DftiSetValue(fft_c2r_wcorr, DFTI_INPUT_DISTANCE, 2 * r2c_len);
DftiSetValue(fft_c2r_wcorr, DFTI_OUTPUT_DISTANCE, fft_len);
DftiCommit(fft_c2r_wcorr);
```

### Step 3: Normalize and Weight

```c
// Pre-computed: sqrt_inv_c[t] = sqrt(1 / count[t])
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    double *h = &all_h[i * fft_len];
    double sigma = ssa->sigma[i];
    
    // Compute weighted norm: ||h||_w² = Σ h[t]² · w[t]
    double norm_sq = ssa_opt_weighted_norm_sq(h, inv_diag_count, N);
    norm_sq *= sigma * sigma;
    double norm = sqrt(norm_sq);
    
    // Normalize: g[i,t] = (σᵢ / ||hᵢ||_w) · hᵢ[t] · √w[t]
    double scale = (norm > 1e-12) ? sigma / norm : 0.0;
    double *g_row = &G[i * N];
    ssa_opt_scale_weighted(h, sqrt_inv_c, scale, g_row, N);  // AVX2 optimized
}
```

Now G is an n×N matrix where each row is normalized and pre-weighted.

### Step 4: DSYRK for Gram Matrix

```c
// W = G · Gᵀ (upper triangle)
cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
            n,      // rows of G (number of components)
            N,      // cols of G (signal length)
            1.0,    // alpha
            G, N,   // G matrix, leading dimension
            0.0,    // beta
            W, n);  // output W, leading dimension

// Mirror to lower triangle
for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
        W[j * n + i] = W[i * n + j];
    }
}
```

DSYRK is **much faster** than computing n² dot products because:
- BLAS3 operation (cache-optimal blocking)
- Only computes upper triangle (n(n+1)/2 instead of n²)
- Highly optimized in MKL

## Complexity Comparison

| Method | FFTs | Dot Products | Total |
|--------|------|--------------|-------|
| Naive (pairwise) | O(n²) | O(n² · N) | O(n² · N log N) |
| Cached FFT | O(n) IFFT | O(n² · N) | O(n · N log N + n² · N) |
| DSYRK | O(n) IFFT | O(n · N) via DSYRK | **O(n · N log N + n · N)** |

For n=50, N=10000:
- Naive: ~2500 FFTs, 2500 × 10000 = 25M dot products
- DSYRK: 50 IFFTs, one DSYRK call ≈ 50 × 10000 = 500K ops

**Speedup: ~3-5×** in practice.

## AVX2 SIMD Helpers

The weighted operations use AVX2 intrinsics:

```c
// ssa_opt_weighted_norm_sq: Σ h[t]² · w[t]
// Processes 4 doubles per iteration
__m256d sum_vec = _mm256_setzero_pd();
for (int i = 0; i + 4 <= N; i += 4) {
    __m256d h_vec = _mm256_loadu_pd(&h[i]);
    __m256d w_vec = _mm256_loadu_pd(&w[i]);
    __m256d h_sq = _mm256_mul_pd(h_vec, h_vec);
    sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(h_sq, w_vec));
}
// Horizontal sum...
```

```c
// ssa_opt_scale_weighted: dst[t] = scale · src[t] · w[t]
__m256d scale_vec = _mm256_set1_pd(scale);
for (int i = 0; i + 4 <= N; i += 4) {
    __m256d s_vec = _mm256_loadu_pd(&src[i]);
    __m256d w_vec = _mm256_loadu_pd(&w[i]);
    __m256d prod = _mm256_mul_pd(_mm256_mul_pd(s_vec, w_vec), scale_vec);
    _mm256_storeu_pd(&dst[i], prod);
}
```

## Memory Layout

```
U_fft:           [n × 2 × r2c_len]  Pre-cached FFT of scaled left vectors
V_fft:           [n × 2 × r2c_len]  Pre-cached FFT of right vectors
wcorr_ws_complex:[n × 2 × r2c_len]  Workspace for batch complex multiply
wcorr_h:         [n × fft_len]      Workspace for batch IFFT results
wcorr_G:         [n × N]            Normalized weighted reconstructions
wcorr_sqrt_inv_c:[N]                Pre-computed √(1/w[t])
```

All allocated at `ssa_opt_cache_ffts()` or `ssa_opt_prepare()`.

## When to Cache

```c
// Call after decomposition if you'll compute W-correlation
ssa_opt_decompose_randomized(&ssa, k, 8);
ssa_opt_cache_ffts(&ssa);  // Allocates and computes U_fft, V_fft

// Now wcorr is fast
double *W = malloc(k * k * sizeof(double));
ssa_opt_wcorr_matrix(&ssa, W);  // Auto-dispatches to fast path
```

If you only need a few pairwise correlations, use `ssa_opt_wcorr_pair(i, j)` instead.

## Interpretation

| W(i,j) | Meaning |
|--------|---------|
| ≈ 0 | Components i,j are separable (orthogonal) |
| ≈ 1 | Components are nearly identical (should group together) |
| > 0.5 | Likely the same underlying signal (e.g., sine pair) |

Typical grouping: periodic components come in pairs (sine + cosine at same frequency), so W(2i, 2i+1) ≈ 0.5-0.9 for harmonic pairs.
