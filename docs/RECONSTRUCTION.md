# SSA Reconstruction: Diagonal Averaging

Converting rank-1 matrices back to time series signals.

## The Process

After SVD, each component is a rank-1 matrix:

```
Xᵢ = σᵢ · uᵢ · vᵢᵀ    (L × K matrix)
```

To get back a 1D signal, we **average along anti-diagonals** (Hankel averaging):

```
Trajectory matrix (example):

       j=0   j=1   j=2   j=3
      ┌─────────────────────┐
 i=0  │  a     b     c     d │
 i=1  │  e     f     g     h │
 i=2  │  i     j     k     l │
      └─────────────────────┘

Anti-diagonals (same i+j):
  t=0: a                    → x̃[0] = a
  t=1: b, e                 → x̃[1] = (b + e) / 2
  t=2: c, f, i              → x̃[2] = (c + f + i) / 3
  t=3: d, g, j              → x̃[3] = (d + g + j) / 3
  t=4: h, k                 → x̃[4] = (h + k) / 2
  t=5: l                    → x̃[5] = l
```

## Weights

The weight (count) for position t depends on how many elements fall on that anti-diagonal:

```
w[t] = min(t + 1, L, K, N - t)

Example (L=3, K=4, N=6):
  w = [1, 2, 3, 3, 2, 1]
```

We precompute `inv_diag_count[t] = 1 / w[t]` at initialization for efficiency.

## Naive vs FFT-Based

### Naive O(N²)

```python
# Form the rank-1 matrix explicitly
X = sigma * np.outer(u, v)  # L×K matrix, O(L*K) memory

# Average anti-diagonals
x_recon = np.zeros(N)
for t in range(N):
    for i in range(L):
        j = t - i
        if 0 <= j < K:
            x_recon[t] += X[i, j]
    x_recon[t] /= count[t]
```

This requires O(L × K) ≈ O(N²) memory and O(N²) time.

### FFT-Based O(N log N)

Key insight: the sum along anti-diagonal t equals:

```
Σᵢ X[i, t-i] = Σᵢ σ·u[i]·v[t-i] = σ · (u ∗ v)[t]
```

This is a **convolution**! We can compute it via FFT:

```
reconstruction = IFFT(FFT(σ·u) ⊙ FFT(v)) / w
```

## Implementation

```c
int ssa_opt_reconstruct(const SSA_Opt *ssa, const int *group, int n_group, double *output) {
    // Accumulate in frequency domain
    double *freq_accum = workspace;
    ssa_opt_zero(freq_accum, 2 * r2c_len);
    
    if (ssa->fft_cached) {
        // Fast path: use pre-cached FFTs
        for (int g = 0; g < n_group; g++) {
            int idx = group[g];
            const double *u_fft = &ssa->U_fft[idx * 2 * r2c_len];
            const double *v_fft = &ssa->V_fft[idx * 2 * r2c_len];
            
            // Pointwise multiply in frequency domain
            ssa_opt_complex_mul_r2c(u_fft, v_fft, ws_complex, r2c_len);
            
            // Accumulate (sum of convolutions = convolution of sums)
            cblas_daxpy(2 * r2c_len, 1.0, ws_complex, 1, freq_accum, 1);
        }
    } else {
        // Slow path: compute FFTs on the fly
        for (int g = 0; g < n_group; g++) {
            int idx = group[g];
            double sigma = ssa->sigma[idx];
            
            // FFT of scaled u
            for (int i = 0; i < L; i++) ws_real[i] = sigma * U[idx * L + i];
            DftiComputeForward(fft_r2c, ws_real, ws_complex);
            
            // FFT of v
            memcpy(ws_real2, &V[idx * K], K * sizeof(double));
            DftiComputeForward(fft_r2c, ws_real2, temp_fft);
            
            // Multiply and accumulate
            ssa_opt_complex_mul_r2c(ws_complex, temp_fft, ws_complex, r2c_len);
            cblas_daxpy(2 * r2c_len, 1.0, ws_complex, 1, freq_accum, 1);
        }
    }
    
    // Single IFFT for combined result
    DftiComputeBackward(fft_c2r, freq_accum, ws_real);
    memcpy(output, ws_real, N * sizeof(double));
    
    // Divide by weights (diagonal averaging)
    vdMul(N, output, ssa->inv_diag_count, output);  // MKL vectorized multiply
}
```

## Frequency-Domain Accumulation

The key optimization: instead of reconstructing each component separately, we **accumulate in frequency domain**:

```
FFT(Σᵢ σᵢ·uᵢ∗vᵢ) = Σᵢ σᵢ·FFT(uᵢ)⊙FFT(vᵢ)
```

This means:
- n_group complex multiplies (cheap)
- Only **one IFFT** at the end (expensive)

vs naive approach:
- n_group IFFTs (expensive)
- n_group real additions (cheap)

## Cached FFTs

With `ssa_opt_cache_ffts()`, we precompute:

```c
U_fft[i] = FFT(sigma[i] * U[:,i])   // Scaled left vectors
V_fft[i] = FFT(V[:,i])              // Right vectors
```

This makes reconstruction nearly free: just complex multiplies and one IFFT.

## Complexity

| Method | Per Component | Total (n components) |
|--------|---------------|----------------------|
| Naive | O(N²) | O(n × N²) |
| FFT (no cache) | O(N log N) | O(n × N log N) |
| FFT (cached) | O(N) multiply | O(n × N) + O(N log N) |

For N=10000, n=20: FFT cached is ~100× faster than naive.
