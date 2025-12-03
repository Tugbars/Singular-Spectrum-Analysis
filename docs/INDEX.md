# SSA-Opt Documentation

Technical documentation for the SSA-Opt library implementation.

## Core Operations

| Document | Description |
|----------|-------------|
| [HANKEL_MATVEC.md](HANKEL_MATVEC.md) | FFT-based Hankel matrix-vector product |
| [RANDOMIZED_SVD.md](RANDOMIZED_SVD.md) | Randomized SVD decomposition algorithm |
| [RECONSTRUCTION.md](RECONSTRUCTION.md) | Diagonal averaging via FFT |
| [WCORR.md](WCORR.md) | W-correlation matrix computation |

## Analysis & Forecasting

| Document | Description |
|----------|-------------|
| [FORECAST.md](FORECAST.md) | R-forecast (LRF) and V-forecast methods |
| [ESPRIT.md](ESPRIT.md) | Automatic frequency/period detection |
| [CADZOW.md](CADZOW.md) | Iterative denoising for exact rank-r |
| [GAPFILL.md](GAPFILL.md) | Handling missing values (NaN) |

## Quick Reference

### Complexity Summary

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Hankel matvec | O(N log N) | FFT-based |
| Decomposition (randomized) | O(kp · N log N) | kp = k + oversampling |
| Decomposition (block) | O(k · iter · N log N) | iter ≈ 50-100 |
| Reconstruction | O(n · N log N) | n = group size |
| Reconstruction (cached) | O(n · N) + O(N log N) | One IFFT |
| W-correlation matrix | O(n · N log N) | DSYRK optimization |
| Forecast (R) | O(L) per step | Precomputed LRF |
| Forecast (V) | O(n · L) per step | Per-step projection |
| ESPRIT | O(r³) | r = components analyzed |
| Cadzow | O(iter · N log N) | iter ≈ 10-30 |
| Gap filling | O(iter · N log N) | iter ≈ 5-20 |

### Memory Usage

| Buffer | Size | When Allocated |
|--------|------|----------------|
| Signal FFT (fft_x) | N/2+1 complex | init() |
| Workspace | ~4N doubles | init() |
| U, V, σ | k·L + k·K + k | decompose() |
| Cached FFTs | k·N complex | cache_ffts() |
| Prepared workspace | ~4·kp·max(L,K) | prepare() |

### Function Call Graph

```
ssa_opt_init()
    └── Pre-compute FFT(signal)

ssa_opt_prepare()          [Optional: enables malloc-free path]
    └── Pre-allocate all decomposition workspace

ssa_opt_decompose_randomized()
    ├── vdRngGaussian()    [Random Ω]
    ├── hankel_matvec_block() [Y = H·Ω]
    ├── LAPACKE_dgeqrf()   [QR factorization]
    ├── hankel_matvec_T_block() [B = Hᵀ·Q]
    ├── LAPACKE_dgesdd()   [SVD of small B]
    └── cblas_dgemm()      [Recover U]

ssa_opt_reconstruct()
    ├── [Loop] Complex multiply in frequency domain
    ├── DftiComputeBackward() [Single IFFT]
    └── vdMul() [Diagonal averaging weights]

ssa_opt_forecast()
    ├── ssa_opt_lrf_init()  [Compute LRF coefficients]
    ├── ssa_opt_reconstruct() [Get base signal]
    └── [Loop] cblas_ddot() [Apply LRF]

ssa_opt_wcorr_matrix_fast()
    ├── [Parallel] Complex multiply (cached FFTs)
    ├── DftiComputeBackward() [Batched IFFT]
    ├── [Parallel] Normalize with AVX2
    └── cblas_dsyrk() [Gram matrix]
```
