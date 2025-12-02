# SSA-Opt: High-Performance Singular Spectrum Analysis

A fast, header-only C library for Singular Spectrum Analysis (SSA) with FFT-accelerated matrix operations and optional Intel MKL optimization.

## 1. What is SSA?

Singular Spectrum Analysis is a powerful technique for time series decomposition. It extracts:

- **Trends** - Long-term movements in data
- **Periodic components** - Seasonal patterns, oscillations
- **Noise** - Random fluctuations

Unlike Fourier analysis, SSA is non-parametric and handles non-stationary signals well. Applications include financial data analysis, climate science, biomedical signal processing, and anywhere you need to separate signal from noise.

## 2. Results

### 2.1 Signal Denoising: 159× Noise Reduction

<img width="2779" height="1576" alt="ssa_denoising_best" src="https://github.com/user-attachments/assets/980caf63-7c12-4a8c-86ed-35a6e237bbc2" />

*SSA recovers clean signal from heavy noise (SNR: 4.1 → 26.1 dB). Top-left: noisy input. Top-right: noise removal transformation. Bottom-left: accuracy comparison. Bottom-right: zoomed detail.*

### 2.2 Trend Extraction: Bitcoin Price Analysis

<img width="3173" height="1966" alt="linkedin_btc_max_accuracy" src="https://github.com/user-attachments/assets/5eead1c6-248b-4feb-b327-b993efeee711" />

*Ensemble SSA extracts smooth trend from volatile Bitcoin prices (r = 0.987). Automatically separates trend from market cycles without parameter tuning.*

### 2.3 Forecasting: Airline Passengers

<img width="2303" height="1077" alt="linkedin_forecast" src="https://github.com/user-attachments/assets/6a6400ae-c5f3-41b1-ba8f-ea8ce8193a49" />

*2-year ahead forecast on classic airline passengers dataset (RMSE: 47.5). SSA captures both trend and 12-month seasonality.*

## 3. Benchmarks

### 3.1 Speed: SSA-Opt vs Rssa

Comparison against Rssa (gold-standard R implementation):

| N | L | k | SSA-Opt (ms) | Rssa (ms) | Speedup |
|---|---|---|--------------|-----------|---------|
| 500 | 125 | 30 | 5.6 | 26.1 | **4.5x** |
| 1000 | 250 | 30 | 3.3 | 41.3 | **12.6x** |
| 2000 | 500 | 30 | 4.4 | 36.1 | **8.1x** |
| 5000 | 1250 | 30 | 12.7 | 122.2 | **9.6x** |
| 10000 | 2500 | 30 | 18.6 | 142.5 | **7.6x** |

*Intel Core i9 14900KF, Windows, MKL 2025.3. Rssa uses single-threaded BLAS.*

### 3.2 Accuracy: Validated Against Rssa

| Test Signal | Trend Correlation | Forecast Correlation |
|-------------|-------------------|----------------------|
| sine_noise | 0.9998 | - |
| trend_seasonal | 0.99999 | 0.9996 |
| multi_periodic | 0.9996 | - |
| nonlinear | 0.99999 | - |
| stock_sim | 0.99999 | - |

## 4. Features

- **FFT-accelerated Hankel matrix-vector products** - O(N log N) instead of O(N²)
- **Block power method** - Process multiple singular vectors simultaneously
- **Forecasting (LRF)** - Linear Recurrent Formula for time series prediction
- **MSSA** - Multivariate SSA for analyzing multiple correlated series
- **Python bindings** - Full-featured ctypes wrapper with NumPy integration
- **Two backends**: Reference (portable C) and MKL-optimized (2-3x faster)
- **Header-only design** - Just include `ssa_opt.h`

## 5. Algorithm

### 5.1 Trajectory Matrix (Hankel Embedding)

SSA embeds a 1D time series into a 2D trajectory matrix. Given signal `x = [x₀, x₁, ..., x_{N-1}]` and window length `L`, we construct an `L × K` Hankel matrix where `K = N - L + 1`:

```
         Column 0   Column 1   Column 2   ...   Column K-1
       ┌─────────────────────────────────────────────────┐
Row 0  │   x₀        x₁         x₂       ...    x_{K-1}  │
Row 1  │   x₁        x₂         x₃       ...    x_K      │
Row 2  │   x₂        x₃         x₄       ...    x_{K+1}  │
  ⋮     │   ⋮         ⋮          ⋮        ⋱       ⋮        │
Row L-1│  x_{L-1}    x_L       x_{L+1}   ...    x_{N-1}  │
       └─────────────────────────────────────────────────┘

H[i,j] = x[i+j]    (constant along anti-diagonals)
```

This matrix is never explicitly formed - it would require O(L × K) ≈ O(N²) memory.

### 5.2 Singular Value Decomposition

We compute the top-k singular triplets of H. Direct SVD is O(N³), so we use power iteration:

```
H ≈ σ₀·u₀·v₀ᵀ + σ₁·u₁·v₁ᵀ + ... + σₖ₋₁·uₖ₋₁·vₖ₋₁ᵀ

     ┌─────────┐       ┌───┐   ┌───┐ᵀ       ┌───┐   ┌───┐ᵀ
     │         │       │ u │   │ v │        │ u │   │ v │
H  = │  L × K  │  ≈ σ₀·│ ₀ │ × │ ₀ │  + σ₁· │ ₁ │ × │ ₁ │  + ...
     │         │       │   │   │   │        │   │   │   │
     └─────────┘       └───┘   └───┘        └───┘   └───┘
                       (L×1)   (K×1)        (L×1)   (K×1)
```

Power iteration finds each triplet (σᵢ, uᵢ, vᵢ) by repeatedly multiplying:

```
v⁽⁰⁾ = random       ─┐
                     │  repeat until convergence:
u = H·v,  u = u/‖u‖  │    • Forward:  u = H·v   (L×K · K×1 → L×1)
v = Hᵀ·u, σ = ‖v‖    │    • Adjoint:  v = Hᵀ·u  (K×L · L×1 → K×1)
v = v/σ             ─┘    • σ = ‖v‖ before normalizing
```

### 5.3 Reconstruction (Diagonal Averaging)

Each component forms a rank-1 matrix, then averaged back to 1D:

```
Step 1: Outer product             Step 2: Diagonal averaging
                                  
     u · vᵀ = Xᵢ                  Average along anti-diagonals:
                                  
┌───┐   ┌──────────────────┐      t=0: x̃[0] = X[0,0]
│ u₀│   │u₀v₀ u₀v₁ u₀v₂ ...│      t=1: x̃[1] = (X[0,1] + X[1,0])/2
│ u₁│ × │u₁v₀ u₁v₁ u₁v₂ ...│  →   t=2: x̃[2] = (X[0,2] + X[1,1] + X[2,0])/3
│ u₂│   │u₂v₀ u₂v₁ u₂v₂ ...│       ⋮
│ ⋮ │   │ ⋮    ⋮    ⋮        │      
└───┘   └──────────────────┘      Result: x̃ = [x̃₀, x̃₁, ..., x̃_{N-1}]
```

### 5.4 Forecasting (LRF)

The Linear Recurrent Formula extracts coefficients from left singular vectors:

```
Given L singular vectors U = [u₀, u₁, ..., uₖ₋₁], each of length L:

┌──────────────────────────┐
│  u₀[0]   u₁[0]  ...      │  ← first L-1 rows: "past" values
│  u₀[1]   u₁[1]  ...      │
│   ⋮       ⋮               │
│  u₀[L-2] u₁[L-2] ...     │
├──────────────────────────┤
│  u₀[L-1] u₁[L-1] ...     │  ← last row: determines recurrence
└──────────────────────────┘

Coefficients: a = (last row) · (first L-1 rows)ᵀ · (I - last_row·last_rowᵀ)⁻¹

Forecast: x[n] = a₁·x[n-1] + a₂·x[n-2] + ... + a_{L-1}·x[n-L+1]
```

### 5.5 MSSA (Multivariate SSA)

For M time series, stack trajectory matrices vertically:

```
Series 1: x⁽¹⁾ → H⁽¹⁾ (L × K)      ┌─────────┐
Series 2: x⁽²⁾ → H⁽²⁾ (L × K)  →   │  H⁽¹⁾   │  L rows
Series 3: x⁽³⁾ → H⁽³⁾ (L × K)      │─────────│
                                   │  H⁽²⁾   │  L rows    = H_stacked
                                   │─────────│               (M·L × K)
                                   │  H⁽³⁾   │  L rows
                                   └─────────┘
                                      K cols

SVD of H_stacked extracts common patterns across all series.
Each series reconstructed separately from shared singular vectors.
```

## 6. Optimizations

### 6.1 FFT-Accelerated Matrix-Vector Product

Hankel matrix-vector multiplication equals convolution:

```
Forward: y = H·v                    Adjoint: z = Hᵀ·u

y[i] = Σⱼ x[i+j]·v[j]              z[j] = Σᵢ x[i+j]·u[i]
     = (x ∗ v)[i]                       = (x ∗ flip(u))[L-1+j]

┌─────────────────────────────────────────────────────────┐
│  conv(a,b) = IFFT( FFT(a) ⊙ FFT(b) )                   │
│                                                         │
│  Complexity: O(N²) direct  →  O(N log N) with FFT       │
└─────────────────────────────────────────────────────────┘

Pre-compute FFT(signal) once at init.
Each matvec: 1 FFT + 1 pointwise multiply + 1 IFFT
```

### 6.2 Block Power Method

Process b vectors simultaneously instead of sequentially:

```
Sequential (b=1):                 Block (b=32):
┌───┐                             ┌───────────────────┐
│ v │  → H·v → u → Hᵀ·u → v       │ v₀ v₁ v₂ ... v₃₁  │   = V_block
└───┘                             └───────────────────┘
                                           ↓
× k components                    H·V_block (32 FFTs batched)
× 100 iterations                           ↓
= 6400 individual FFTs            QR(U_block)
                                           ↓
                                  Hᵀ·U_block (32 FFTs batched)
                                           ↓
                                  = 200 batched calls total

Benefits:
• MKL batched FFT: single function call for 32 transforms
• Memory locality: V_block columns contiguous in memory
• BLAS3: QR uses GEMM instead of GEMV
```

### 6.3 Periodic QR

```
Every iteration:        vs.     Every 5 iterations:

iter 1: QR                      iter 1: (skip)
iter 2: QR                      iter 2: (skip)
iter 3: QR                      iter 3: (skip)
iter 4: QR                      iter 4: (skip)
iter 5: QR                      iter 5: QR ←
  ⋮                               ⋮
100 QR calls                    20 QR calls (5× reduction)

Stability maintained by:
• Final QR before Rayleigh-Ritz extraction
• Deflation orthogonalization against previous blocks each iteration
```

### 6.4 Rayleigh-Ritz Extraction

After block iteration converges, vectors span the correct subspace but aren't individually optimal:

```
┌─────────────────────────────────────────────────────┐
│ 1. Project: M = U_blockᵀ · (H · V_block)   [b × b]  │
│                                                     │
│ 2. Small SVD: M = Uₛ · Σ · Vₛᵀ             [b × b]   │
│                                                     │
│ 3. Rotate:  U_final = U_block · Uₛ                   │
│             V_final = V_block · Vₛ                   │
│             σ_final = diag(Σ)                       │
└─────────────────────────────────────────────────────┘

Cost: one 32×32 SVD (microseconds) → optimal singular values
```

### 6.5 GEMM-Based Operations

All multi-vector operations use BLAS Level 3:

| Operation | BLAS2 (sequential) | BLAS3 (block) |
|-----------|-------------------|---------------|
| Orthogonalize against previous | k × GEMV | Single GEMM |
| Project out components | k × AXPY | Single GEMM |

GEMM achieves near-peak FLOPS due to cache reuse and SIMD.

## 7. API Reference

### 7.1 Core Functions

```c
int ssa_opt_init(SSA_Opt* ssa, const double* data, int N, int L);
int ssa_opt_decompose(SSA_Opt* ssa, int k, int max_iter);
int ssa_opt_decompose_block(SSA_Opt* ssa, int k, int block_size, int max_iter);
int ssa_opt_reconstruct(SSA_Opt* ssa, const int* group, int group_size, double* result);
int ssa_opt_forecast(SSA_Opt* ssa, const int* group, int group_size, int n_forecast, double* forecast);
void ssa_opt_free(SSA_Opt* ssa);
```

### 7.2 MSSA Functions

```c
int ssa_opt_mssa_init(SSA_MSSA* mssa, const double* data, int M, int N, int L);
int ssa_opt_mssa_decompose(SSA_MSSA* mssa, int k, int max_iter);
int ssa_opt_mssa_reconstruct(SSA_MSSA* mssa, int series_idx, const int* group, int group_size, double* result);
void ssa_opt_mssa_free(SSA_MSSA* mssa);
```

### 7.3 After Decomposition

Access results via:
- `ssa->sigma[i]` - Singular values (descending)
- `ssa->eigenvalues[i]` - Squared singular values (variance explained)
- `ssa->U[i * L]` - Left singular vector i (length L)
- `ssa->V[i * K]` - Right singular vector i (length K)

## 8. Parameter Selection

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| L (window) | N/3 to N/2 | Larger = better resolution, slower |
| k (components) | 20-50 | More than needed, then select |
| block_size | 32 | Optimal for MKL batched FFT |

## 9. Building

### 9.1 Prerequisites

- C compiler with C99 support (MSVC 2019+, GCC 7+, Clang 5+)
- CMake 3.15+
- Intel oneAPI (optional, for MKL backend)

### 9.2 Windows with MKL

```powershell
mkdir build && cd build
cmake .. -DSSA_USE_MKL=ON
cmake --build . --config Release
```

Copy MKL DLLs to build folder or add to PATH:
- `mkl_core.2.dll`, `mkl_intel_thread.2.dll`, `mkl_intel_lp64.2.dll`
- `libiomp5md.dll` (from compiler folder)

### 9.3 Linux with MKL

```bash
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DSSA_USE_MKL=ON
make -j$(nproc)
```

### 9.4 Without MKL

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Note: Reference implementation is significantly slower.

## 10. Limitations

- **Memory**: Stores L×k for U vectors and K×k for V vectors
- **Not suitable for**: Very short signals (N < 100) or streaming data
- **Block overhead**: For small problems (N < 2000), sequential may be faster

## 11. Future Work

- [x] Forecasting (LRF)
- [x] MSSA
- [x] Python bindings
- [ ] Real-to-complex FFT (2x speedup)
- [ ] Convergence detection
- [ ] GPU acceleration (cuFFT)

## 12. References

1. Golyandina, N., & Zhigljavsky, A. (2013). *Singular Spectrum Analysis for Time Series*. Springer.
2. Golyandina, N., & Korobeynikov, A. (2014). Basic Singular Spectrum Analysis and Forecasting with R. *Computational Statistics & Data Analysis*.
3. Korobeynikov, A. (2010). Computation- and space-efficient implementation of SSA. *Statistics and Its Interface*, 3(3), 357-368.

## 13. License

GPL 3.0 - See LICENSE file for details.
