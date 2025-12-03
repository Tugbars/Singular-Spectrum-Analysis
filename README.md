# SSA-Opt: High-Performance Singular Spectrum Analysis

A fast, header-only C library for Singular Spectrum Analysis with **Python bindings**, FFT-accelerated operations, and Intel MKL optimization.

**7-20× faster than Rssa** (the gold-standard R implementation).

## 1. What is SSA?

Singular Spectrum Analysis decomposes time series into:

- **Trends** - Long-term movements
- **Periodic components** - Cycles, seasonality
- **Noise** - Random fluctuations

Unlike Fourier analysis, SSA is non-parametric and adapts to your data's structure. Ideal for financial analysis, signal processing, and anywhere you need clean signal extraction.

## 2. Results

### 2.1 Signal Denoising: 159× Noise Reduction

<img width="2779" height="1576" alt="ssa_denoising_best" src="https://github.com/user-attachments/assets/980caf63-7c12-4a8c-86ed-35a6e237bbc2" />

*SSA recovers clean signal from heavy noise (SNR: 4.1 → 26.1 dB).*

### 2.2 Trend Extraction: Bitcoin Price Analysis

<img width="3173" height="1966" alt="linkedin_btc_max_accuracy" src="https://github.com/user-attachments/assets/5eead1c6-248b-4feb-b327-b993efeee711" />

*Ensemble SSA extracts smooth trend from volatile Bitcoin prices (r = 0.987).*

### 2.3 Forecasting: Airline Passengers

<img width="2303" height="1077" alt="linkedin_forecast" src="https://github.com/user-attachments/assets/6a6400ae-c5f3-41b1-ba8f-ea8ce8193a49" />

*2-year ahead forecast capturing trend and 12-month seasonality (RMSE: 47.5).*

### 2.3 Signal Denoising with Cadzow. 

<img width="3472" height="1971" alt="cadzow_comparison" src="https://github.com/user-attachments/assets/5bf60b04-05c7-4f58-af48-b1b883df4468" />

## 3. Benchmarks

### 3.1 Speed: SSA-Opt vs Rssa

Benchmarked with `benchmark_vs_rssa.py` on Intel Core i9-14900KF, MKL 2025.0.3:

<img width="2081" height="730" alt="benchmark_chart" src="https://github.com/user-attachments/assets/788a399f-ae8c-4e64-9b83-a9ee0d0da8ad" />

| N | L | k | SSA-Opt (ms) | Rssa (ms) | Speedup | Correlation |
|---|---|---|--------------|-----------|---------|-------------|
| 500 | 125 | 30 | 1.2 | 23.8 | **20.3×** | 0.9895 |
| 1000 | 250 | 30 | 2.1 | 36.9 | **17.2×** | 0.9973 |
| 5000 | 1250 | 30 | 8.9 | 118.1 | **13.3×** | 0.9996 |
| 10000 | 2500 | 50 | 20.7 | 368.4 | **17.8×** | 0.9999 |
| 20000 | 5000 | 50 | 36.1 | 620.3 | **17.2×** | 1.0000 |

*Rssa uses PROPACK (Lanczos bidiagonalization). SSA-Opt uses randomized SVD with MKL.*

### 3.2 MKL Configuration Impact

Using `mkl_config.h` for hybrid CPU optimization (P-core affinity, thread tuning):

| N | Default MKL | Optimized | Improvement |
|---|-------------|-----------|-------------|
| 1000 | 2.7 ms | 2.1 ms | **1.25×** |
| 5000 | 10.7 ms | 8.9 ms | **1.20×** |
| 10000 | 23.4 ms | 20.7 ms | **1.13×** |
| 20000 | 41.3 ms | 36.1 ms | **1.14×** |

The `mkl_config.h` header auto-detects hybrid CPUs (Intel 12th-14th gen) and pins threads to P-cores only, avoiding E-core slowdown.

### 3.3 Accuracy Validation

| Test Signal | Correlation with True Signal |
|-------------|------------------------------|
| N=500, SNR=7dB | 0.9895 |
| N=1000, SNR=7dB | 0.9973 |
| N=10000, SNR=7dB | 0.9999 |
| N=20000, SNR=7dB | 1.0000 |

## 4. Features

- **Python bindings** - Full ctypes wrapper with NumPy integration
- **3 decomposition methods** - Sequential, block, and randomized SVD
- **FFT-accelerated** - O(N log N) Hankel matrix operations
- **Forecasting (LRF)** - Linear Recurrent Formula prediction
- **MSSA** - Multivariate SSA for correlated series
- **MKL-optimized** - Hybrid CPU auto-configuration
- **Header-only** - Just `#include "ssa_opt_r2c.h"`

## 5. Quick Start

### 5.1 Python (Recommended)

```python
from ssa_wrapper import SSA

# Load your data
prices = np.array([...])  # Your time series

# Decompose
ssa = SSA(prices, L=250)
ssa.decompose(k=30)  # Uses fast randomized SVD

# Extract components
trend = ssa.reconstruct([0])
cycle = ssa.reconstruct([1, 2])

# Forecast
forecast = ssa.forecast([0, 1, 2], n_forecast=50)

# Analysis
print(f"Variance explained: {ssa.variance_explained(0, 2):.1%}")
pairs = ssa.find_periodic_pairs()
```

### 5.2 C

```c
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt_r2c.h"
#include "mkl_config.h"

int main() {
    // Initialize MKL (call once at startup)
    mkl_config_ssa_full(1);  // 1 = verbose
    
    // Setup
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, signal, N, L);
    
    // Decompose (randomized is fastest)
    ssa_opt_decompose_randomized(&ssa, k, 8);
    
    // Reconstruct trend
    int trend_group[] = {0};
    double* trend = malloc(N * sizeof(double));
    ssa_opt_reconstruct(&ssa, trend_group, 1, trend);
    
    // Cleanup
    ssa_opt_free(&ssa);
    free(trend);
}
```

### 5.3 After Decomposition

Access results directly via the struct:

```c
ssa->sigma[i]        // Singular values (descending order)
ssa->eigenvalues[i]  // Squared singular values (variance explained)
ssa->U[i * L]        // Left singular vector i (length L)
ssa->V[i * K]        // Right singular vector i (length K)
```

## 6. Algorithm

### 6.1 Trajectory Matrix (Hankel Embedding)

SSA embeds a 1D time series into a 2D trajectory matrix. Given signal `x = [x₀, x₁, ..., x_{N-1}]` and window length `L`, we construct an `L × K` Hankel matrix where `K = N - L + 1`:

```
         Column 0   Column 1   Column 2   ...   Column K-1
       ┌─────────────────────────────────────────────────┐
Row 0  │   x₀        x₁         x₂       ...    x_{K-1}  │
Row 1  │   x₁        x₂         x₃       ...    x_K      │
Row 2  │   x₂        x₃         x₄       ...    x_{K+1}  │
  ⋮    │   ⋮         ⋮          ⋮        ⋱       ⋮       │
Row L-1│  x_{L-1}    x_L       x_{L+1}   ...    x_{N-1}  │
       └─────────────────────────────────────────────────┘

H[i,j] = x[i+j]    (constant along anti-diagonals)
```

This matrix is never explicitly formed - it would require O(L × K) ≈ O(N²) memory.

### 6.2 Singular Value Decomposition

We compute the top-k singular triplets of H:

```
H ≈ σ₀·u₀·v₀ᵀ + σ₁·u₁·v₁ᵀ + ... + σₖ₋₁·uₖ₋₁·vₖ₋₁ᵀ
```

Each triplet (σᵢ, uᵢ, vᵢ) represents one component. Larger σ = more signal energy.

### 6.3 Reconstruction (Diagonal Averaging)

Each component forms a rank-1 matrix, then averaged back to 1D along anti-diagonals:

```
t=0: x̃[0] = X[0,0]
t=1: x̃[1] = (X[0,1] + X[1,0]) / 2
t=2: x̃[2] = (X[0,2] + X[1,1] + X[2,0]) / 3
...
```

### 6.4 Forecasting (Linear Recurrent Formula)

SSA signals satisfy a linear recurrence. The LRF extracts coefficients from left singular vectors to predict future values:

```
x[n] = a₁·x[n-1] + a₂·x[n-2] + ... + a_{L-1}·x[n-L+1]
```

## 7. Optimizations

### 7.1 FFT-Accelerated Matrix-Vector Product

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

### 7.2 Block Power Method

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

### 7.3 Periodic QR

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

### 7.4 Rayleigh-Ritz Extraction

After block iteration converges, vectors span the correct subspace but aren't individually optimal:

```
┌─────────────────────────────────────────────────────┐
│ 1. Project: M = U_blockᵀ · (H · V_block)   [b × b]  │
│                                                     │
│ 2. Small SVD: M = Uₛ · Σ · Vₛᵀ             [b × b]  │
│                                                     │
│ 3. Rotate:  U_final = U_block · Uₛ                  │
│             V_final = V_block · Vₛ                  │
│             σ_final = diag(Σ)                       │
└─────────────────────────────────────────────────────┘

Cost: one 32×32 SVD (microseconds) → optimal singular values
```

### 7.5 GEMM-Based Operations

All multi-vector operations use BLAS Level 3:

| Operation | BLAS2 (sequential) | BLAS3 (block) |
|-----------|-------------------|---------------|
| Orthogonalize against previous | k × GEMV | Single GEMM |
| Project out components | k × AXPY | Single GEMM |

GEMM achieves near-peak FLOPS due to cache reuse and SIMD.

### 7.6 Randomized SVD

For very large problems, randomized SVD is fastest:

```
1. Random projection: Y = H · Ω,  Ω is K × (k+p) random
2. Orthogonalize: Q = qr(Y)
3. Project: B = Qᵀ · H
4. SVD of small matrix: B = U_B · Σ · Vᵀ
5. Recover: U = Q · U_B
```

Complexity: O((k+p) × N log N) vs O(k × iterations × N log N) for power iteration.

### 7.7 R2C FFT Optimization

Real-to-Complex FFT exploits conjugate symmetry:

- Output: N/2+1 complex values (vs N for C2C)
- 50% less memory for FFT buffers
- ~17% faster than complex-to-complex

### 7.8 Cached FFTs for W-Correlation

W-correlation matrix requires FFTs of all component reconstructions. With caching:

| Operation | Without Cache | With Cache |
|-----------|---------------|------------|
| wcorr_matrix | 2n forward FFTs | 0 forward FFTs |
| Speedup | 1× | **3×** |

Call `ssa_opt_cache_ffts()` once after decomposition if computing W-correlation multiple times.

## 8. Installation

### 8.1 Python

```bash
# Build the shared library
cd MKL
mkdir build && cd build
cmake .. -DUSE_MKL=ON
cmake --build . --config Release

# Copy ssa.dll/libssa.so to py/ folder
# Then:
cd ../py
python -c "from ssa_wrapper import SSA; print('OK')"
```

### 8.2 C/C++ (Header-Only)

```c
#define SSA_OPT_IMPLEMENTATION
#include "ssa_opt_r2c.h"
```

Link with MKL:
```bash
# Windows
cl /O2 your_code.c /I"%MKLROOT%\include" mkl_rt.lib

# Linux
gcc -O3 your_code.c -I${MKLROOT}/include -lmkl_rt -lm
```

## 9. Benchmarking

Run the benchmark suite:

```bash
cd py

# Speed comparison vs Rssa
python benchmark_vs_rssa.py --speed

# Compare MKL configurations
python benchmark_vs_rssa.py --mkl-config

# Internal method comparison (seq/block/randomized)
python benchmark_vs_rssa.py --internal

# All benchmarks
python benchmark_vs_rssa.py --all
```

Parameter sensitivity tests:

```bash
python ssa_parameter_tests.py --snr        # Noise robustness
python ssa_parameter_tests.py --window     # L sensitivity
python ssa_parameter_tests.py --components # k sweep
python ssa_parameter_tests.py --all
```

## 10. Limitations

- **Memory**: O(k × N) for storing singular vectors
- **Not for streaming**: Batch processing only
- **Minimum size**: N > 100 recommended

## 11. References

1. Golyandina, N., & Zhigljavsky, A. (2013). *Singular Spectrum Analysis for Time Series*. Springer.
2. Korobeynikov, A. (2010). Computation- and space-efficient implementation of SSA. *Statistics and Its Interface*, 3(3).
3. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness. *SIAM Review*, 53(2).

## 12. License

GPL 3.0 - See LICENSE file.
