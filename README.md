# SSA-Opt: High-Performance Singular Spectrum Analysis

A fast, header-only C library for Singular Spectrum Analysis (SSA) with FFT-accelerated matrix operations and optional Intel MKL optimization.

## What is SSA?

Singular Spectrum Analysis is a powerful technique for time series decomposition. It extracts:

- **Trends** - Long-term movements in data
- **Periodic components** - Seasonal patterns, oscillations
- **Noise** - Random fluctuations

Unlike Fourier analysis, SSA is non-parametric and handles non-stationary signals well. Applications include financial data analysis, climate science, biomedical signal processing, and anywhere you need to separate signal from noise.

## Features

- **FFT-accelerated Hankel matrix-vector products** - O(N log N) instead of O(N²)
- **Block power method** - Process multiple singular vectors simultaneously
- **Two backends**:
  - Reference implementation (portable C)
  - MKL-optimized (Intel processors, 2-3x faster)
- **Header-only design** - Just include `ssa_opt.h`
- **Batched FFT operations** - Efficient use of SIMD
- **Rayleigh-Ritz extraction** - Accurate singular value computation

## Algorithm

### Step 1: Trajectory Matrix (Hankel Embedding)

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

### Step 2: Singular Value Decomposition

We compute the top-k singular triplets of H:

```
H ≈ Σᵢ σᵢ · uᵢ · vᵢᵀ    for i = 0, 1, ..., k-1
```

Where:
- `σᵢ` = singular values (importance/strength of component)
- `uᵢ` = left singular vectors (L-dimensional)
- `vᵢ` = right singular vectors (K-dimensional)

Direct SVD is O(L × K × min(L,K)) ≈ O(N³) - prohibitively expensive. Instead, we use power iteration which only requires matrix-vector products.

### Step 3: Power Iteration

To find the dominant singular triplet, we iterate:

```
v⁽⁰⁾ = random unit vector (length K)

repeat:
    u = H · v        # L × K matrix times K-vector → L-vector
    u = u / ||u||    # normalize
    
    v = Hᵀ · u       # K × L matrix times L-vector → K-vector  
    σ = ||v||        # singular value
    v = v / σ        # normalize
    
until converged
```

Each iteration requires one `H·v` and one `Hᵀ·u` operation. For k components, we use deflation: after finding each triplet, we orthogonalize against previous ones.

### Step 4: Reconstruction (Diagonal Averaging)

To reconstruct a time series component from singular triplet (σ, u, v):

1. Form rank-1 matrix: `Xᵢ = σ · u · vᵀ`
2. Average along anti-diagonals to get 1D signal

```
For each time point t = 0, 1, ..., N-1:
    x̃[t] = average of all Xᵢ[i,j] where i + j = t
```

This "hankelization" projects the rank-1 matrix back to a valid trajectory matrix structure.

## Optimizations

### Optimization 1: FFT-Accelerated Matrix-Vector Product

The key insight: Hankel matrix-vector multiplication is equivalent to convolution.

**Forward product `y = H · v`:**
```
y[i] = Σⱼ H[i,j] · v[j] = Σⱼ x[i+j] · v[j]
```
This is the convolution of `x` with `v`, extracting elements `[0:L]`.

**Adjoint product `z = Hᵀ · u`:**
```
z[j] = Σᵢ H[i,j] · u[i] = Σᵢ x[i+j] · u[i]
```
This is the convolution of `x` with `reversed(u)`, extracting elements `[L-1:L-1+K]`.

**FFT Convolution:**
```
conv(a, b) = IFFT(FFT(a) ⊙ FFT(b))
```

Where `⊙` is element-wise multiplication. This reduces O(N²) direct convolution to O(N log N).

**Implementation details:**
- Pad to next power of 2: `fft_len = 2^ceil(log2(N + K - 1))`
- Pre-compute `FFT(signal)` once during initialization
- Each matvec: one FFT of vector, one element-wise multiply, one IFFT

### Optimization 2: Block Power Method

Instead of computing singular vectors one at a time, process blocks of `b` vectors simultaneously:

**Sequential method:**
```
for each component i = 0 to k-1:
    for iter = 0 to max_iter-1:
        u = H · v           # single FFT
        v = Hᵀ · u          # single FFT
    orthogonalize against previous components
```

**Block method:**
```
for each block of b components:
    V_block = [v₀, v₁, ..., v_{b-1}]    # K × b matrix
    
    for iter = 0 to max_iter-1:
        U_block = H · V_block           # batched FFT (b transforms)
        QR(U_block)                     # orthonormalize columns
        V_block = Hᵀ · U_block          # batched FFT (b transforms)
        QR(V_block)                     # orthonormalize columns
    
    Rayleigh-Ritz refinement
```

**Why it's faster:**
1. Batched FFT: MKL processes 32 transforms with single function call, better SIMD utilization
2. Memory locality: V_block columns are contiguous, better cache behavior
3. BLAS3 operations: Orthogonalization uses GEMM instead of GEMV

### Optimization 3: Periodic QR Orthogonalization

Full QR factorization every iteration is expensive. In practice, orthogonality drifts slowly:

```
QR every iteration:     100 × 2 = 200 QR calls per block
QR every 5 iterations:   20 × 2 =  40 QR calls per block (5x reduction)
```

Numerical stability is maintained by:
- Final QR before Rayleigh-Ritz extraction
- Deflation orthogonalization against previous components every iteration

### Optimization 4: Rayleigh-Ritz Extraction

After block power iteration converges, vectors span the correct subspace but may not be individually optimal (especially for nearly-degenerate singular values). Rayleigh-Ritz refines them:

```
1. Compute small matrix: M = U_blockᵀ · (H · V_block)    # b × b
2. SVD(M) = U_small · Σ · V_smallᵀ
3. Rotate: U_final = U_block · U_small
           V_final = V_block · V_small
4. Singular values: diagonal of Σ
```

Cost: One b×b SVD (negligible for b=32) gives optimal singular values within the subspace.

### Optimization 5: GEMM-Based Operations

All multi-vector operations use BLAS Level 3 (matrix-matrix) instead of Level 2 (matrix-vector):

| Operation | BLAS2 (sequential) | BLAS3 (block) |
|-----------|-------------------|---------------|
| Orthogonalize against previous | `k × GEMV` | Single `GEMM` |
| Project out components | `k × AXPY` | Single `GEMM` |
| Normalize columns | `b × NRM2 + SCAL` | Batched |

GEMM achieves near-peak FLOPS due to better cache reuse and SIMD utilization.

## Performance Summary

### Optimization Gains Table

| Optimization | Technique | Complexity Reduction | Typical Speedup |
|--------------|-----------|---------------------|-----------------|
| FFT matvec | Convolution via FFT | O(N²) → O(N log N) | 10-100x |
| Batched FFT | Process b vectors together | b FFT calls → 1 batch call | 1.5-2x |
| Periodic QR | QR every 5 iters, not every | 200 → 40 QR calls | 1.3-1.5x |
| Block method | GEMM instead of GEMV | Better cache/SIMD | 1.2-1.5x |
| Rayleigh-Ritz | Small SVD instead of re-iteration | Faster convergence | 1.1-1.2x |
| **Combined** | All above | | **2-3x vs sequential FFT** |

### Benchmark Results

Intel Core i7, Windows, MSVC, MKL 2025.3:

| N | L | k | Sequential | Block (b=32) | Speedup |
|---|---|---|------------|--------------|---------|
| 1,000 | 400 | 20 | 36 ms | 80 ms | 0.5x (overhead dominates) |
| 5,000 | 2,000 | 32 | 548 ms | 226 ms | **2.4x** |
| 10,000 | 4,000 | 32 | ~2.2 s | ~0.8 s | **~2.8x** |
| 20,000 | 8,000 | 32 | ~9 s | ~3 s | **~3x** |

**Crossover point:** Block method wins when N > ~3000 and k > ~16.

### Operation Count Analysis

For N=5000, L=2000, K=3001, k=32, max_iter=100:

**Sequential:**
```
FFT calls: k × max_iter × 2 = 32 × 100 × 2 = 6,400 individual FFTs
Orthogonalization: k × (k-1)/2 × max_iter GEMV operations
```

**Block (b=32):**
```
Batched FFT calls: ceil(k/b) × max_iter × 2 = 1 × 100 × 2 = 200 batched operations
QR factorizations: ceil(k/b) × (max_iter/5) × 2 = 1 × 20 × 2 = 40 QR calls
Orthogonalization: GEMM operations (much faster than GEMV)
```

## API Reference

### Initialization

```c
int ssa_opt_init(SSA_Opt* ssa, const double* data, int N, int L);
```
- `ssa` - Pointer to SSA_Opt structure
- `data` - Input time series (copied internally)
- `N` - Signal length
- `L` - Window length (embedding dimension). Typical choice: N/3 to N/2

Returns 0 on success, -1 on failure.

### Decomposition

```c
// Sequential power iteration (one component at a time)
int ssa_opt_decompose(SSA_Opt* ssa, int k, int max_iter);

// Block power iteration (multiple components simultaneously)  
int ssa_opt_decompose_block(SSA_Opt* ssa, int k, int block_size, int max_iter);
```
- `k` - Number of singular triplets to compute
- `block_size` - Block size for batched computation (use 32 for MKL, or 0 for auto)
- `max_iter` - Maximum power iterations (50-100 typically sufficient)

After decomposition, access results via:
- `ssa->sigma[i]` - Singular values (descending order)
- `ssa->eigenvalues[i]` - Squared singular values (variance explained)
- `ssa->total_variance` - Sum of all eigenvalues
- `ssa->U[i * L]` - Left singular vector i (length L)
- `ssa->V[i * K]` - Right singular vector i (length K)

### Reconstruction

```c
int ssa_opt_reconstruct(SSA_Opt* ssa, const int* group, int group_size, double* result);
```
- `group` - Array of component indices to include
- `group_size` - Number of components in group
- `result` - Output buffer (length N, caller-allocated)

### Cleanup

```c
void ssa_opt_free(SSA_Opt* ssa);
```

## Choosing Parameters

### Window Length (L)

| L value | Trade-off |
|---------|-----------|
| N/2 | Maximum frequency resolution, slower |
| N/3 | Good balance (recommended default) |
| N/4 | Faster, may miss long-period components |

**Rule:** L should contain at least one full period of the slowest oscillation you want to extract.

### Number of Components (k)

- Start with k = 10-20 and examine singular value spectrum
- Look for gaps: large drop indicates separation between signal and noise
- Periodic signals appear as pairs with similar singular values (sine/cosine)
- Trend typically dominates as component 0

### Block Size

| Backend | Recommended block_size |
|---------|----------------------|
| MKL | 32 (matches SSA_BATCH_SIZE) |
| Reference | Use sequential method |

## Building

### Prerequisites

**Required:**
- C compiler with C99 support (MSVC 2019+, GCC 7+, Clang 5+)
- CMake 3.15+

**For MKL backend (recommended):**
- Intel oneAPI Base Toolkit (includes MKL)
- Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

### Windows with Intel MKL

Building on Windows with MKL can be tricky due to DLL dependencies. Here's the recommended approach:

#### Step 1: Install Intel oneAPI

1. Download Intel oneAPI Base Toolkit from Intel's website
2. Run installer, select "Intel oneAPI Math Kernel Library" at minimum
3. Default install path: `C:\Program Files (x86)\Intel\oneAPI\`

#### Step 2: Build with CMake

```powershell
cd Singular-Spectrum-Analysis
mkdir build
cd build
cmake .. -DSSA_USE_MKL=ON
cmake --build . --config Release
```

#### Step 3: Handle DLL Dependencies (Important!)

The executable needs MKL DLLs at runtime. **Easiest method - copy DLLs to your build folder:**

```powershell
# From your build\MKL\Release folder, copy these DLLs:
copy "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin\mkl_core.2.dll" .
copy "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin\mkl_intel_thread.2.dll" .
copy "C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin\mkl_intel_lp64.2.dll" .
copy "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\libiomp5md.dll" .
```

Adjust version numbers (2025.3) to match your installation.

**Alternative - Add to System PATH:**

Add these directories to your PATH environment variable:
- `C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin`
- `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin`

**Troubleshooting:**

If you see error code `0xC0000135` (DLL not found):
1. Run `where mkl_core.2.dll` to check if DLLs are findable
2. Use Dependency Walker or `dumpbin /dependents yourprogram.exe` to see missing DLLs
3. Ensure you're running from a directory with the DLLs or they're in PATH

### Linux with Intel MKL

```bash
# Install oneAPI (Ubuntu/Debian)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt install intel-oneapi-mkl-devel

# Source the environment
source /opt/intel/oneapi/setvars.sh

# Build
mkdir build && cd build
cmake .. -DSSA_USE_MKL=ON
make -j$(nproc)
```

### Building without MKL (Reference Implementation)

If you don't have MKL, the library falls back to a portable reference implementation:

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Note: Reference implementation is significantly slower for large signals.

## Limitations

- **Memory**: Stores L×k for U vectors and K×k for V vectors
- **Not suitable for**: Very short signals (N < 100) or streaming data
- **MKL dependency**: Best performance requires Intel MKL
- **Block method overhead**: For small problems (N < 2000), sequential is faster

## Future Work

- [ ] Real-to-complex FFT (r2c/c2r) for 2x FFT speedup
- [ ] Convergence detection for early termination
- [ ] Multi-threaded diagonal averaging
- [ ] GPU acceleration (cuFFT)
- [ ] Forecasting/prediction API

## References

1. Golyandina, N., & Zhigljavsky, A. (2013). *Singular Spectrum Analysis for Time Series*. Springer.
2. Korobeynikov, A. (2010). Computation- and space-efficient implementation of SSA. *Statistics and Its Interface*, 3(3), 357-368.
3. Intel oneAPI Math Kernel Library Documentation: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/

## License

MIT License - See LICENSE file for details.
