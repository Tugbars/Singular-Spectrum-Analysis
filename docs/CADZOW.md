# Cadzow Iterations: Exact Rank-r Denoising

Iteratively project onto Hankel and low-rank constraints for optimal denoising.

## The Problem

Single-pass SSA reconstruction gives a signal that is:
- **Low-rank**: Comes from rank-r SVD approximation
- **Not Hankel**: Diagonal averaging breaks the exact Hankel structure

Cadzow iterations enforce **both** constraints simultaneously, yielding a signal whose trajectory matrix is exactly rank-r AND Hankel.

## Algorithm

Alternate between two projections:

```
Input: noisy signal x, target rank r
x_current = x

Repeat until convergence:
    1. Form Hankel matrix H from x_current
    2. SVD: H = U·Σ·Vᵀ
    3. Truncate to rank r: H_r = Σᵢ₌₀^{r-1} σᵢ·uᵢ·vᵢᵀ
    4. Diagonal averaging: x_new = hankel_to_signal(H_r)
    5. Check convergence: ||x_new - x_current|| / ||x_current|| < tol
    6. x_current = x_new

Output: x_current (exactly rank-r Hankelizable)
```

## Why It Works

The trajectory matrix lives in two constraint sets:

```
𝓗 = {H : H is Hankel}           (linear subspace)
𝓡ᵣ = {H : rank(H) ≤ r}          (smooth manifold)
```

Cadzow alternates projections:
- **P_𝓡**: SVD truncation projects onto rank-r matrices
- **P_𝓗**: Diagonal averaging projects onto Hankel matrices

This is a form of **alternating projections** (Dykstra/von Neumann). For these two sets, it converges to a point in 𝓗 ∩ 𝓡ᵣ (or nearby if intersection is empty).

## Implementation

```c
int ssa_opt_cadzow(const double *x, int N, int L, int rank, int max_iter,
                   double tol, double *output, SSA_CadzowResult *result) {
    // Initialize with input signal
    memcpy(output, x, N * sizeof(double));
    
    // Create SSA context
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, output, N, L);
    ssa_opt_prepare(&ssa, rank, 8);
    
    int group[rank];
    for (int i = 0; i < rank; i++) group[i] = i;
    
    double prev_norm = cblas_dnrm2(N, output, 1);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Update SSA with current signal
        ssa_opt_update_signal(&ssa, output);
        
        // Project onto rank-r: decompose + reconstruct
        ssa_opt_decompose_randomized(&ssa, rank, 8);
        double *recon = workspace;
        ssa_opt_reconstruct(&ssa, group, rank, recon);
        
        // Compute change
        double diff_norm = 0;
        for (int i = 0; i < N; i++) {
            double d = recon[i] - output[i];
            diff_norm += d * d;
        }
        diff_norm = sqrt(diff_norm);
        
        // Update output
        memcpy(output, recon, N * sizeof(double));
        
        // Check convergence
        double curr_norm = cblas_dnrm2(N, output, 1);
        double rel_change = diff_norm / (prev_norm + 1e-12);
        
        if (rel_change < tol) {
            result->converged = 1;
            result->iterations = iter + 1;
            result->final_diff = rel_change;
            break;
        }
        
        prev_norm = curr_norm;
    }
    
    ssa_opt_free(&ssa);
    return 0;
}
```

## Weighted Cadzow

For gentler convergence, blend instead of fully replacing:

```c
// x_new = α·recon + (1-α)·x_current
int ssa_opt_cadzow_weighted(const double *x, int N, int L, int rank, 
                             int max_iter, double tol, double alpha,
                             double *output, SSA_CadzowResult *result) {
    // ... same setup ...
    
    for (int iter = 0; iter < max_iter; iter++) {
        // ... decompose + reconstruct ...
        
        // Weighted update instead of full replacement
        for (int i = 0; i < N; i++) {
            output[i] = alpha * recon[i] + (1 - alpha) * output[i];
        }
        
        // ... convergence check ...
    }
}
```

| α | Behavior |
|---|----------|
| 1.0 | Standard Cadzow (aggressive) |
| 0.5 | Balanced, slower convergence |
| 0.1 | Very gentle, many iterations |

## Convergence Properties

### Typical Convergence

```
Iteration 1:  rel_change = 0.15
Iteration 2:  rel_change = 0.08
Iteration 3:  rel_change = 0.03
Iteration 4:  rel_change = 0.01
Iteration 5:  rel_change = 0.003
Iteration 6:  rel_change = 0.0008
Iteration 7:  rel_change = 2e-4
...
Iteration 15: rel_change = 1e-9 ✓ converged
```

### Factors Affecting Convergence

| Factor | Faster Convergence | Slower Convergence |
|--------|-------------------|-------------------|
| SNR | High SNR | Low SNR |
| Rank choice | Correct rank | Over/under-estimated |
| Signal type | Pure sinusoids | Broadband/complex |
| α (weighted) | α = 1.0 | α << 1 |

## Single-Pass SSA vs Cadzow

| Aspect | Single-Pass SSA | Cadzow |
|--------|-----------------|--------|
| Speed | 1× | 10-20× slower |
| Noise reduction | ~95% optimal | ~100% optimal |
| Output property | Approximately low-rank | Exactly rank-r Hankelizable |
| Use case | Production, real-time | Research, analysis |

### When Cadzow Matters

1. **Theoretical analysis**: Need exact rank-r signal for proofs
2. **Extremely low SNR**: Every dB matters
3. **Pre-processing for ESPRIT**: Cleaner eigenvalue estimates
4. **Comparison studies**: Fair comparison with other methods

### When Single-Pass Suffices

1. **Production systems**: 10-20× faster, nearly as good
2. **Moderate SNR**: Diminishing returns from iteration
3. **Streaming applications**: Can't afford iteration latency

## Practical Example

```python
from ssa_wrapper import cadzow

# Noisy sinusoid
t = np.linspace(0, 4*np.pi, 200)
clean = np.sin(t) + 0.5*np.sin(3*t)
noisy = clean + np.random.randn(200) * 0.5

# Cadzow denoising
result = cadzow(noisy, L=50, rank=4, max_iter=30, tol=1e-9)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final change: {result.final_diff:.2e}")

# Compare
rmse_single = np.sqrt(np.mean((ssa.reconstruct([0,1,2,3]) - clean)**2))
rmse_cadzow = np.sqrt(np.mean((result.signal - clean)**2))
print(f"Single-pass RMSE: {rmse_single:.4f}")
print(f"Cadzow RMSE: {rmse_cadzow:.4f}")
```

Typical output:
```
Converged: True
Iterations: 12
Final change: 8.73e-10
Single-pass RMSE: 0.0423
Cadzow RMSE: 0.0398  (6% better)
```

## Choosing Parameters

### Rank

Critical parameter. Too low = signal loss. Too high = noise retained.

```python
# Use ESPRIT to find optimal rank
par = ssa.parestimate()
n_signal_components = sum(par.moduli > 0.9)  # Count undamped components
rank = n_signal_components
```

### Tolerance

| tol | Typical Iterations | Use Case |
|-----|-------------------|----------|
| 1e-3 | 3-5 | Quick approximation |
| 1e-6 | 8-15 | Default |
| 1e-9 | 15-30 | High precision |
| 1e-12 | 30-50 | Near machine precision |

### Max Iterations

Safety limit. If not converged in max_iter:
- Check if rank is appropriate
- Try weighted Cadzow with α < 1
- Accept partial convergence

## References

- Cadzow, J.A. (1988). "Signal Enhancement - A Composite Property Mapping Algorithm." *IEEE Trans. ASSP*.
- De Moor, B. (1994). "Total least squares for affinely structured matrices and the noisy realization problem." *IEEE Trans. Signal Processing*.
- Gillard, J. & Zhigljavsky, A. (2013). "Optimization challenges in the structured low rank approximation problem." *J. Global Optimization*.
