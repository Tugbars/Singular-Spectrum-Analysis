# SSA Forecasting: LRF and V-Forecast

Two methods for extending SSA-decomposed signals into the future.

## The Key Property

SSA components satisfy a **Linear Recurrence Formula (LRF)**:

```
x[n] = a₁·x[n-1] + a₂·x[n-2] + ... + a_{L-1}·x[n-L+1]
```

This arises from the structure of the trajectory matrix: if rank(H) = r, then any r+1 consecutive rows are linearly dependent.

## R-Forecast (Default)

### Computing LRF Coefficients

Given left singular vectors U = [u₀, u₁, ..., u_{r-1}], each of length L:

```
Let πᵢ = uᵢ[L-1]         (last element of each vector)
Let Uᵗ = uᵢ[0:L-1]       (first L-1 elements, "truncated")

ν² = Σᵢ πᵢ²              (verticality coefficient)

If ν² < 1 (non-degenerate case):
    a = (1/(1-ν²)) · Σᵢ πᵢ · Uᵗ[:,i]
```

The vector `a` has length L-1 and contains the LRF coefficients.

### Forecasting Loop

```c
// Reconstruct signal first, then extend
for (int step = 0; step < n_forecast; step++) {
    double x_new = 0.0;
    for (int j = 0; j < L - 1; j++) {
        x_new += lrf.coeffs[j] * signal[current_len - L + 1 + j];
    }
    forecast[step] = x_new;
    signal[current_len++] = x_new;  // Extend for next step
}
```

**Complexity**: O(L) per forecast step.

## V-Forecast (Alternative)

V-forecast recomputes the projection at each step instead of using precomputed LRF coefficients.

### Algorithm

At each step, given the last L-1 values z = [x_{n-L+1}, ..., x_{n-1}]:

```
For each component i in group:
    Uᵗᵢ = U[i, 0:L-1]     (truncated left singular vector)
    πᵢ = U[i, L-1]        (last element)
    
x_new = (1/(1-ν²)) · Σᵢ πᵢ · (Uᵗᵢ · z)
```

### Implementation

```c
int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group,
                      int n_forecast, double *output) {
    // Reconstruct signal from group
    double *signal = reconstruct(ssa, group, n_group);
    
    // Pre-compute ν² = Σ πᵢ²
    double nu_sq = 0.0;
    for (int g = 0; g < n_group; g++) {
        int idx = group[g];
        double pi = ssa->U[idx * L + (L - 1)];  // Last element
        nu_sq += pi * pi;
    }
    double scale = 1.0 / (1.0 - nu_sq);
    
    // Forecast loop
    for (int step = 0; step < n_forecast; step++) {
        double *z = &signal[N + step - (L - 1)];  // Last L-1 values
        double x_new = 0.0;
        
        for (int g = 0; g < n_group; g++) {
            int idx = group[g];
            const double *u_trunc = &ssa->U[idx * L];  // First L-1 elements
            double pi = ssa->U[idx * L + (L - 1)];
            
            // Inner product: Uᵗᵢ · z
            double inner = cblas_ddot(L - 1, u_trunc, 1, z, 1);
            x_new += pi * inner;
        }
        
        output[step] = scale * x_new;
        signal[N + step] = output[step];  // Extend for next step
    }
}
```

## R-Forecast vs V-Forecast

| Aspect | R-Forecast | V-Forecast |
|--------|------------|------------|
| Speed | Faster (single dot product/step) | Slower (n_group dot products/step) |
| Single-step | Identical result | Identical result |
| Multi-step | Accumulates via LRF | Recomputes projection |
| Long horizon | May drift/explode | More stable |
| When to use | Default, < 100 steps | > 100 steps, instability |

## Mathematical Equivalence (Single Step)

For a single forecast step, R-forecast and V-forecast are mathematically identical:

```
R-forecast:  x_new = a · z  where a = (1/(1-ν²)) · Σᵢ πᵢ · Uᵗᵢ

V-forecast:  x_new = (1/(1-ν²)) · Σᵢ πᵢ · (Uᵗᵢ · z)
           = (1/(1-ν²)) · (Σᵢ πᵢ · Uᵗᵢ) · z
           = a · z
```

The difference emerges in multi-step forecasting due to numerical accumulation.

## Verticality Coefficient ν²

The value ν² = Σᵢ πᵢ² indicates forecast reliability:

| ν² | Meaning |
|----|---------|
| Close to 0 | Safe, stable forecast |
| Close to 1 | Unstable, signal doesn't fit LRF well |
| = 1 | Degenerate, no valid forecast |

If ν² ≥ 1, the function returns an error (forecast impossible).

## Implementation Notes

### R-Forecast Full

`ssa_opt_forecast_full` returns reconstruction + forecast concatenated:

```c
// Output has length N + n_forecast
// [0:N] = reconstruction of group
// [N:N+n_forecast] = forecast
```

### V-Forecast Fast

`ssa_opt_vforecast_fast` takes pre-reconstructed signal as input, avoiding redundant reconstruction:

```c
// For streaming: reconstruct once, forecast multiple times
double *recon = ssa_opt_reconstruct(ssa, group, ...);
ssa_opt_vforecast_fast(ssa, group, n_group, recon, N, n_forecast, output);
```
