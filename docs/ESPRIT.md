# ESPRIT: Frequency Estimation from SSA Eigenvectors

Automatically detect periods/frequencies in your signal.

## What ESPRIT Does

Given SSA eigenvectors, ESPRIT extracts the underlying signal frequencies without manual inspection. Useful for:

- Detecting market cycles (20-day, 60-day, etc.)
- Identifying seasonal patterns
- Setting optimal L (window should capture at least one full cycle)
- Filtering noise components from signal components

## The Key Property: Shift Invariance

For a sinusoidal signal, the left singular vectors U have a special structure:

```
If x[t] = A·cos(ωt + φ), then:
    u[t+1] / u[t] ≈ e^{iω}    (constant ratio)
```

This means shifting U by one row is approximately equivalent to multiplying by a diagonal matrix of complex exponentials.

## Algorithm

### Step 1: Form Shifted Matrices

Split each eigenvector into "up" and "down" parts:

```
U_up   = U[0:L-1, :]    (first L-1 rows)
U_down = U[1:L, :]      (last L-1 rows)
```

For pure sinusoids: `U_down ≈ U_up · diag(e^{iω₁}, e^{iω₂}, ...)`

### Step 2: Compute Shift Matrix

Solve the least-squares problem:

```
Z = pinv(U_up) · U_down
```

Z is an r×r matrix whose eigenvalues are the signal poles.

### Step 3: Eigenvalue Decomposition

```
eigenvalues(Z) = {λ₁, λ₂, ..., λᵣ}
```

Each eigenvalue λ = |λ| · e^{iθ} gives:
- **Frequency**: f = θ / (2π) cycles per sample
- **Period**: T = 1/|f| samples (Inf for DC/trend)
- **Modulus**: |λ| (1.0 = undamped sinusoid, <1 = decaying)
- **Rate**: log(|λ|) (0 = undamped, negative = decaying)

## Implementation

```c
int ssa_opt_parestimate(const SSA_Opt *ssa, const int *group, int n_group, 
                         SSA_ParEstimate *result) {
    int L = ssa->L;
    int r = (group && n_group > 0) ? n_group : ssa->n_components;
    
    // Build U matrix (L × r) from selected components
    double *U_mat = alloc(L * r);
    for (int i = 0; i < r; i++) {
        int idx = group ? group[i] : i;
        memcpy(&U_mat[i * L], &ssa->U[idx * L], L * sizeof(double));
    }
    
    // U_up = U[0:L-1, :], U_down = U[1:L, :]
    double *U_up = U_mat;           // First L-1 rows (reuse)
    double *U_down = U_mat + 1;     // Offset by 1 (same storage, strided)
    
    // Solve U_up · Z = U_down via least squares (DGELS)
    // Z is r × r
    LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', L-1, r, r, U_up, L, U_down, L);
    
    // Eigenvalue decomposition of Z (DGEEV for real non-symmetric)
    // Returns complex eigenvalues as (real, imag) pairs
    LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', r, Z, r, 
                  eigenvalues_real, eigenvalues_imag, NULL, 1, NULL, 1);
    
    // Convert eigenvalues to periods/frequencies
    for (int i = 0; i < r; i++) {
        double re = eigenvalues_real[i];
        double im = eigenvalues_imag[i];
        double modulus = sqrt(re*re + im*im);
        double argument = atan2(im, re);
        
        result->frequencies[i] = argument / (2 * M_PI);
        result->periods[i] = (fabs(argument) > 1e-10) ? 
                             fabs(2 * M_PI / argument) : INFINITY;
        result->moduli[i] = modulus;
        result->rates[i] = log(modulus);
    }
}
```

## Interpreting Results

### Modulus (Damping)

| Modulus | Interpretation |
|---------|----------------|
| ≈ 1.0 | Pure sinusoid (undamped) |
| 0.9 - 1.0 | Slowly decaying oscillation |
| < 0.9 | Rapidly decaying or noise |
| > 1.0 | Growing oscillation (unstable) |

### Period

| Period | Interpretation |
|--------|----------------|
| Inf | DC component (trend) |
| > L/2 | Well-captured cycle |
| 2-5 | High frequency (may be noise) |
| < 2 | Nyquist limit, likely noise |

### Filtering Strategy

```python
# Get only "real" periodic components (not noise)
periodic = par.get_periodic_components(
    min_period=5,      # Ignore very high frequencies
    max_period=None,   # No upper limit
    min_modulus=0.9    # Only undamped sinusoids
)

# These are your signal components
signal_group = list(periodic)

# Everything else is noise
noise_group = [i for i in range(k) if i not in signal_group]
```

## Practical Example

```python
# Detect cycles in daily stock prices
ssa = SSA(prices, L=120)  # ~6 months window
ssa.decompose(k=20)

par = ssa.parestimate()
print(par.summary())

# Output:
# Comp   Period    Modulus
# 0,1    60.2      0.98    → ~60-day cycle (quarterly)
# 2,3    20.1      0.97    → ~20-day cycle (monthly)
# 4,5    5.3       0.91    → ~weekly cycle
# 6-19   2-4       <0.5    → noise

# Use detected periods to set better L
dominant_period = par.periods[0]  # 60.2
optimal_L = int(2 * dominant_period)  # L=120 captures 2 full cycles
```

## Limitations

1. **Complex eigenvalues come in conjugate pairs**: For real signals, frequencies appear as ±f. The positive frequency is the physical one.

2. **Accuracy depends on rank selection**: If k is too low, you miss frequencies. If too high, noise pollutes estimates.

3. **Works best for stationary signals**: Non-stationary signals (changing frequency) give smeared estimates.

4. **Edge effects**: Short signals or L too small give poor estimates.

## References

- Roy, R. & Kailath, T. (1989). "ESPRIT - Estimation of Signal Parameters via Rotational Invariance Techniques." IEEE Trans. ASSP.
- Golyandina & Zhigljavsky (2013). "Singular Spectrum Analysis for Time Series." Chapter on frequency estimation.
