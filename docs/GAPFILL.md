# Gap Filling: Handling Missing Values

Reconstruct missing data points using SSA's low-rank structure.

## The Problem

Time series often have gaps:
- Weekend/holiday gaps in financial data
- Sensor dropouts
- Corrupted measurements
- Irregular sampling

SSA can fill these gaps because the underlying signal has low-rank structure that spans the missing regions.

## Two Methods

### 1. Iterative Gap Filling (Recommended)

Repeatedly apply SSA until gap values converge:

```
Input: x with NaN at gap positions
1. Initialize gaps with linear interpolation
2. Repeat until convergence:
   a. Decompose x with SSA
   b. Reconstruct using rank-r approximation
   c. Update ONLY gap positions with new values
   d. Check convergence: ||x_gaps_new - x_gaps_old|| / ||x_gaps_old|| < tol
3. Return filled signal
```

**Why it works**: Each iteration projects the signal onto the low-rank subspace. Non-gap values anchor the solution while gap values adjust to be consistent with the global structure.

### 2. Simple Gap Filling (Faster)

Forecast from left, backcast from right, blend:

```
Input: x with gap at positions [a, b]

1. Left segment: x[0:a]
   - Fit SSA on left segment
   - Forecast into gap region

2. Right segment: x[b:N] (reversed)
   - Fit SSA on reversed right segment  
   - Forecast backward into gap region

3. Blend with position-based weights:
   - Weight toward left forecast near position a
   - Weight toward right forecast near position b
   - Linear interpolation of weights across gap
```

**When to use**: Single gaps, speed critical, approximate fill acceptable.

## Algorithm Details: Iterative Method

### Initialization

Linear interpolation provides a reasonable starting point:

```c
static void ssa_linear_interp_gaps(double *x, const int *gap_pos, int n_gaps, int N) {
    for (int g = 0; g < n_gaps; g++) {
        int pos = gap_pos[g];
        
        // Find nearest non-gap neighbors
        int left = pos - 1, right = pos + 1;
        while (left >= 0 && ssa_is_nan(x[left])) left--;
        while (right < N && ssa_is_nan(x[right])) right++;
        
        // Interpolate
        if (left >= 0 && right < N) {
            double t = (double)(pos - left) / (right - left);
            x[pos] = (1 - t) * x[left] + t * x[right];
        } else if (left >= 0) {
            x[pos] = x[left];  // Extrapolate from left
        } else if (right < N) {
            x[pos] = x[right]; // Extrapolate from right
        } else {
            x[pos] = 0.0;      // No neighbors (shouldn't happen)
        }
    }
}
```

### Main Loop

```c
int ssa_opt_gapfill(double *x, int N, int L, int rank, 
                    int max_iter, double tol, SSA_GapFillResult *result) {
    // 1. Identify gap positions (where x[t] is NaN)
    int *gap_pos = find_nan_positions(x, N, &n_gaps);
    
    // 2. Initialize gaps
    ssa_linear_interp_gaps(x, gap_pos, n_gaps, N);
    
    // 3. Store previous gap values for convergence check
    double *prev_gaps = alloc(n_gaps);
    for (int i = 0; i < n_gaps; i++) 
        prev_gaps[i] = x[gap_pos[i]];
    
    // 4. Iterate
    SSA_Opt ssa = {0};
    ssa_opt_init(&ssa, x, N, L);
    ssa_opt_prepare(&ssa, rank, 8);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Update SSA with current signal
        ssa_opt_update_signal(&ssa, x);
        
        // Decompose and reconstruct
        ssa_opt_decompose_randomized(&ssa, rank, 8);
        int group[rank];
        for (int i = 0; i < rank; i++) group[i] = i;
        double *recon = alloc(N);
        ssa_opt_reconstruct(&ssa, group, rank, recon);
        
        // Update ONLY gap positions
        for (int i = 0; i < n_gaps; i++) {
            x[gap_pos[i]] = recon[gap_pos[i]];
        }
        
        // Check convergence
        double sum_sq_diff = 0, sum_sq_old = 0;
        for (int i = 0; i < n_gaps; i++) {
            double diff = x[gap_pos[i]] - prev_gaps[i];
            sum_sq_diff += diff * diff;
            sum_sq_old += prev_gaps[i] * prev_gaps[i];
        }
        double rel_change = sqrt(sum_sq_diff / (sum_sq_old + 1e-12));
        
        if (rel_change < tol) {
            result->converged = 1;
            result->iterations = iter + 1;
            result->final_diff = rel_change;
            break;
        }
        
        // Save for next iteration
        for (int i = 0; i < n_gaps; i++)
            prev_gaps[i] = x[gap_pos[i]];
    }
    
    ssa_opt_free(&ssa);
    return 0;
}
```

### Convergence

The iteration converges when gap values stabilize:

```
rel_change = ||x_gaps^(k+1) - x_gaps^(k)||₂ / ||x_gaps^(k)||₂
```

Typical convergence: 5-20 iterations for tol=1e-6.

## Algorithm Details: Simple Method

```c
int ssa_opt_gapfill_simple(double *x, int N, int L, int rank, 
                            SSA_GapFillResult *result) {
    // Find gap region [gap_start, gap_end)
    int gap_start = -1, gap_end = -1;
    for (int i = 0; i < N; i++) {
        if (ssa_is_nan(x[i])) {
            if (gap_start < 0) gap_start = i;
            gap_end = i + 1;
        }
    }
    int gap_len = gap_end - gap_start;
    
    // === LEFT FORECAST ===
    // Fit SSA on x[0:gap_start]
    SSA_Opt ssa_left = {0};
    ssa_opt_init(&ssa_left, x, gap_start, L);
    ssa_opt_decompose_randomized(&ssa_left, rank, 8);
    
    // Forecast gap_len points forward
    int group[rank];
    for (int i = 0; i < rank; i++) group[i] = i;
    double *left_forecast = alloc(gap_len);
    ssa_opt_forecast(&ssa_left, group, rank, gap_len, left_forecast);
    
    // === RIGHT BACKCAST ===
    // Reverse right segment: x[gap_end:N]
    int right_len = N - gap_end;
    double *right_rev = alloc(right_len);
    for (int i = 0; i < right_len; i++)
        right_rev[i] = x[N - 1 - i];
    
    // Fit SSA on reversed segment
    SSA_Opt ssa_right = {0};
    ssa_opt_init(&ssa_right, right_rev, right_len, L);
    ssa_opt_decompose_randomized(&ssa_right, rank, 8);
    
    // Forecast (which is backcast in original orientation)
    double *right_forecast_rev = alloc(gap_len);
    ssa_opt_forecast(&ssa_right, group, rank, gap_len, right_forecast_rev);
    
    // Reverse back
    double *right_forecast = alloc(gap_len);
    for (int i = 0; i < gap_len; i++)
        right_forecast[i] = right_forecast_rev[gap_len - 1 - i];
    
    // === BLEND ===
    // Linear weight: w=0 at gap_start (use left), w=1 at gap_end (use right)
    for (int i = 0; i < gap_len; i++) {
        double w = (double)i / (gap_len - 1);  // 0 to 1
        x[gap_start + i] = (1 - w) * left_forecast[i] + w * right_forecast[i];
    }
    
    result->iterations = 1;
    result->converged = 1;
    result->n_gaps = gap_len;
}
```

## Choosing Parameters

### Window Length L

Should be large enough to capture signal structure but leave enough non-gap data:

```
L ≤ (N - n_gaps) / 2
```

Rule of thumb: `L = min(N/4, dominant_period * 2)`

### Rank

Should capture the signal but not noise:
- Too low: Underfits, fills with overly smooth values
- Too high: Overfits, fills with noise

Rule of thumb: `rank = 2 * (number of periodic components) + 1`

### Convergence Tolerance

| tol | Iterations | Use Case |
|-----|------------|----------|
| 1e-3 | 3-5 | Quick approximation |
| 1e-6 | 10-20 | Default, good accuracy |
| 1e-9 | 20-50 | High precision |

## Practical Examples

### Weekend Gaps in Financial Data

```python
# Daily prices with NaN for weekends
prices = load_prices()  # Has NaN on Sat/Sun

# Fill weekends
result = gapfill(prices, L=100, rank=6, max_iter=20, tol=1e-6)
prices_filled = result.signal

# Now have smooth 7-day-a-week series
```

### Sensor Dropout

```python
# Temperature sensor with random dropouts
temp = np.array([20.1, 20.3, np.nan, np.nan, 20.8, 20.9, ...])

# Fill dropouts
result = gapfill(temp, L=50, rank=4)
print(f"Filled {result.n_gaps} gaps in {result.iterations} iterations")
```

### Multiple Gap Regions

Iterative method handles multiple gaps naturally:

```python
# Signal with several gap regions
x = np.sin(np.linspace(0, 10*np.pi, 500))
x[50:60] = np.nan    # Gap 1
x[150:170] = np.nan  # Gap 2
x[300:310] = np.nan  # Gap 3

result = gapfill(x, L=100, rank=4)
# All gaps filled simultaneously
```

## Comparison

| Aspect | Iterative | Simple |
|--------|-----------|--------|
| Accuracy | Higher | Lower |
| Speed | Slower (10-20× SSA) | Fast (2× SSA) |
| Multiple gaps | Natural | Needs adaptation |
| Edge gaps | Works | Fails (no forecast anchor) |
| Convergence check | Yes | N/A |

## Limitations

1. **Requires sufficient non-gap data**: At minimum `N - n_gaps ≥ L + rank`

2. **Assumes low-rank structure**: Random noise cannot be filled meaningfully

3. **Large gaps degrade quality**: If gap > L, the method extrapolates blindly

4. **Edge gaps are harder**: No anchor data on one side (simple method fails)

## References

- Golyandina, N. & Osipov, E. (2007). "The 'Caterpillar'-SSA method for analysis of time series with missing values." *Journal of Statistical Planning and Inference*.
- Kondrashov, D. & Ghil, M. (2006). "Spatio-temporal filling of missing points in geophysical data sets." *Nonlinear Processes in Geophysics*.
