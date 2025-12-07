// ============================================================================
// SSA_OPT_FORECAST.H - SSA Forecasting Methods
// ============================================================================
// Part of the SSA optimization library. Requires ssa_opt_r2c.h
//
// Contents:
//   - Linear Recurrence Formula (LRF) forecasting
//   - Vector forecasting (V-forecast)
//   - Utility functions (get_trend, get_noise, variance_explained)
//
// Usage:
//   #define SSA_OPT_FORECAST_IMPLEMENTATION  // in ONE .c file
//   #include "ssa_opt_forecast.h"
//
// Precision:
//   #define SSA_USE_FLOAT before including for single precision
// ============================================================================

#ifndef SSA_OPT_FORECAST_H
#define SSA_OPT_FORECAST_H

#include "ssa_opt_r2c.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// API Declarations
// ============================================================================

// Linear Recurrence Formula (LRF) Forecasting
// Computes LRF coefficients R such that x[t] = Σ R[j] * x[t-L+1+j]
int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf);
void ssa_opt_lrf_free(SSA_LRF *lrf);

// Forecast using pre-computed LRF
int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const ssa_real *base_signal,
                              int base_len, int n_forecast, ssa_real *output);

// One-shot forecast: compute LRF + reconstruct + forecast
int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group,
                     int n_forecast, ssa_real *output);

// Full forecast: returns N + n_forecast values (reconstructed signal + forecast)
int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                          int n_forecast, ssa_real *output);

// Vector Forecast (V-forecast)
// Alternative to recurrent forecast - projects directly onto eigenvector subspace
int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group,
                      int n_forecast, ssa_real *output);
int ssa_opt_vforecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                           int n_forecast, ssa_real *output);

// Fast V-forecast for hot path - uses BLAS, works with external base signal
int ssa_opt_vforecast_fast(const SSA_Opt *ssa, const int *group, int n_group,
                           const ssa_real *base_signal, int base_len,
                           int n_forecast, ssa_real *output);

// Convenience Functions
int ssa_opt_get_trend(const SSA_Opt *ssa, ssa_real *output);
int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, ssa_real *output);
ssa_real ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end);
int ssa_opt_get_singular_values(const SSA_Opt *ssa, ssa_real *output, int max_n);
int ssa_opt_get_eigenvalues(const SSA_Opt *ssa, ssa_real *output, int max_n);
ssa_real ssa_opt_get_total_variance(const SSA_Opt *ssa);

// ============================================================================
// Implementation
// ============================================================================

#ifdef SSA_OPT_FORECAST_IMPLEMENTATION

void ssa_opt_lrf_free(SSA_LRF *lrf)
{
    if (!lrf)
        return;
    ssa_opt_free_ptr(lrf->R);
    memset(lrf, 0, sizeof(SSA_LRF));
}

int ssa_opt_compute_lrf(const SSA_Opt *ssa, const int *group, int n_group, SSA_LRF *lrf)
{
    if (!ssa || !ssa->decomposed || !group || !lrf || n_group < 1)
        return -1;

    int L = ssa->L;
    memset(lrf, 0, sizeof(SSA_LRF));
    lrf->L = L;
    lrf->valid = false;

    ssa_real *pi = (ssa_real *)ssa_opt_alloc(n_group * sizeof(ssa_real));
    if (!pi)
        return -1;

    ssa_real nu_sq = (ssa_real)0.0;
    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
        {
            ssa_opt_free_ptr(pi);
            return -1;
        }
        pi[g] = ssa->U[idx * L + (L - 1)];
        nu_sq += pi[g] * pi[g];
    }

    lrf->verticality = nu_sq;

    if (nu_sq >= (ssa_real)1.0 - (ssa_real)1e-10)
    {
        ssa_opt_free_ptr(pi);
        lrf->valid = false;
        return -1;
    }

    lrf->R = (ssa_real *)ssa_opt_alloc((L - 1) * sizeof(ssa_real));
    if (!lrf->R)
    {
        ssa_opt_free_ptr(pi);
        return -1;
    }

    ssa_real scale = (ssa_real)1.0 / ((ssa_real)1.0 - nu_sq);
    for (int j = 0; j < L - 1; j++)
    {
        ssa_real sum = (ssa_real)0.0;
        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            sum += pi[g] * ssa->U[idx * L + j];
        }
        lrf->R[j] = sum * scale;
    }

    ssa_opt_free_ptr(pi);
    lrf->valid = true;
    return 0;
}

int ssa_opt_forecast_with_lrf(const SSA_LRF *lrf, const ssa_real *base_signal,
                              int base_len, int n_forecast, ssa_real *output)
{
    if (!lrf || !lrf->valid || !lrf->R || !base_signal || !output)
        return -1;

    int L = lrf->L;
    if (base_len < L - 1 || n_forecast < 1)
        return -1;

    int window_size = L - 1;
    ssa_real *buffer = (ssa_real *)ssa_opt_alloc((window_size + n_forecast) * sizeof(ssa_real));
    if (!buffer)
        return -1;

    for (int i = 0; i < window_size; i++)
        buffer[i] = base_signal[base_len - window_size + i];

    for (int h = 0; h < n_forecast; h++)
    {
        ssa_real forecast = (ssa_real)0.0;
        for (int j = 0; j < window_size; j++)
            forecast += lrf->R[j] * buffer[h + j];
        buffer[window_size + h] = forecast;
        output[h] = forecast;
    }

    ssa_opt_free_ptr(buffer);
    return 0;
}

int ssa_opt_forecast(const SSA_Opt *ssa, const int *group, int n_group,
                     int n_forecast, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
        return -1;

    int N = ssa->N;

    SSA_LRF lrf = {0};
    if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
        return -1;

    ssa_real *reconstructed = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    if (!reconstructed)
    {
        ssa_opt_lrf_free(&lrf);
        return -1;
    }

    if (ssa_opt_reconstruct(ssa, group, n_group, reconstructed) != 0)
    {
        ssa_opt_free_ptr(reconstructed);
        ssa_opt_lrf_free(&lrf);
        return -1;
    }

    int result = ssa_opt_forecast_with_lrf(&lrf, reconstructed, N, n_forecast, output);

    ssa_opt_free_ptr(reconstructed);
    ssa_opt_lrf_free(&lrf);
    return result;
}

int ssa_opt_forecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                          int n_forecast, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
        return -1;

    int N = ssa->N;

    SSA_LRF lrf = {0};
    if (ssa_opt_compute_lrf(ssa, group, n_group, &lrf) != 0)
        return -1;

    if (ssa_opt_reconstruct(ssa, group, n_group, output) != 0)
    {
        ssa_opt_lrf_free(&lrf);
        return -1;
    }

    int result = ssa_opt_forecast_with_lrf(&lrf, output, N, n_forecast, output + N);

    ssa_opt_lrf_free(&lrf);
    return result;
}

// ========================================================================
// Vector Forecast (V-forecast)
// ========================================================================
// Alternative to recurrent forecast. Instead of precomputing LRR coefficients,
// V-forecast projects directly onto eigenvector subspace at each step.
//
// Formula: x_new = (1/(1-ν²)) * Σ_i π_i * (U_trunc_i · z)
// where:
//   z = last L-1 values of signal
//   U_trunc_i = eigenvector i without last element
//   π_i = last element of eigenvector i
//   ν² = Σ π_i²
// ========================================================================

int ssa_opt_vforecast(const SSA_Opt *ssa, const int *group, int n_group,
                      int n_forecast, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
        return -1;

    int N = ssa->N, L = ssa->L;

    // Compute verticality ν² = Σ π_i²
    ssa_real nu_sq = (ssa_real)0.0;
    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
            return -1;
        ssa_real pi_i = ssa->U[idx * L + (L - 1)];
        nu_sq += pi_i * pi_i;
    }

    if (nu_sq >= (ssa_real)1.0 - (ssa_real)1e-10)
        return -1;  // Not forecastable (vertical eigenvectors)

    ssa_real scale = (ssa_real)1.0 / ((ssa_real)1.0 - nu_sq);

    // Get reconstructed signal and allocate space for forecasts
    ssa_real *signal = (ssa_real *)ssa_opt_alloc((N + n_forecast) * sizeof(ssa_real));
    if (!signal)
        return -1;

    if (ssa_opt_reconstruct(ssa, group, n_group, signal) != 0)
    {
        ssa_opt_free_ptr(signal);
        return -1;
    }

    // V-forecast loop
    for (int h = 0; h < n_forecast; h++)
    {
        ssa_real *z = &signal[N + h - (L - 1)];  // Last L-1 values
        ssa_real x_new = (ssa_real)0.0;

        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            const ssa_real *u_col = &ssa->U[idx * L];
            ssa_real pi_i = u_col[L - 1];

            ssa_real dot = (ssa_real)0.0;
            for (int j = 0; j < L - 1; j++)
                dot += u_col[j] * z[j];

            x_new += pi_i * dot;
        }
        x_new *= scale;

        signal[N + h] = x_new;
        output[h] = x_new;
    }

    ssa_opt_free_ptr(signal);
    return 0;
}

int ssa_opt_vforecast_full(const SSA_Opt *ssa, const int *group, int n_group,
                           int n_forecast, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || !group || !output || n_group < 1 || n_forecast < 1)
        return -1;

    int N = ssa->N, L = ssa->L;

    // Compute verticality
    ssa_real nu_sq = (ssa_real)0.0;
    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
            return -1;
        ssa_real pi_i = ssa->U[idx * L + (L - 1)];
        nu_sq += pi_i * pi_i;
    }

    if (nu_sq >= (ssa_real)1.0 - (ssa_real)1e-10)
        return -1;

    ssa_real scale = (ssa_real)1.0 / ((ssa_real)1.0 - nu_sq);

    // Reconstruct into output buffer
    if (ssa_opt_reconstruct(ssa, group, n_group, output) != 0)
        return -1;

    // V-forecast loop
    for (int h = 0; h < n_forecast; h++)
    {
        ssa_real *z = &output[N + h - (L - 1)];
        ssa_real x_new = (ssa_real)0.0;

        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            const ssa_real *u_col = &ssa->U[idx * L];
            ssa_real pi_i = u_col[L - 1];

            ssa_real dot = (ssa_real)0.0;
            for (int j = 0; j < L - 1; j++)
                dot += u_col[j] * z[j];

            x_new += pi_i * dot;
        }
        x_new *= scale;
        output[N + h] = x_new;
    }

    return 0;
}

// Optimized V-forecast for hot path - uses BLAS for inner products
int ssa_opt_vforecast_fast(const SSA_Opt *ssa, const int *group, int n_group,
                           const ssa_real *base_signal, int base_len,
                           int n_forecast, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || !group || !base_signal || !output)
        return -1;
    if (n_group < 1 || n_forecast < 1 || base_len < ssa->L - 1)
        return -1;

    int L = ssa->L;

    // Precompute π values and ν²
    ssa_real *pi = (ssa_real *)ssa_opt_alloc(n_group * sizeof(ssa_real));
    if (!pi)
        return -1;

    ssa_real nu_sq = (ssa_real)0.0;
    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= ssa->n_components)
        {
            ssa_opt_free_ptr(pi);
            return -1;
        }
        pi[g] = ssa->U[idx * L + (L - 1)];
        nu_sq += pi[g] * pi[g];
    }

    if (nu_sq >= (ssa_real)1.0 - (ssa_real)1e-10)
    {
        ssa_opt_free_ptr(pi);
        return -1;
    }

    ssa_real scale = (ssa_real)1.0 / ((ssa_real)1.0 - nu_sq);

    // Working buffer for signal extension
    int window = L - 1;
    ssa_real *buffer = (ssa_real *)ssa_opt_alloc((window + n_forecast) * sizeof(ssa_real));
    if (!buffer)
    {
        ssa_opt_free_ptr(pi);
        return -1;
    }

    // Copy last L-1 values of base signal
    memcpy(buffer, &base_signal[base_len - window], window * sizeof(ssa_real));

    // V-forecast loop with BLAS
    for (int h = 0; h < n_forecast; h++)
    {
        ssa_real *z = &buffer[h];
        ssa_real x_new = (ssa_real)0.0;

        for (int g = 0; g < n_group; g++)
        {
            int idx = group[g];
            // BLAS dot product: U_trunc · z
            ssa_real dot = ssa_cblas_dot(window, &ssa->U[idx * L], 1, z, 1);
            x_new += pi[g] * dot;
        }
        x_new *= scale;

        buffer[window + h] = x_new;
        output[h] = x_new;
    }

    ssa_opt_free_ptr(pi);
    ssa_opt_free_ptr(buffer);
    return 0;
}

// ========================================================================
// Convenience Functions
// ========================================================================

int ssa_opt_get_trend(const SSA_Opt *ssa, ssa_real *output)
{
    int group[] = {0};
    return ssa_opt_reconstruct(ssa, group, 1, output);
}

int ssa_opt_get_noise(const SSA_Opt *ssa, int noise_start, ssa_real *output)
{
    if (!ssa || !ssa->decomposed || noise_start < 0)
        return -1;

    int n_noise = ssa->n_components - noise_start;
    if (n_noise <= 0)
    {
        ssa_opt_zero(output, ssa->N);
        return 0;
    }

    int *group = (int *)malloc(n_noise * sizeof(int));
    if (!group)
        return -1;

    for (int i = 0; i < n_noise; i++)
        group[i] = noise_start + i;

    int ret = ssa_opt_reconstruct(ssa, group, n_noise, output);
    free(group);
    return ret;
}

ssa_real ssa_opt_variance_explained(const SSA_Opt *ssa, int start, int end)
{
    if (!ssa || !ssa->decomposed || start < 0 || ssa->total_variance <= (ssa_real)0)
        return (ssa_real)0.0;

    if (end < 0 || end >= ssa->n_components)
        end = ssa->n_components - 1;

    ssa_real sum = (ssa_real)0.0;
    for (int i = start; i <= end; i++)
        sum += ssa->eigenvalues[i];

    return sum / ssa->total_variance;
}

int ssa_opt_get_singular_values(const SSA_Opt *ssa, ssa_real *output, int max_n)
{
    if (!ssa || !ssa->decomposed || !output)
        return -1;

    int n = (max_n < ssa->n_components) ? max_n : ssa->n_components;
    memcpy(output, ssa->sigma, n * sizeof(ssa_real));
    return n;
}

int ssa_opt_get_eigenvalues(const SSA_Opt *ssa, ssa_real *output, int max_n)
{
    if (!ssa || !ssa->decomposed || !output)
        return -1;

    int n = (max_n < ssa->n_components) ? max_n : ssa->n_components;
    memcpy(output, ssa->eigenvalues, n * sizeof(ssa_real));
    return n;
}

ssa_real ssa_opt_get_total_variance(const SSA_Opt *ssa)
{
    if (!ssa || !ssa->decomposed)
        return (ssa_real)0.0;
    return ssa->total_variance;
}

#endif // SSA_OPT_FORECAST_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_FORECAST_H
