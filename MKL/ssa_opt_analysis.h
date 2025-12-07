// ============================================================================
// SSA_OPT_ANALYSIS.H - W-Correlation and Component Analysis
// ============================================================================
// Part of the SSA optimization library. Requires ssa_opt_r2c.h
//
// Contents:
//   - W-correlation matrix computation (sequential and fast batched)
//   - Component statistics (singular values, gaps, variance explained)
//   - Periodic pair detection
//
// Usage:
//   #define SSA_OPT_ANALYSIS_IMPLEMENTATION  // in ONE .c file
//   #include "ssa_opt_analysis.h"
//
// Precision:
//   #define SSA_USE_FLOAT before including for single precision
// ============================================================================

#ifndef SSA_OPT_ANALYSIS_H
#define SSA_OPT_ANALYSIS_H

#include "ssa_opt_r2c.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Additional precision-switching macros for analysis functions
// ============================================================================

#ifdef SSA_USE_FLOAT
    #define ssa_cblas_syrk          cblas_ssyrk
#else
    #define ssa_cblas_syrk          cblas_dsyrk
#endif

// ============================================================================
// API Declarations
// ============================================================================

// W-Correlation Matrix
// Computes W[i,j] = ⟨Xᵢ,Xⱼ⟩_w / (||Xᵢ||_w · ||Xⱼ||_w) for all component pairs
// W must be pre-allocated as n_components × n_components
int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, ssa_real *W);

// Fast W-correlation using pre-cached FFTs and batched operations
// Requires ssa_opt_cache_ffts() called first, auto-dispatched from wcorr_matrix
int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, ssa_real *W);

// Single-pair W-correlation (for selective queries without full matrix)
ssa_real ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j);

// Component Statistics
// Computes singular value analysis: gaps, cumulative variance, second differences
int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats);
void ssa_opt_component_stats_free(SSA_ComponentStats *stats);

// Periodic Pair Detection
// Finds sine/cosine pairs based on singular value similarity and W-correlation
// Returns number of pairs found, pairs array contains [i0,j0, i1,j1, ...]
int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                ssa_real sv_tol, ssa_real wcorr_thresh);

// ============================================================================
// Implementation
// ============================================================================

#ifdef SSA_OPT_ANALYSIS_IMPLEMENTATION

// Forward declaration for auto-dispatch
int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, ssa_real *W);

int ssa_opt_wcorr_matrix(const SSA_Opt *ssa, ssa_real *W)
{
    if (!ssa || !ssa->decomposed || !W)
        return -1;

    // AUTO-DISPATCH: Use fast path if workspace is ready
    if (ssa->fft_cached && ssa->wcorr_ws_complex)
        return ssa_opt_wcorr_matrix_fast(ssa, W);

    // Original sequential implementation
    int N = ssa->N, L = ssa->L, K = ssa->K, n = ssa->n_components;
    int fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
    SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

    int use_cache = (ssa->U_fft != NULL && ssa->V_fft != NULL);

    ssa_real *sqrt_inv_c = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    if (!sqrt_inv_c)
        return -1;

    for (int t = 0; t < N; t++)
        sqrt_inv_c[t] = (ssa_real)sqrt((double)ssa->inv_diag_count[t]);

    ssa_real *G = (ssa_real *)ssa_opt_alloc(n * N * sizeof(ssa_real));
    ssa_real *norms = (ssa_real *)ssa_opt_alloc(n * sizeof(ssa_real));
    ssa_real *h_temp = (ssa_real *)ssa_opt_alloc(fft_len * sizeof(ssa_real));
    ssa_real *u_fft_local = NULL, *v_fft_local = NULL;

    if (!use_cache)
    {
        u_fft_local = (ssa_real *)ssa_opt_alloc(2 * r2c_len * sizeof(ssa_real));
        v_fft_local = (ssa_real *)ssa_opt_alloc(2 * r2c_len * sizeof(ssa_real));
    }

    if (!G || !norms || !h_temp || (!use_cache && (!u_fft_local || !v_fft_local)))
    {
        ssa_opt_free_ptr(sqrt_inv_c);
        ssa_opt_free_ptr(G);
        ssa_opt_free_ptr(norms);
        ssa_opt_free_ptr(h_temp);
        ssa_opt_free_ptr(u_fft_local);
        ssa_opt_free_ptr(v_fft_local);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        ssa_real sigma = ssa->sigma[i];
        const ssa_real *u_fft_ptr, *v_fft_ptr;

        if (use_cache)
        {
            u_fft_ptr = &ssa->U_fft[i * 2 * r2c_len];
            v_fft_ptr = &ssa->V_fft[i * 2 * r2c_len];
        }
        else
        {
            const ssa_real *u_vec = &ssa->U[i * L];
            const ssa_real *v_vec = &ssa->V[i * K];

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);

            u_fft_ptr = u_fft_local;
            v_fft_ptr = v_fft_local;
        }

        ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
        DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, h_temp);

        ssa_real norm_sq = ssa_opt_weighted_norm_sq(h_temp, ssa->inv_diag_count, N);
        norm_sq *= sigma * sigma;
        norms[i] = (ssa_real)sqrt((double)norm_sq);

        ssa_real scale = (norms[i] > (ssa_real)1e-12) ? sigma / norms[i] : (ssa_real)0.0;
        ssa_real *g_row = &G[i * N];
        ssa_opt_scale_weighted(h_temp, sqrt_inv_c, scale, g_row, N);
    }

    // W = G·Gᵀ via DSYRK (upper triangle only, then mirror)
    ssa_cblas_syrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, N,
                   (ssa_real)1.0, G, N, (ssa_real)0.0, W, n);

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            W[j * n + i] = W[i * n + j];

    ssa_opt_free_ptr(sqrt_inv_c);
    ssa_opt_free_ptr(G);
    ssa_opt_free_ptr(norms);
    ssa_opt_free_ptr(h_temp);
    ssa_opt_free_ptr(u_fft_local);
    ssa_opt_free_ptr(v_fft_local);

    return 0;
}

// ========================================================================
// W-Correlation Matrix (Fast Path)
// Computes W[i,j] = ⟨Xᵢ,Xⱼ⟩_w / (||Xᵢ||_w · ||Xⱼ||_w) for all pairs
// Key optimization: Use DSYRK instead of O(n²) pairwise correlations
// ========================================================================
int ssa_opt_wcorr_matrix_fast(const SSA_Opt *ssa, ssa_real *W)
{
    int N, n, fft_len, r2c_len, i, t, k, ii, jj;

    if (!ssa || !ssa->decomposed || !W)
        return -1;
    if (!ssa->fft_cached || !ssa->wcorr_ws_complex)
        return ssa_opt_wcorr_matrix(ssa, W);

    N = ssa->N;
    n = ssa->n_components;
    fft_len = ssa->fft_len;
    r2c_len = ssa->r2c_len;

    ssa_real *all_ws_complex = ssa->wcorr_ws_complex;
    ssa_real *all_h = ssa->wcorr_h;
    ssa_real *G = ssa->wcorr_G;
    ssa_real *sqrt_inv_c = ssa->wcorr_sqrt_inv_c;

    // Step 1: Compute FFT(uᵢ) ⊙ FFT(vᵢ) for all components (parallel)
    #pragma omp parallel for private(k)
    for (i = 0; i < n; i++)
    {
        const ssa_real *u_fft = &ssa->U_fft[i * 2 * r2c_len];
        const ssa_real *v_fft = &ssa->V_fft[i * 2 * r2c_len];
        ssa_real *ws = &all_ws_complex[i * 2 * r2c_len];

        for (k = 0; k < r2c_len; k++)
        {
            ssa_real ar = u_fft[2 * k], ai = u_fft[2 * k + 1];
            ssa_real br = v_fft[2 * k], bi = v_fft[2 * k + 1];
            ws[2 * k] = ar * br - ai * bi;
            ws[2 * k + 1] = ar * bi + ai * br;
        }
    }

    // Step 2: Batched IFFT for all n components (single MKL call)
    {
        MKL_LONG status = DftiComputeBackward(ssa->fft_c2r_wcorr, all_ws_complex, all_h);
        if (status != 0)
            return -1;
    }

    // Step 3: Normalize and weight: G[i,t] = (σᵢ/||hᵢ||_w) · hᵢ[t] · √w[t]
    #pragma omp parallel for private(t)
    for (i = 0; i < n; i++)
    {
        ssa_real *h = &all_h[i * fft_len];
        ssa_real sigma = ssa->sigma[i];
        ssa_real norm_sq = ssa_opt_weighted_norm_sq(h, ssa->inv_diag_count, N);
        norm_sq *= sigma * sigma;
        ssa_real norm = (ssa_real)sqrt((double)norm_sq);
        ssa_real scale = (norm > (ssa_real)1e-12) ? sigma / norm : (ssa_real)0.0;
        ssa_real *g_row = &G[i * N];
        ssa_opt_scale_weighted(h, sqrt_inv_c, scale, g_row, N);
    }

    // Step 4: W = G·Gᵀ via DSYRK (upper triangle only, then mirror)
    ssa_cblas_syrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, N,
                   (ssa_real)1.0, G, N, (ssa_real)0.0, W, n);

    for (ii = 0; ii < n; ii++)
        for (jj = ii + 1; jj < n; jj++)
            W[jj * n + ii] = W[ii * n + jj];

    return 0;
}

// Single-pair W-correlation (for selective queries)
ssa_real ssa_opt_wcorr_pair(const SSA_Opt *ssa, int i, int j)
{
    if (!ssa || !ssa->decomposed || i < 0 || i >= ssa->n_components ||
        j < 0 || j >= ssa->n_components)
        return (ssa_real)0.0;

    int N = ssa->N, L = ssa->L, K = ssa->K;
    int fft_len = ssa->fft_len, r2c_len = ssa->r2c_len;
    SSA_Opt *ssa_mut = (SSA_Opt *)ssa;

    int use_cache = (ssa->U_fft != NULL && ssa->V_fft != NULL);

    ssa_real *h_i = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    ssa_real *h_j = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    ssa_real *u_fft_local = NULL, *v_fft_local = NULL;

    if (!use_cache)
    {
        u_fft_local = (ssa_real *)ssa_opt_alloc(2 * r2c_len * sizeof(ssa_real));
        v_fft_local = (ssa_real *)ssa_opt_alloc(2 * r2c_len * sizeof(ssa_real));
    }

    if (!h_i || !h_j || (!use_cache && (!u_fft_local || !v_fft_local)))
    {
        ssa_opt_free_ptr(h_i);
        ssa_opt_free_ptr(h_j);
        ssa_opt_free_ptr(u_fft_local);
        ssa_opt_free_ptr(v_fft_local);
        return (ssa_real)0.0;
    }

    // Compute h_i
    {
        const ssa_real *u_fft_ptr, *v_fft_ptr;
        if (use_cache)
        {
            u_fft_ptr = &ssa->U_fft[i * 2 * r2c_len];
            v_fft_ptr = &ssa->V_fft[i * 2 * r2c_len];
        }
        else
        {
            const ssa_real *u_vec = &ssa->U[i * L];
            const ssa_real *v_vec = &ssa->V[i * K];

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);

            u_fft_ptr = u_fft_local;
            v_fft_ptr = v_fft_local;
        }

        ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
        DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
        memcpy(h_i, ssa_mut->ws_real, N * sizeof(ssa_real));
    }

    // Compute h_j
    {
        const ssa_real *u_fft_ptr, *v_fft_ptr;
        if (use_cache)
        {
            u_fft_ptr = &ssa->U_fft[j * 2 * r2c_len];
            v_fft_ptr = &ssa->V_fft[j * 2 * r2c_len];
        }
        else
        {
            const ssa_real *u_vec = &ssa->U[j * L];
            const ssa_real *v_vec = &ssa->V[j * K];

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, u_vec, L * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, u_fft_local);

            ssa_opt_zero(ssa_mut->ws_real, fft_len);
            memcpy(ssa_mut->ws_real, v_vec, K * sizeof(ssa_real));
            DftiComputeForward(ssa_mut->fft_r2c, ssa_mut->ws_real, v_fft_local);

            u_fft_ptr = u_fft_local;
            v_fft_ptr = v_fft_local;
        }

        ssa_opt_complex_mul_r2c(u_fft_ptr, v_fft_ptr, ssa_mut->ws_complex, r2c_len);
        DftiComputeBackward(ssa_mut->fft_c2r, ssa_mut->ws_complex, ssa_mut->ws_real);
        memcpy(h_j, ssa_mut->ws_real, N * sizeof(ssa_real));
    }

    ssa_real sigma_i = ssa->sigma[i], sigma_j = ssa->sigma[j];
    ssa_real inner, norm_i_sq, norm_j_sq;
    ssa_opt_weighted_inner3(h_i, h_j, ssa->inv_diag_count, N, &inner, &norm_i_sq, &norm_j_sq);

    inner *= sigma_i * sigma_j;
    norm_i_sq *= sigma_i * sigma_i;
    norm_j_sq *= sigma_j * sigma_j;

    ssa_opt_free_ptr(h_i);
    ssa_opt_free_ptr(h_j);
    ssa_opt_free_ptr(u_fft_local);
    ssa_opt_free_ptr(v_fft_local);

    ssa_real denom = (ssa_real)sqrt((double)norm_i_sq) * (ssa_real)sqrt((double)norm_j_sq);
    return (denom > (ssa_real)1e-12) ? inner / denom : (ssa_real)0.0;
}

int ssa_opt_component_stats(const SSA_Opt *ssa, SSA_ComponentStats *stats)
{
    if (!ssa || !ssa->decomposed || !stats)
        return -1;

    int n = ssa->n_components;
    if (n < 2)
        return -1;

    memset(stats, 0, sizeof(SSA_ComponentStats));
    stats->n = n;

    stats->singular_values = (ssa_real *)ssa_opt_alloc(n * sizeof(ssa_real));
    stats->log_sv = (ssa_real *)ssa_opt_alloc(n * sizeof(ssa_real));
    stats->gaps = (ssa_real *)ssa_opt_alloc((n - 1) * sizeof(ssa_real));
    stats->cumulative_var = (ssa_real *)ssa_opt_alloc(n * sizeof(ssa_real));
    stats->second_diff = (ssa_real *)ssa_opt_alloc(n * sizeof(ssa_real));

    if (!stats->singular_values || !stats->log_sv || !stats->gaps ||
        !stats->cumulative_var || !stats->second_diff)
    {
        ssa_opt_component_stats_free(stats);
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        stats->singular_values[i] = ssa->sigma[i];
        stats->log_sv[i] = (ssa_real)log((double)ssa->sigma[i] + 1e-300);
    }

    ssa_real max_gap = (ssa_real)0.0;
    int max_gap_idx = 0;
    for (int i = 0; i < n - 1; i++)
    {
        ssa_real gap = ssa->sigma[i] / (ssa->sigma[i + 1] + (ssa_real)1e-300);
        stats->gaps[i] = gap;
        if (gap > max_gap)
        {
            max_gap = gap;
            max_gap_idx = i;
        }
    }

    ssa_real cumsum = (ssa_real)0.0;
    for (int i = 0; i < n; i++)
    {
        cumsum += ssa->eigenvalues[i];
        stats->cumulative_var[i] = cumsum / ssa->total_variance;
    }

    stats->second_diff[0] = (ssa_real)0.0;
    stats->second_diff[n - 1] = (ssa_real)0.0;
    for (int i = 1; i < n - 1; i++)
        stats->second_diff[i] = stats->log_sv[i - 1] - (ssa_real)2.0 * stats->log_sv[i] + stats->log_sv[i + 1];

    stats->suggested_signal = max_gap_idx + 1;
    stats->gap_threshold = max_gap;

    return 0;
}

void ssa_opt_component_stats_free(SSA_ComponentStats *stats)
{
    if (!stats)
        return;
    ssa_opt_free_ptr(stats->singular_values);
    ssa_opt_free_ptr(stats->log_sv);
    ssa_opt_free_ptr(stats->gaps);
    ssa_opt_free_ptr(stats->cumulative_var);
    ssa_opt_free_ptr(stats->second_diff);
    memset(stats, 0, sizeof(SSA_ComponentStats));
}

int ssa_opt_find_periodic_pairs(const SSA_Opt *ssa, int *pairs, int max_pairs,
                                ssa_real sv_tol, ssa_real wcorr_thresh)
{
    if (!ssa || !ssa->decomposed || !pairs || max_pairs < 1)
        return 0;

    // Default tolerances if not specified
    if (sv_tol <= (ssa_real)0)
        sv_tol = (ssa_real)0.1;  // 10% singular value difference allowed
    if (wcorr_thresh <= (ssa_real)0)
        wcorr_thresh = (ssa_real)0.5;  // 50% W-correlation minimum

    int n = ssa->n_components, n_pairs = 0;

    // Track which components have been paired
    bool *used = (bool *)calloc(n, sizeof(bool));
    if (!used)
        return 0;

    // Greedy pairing: scan components in order of decreasing singular value
    for (int i = 0; i < n - 1 && n_pairs < max_pairs; i++)
    {
        if (used[i])
            continue;

        for (int j = i + 1; j < n && n_pairs < max_pairs; j++)
        {
            if (used[j])
                continue;

            // Test 1: Singular Value Similarity
            ssa_real sv_ratio = ssa->sigma[j] / (ssa->sigma[i] + (ssa_real)1e-300);
            if ((ssa_real)fabs((double)((ssa_real)1.0 - sv_ratio)) > sv_tol)
                continue;

            // Test 2: W-Correlation
            ssa_real wcorr = (ssa_real)fabs((double)ssa_opt_wcorr_pair(ssa, i, j));
            if (wcorr < wcorr_thresh)
                continue;

            // Found a pair
            pairs[2 * n_pairs] = i;
            pairs[2 * n_pairs + 1] = j;
            used[i] = true;
            used[j] = true;
            n_pairs++;
            break;
        }
    }

    free(used);
    return n_pairs;
}

#endif // SSA_OPT_ANALYSIS_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_ANALYSIS_H
