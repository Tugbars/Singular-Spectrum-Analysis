// ============================================================================
// SSA_OPT_ADVANCED.H - Advanced SSA Methods
// ============================================================================
// Part of the SSA optimization library. Requires ssa_opt_r2c.h
//
// Contents:
//   - Cadzow iterations (finite-rank signal approximation)
//   - Gap filling (missing value imputation)
//   - MSSA (Multivariate SSA)
//   - ESPRIT (parameter estimation via rotational invariance)
//
// Usage:
//   #define SSA_OPT_ADVANCED_IMPLEMENTATION  // in ONE .c file
//   #include "ssa_opt_advanced.h"
//
// Precision:
//   #define SSA_USE_FLOAT before including for single precision
// ============================================================================

#ifndef SSA_OPT_ADVANCED_H
#define SSA_OPT_ADVANCED_H

#include "ssa_opt_r2c.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Additional precision-switching macros for advanced functions
// ============================================================================

#ifdef SSA_USE_FLOAT
    #define ssa_cblas_syrk          cblas_ssyrk
    #define ssa_vRngGaussian        vsRngGaussian
    #define ssa_lapack_geqrf        LAPACKE_sgeqrf
    #define ssa_lapack_orgqr        LAPACKE_sorgqr
    #define ssa_lapack_gesdd        LAPACKE_sgesdd
    #define ssa_lapack_gesdd_work   LAPACKE_sgesdd_work
    #define ssa_lapack_dgels        LAPACKE_sgels
    #define ssa_lapack_dgels_work   LAPACKE_sgels_work
    #define ssa_lapack_geev         LAPACKE_sgeev
    #define SSA_DFTI_PRECISION      DFTI_SINGLE
#else
    #define ssa_cblas_syrk          cblas_dsyrk
    #define ssa_vRngGaussian        vdRngGaussian
    #define ssa_lapack_geqrf        LAPACKE_dgeqrf
    #define ssa_lapack_orgqr        LAPACKE_dorgqr
    #define ssa_lapack_gesdd        LAPACKE_dgesdd
    #define ssa_lapack_gesdd_work   LAPACKE_dgesdd_work
    #define ssa_lapack_dgels        LAPACKE_dgels
    #define ssa_lapack_dgels_work   LAPACKE_dgels_work
    #define ssa_lapack_geev         LAPACKE_dgeev
    #define SSA_DFTI_PRECISION      DFTI_DOUBLE
#endif

// ============================================================================
// API Declarations
// ============================================================================

// Cadzow Iterations
int ssa_opt_cadzow(const ssa_real *x, int N, int L, int rank, int max_iter,
                   ssa_real tol, ssa_real *output, SSA_CadzowResult *result);
int ssa_opt_cadzow_weighted(const ssa_real *x, int N, int L, int rank, int max_iter,
                            ssa_real tol, ssa_real alpha, ssa_real *output,
                            SSA_CadzowResult *result);
int ssa_opt_cadzow_inplace(SSA_Opt *ssa, int rank, int max_iter, ssa_real tol,
                           SSA_CadzowResult *result);

// Gap Filling
int ssa_opt_gapfill(ssa_real *x, int N, int L, int rank, int max_iter,
                    ssa_real tol, SSA_GapFillResult *result);
int ssa_opt_gapfill_simple(ssa_real *x, int N, int L, int rank,
                           SSA_GapFillResult *result);

// MSSA (Multivariate SSA)
int mssa_opt_init(MSSA_Opt *mssa, const ssa_real *X, int M, int N, int L);
void mssa_opt_free(MSSA_Opt *mssa);
int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling);
int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx, const int *group,
                         int n_group, ssa_real *output);
int mssa_opt_reconstruct_all(const MSSA_Opt *mssa, const int *group,
                             int n_group, ssa_real *output);
int mssa_opt_series_contributions(const MSSA_Opt *mssa, ssa_real *contributions);
ssa_real mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end);

// ESPRIT (Parameter Estimation)
int ssa_opt_parestimate(const SSA_Opt *ssa, const int *group, int n_group,
                        SSA_ParEstimate *result);
void ssa_opt_parestimate_free(SSA_ParEstimate *result);

// ============================================================================
// Implementation
// ============================================================================

#ifdef SSA_OPT_ADVANCED_IMPLEMENTATION

// ========================================================================
// Helper: check if value is NaN
// ========================================================================
static inline int ssa_is_nan(ssa_real x)
{
    return isnan(x);
}

// ========================================================================
// Helper: linear interpolation for initial gap filling
// ========================================================================
static void ssa_linear_interp_gaps(ssa_real *x, const int *gap_mask, int N)
{
    int i = 0;
    while (i < N)
    {
        if (gap_mask[i])
        {
            int gap_start = i;
            while (i < N && gap_mask[i])
                i++;
            int gap_end = i;

            ssa_real left_val = (ssa_real)0.0, right_val = (ssa_real)0.0;
            int left_idx = gap_start - 1;
            int right_idx = gap_end;

            if (left_idx >= 0)
                left_val = x[left_idx];
            else
                left_val = (right_idx < N) ? x[right_idx] : (ssa_real)0.0;

            if (right_idx < N)
                right_val = x[right_idx];
            else
                right_val = left_val;

            int gap_len = gap_end - gap_start;
            for (int j = 0; j < gap_len; j++)
            {
                ssa_real t = (gap_len > 1) ? (ssa_real)(j + 1) / (ssa_real)(gap_len + 1) : (ssa_real)0.5;
                x[gap_start + j] = left_val + t * (right_val - left_val);
            }
        }
        else
        {
            i++;
        }
    }
}

// ========================================================================
// Cadzow Iterations (Finite-Rank Signal Approximation)
// ========================================================================

int ssa_opt_cadzow(const ssa_real *x, int N, int L, int rank, int max_iter,
                   ssa_real tol, ssa_real *output, SSA_CadzowResult *result)
{
    if (!x || !output || N < 4 || L < 2 || L > N - 1 || rank < 1 || max_iter < 1)
        return -1;

    ssa_real *y = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    ssa_real *y_new = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    int *group = (int *)malloc(rank * sizeof(int));

    if (!y || !y_new || !group)
    {
        ssa_opt_free_ptr(y);
        ssa_opt_free_ptr(y_new);
        free(group);
        return -1;
    }

    for (int i = 0; i < rank; i++)
        group[i] = i;

    memcpy(y, x, N * sizeof(ssa_real));

    ssa_real y_norm = ssa_cblas_nrm2(N, y, 1);
    if (y_norm < (ssa_real)1e-15)
        y_norm = (ssa_real)1.0;

    int iter;
    ssa_real diff = (ssa_real)0.0;
    bool converged = false;

    for (iter = 0; iter < max_iter; iter++)
    {
        SSA_Opt ssa = {0};
        if (ssa_opt_init(&ssa, y, N, L) != 0)
        {
            ssa_opt_free_ptr(y);
            ssa_opt_free_ptr(y_new);
            free(group);
            return -1;
        }

        int K = N - L + 1;
        int min_dim = (L < K) ? L : K;
        int use_randomized = (rank + 8 < min_dim / 2);

        if (use_randomized)
        {
            if (ssa_opt_prepare(&ssa, rank, 8) != 0 ||
                ssa_opt_decompose_randomized(&ssa, rank, 8) != 0)
            {
                use_randomized = 0;
            }
        }

        if (!use_randomized)
        {
            int block_sz = (rank < 16) ? rank : 16;
            if (ssa_opt_decompose_block(&ssa, rank, block_sz, 3) != 0)
            {
                ssa_opt_free(&ssa);
                ssa_opt_free_ptr(y);
                ssa_opt_free_ptr(y_new);
                free(group);
                return -1;
            }
        }

        if (ssa_opt_reconstruct(&ssa, group, rank, y_new) != 0)
        {
            ssa_opt_free(&ssa);
            ssa_opt_free_ptr(y);
            ssa_opt_free_ptr(y_new);
            free(group);
            return -1;
        }

        ssa_opt_free(&ssa);

        ssa_real diff_sq = (ssa_real)0.0;
        for (int i = 0; i < N; i++)
        {
            ssa_real d = y_new[i] - y[i];
            diff_sq += d * d;
        }
        diff = (ssa_real)sqrt((double)diff_sq) / y_norm;

        ssa_real *tmp = y;
        y = y_new;
        y_new = tmp;

        if (diff < tol)
        {
            converged = true;
            iter++;
            break;
        }
    }

    memcpy(output, y, N * sizeof(ssa_real));

    if (result)
    {
        result->iterations = iter;
        result->final_diff = diff;
        result->converged = converged ? (ssa_real)1.0 : (ssa_real)0.0;
    }

    ssa_opt_free_ptr(y);
    ssa_opt_free_ptr(y_new);
    free(group);

    return iter;
}

int ssa_opt_cadzow_weighted(const ssa_real *x, int N, int L, int rank, int max_iter,
                            ssa_real tol, ssa_real alpha, ssa_real *output,
                            SSA_CadzowResult *result)
{
    if (alpha <= (ssa_real)0.0 || alpha > (ssa_real)1.0)
        return -1;

    int ret = ssa_opt_cadzow(x, N, L, rank, max_iter, tol, output, result);
    if (ret < 0)
        return ret;

    if (alpha < (ssa_real)1.0)
    {
        ssa_real beta = (ssa_real)1.0 - alpha;
        for (int i = 0; i < N; i++)
            output[i] = alpha * output[i] + beta * x[i];
    }

    return ret;
}

int ssa_opt_cadzow_inplace(SSA_Opt *ssa, int rank, int max_iter, ssa_real tol,
                           SSA_CadzowResult *result)
{
    if (!ssa || !ssa->initialized || rank < 1 || max_iter < 1)
        return -1;

    int N = ssa->N, L = ssa->L;

    ssa_real *y_new = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    ssa_real *y_current = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    int *group = (int *)malloc(rank * sizeof(int));

    if (!y_new || !y_current || !group)
    {
        ssa_opt_free_ptr(y_new);
        ssa_opt_free_ptr(y_current);
        free(group);
        return -1;
    }

    for (int i = 0; i < rank; i++)
        group[i] = i;

    ssa_real y_norm = (ssa_real)1.0;
    int iter;
    ssa_real diff = (ssa_real)0.0;
    bool converged = false;

    for (iter = 0; iter < max_iter; iter++)
    {
        int K = N - L + 1;
        int min_dim = (L < K) ? L : K;
        int use_randomized = (rank + 8 < min_dim / 2);

        ssa->decomposed = false;
        ssa->n_components = 0;
        ssa->total_variance = (ssa_real)0.0;
        ssa_opt_free_cached_ffts(ssa);

        if (use_randomized && ssa->prepared && rank + 8 <= ssa->prepared_kp)
        {
            if (ssa_opt_decompose_randomized(ssa, rank, 8) != 0)
                use_randomized = 0;
        }

        if (!use_randomized || !ssa->decomposed)
        {
            int block_sz = (rank < 16) ? rank : 16;
            if (ssa_opt_decompose_block(ssa, rank, block_sz, 3) != 0)
            {
                ssa_opt_free_ptr(y_new);
                ssa_opt_free_ptr(y_current);
                free(group);
                return -1;
            }
        }

        if (iter > 0)
            memcpy(y_current, y_new, N * sizeof(ssa_real));

        if (ssa_opt_reconstruct(ssa, group, rank, y_new) != 0)
        {
            ssa_opt_free_ptr(y_new);
            ssa_opt_free_ptr(y_current);
            free(group);
            return -1;
        }

        if (iter == 0)
        {
            y_norm = ssa_cblas_nrm2(N, y_new, 1);
            if (y_norm < (ssa_real)1e-15)
                y_norm = (ssa_real)1.0;
        }
        else
        {
            ssa_real diff_sq = (ssa_real)0.0;
            for (int i = 0; i < N; i++)
            {
                ssa_real d = y_new[i] - y_current[i];
                diff_sq += d * d;
            }
            diff = (ssa_real)sqrt((double)diff_sq) / y_norm;

            if (diff < tol)
            {
                converged = true;
                iter++;
                break;
            }
        }

        if (iter < max_iter - 1)
            ssa_opt_update_signal(ssa, y_new);
    }

    if (result)
    {
        result->iterations = iter;
        result->final_diff = diff;
        result->converged = converged ? (ssa_real)1.0 : (ssa_real)0.0;
    }

    ssa_opt_free_ptr(y_new);
    ssa_opt_free_ptr(y_current);
    free(group);

    return iter;
}

// ========================================================================
// Gap Filling - Handle Missing Values in Time Series
// ========================================================================

int ssa_opt_gapfill(ssa_real *x, int N, int L, int rank, int max_iter,
                    ssa_real tol, SSA_GapFillResult *result)
{
    if (!x || N < 4 || L < 2 || L > N - 1 || rank < 1)
        return -1;

    int *gap_mask = (int *)calloc(N, sizeof(int));
    if (!gap_mask)
        return -1;

    int n_gaps = 0;
    for (int i = 0; i < N; i++)
    {
        if (ssa_is_nan(x[i]))
        {
            gap_mask[i] = 1;
            n_gaps++;
        }
    }

    if (n_gaps == 0)
    {
        free(gap_mask);
        if (result)
        {
            result->iterations = 0;
            result->final_diff = (ssa_real)0.0;
            result->converged = 1;
            result->n_gaps = 0;
        }
        return 0;
    }

    if (N - n_gaps < L + rank)
    {
        free(gap_mask);
        return -1;
    }

    ssa_linear_interp_gaps(x, gap_mask, N);

    ssa_real *gap_values_old = (ssa_real *)ssa_opt_alloc(n_gaps * sizeof(ssa_real));
    ssa_real *gap_values_new = (ssa_real *)ssa_opt_alloc(n_gaps * sizeof(ssa_real));

    if (!gap_values_old || !gap_values_new)
    {
        free(gap_mask);
        if (gap_values_old)
            ssa_opt_free_ptr(gap_values_old);
        if (gap_values_new)
            ssa_opt_free_ptr(gap_values_new);
        return -1;
    }

    int gi = 0;
    for (int i = 0; i < N; i++)
    {
        if (gap_mask[i])
            gap_values_old[gi++] = x[i];
    }

    ssa_real *recon = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    if (!recon)
    {
        free(gap_mask);
        ssa_opt_free_ptr(gap_values_old);
        ssa_opt_free_ptr(gap_values_new);
        return -1;
    }

    int *group = (int *)malloc(rank * sizeof(int));
    if (!group)
    {
        free(gap_mask);
        ssa_opt_free_ptr(gap_values_old);
        ssa_opt_free_ptr(gap_values_new);
        ssa_opt_free_ptr(recon);
        return -1;
    }
    for (int i = 0; i < rank; i++)
        group[i] = i;

    int iter;
    ssa_real diff = (ssa_real)1e10;
    int converged = 0;

    for (iter = 0; iter < max_iter; iter++)
    {
        SSA_Opt ssa;
        if (ssa_opt_init(&ssa, x, N, L) != 0)
        {
            free(gap_mask);
            free(group);
            ssa_opt_free_ptr(gap_values_old);
            ssa_opt_free_ptr(gap_values_new);
            ssa_opt_free_ptr(recon);
            return -1;
        }

        ssa_opt_prepare(&ssa, rank, 8);
        if (ssa_opt_decompose_randomized(&ssa, rank, 8) != 0)
        {
            ssa_opt_free(&ssa);
            free(gap_mask);
            free(group);
            ssa_opt_free_ptr(gap_values_old);
            ssa_opt_free_ptr(gap_values_new);
            ssa_opt_free_ptr(recon);
            return -1;
        }

        if (ssa_opt_reconstruct(&ssa, group, rank, recon) != 0)
        {
            ssa_opt_free(&ssa);
            free(gap_mask);
            free(group);
            ssa_opt_free_ptr(gap_values_old);
            ssa_opt_free_ptr(gap_values_new);
            ssa_opt_free_ptr(recon);
            return -1;
        }

        ssa_opt_free(&ssa);

        gi = 0;
        for (int i = 0; i < N; i++)
        {
            if (gap_mask[i])
            {
                gap_values_new[gi++] = recon[i];
                x[i] = recon[i];
            }
        }

        ssa_real sum_sq_diff = (ssa_real)0.0, sum_sq_old = (ssa_real)0.0;
        for (int i = 0; i < n_gaps; i++)
        {
            ssa_real d = gap_values_new[i] - gap_values_old[i];
            sum_sq_diff += d * d;
            sum_sq_old += gap_values_old[i] * gap_values_old[i];
        }

        diff = (sum_sq_old > (ssa_real)1e-12) ?
               (ssa_real)sqrt((double)sum_sq_diff / (double)sum_sq_old) :
               (ssa_real)sqrt((double)sum_sq_diff);

        if (diff < tol)
        {
            converged = 1;
            iter++;
            break;
        }

        memcpy(gap_values_old, gap_values_new, n_gaps * sizeof(ssa_real));
    }

    if (result)
    {
        result->iterations = iter;
        result->final_diff = diff;
        result->converged = converged;
        result->n_gaps = n_gaps;
    }

    free(gap_mask);
    free(group);
    ssa_opt_free_ptr(gap_values_old);
    ssa_opt_free_ptr(gap_values_new);
    ssa_opt_free_ptr(recon);

    return 0;
}

int ssa_opt_gapfill_simple(ssa_real *x, int N, int L, int rank,
                           SSA_GapFillResult *result)
{
    if (!x || N < 4 || L < 2 || L > N - 1 || rank < 1)
        return -1;

    int *gap_mask = (int *)calloc(N, sizeof(int));
    if (!gap_mask)
        return -1;

    int n_gaps = 0;
    for (int i = 0; i < N; i++)
    {
        if (ssa_is_nan(x[i]))
        {
            gap_mask[i] = 1;
            n_gaps++;
        }
    }

    if (n_gaps == 0)
    {
        free(gap_mask);
        if (result)
        {
            result->iterations = 1;
            result->final_diff = (ssa_real)0.0;
            result->converged = 1;
            result->n_gaps = 0;
        }
        return 0;
    }

    int *group = (int *)malloc(rank * sizeof(int));
    if (!group)
    {
        free(gap_mask);
        return -1;
    }
    for (int i = 0; i < rank; i++)
        group[i] = i;

    int i = 0;
    while (i < N)
    {
        if (!gap_mask[i])
        {
            i++;
            continue;
        }

        int gap_start = i;
        while (i < N && gap_mask[i])
            i++;
        int gap_end = i;
        int gap_len = gap_end - gap_start;

        int left_len = gap_start;
        int right_start = gap_end;
        int right_len = N - gap_end;

        ssa_real *forecast_left = NULL;
        ssa_real *forecast_right = NULL;

        if (left_len >= L + rank)
        {
            SSA_Opt ssa_left;
            ssa_real *left_data = (ssa_real *)ssa_opt_alloc(left_len * sizeof(ssa_real));
            if (left_data)
            {
                memcpy(left_data, x, left_len * sizeof(ssa_real));

                if (ssa_opt_init(&ssa_left, left_data, left_len, L) == 0)
                {
                    ssa_opt_prepare(&ssa_left, rank, 8);
                    if (ssa_opt_decompose_randomized(&ssa_left, rank, 8) == 0)
                    {
                        forecast_left = (ssa_real *)ssa_opt_alloc(gap_len * sizeof(ssa_real));
                        if (forecast_left)
                        {
                            // Need forecast function - include forecast header or inline
                            // For now, use simple extrapolation
                            ssa_real *recon = (ssa_real *)ssa_opt_alloc(left_len * sizeof(ssa_real));
                            if (recon && ssa_opt_reconstruct(&ssa_left, group, rank, recon) == 0)
                            {
                                // Linear extrapolation from last two points
                                ssa_real slope = recon[left_len - 1] - recon[left_len - 2];
                                for (int j = 0; j < gap_len; j++)
                                    forecast_left[j] = recon[left_len - 1] + slope * (j + 1);
                            }
                            ssa_opt_free_ptr(recon);
                        }
                    }
                    ssa_opt_free(&ssa_left);
                }
                ssa_opt_free_ptr(left_data);
            }
        }

        if (right_len >= L + rank)
        {
            SSA_Opt ssa_right;
            ssa_real *right_data = (ssa_real *)ssa_opt_alloc(right_len * sizeof(ssa_real));
            if (right_data)
            {
                for (int j = 0; j < right_len; j++)
                    right_data[j] = x[N - 1 - j];

                if (ssa_opt_init(&ssa_right, right_data, right_len, L) == 0)
                {
                    ssa_opt_prepare(&ssa_right, rank, 8);
                    if (ssa_opt_decompose_randomized(&ssa_right, rank, 8) == 0)
                    {
                        ssa_real *backcast_rev = (ssa_real *)ssa_opt_alloc(gap_len * sizeof(ssa_real));
                        if (backcast_rev)
                        {
                            ssa_real *recon = (ssa_real *)ssa_opt_alloc(right_len * sizeof(ssa_real));
                            if (recon && ssa_opt_reconstruct(&ssa_right, group, rank, recon) == 0)
                            {
                                ssa_real slope = recon[right_len - 1] - recon[right_len - 2];
                                for (int j = 0; j < gap_len; j++)
                                    backcast_rev[j] = recon[right_len - 1] + slope * (j + 1);

                                forecast_right = (ssa_real *)ssa_opt_alloc(gap_len * sizeof(ssa_real));
                                if (forecast_right)
                                {
                                    for (int j = 0; j < gap_len; j++)
                                        forecast_right[j] = backcast_rev[gap_len - 1 - j];
                                }
                            }
                            ssa_opt_free_ptr(recon);
                            ssa_opt_free_ptr(backcast_rev);
                        }
                    }
                    ssa_opt_free(&ssa_right);
                }
                ssa_opt_free_ptr(right_data);
            }
        }

        for (int j = 0; j < gap_len; j++)
        {
            ssa_real val = (ssa_real)0.0;
            ssa_real weight = (ssa_real)0.0;

            if (forecast_left)
            {
                ssa_real w = (ssa_real)(gap_len - j) / (ssa_real)gap_len;
                val += w * forecast_left[j];
                weight += w;
            }

            if (forecast_right)
            {
                ssa_real w = (ssa_real)(j + 1) / (ssa_real)gap_len;
                val += w * forecast_right[j];
                weight += w;
            }

            if (weight > (ssa_real)0)
            {
                x[gap_start + j] = val / weight;
            }
            else
            {
                ssa_real left_val = (gap_start > 0) ? x[gap_start - 1] : (ssa_real)0.0;
                ssa_real right_val = (gap_end < N) ? x[gap_end] : left_val;
                x[gap_start + j] = (left_val + right_val) / (ssa_real)2.0;
            }
        }

        if (forecast_left)
            ssa_opt_free_ptr(forecast_left);
        if (forecast_right)
            ssa_opt_free_ptr(forecast_right);
    }

    if (result)
    {
        result->iterations = 1;
        result->final_diff = (ssa_real)0.0;
        result->converged = 1;
        result->n_gaps = n_gaps;
    }

    free(gap_mask);
    free(group);

    return 0;
}

// ========================================================================
// MSSA Implementation
// ========================================================================

void mssa_opt_free(MSSA_Opt *mssa)
{
    if (!mssa)
        return;
    if (mssa->fft_r2c)
        DftiFreeDescriptor(&mssa->fft_r2c);
    if (mssa->fft_c2r)
        DftiFreeDescriptor(&mssa->fft_c2r);
    if (mssa->fft_r2c_batch)
        DftiFreeDescriptor(&mssa->fft_r2c_batch);
    if (mssa->fft_c2r_batch)
        DftiFreeDescriptor(&mssa->fft_c2r_batch);
    if (mssa->rng)
        vslDeleteStream(&mssa->rng);
    ssa_opt_free_ptr(mssa->fft_x);
    ssa_opt_free_ptr(mssa->ws_real);
    ssa_opt_free_ptr(mssa->ws_complex);
    ssa_opt_free_ptr(mssa->ws_batch_real);
    ssa_opt_free_ptr(mssa->ws_batch_complex);
    ssa_opt_free_ptr(mssa->U);
    ssa_opt_free_ptr(mssa->V);
    ssa_opt_free_ptr(mssa->sigma);
    ssa_opt_free_ptr(mssa->eigenvalues);
    ssa_opt_free_ptr(mssa->inv_diag_count);
    memset(mssa, 0, sizeof(MSSA_Opt));
}

int mssa_opt_init(MSSA_Opt *mssa, const ssa_real *X, int M, int N, int L)
{
    if (!mssa || !X || M < 1 || N < 4 || L < 2 || L > N - 1)
        return -1;

    memset(mssa, 0, sizeof(MSSA_Opt));

    mssa->M = M;
    mssa->N = N;
    mssa->L = L;
    mssa->K = N - L + 1;

    int conv_len = N + mssa->K - 1;
    int fft_n = ssa_opt_next_pow2(conv_len);
    mssa->fft_len = fft_n;
    mssa->r2c_len = fft_n / 2 + 1;

    mssa->ws_real = (ssa_real *)ssa_opt_alloc(fft_n * sizeof(ssa_real));
    mssa->ws_complex = (ssa_real *)ssa_opt_alloc(2 * mssa->r2c_len * sizeof(ssa_real));
    mssa->ws_batch_real = (ssa_real *)ssa_opt_alloc(M * fft_n * sizeof(ssa_real));
    mssa->ws_batch_complex = (ssa_real *)ssa_opt_alloc(M * 2 * mssa->r2c_len * sizeof(ssa_real));
    mssa->fft_x = (ssa_real *)ssa_opt_alloc(M * 2 * mssa->r2c_len * sizeof(ssa_real));

    if (!mssa->ws_real || !mssa->ws_complex || !mssa->ws_batch_real ||
        !mssa->ws_batch_complex || !mssa->fft_x)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    MKL_LONG status;

    status = DftiCreateDescriptor(&mssa->fft_r2c, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
    if (status != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }
    DftiSetValue(mssa->fft_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(mssa->fft_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (DftiCommitDescriptor(mssa->fft_r2c) != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    status = DftiCreateDescriptor(&mssa->fft_c2r, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
    if (status != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }
    DftiSetValue(mssa->fft_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(mssa->fft_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mssa->fft_c2r, DFTI_BACKWARD_SCALE, (ssa_real)1.0 / fft_n);
    if (DftiCommitDescriptor(mssa->fft_c2r) != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    status = DftiCreateDescriptor(&mssa->fft_r2c_batch, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
    if (status != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }
    DftiSetValue(mssa->fft_r2c_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(mssa->fft_r2c_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mssa->fft_r2c_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)M);
    DftiSetValue(mssa->fft_r2c_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)fft_n);
    DftiSetValue(mssa->fft_r2c_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)mssa->r2c_len);
    if (DftiCommitDescriptor(mssa->fft_r2c_batch) != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    status = DftiCreateDescriptor(&mssa->fft_c2r_batch, SSA_DFTI_PRECISION, DFTI_REAL, 1, fft_n);
    if (status != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }
    DftiSetValue(mssa->fft_c2r_batch, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(mssa->fft_c2r_batch, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mssa->fft_c2r_batch, DFTI_BACKWARD_SCALE, (ssa_real)1.0 / fft_n);
    DftiSetValue(mssa->fft_c2r_batch, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)M);
    DftiSetValue(mssa->fft_c2r_batch, DFTI_INPUT_DISTANCE, (MKL_LONG)mssa->r2c_len);
    DftiSetValue(mssa->fft_c2r_batch, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fft_n);
    if (DftiCommitDescriptor(mssa->fft_c2r_batch) != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    for (int m = 0; m < M; m++)
    {
        ssa_real *fft_xm = mssa->fft_x + m * 2 * mssa->r2c_len;
        ssa_opt_zero(mssa->ws_real, fft_n);
        memcpy(mssa->ws_real, X + m * N, N * sizeof(ssa_real));
        DftiComputeForward(mssa->fft_r2c, mssa->ws_real, fft_xm);
    }

    if (vslNewStream(&mssa->rng, VSL_BRNG_MT19937, 42) != 0)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    mssa->inv_diag_count = (ssa_real *)ssa_opt_alloc(N * sizeof(ssa_real));
    if (!mssa->inv_diag_count)
    {
        mssa_opt_free(mssa);
        return -1;
    }

    for (int t = 0; t < N; t++)
    {
        int L_t = (t + 1 < L) ? t + 1 : L;
        int K_t = (mssa->K < N - t) ? mssa->K : N - t;
        int count = (L_t < K_t) ? L_t : K_t;
        mssa->inv_diag_count[t] = (count > 0) ? (ssa_real)1.0 / count : (ssa_real)0.0;
    }

    mssa->initialized = true;
    return 0;
}

static void mssa_opt_hankel_matvec(MSSA_Opt *mssa, const ssa_real *v, ssa_real *y)
{
    int M = mssa->M, K = mssa->K, L = mssa->L;
    int fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;

    ssa_opt_zero(mssa->ws_real, fft_len);
    ssa_opt_reverse_copy(v, mssa->ws_real, K);
    DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);

    for (int m = 0; m < M; m++)
    {
        ssa_real *fft_xm = mssa->fft_x + m * 2 * r2c_len;
        ssa_real *ym = y + m * L;
        ssa_real *ws_out = mssa->ws_batch_complex;

        ssa_vMul(r2c_len, (const SSA_MKL_COMPLEX *)fft_xm,
                 (const SSA_MKL_COMPLEX *)mssa->ws_complex,
                 (SSA_MKL_COMPLEX *)ws_out);

        DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);
        memcpy(ym, mssa->ws_batch_real + (K - 1), L * sizeof(ssa_real));
    }
}

static void mssa_opt_hankel_matvec_T(MSSA_Opt *mssa, const ssa_real *u, ssa_real *y)
{
    int M = mssa->M, K = mssa->K, L = mssa->L;
    int fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;

    ssa_opt_zero(y, K);

    for (int m = 0; m < M; m++)
    {
        const ssa_real *um = u + m * L;
        ssa_real *fft_xm = mssa->fft_x + m * 2 * r2c_len;

        ssa_opt_zero(mssa->ws_real, fft_len);
        ssa_opt_reverse_copy(um, mssa->ws_real, L);
        DftiComputeForward(mssa->fft_r2c, mssa->ws_real, mssa->ws_complex);

        ssa_real *ws_out = mssa->ws_batch_complex;
        ssa_vMul(r2c_len, (const SSA_MKL_COMPLEX *)fft_xm,
                 (const SSA_MKL_COMPLEX *)mssa->ws_complex,
                 (SSA_MKL_COMPLEX *)ws_out);

        DftiComputeBackward(mssa->fft_c2r, ws_out, mssa->ws_batch_real);

        for (int j = 0; j < K; j++)
            y[j] += mssa->ws_batch_real[(L - 1) + j];
    }
}

static void mssa_opt_hankel_matvec_batch(MSSA_Opt *mssa, const ssa_real *V_block,
                                         ssa_real *Y_block, int b)
{
    int ML = mssa->M * mssa->L;
    for (int j = 0; j < b; j++)
        mssa_opt_hankel_matvec(mssa, V_block + j * mssa->K, Y_block + j * ML);
}

static void mssa_opt_hankel_matvec_T_batch(MSSA_Opt *mssa, const ssa_real *U_block,
                                           ssa_real *Y_block, int b)
{
    int ML = mssa->M * mssa->L, K = mssa->K;
    for (int j = 0; j < b; j++)
        mssa_opt_hankel_matvec_T(mssa, U_block + j * ML, Y_block + j * K);
}

int mssa_opt_decompose(MSSA_Opt *mssa, int k, int oversampling)
{
    if (!mssa || !mssa->initialized || k < 1)
        return -1;

    int M = mssa->M, L = mssa->L, K = mssa->K, ML = M * L;
    int p = (oversampling <= 0) ? 8 : oversampling;
    int kp = k + p;

    int min_dim = (ML < K) ? ML : K;
    kp = (kp < min_dim) ? kp : min_dim;
    k = (k < kp) ? k : kp;

    mssa->U = (ssa_real *)ssa_opt_alloc(ML * k * sizeof(ssa_real));
    mssa->V = (ssa_real *)ssa_opt_alloc(K * k * sizeof(ssa_real));
    mssa->sigma = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    mssa->eigenvalues = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));

    if (!mssa->U || !mssa->V || !mssa->sigma || !mssa->eigenvalues)
        return -1;

    mssa->n_components = k;
    mssa->total_variance = (ssa_real)0.0;

    ssa_real *Omega = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));
    ssa_real *Y = (ssa_real *)ssa_opt_alloc(ML * kp * sizeof(ssa_real));
    ssa_real *Q = (ssa_real *)ssa_opt_alloc(ML * kp * sizeof(ssa_real));
    ssa_real *B = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));
    ssa_real *tau = (ssa_real *)ssa_opt_alloc(kp * sizeof(ssa_real));
    ssa_real *U_svd = (ssa_real *)ssa_opt_alloc(K * kp * sizeof(ssa_real));
    ssa_real *Vt_svd = (ssa_real *)ssa_opt_alloc(kp * kp * sizeof(ssa_real));
    ssa_real *S_svd = (ssa_real *)ssa_opt_alloc(kp * sizeof(ssa_real));
    int *iwork = (int *)ssa_opt_alloc(8 * kp * sizeof(int));

    ssa_real work_query;
    int lwork = -1, info;
    ssa_lapack_gesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd, U_svd, K,
                          Vt_svd, kp, &work_query, lwork, iwork);
    lwork = (int)work_query + 1;
    ssa_real *work = (ssa_real *)ssa_opt_alloc(lwork * sizeof(ssa_real));

    if (!Omega || !Y || !Q || !B || !tau || !U_svd || !Vt_svd || !S_svd || !iwork || !work)
    {
        ssa_opt_free_ptr(Omega);
        ssa_opt_free_ptr(Y);
        ssa_opt_free_ptr(Q);
        ssa_opt_free_ptr(B);
        ssa_opt_free_ptr(tau);
        ssa_opt_free_ptr(U_svd);
        ssa_opt_free_ptr(Vt_svd);
        ssa_opt_free_ptr(S_svd);
        ssa_opt_free_ptr(iwork);
        ssa_opt_free_ptr(work);
        return -1;
    }

    ssa_vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mssa->rng, K * kp, Omega,
                     (ssa_real)0.0, (ssa_real)1.0);

    mssa_opt_hankel_matvec_batch(mssa, Omega, Y, kp);

    ssa_cblas_copy(ML * kp, Y, 1, Q, 1);
    ssa_lapack_geqrf(LAPACK_COL_MAJOR, ML, kp, Q, ML, tau);
    ssa_lapack_orgqr(LAPACK_COL_MAJOR, ML, kp, kp, Q, ML, tau);

    mssa_opt_hankel_matvec_T_batch(mssa, Q, B, kp);

    info = ssa_lapack_gesdd_work(LAPACK_COL_MAJOR, 'S', K, kp, B, K, S_svd,
                                 U_svd, K, Vt_svd, kp, work, lwork, iwork);

    if (info != 0)
    {
        ssa_opt_free_ptr(Omega);
        ssa_opt_free_ptr(Y);
        ssa_opt_free_ptr(Q);
        ssa_opt_free_ptr(B);
        ssa_opt_free_ptr(tau);
        ssa_opt_free_ptr(U_svd);
        ssa_opt_free_ptr(Vt_svd);
        ssa_opt_free_ptr(S_svd);
        ssa_opt_free_ptr(iwork);
        ssa_opt_free_ptr(work);
        return -1;
    }

    ssa_cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, ML, k, kp,
                   (ssa_real)1.0, Q, ML, Vt_svd, kp, (ssa_real)0.0, mssa->U, ML);

    for (int i = 0; i < k; i++)
        ssa_cblas_copy(K, &U_svd[i * K], 1, &mssa->V[i * K], 1);

    for (int i = 0; i < k; i++)
    {
        mssa->sigma[i] = S_svd[i];
        mssa->eigenvalues[i] = S_svd[i] * S_svd[i];
        mssa->total_variance += mssa->eigenvalues[i];
    }

    for (int i = 0; i < k; i++)
    {
        ssa_real sum = (ssa_real)0.0;
        for (int t = 0; t < ML; t++)
            sum += mssa->U[i * ML + t];
        if (sum < (ssa_real)0.0)
        {
            ssa_cblas_scal(ML, (ssa_real)-1.0, &mssa->U[i * ML], 1);
            ssa_cblas_scal(K, (ssa_real)-1.0, &mssa->V[i * K], 1);
        }
    }

    ssa_opt_free_ptr(Omega);
    ssa_opt_free_ptr(Y);
    ssa_opt_free_ptr(Q);
    ssa_opt_free_ptr(B);
    ssa_opt_free_ptr(tau);
    ssa_opt_free_ptr(U_svd);
    ssa_opt_free_ptr(Vt_svd);
    ssa_opt_free_ptr(S_svd);
    ssa_opt_free_ptr(iwork);
    ssa_opt_free_ptr(work);

    mssa->decomposed = true;
    return 0;
}

int mssa_opt_reconstruct(const MSSA_Opt *mssa, int series_idx, const int *group,
                         int n_group, ssa_real *output)
{
    if (!mssa || !mssa->decomposed || !group || !output ||
        n_group < 1 || series_idx < 0 || series_idx >= mssa->M)
        return -1;

    int M = mssa->M, N = mssa->N, L = mssa->L, K = mssa->K, ML = M * L;
    int fft_len = mssa->fft_len, r2c_len = mssa->r2c_len;

    ssa_opt_zero(output, N);

    MSSA_Opt *mssa_mut = (MSSA_Opt *)mssa;

    for (int g = 0; g < n_group; g++)
    {
        int idx = group[g];
        if (idx < 0 || idx >= mssa->n_components)
            continue;

        ssa_real sigma = mssa->sigma[idx];
        const ssa_real *u_full = &mssa->U[idx * ML];
        const ssa_real *u_m = u_full + series_idx * L;
        const ssa_real *v = &mssa->V[idx * K];

        ssa_opt_zero(mssa_mut->ws_real, fft_len);
        for (int i = 0; i < L; i++)
            mssa_mut->ws_real[i] = sigma * u_m[i];
        DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_real, mssa_mut->ws_complex);

        ssa_opt_zero(mssa_mut->ws_batch_real, fft_len);
        for (int i = 0; i < K; i++)
            mssa_mut->ws_batch_real[i] = v[i];
        ssa_real *ws_v = mssa_mut->ws_batch_complex;
        DftiComputeForward(mssa_mut->fft_r2c, mssa_mut->ws_batch_real, ws_v);

        ssa_vMul(r2c_len, (const SSA_MKL_COMPLEX *)mssa_mut->ws_complex,
                 (const SSA_MKL_COMPLEX *)ws_v,
                 (SSA_MKL_COMPLEX *)mssa_mut->ws_complex);

        DftiComputeBackward(mssa_mut->fft_c2r, mssa_mut->ws_complex, mssa_mut->ws_real);

        for (int t = 0; t < N; t++)
            output[t] += mssa_mut->ws_real[t];
    }

    ssa_vMul_real(N, output, mssa->inv_diag_count, output);
    return 0;
}

int mssa_opt_reconstruct_all(const MSSA_Opt *mssa, const int *group,
                             int n_group, ssa_real *output)
{
    if (!mssa || !mssa->decomposed || !group || !output || n_group < 1)
        return -1;

    int M = mssa->M, N = mssa->N;
    for (int m = 0; m < M; m++)
        if (mssa_opt_reconstruct(mssa, m, group, n_group, output + m * N) != 0)
            return -1;
    return 0;
}

int mssa_opt_series_contributions(const MSSA_Opt *mssa, ssa_real *contributions)
{
    if (!mssa || !mssa->decomposed || !contributions)
        return -1;

    int M = mssa->M, L = mssa->L, ML = M * L, k = mssa->n_components;

    for (int i = 0; i < k; i++)
    {
        const ssa_real *u_col = &mssa->U[i * ML];
        for (int m = 0; m < M; m++)
        {
            const ssa_real *u_block = u_col + m * L;
            ssa_real norm_sq = ssa_cblas_dot(L, u_block, 1, u_block, 1);
            contributions[m * k + i] = norm_sq;
        }
    }
    return 0;
}

ssa_real mssa_opt_variance_explained(const MSSA_Opt *mssa, int start, int end)
{
    if (!mssa || !mssa->decomposed || start < 0 || mssa->total_variance <= (ssa_real)0)
        return (ssa_real)0.0;

    if (end < 0 || end >= mssa->n_components)
        end = mssa->n_components - 1;

    ssa_real sum = (ssa_real)0.0;
    for (int i = start; i <= end; i++)
        sum += mssa->eigenvalues[i];

    return sum / mssa->total_variance;
}

// ========================================================================
// ESPRIT - Estimation of Signal Parameters via Rotational Invariance
// ========================================================================

void ssa_opt_parestimate_free(SSA_ParEstimate *result)
{
    if (!result)
        return;
    if (result->periods)
        ssa_opt_free_ptr(result->periods);
    if (result->frequencies)
        ssa_opt_free_ptr(result->frequencies);
    if (result->moduli)
        ssa_opt_free_ptr(result->moduli);
    if (result->rates)
        ssa_opt_free_ptr(result->rates);
    memset(result, 0, sizeof(*result));
}

int ssa_opt_parestimate(const SSA_Opt *ssa, const int *group, int n_group,
                        SSA_ParEstimate *result)
{
    if (!ssa || !ssa->decomposed || !result)
        return -1;

    int k = (n_group > 0 && group) ? n_group : ssa->n_components;
    if (k < 1 || k > ssa->n_components)
        return -1;

    int L = ssa->L;
    if (L < 3)
        return -1;

    int *idx = NULL;
    int idx_allocated = 0;
    if (group && n_group > 0)
    {
        idx = (int *)group;
    }
    else
    {
        idx = (int *)ssa_opt_alloc(k * sizeof(int));
        if (!idx)
            return -1;
        for (int i = 0; i < k; i++)
            idx[i] = i;
        idx_allocated = 1;
    }

    ssa_real *U_sel = (ssa_real *)ssa_opt_alloc(L * k * sizeof(ssa_real));
    if (!U_sel)
    {
        if (idx_allocated)
            ssa_opt_free_ptr(idx);
        return -1;
    }

    for (int j = 0; j < k; j++)
    {
        int comp_idx = idx[j];
        if (comp_idx < 0 || comp_idx >= ssa->n_components)
        {
            ssa_opt_free_ptr(U_sel);
            if (idx_allocated)
                ssa_opt_free_ptr(idx);
            return -1;
        }
        ssa_cblas_copy(L, &ssa->U[comp_idx * L], 1, &U_sel[j * L], 1);
    }

    if (idx_allocated)
        ssa_opt_free_ptr(idx);

    int Lm1 = L - 1;

    ssa_real *U_up = (ssa_real *)ssa_opt_alloc(Lm1 * k * sizeof(ssa_real));
    ssa_real *U_down = (ssa_real *)ssa_opt_alloc(Lm1 * k * sizeof(ssa_real));

    if (!U_up || !U_down)
    {
        if (U_up)
            ssa_opt_free_ptr(U_up);
        if (U_down)
            ssa_opt_free_ptr(U_down);
        ssa_opt_free_ptr(U_sel);
        return -1;
    }

    for (int j = 0; j < k; j++)
    {
        ssa_cblas_copy(Lm1, &U_sel[j * L], 1, &U_up[j * Lm1], 1);
        ssa_cblas_copy(Lm1, &U_sel[j * L + 1], 1, &U_down[j * Lm1], 1);
    }

    ssa_opt_free_ptr(U_sel);

    ssa_real *Z = (ssa_real *)ssa_opt_alloc(k * k * sizeof(ssa_real));
    ssa_real *U_up_copy = (ssa_real *)ssa_opt_alloc(Lm1 * k * sizeof(ssa_real));

    if (!Z || !U_up_copy)
    {
        if (Z)
            ssa_opt_free_ptr(Z);
        if (U_up_copy)
            ssa_opt_free_ptr(U_up_copy);
        ssa_opt_free_ptr(U_up);
        ssa_opt_free_ptr(U_down);
        return -1;
    }

    ssa_cblas_copy(Lm1 * k, U_up, 1, U_up_copy, 1);

    lapack_int info;
    lapack_int lwork = -1;
    ssa_real work_query;

    info = ssa_lapack_dgels_work(LAPACK_COL_MAJOR, 'N', Lm1, k, k,
                                  U_up_copy, Lm1, U_down, Lm1, &work_query, lwork);

    lwork = (lapack_int)work_query + 1;
    ssa_real *work = (ssa_real *)ssa_opt_alloc(lwork * sizeof(ssa_real));
    if (!work)
    {
        ssa_opt_free_ptr(U_up_copy);
        ssa_opt_free_ptr(Z);
        ssa_opt_free_ptr(U_up);
        ssa_opt_free_ptr(U_down);
        return -1;
    }

    info = ssa_lapack_dgels_work(LAPACK_COL_MAJOR, 'N', Lm1, k, k,
                                  U_up_copy, Lm1, U_down, Lm1, work, lwork);

    ssa_opt_free_ptr(work);
    ssa_opt_free_ptr(U_up_copy);
    ssa_opt_free_ptr(U_up);

    if (info != 0)
    {
        ssa_opt_free_ptr(Z);
        ssa_opt_free_ptr(U_down);
        return -1;
    }

    for (int j = 0; j < k; j++)
        ssa_cblas_copy(k, &U_down[j * Lm1], 1, &Z[j * k], 1);

    ssa_opt_free_ptr(U_down);

    ssa_real *wr = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    ssa_real *wi = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));

    if (!wr || !wi)
    {
        if (wr)
            ssa_opt_free_ptr(wr);
        if (wi)
            ssa_opt_free_ptr(wi);
        ssa_opt_free_ptr(Z);
        return -1;
    }

    info = ssa_lapack_geev(LAPACK_COL_MAJOR, 'N', 'N', k, Z, k, wr, wi, NULL, 1, NULL, 1);

    ssa_opt_free_ptr(Z);

    if (info != 0)
    {
        ssa_opt_free_ptr(wr);
        ssa_opt_free_ptr(wi);
        return -1;
    }

    result->periods = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    result->frequencies = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    result->moduli = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    result->rates = (ssa_real *)ssa_opt_alloc(k * sizeof(ssa_real));
    result->n_components = k;

    if (!result->periods || !result->frequencies || !result->moduli || !result->rates)
    {
        ssa_opt_parestimate_free(result);
        ssa_opt_free_ptr(wr);
        ssa_opt_free_ptr(wi);
        return -1;
    }

    const ssa_real TWO_PI = (ssa_real)(2.0 * M_PI);

    for (int i = 0; i < k; i++)
    {
        ssa_real re = wr[i];
        ssa_real im = wi[i];

        ssa_real mod = (ssa_real)sqrt((double)(re * re + im * im));
        result->moduli[i] = mod;

        result->rates[i] = (mod > (ssa_real)1e-12) ? (ssa_real)log((double)mod) : (ssa_real)-30.0;

        ssa_real arg = (ssa_real)atan2((double)im, (double)re);
        ssa_real freq = arg / TWO_PI;
        result->frequencies[i] = freq;

        if ((ssa_real)fabs((double)freq) > (ssa_real)1e-12)
            result->periods[i] = (ssa_real)1.0 / (ssa_real)fabs((double)freq);
        else
            result->periods[i] = (ssa_real)INFINITY;
    }

    ssa_opt_free_ptr(wr);
    ssa_opt_free_ptr(wi);

    return 0;
}

#endif // SSA_OPT_ADVANCED_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // SSA_OPT_ADVANCED_H
