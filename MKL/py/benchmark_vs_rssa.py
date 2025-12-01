"""
SSA Benchmark: Compare against Rssa (R implementation)
======================================================

This script:
1. Generates test signals
2. Runs our SSA implementation
3. Generates an R script to run Rssa
4. Compares results

Requirements:
- Python: numpy, ssa_wrapper
- R: Rssa package (install.packages("Rssa"))

Usage:
    python benchmark_vs_rssa.py
    Rscript benchmark_rssa.R
    python benchmark_vs_rssa.py --compare
"""

import numpy as np
import argparse
import os
from pathlib import Path

# ============================================================================
# Test Signal Generation
# ============================================================================

def generate_test_signals():
    """Generate various test signals for benchmarking."""
    np.random.seed(42)
    N = 500
    t = np.linspace(0, 10, N)
    
    signals = {}
    
    # 1. Simple sine + noise
    signals['sine_noise'] = {
        'data': np.sin(2 * np.pi * t) + 0.3 * np.random.randn(N),
        'description': 'Sine wave with Gaussian noise'
    }
    
    # 2. Trend + seasonal
    signals['trend_seasonal'] = {
        'data': 10 + 0.5*t + 2*np.sin(2*np.pi*t) + 0.5*np.random.randn(N),
        'description': 'Linear trend + seasonal + noise'
    }
    
    # 3. Multiple periodicities
    signals['multi_periodic'] = {
        'data': (np.sin(2*np.pi*t/0.5) + 
                 0.7*np.sin(2*np.pi*t/1.2) + 
                 0.4*np.sin(2*np.pi*t/2.5) + 
                 0.3*np.random.randn(N)),
        'description': 'Three periodic components + noise'
    }
    
    # 4. Nonlinear trend
    signals['nonlinear'] = {
        'data': np.exp(0.3*t) * np.sin(2*np.pi*t) + 0.5*np.random.randn(N),
        'description': 'Exponentially modulated sine'
    }
    
    # 5. Real-world like (stock price simulation)
    returns = 0.001 + 0.02 * np.random.randn(N)
    price = 100 * np.exp(np.cumsum(returns))
    signals['stock_sim'] = {
        'data': price,
        'description': 'Simulated stock price (geometric Brownian motion)'
    }
    
    return signals


# ============================================================================
# Run Our Implementation
# ============================================================================

def run_our_ssa(signals, L=100, k=20, output_dir='benchmark_data'):
    """Run our SSA implementation and save results."""
    from ssa_wrapper import SSA
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for name, sig_info in signals.items():
        x = sig_info['data']
        print(f"Processing {name}...")
        
        # Save input signal
        np.savetxt(f'{output_dir}/{name}_signal.csv', x, delimiter=',')
        
        # Run SSA
        ssa = SSA(x, L=L)
        ssa.decompose(k=k)
        
        # Extract results
        results[name] = {
            'singular_values': [],
            'variance_explained': [],
            'reconstructions': {},
            'wcorr': None
        }
        
        # Singular values (via variance explained for individual components)
        total_var = ssa.variance_explained(0, k-1)
        for i in range(k):
            var_i = ssa.variance_explained(i, i)
            results[name]['variance_explained'].append(var_i)
        
        # Save variance explained
        np.savetxt(f'{output_dir}/{name}_variance.csv', 
                   results[name]['variance_explained'], delimiter=',')
        
        # Reconstructions for key component groups
        # Trend (component 0)
        trend = ssa.reconstruct([0])
        np.savetxt(f'{output_dir}/{name}_trend.csv', trend, delimiter=',')
        results[name]['reconstructions']['trend'] = trend
        
        # First periodic pair (components 1,2)
        if k >= 3:
            periodic = ssa.reconstruct([1, 2])
            np.savetxt(f'{output_dir}/{name}_periodic.csv', periodic, delimiter=',')
            results[name]['reconstructions']['periodic'] = periodic
        
        # Full reconstruction (all k components)
        full_recon = ssa.reconstruct(list(range(k)))
        np.savetxt(f'{output_dir}/{name}_full_recon.csv', full_recon, delimiter=',')
        results[name]['reconstructions']['full'] = full_recon
        
        # W-correlation matrix
        wcorr = ssa.wcorr_matrix()
        np.savetxt(f'{output_dir}/{name}_wcorr.csv', wcorr, delimiter=',')
        results[name]['wcorr'] = wcorr
        
        # Forecast (for trend+seasonal signal)
        if name == 'trend_seasonal':
            forecast = ssa.forecast([0, 1, 2], n_forecast=50)
            np.savetxt(f'{output_dir}/{name}_forecast.csv', forecast, delimiter=',')
            results[name]['forecast'] = forecast
    
    print(f"\nResults saved to {output_dir}/")
    return results


# ============================================================================
# Generate R Script
# ============================================================================

def generate_r_script(signals, L=100, k=20, output_dir='benchmark_data'):
    """Generate R script to run Rssa on same data."""
    
    r_script = f'''# Rssa Benchmark Script
# Run with: Rscript benchmark_rssa.R

# Install Rssa if needed
if (!require("Rssa")) {{
    install.packages("Rssa", repos="https://cloud.r-project.org")
    library(Rssa)
}}

L <- {L}
k <- {k}
output_dir <- "{output_dir}"

'''
    
    for name in signals.keys():
        r_script += f'''
# === {name} ===
cat("Processing {name}...\\n")
x <- scan(paste0(output_dir, "/{name}_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/{name}_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/{name}_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {{
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/{name}_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/{name}_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/{name}_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")

'''
    
    # Add forecasting for trend_seasonal
    r_script += '''
# === Forecasting (trend_seasonal) ===
cat("Forecasting trend_seasonal...\\n")
x <- scan(paste0(output_dir, "/trend_seasonal_signal.csv"), quiet=TRUE)
s <- ssa(x, L=L, neig=k)

# Forecast using components 1,2,3 (R indexing)
forecast_result <- rforecast(s, groups=list(c(1,2,3)), len=50)
write.table(forecast_result, paste0(output_dir, "/trend_seasonal_forecast_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

cat("\\nRssa results saved to", output_dir, "\\n")
'''
    
    with open('benchmark_rssa.R', 'w') as f:
        f.write(r_script)
    
    print("Generated benchmark_rssa.R")


# ============================================================================
# Compare Results
# ============================================================================

def compare_results(signals, output_dir='benchmark_data'):
    """Compare our results with Rssa results."""
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: Our SSA vs Rssa")
    print("="*70)
    
    for name in signals.keys():
        print(f"\n--- {name} ---")
        
        # Check if Rssa results exist
        rssa_var_file = f'{output_dir}/{name}_variance_rssa.csv'
        if not os.path.exists(rssa_var_file):
            print("  [Rssa results not found - run: Rscript benchmark_rssa.R]")
            continue
        
        # Compare variance explained
        our_var = np.loadtxt(f'{output_dir}/{name}_variance.csv')
        rssa_var = np.loadtxt(rssa_var_file)
        
        # Rssa might return fewer components
        n_compare = min(len(our_var), len(rssa_var))
        var_diff = np.abs(our_var[:n_compare] - rssa_var[:n_compare])
        
        print(f"  Variance explained (first 5):")
        print(f"    Ours:  {our_var[:5]}")
        print(f"    Rssa:  {rssa_var[:5]}")
        print(f"    Max diff: {var_diff.max():.2e}")
        
        # Compare trend reconstruction
        our_trend = np.loadtxt(f'{output_dir}/{name}_trend.csv')
        rssa_trend = np.loadtxt(f'{output_dir}/{name}_trend_rssa.csv')
        
        trend_rmse = np.sqrt(np.mean((our_trend - rssa_trend)**2))
        trend_corr = np.corrcoef(our_trend, rssa_trend)[0, 1]
        
        print(f"  Trend reconstruction:")
        print(f"    RMSE: {trend_rmse:.6f}")
        print(f"    Correlation: {trend_corr:.6f}")
        
        # Compare periodic reconstruction
        our_periodic_file = f'{output_dir}/{name}_periodic.csv'
        rssa_periodic_file = f'{output_dir}/{name}_periodic_rssa.csv'
        
        if os.path.exists(our_periodic_file) and os.path.exists(rssa_periodic_file):
            our_periodic = np.loadtxt(our_periodic_file)
            rssa_periodic = np.loadtxt(rssa_periodic_file)
            
            periodic_rmse = np.sqrt(np.mean((our_periodic - rssa_periodic)**2))
            periodic_corr = np.corrcoef(our_periodic, rssa_periodic)[0, 1]
            
            print(f"  Periodic reconstruction:")
            print(f"    RMSE: {periodic_rmse:.6f}")
            print(f"    Correlation: {periodic_corr:.6f}")
        
        # Compare full reconstruction
        our_full = np.loadtxt(f'{output_dir}/{name}_full_recon.csv')
        rssa_full = np.loadtxt(f'{output_dir}/{name}_full_recon_rssa.csv')
        
        full_rmse = np.sqrt(np.mean((our_full - rssa_full)**2))
        full_corr = np.corrcoef(our_full, rssa_full)[0, 1]
        
        print(f"  Full reconstruction (k components):")
        print(f"    RMSE: {full_rmse:.6f}")
        print(f"    Correlation: {full_corr:.6f}")
        
        # Compare W-correlation matrices
        our_wcorr = np.loadtxt(f'{output_dir}/{name}_wcorr.csv', delimiter=',')
        rssa_wcorr_file = f'{output_dir}/{name}_wcorr_rssa.csv'
        
        if os.path.exists(rssa_wcorr_file):
            rssa_wcorr = np.loadtxt(rssa_wcorr_file, delimiter=',')
            n_w = min(our_wcorr.shape[0], rssa_wcorr.shape[0])
            wcorr_diff = np.abs(our_wcorr[:n_w, :n_w] - rssa_wcorr[:n_w, :n_w])
            
            print(f"  W-correlation matrix:")
            print(f"    Max absolute diff: {wcorr_diff.max():.6f}")
            print(f"    Mean absolute diff: {wcorr_diff.mean():.6f}")
    
    # Compare forecasts
    our_fc_file = f'{output_dir}/trend_seasonal_forecast.csv'
    rssa_fc_file = f'{output_dir}/trend_seasonal_forecast_rssa.csv'
    
    if os.path.exists(our_fc_file) and os.path.exists(rssa_fc_file):
        print(f"\n--- Forecasting (trend_seasonal) ---")
        our_fc = np.loadtxt(our_fc_file)
        rssa_fc = np.loadtxt(rssa_fc_file)
        
        n_fc = min(len(our_fc), len(rssa_fc))
        fc_rmse = np.sqrt(np.mean((our_fc[:n_fc] - rssa_fc[:n_fc])**2))
        fc_corr = np.corrcoef(our_fc[:n_fc], rssa_fc[:n_fc])[0, 1]
        
        print(f"  50-step forecast:")
        print(f"    RMSE: {fc_rmse:.6f}")
        print(f"    Correlation: {fc_corr:.6f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("  - Correlation > 0.999: Excellent match")
    print("  - Correlation > 0.99:  Good match (numerical differences)")
    print("  - Correlation < 0.99:  Investigate differences")
    print("  - Small RMSE: Absolute values match well")
    print("="*70)


# ============================================================================
# Performance Benchmark
# ============================================================================

def benchmark_performance(signals, L=100, k=20, n_runs=5):
    """Benchmark our implementation's performance."""
    import time
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    for name, sig_info in signals.items():
        x = sig_info['data']
        N = len(x)
        
        times = {'init': [], 'decompose': [], 'reconstruct': [], 'forecast': []}
        
        for _ in range(n_runs):
            # Init
            t0 = time.perf_counter()
            ssa = SSA(x, L=L)
            times['init'].append(time.perf_counter() - t0)
            
            # Decompose
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            times['decompose'].append(time.perf_counter() - t0)
            
            # Reconstruct
            t0 = time.perf_counter()
            _ = ssa.reconstruct(list(range(k)))
            times['reconstruct'].append(time.perf_counter() - t0)
            
            # Forecast
            t0 = time.perf_counter()
            _ = ssa.forecast([0, 1, 2], n_forecast=50)
            times['forecast'].append(time.perf_counter() - t0)
        
        print(f"\n{name} (N={N}, L={L}, k={k}):")
        print(f"  Init:        {np.mean(times['init'])*1000:.3f} ms")
        print(f"  Decompose:   {np.mean(times['decompose'])*1000:.3f} ms")
        print(f"  Reconstruct: {np.mean(times['reconstruct'])*1000:.3f} ms")
        print(f"  Forecast:    {np.mean(times['forecast'])*1000:.3f} ms")
        print(f"  Total:       {sum(np.mean(times[k]) for k in times)*1000:.3f} ms")
    
    # Large scale test
    print("\n--- Large Scale Test ---")
    for N in [1000, 5000, 10000]:
        L = N // 5
        k = 50
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        t0 = time.perf_counter()
        ssa = SSA(x, L=L)
        ssa.decompose(k=k)
        _ = ssa.reconstruct(list(range(k)))
        total_time = time.perf_counter() - t0
        
        print(f"N={N:5d}, L={L:4d}, k={k}: {total_time*1000:.1f} ms")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSA Benchmark vs Rssa')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare results (run after Rscript)')
    parser.add_argument('--perf', action='store_true',
                        help='Run performance benchmark only')
    parser.add_argument('-L', type=int, default=100, help='Window length')
    parser.add_argument('-k', type=int, default=20, help='Number of components')
    args = parser.parse_args()
    
    signals = generate_test_signals()
    
    if args.compare:
        compare_results(signals)
    elif args.perf:
        benchmark_performance(signals, L=args.L, k=args.k)
    else:
        # Run our implementation
        print("Running our SSA implementation...")
        run_our_ssa(signals, L=args.L, k=args.k)
        
        # Generate R script
        generate_r_script(signals, L=args.L, k=args.k)
        
        # Run performance benchmark
        benchmark_performance(signals, L=args.L, k=args.k)
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("  1. Install R and Rssa: install.packages('Rssa')")
        print("  2. Run: Rscript benchmark_rssa.R")
        print("  3. Compare: python benchmark_vs_rssa.py --compare")
        print("="*70)
