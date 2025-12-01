"""
SSA Benchmark: Compare against Rssa (R implementation)
======================================================

Usage:
    python benchmark_vs_rssa.py           # Run our SSA + generate R script
    Rscript benchmark_rssa.R              # Run Rssa
    python benchmark_vs_rssa.py --compare # Compare accuracy
    python benchmark_vs_rssa.py --speed   # Compare speed
"""

import numpy as np
import argparse
import os
import time
import subprocess

# ============================================================================
# Test Signal Generation
# ============================================================================

def generate_test_signals():
    np.random.seed(42)
    N = 500
    t = np.linspace(0, 10, N)
    
    signals = {}
    signals['sine_noise'] = {'data': np.sin(2*np.pi*t) + 0.3*np.random.randn(N)}
    signals['trend_seasonal'] = {'data': 10 + 0.5*t + 2*np.sin(2*np.pi*t) + 0.5*np.random.randn(N)}
    signals['multi_periodic'] = {'data': np.sin(2*np.pi*t/0.5) + 0.7*np.sin(2*np.pi*t/1.2) + 0.4*np.sin(2*np.pi*t/2.5) + 0.3*np.random.randn(N)}
    signals['nonlinear'] = {'data': np.exp(0.3*t) * np.sin(2*np.pi*t) + 0.5*np.random.randn(N)}
    
    returns = 0.001 + 0.02 * np.random.randn(N)
    signals['stock_sim'] = {'data': 100 * np.exp(np.cumsum(returns))}
    
    return signals

# ============================================================================
# Run Our Implementation
# ============================================================================

def run_our_ssa(signals, L=100, k=20, output_dir='benchmark_data'):
    from ssa_wrapper import SSA
    os.makedirs(output_dir, exist_ok=True)
    
    for name, sig_info in signals.items():
        x = sig_info['data']
        print(f"Processing {name}...")
        
        np.savetxt(f'{output_dir}/{name}_signal.csv', x, delimiter=',')
        
        ssa = SSA(x, L=L)
        ssa.decompose(k=k)
        
        var_explained = [ssa.variance_explained(i, i) for i in range(k)]
        np.savetxt(f'{output_dir}/{name}_variance.csv', var_explained, delimiter=',')
        
        np.savetxt(f'{output_dir}/{name}_trend.csv', ssa.reconstruct([0]), delimiter=',')
        if k >= 3:
            np.savetxt(f'{output_dir}/{name}_periodic.csv', ssa.reconstruct([1, 2]), delimiter=',')
        np.savetxt(f'{output_dir}/{name}_full_recon.csv', ssa.reconstruct(list(range(k))), delimiter=',')
        np.savetxt(f'{output_dir}/{name}_wcorr.csv', ssa.wcorr_matrix(), delimiter=',')
        
        if name == 'trend_seasonal':
            np.savetxt(f'{output_dir}/{name}_forecast.csv', ssa.forecast([0, 1, 2], n_forecast=50), delimiter=',')
    
    print(f"Results saved to {output_dir}/")

# ============================================================================
# Generate R Script
# ============================================================================

def generate_r_script(signals, L=100, k=20, output_dir='benchmark_data'):
    r_script = f'''if (!require("Rssa")) {{ install.packages("Rssa", repos="https://cloud.r-project.org"); library(Rssa) }}
L <- {L}; k <- {k}; output_dir <- "{output_dir}"
'''
    for name in signals.keys():
        r_script += f'''
cat("Processing {name}...\\n")
x <- scan(paste0(output_dir, "/{name}_signal.csv"), quiet=TRUE)
s <- ssa(x, L=L, neig=k)
write.table(s$sigma[1:k]^2/sum(s$sigma^2), paste0(output_dir, "/{name}_variance_rssa.csv"), row.names=F, col.names=F)
write.table(reconstruct(s, groups=list(1))$F1, paste0(output_dir, "/{name}_trend_rssa.csv"), row.names=F, col.names=F)
if(k>=3) write.table(reconstruct(s, groups=list(c(2,3)))$F1, paste0(output_dir, "/{name}_periodic_rssa.csv"), row.names=F, col.names=F)
write.table(reconstruct(s, groups=list(1:k))$F1, paste0(output_dir, "/{name}_full_recon_rssa.csv"), row.names=F, col.names=F)
write.table(wcor(s, groups=1:k), paste0(output_dir, "/{name}_wcorr_rssa.csv"), row.names=F, col.names=F, sep=",")
'''
    r_script += f'''
x <- scan(paste0(output_dir, "/trend_seasonal_signal.csv"), quiet=TRUE)
s <- ssa(x, L=L, neig=k)
write.table(rforecast(s, groups=list(c(1,2,3)), len=50), paste0(output_dir, "/trend_seasonal_forecast_rssa.csv"), row.names=F, col.names=F)
cat("Done\\n")
'''
    with open('benchmark_rssa.R', 'w') as f:
        f.write(r_script)
    print("Generated benchmark_rssa.R")

# ============================================================================
# Compare Results
# ============================================================================

def compare_results(signals, output_dir='benchmark_data'):
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: Our SSA vs Rssa")
    print("="*70)
    
    for name in signals.keys():
        print(f"\n--- {name} ---")
        rssa_var_file = f'{output_dir}/{name}_variance_rssa.csv'
        if not os.path.exists(rssa_var_file):
            print("  [Rssa results not found]")
            continue
        
        our_var = np.loadtxt(f'{output_dir}/{name}_variance.csv')
        rssa_var = np.loadtxt(rssa_var_file)
        print(f"  Variance (first 5): Max diff: {np.abs(our_var[:5] - rssa_var[:5]).max():.2e}")
        
        our_trend = np.loadtxt(f'{output_dir}/{name}_trend.csv')
        rssa_trend = np.loadtxt(f'{output_dir}/{name}_trend_rssa.csv')
        print(f"  Trend: Corr={np.corrcoef(our_trend, rssa_trend)[0,1]:.6f}")
        
        our_full = np.loadtxt(f'{output_dir}/{name}_full_recon.csv')
        rssa_full = np.loadtxt(f'{output_dir}/{name}_full_recon_rssa.csv')
        print(f"  Full:  Corr={np.corrcoef(our_full, rssa_full)[0,1]:.6f}")
    
    our_fc = f'{output_dir}/trend_seasonal_forecast.csv'
    rssa_fc = f'{output_dir}/trend_seasonal_forecast_rssa.csv'
    if os.path.exists(our_fc) and os.path.exists(rssa_fc):
        our = np.loadtxt(our_fc)
        rssa = np.loadtxt(rssa_fc)
        print(f"\n--- Forecast ---")
        print(f"  Corr={np.corrcoef(our, rssa)[0,1]:.6f}")

# ============================================================================
# Speed Comparison
# ============================================================================

def compare_speed(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("SPEED COMPARISON: Our SSA vs Rssa")
    print("="*70)
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Ours (ms)':>10} {'Rssa (ms)':>10} {'Speedup':>8}")
    print("-"*55)
    
    n_runs = 3
    
    for N in [500, 1000, 2000, 5000, 10000]:
        L = N // 4
        k = 30
        
        # Our implementation
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ssa = SSA(x, L=L)
            ssa.decompose(k=k)
            _ = ssa.reconstruct(list(range(k)))
        our_time = (time.perf_counter() - t0) / n_runs * 1000
        
        # Rssa - generate data in R directly, suppress ALL output
        r_code = f'''suppressMessages(suppressWarnings(library(Rssa)));options(warn=-1);set.seed(42);x<-sin(seq(0,50*pi,length.out={N}))+0.3*rnorm({N});t0<-Sys.time();for(i in 1:{n_runs}){{invisible(capture.output({{s<-ssa(x,L={L},neig={k});r<-reconstruct(s,groups=list(1:{k}))}}));}};cat(as.numeric(Sys.time()-t0)/{n_runs}*1000)'''
        
        try:
            result = subprocess.run([r_path, '-e', r_code], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and result.stdout.strip():
                # Robust parsing: extract last numeric value
                out = result.stdout.strip().split()
                nums = [x for x in out if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if nums:
                    rssa_time = float(nums[-1])
                    speedup = rssa_time / our_time
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {rssa_time:>10.1f} {speedup:>7.1f}x")
                else:
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'parse err':>10} {'-':>8}")
                    print(f"  R stdout: {result.stdout[:100]}")
            else:
                print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>8}")
                if result.stderr:
                    print(f"  R error: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'timeout':>10} {'-':>8}")
        except Exception as e:
            print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>8}")
            print(f"  Exception: {e}")
    
    print("-"*55)
    print("Rssa: single-threaded BLAS | Ours: Intel MKL (multi-threaded)")

# ============================================================================
# Performance Benchmark
# ============================================================================

def benchmark_performance(L=100, k=20):
    from ssa_wrapper import SSA
    
    print("\n--- Large Scale Test ---")
    for N in [1000, 5000, 10000, 20000]:
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        L = N // 4
        
        t0 = time.perf_counter()
        ssa = SSA(x, L=L)
        ssa.decompose(k=50)
        _ = ssa.reconstruct(list(range(50)))
        total_time = time.perf_counter() - t0
        
        print(f"N={N:5d}, L={L:5d}, k=50: {total_time*1000:>8.1f} ms")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSA Benchmark vs Rssa')
    parser.add_argument('--compare', action='store_true', help='Compare accuracy')
    parser.add_argument('--perf', action='store_true', help='Performance benchmark')
    parser.add_argument('--speed', action='store_true', help='Speed comparison vs Rssa')
    parser.add_argument('-L', type=int, default=100)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('--r-path', default=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe')
    args = parser.parse_args()
    
    signals = generate_test_signals()
    
    if args.compare:
        compare_results(signals)
    elif args.perf:
        benchmark_performance(args.L, args.k)
    elif args.speed:
        compare_speed(args.r_path)
    else:
        run_our_ssa(signals, args.L, args.k)
        generate_r_script(signals, args.L, args.k)
        benchmark_performance(args.L, args.k)
        print("\nNEXT: Rscript benchmark_rssa.R && python benchmark_vs_rssa.py --compare")
        print("      python benchmark_vs_rssa.py --speed")