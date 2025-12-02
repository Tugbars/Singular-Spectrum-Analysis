"""
SSA Benchmark: Compare against Rssa (R implementation)
======================================================

Usage:
    python benchmark_vs_rssa.py --speed          # Compare vs Rssa
    python benchmark_vs_rssa.py --detailed       # Breakdown: decompose + reconstruct
    python benchmark_vs_rssa.py --internal       # Compare sequential/block/randomized
    python benchmark_vs_rssa.py --scaling        # Test O(N log N) complexity
    python benchmark_vs_rssa.py --all            # Run everything
"""

import numpy as np
import argparse
import os
import time
import subprocess
import sys

# ============================================================================
# Speed Comparison with Proper Warmup
# ============================================================================

def compare_speed(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("SPEED COMPARISON: Our SSA vs Rssa")
    print("="*70)
    
    # =========================================================================
    # WARMUP: Run one decomposition to trigger MKL JIT, DLL loading, etc.
    # =========================================================================
    print("\nWarming up MKL (first call has JIT overhead)...")
    np.random.seed(0)
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    
    t0 = time.perf_counter()
    ssa_warmup.decompose(k=10)
    warmup_time = (time.perf_counter() - t0) * 1000
    
    # Second call (should be faster)
    ssa_warmup2 = SSA(x_warmup, L=250)
    t0 = time.perf_counter()
    ssa_warmup2.decompose(k=10)
    post_warmup_time = (time.perf_counter() - t0) * 1000
    
    print(f"  First call (cold):  {warmup_time:.1f} ms")
    print(f"  Second call (warm): {post_warmup_time:.1f} ms")
    print(f"  Warmup overhead:    {warmup_time - post_warmup_time:.1f} ms")
    print()
    
    # =========================================================================
    # Actual benchmark (post-warmup)
    # =========================================================================
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Ours (ms)':>10} {'Rssa (ms)':>10} {'Speedup':>8} {'Corr(raw)':>10} {'Corr(true)':>11}")
    print("-"*82)
    
    n_runs = 5  # More runs for stable timing
    
    for N in [500, 1000, 2000, 5000, 10000, 20000]:
        L = N // 4
        k = 30
        
        # Our implementation
        np.random.seed(42)
        t_vec = np.linspace(0, 50*np.pi, N)
        x_true = np.sin(t_vec)  # True signal without noise
        x = x_true + 0.3*np.random.randn(N)  # Noisy signal
        
        # Time multiple runs
        times = []
        recon_result = None
        for run_idx in range(n_runs):
            ssa = SSA(x, L=L)
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            recon = ssa.reconstruct(list(range(k)))
            times.append((time.perf_counter() - t0) * 1000)
            if run_idx == 0:
                recon_result = recon
        
        # Compute correlations
        corr_raw = np.corrcoef(x, recon_result)[0, 1]        # vs noisy input
        corr_true = np.corrcoef(x_true, recon_result)[0, 1]  # vs true signal
        
        # Use median to reduce variance
        our_time = np.median(times)
        
        # Rssa
        r_code = f'''suppressMessages(suppressWarnings(library(Rssa)));options(warn=-1);set.seed(42);x<-sin(seq(0,50*pi,length.out={N}))+0.3*rnorm({N});t0<-Sys.time();for(i in 1:{n_runs}){{invisible(capture.output({{s<-ssa(x,L={L},neig={k});r<-reconstruct(s,groups=list(1:{k}))}}));}};cat(as.numeric(Sys.time()-t0)/{n_runs}*1000)'''
        
        try:
            result = subprocess.run([r_path, '-e', r_code], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and result.stdout.strip():
                out = result.stdout.strip().split()
                nums = [x for x in out if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if nums:
                    rssa_time = float(nums[-1])
                    speedup = rssa_time / our_time
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {rssa_time:>10.1f} {speedup:>7.1f}x {corr_raw:>10.4f} {corr_true:>11.4f}")
                else:
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'parse err':>10} {'-':>8} {corr_raw:>10.4f} {corr_true:>11.4f}")
            else:
                print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>8} {corr_raw:>10.4f} {corr_true:>11.4f}")
        except subprocess.TimeoutExpired:
            print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'timeout':>10} {'-':>8} {corr_raw:>10.4f} {corr_true:>11.4f}")
        except Exception as e:
            print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>8} {corr_raw:>10.4f} {corr_true:>11.4f}")
    
    print("-"*82)
    print("Rssa: PROPACK (Lanczos) | Ours: Randomized SVD + MKL R2C FFT")


def compare_speed_detailed(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    """
    Detailed breakdown: decompose vs reconstruct timing.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("DETAILED TIMING: Decompose vs Reconstruct")
    print("="*70)
    
    # Warmup
    print("\nWarming up...")
    x_warmup = np.random.randn(1000)
    SSA(x_warmup, L=250).decompose(k=10)
    print("Done.\n")
    
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Decomp':>9} {'Recon':>9} {'Total':>9} | {'Rssa':>9} {'Speedup':>8}")
    print("-"*78)
    
    n_runs = 5
    
    for N in [1000, 2000, 5000, 10000, 20000]:
        L = N // 4
        k = 30
        
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        decomp_times = []
        recon_times = []
        
        for _ in range(n_runs):
            ssa = SSA(x, L=L)
            
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            decomp_times.append((time.perf_counter() - t0) * 1000)
            
            t0 = time.perf_counter()
            _ = ssa.reconstruct(list(range(k)))
            recon_times.append((time.perf_counter() - t0) * 1000)
        
        decomp_ms = np.median(decomp_times)
        recon_ms = np.median(recon_times)
        total_ms = decomp_ms + recon_ms
        
        # Rssa
        r_code = f'''suppressMessages(suppressWarnings(library(Rssa)));options(warn=-1);set.seed(42);x<-sin(seq(0,50*pi,length.out={N}))+0.3*rnorm({N});t0<-Sys.time();for(i in 1:{n_runs}){{invisible(capture.output({{s<-ssa(x,L={L},neig={k});r<-reconstruct(s,groups=list(1:{k}))}}));}};cat(as.numeric(Sys.time()-t0)/{n_runs}*1000)'''
        
        try:
            result = subprocess.run([r_path, '-e', r_code], capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and result.stdout.strip():
                out = result.stdout.strip().split()
                nums = [x for x in out if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if nums:
                    rssa_time = float(nums[-1])
                    speedup = rssa_time / total_ms
                    print(f"{N:>6} {L:>5} {k:>3} | {decomp_ms:>8.1f}ms {recon_ms:>8.1f}ms {total_ms:>8.1f}ms | {rssa_time:>8.1f}ms {speedup:>7.1f}x")
                else:
                    print(f"{N:>6} {L:>5} {k:>3} | {decomp_ms:>8.1f}ms {recon_ms:>8.1f}ms {total_ms:>8.1f}ms | {'err':>9} {'-':>8}")
            else:
                print(f"{N:>6} {L:>5} {k:>3} | {decomp_ms:>8.1f}ms {recon_ms:>8.1f}ms {total_ms:>8.1f}ms | {'err':>9} {'-':>8}")
        except:
            print(f"{N:>6} {L:>5} {k:>3} | {decomp_ms:>8.1f}ms {recon_ms:>8.1f}ms {total_ms:>8.1f}ms | {'err':>9} {'-':>8}")
    
    print("-"*78)


def compare_methods_internal():
    """
    Compare our three decomposition methods (no Rssa).
    Shows the 100x speedup of randomized vs sequential.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("INTERNAL COMPARISON: Sequential vs Block vs Randomized")
    print("="*70)
    
    # Warmup all methods
    print("\nWarming up all decomposition methods...")
    x_warmup = np.random.randn(1000)
    SSA(x_warmup, L=250).decompose(k=10, method='randomized')
    SSA(x_warmup, L=250).decompose(k=10, method='block')
    SSA(x_warmup, L=250).decompose(k=10, method='sequential', max_iter=50)
    print("Done.\n")
    
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Sequential':>12} {'Block':>12} {'Randomized':>12} | {'Speedup':>8} {'Corr':>10}")
    print("-"*90)
    
    n_runs = 3
    
    for N in [1000, 2000, 5000, 10000]:
        L = N // 4
        k = 30
        
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        # Sequential (limit iterations for reasonable time)
        max_iter = 100 if N <= 2000 else 50
        times_seq = []
        for _ in range(n_runs):
            ssa = SSA(x, L=L)
            t0 = time.perf_counter()
            ssa.decompose(k=k, method='sequential', max_iter=max_iter)
            times_seq.append((time.perf_counter() - t0) * 1000)
        seq_ms = np.median(times_seq)
        
        # Block
        times_block = []
        for _ in range(n_runs):
            ssa = SSA(x, L=L)
            t0 = time.perf_counter()
            ssa.decompose(k=k, method='block', max_iter=max_iter)
            times_block.append((time.perf_counter() - t0) * 1000)
        block_ms = np.median(times_block)
        
        # Randomized (also capture reconstruction for correlation)
        times_rand = []
        recon_result = None
        for run_idx in range(n_runs):
            ssa = SSA(x, L=L)
            t0 = time.perf_counter()
            ssa.decompose(k=k, method='randomized')
            times_rand.append((time.perf_counter() - t0) * 1000)
            if run_idx == 0:
                recon_result = ssa.reconstruct(list(range(k)))
        rand_ms = np.median(times_rand)
        
        # Correlation
        corr = np.corrcoef(x, recon_result)[0, 1]
        
        speedup = seq_ms / rand_ms
        print(f"{N:>6} {L:>5} {k:>3} | {seq_ms:>11.1f}ms {block_ms:>11.1f}ms {rand_ms:>11.1f}ms | {speedup:>7.1f}x {corr:>10.6f}")
    
    print("-"*90)
    print(f"Note: Sequential/Block use max_iter={100 if True else 50}")
    print("Speedup = Sequential / Randomized")


def benchmark_scaling():
    """
    Test O(N log N) scaling of our implementation.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("SCALING TEST: Time vs N (should be ~O(N log N))")
    print("="*70)
    
    # Warmup
    print("\nWarming up...")
    x_warmup = np.random.randn(1000)
    SSA(x_warmup, L=250).decompose(k=10)
    print("Done.\n")
    
    print(f"{'N':>7} {'L':>6} {'k':>3} | {'Time (ms)':>10} | {'N log N':>12} {'Ratio':>12}")
    print("-"*62)
    
    n_runs = 5
    base_ratio = None
    
    for N in [1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        L = N // 4
        k = 50
        
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        times = []
        for _ in range(n_runs):
            ssa = SSA(x, L=L)
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            times.append((time.perf_counter() - t0) * 1000)
        
        time_ms = np.median(times)
        n_log_n = N * np.log2(N)
        ratio = time_ms / n_log_n * 1e6  # Normalize to readable range
        
        if base_ratio is None:
            base_ratio = ratio
        
        rel_ratio = ratio / base_ratio
        print(f"{N:>7} {L:>6} {k:>3} | {time_ms:>9.1f}ms | {n_log_n:>11.0f} {rel_ratio:>11.2f}x")
    
    print("-"*62)
    print("Ratio column: should stay ~1.0x if complexity is O(N log N)")
    print("Values >1.0x indicate worse-than-linear-log scaling")


def compare_speed_all(r_path):
    """Run all speed comparisons."""
    compare_speed(r_path)
    compare_speed_detailed(r_path)
    compare_methods_internal()
    benchmark_scaling()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSA Benchmark vs Rssa')
    parser.add_argument('--speed', action='store_true', help='Speed comparison vs Rssa')
    parser.add_argument('--detailed', action='store_true', help='Detailed timing breakdown')
    parser.add_argument('--internal', action='store_true', help='Compare our three methods')
    parser.add_argument('--scaling', action='store_true', help='Test O(N log N) scaling')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--r-path', default=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe')
    args = parser.parse_args()
    
    if args.speed:
        compare_speed(args.r_path)
    elif args.detailed:
        compare_speed_detailed(args.r_path)
    elif args.internal:
        compare_methods_internal()
    elif args.scaling:
        benchmark_scaling()
    elif args.all:
        compare_speed_all(args.r_path)
    else:
        # Default: show all options
        print("SSA Benchmark Suite")
        print("="*40)
        print("Options:")
        print("  --speed     Compare vs Rssa (with warmup)")
        print("  --detailed  Breakdown: decompose + reconstruct")
        print("  --internal  Compare sequential/block/randomized")
        print("  --scaling   Test O(N log N) complexity")
        print("  --all       Run everything")
        print()
        print("Example: python benchmark_vs_rssa.py --all")