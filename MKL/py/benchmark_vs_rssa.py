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
import ctypes

# ============================================================================
# Speed Comparison with Proper Warmup
# ============================================================================

def compare_speed(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*70)
    print("SPEED COMPARISON: Our SSA vs Rssa (Malloc-Free Hot Path)")
    print("="*70)
    
    # Apply MKL config
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
        print("\nApplying MKL configuration...")
        _lib.ssa_mkl_init(1)
    except Exception as e:
        print(f"Note: MKL config not available: {e}")
    
    # =========================================================================
    # WARMUP
    # =========================================================================
    print("\nWarming up MKL (first call has JIT overhead)...")
    np.random.seed(0)
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), 30, 8)
    
    t0 = time.perf_counter()
    ssa_warmup.decompose(k=10)
    warmup_time = (time.perf_counter() - t0) * 1000
    
    # Second call (should be faster - uses prepared workspace)
    x_warmup[0] += 0.001
    _lib.ssa_opt_update_signal(ctypes.byref(ssa_warmup._ctx), x_warmup.ctypes.data_as(c_double_p))
    t0 = time.perf_counter()
    ssa_warmup.decompose(k=10)
    post_warmup_time = (time.perf_counter() - t0) * 1000
    
    print(f"  First call (cold):  {warmup_time:.1f} ms")
    print(f"  Second call (warm): {post_warmup_time:.1f} ms")
    print(f"  Warmup overhead:    {warmup_time - post_warmup_time:.1f} ms")
    print()
    
    # =========================================================================
    # Actual benchmark (post-warmup, malloc-free hot path)
    # =========================================================================
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Ours (ms)':>10} {'Rssa (ms)':>10} {'Speedup':>8} {'Corr(raw)':>10} {'Corr(true)':>11}")
    print("-"*82)
    
    n_runs = 5  # More runs for stable timing
    
    for N in [500, 1000, 2000, 5000, 10000, 20000]:
        L = N // 4
        k = 50
        
        # Our implementation with malloc-free path
        np.random.seed(42)
        t_vec = np.linspace(0, 50*np.pi, N)
        x_true = np.sin(t_vec)  # True signal without noise
        x = x_true + 0.3*np.random.randn(N)  # Noisy signal
        
        # Setup once
        ssa = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
        
        # Time multiple runs using prepared path
        times = []
        recon_result = None
        for run_idx in range(n_runs):
            # Update signal (simulates new data arriving)
            x[0] += 0.0001  # Tiny perturbation
            _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x.ctypes.data_as(c_double_p))
            
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
    print("Rssa: PROPACK (Lanczos) | Ours: Randomized SVD + MKL R2C FFT + Malloc-Free Path")


def compare_speed_detailed(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    """
    Detailed breakdown: decompose vs reconstruct timing (malloc-free path).
    """
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*70)
    print("DETAILED TIMING: Decompose vs Reconstruct (Malloc-Free)")
    print("="*70)
    
    # Apply MKL config and setup function signatures
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
        _lib.ssa_mkl_init(1)
    except Exception as e:
        print(f"Note: MKL config not available: {e}")
    
    # Warmup
    print("\nWarming up...")
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), 30, 8)
    ssa_warmup.decompose(k=10)
    print("Done.\n")
    
    print(f"{'N':>6} {'L':>5} {'k':>3} | {'Decomp':>9} {'Recon':>9} {'Total':>9} | {'Rssa':>9} {'Speedup':>8}")
    print("-"*78)
    
    n_runs = 5
    
    for N in [1000, 2000, 5000, 10000, 20000]:
        L = N // 4
        k = 30
        
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        
        # Setup once with prepare
        ssa = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
        
        decomp_times = []
        recon_times = []
        
        for run_idx in range(n_runs):
            # Update signal
            x[0] += 0.0001
            _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x.ctypes.data_as(c_double_p))
            
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
    Uses malloc-free path for randomized method.
    """
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*70)
    print("INTERNAL COMPARISON: Sequential vs Block vs Randomized")
    print("="*70)
    
    # Apply MKL config
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
        _lib.ssa_mkl_init(1)
    except Exception as e:
        print(f"Note: MKL config not available: {e}")
    
    # Warmup all methods
    print("\nWarming up all decomposition methods...")
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), 30, 8)
    ssa_warmup.decompose(k=10, method='randomized')
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
        
        # Randomized with malloc-free path
        ssa_rand = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa_rand._ctx), k, 8)
        
        times_rand = []
        recon_result = None
        for run_idx in range(n_runs):
            x[0] += 0.0001
            _lib.ssa_opt_update_signal(ctypes.byref(ssa_rand._ctx), x.ctypes.data_as(c_double_p))
            
            t0 = time.perf_counter()
            ssa_rand.decompose(k=k, method='randomized')
            times_rand.append((time.perf_counter() - t0) * 1000)
            if run_idx == 0:
                recon_result = ssa_rand.reconstruct(list(range(k)))
        rand_ms = np.median(times_rand)
        
        # Correlation
        corr = np.corrcoef(x, recon_result)[0, 1]
        
        speedup = seq_ms / rand_ms
        print(f"{N:>6} {L:>5} {k:>3} | {seq_ms:>11.1f}ms {block_ms:>11.1f}ms {rand_ms:>11.1f}ms | {speedup:>7.1f}x {corr:>10.6f}")
    
    print("-"*90)
    print(f"Note: Sequential/Block use max_iter={100 if True else 50}")
    print("Speedup = Sequential / Randomized (malloc-free)")


def benchmark_scaling():
    """
    Test O(N log N) scaling of our implementation (malloc-free path).
    """
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*70)
    print("SCALING TEST: Time vs N (should be ~O(N log N)) - Malloc-Free")
    print("="*70)
    
    # Apply MKL config
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
        _lib.ssa_mkl_init(1)
    except Exception as e:
        print(f"Note: MKL config not available: {e}")
    
    # Warmup
    print("\nWarming up...")
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), 50, 8)
    ssa_warmup.decompose(k=10)
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
        
        # Setup with malloc-free path
        ssa = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
        
        times = []
        for run_idx in range(n_runs):
            x[0] += 0.0001
            _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x.ctypes.data_as(c_double_p))
            
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


def compare_malloc_free(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    """
    Benchmark malloc-free hot path vs Rssa.
    
    This tests the ssa_opt_prepare() / ssa_opt_update_signal() API
    that provides consistent, allocation-free performance.
    """
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*80)
    print("MALLOC-FREE HOT PATH BENCHMARK vs Rssa")
    print("="*80)
    
    # Always apply MKL config first
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        print("\nApplying MKL configuration...")
        _lib.ssa_mkl_init(1)  # verbose=1
    except Exception as e:
        print(f"Note: MKL config not available: {e}")
    
    # Setup function signatures
    try:
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
    except Exception as e:
        print(f"\nERROR: Malloc-free functions not available: {e}")
        return
    
    print("\nSimulating trading hot loop (malloc-free path):")
    print("  init → prepare → [update_signal → decompose → reconstruct] × N")
    print()
    
    test_configs = [
        (1000, 250, 30, 100),   # N, L, k, iterations
        (2000, 500, 30, 50),
        (5000, 1250, 30, 20),
        (10000, 2500, 50, 10),
    ]
    
    n_rssa_runs = 5
    
    print(f"{'N':>6} {'L':>5} {'k':>3} {'iters':>6} | {'Ours/iter':>10} {'Rssa/iter':>10} {'Speedup':>9}")
    print("-"*70)
    
    for N, L, k, n_iterations in test_configs:
        np.random.seed(42)
        x = np.sin(np.linspace(0, 50*np.pi, N)) + 0.3*np.random.randn(N)
        group = list(range(k))
        
        # Warmup
        ssa_warmup = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), k, 8)
        ssa_warmup.decompose(k=k)
        
        # Benchmark malloc-free hot loop
        times = []
        for _ in range(3):  # 3 trials
            x_copy = x.copy()
            ssa = SSA(x_copy, L=L)
            _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
            
            t0 = time.perf_counter()
            for i in range(n_iterations):
                x_copy[0] += 0.001
                x_ptr = x_copy.ctypes.data_as(c_double_p)
                _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x_ptr)
                ssa.decompose(k=k)
                _ = ssa.reconstruct(group)
            times.append((time.perf_counter() - t0) * 1000)
        
        our_per_iter = np.median(times) / n_iterations
        
        # Rssa timing
        r_code = f'''suppressMessages(suppressWarnings(library(Rssa)));options(warn=-1);set.seed(42);x<-sin(seq(0,50*pi,length.out={N}))+0.3*rnorm({N});t0<-Sys.time();for(i in 1:{n_rssa_runs}){{invisible(capture.output({{s<-ssa(x,L={L},neig={k});r<-reconstruct(s,groups=list(1:{k}))}}));}};cat(as.numeric(Sys.time()-t0)/{n_rssa_runs}*1000)'''
        
        rssa_per_iter = None
        try:
            result = subprocess.run([r_path, '-e', r_code], capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and result.stdout.strip():
                out = result.stdout.strip().split()
                nums = [n for n in out if n.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if nums:
                    rssa_per_iter = float(nums[-1])
        except:
            pass
        
        if rssa_per_iter:
            speedup = rssa_per_iter / our_per_iter
            print(f"{N:>6} {L:>5} {k:>3} {n_iterations:>6} | {our_per_iter:>8.2f}ms {rssa_per_iter:>8.1f}ms {speedup:>8.1f}x")
        else:
            print(f"{N:>6} {L:>5} {k:>3} {n_iterations:>6} | {our_per_iter:>8.2f}ms {'err':>10} {'-':>9}")
    
    print("-"*70)
    print("\nNote: All timings use malloc-free hot path (prepare + update_signal)")


def compare_mkl_config(r_path=r'C:\Program Files\R\R-4.5.2\bin\Rscript.exe'):
    """
    Compare performance with and without MKL configuration.
    Uses malloc-free hot path throughout.
    
    Tests:
    1. Default MKL settings (no config)
    2. With ssa_mkl_init() - full SSA-optimized config
    3. Compare both against Rssa
    """
    from ssa_wrapper import SSA, _lib, _SSA_Opt, c_double_p
    
    print("\n" + "="*80)
    print("MKL CONFIGURATION COMPARISON (Malloc-Free Path)")
    print("="*80)
    
    # Check if MKL config functions are available
    try:
        _lib.ssa_mkl_init.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_init.restype = ctypes.c_int
        _lib.ssa_mkl_get_threads.argtypes = []
        _lib.ssa_mkl_get_threads.restype = ctypes.c_int
        _lib.ssa_mkl_set_threads.argtypes = [ctypes.c_int]
        _lib.ssa_mkl_set_threads.restype = None
        _lib.ssa_mkl_get_cpu_info.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        _lib.ssa_mkl_get_cpu_info.restype = None
        _lib.ssa_opt_prepare.argtypes = [ctypes.POINTER(_SSA_Opt), ctypes.c_int, ctypes.c_int]
        _lib.ssa_opt_prepare.restype = ctypes.c_int
        _lib.ssa_opt_update_signal.argtypes = [ctypes.POINTER(_SSA_Opt), c_double_p]
        _lib.ssa_opt_update_signal.restype = ctypes.c_int
        has_mkl_config = True
    except Exception as e:
        print(f"\nWARNING: MKL config functions not available in DLL: {e}")
        print("Rebuild DLL with ssa_dll.c to enable MKL configuration.\n")
        has_mkl_config = False
        return
    
    # Get CPU info
    p_cores = ctypes.c_int()
    e_cores = ctypes.c_int()
    is_hybrid = ctypes.c_int()
    has_avx512 = ctypes.c_int()
    _lib.ssa_mkl_get_cpu_info(
        ctypes.byref(p_cores), ctypes.byref(e_cores),
        ctypes.byref(is_hybrid), ctypes.byref(has_avx512)
    )
    
    print(f"\nCPU Info:")
    print(f"  P-cores: {p_cores.value}")
    print(f"  E-cores: {e_cores.value}")
    print(f"  Hybrid:  {'Yes' if is_hybrid.value else 'No'}")
    print(f"  AVX-512: {'Yes' if has_avx512.value else 'No'}")
    
    # Test parameters
    test_configs = [
        (1000, 250, 30),
        (5000, 1250, 30),
        (10000, 2500, 50),
        (20000, 5000, 50),
    ]
    
    n_runs = 5
    
    # =========================================================================
    # Test 1: Default MKL settings (with malloc-free path)
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Default MKL Settings (no optimization, but malloc-free)")
    print("-"*80)
    
    # Reset to default
    default_threads = _lib.ssa_mkl_get_threads()
    print(f"Current MKL threads: {default_threads}")
    
    # Warmup
    np.random.seed(0)
    x_warmup = np.random.randn(1000)
    ssa_warmup = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup._ctx), 50, 8)
    ssa_warmup.decompose(k=10)
    
    default_times = {}
    
    print(f"\n{'N':>6} {'L':>5} {'k':>3} | {'Time (ms)':>10} {'Corr(true)':>11}")
    print("-"*45)
    
    for N, L, k in test_configs:
        np.random.seed(42)
        t_vec = np.linspace(0, 50*np.pi, N)
        x_true = np.sin(t_vec)
        x = x_true + 0.3*np.random.randn(N)
        
        # Setup with malloc-free path
        ssa = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
        
        times = []
        recon = None
        for run_idx in range(n_runs):
            x[0] += 0.0001
            _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x.ctypes.data_as(c_double_p))
            
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            recon = ssa.reconstruct(list(range(k)))
            times.append((time.perf_counter() - t0) * 1000)
        
        time_ms = np.median(times)
        corr = np.corrcoef(x_true, recon)[0, 1]
        default_times[(N, L, k)] = time_ms
        
        print(f"{N:>6} {L:>5} {k:>3} | {time_ms:>10.1f} {corr:>11.4f}")
    
    # =========================================================================
    # Test 2: With MKL SSA-optimized config (malloc-free path)
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 2: With ssa_mkl_init() - SSA-Optimized Config (malloc-free)")
    print("-"*80)
    
    # Apply SSA-optimized configuration
    print("\nApplying MKL configuration...")
    _lib.ssa_mkl_init(1)  # verbose=1
    
    optimized_threads = _lib.ssa_mkl_get_threads()
    print(f"MKL threads after config: {optimized_threads}")
    
    # Warmup again after config change
    ssa_warmup2 = SSA(x_warmup, L=250)
    _lib.ssa_opt_prepare(ctypes.byref(ssa_warmup2._ctx), 50, 8)
    ssa_warmup2.decompose(k=10)
    
    optimized_times = {}
    
    print(f"\n{'N':>6} {'L':>5} {'k':>3} | {'Time (ms)':>10} {'Corr(true)':>11} {'Speedup':>10}")
    print("-"*58)
    
    for N, L, k in test_configs:
        np.random.seed(42)
        t_vec = np.linspace(0, 50*np.pi, N)
        x_true = np.sin(t_vec)
        x = x_true + 0.3*np.random.randn(N)
        
        # Setup with malloc-free path
        ssa = SSA(x, L=L)
        _lib.ssa_opt_prepare(ctypes.byref(ssa._ctx), k, 8)
        
        times = []
        recon = None
        for run_idx in range(n_runs):
            x[0] += 0.0001
            _lib.ssa_opt_update_signal(ctypes.byref(ssa._ctx), x.ctypes.data_as(c_double_p))
            
            t0 = time.perf_counter()
            ssa.decompose(k=k)
            recon = ssa.reconstruct(list(range(k)))
            times.append((time.perf_counter() - t0) * 1000)
        
        time_ms = np.median(times)
        corr = np.corrcoef(x_true, recon)[0, 1]
        optimized_times[(N, L, k)] = time_ms
        
        speedup = default_times[(N, L, k)] / time_ms
        print(f"{N:>6} {L:>5} {k:>3} | {time_ms:>10.1f} {corr:>11.4f} {speedup:>9.2f}x")
    
    # =========================================================================
    # Test 3: Compare against Rssa
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Optimized SSA (malloc-free) vs Rssa")
    print("-"*80)
    
    print(f"\n{'N':>6} {'L':>5} {'k':>3} | {'Ours (ms)':>10} {'Rssa (ms)':>10} {'Speedup':>10}")
    print("-"*58)
    
    for N, L, k in test_configs:
        our_time = optimized_times[(N, L, k)]
        
        # Rssa timing
        r_code = f'''suppressMessages(suppressWarnings(library(Rssa)));options(warn=-1);set.seed(42);x<-sin(seq(0,50*pi,length.out={N}))+0.3*rnorm({N});t0<-Sys.time();for(i in 1:{n_runs}){{invisible(capture.output({{s<-ssa(x,L={L},neig={k});r<-reconstruct(s,groups=list(1:{k}))}}));}};cat(as.numeric(Sys.time()-t0)/{n_runs}*1000)'''
        
        try:
            result = subprocess.run([r_path, '-e', r_code], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and result.stdout.strip():
                out = result.stdout.strip().split()
                nums = [x for x in out if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if nums:
                    rssa_time = float(nums[-1])
                    speedup = rssa_time / our_time
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {rssa_time:>10.1f} {speedup:>9.1f}x")
                else:
                    print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'parse err':>10} {'-':>10}")
            else:
                print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>10}")
        except Exception as e:
            print(f"{N:>6} {L:>5} {k:>3} | {our_time:>10.1f} {'error':>10} {'-':>10}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY (All tests use malloc-free hot path)")
    print("="*80)
    
    print(f"\n{'Config':>25} | ", end="")
    for N, L, k in test_configs:
        print(f"N={N:>5} ", end="")
    print()
    print("-"*80)
    
    print(f"{'Default MKL':>25} | ", end="")
    for key in test_configs:
        print(f"{default_times[key]:>7.1f}ms ", end="")
    print()
    
    print(f"{'Optimized (ssa_mkl_init)':>25} | ", end="")
    for key in test_configs:
        print(f"{optimized_times[key]:>7.1f}ms ", end="")
    print()
    
    print(f"{'Improvement':>25} | ", end="")
    for key in test_configs:
        speedup = default_times[key] / optimized_times[key]
        print(f"{speedup:>7.2f}x ", end="")
    print()
    
    print("="*80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSA Benchmark vs Rssa')
    parser.add_argument('--speed', action='store_true', help='Speed comparison vs Rssa')
    parser.add_argument('--detailed', action='store_true', help='Detailed timing breakdown')
    parser.add_argument('--internal', action='store_true', help='Compare our three methods')
    parser.add_argument('--scaling', action='store_true', help='Test O(N log N) scaling')
    parser.add_argument('--mkl-config', action='store_true', help='Compare with/without MKL config')
    parser.add_argument('--malloc-free', action='store_true', help='Benchmark malloc-free hot path vs Rssa')
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
    elif args.mkl_config:
        compare_mkl_config(args.r_path)
    elif args.malloc_free:
        compare_malloc_free(args.r_path)
    elif args.all:
        compare_speed_all(args.r_path)
    else:
        # Default: show all options
        print("SSA Benchmark Suite")
        print("="*40)
        print("Options:")
        print("  --speed      Compare vs Rssa (with warmup)")
        print("  --detailed   Breakdown: decompose + reconstruct")
        print("  --internal   Compare sequential/block/randomized")
        print("  --scaling    Test O(N log N) complexity")
        print("  --mkl-config Compare with/without MKL optimization")
        print("  --malloc-free Benchmark malloc-free hot path vs Rssa")
        print("  --all        Run everything")
        print()
        print("Example: python benchmark_vs_rssa.py --all")