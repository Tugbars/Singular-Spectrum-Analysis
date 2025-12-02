"""
SSA Parameter Sensitivity Analysis
==================================

Tests how SSA performance varies with:
  - Signal-to-Noise Ratio (SNR)
  - Window Length (L)
  - Component Count (k)

This helps users understand how to tune SSA for their data.

Usage:
    python ssa_parameter_tests.py --snr        # SNR sweep
    python ssa_parameter_tests.py --window     # L sensitivity
    python ssa_parameter_tests.py --components # k sweep
    python ssa_parameter_tests.py --all        # Run everything
"""

import numpy as np
import time
import argparse

# ============================================================================
# Test 1: Signal-to-Noise Ratio Sweep
# ============================================================================

def test_snr_sweep():
    """
    How does SSA perform at different noise levels?
    
    At high SNR (low noise), SSA should achieve near-perfect reconstruction.
    At low SNR (high noise), performance degrades gracefully.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("TEST: Signal-to-Noise Ratio Sweep")
    print("="*70)
    print("\nSignal: sin(2πt) with varying noise levels")
    print("N=5000, L=1250, k=10\n")
    
    N = 5000
    L = N // 4
    k = 10
    
    # Generate true signal
    np.random.seed(42)
    t = np.linspace(0, 50*np.pi, N)
    x_true = np.sin(t)
    noise = np.random.randn(N)
    
    print(f"{'Noise Std':>10} {'SNR (dB)':>10} | {'Time (ms)':>10} {'Corr(raw)':>10} {'Corr(true)':>11}")
    print("-"*65)
    
    # Warmup
    SSA(np.random.randn(1000), L=250).decompose(k=5)
    
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    for noise_std in noise_levels:
        x = x_true + noise_std * noise
        
        # SNR in dB: 10 * log10(signal_power / noise_power)
        signal_power = np.var(x_true)
        noise_power = noise_std**2
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Time decomposition + reconstruction
        ssa = SSA(x, L=L)
        t0 = time.perf_counter()
        ssa.decompose(k=k)
        recon = ssa.reconstruct(list(range(k)))
        time_ms = (time.perf_counter() - t0) * 1000
        
        corr_raw = np.corrcoef(x, recon)[0, 1]
        corr_true = np.corrcoef(x_true, recon)[0, 1]
        
        print(f"{noise_std:>10.2f} {snr_db:>10.1f} | {time_ms:>10.1f} {corr_raw:>10.4f} {corr_true:>11.4f}")
    
    print("-"*65)
    print("\nInterpretation:")
    print("  - Corr(true) shows how well SSA recovers the underlying signal")
    print("  - At SNR < 0 dB (noise > signal), performance degrades")
    print("  - SSA is robust up to ~0 dB SNR with k=10 components")


# ============================================================================
# Test 2: Window Length (L) Sensitivity
# ============================================================================

def test_window_sensitivity():
    """
    How does window length L affect accuracy and speed?
    
    L controls the trade-off between:
    - Frequency resolution (larger L = better separation)
    - Statistical stability (smaller L = more trajectory vectors)
    
    Rule of thumb: L = N/3 to N/2 is usually good.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("TEST: Window Length (L) Sensitivity")
    print("="*70)
    print("\nSignal: sin(2πt/10) + 0.5*sin(2πt/25) + noise(0.3)")
    print("N=5000, k=10\n")
    
    N = 5000
    k = 10
    
    # Multi-periodic signal
    np.random.seed(42)
    t = np.linspace(0, 100*np.pi, N)
    x_true = np.sin(2*np.pi*t/10) + 0.5*np.sin(2*np.pi*t/25)
    x = x_true + 0.3*np.random.randn(N)
    
    print(f"{'L':>6} {'L/N':>6} {'K':>6} | {'Time (ms)':>10} {'Corr(true)':>11} {'Note':>20}")
    print("-"*75)
    
    # Warmup
    SSA(np.random.randn(1000), L=250).decompose(k=5)
    
    l_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.7]
    
    best_corr = 0
    best_L = 0
    
    for l_ratio in l_ratios:
        L = int(N * l_ratio)
        K = N - L + 1
        
        ssa = SSA(x, L=L)
        t0 = time.perf_counter()
        ssa.decompose(k=k)
        recon = ssa.reconstruct(list(range(k)))
        time_ms = (time.perf_counter() - t0) * 1000
        
        corr_true = np.corrcoef(x_true, recon)[0, 1]
        
        # Note for user guidance
        if l_ratio < 0.1:
            note = "Too small"
        elif l_ratio < 0.2:
            note = "Low resolution"
        elif l_ratio <= 0.5:
            note = "Recommended"
        else:
            note = "May overfit"
        
        if corr_true > best_corr:
            best_corr = corr_true
            best_L = L
        
        print(f"{L:>6} {l_ratio:>6.2f} {K:>6} | {time_ms:>10.1f} {corr_true:>11.4f} {note:>20}")
    
    print("-"*75)
    print(f"\nBest: L={best_L} (L/N={best_L/N:.2f}) with Corr(true)={best_corr:.4f}")
    print("\nGuidelines:")
    print("  - L/N = 0.2 to 0.5 is generally optimal")
    print("  - Larger L = better frequency separation, slower")
    print("  - Smaller L = more averaging, may blur components")


# ============================================================================
# Test 3: Component Count (k) Sweep
# ============================================================================

def test_component_sweep():
    """
    How many components are needed?
    
    Shows diminishing returns - after capturing signal, more k just adds noise.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("TEST: Component Count (k) Sweep")
    print("="*70)
    print("\nSignal: trend + sin(2πt/50) + noise(0.3)")
    print("N=5000, L=1250\n")
    
    N = 5000
    L = N // 4
    
    # Trend + seasonal + noise
    np.random.seed(42)
    t = np.linspace(0, 100*np.pi, N)
    trend = 0.01 * np.arange(N)
    seasonal = np.sin(2*np.pi*t/50)
    x_true = trend + seasonal
    x = x_true + 0.3*np.random.randn(N)
    
    print(f"{'k':>5} | {'Time (ms)':>10} {'Corr(true)':>11} {'Var Expl':>10} {'Δ Corr':>10}")
    print("-"*55)
    
    # Warmup
    SSA(np.random.randn(1000), L=250).decompose(k=5)
    
    k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    
    prev_corr = 0
    
    for k in k_values:
        ssa = SSA(x, L=L)
        t0 = time.perf_counter()
        ssa.decompose(k=k)
        recon = ssa.reconstruct(list(range(k)))
        time_ms = (time.perf_counter() - t0) * 1000
        
        corr_true = np.corrcoef(x_true, recon)[0, 1]
        var_explained = ssa.variance_explained(0, k-1)
        delta_corr = corr_true - prev_corr
        
        delta_str = f"+{delta_corr:.4f}" if delta_corr > 0 else f"{delta_corr:.4f}"
        
        print(f"{k:>5} | {time_ms:>10.1f} {corr_true:>11.4f} {var_explained:>9.2%} {delta_str:>10}")
        
        prev_corr = corr_true
    
    print("-"*55)
    print("\nInterpretation:")
    print("  - Signal has ~3 components (trend + sin/cos pair)")
    print("  - Corr(true) plateaus after k=3-5")
    print("  - More components add noise, not signal")
    print("  - Time scales linearly with k (randomized SVD)")


# ============================================================================
# Test 4: Combined Analysis - Optimal Parameters
# ============================================================================

def test_optimal_parameters():
    """
    Grid search to find optimal (L, k) for a given signal.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("TEST: Optimal Parameter Search (L × k grid)")
    print("="*70)
    print("\nSignal: sin(2πt/20) + 0.5*sin(2πt/50) + noise(0.5)")
    print("N=2000\n")
    
    N = 2000
    
    np.random.seed(42)
    t = np.linspace(0, 100*np.pi, N)
    x_true = np.sin(2*np.pi*t/20) + 0.5*np.sin(2*np.pi*t/50)
    x = x_true + 0.5*np.random.randn(N)
    
    # Warmup
    SSA(np.random.randn(1000), L=250).decompose(k=5)
    
    l_ratios = [0.1, 0.2, 0.25, 0.33, 0.5]
    k_values = [3, 5, 10, 20, 30]
    
    print(f"{'':>8}", end="")
    for k in k_values:
        print(f"{'k='+str(k):>10}", end="")
    print()
    print("-"*60)
    
    best_corr = 0
    best_params = (0, 0)
    
    for l_ratio in l_ratios:
        L = int(N * l_ratio)
        print(f"L={L:>4} |", end="")
        
        for k in k_values:
            ssa = SSA(x, L=L)
            ssa.decompose(k=k)
            recon = ssa.reconstruct(list(range(k)))
            corr = np.corrcoef(x_true, recon)[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_params = (L, k)
            
            print(f"{corr:>10.4f}", end="")
        print()
    
    print("-"*60)
    print(f"\nBest: L={best_params[0]}, k={best_params[1]} → Corr={best_corr:.4f}")


# ============================================================================
# Test 5: Periodic Component Detection
# ============================================================================

def test_periodic_detection():
    """
    Test automatic detection of periodic pairs.
    """
    from ssa_wrapper import SSA
    
    print("\n" + "="*70)
    print("TEST: Periodic Component Detection")
    print("="*70)
    print("\nSignal: sin(2πt/20) + 0.7*sin(2πt/50) + 0.4*sin(2πt/100) + noise")
    print("N=5000, L=1250, k=20\n")
    
    N = 5000
    L = N // 4
    k = 20
    
    np.random.seed(42)
    t = np.linspace(0, 200*np.pi, N)
    x_true = np.sin(2*np.pi*t/20) + 0.7*np.sin(2*np.pi*t/50) + 0.4*np.sin(2*np.pi*t/100)
    x = x_true + 0.3*np.random.randn(N)
    
    # Warmup
    SSA(np.random.randn(1000), L=250).decompose(k=5)
    
    ssa = SSA(x, L=L)
    ssa.decompose(k=k)
    
    # Find periodic pairs
    pairs = ssa.find_periodic_pairs(max_pairs=10, sv_tol=0.15, wcorr_thresh=0.5)
    
    print(f"Detected {len(pairs)} periodic pairs:")
    print(f"{'Pair':>6} {'Components':>15} {'Variance':>12}")
    print("-"*40)
    
    for i, (c1, c2) in enumerate(pairs):
        var = ssa.variance_explained(c1, c2)
        print(f"{i+1:>6} {f'({c1}, {c2})':>15} {var:>11.2%}")
    
    print("-"*40)
    
    # Reconstruct using detected pairs
    if pairs:
        pair_components = [c for pair in pairs for c in pair]
        recon = ssa.reconstruct(pair_components)
        corr = np.corrcoef(x_true, recon)[0, 1]
        print(f"\nReconstruction using detected pairs: Corr(true) = {corr:.4f}")
    
    print("\nExpected: 3 pairs for the 3 sinusoidal components")


# ============================================================================
# Main
# ============================================================================

def run_all():
    test_snr_sweep()
    test_window_sensitivity()
    test_component_sweep()
    test_optimal_parameters()
    test_periodic_detection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSA Parameter Sensitivity Tests')
    parser.add_argument('--snr', action='store_true', help='Signal-to-Noise Ratio sweep')
    parser.add_argument('--window', action='store_true', help='Window length (L) sensitivity')
    parser.add_argument('--components', action='store_true', help='Component count (k) sweep')
    parser.add_argument('--optimal', action='store_true', help='Optimal parameter grid search')
    parser.add_argument('--periodic', action='store_true', help='Periodic detection test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    args = parser.parse_args()
    
    if args.snr:
        test_snr_sweep()
    elif args.window:
        test_window_sensitivity()
    elif args.components:
        test_component_sweep()
    elif args.optimal:
        test_optimal_parameters()
    elif args.periodic:
        test_periodic_detection()
    elif args.all:
        run_all()
    else:
        print("SSA Parameter Sensitivity Tests")
        print("="*40)
        print("Options:")
        print("  --snr        Signal-to-Noise Ratio sweep")
        print("  --window     Window length (L) sensitivity")
        print("  --components Component count (k) sweep")
        print("  --optimal    Optimal parameter grid search")
        print("  --periodic   Periodic detection test")
        print("  --all        Run everything")
        print()
        print("Example: python ssa_parameter_tests.py --all")
