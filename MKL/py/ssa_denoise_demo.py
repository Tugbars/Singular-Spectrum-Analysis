"""
SSA Denoising Showcase
======================

Visual demonstration of SSA's denoising capabilities.
Compares single-pass SSA vs Cadzow iterations vs other methods.

Run: python ssa_denoise_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Try to import our SSA library
try:
    from ssa_wrapper import SSA, cadzow
    HAS_SSA = True
except ImportError:
    HAS_SSA = False
    print("Warning: ssa_wrapper not available. Using numpy fallback for comparison methods only.")

# ============================================================================
# Signal Generators
# ============================================================================

def generate_sinusoidal(N: int, freqs: list, amps: list) -> np.ndarray:
    """Sum of sinusoids - exactly finite rank."""
    t = np.arange(N) / N
    signal = np.zeros(N)
    for f, a in zip(freqs, amps):
        signal += a * np.sin(2 * np.pi * f * t)
    return signal

def generate_chirp(N: int, f0: float = 1, f1: float = 20) -> np.ndarray:
    """Chirp signal - frequency sweep."""
    t = np.linspace(0, 1, N)
    return np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)

def generate_trend_seasonal(N: int) -> np.ndarray:
    """Trend + seasonal pattern (like financial/economic data)."""
    t = np.arange(N) / N
    trend = 2 * t + 0.5 * t**2
    seasonal = 0.5 * np.sin(2 * np.pi * 12 * t) + 0.3 * np.sin(2 * np.pi * 52 * t)
    return trend + seasonal

def generate_ecg_like(N: int) -> np.ndarray:
    """ECG-like periodic spikes."""
    t = np.linspace(0, 4 * np.pi, N)
    # Approximate QRS complex shape
    signal = np.zeros(N)
    period = N // 8
    for i in range(8):
        center = i * period + period // 2
        if center < N:
            width = period // 10
            signal[max(0, center-width):min(N, center+width)] += \
                np.exp(-0.5 * ((np.arange(min(N, center+width) - max(0, center-width)) - width) / (width/3))**2)
    return signal

def add_noise(signal: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    """Add Gaussian noise to achieve target SNR."""
    sig_power = np.mean(signal**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.randn(len(signal)) * noise_std
    return signal + noise, noise_std

# ============================================================================
# Comparison Methods
# ============================================================================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')

def savgol_filter(x: np.ndarray, window: int, order: int = 3) -> np.ndarray:
    """Savitzky-Golay filter."""
    try:
        from scipy.signal import savgol_filter as sg
        return sg(x, window, order)
    except ImportError:
        # Fallback to moving average
        return moving_average(x, window)

def lowpass_filter(x: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    """FFT lowpass filter."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    X[freqs > cutoff] = 0
    return np.fft.irfft(X, len(x))

# ============================================================================
# Metrics
# ============================================================================

def compute_snr(true: np.ndarray, estimate: np.ndarray) -> float:
    """Compute SNR in dB."""
    signal_power = np.mean(true**2)
    noise_power = np.mean((true - estimate)**2)
    if noise_power < 1e-15:
        return 100.0  # Essentially perfect
    return 10 * np.log10(signal_power / noise_power)

def compute_rmse(true: np.ndarray, estimate: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(np.mean((true - estimate)**2))

def compute_correlation(true: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return np.corrcoef(true, estimate)[0, 1]

# ============================================================================
# Demo Functions
# ============================================================================

def demo_single_signal(signal_name: str, true_signal: np.ndarray, 
                       snr_db: float = 10, L: int = None, rank: int = None):
    """Demonstrate denoising on a single signal."""
    N = len(true_signal)
    if L is None:
        L = N // 4
    if rank is None:
        rank = 6
    
    # Add noise
    noisy, noise_std = add_noise(true_signal, snr_db)
    
    print(f"\n{'='*60}")
    print(f"Signal: {signal_name}")
    print(f"N={N}, L={L}, rank={rank}, input SNR={snr_db} dB")
    print(f"{'='*60}")
    
    results = {}
    
    # Input (noisy)
    results['Noisy'] = {
        'signal': noisy,
        'snr': snr_db,
        'rmse': compute_rmse(true_signal, noisy),
        'corr': compute_correlation(true_signal, noisy)
    }
    
    # Moving average
    ma_window = max(5, N // 50)
    ma_result = moving_average(noisy, ma_window)
    results['Moving Avg'] = {
        'signal': ma_result,
        'snr': compute_snr(true_signal, ma_result),
        'rmse': compute_rmse(true_signal, ma_result),
        'corr': compute_correlation(true_signal, ma_result)
    }
    
    # Savitzky-Golay
    sg_window = ma_window if ma_window % 2 == 1 else ma_window + 1
    sg_result = savgol_filter(noisy, sg_window, 3)
    results['Savitzky-Golay'] = {
        'signal': sg_result,
        'snr': compute_snr(true_signal, sg_result),
        'rmse': compute_rmse(true_signal, sg_result),
        'corr': compute_correlation(true_signal, sg_result)
    }
    
    # FFT lowpass
    lp_result = lowpass_filter(noisy, 0.05)
    results['FFT Lowpass'] = {
        'signal': lp_result,
        'snr': compute_snr(true_signal, lp_result),
        'rmse': compute_rmse(true_signal, lp_result),
        'corr': compute_correlation(true_signal, lp_result)
    }
    
    if HAS_SSA:
        # Single-pass SSA
        ssa = SSA(noisy, L=L)
        ssa.decompose(k=rank)
        ssa_result = ssa.reconstruct(list(range(rank)))
        results['SSA (single-pass)'] = {
            'signal': ssa_result,
            'snr': compute_snr(true_signal, ssa_result),
            'rmse': compute_rmse(true_signal, ssa_result),
            'corr': compute_correlation(true_signal, ssa_result)
        }
        
        # Cadzow iterations
        cadzow_res = cadzow(noisy, L=L, rank=rank, max_iter=20, tol=1e-9)
        results['SSA + Cadzow'] = {
            'signal': cadzow_res.signal,
            'snr': compute_snr(true_signal, cadzow_res.signal),
            'rmse': compute_rmse(true_signal, cadzow_res.signal),
            'corr': compute_correlation(true_signal, cadzow_res.signal),
            'iterations': cadzow_res.iterations
        }
    
    # Print results table
    print(f"\n{'Method':<20} {'SNR (dB)':<12} {'RMSE':<12} {'Correlation':<12}")
    print("-" * 56)
    for name, res in results.items():
        extra = f" ({res.get('iterations', '')} iter)" if 'iterations' in res else ""
        print(f"{name:<20} {res['snr']:>8.2f}    {res['rmse']:>8.5f}    {res['corr']:>8.5f}{extra}")
    
    return true_signal, noisy, results

def plot_comparison(true_signal, noisy, results, title="SSA Denoising Comparison"):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    N = len(true_signal)
    t = np.arange(N)
    
    # Top left: Full signal comparison
    ax = axes[0, 0]
    ax.plot(t, noisy, 'gray', alpha=0.5, linewidth=0.5, label='Noisy')
    ax.plot(t, true_signal, 'k-', linewidth=2, label='True')
    if 'SSA + Cadzow' in results:
        ax.plot(t, results['SSA + Cadzow']['signal'], 'r-', linewidth=1.5, label='SSA + Cadzow')
    elif 'SSA (single-pass)' in results:
        ax.plot(t, results['SSA (single-pass)']['signal'], 'r-', linewidth=1.5, label='SSA')
    ax.set_title('Full Signal')
    ax.legend(loc='upper right')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    
    # Top right: Zoomed comparison
    ax = axes[0, 1]
    zoom_start, zoom_end = N // 4, N // 4 + N // 5
    ax.plot(t[zoom_start:zoom_end], noisy[zoom_start:zoom_end], 'gray', alpha=0.5, linewidth=1, label='Noisy')
    ax.plot(t[zoom_start:zoom_end], true_signal[zoom_start:zoom_end], 'k-', linewidth=2, label='True')
    for name, res in results.items():
        if name in ['SSA (single-pass)', 'SSA + Cadzow']:
            ax.plot(t[zoom_start:zoom_end], res['signal'][zoom_start:zoom_end], 
                   linewidth=1.5, label=name)
    ax.set_title('Zoomed View')
    ax.legend(loc='upper right')
    ax.set_xlabel('Sample')
    
    # Bottom left: All methods comparison (zoomed)
    ax = axes[1, 0]
    ax.plot(t[zoom_start:zoom_end], true_signal[zoom_start:zoom_end], 'k-', linewidth=2.5, label='True')
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, res), color in zip(results.items(), colors):
        if name != 'Noisy':
            ax.plot(t[zoom_start:zoom_end], res['signal'][zoom_start:zoom_end], 
                   color=color, linewidth=1, alpha=0.8, label=f"{name} (SNR={res['snr']:.1f})")
    ax.set_title('All Methods Comparison (Zoomed)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Sample')
    
    # Bottom right: SNR bar chart
    ax = axes[1, 1]
    methods = [k for k in results.keys() if k != 'Noisy']
    snrs = [results[k]['snr'] for k in methods]
    colors = ['green' if 'SSA' in m else 'steelblue' for m in methods]
    bars = ax.bar(methods, snrs, color=colors, edgecolor='black')
    ax.axhline(y=results['Noisy']['snr'], color='red', linestyle='--', label=f"Input SNR={results['Noisy']['snr']:.1f}")
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Denoising Performance')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    
    # Highlight best
    best_idx = np.argmax(snrs)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    return fig

def run_full_demo():
    """Run complete denoising showcase."""
    print("=" * 70)
    print("SSA DENOISING SHOWCASE")
    print("=" * 70)
    
    if not HAS_SSA:
        print("\nERROR: ssa_wrapper not available!")
        print("Build the library first with:")
        print("  gcc -shared -fPIC -O3 -o libssa.so ssa_impl.c ...")
        return
    
    np.random.seed(42)
    
    # Test cases
    test_cases = [
        ("Sinusoidal (rank 4)", generate_sinusoidal(500, [5, 12], [3, 2]), 10, 125, 4),
        ("Chirp Signal", generate_chirp(500), 8, 100, 10),
        ("Trend + Seasonal", generate_trend_seasonal(500), 10, 125, 8),
        ("ECG-like Spikes", generate_ecg_like(500), 5, 100, 12),
    ]
    
    all_results = []
    
    for name, true_signal, snr, L, rank in test_cases:
        true, noisy, results = demo_single_signal(name, true_signal, snr, L, rank)
        all_results.append((name, true, noisy, results))
        
        # Plot
        fig = plot_comparison(true, noisy, results, f"SSA Denoising: {name}")
        fig.savefig(f"denoise_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: SNR Improvement (dB)")
    print("=" * 70)
    print(f"{'Signal':<25} {'Input':<8} {'MA':<8} {'S-G':<8} {'FFT':<8} {'SSA':<8} {'Cadzow':<8}")
    print("-" * 70)
    
    for name, _, _, results in all_results:
        row = f"{name:<25} "
        row += f"{results['Noisy']['snr']:<8.1f} "
        row += f"{results['Moving Avg']['snr']:<8.1f} "
        row += f"{results['Savitzky-Golay']['snr']:<8.1f} "
        row += f"{results['FFT Lowpass']['snr']:<8.1f} "
        if 'SSA (single-pass)' in results:
            row += f"{results['SSA (single-pass)']['snr']:<8.1f} "
            row += f"{results['SSA + Cadzow']['snr']:<8.1f}"
        print(row)
    
    print("\nPlots saved to denoise_*.png")
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("• SSA adapts to signal structure (periodic, trend, etc.)")
    print("• Cadzow iterations provide extra refinement for finite-rank signals")
    print("• Unlike FFT lowpass, SSA preserves sharp features and phase")
    print("• SSA works well even without knowing signal frequencies in advance")

if __name__ == "__main__":
    run_full_demo()
