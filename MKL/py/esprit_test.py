# ESPRIT Frequency Estimation Test
import numpy as np
import matplotlib.pyplot as plt
from ssa_wrapper import SSA

# Generate signal with known periods
N = 500
t = np.arange(N)

# Known periods: 50, 20, 10 samples (frequencies: 0.02, 0.05, 0.1)
true_periods = [50, 20, 10]
signal = (3.0 * np.sin(2 * np.pi * t / 50) +   # Period 50
          2.0 * np.sin(2 * np.pi * t / 20) +   # Period 20
          1.0 * np.sin(2 * np.pi * t / 10))    # Period 10

# Add some noise
np.random.seed(42)
noisy = signal + 0.5 * np.random.randn(N)

# SSA decomposition
L = 100  # Should capture at least 2 cycles of longest period
k = 10   # Enough to capture 3 periodic pairs + some noise
ssa = SSA(noisy, L=L)
ssa.decompose(k=k)

# ESPRIT frequency estimation
par = ssa.parestimate()
print(par.summary())

# Find periodic components
periodic_idx = par.get_periodic_components(min_period=5, min_modulus=0.8)
print(f"\nPeriodic component indices: {periodic_idx}")
print(f"Detected periods: {np.sort(par.periods[periodic_idx])}")
print(f"True periods: {true_periods}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Signal
axes[0, 0].plot(noisy, 'gray', alpha=0.5, lw=0.5, label='Noisy')
axes[0, 0].plot(signal, 'k-', lw=1.5, label='True')
recon = ssa.reconstruct(list(range(6)))  # First 6 components (3 pairs)
axes[0, 0].plot(recon, 'r-', lw=1, label='SSA Reconstruction')
axes[0, 0].set_title('Signal')
axes[0, 0].legend()

# Eigenvalue spectrum
axes[0, 1].bar(range(k), ssa.eigenvalues[:k])
axes[0, 1].set_xlabel('Component')
axes[0, 1].set_ylabel('Eigenvalue')
axes[0, 1].set_title('Eigenvalue Spectrum')

# Detected periods
valid_periods = par.periods[~np.isinf(par.periods)]
valid_moduli = par.moduli[~np.isinf(par.periods)]
axes[1, 0].scatter(valid_periods, valid_moduli, s=100, c='blue', edgecolors='black')
for p in true_periods:
    axes[1, 0].axvline(x=p, color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('Period (samples)')
axes[1, 0].set_ylabel('Modulus (damping)')
axes[1, 0].set_title('ESPRIT: Detected Periods vs Modulus')
axes[1, 0].axhline(y=1.0, color='green', linestyle=':', label='Undamped')
axes[1, 0].legend()

# Period detection accuracy
axes[1, 1].stem(range(len(valid_periods)), np.sort(valid_periods)[::-1])
for i, p in enumerate(true_periods):
    axes[1, 1].axhline(y=p, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Component')
axes[1, 1].set_ylabel('Period (samples)')
axes[1, 1].set_title('Sorted Detected Periods (red = true)')

plt.tight_layout()
plt.show()
