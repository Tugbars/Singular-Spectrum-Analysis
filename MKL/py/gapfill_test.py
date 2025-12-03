# Gap Filling Test
import numpy as np
import matplotlib.pyplot as plt
from ssa_wrapper import gapfill

# Generate test signal: sum of sinusoids
N = 300
t = np.arange(N) / N
true_signal = (3.0 * np.sin(2 * np.pi * 5 * t) + 
               2.0 * np.sin(2 * np.pi * 12 * t) +
               1.0 * np.sin(2 * np.pi * 20 * t))

# Create copy with gaps
signal_with_gaps = true_signal.copy()

# Add multiple gaps
gap_regions = [
    (50, 60),    # 10-point gap
    (120, 135),  # 15-point gap
    (200, 210),  # 10-point gap
    (250, 255),  # 5-point gap
]

for start, end in gap_regions:
    signal_with_gaps[start:end] = np.nan

total_gaps = sum(end - start for start, end in gap_regions)
print(f"Created {total_gaps} gap points in {len(gap_regions)} regions")

# Parameters
L = 75  # Window length
rank = 6  # 3 sinusoids = 6 components

# Fill gaps - iterative method
print("\nIterative method:")
result_iter = gapfill(signal_with_gaps, L=L, rank=rank, max_iter=30, tol=1e-7, method="iterative")
print(f"  Iterations: {result_iter.iterations}")
print(f"  Converged: {result_iter.converged}")
print(f"  Gaps filled: {result_iter.n_gaps}")

# Fill gaps - simple method
print("\nSimple method:")
result_simple = gapfill(signal_with_gaps, L=L, rank=rank, method="simple")
print(f"  Gaps filled: {result_simple.n_gaps}")

# Compute errors only at gap positions
gap_mask = np.isnan(signal_with_gaps)
rmse_iter = np.sqrt(np.mean((result_iter.signal[gap_mask] - true_signal[gap_mask])**2))
rmse_simple = np.sqrt(np.mean((result_simple.signal[gap_mask] - true_signal[gap_mask])**2))

print(f"\nGap reconstruction RMSE:")
print(f"  Iterative: {rmse_iter:.4f}")
print(f"  Simple:    {rmse_simple:.4f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Original with gaps
axes[0].plot(true_signal, 'k-', lw=1.5, label='True signal')
axes[0].plot(signal_with_gaps, 'r.', markersize=2, label='With gaps (NaN)')
for start, end in gap_regions:
    axes[0].axvspan(start, end, alpha=0.3, color='yellow')
axes[0].set_title('Original Signal with Gaps')
axes[0].legend(loc='upper right')
axes[0].set_ylabel('Value')

# Iterative method result
axes[1].plot(true_signal, 'k-', lw=1.5, alpha=0.5, label='True')
axes[1].plot(result_iter.signal, 'b-', lw=1, label=f'Iterative (RMSE={rmse_iter:.4f})')
for start, end in gap_regions:
    axes[1].axvspan(start, end, alpha=0.3, color='yellow')
axes[1].set_title(f'Iterative Gap Fill ({result_iter.iterations} iterations)')
axes[1].legend(loc='upper right')
axes[1].set_ylabel('Value')

# Simple method result
axes[2].plot(true_signal, 'k-', lw=1.5, alpha=0.5, label='True')
axes[2].plot(result_simple.signal, 'g-', lw=1, label=f'Simple (RMSE={rmse_simple:.4f})')
for start, end in gap_regions:
    axes[2].axvspan(start, end, alpha=0.3, color='yellow')
axes[2].set_title('Simple Gap Fill (forecast/backcast blend)')
axes[2].legend(loc='upper right')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('Sample')

plt.tight_layout()
plt.show()

# Zoomed comparison at one gap
fig2, ax = plt.subplots(figsize=(12, 5))
gap_start, gap_end = gap_regions[1]  # 15-point gap
z = slice(gap_start - 20, gap_end + 20)

ax.plot(range(z.start, z.stop), true_signal[z], 'k-', lw=2.5, label='True')
ax.plot(range(z.start, z.stop), result_iter.signal[z], 'b--', lw=2, label='Iterative')
ax.plot(range(z.start, z.stop), result_simple.signal[z], 'g:', lw=2, label='Simple')
ax.axvspan(gap_start, gap_end, alpha=0.3, color='yellow', label='Gap region')
ax.legend(loc='upper right')
ax.set_title(f'Zoomed: Gap from {gap_start} to {gap_end}')
ax.set_xlabel('Sample')
ax.set_ylabel('Value')

plt.tight_layout()
plt.show()
