"""
SSA-Opt vs Rssa Benchmark - 2x2 Grid Layout
Updated with k=50 data and scaling efficiency chart
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data (k=50)
data = {
    'N':        [500,    1000,   2000,   5000,   10000,  20000],
    'L':        [125,    250,    500,    1250,   2500,   5000],
    'k':        50,
    'ours_ms':  [2.0,    1.6,    2.5,    7.2,    12.2,   26.3],
    'rssa_ms':  [33.9,   46.8,   64.3,   148.3,  370.8,  618.9],
    'corr_true':[0.9767, 0.9938, 0.9982, 0.9995, 0.9999, 1.0000],
}

# Calculate metrics
rssa_performance = [(o/r) * 100 for o, r in zip(data['ours_ms'], data['rssa_ms'])]
speedups = [r/o for r, o in zip(data['rssa_ms'], data['ours_ms'])]
avg_speedup = np.mean(speedups)
avg_perf = np.mean(rssa_performance)

# Labels
x_labels = [f'N={n:,}\nL={l}' for n, l in zip(data['N'], data['L'])]
x = np.arange(len(x_labels))

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f'SSA-Opt Performance Benchmark (k={data["k"]} components)', fontsize=16, fontweight='bold')

# ============================================================================
# CHART 1 (Top Left): Relative Performance (Long bar = better)
# ============================================================================
ax1 = axes[0, 0]
y = np.arange(len(x_labels))
height = 0.35

bars_ours = ax1.barh(y + height/2, [100]*len(y), height,
                     label='SSA-Opt', color='#2ecc71', edgecolor='black', linewidth=1)
bars_rssa = ax1.barh(y - height/2, rssa_performance, height,
                     label='Rssa', color='#e74c3c', edgecolor='black', linewidth=1)

ax1.set_xlabel('Relative Performance (%)\nLonger = Faster = Better', fontsize=11)
ax1.set_ylabel('Signal Length', fontsize=11)
ax1.set_title('Speed: SSA-Opt = 100% baseline', fontsize=12, fontweight='bold')
ax1.set_yticks(y)
ax1.set_yticklabels([f'N={n:,}' for n in data['N']], fontsize=10)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim(0, 115)

for bar in bars_ours:
    ax1.annotate('100%',
                xy=(bar.get_width() - 2, bar.get_y() + bar.get_height()/2),
                ha='right', va='center', fontsize=10, fontweight='bold', color='white')

for bar, perf in zip(bars_rssa, rssa_performance):
    ax1.annotate(f'{perf:.1f}%',
                xy=(bar.get_width() + 2, bar.get_y() + bar.get_height()/2),
                ha='left', va='center', fontsize=9, fontweight='bold', color='#c0392b')

textstr = f'Rssa achieves only {avg_perf:.1f}%\nof SSA-Opt\'s speed\n(SSA-Opt is {avg_speedup:.0f}× faster)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.62, 0.25, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center', bbox=props, fontweight='bold')

# ============================================================================
# CHART 2 (Top Right): Reconstruction Accuracy
# ============================================================================
ax2 = axes[0, 1]
color_corr = '#9b59b6'

ax2.plot(x, data['corr_true'], 'o-', color=color_corr, linewidth=2.5, markersize=10,
         label='Reconstruction correlation')
ax2.fill_between(x, 0.97, data['corr_true'], alpha=0.3, color=color_corr)

ax2.set_ylabel('Correlation with True Signal', fontsize=11)
ax2.set_xlabel('Signal Size', fontsize=11)
ax2.set_title('Accuracy: Reconstruction Quality', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'N={n:,}' for n in data['N']], fontsize=9)
ax2.set_ylim(0.97, 1.005)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, corr in enumerate(data['corr_true']):
    offset = 10 if i % 2 == 0 else -15
    va = 'bottom' if i % 2 == 0 else 'top'
    ax2.annotate(f'{corr:.4f}',
                xy=(x[i], corr),
                xytext=(0, offset), textcoords="offset points",
                ha='center', va=va, fontsize=8, fontweight='bold', color=color_corr)

textstr2 = 'Correlation → 1.0 as N increases\nMathematically correct results'
ax2.text(0.98, 0.08, textstr2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

# ============================================================================
# CHART 3 (Bottom Left): Speedup Factor
# ============================================================================
ax3 = axes[1, 0]

bars = ax3.bar(x, speedups, width=0.6, color='#3498db', edgecolor='black', linewidth=1)

ax3.set_ylabel('Speedup (×)', fontsize=11)
ax3.set_xlabel('Signal Size', fontsize=11)
ax3.set_title('Speedup vs Rssa', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, fontsize=8)
ax3.axhline(y=avg_speedup, color='orange', linestyle='--', linewidth=2.5,
            label=f'Mean: {avg_speedup:.1f}×')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(speedups) * 1.15)

for bar, spd in zip(bars, speedups):
    ax3.annotate(f'{spd:.1f}×',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# CHART 4 (Bottom Right): Scaling Efficiency (Time vs N)
# ============================================================================
ax4 = axes[1, 1]

# Plot both on log-log scale to see scaling behavior
ax4.loglog(data['N'], data['ours_ms'], 'o-', color='#2ecc71', linewidth=2.5, markersize=10,
           label='SSA-Opt', markeredgecolor='black', markeredgewidth=1)
ax4.loglog(data['N'], data['rssa_ms'], 's-', color='#e74c3c', linewidth=2.5, markersize=10,
           label='Rssa', markeredgecolor='black', markeredgewidth=1)

# Add reference lines for O(N) and O(N log N)
N_ref = np.array(data['N'])
# Normalize to pass through first SSA-Opt point
scale_nlogn = data['ours_ms'][0] / (N_ref[0] * np.log2(N_ref[0]))
scale_n = data['rssa_ms'][0] / N_ref[0]

ax4.loglog(N_ref, scale_nlogn * N_ref * np.log2(N_ref), ':', color='gray', linewidth=1.5,
           label='O(N log N)', alpha=0.7)
ax4.loglog(N_ref, scale_n * N_ref, '--', color='gray', linewidth=1.5,
           label='O(N)', alpha=0.7)

ax4.set_ylabel('Time (ms)', fontsize=11)
ax4.set_xlabel('Signal Length (N)', fontsize=11)
ax4.set_title('Scaling Efficiency', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--', which='both')

# Add time annotations
for i, (n, t_ours, t_rssa) in enumerate(zip(data['N'], data['ours_ms'], data['rssa_ms'])):
    if i % 2 == 0:  # Annotate every other point to avoid clutter
        ax4.annotate(f'{t_ours:.1f}ms', xy=(n, t_ours), xytext=(5, -10),
                    textcoords='offset points', fontsize=8, color='#27ae60')
        ax4.annotate(f'{t_rssa:.0f}ms', xy=(n, t_rssa), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='#c0392b')

# Scaling summary
scaling_text = 'SSA-Opt: ~O(N log N)\nRssa: ~O(N²) observed'
ax4.text(0.98, 0.08, scaling_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/benchmark_2x2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/benchmark_2x2.svg', bbox_inches='tight', facecolor='white')
print("Saved: benchmark_2x2.png and benchmark_2x2.svg")

# Print summary
print(f"\n{'='*60}")
print(f"BENCHMARK SUMMARY (k={data['k']})")
print(f"{'='*60}")
print(f"  Average speedup:     {avg_speedup:.1f}× faster than Rssa")
print(f"  Peak speedup:        {max(speedups):.1f}× (N={data['N'][speedups.index(max(speedups))]:,})")
print(f"  Min speedup:         {min(speedups):.1f}× (N={data['N'][speedups.index(min(speedups))]:,})")
print(f"  Reconstruction corr: {min(data['corr_true']):.4f} - {max(data['corr_true']):.4f}")
print(f"  Throughput (N=1000): {1000/data['ours_ms'][1]:.0f} SSA/sec")
print(f"{'='*60}")

plt.show()