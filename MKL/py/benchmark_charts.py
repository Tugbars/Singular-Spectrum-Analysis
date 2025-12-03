"""
SSA-Opt vs Rssa Benchmark - Individual Charts
Each chart saved separately (not combined horizontally)
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
data = {
    'N':        [500,    1000,   2000,   5000,   10000,  20000],
    'L':        [125,    250,    500,    1250,   2500,   5000],
    'ours_ms':  [1.5,    1.5,    2.5,    4.9,    8.4,    17.7],
    'rssa_ms':  [26.8,   35.6,   33.4,   117.2,  138.8,  337.9],
    'corr_true':[0.9895, 0.9973, 0.9992, 0.9996, 0.9999, 1.0000],
}

# Calculate metrics
rssa_performance = [(o/r) * 100 for o, r in zip(data['ours_ms'], data['rssa_ms'])]
speedups = [r/o for r, o in zip(data['rssa_ms'], data['ours_ms'])]

# Labels
x_labels = [f'N={n:,}\nL={l}' for n, l in zip(data['N'], data['L'])]
x = np.arange(len(x_labels))

# ============================================================================
# CHART 1: Relative Performance (Long bar = better)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

y = np.arange(len(x_labels))
height = 0.35

bars_ours = ax1.barh(y + height/2, [100]*len(y), height,
                     label='SSA-Opt', color='#2ecc71', edgecolor='black', linewidth=1)
bars_rssa = ax1.barh(y - height/2, rssa_performance, height,
                     label='Rssa', color='#e74c3c', edgecolor='black', linewidth=1)

ax1.set_xlabel('Relative Performance (%)\nLonger = Faster = Better', fontsize=13)
ax1.set_ylabel('Signal Length', fontsize=13)
ax1.set_title('SSA-Opt vs Rssa: Performance Comparison\n(SSA-Opt = 100%)', fontsize=15, fontweight='bold')
ax1.set_yticks(y)
ax1.set_yticklabels([f'N={n:,}' for n in data['N']], fontsize=11)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim(0, 115)

for bar in bars_ours:
    ax1.annotate('100%',
                xy=(bar.get_width() - 2, bar.get_y() + bar.get_height()/2),
                ha='right', va='center', fontsize=11, fontweight='bold', color='white')

for bar, perf, spd in zip(bars_rssa, rssa_performance, speedups):
    ax1.annotate(f'{perf:.1f}%',
                xy=(bar.get_width() + 2, bar.get_y() + bar.get_height()/2),
                ha='left', va='center', fontsize=10, fontweight='bold', color='#c0392b')

avg_speedup = np.mean(speedups)
avg_perf = np.mean(rssa_performance)
textstr = f'Rssa achieves only {avg_perf:.1f}%\nof SSA-Opt\'s speed\n(SSA-Opt is {avg_speedup:.0f}× faster)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.65, 0.25, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='center', horizontalalignment='center', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/chart1_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/chart1_performance.svg', bbox_inches='tight', facecolor='white')
print("Saved: chart1_performance.png")
plt.close()

# ============================================================================
# CHART 2: Speedup Factor (the middle one you liked)
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

bars = ax2.bar(x, speedups, width=0.6, color='#3498db', edgecolor='black', linewidth=1)

ax2.set_ylabel('Speedup (×)', fontsize=13)
ax2.set_xlabel('Signal Size', fontsize=13)
ax2.set_title('SSA-Opt Speedup vs Rssa (k=30 components)', fontsize=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=9)
ax2.axhline(y=np.mean(speedups), color='orange', linestyle='--', linewidth=2.5, 
            label=f'Mean: {np.mean(speedups):.1f}×')
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(speedups) * 1.15)

for bar, spd in zip(bars, speedups):
    ax2.annotate(f'{spd:.1f}×',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/chart2_speedup.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/chart2_speedup.svg', bbox_inches='tight', facecolor='white')
print("Saved: chart2_speedup.png")
plt.close()

# ============================================================================
# CHART 3: Reconstruction Accuracy
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

color_corr = '#9b59b6'
ax3.plot(x, data['corr_true'], 'o-', color=color_corr, linewidth=2.5, markersize=10, 
         label='Reconstruction correlation')
ax3.fill_between(x, 0.98, data['corr_true'], alpha=0.3, color=color_corr)

ax3.set_ylabel('Correlation with True Signal', fontsize=13)
ax3.set_xlabel('Signal Size', fontsize=13)
ax3.set_title('SSA-Opt Reconstruction Accuracy (k=30 components)', fontsize=15, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, fontsize=9)
ax3.set_ylim(0.985, 1.003)
ax3.legend(loc='lower right', fontsize=12)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for i, corr in enumerate(data['corr_true']):
    offset = 12 if i % 2 == 0 else -18
    va = 'bottom' if i % 2 == 0 else 'top'
    ax3.annotate(f'{corr:.4f}',
                xy=(x[i], corr),
                xytext=(0, offset), textcoords="offset points",
                ha='center', va=va, fontsize=9, fontweight='bold', color=color_corr)

textstr = 'Correlation → 1.0 as N increases\nMathematically correct results'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax3.text(0.98, 0.15, textstr, transform=ax3.transAxes, fontsize=11,
         verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/chart3_accuracy.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/chart3_accuracy.svg', bbox_inches='tight', facecolor='white')
print("Saved: chart3_accuracy.png")
plt.close()

# ============================================================================
# CHART 4: Execution Time Comparison (log scale)
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

width = 0.35
bars_ours = ax4.bar(x - width/2, data['ours_ms'], width, label='SSA-Opt', 
                    color='#2ecc71', edgecolor='black', linewidth=0.5)
bars_rssa = ax4.bar(x + width/2, data['rssa_ms'], width, label='Rssa', 
                    color='#e74c3c', edgecolor='black', linewidth=0.5)

ax4.set_ylabel('Time (ms)', fontsize=13)
ax4.set_xlabel('Signal Size', fontsize=13)
ax4.set_title('Execution Time Comparison (k=30 components)', fontsize=15, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(x_labels, fontsize=9)
ax4.legend(loc='upper left', fontsize=12)
ax4.set_yscale('log')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars_ours:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, color='#27ae60')

for bar in bars_rssa:
    height = bar.get_height()
    ax4.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, color='#c0392b')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/chart4_time.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/chart4_time.svg', bbox_inches='tight', facecolor='white')
print("Saved: chart4_time.png")
plt.close()

print("\n=== All 4 charts saved separately ===")
print("  chart1_performance.png - Relative performance (long bar = better)")
print("  chart2_speedup.png     - Speedup factor (the one you liked)")
print("  chart3_accuracy.png    - Reconstruction correlation")
print("  chart4_time.png        - Execution time (log scale)")
