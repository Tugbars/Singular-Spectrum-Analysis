# Gap Filling Test - Financial Data Simulation
import numpy as np
import matplotlib.pyplot as plt
from ssa_wrapper import gapfill

# Simulate daily price data with weekend gaps
# 100 weeks = 700 days, but only 500 trading days
N_trading = 500
N_calendar = 700

# Generate "true" trading day prices (trend + cycles + noise)
np.random.seed(42)
t = np.arange(N_trading) / N_trading
prices_trading = (100 + 
                  10 * t +  # upward trend
                  5 * np.sin(2 * np.pi * 20 * t) +  # ~25-day cycle
                  3 * np.sin(2 * np.pi * 50 * t) +  # ~10-day cycle
                  2 * np.random.randn(N_trading))   # noise

# Expand to calendar days (insert NaN for weekends)
prices_calendar = np.full(N_calendar, np.nan)
trading_idx = 0
for i in range(N_calendar):
    day_of_week = i % 7
    if day_of_week < 5:  # Monday-Friday
        if trading_idx < N_trading:
            prices_calendar[i] = prices_trading[trading_idx]
            trading_idx += 1

print(f"Calendar days: {N_calendar}")
print(f"Trading days: {np.sum(~np.isnan(prices_calendar))}")
print(f"Gap days (weekends): {np.sum(np.isnan(prices_calendar))}")

# Fill gaps
L = 100
rank = 6

print("\nFilling weekend gaps...")
result = gapfill(prices_calendar, L=L, rank=rank, max_iter=20, method="iterative")
print(f"  Iterations: {result.iterations}")
print(f"  Converged: {result.converged}")
print(f"  Gaps filled: {result.n_gaps}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full view
axes[0].plot(prices_calendar, 'b.', markersize=2, label='Trading days only')
axes[0].plot(result.signal, 'r-', lw=0.5, alpha=0.7, label='Gap-filled')
axes[0].set_title('Financial Data: Weekend Gap Filling')
axes[0].legend(loc='upper left')
axes[0].set_ylabel('Price')
axes[0].set_xlabel('Calendar Day')

# Zoomed view (2 weeks)
z = slice(50, 100)
axes[1].plot(range(z.start, z.stop), prices_calendar[z], 'bo', markersize=8, label='Trading days')
axes[1].plot(range(z.start, z.stop), result.signal[z], 'r-', lw=2, label='Interpolated')

# Highlight weekends
for i in range(z.start, z.stop):
    if i % 7 >= 5:  # Weekend
        axes[1].axvspan(i - 0.5, i + 0.5, alpha=0.2, color='gray')

axes[1].set_title('Zoomed: 2 Weeks (gray = weekends)')
axes[1].legend(loc='upper left')
axes[1].set_ylabel('Price')
axes[1].set_xlabel('Calendar Day')

plt.tight_layout()
plt.show()

# Also add some random "bad tick" gaps
print("\n--- Adding random bad tick gaps ---")
prices_with_bad_ticks = result.signal.copy()
bad_tick_positions = [75, 150, 225, 310, 420, 555]
for pos in bad_tick_positions:
    prices_with_bad_ticks[pos] = np.nan

result2 = gapfill(prices_with_bad_ticks, L=L, rank=rank, max_iter=20)
print(f"Additional gaps filled: {result2.n_gaps}")

fig2, ax = plt.subplots(figsize=(14, 5))
ax.plot(result.signal, 'b-', lw=1, alpha=0.5, label='Original (weekends filled)')
ax.plot(prices_with_bad_ticks, 'r.', markersize=10, label='With bad ticks')
ax.plot(result2.signal, 'g-', lw=1, label='All gaps filled')
for pos in bad_tick_positions:
    ax.axvline(pos, color='red', linestyle=':', alpha=0.5)
ax.set_title('Bad Tick Recovery')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Day')
ax.set_ylabel('Price')
plt.tight_layout()
plt.show()
