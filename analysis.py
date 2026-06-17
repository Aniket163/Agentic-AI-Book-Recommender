import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

fg = pd.read_csv("/home/claude/assignment/fear_greed_index.csv", parse_dates=["Date"])
tr = pd.read_csv("/home/claude/assignment/historical_trader_data.csv", parse_dates=["time"])

tr["date"] = tr["time"].dt.date.astype(str)
fg["date"] = fg["Date"].dt.strftime("%Y-%m-%d")

merged = tr.merge(fg[["date","Classification","Value"]], on="date", how="left")

# ===================== FIGURE 1: Sentiment Distribution + PnL by Sentiment =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
for ax in axes: ax.set_facecolor("#161b22"); ax.tick_params(colors="white"); [sp.set_color("#30363d") for sp in ax.spines.values()]

sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
colors_map = {"Extreme Fear":"#ff4444","Fear":"#ff8c42","Neutral":"#f0e68c","Greed":"#7fff7f","Extreme Greed":"#00e676"}

# Trade count by sentiment
counts = merged["Classification"].value_counts().reindex(sentiment_order).fillna(0)
bars = axes[0].bar(sentiment_order, counts, color=[colors_map[s] for s in sentiment_order], edgecolor="#30363d", linewidth=0.8)
axes[0].set_title("Trade Volume by Market Sentiment", color="white", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Number of Trades", color="white"); axes[0].tick_params(axis="x", rotation=20)
for bar, val in zip(bars, counts): axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(int(val)), ha="center", color="white", fontsize=9)

# Avg PnL by sentiment
avg_pnl = merged.groupby("Classification")["closedPnL"].mean().reindex(sentiment_order)
bar_colors = ["#ff4444" if v < 0 else "#00e676" for v in avg_pnl]
bars2 = axes[1].bar(sentiment_order, avg_pnl, color=bar_colors, edgecolor="#30363d", linewidth=0.8)
axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("Avg Closed PnL by Sentiment", color="white", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Avg PnL (USD)", color="white"); axes[1].tick_params(axis="x", rotation=20)
for bar, val in zip(bars2, avg_pnl): axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5 if val>=0 else bar.get_height()-15, f"${val:.1f}", ha="center", color="white", fontsize=9)

plt.tight_layout(pad=2)
plt.savefig("/home/claude/assignment/fig1_sentiment_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Fig 1 saved")

# ===================== FIGURE 2: Top Traders + Leverage Analysis =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
for ax in axes: ax.set_facecolor("#161b22"); ax.tick_params(colors="white"); [sp.set_color("#30363d") for sp in ax.spines.values()]

top_traders = merged.groupby("account")["closedPnL"].sum().nlargest(10)
short_labels = [f"Trader {i+1}" for i in range(len(top_traders))]
bar_colors3 = ["#00e676" if v >= 0 else "#ff4444" for v in top_traders.values]
axes[0].barh(short_labels[::-1], top_traders.values[::-1], color=bar_colors3[::-1], edgecolor="#30363d")
axes[0].set_title("Top 10 Traders by Total PnL", color="white", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Total Closed PnL (USD)", color="white"); axes[0].axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

# Leverage vs PnL
lev_pnl = merged.groupby("leverage")["closedPnL"].agg(["mean","std","count"]).reset_index()
sc = axes[1].scatter(lev_pnl["leverage"], lev_pnl["mean"], s=lev_pnl["count"]/5, c=lev_pnl["mean"], cmap="RdYlGn", vmin=-200, vmax=200, edgecolors="#30363d", linewidth=0.5, alpha=0.9)
axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("Avg PnL vs Leverage (bubble = trade count)", color="white", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Leverage", color="white"); axes[1].set_ylabel("Avg PnL (USD)", color="white")
plt.colorbar(sc, ax=axes[1]).ax.yaxis.set_tick_params(color="white")

plt.tight_layout(pad=2)
plt.savefig("/home/claude/assignment/fig2_trader_leverage.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Fig 2 saved")

# ===================== FIGURE 3: Time Series + Symbol Analysis =====================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
for ax in axes: ax.set_facecolor("#161b22"); ax.tick_params(colors="white"); [sp.set_color("#30363d") for sp in ax.spines.values()]

# Monthly PnL trend
merged["month"] = pd.to_datetime(merged["date"]).dt.to_period("M")
monthly = merged.groupby("month")["closedPnL"].sum()
months_str = [str(m) for m in monthly.index]
bar_colors4 = ["#00e676" if v >= 0 else "#ff4444" for v in monthly.values]
axes[0].bar(months_str, monthly.values, color=bar_colors4, edgecolor="#30363d", linewidth=0.5)
axes[0].set_title("Monthly Total PnL Across All Traders", color="white", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Total PnL (USD)", color="white"); axes[0].tick_params(axis="x", rotation=45)
axes[0].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

# Symbol performance
sym_stats = merged.groupby("symbol")["closedPnL"].agg(["sum","mean","count"]).reset_index()
x = np.arange(len(sym_stats))
w = 0.35
axes[1].bar(x - w/2, sym_stats["sum"]/1000, w, label="Total PnL (K)", color="#58a6ff", edgecolor="#30363d")
axes[1].bar(x + w/2, sym_stats["mean"], w, label="Avg PnL", color="#f0883e", edgecolor="#30363d")
axes[1].set_xticks(x); axes[1].set_xticklabels(sym_stats["symbol"], color="white")
axes[1].set_title("PnL Performance by Symbol", color="white", fontsize=14, fontweight="bold")
axes[1].set_ylabel("PnL (USD)", color="white")
axes[1].legend(facecolor="#161b22", labelcolor="white")
axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

plt.tight_layout(pad=2)
plt.savefig("/home/claude/assignment/fig3_timeseries_symbols.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Fig 3 saved")

# ===================== FIGURE 4: Heatmap + Win Rate =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
for ax in axes: ax.set_facecolor("#161b22"); ax.tick_params(colors="white"); [sp.set_color("#30363d") for sp in ax.spines.values()]

# Win rate by sentiment & symbol
win_rate_pivot = merged.copy()
win_rate_pivot["win"] = (win_rate_pivot["closedPnL"] > 0).astype(int)
pivot = win_rate_pivot.groupby(["Classification","symbol"])["win"].mean().unstack().reindex(sentiment_order).fillna(0)
cmap = LinearSegmentedColormap.from_list("rg", ["#ff4444","#f0e68c","#00e676"])
im = axes[0].imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
axes[0].set_xticks(range(len(pivot.columns))); axes[0].set_xticklabels(pivot.columns, color="white")
axes[0].set_yticks(range(len(pivot.index))); axes[0].set_yticklabels(pivot.index, color="white")
axes[0].set_title("Win Rate: Sentiment × Symbol", color="white", fontsize=14, fontweight="bold")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        axes[0].text(j, i, f"{pivot.values[i,j]:.0%}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
plt.colorbar(im, ax=axes[0]).ax.yaxis.set_tick_params(color="white")

# BUY vs SELL PnL by sentiment
side_pnl = merged.groupby(["Classification","side"])["closedPnL"].mean().unstack().reindex(sentiment_order)
x2 = np.arange(len(sentiment_order))
if "BUY" in side_pnl.columns: axes[1].bar(x2 - 0.2, side_pnl["BUY"], 0.4, label="BUY", color="#00e676", edgecolor="#30363d")
if "SELL" in side_pnl.columns: axes[1].bar(x2 + 0.2, side_pnl["SELL"], 0.4, label="SELL", color="#ff4444", edgecolor="#30363d")
axes[1].set_xticks(x2); axes[1].set_xticklabels(sentiment_order, rotation=20, color="white")
axes[1].set_title("Avg PnL: Buy vs Sell by Sentiment", color="white", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Avg PnL (USD)", color="white"); axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].legend(facecolor="#161b22", labelcolor="white")

plt.tight_layout(pad=2)
plt.savefig("/home/claude/assignment/fig4_winrate_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Fig 4 saved")

# =================== SUMMARY STATS ===================
print("\n=== KEY STATS ===")
print(f"Total trades: {len(merged)}")
print(f"Total traders: {merged['account'].nunique()}")
print(f"Overall win rate: {(merged['closedPnL']>0).mean():.1%}")
print(f"Avg PnL per trade: ${merged['closedPnL'].mean():.2f}")
print(f"Best sentiment for trading: {merged.groupby('Classification')['closedPnL'].mean().idxmax()}")
print(f"Best symbol: {merged.groupby('symbol')['closedPnL'].mean().idxmax()}")
