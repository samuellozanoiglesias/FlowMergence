"""deck_plots.py -- explanatory plots for the presentation, from paper_numbers.json only
(plot 1 is conceptual: pure power laws with literature exponents)."""
import json
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("/mnt/user-data/uploads/paper_numbers.json"))
f = lambda k: float(str(D[k]).replace(",", "").replace("+", ""))
INK, PAPER, MUTED = "#1b1f24", "#f6f4ef", "#5f656e"
BLUE, ORANGE, HUBC = "#2b6cb0", "#d9822b", "#6a51a3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 1.4, "figure.facecolor": PAPER,
    "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "legend.frameon": False})

def save(fig, name):
    fig.savefig(name, dpi=110, bbox_inches="tight"); plt.close(fig)

# 1. what an exponent means (conceptual)
fig, ax = plt.subplots(figsize=(10, 7.2))
N = np.logspace(0, 3, 50)
for b, c, lab in [(1.15, BLUE, "β = 1.15  superlinear (city output)"), (1.0, MUTED, "β = 1  linear"),
                  (0.85, ORANGE, "β = 0.85  sublinear")]:
    ax.plot(N, N ** b, color=c, lw=4, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("size N (relative)"); ax.set_ylabel("total output Y (relative)")
ax.legend(loc="upper left", fontsize=19)
ax.text(30, 1.6, "on log–log axes a power law\nY ∝ N^β is a straight line\nwith slope β", fontsize=18, color=MUTED)
save(fig, "plot_concept.png")

# 2. beta vs c
cs = ["0", "0.02", "0.05", "0.1", "0.2", "0.35", "0.5", "1"]
x = np.arange(len(cs))
b = np.array([f(f"beta_c{c}") for c in cs]); lo = np.array([f(f"beta_c{c}_lo") for c in cs])
hi = np.array([f(f"beta_c{c}_hi") for c in cs])
fig, ax = plt.subplots(figsize=(12, 7.2))
ax.axhspan(1, 1.3, color=BLUE, alpha=0.07); ax.axhspan(0.7, 1, color=ORANGE, alpha=0.08)
ax.axhline(1, color=INK, lw=1.4, ls="--")
ax.plot(x, b, color=INK, lw=2.5, zorder=2)
ax.errorbar(x, b, yerr=[b - lo, hi - b], fmt="none", ecolor=INK, capsize=4, zorder=3)
ax.scatter(x, b, s=170, c=[BLUE if v > 1 else ORANGE for v in b], edgecolor=INK, zorder=4)
for xi, v in zip(x, b):
    ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", fontsize=17)
xs = np.interp(1.0, b[::-1], x[::-1])
ax.axvline(xs, color=HUBC, lw=1.6, ls=":")
ax.text(xs + 0.1, 1.22, f"c* ≈ {D['cstar']}", color=HUBC, fontsize=20)
ax.text(5.6, 1.14, "city-like", color=BLUE, fontsize=22, fontweight="bold")
ax.text(0.05, 0.79, "company-like", color=ORANGE, fontsize=22, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cs); ax.set_ylim(0.72, 1.29)
ax.set_xlabel("centralization c  (share of jobs that must go to the hub)")
ax.set_ylabel("scaling exponent β")
save(fig, "plot_beta_vs_c.png")

# 3. agents vs planner switch size
pis = ["1", "2", "5", "10", "20", "50"]
na = np.array([f(f"choice_Nhalf_agents_pi{p}") for p in pis])
ns = np.array([f(f"choice_Nstar_planner_pi{p}") for p in pis])
fig, ax = plt.subplots(figsize=(8.6, 7.6))
g = np.logspace(2.5, 5.1, 10)
ax.fill_between(g, g / 2, g * 2, color=MUTED, alpha=0.12, label="within a factor 2")
ax.plot(g, g, color=INK, ls="--", lw=1.5, label="agents = planner")
ax.scatter(ns, na, s=200, c=BLUE, edgecolor=INK, zorder=3)
for p, a_, s_ in zip(pis, na, ns):
    ax.annotate(f"π = {p}", (s_, a_), xytext=(10, -22), textcoords="offset points", fontsize=17)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("planner's switch size N*"); ax.set_ylabel("agents' switch size N½")
ax.legend(loc="upper left", fontsize=17)
save(fig, "plot_switch.png")

# 4. exploration cost at pi = 0
Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
gap = [f(f"choice_gap_pi0_N{n}") for n in Ns]
fig, ax = plt.subplots(figsize=(10, 6.6))
ax.plot(Ns, gap, "o-", color=ORANGE, lw=3, ms=11, mec=INK)
for n, gv in zip(Ns, gap):
    if n in (128, 2048, 131072):
        ax.annotate(f"{gv:.1f}%", (n, gv), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=18)
ax.set_xscale("log"); ax.set_ylim(0, 13)
ax.set_xlabel("system size N"); ax.set_ylabel("value lost vs planner (%)")
save(fig, "plot_exploration.png")

# 5. competency trap: believed vs realized (pi = 10)
rows = [("N = 16384\nskill-adjusted", "N16384_perskill"), ("N = 16384\nraw payoff", "N16384_raw"),
        ("N = 65536\nskill-adjusted", "N65536_perskill"), ("N = 65536\nraw payoff", "N65536_raw")]
fig, ax = plt.subplots(figsize=(11, 6.8))
for i, (lab, k) in enumerate(rows):
    r_ = f(f"trap_realized_pi10_{k}"); bl = f(f"trap_believed_pi10_{k}")
    ax.plot([r_, bl], [i, i], color=MUTED, lw=3, zorder=1)
    ax.scatter([r_], [i], s=230, c=INK, zorder=3, label="realized" if i == 0 else None)
    ax.scatter([bl], [i], s=230, c=ORANGE if "raw" in k else BLUE, edgecolor=INK, zorder=3,
               label=("believed (raw)" if i == 1 else ("believed (skill-adj.)" if i == 0 else None)))
ax.axvline(0, color=INK, lw=1.2, ls="--")
ax.text(0.05, 3.45, "hub better →", fontsize=17, color=MUTED); ax.text(-1.05, 3.45, "← hub worse", fontsize=17, color=MUTED)
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows], fontsize=17)
ax.set_xlabel("advantage of the hub  (log payoff-rate ratio)"); ax.set_ylim(-0.6, 3.8)
ax.legend(loc="lower right", fontsize=16)
save(fig, "plot_trap.png")

# 6. lock-in: committed fraction from each start
cells = [("5", "4096"), ("5", "16384"), ("5", "65536"), ("10", "16384"), ("10", "65536"), ("20", "65536")]
hiv = [f(f"lockin_commit_hi_pi{p}_N{n}") for p, n in cells]
lov = [f(f"lockin_commit_lo_pi{p}_N{n}") for p, n in cells]
fig, ax = plt.subplots(figsize=(12, 6.6))
xx = np.arange(len(cells)); w = 0.38
ax.bar(xx - w / 2, hiv, w, color=ORANGE, edgecolor=INK, label="started centralized (p₀ = 0.95)")
ax.bar(xx + w / 2, lov, w, color=BLUE, edgecolor=INK, label="started decentralized (p₀ = 0.05)")
ax.set_xticks(xx); ax.set_xticklabels([f"π={p}\nN={int(n):,}" for p, n in cells], fontsize=16)
ax.set_ylabel("fraction committed to the hub"); ax.set_ylim(0, 1.25)
ax.legend(loc="upper right", fontsize=16, ncol=1)
save(fig, "plot_lockin.png")
print("ok")
