"""
fig0_model_schematic.py -- Figure 1 of the FlowMergence paper: the model at a glance.

Panels
  (a) The lattice: two BML layers, the hub (central row and column), a delivery trip,
      a crossing block, a queue in a lane and the spillover neighbourhood.
  (b) The life cycle of an agent: off-lattice idle agent, role choice, entry, work, job
      completion, local vs hub delivery; learning continues while delivering.
  (c) Where each experiment sits: density rho(N) = rho0 * N^(1-a) for every density path used,
      with the BML jamming band.
  (d) Response thresholds: probability of taking a role, T(s) = s^2 / (s^2 + theta^2).
  (e) Learning and spillovers: moves needed per job versus same-task neighbours, with the
      mean-field link between neighbours and density.

Everything quantitative in (c)-(e) is computed from the parameters of the final
configuration (config.py: BEST, DENSITY_PATH and the paper_* experiments). Panel (a) is an
illustrative random configuration on a small lattice, not a simulation snapshot.

Usage
    python fig0_model_schematic.py            # writes fig0_model_schematic.pdf / .png
    python fig0_model_schematic.py --out figs/fig0

If config.py (and its imports) is importable, BEST and DENSITY_PATH are read from it;
otherwise the values below, copied from config.py, are used.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------- parameters
# Copied from config.py (BEST, DENSITY_PATH, paper_* experiments). Values marked "params.py"
# are defaults of params.py, which was not available when this script was written: check them.
P = dict(
    learning_exponent=1.0,      # b            (BEST)
    spillover=8.0,              # kappa        (BEST)
    spillover_radius=4,         # R            (BEST)
    route_length=225000.0,      # D            (BEST)
    area_exponent=2 / 3,        # a            (DENSITY_PATH)
    rho0=0.0125,                # rho0         (DENSITY_PATH)
    m=4,                        # tasks
    practice_gain=1.0,          # g            (params.py)
    exp_decay=0.002,            # rho_E        (params.py)
    theta_init=200.0,           # (params.py)
    theta_max=4000.0,           # (params.py)
    s0=200.0,                   # (params.py)
    delta=2.0,                  # (params.py)
    xi=3.0,                     # (params.py)
    phi=1.5,                    # (params.py)
)
try:  # prefer the live configuration when it can be imported
    import config  # noqa: F401

    P.update({k: v for k, v in config.BEST.items() if k in P})
    P.update({k: v for k, v in config.DENSITY_PATH.items() if k in P})
    _model = getattr(config, "MODEL", {})          # Paper 1: every other model parameter
    for k_cfg, k_here in (("m", "m"), ("practice_gain", "practice_gain"), ("exp_decay", "exp_decay"),
                          ("theta_init", "theta_init"), ("theta_max", "theta_max"), ("s_init", "s0"),
                          ("s_gain", "delta"), ("xi", "xi"), ("phi", "phi")):
        if k_cfg in _model:
            P[k_here] = _model[k_cfg]
except Exception:  # config.py imports paths/params, which may be absent here
    pass

N_REF = 64
JAM_BAND = (0.30, 0.40)   # BML jamming density range quoted in config.py (density_scan, λ = 1)
PL_RHO_REF, PL_N_REF = 0.0386, 1448           # paper_density_exponent: ρ(1448) for every a
PL_A = (0.76, 1.24)                           # its extreme area exponents
SCAN_RHO = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]   # density_window_scan

# --------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
})
C_H = ["#1f5fa8", "#7fb2e5"]          # horizontal-layer tasks 0, 2
C_V = ["#c4501b", "#f2a878"]          # vertical-layer tasks 1, 3
C_HUB = "#e9e3f5"
C_HUB_EDGE = "#6a51a3"
C_COORD = "#2e8b57"
C_BLOCK = "#d62728"
C_GREY = "#555555"


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def callout(ax, x, y, k, color="black"):
    ax.text(x, y, str(k), ha="center", va="center", fontsize=6, fontweight="bold",
            color="white", zorder=12,
            bbox=dict(boxstyle="circle,pad=0.18", fc=color, ec="none"))


# --------------------------------------------------------------------------- (a) lattice
def panel_lattice(ax, L=21, seed=7):
    rng = np.random.default_rng(seed)
    hub = L // 2
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color(C_GREY)

    # hub: central column (horizontal agents) and central row (vertical agents)
    ax.add_patch(Rectangle((hub - 0.5, -0.5), 1, L, fc=C_HUB, ec="none", zorder=0))
    ax.add_patch(Rectangle((-0.5, hub - 0.5), L, 1, fc=C_HUB, ec="none", zorder=0))
    ax.plot([hub - .5, hub - .5], [-.5, L - .5], color=C_HUB_EDGE, lw=0.5, zorder=1)
    ax.plot([hub + .5, hub + .5], [-.5, L - .5], color=C_HUB_EDGE, lw=0.5, zorder=1)
    ax.plot([-.5, L - .5], [hub - .5, hub - .5], color=C_HUB_EDGE, lw=0.5, zorder=1)
    ax.plot([-.5, L - .5], [hub + .5, hub + .5], color=C_HUB_EDGE, lw=0.5, zorder=1)
    for k in range(L + 1):
        ax.plot([k - .5, k - .5], [-.5, L - .5], color="#dddddd", lw=0.3, zorder=0.5)
        ax.plot([-.5, L - .5], [k - .5, k - .5], color="#dddddd", lw=0.3, zorder=0.5)

    cx, cy = 16, 4                # position of a vertical agent in the lower-right quadrant
    # spillover neighbourhood ((2R+1)^2) around a focal agent -- upper-left quadrant
    R = int(P["spillover_radius"])
    fx, fy = 4, 15
    ax.add_patch(Rectangle((fx - R - .5, fy - R - .5), 2 * R + 1, 2 * R + 1, fc="none",
                           ec="black", lw=0.8, ls=(0, (3, 2)), zorder=6))

    # random agents (density ~0.16 per layer), hard exclusion within each layer
    occ_h, occ_v = set(), set()
    fixed_h = {(fx, fy): 0, (3, 17): 0, (6, 13): 0, (2, 12): 0,    # focal + same-task peers
               (5, 7): 0, (6, 7): 0, (7, 7): 0,                    # same-lane queue
               (13, 13): 2}                                       # crossing block (with V below)
    fixed_v = {(14, 13): 1, (cx, cy + 1): 3}
    for (x, y), t in fixed_h.items():
        occ_h.add((x, y))
    for (x, y), t in fixed_v.items():
        occ_v.add((x, y))
    agents_h = dict(fixed_h)
    agents_v = dict(fixed_v)
    reserved = {(x, 20) for x in range(L)}   # top row kept free for the delivery trip
    reserved |= {(x, hub) for x in range(L)} | {(hub, y) for y in range(L)}
    reserved |= {(x, y) for x in range(fx - R, fx + R + 1) for y in range(fy - R, fy + R + 1)}
    for (x, y) in [(x, y) for x in range(L) for y in range(L)]:
        if (x, y) in reserved or (x, y) in occ_h or (x, y) in occ_v:
            continue
        u = rng.random()
        if u < 0.08:
            agents_h[(x, y)] = int(rng.choice([0, 2]))
        elif u < 0.16:
            agents_v[(x, y)] = int(rng.choice([1, 3]))
    # a few non-peer agents inside the spillover box (different tasks)
    for (x, y), t in {(1, 15): 2, (7, 16): 2, (5, 18): 1, (2, 19): 3, (8, 12): 1}.items():
        (agents_h if t in (0, 2) else agents_v)[(x, y)] = t

    def draw_h(x, y, t, hollow=False, size=34):
        c = C_H[t // 2]
        ax.scatter([x], [y], marker=">", s=size, c="white" if hollow else c,
                   edgecolors=c if hollow else "black", linewidths=0.9 if hollow else 0.3,
                   zorder=8)

    def draw_v(x, y, t, hollow=False, size=34):
        c = C_V[(t - 1) // 2]
        ax.scatter([x], [y], marker="^", s=size, c="white" if hollow else c,
                   edgecolors=c if hollow else "black", linewidths=0.9 if hollow else 0.3,
                   zorder=8)

    for (x, y), t in agents_h.items():
        draw_h(x, y, t)
    for (x, y), t in agents_v.items():
        draw_v(x, y, t)

    # same-task peers of the focal agent: ring
    for (x, y) in [(3, 17), (6, 13), (2, 12)]:
        ax.scatter([x], [y], s=95, facecolors="none", edgecolors="black", lw=0.6, zorder=7)
    ax.scatter([fx], [fy], s=120, facecolors="none", edgecolors="black", lw=1.1, zorder=7)

    # delivery trip of a horizontal agent carrying a finished job to the hub column
    dx0, dy = 15, 20           # moves right, wraps around the periodic boundary
    draw_h(dx0, dy, 0, hollow=True)
    ax.plot([dx0 + 0.35, L - 0.5], [dy, dy], color=C_H[0], lw=1.0, ls=(0, (2, 1.5)), zorder=9)
    ax.annotate("", xy=(hub - 0.3, dy), xytext=(-0.5, dy),
                arrowprops=dict(arrowstyle="-|>", color=C_H[0], lw=1.0, ls=(0, (2, 1.5)),
                                mutation_scale=7), zorder=9)
    ax.scatter([hub], [dy], marker="s", s=22, c=C_HUB_EDGE, zorder=9)

    # crossing block: H at (13,13) wants (14,13), occupied in the vertical layer
    ax.plot([13.45, 13.75], [12.7, 13.3], color=C_BLOCK, lw=1.2, zorder=10)
    ax.plot([13.45, 13.75], [13.3, 12.7], color=C_BLOCK, lw=1.2, zorder=10)

    # callouts
    callout(ax, fx + R + 0.9, fy + R - 0.4, 1)           # spillover box
    callout(ax, 13.6, 14.4, 2, C_BLOCK)                   # crossing
    callout(ax, 8.2, 7.0, 3)                               # queue
    callout(ax, 18.0, 18.9, 4, C_H[0])                     # delivery
    ax.text(hub, -1.3, "hub", ha="center", va="top", fontsize=6.5, color=C_HUB_EDGE)
    ax.text(L - 0.2, hub, "hub", ha="left", va="center", fontsize=6.5, color=C_HUB_EDGE,
            rotation=90)

    # legend (markers only)
    h = [plt.Line2D([], [], marker=">", ls="", mfc=C_H[0], mec="black", mew=0.3, ms=5,
                    label="horizontal, tasks 0/2"),
         plt.Line2D([], [], marker="^", ls="", mfc=C_V[0], mec="black", mew=0.3, ms=5,
                    label="vertical, tasks 1/3"),
         plt.Line2D([], [], marker=">", ls="", mfc="white", mec=C_H[0], mew=0.9, ms=5,
                    label="delivering to hub")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2,
              frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=5.8)
    ax.set_title("Lattice (illustrative)", loc="left", pad=3)


# --------------------------------------------------------------------------- (b) cycle
def box(ax, x, y, w, h, title, body="", fc="white", ec="black", fs=5.4, lw=0.7, dy=0.05):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    lines = [title] + ([l for l in body.split("\n")] if body else [])
    y0 = y + (len(lines) - 1) * dy / 2
    for i, l in enumerate(lines):
        ax.text(x, y0 - i * dy, l, ha="center", va="center", zorder=4,
                fontsize=fs + 0.4 if i == 0 else fs, fontweight="bold" if i == 0 else "normal")


def arrow(ax, p, q, text=None, tpos=0.5, toff=(0, 0.035), color="black", rad=0.0,
          fs=6.0, ha="center", lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=2))
    if text:
        x = p[0] + tpos * (q[0] - p[0]) + toff[0]
        y = p[1] + tpos * (q[1] - p[1]) + toff[1]
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5)


def panel_cycle(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y1, h, w = 0.64, 0.25, 0.16
    X = dict(idle=0.08, take=0.275, work=0.47, done=0.665, out=0.895)
    box(ax, X["idle"], y1, w, h, "Idle", "off the lattice\nmeets role j\nw.p. $\\propto s_j$")
    box(ax, X["take"], y1, w, h, "Take role j", "w.p. $s_j^2/(s_j^2+\\theta_{ij}^2)$\nenters a free cell")
    box(ax, X["work"], y1, w, h, "Work on task j", "direction $j$ mod 2\n$w=(1+e)^b$ per move")
    box(ax, X["done"], y1, w, h, "Job done", "after distance $D$\nleaves the lattice")
    box(ax, X["out"], y1, w * 0.85, h, "Output Y", "counts", fc="#f3f3f3")
    y2 = 0.22
    box(ax, X["work"], y2, w * 1.2, h, "Reinforce", "$\\theta$: $-\\xi$ own task, $+\\varphi$ other\n"
        "$e_j$: $+(g+\\kappa n)$ on task $j$\n$e$: $\\times(1-\\rho_E)$ every step",
        fc="#f7f7f7", ec="#999999", lw=0.5, fs=5.0)
    box(ax, X["done"] + 0.02, y2, w, h, "Deliver to hub", "trip to central\nrow / column\n(still learning)",
        fc=C_HUB, ec=C_HUB_EDGE)

    e = w / 2
    arrow(ax, (X["idle"] + e, y1), (X["take"] - e, y1))
    arrow(ax, (X["take"] + e, y1), (X["work"] - e, y1))
    arrow(ax, (X["work"] + e, y1), (X["done"] - e, y1))
    arrow(ax, (X["done"] + e, y1), (X["out"] - e * 0.85, y1), "$1-c$", toff=(0, 0.04), fs=6.5)
    arrow(ax, (X["done"], y1 - h / 2), (X["done"], y2 + h / 2), "$c$", toff=(0.02, 0), fs=6.5)
    arrow(ax, (X["done"] + e, y2), (X["out"], y1 - h / 2), rad=0.3)
    ax.plot([X["work"]] * 2, [y2 + h / 2, y1 - h / 2], color="#999999", lw=0.5, ls=":",
            zorder=1)
    ax.text(X["done"] + 0.11, 0.05, "$\\Gamma$ = time delivering / other time",
            fontsize=6.0, color=C_HUB_EDGE, ha="center", va="center")
    arrow(ax, (X["out"], y1 + h / 2), (X["idle"], y1 + h / 2), rad=0.16, color=C_GREY, lw=0.6)
    ax.text(0.49, 0.975, "next job", fontsize=6.0, color=C_GREY, ha="center", va="top")
    ax.set_title("Agent life cycle", loc="left", pad=3)


# --------------------------------------------------------------------------- (c) density
def panel_density(ax):
    N = np.logspace(np.log10(64), np.log10(131072), 200)
    ax.axhspan(*JAM_BAND, color="#fbe3e3", zorder=0, lw=0)
    ax.text(70, 0.345, "BML jam ($\\lambda=1$)", fontsize=5.8, color=C_BLOCK, va="center")
    a0, r0 = P["area_exponent"], P["rho0"]
    ax.plot(N, r0 * N ** (1 - a0), color="black", lw=1.4, zorder=4)
    ax.text(75, 0.118, "city path, $a=2/3$", fontsize=5.8, ha="left", va="bottom")
    Nd = N[N <= 32768]
    for a, ls in [(0.75, "--"), (0.85, "-."), (1.0, ":")]:
        rho0 = 0.05 / N_REF ** (1 - a)
        ax.plot(Nd, rho0 * Nd ** (1 - a), color="#4d4d4d", lw=0.9, ls=ls, zorder=3)
        ax.text(4.2e4, rho0 * 32768 ** (1 - a), f"$a={a:g}$", fontsize=5.6, va="center",
                color="#4d4d4d")
    ax.plot(Nd, np.full_like(Nd, 0.2), color="#3b7dd8", lw=1.0, zorder=3)
    ax.text(70, 0.215, "constant ρ = 0.2", fontsize=5.6, color="#3b7dd8", va="bottom")
    # power-law runs of Fig. 2: every a passes through ρ(1448) = 0.0386
    lo = PL_RHO_REF * (Nd / PL_N_REF) ** (1 - PL_A[1])
    hi = PL_RHO_REF * (Nd / PL_N_REF) ** (1 - PL_A[0])
    ax.fill_between(Nd, np.minimum(lo, hi), np.maximum(lo, hi), color="#c9a227", alpha=0.25,
                    lw=0, zorder=2)
    ax.text(90, 0.023, "Fig. 2 runs", fontsize=5.6, color="#8a6d00", va="center")
    # constant-density scan (N = 4096)
    ax.plot(np.full(len(SCAN_RHO), 4096), SCAN_RHO, "_", ms=4, mew=0.8, color="#8a6d00", zorder=5)
    ax.text(4096 * 1.25, 0.0105, "scan", fontsize=5.4, color="#8a6d00", va="bottom")
    for n_mark in (64, 131072):
        rho = r0 * n_mark ** (1 - a0)
        ax.plot([n_mark], [rho], "o", ms=3, color="black", zorder=6)
    ax.annotate("0.63", (131072, 0.63), xytext=(-3, 4), textcoords="offset points",
                fontsize=5.6, ha="right")
    ax.annotate("0.05", (64, 0.05), xytext=(3, -7), textcoords="offset points", fontsize=5.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(50, 2.6e5); ax.set_ylim(0.009, 1.0)
    ax.set_xlabel("number of agents N")
    ax.set_ylabel("density ρ = N / L²")
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.4", "0.8"])
    ax.set_title("Density paths, $\\rho=\\rho_0N^{1-a}$", loc="left", pad=3)


# --------------------------------------------------------------------------- (d) thresholds
def panel_threshold(ax):
    s = np.logspace(0, 4, 400)
    s0, d = P["s0"], P["delta"]
    s_max = s0 * (1 + d)                      # stimulus when a task has zero share
    ax.axvspan(1, s_max, color="#f0f0f0", lw=0, zorder=0)
    ax.text(28, 0.56, "stimulus range\n$s\\leq s_0(1+\\delta)$", fontsize=5.6,
            color=C_GREY, ha="center", va="center")
    for th, c, lab in [(1.0, "black", "specialist, θ = 1"),
                       (P["theta_init"], "#3b7dd8", f"initial, θ = {P['theta_init']:g}"),
                       (P["theta_max"], "#9a9a9a", f"deaf, θ = {P['theta_max']:g}")]:
        ax.plot(s, s ** 2 / (s ** 2 + th ** 2), color=c, lw=1.1, label=lab)
    ax.axvline(s0, color=C_GREY, lw=0.5, ls=":")
    ax.text(s0 * 1.08, 0.04, "$s_0$", fontsize=6, color=C_GREY, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1, 1e4); ax.set_ylim(0, 1.02)
    ax.set_xlabel("stimulus s")
    ax.set_ylabel("P(take role)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1), frameon=False, fontsize=5.6,
              handlelength=1.6)
    ax.set_title("Response thresholds", loc="left", pad=3)


# --------------------------------------------------------------------------- (e) spillovers
def panel_spillover(ax):
    g, k, rE, b, D = (P["practice_gain"], P["spillover"], P["exp_decay"],
                      P["learning_exponent"], P["route_length"])
    box_cells = (2 * P["spillover_radius"] + 1) ** 2 - 1    # other cells (traffic.same_task_counts)
    n = np.linspace(0, 16, 300)
    moves = D / ((g + k * n) / rE) ** b       # same w_ref as config.build_jobs (route_moves)
    ax.plot(n, moves, color="black", lw=1.2)
    for nn in (0, 1, 8):  # n = 8: annotate on the left
        mv = D / ((g + k * nn) / rE) ** b
        ax.plot([nn], [mv], "o", ms=3, color="black")
        ax.annotate(f"{mv:.0f}" if mv >= 10 else f"{mv:.1f}", (nn, mv),
                    xytext=(4, -9) if nn == 0 else ((5, 3) if nn < 8 else (6, 2)), textcoords="offset points", fontsize=5.8)
    ax.set_yscale("log"); ax.set_xlim(-0.3, 16); ax.set_ylim(2, 3000)
    ax.set_xlabel("same-task neighbours n in $(2R+1)^2$")
    ax.set_ylabel("moves per job, $D/w^*$")
    ax.text(9.0, 120,
            f"$w^*=[(g+\\kappa n)/\\rho_E]^b$\n$b={b:g},\\ \\kappa={k:g},\\ R={P['spillover_radius']}$",
            ha="left", va="center", fontsize=5.3)
    # mean-field link n = ((2R+1)^2 - 1) * rho / m (no idle time): the city path at its two ends
    a0, r0 = P["area_exponent"], P["rho0"]
    for n_mark in (64, 32768):
        rho = r0 * n_mark ** (1 - a0)
        nn = box_cells * rho / P["m"]
        ax.axvline(nn, color="#3b7dd8", lw=0.5, ls=":")
        ax.text(nn + 0.3, 2600, f"ρ = {rho:.2f}\nN = {n_mark}", fontsize=5.4,
                color="#3b7dd8", va="top", ha="left")
    ax.set_title("Learning and spillovers", loc="left", pad=3)


# --------------------------------------------------------------------------- figure
def make_figure():
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.12, 1.0], hspace=0.5, wspace=0.42,
                  left=0.075, right=0.985, top=0.955, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    panel_lattice(ax_a); panel_label(ax_a, "a")
    panel_cycle(ax_b); panel_label(ax_b, "b", x=0.0)
    panel_density(ax_c); panel_label(ax_c, "c", x=-0.2)
    panel_threshold(ax_d); panel_label(ax_d, "d", x=-0.2)
    panel_spillover(ax_e); panel_label(ax_e, "e", x=-0.2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="fig0_model_schematic", help="output path without suffix")
    args = ap.parse_args()
    fig = make_figure()
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("wrote", args.out + ".pdf", "and", args.out + ".png")


if __name__ == "__main__":
    main()