"""
config.py — Experiment definitions (sweeps). Each experiment has:
    base           : fixed overrides on Params
    grid           : {parameter: [values]}, expanded as a Cartesian product
    seeds          : replicas per point. The SAME seeds are used at every point (common
                     random numbers), which reduces the variance of differences between points.
    timeseries     : (optional) store time series in each run's result.json
    snapshot_every : (optional) store lattice snapshots every k steps (snapshots.npz)

Recommended order:
    smoke -> calibration_fine -> calibration_robustness -> density_scan (+ _nocoord,
    _spillover) -> encounter_check -> control_density -> lambda_sweep (+ _nocoord)
    -> centralization_sweep (PRIORITY) -> centralization_constant_density -> coord_value
    (+ _endogenous) -> coord_frontier_fixed (+ _endogenous) -> jam_transition
    -> lambda_sweep_nocoord -> phase_diagram -> area_exponent -> ossification
    -> useless_bureaucracy -> learning_ablation -> large_N (GPU)
"""
from __future__ import annotations

import hashlib
import itertools
import json

import paths
from params import Params

N_GRID = [2 ** k for k in range(6, 16)]         # 64 … 32768  (2.7 decades)
N_COARSE = [2 ** k for k in range(6, 16, 2)]    # 64, 256, 1024, 4096, 16384
LAMBDAS = [round(0.1 * i, 3) for i in range(11)]
N_WIDE = [2 ** k for k in range(6, 18)]         # 64 … 131072 (3.3 decades)
# Density path chosen from density_scan: with a = 2/3 and rho0 = 0.0125, density goes
# 0.05 (N=64) -> 0.40 (N=32768) -> 0.63 (N=131072): it crosses the whole hump of
# y(ρ), from the spillover-dominated regime (ε > 0) to beyond the BML jam (ε < 0).
DENSITY_PATH = dict(area_exponent=2 / 3, rho0=0.0125)
N_REF = 64   # reference size for the derived key rho_at_Nref
# Final configuration for the paper (from superlinear_city / superlinear_confirm):
# b = 1, κ = 8, R = 4, jobs ≈ 50 moves of a typically skilled agent, no efficiency cap.
# Applied EXPLICITLY to every paper_* experiment, so it does not depend on params.py defaults.
BEST = dict(learning_exponent=1.0, spillover=8.0, spillover_radius=4, route_length=225000.0,
            w_max=1e12)
# The OLD params.py defaults (changed to BEST between the choice_sweep launch, 21:32, and the
# choice_lockin launch, 22:59, on 2026-09-30). choice_sweep and choice_fixed_premium ran with
# these; pinned here so a relaunch reproduces them. choice_lockin ran with BEST.
OLD = dict(learning_exponent=0.5, spillover=2.0, spillover_radius=2, route_length=200.0,
           w_max=200.0)
# Paper 1: EVERY other model and run parameter, pinned to the values of the final runs, so new
# experiments do not depend on params.py defaults either (BEST pins only the learning/job
# parameters). Values = params.py defaults as of the paper_* launches; paper_figures.py
# (check_pinned) verifies that the stored params of every experiment it uses match them.
MODEL = dict(m=4, practice_gain=1.0, exp_decay=0.002, theta_min=1.0, theta_max=4000.0,
             theta_init=200.0, theta_init_noise=0.0, xi=3.0, phi=1.5, s_init=200.0, s_max=1000.0,
             s_gain=2.0, share_ema=0.02, p_quit_work=0.0, encounter="stimulus", traffic_light=True,
             L_min=8, route_length_spread=0.0, demand_skew=0.0, T_burn=6000, T_measure=2000,
             hub_count_exponent=0.0, hub_spacing_ref=16.0, learn_while_delivering=True)
P1 = dict(BEST, **MODEL)                 # disjoint keys: safe
# Bases of the Paper-1 experiments (override keys with dict(BASE, key=value), never dict(**BASE, ...))
PL_BASE = dict(P1, exclusion=0.5, coordination=False, centralization=0.0, rho_ref_N=1448)
CITY_BASE = dict(P1, **DENSITY_PATH, exclusion=0.5, coordination=False)
PL_AVALS_SHORT = [0.824, 1.0, 1.176]
# Seeds of the comparison experiments = seeds of their baselines, so every run has a partner with
# the same seed (common random numbers): paper_density_exponent has seeds 0–2,
# paper_centralization_sweep seeds 0–3. More seeds here would have no partner.
ABL_SEEDS_PL = 3
ABL_SEEDS_C = 4
C_SHORT = [0.0, 0.2, 1.0]
# E3 ablations: each removes ONE ingredient. route_moves = 50 keeps jobs at 50 moves of a
# typically skilled agent when the change would otherwise make them absurdly long (κ = 0:
# w ≈ 500 -> D = 25000; b = 0: w = 1 -> D = 50).
ABLATIONS = {
    "kappa0": dict(spillover=0.0, route_moves=50),    # no spillovers (practice only)
    "nodol": dict(xi=0.0, phi=0.0),                   # thresholds frozen at θ_init ≈ s0: no
                                                      # specialization, tasks taken at random
    "lambda0": dict(exclusion=0.0),                   # crossings never block
    "b0": dict(learning_exponent=0.0, route_moves=50),  # no learning at all (w = 1)
}

EXPERIMENTS = {
    # ================================================================== FINAL PAPER (best config)
    "paper_centralization_sweep": dict(
        description="PAPER Fig. 1–2: one knob c on the city density path (a = 2/3), best config. "
                    "Fine c grid around the crossover (c* ≈ 0.15 in the final runs).",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False),
        grid=dict(N=N_WIDE, centralization=[0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]),
        seeds=4,
    ),
    "paper_constant_density": dict(
        description="PAPER control: constant density (a = 1). c = 0 must give β ≈ 1 (extensivity); "
                    "c > 0 isolates the organizational effect.",
        base=dict(**BEST, area_exponent=1.0, rho0=0.2, exclusion=0.5, coordination=False),
        grid=dict(N=N_GRID, centralization=[0.0, 0.05, 0.2, 0.5, 1.0]),
        seeds=4,
    ),
    "paper_densification": dict(
        description="PAPER (SI S2): β − 1 ≈ (1 − a)·ε. Densification rate a at c = 0 and c = 0.2, same "
                    "starting density 0.05 at N = 64.",
        base=dict(**BEST, exclusion=0.5, coordination=False),
        grid=dict(N=N_GRID, area_exponent=[1.0, 0.85, 0.75, 2 / 3], rho_at_Nref=[0.05],
                  centralization=[0.0, 0.2]),
        seeds=3,
    ),
    "paper_coord_value": dict(
        description="PAPER Fig. 5: value of coordination (fixed bureaucracy) vs size, decentralized "
                    "vs centralized, near the jam (ρ = 0.4, λ = 1), best config.",
        base=dict(**BEST, area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True,
                  coord_mode="fixed", coord_relocate_prob=0.1),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.0, 0.5, 1.0],
                  coord_fixed_frac=[0.0, 0.01, 0.02, 0.04, 0.08, 0.12]),
        seeds=3,
    ),
    "paper_coord_value_endogenous": dict(
        description="PAPER Fig. 5: the same points with emergent bureaucracy.",
        base=dict(**BEST, area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.0, 0.5, 1.0]),
        seeds=3,
    ),
    "paper_choice_fixed": dict(
        description="PAPER option B (optional): planner benchmark + baseline with the best config.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False, T_burn=10000),
        grid=dict(N=N_WIDE, centralization=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]),
        seeds=3,
    ),
    "paper_choice_sweep": dict(
        description="PAPER option B (optional): emergent centralization, knob = hub premium π.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", T_burn=10000),
        grid=dict(N=N_WIDE, hub_premium=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
        seeds=4,
    ),
    "paper_choice_lockin": dict(
        description="PAPER option B (optional): lock-in, started centralized vs decentralized.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", T_burn=10000),
        grid=dict(N=[1024, 4096, 16384, 65536], hub_premium=[5.0, 10.0, 20.0],
                  choice_p0=[0.05, 0.95]),
        seeds=4,
    ),
    # ---------------------------------------------------------- option B, logit choice rule
    # Bush–Mosteller with one shared aspiration has a noise-dependent floor (≈ 0.1–0.2) and
    # ceiling (≈ 0.8–0.9); at large N the floor alone makes the hub a bottleneck. The logit
    # rule (per-mode payoff estimates) reaches the explore level 0.02 / 0.98.
    "choice_logit_check": dict(
        description="Quick check of choice_rule = logit (best config): floor at π = 0 should be "
                    "≈ 0.02 and β of value ≈ baseline; compare with paper_choice_fixed.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", choice_rule="logit", T_burn=10000),
        grid=dict(N=[256, 1024, 4096, 16384, 65536], hub_premium=[0.0, 10.0, 50.0]),
        seeds=2,
    ),
    "paper_choice_logit_sweep": dict(
        description="PAPER option B: emergent centralization, logit rule, skill-adjusted "
                    "payoff (no competency trap), ε = 0.005.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", choice_rule="logit", choice_payoff="per_skill",
                  choice_explore=0.005, T_burn=20000),
        grid=dict(N=N_WIDE, hub_premium=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
        seeds=4,
    ),
    "paper_choice_logit_lockin": dict(
        description="PAPER option B: lock-in with the logit rule, raw payoff "
                    "(competency-trap history dependence).",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", choice_rule="logit", choice_payoff="raw",
                  choice_explore=0.005, T_burn=20000),
        grid=dict(N=[1024, 4096, 16384, 65536], hub_premium=[5.0, 10.0, 20.0],
                  choice_p0=[0.05, 0.95]),
        seeds=4,
    ),
    "paper_choice_logit_coordination": dict(
        description="PAPER option B: logit choice with endogenous coordinators "
                    "(compare with paper_choice_logit_sweep).",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=True,
                  delivery_choice="learned", choice_rule="logit", T_burn=10000),
        grid=dict(N=N_WIDE, hub_premium=[0.0, 5.0, 20.0]),
        seeds=4,
    ),
    "smoke": dict(
        description="Quick end-to-end test that the whole pipeline runs.",
        base=dict(T_burn=300, T_measure=200),
        grid=dict(N=[64, 128, 256, 512], exclusion=[0.0, 1.0]),
        seeds=2,
    ),
    "calibration_xi_phi": dict(
        description="(coarse, superseded by calibration_fine) Locate where division of labour "
                    "emerges over θ_max × ξ × φ.",
        base=dict(N=1024, exclusion=0.5, theta_init=200.0),
        grid=dict(theta_max=[400.0, 600.0, 1000.0],
                  xi=[1.0, 3.0, 10.0],
                  phi=[0.5, 1.5, 5.0]),
        seeds=2,
        timeseries=True,
    ),
    "calibration_fine": dict(
        description="Fine calibration. The ratio φ/ξ is the master knob (window found around "
                    "0.5); θ_max helps (higher = more selective); θ_init near the stimulus "
                    "avoids the deaf trap; small ξ = cleaner but slower. 'phi_over_xi' is a "
                    "derived key: φ = ratio × ξ.",
        base=dict(N=1024, exclusion=0.5),
        grid=dict(phi_over_xi=[0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2],
                  xi=[0.3, 1.0, 3.0],
                  theta_max=[1000.0, 2000.0, 4000.0],
                  theta_init=[100.0, 200.0, 400.0]),
        seeds=4,
        timeseries=True,
    ),
    "calibration_robustness": dict(
        description="Does the chosen (ξ, φ, θ_max, θ_init) keep specializing across N and λ? "
                    "Pass the candidate with --set xi=... phi=... theta_max=... theta_init=...",
        grid=dict(N=[64, 256, 1024, 4096, 16384], exclusion=[0.0, 0.5, 1.0]),
        seeds=4,
        timeseries=True,
    ),
    "density_scan": dict(
        description="Density response at FIXED N: with a=1, density = rho0 exactly. Measures "
                    "ε(ρ, λ) = d log y / d log ρ, so that β − 1 ≈ (1 − a)·ε for any area exponent a. "
                    "Spans from sparse to beyond the BML jam (~0.3–0.4).",
        base=dict(N=4096, area_exponent=1.0),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "density_scan_nocoord": dict(
        description="Same as density_scan without coordinators: what do the cops buy at each density?",
        base=dict(N=4096, area_exponent=1.0, coordination=False),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "density_scan_spillover": dict(
        description="Benefit side: how ε depends on the spillover strength κ (λ fixed at 0.5).",
        base=dict(N=4096, area_exponent=1.0, exclusion=0.5),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6],
                  spillover=[0.0, 1.0, 2.0, 4.0, 8.0]),
        seeds=4,
    ),
    # ================================================================== PRIORITY: one knob
    "centralization_sweep": dict(
        description="PRIORITY. One knob, centralization c: fraction of jobs that must be "
                    "delivered to a central hub. Same density path for every c (a=2/3). "
                    "Expect local β > 1 for small c (city) and < 1 for large c (company), with "
                    "a crossover size that shrinks as c grows. Coordination on and off.",
        base=dict(**DENSITY_PATH, exclusion=0.5),
        grid=dict(N=N_WIDE, centralization=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                  coordination=[True, False]),
        seeds=6,
    ),
    "centralization_examples": dict(
        description="Time series for a few centralized runs: check that output settles "
                    "before T_burn even when delivery trips are long.",
        base=dict(**DENSITY_PATH, exclusion=0.5),
        grid=dict(N=[4096, 65536], centralization=[0.0, 0.1, 1.0]),
        seeds=1,
        timeseries=True,
    ),
    "densification_x_centralization": dict(
        description="Both knobs together: how fast density grows with N (area exponent a; "
                    "density ∝ N^(1−a)) and how centralized flows are (c). Every run starts "
                    "from density 0.05 at N=64 (derived key rho_at_Nref). Prediction: "
                    "β − 1 ≈ (1 − a)·ε − centralization penalty; trips scale as L ∝ N^(a/2).",
        base=dict(exclusion=0.5, coordination=True),
        grid=dict(N=N_GRID, area_exponent=[1.0, 0.85, 0.75, 2 / 3, 0.6],
                  rho_at_Nref=[0.05], centralization=[0.0, 0.05, 0.2]),
        seeds=4,
    ),
    # ================================================================== option B: emergent centralization
    "choice_sweep": dict(
        description="OPTION B. Agents LEARN whether to deliver to the hub (Bush–Mosteller on their "
                    "payoff rate). Knob: hub premium π (value of integration); π = 0 tests whether "
                    "centralization emerges from the endogenous hub benefit alone. Centralization, "
                    "Γ and β emerge. Same density path and λ as centralization_sweep.",
        base=dict(**DENSITY_PATH, **OLD, exclusion=0.5, coordination=False, delivery_choice="learned",
                  T_burn=10000),
        grid=dict(N=N_WIDE, hub_premium=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]),
        seeds=4,
    ),
    "choice_fixed_premium": dict(
        description="Planner benchmark AND decentralized baseline (c = 0), same settings as "
                    "choice_sweep: fixed centralization c on a finer grid; for any premium π the "
                    "optimal c maximizes value per agent y·(1 + π·c).",
        base=dict(**DENSITY_PATH, **OLD, exclusion=0.5, coordination=False, T_burn=10000),
        grid=dict(N=N_WIDE, centralization=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]),
        seeds=3,
    ),
    "choice_lockin": dict(
        description="Lock-in / hysteresis: identical systems that start centralized (p0 = 0.95) or "
                    "decentralized (p0 = 0.05). Different final states = history dependence.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False, delivery_choice="learned",
                  T_burn=10000),
        grid=dict(N=[1024, 4096, 16384, 65536], hub_premium=[5.0, 10.0, 20.0],
                  choice_p0=[0.05, 0.95]),
        seeds=4,
    ),
    "choice_with_coordination": dict(
        description="(optional) choice_sweep with endogenous coordinators.",
        base=dict(**DENSITY_PATH, exclusion=0.5, coordination=True, delivery_choice="learned",
                  T_burn=10000),
        grid=dict(N=N_WIDE, hub_premium=[0.0, 5.0, 20.0]),
        seeds=4,
    ),
    "coord_value_extended": dict(
        description="coord_value for centralized systems with larger coordinator fractions (the "
                    "optimum was at the grid edge, 8%). Concatenate with coord_value.",
        base=dict(area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True,
                  coord_mode="fixed", coord_relocate_prob=0.1),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.5, 1.0],
                  coord_fixed_frac=[0.12, 0.16, 0.24, 0.32]),
        seeds=3,
    ),
    "choice_logit_robust": dict(
        description="Logit check: exploration cost (ε) and convergence (T_burn) at large N.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", choice_rule="logit"),
        grid=dict(N=[16384, 65536], hub_premium=[0.0, 10.0],
                  choice_explore=[0.005], T_burn=[10000, 20000]),
        seeds=2,
    ),
    # Competency-trap test: same cells, learning from raw vs skill-adjusted payoff rates.
    "choice_trap_test": dict(
        description="Is the π = 10 large-N split a competency trap? raw vs per_skill payoff.",
        base=dict(**DENSITY_PATH, **BEST, exclusion=0.5, coordination=False,
                  delivery_choice="learned", choice_rule="logit", choice_explore=0.005,
                  T_burn=20000),
        grid=dict(N=[16384, 65536], hub_premium=[0.0, 10.0],
                  choice_payoff=["raw", "per_skill"]),
        seeds=2,
    ),
    # ================================================================== stronger superlinearity
    "superlinear_tuning": dict(
        description="PRIORITY: city-like β ≈ 1.15+. ε ≈ b × [κn/(1+κn)] × (moving share of job time); "
                    "the defaults give 0.5 × 0.5 × 0.6 ≈ 0.15 (density_scan_spillover). Levers: "
                    "b (learning_exponent), neighbours n via spillover_radius R and κ, and job "
                    "length route_moves (derived key: D = route_moves · w_ref). Density scan at fixed "
                    "N; w_max lifted so the efficiency cap never binds.",
        base=dict(N=4096, area_exponent=1.0, exclusion=0.5, coordination=False, w_max=1e12),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.2, 0.4], learning_exponent=[0.75, 1.0],
                  spillover=[4.0, 8.0], spillover_radius=[2, 4], route_moves=[20, 50, 100]),
        seeds=2,
    ),
    "superlinear_city": dict(
        description="PRIORITY: city-only confirmation (c = 0) on the city density path with the "
                    "predicted best setting (b=1, κ=8, R=4). EDIT these if superlinear_tuning finds "
                    "a better one. Target: β ≥ 1.15.",
        base=dict(**DENSITY_PATH, exclusion=0.5, coordination=False, w_max=1e12,
                  learning_exponent=1.0, spillover=8.0, spillover_radius=4),
        grid=dict(N=N_GRID, route_moves=[20, 50, 100]),
        seeds=4,
    ),
    "superlinear_confirm": dict(
        description="Then the transition: the same setting with centralization from 0 to 1 "
                    "(longer jobs dilute Γ = c·L/T, hence the wide c range).",
        base=dict(**DENSITY_PATH, exclusion=0.5, coordination=False, w_max=1e12,
                  learning_exponent=1.0, spillover=8.0, spillover_radius=4, route_moves=50),
        grid=dict(N=N_GRID, centralization=[0.0, 0.05, 0.2, 0.5, 1.0]),
        seeds=3,
    ),
    "centralization_constant_density": dict(
        description="Same knob at CONSTANT density (a=1, ρ=0.2): isolates the organizational "
                    "effect. c=0 must give β ≈ 1 (extensivity); c>0 should give β < 1.",
        base=dict(area_exponent=1.0, rho0=0.2, exclusion=0.5),
        grid=dict(N=N_GRID, centralization=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                  coordination=[True, False]),
        seeds=4,
    ),
    # ================================================================== coordination value
    "coord_value": dict(
        description="Is coordination LESS USEFUL in big / centralized systems? Fixed "
                    "bureaucracy (coord_mode=fixed) at constant density near the jam "
                    "(ρ=0.4, λ=1). Output vs coordinator fraction for each N and c -> "
                    "marginal value of a coordinator vs size.",
        base=dict(area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True,
                  coord_mode="fixed", coord_relocate_prob=0.1),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.0, 0.5, 1.0],
                  coord_fixed_frac=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08]),
        seeds=3,
    ),
    "coord_value_endogenous": dict(
        description="Same points as coord_value with EMERGENT bureaucracy, to compare the "
                    "amount that emerges with the optimum.",
        base=dict(area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.0, 0.5, 1.0]),
        seeds=3,
    ),
    # ================================================================== coordination frontier
    "coord_frontier_fixed": dict(
        description="Optimal bureaucracy M*(λ, ρ): fixed coordinator fractions on a (λ, ρ) "
                    "grid at N=16384, constant density.",
        base=dict(N=16384, area_exponent=1.0, coordination=True, coord_mode="fixed",
                  coord_relocate_prob=0.1),
        grid=dict(rho0=[0.2, 0.3, 0.4, 0.5, 0.6], exclusion=[0.5, 0.75, 1.0],
                  coord_fixed_frac=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08]),
        seeds=3,
    ),
    "coord_frontier_endogenous": dict(
        description="Same (λ, ρ) grid with emergent bureaucracy: compare with the optimum.",
        base=dict(N=16384, area_exponent=1.0, coordination=True),
        grid=dict(rho0=[0.2, 0.3, 0.4, 0.5, 0.6], exclusion=[0.5, 0.75, 1.0]),
        seeds=3,
    ),
    # ================================================================== jamming transition
    "jam_transition": dict(
        description="Is the collapse a true phase transition? Fine (λ, ρ) grid at three sizes, "
                    "constant density, with and without coordinators. A true transition "
                    "sharpens as N grows (BML).",
        base=dict(area_exponent=1.0),
        grid=dict(N=[4096, 16384, 65536],
                  rho0=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                  exclusion=[0.8, 0.85, 0.9, 0.95, 1.0],
                  coordination=[False, True]),
        seeds=4,
    ),
    # ================================================================== optional
    "spillover_on_path": dict(
        description="(optional) How strong can the city regime get? Spillover κ along the "
                    "density path.",
        base=dict(**DENSITY_PATH, exclusion=0.5),
        grid=dict(N=N_GRID, spillover=[1.0, 2.0, 4.0, 8.0]),
        seeds=4,
    ),
    "control_density": dict(
        description="CONTROL: constant density (a=1). β ≈ 1 is expected for every λ. "
                    "If not, there is a finite-size effect or a bug.",
        base=dict(area_exponent=1.0),
        grid=dict(N=N_GRID, exclusion=[0.0, 0.5, 1.0]),
        seeds=6,
    ),
    "lambda_sweep": dict(
        description="MAIN: β(λ), local β(N) and γ(λ). Density grows from 0.05 to 0.63 across N "
                    "(a=2/3, rho0=0.0125), crossing the hump found in density_scan. "
                    "N ≥ 65536 is slow: split with --max-N / --min-N if needed.",
        base=dict(**DENSITY_PATH),
        grid=dict(N=N_WIDE, exclusion=LAMBDAS),
        seeds=8,
    ),
    "lambda_sweep_nocoord": dict(
        description="Ablation: same as lambda_sweep but without the coordinator role "
                    "(expect a jam collapse at large N for high λ).",
        base=dict(**DENSITY_PATH, coordination=False),
        grid=dict(N=N_WIDE, exclusion=LAMBDAS),
        seeds=8,
    ),
    "lambda_sweep_old_path": dict(
        description="(legacy) original main sweep with a=0.85, rho0=0.05: density 0.09–0.24 only.",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_GRID, exclusion=LAMBDAS),
        seeds=8,
    ),
    "encounter_check": dict(
        description="Confirms the encounter fix: with 'stimulus', enabling coordination at λ=0 "
                    "(where M=0) must no longer cost output; 'uniform' reproduces the artifact.",
        base=dict(N=4096, area_exponent=1.0, exclusion=0.0),
        grid=dict(rho0=[0.05, 0.2, 0.5], coordination=[True, False],
                  encounter=["stimulus", "uniform"]),
        seeds=4,
    ),
    "phase_diagram": dict(
        description="β on the (κ spillover, λ exclusion) plane: benefit versus cost.",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_COARSE, spillover=[0.0, 0.5, 1.0, 2.0, 4.0],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "area_exponent": dict(
        description="How much it matters that density grows with N (a = 2/3 … 1).",
        grid=dict(N=N_GRID, area_exponent=[0.67, 0.75, 0.85, 1.0], exclusion=[0.2, 0.8]),
        seeds=6,
    ),
    "ossification": dict(
        description="Bureaucracy: tenure (p_quit_coord), conflict (η) and relocation.",
        base=dict(**DENSITY_PATH, exclusion=1.0),
        grid=dict(N=N_COARSE, p_quit_coord=[0.001, 0.01, 0.1],
                  coord_conflict=[0.0, 0.1, 0.3], coord_relocate_prob=[0.0, 0.05]),
        seeds=4,
    ),
    "useless_bureaucracy": dict(
        description="Control: coordinators that do not help (q=0). Cost of the role by itself.",
        base=dict(**DENSITY_PATH, coord_strength=0.0, coord_signal="all"),
        grid=dict(N=N_GRID, exclusion=[0.5, 1.0]),
        seeds=6,
    ),
    "learning_ablation": dict(
        description="No payoff from specializing (b=0) versus b>0.",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_COARSE, learning_exponent=[0.0, 0.25, 0.5, 0.75], exclusion=[0.2, 0.8]),
        seeds=4,
    ),
    "dynamics_examples": dict(
        description="A few runs with time series and snapshots to inspect transients.",
        base=dict(area_exponent=0.85),
        grid=dict(N=[1024, 8192], exclusion=[0.0, 0.5, 1.0]),
        seeds=2,
        timeseries=True,
        snapshot_every=100,
    ),
    "large_N": dict(
        description="Extends the N range (recommended on GPU: --backend cupy --workers 8).",
        base=dict(area_exponent=0.85),
        grid=dict(N=[2 ** 16, 2 ** 17, 2 ** 18], exclusion=[0.0, 0.5, 1.0]),
        seeds=3,
    ),
    # ------------------------------------------------ one knob, pure power laws on both sides
    # Output per agent ∝ density^ε when ε is constant, and density ∝ N^(1−a), so
    # β = 1 + ε(1 − a) for EVERY a: a < 1 (densifying, city-like) gives β > 1 at all N,
    # a > 1 (diluting) gives β < 1 at all N, a = 1 gives β = 1. Requirement: all densities
    # visited stay inside the window where ε is constant. Step 1 finds that window.
    "density_window_scan": dict(
        description="Constant density (a = 1) over a wide density range, best config: where is "
                    "ε = d ln y / d ln ρ constant? Sets RHO_MID and the N range of "
                    "paper_density_exponent.",
        base=dict(**BEST, exclusion=0.5, coordination=False, centralization=0.0,
                  area_exponent=1.0, N=4096),
        grid=dict(rho0=[0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
        seeds=2,
    ),
    "density_window_scan_b075": dict(
        description="Same scan for the β ≈ 1.15 candidate (b = 0.75, κ = 4, R = 2, 50 moves).",
        # dict(BEST, ...) lets the keyword arguments OVERRIDE BEST (dict(**BEST, ...) cannot)
        base=dict(BEST, exclusion=0.5, coordination=False, centralization=0.0,
                  area_exponent=1.0, N=4096, learning_exponent=0.75, spillover=4.0,
                  spillover_radius=2),
        grid=dict(rho0=[0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                  route_moves=[50]),
        seeds=2,
    ),
    "paper_density_exponent": dict(
        description="PAPER: one knob a (how interaction density scales with size), c = 0, best "
                    "config. Density 0.0386 at N = 1448 for every a, so all runs stay in the "
                    "window where ε ≈ 0.85 is constant (density_window_scan): β = 1 + ε(1 − a) "
                    "is a pure power law for every a. a = 0.824 -> β ≈ 1.15, a = 1.176 -> β ≈ 0.85.",
        base=dict(**BEST, exclusion=0.5, coordination=False, centralization=0.0,
                  rho_ref_N=1448),                       # geometric middle of 64 … 32768
        grid=dict(N=N_GRID,
                  area_exponent=[0.76, 0.824, 0.9, 1.0, 1.1, 1.176, 1.24],
                  rho_at_Nref=[0.0386]),
        seeds=3,
    ),
}


# ====================================================================== PAPER 1 additions
EXPERIMENTS.update({
    # E4: generality of β = 1 + ε(1 − a) with a second learning setting (ε ≈ 0.49), inside its
    # own window ρ ≈ 0.07–0.30 (density_window_scan_b075): density 0.145 at N = 1448 for every a
    "paper_density_exponent_b075": dict(
        description="PAPER 1, E4: Fig. 2 with b = 0.75, κ = 4, R = 2, 50-move jobs, in its own "
                    "density window (ρ(1448) = 0.145; all runs within 0.07–0.30).",
        base=dict(PL_BASE, learning_exponent=0.75, spillover=4.0, spillover_radius=2),
        grid=dict(N=N_GRID, area_exponent=[0.8, 0.85, 0.9, 1.0, 1.1, 1.15, 1.2],
                  rho_at_Nref=[0.145], route_moves=[50]),
        seeds=3,
    ),
    # E5b: does practice during delivery trips produce p < 1 (and the hub-saturation residual)?
    "deliver_learning_check": dict(
        description="PAPER 1, E5b: city path, no experience gain while carrying a job to the hub. "
                    "Compare with paper_centralization_sweep seeds 0–3 (c = 0 is unaffected).",
        base=dict(CITY_BASE, learn_while_delivering=False),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.2, 1.0]),
        seeds=ABL_SEEDS_C,
    ),
    # E6: transient check for the most expensive cells of Fig. 2 (Fig. 3 is checked without new
    # runs: paper_choice_fixed repeats its cells with T_burn = 10000)
    "transient_check_pl": dict(
        description="PAPER 1, E6: Fig. 2 cells with doubled transient (T_burn = 12000), with time "
                    "series. Compare with paper_density_exponent seeds 0–1.",
        base=dict(PL_BASE, T_burn=12000),
        grid=dict(N=[8192, 32768], area_exponent=[0.76, 1.0, 1.24], rho_at_Nref=[0.0386]),
        seeds=2,
        timeseries=True,
    ),
    # E7: an independent seed block for Fig. 2 (seeds 100–104), to test whether the bootstrap
    # CIs of 3 seeds are honest. Same physics as paper_density_exponent (check_pinned).
    "paper_density_exponent_seedsB": dict(
        description="PAPER 1, E7: paper_density_exponent with independent seeds 100–104.",
        base=dict(PL_BASE),
        grid=dict(N=N_GRID, area_exponent=[0.76, 0.824, 0.9, 1.0, 1.1, 1.176, 1.24],
                  rho_at_Nref=[0.0386]),
        seeds=5,
        seed_offset=100,
    ),
    # E9: polycentric delivery. Constant density inside the window (a = 1, ρ = 0.0386), so
    # β0 ≈ 1 at every N and the only size dependence is Γ: Γ_ff ∝ N^((1 − eta)/2).
    # Pilot first: is there a job length (route_moves) with Γ ≫ 1 already at N = 64 and no
    # runaway hub saturation at N = 16384?
    "poly_pilot": dict(
        description="PAPER 1, E9 pilot: c = 1, eta ∈ {0, 0.5, 1}, short jobs, three sizes.",
        base=dict(P1, area_exponent=1.0, rho0=0.0386, exclusion=0.5, coordination=False,
                  centralization=1.0, hub_spacing_ref=16.0),
        grid=dict(N=[64, 1024, 16384], hub_count_exponent=[0.0, 0.5, 1.0], route_moves=[2, 5, 10]),
        seeds=2,
    ),
    "poly_pilot_base": dict(
        description="PAPER 1, E9 pilot: c = 0 baseline (y0) for poly_pilot.",
        base=dict(P1, area_exponent=1.0, rho0=0.0386, exclusion=0.5, coordination=False,
                  centralization=0.0),
        grid=dict(N=[64, 1024, 16384], route_moves=[2, 5, 10]),
        seeds=2,
    ),
    # E9 main grid. route_moves = 5, chosen from poly_pilot (2026-10-05): it is the only job length
    # for which the local exponent changes by <= 0.05 between N = 64–1024 and 1024–16384 for every
    # eta tested (0.007, 0.048, 0.044 for eta = 0, 0.5, 1) while staying close to the ideal
    # (1 − eta)/2 growth of the delivery cost. route_moves = 2 fails at eta = 0.5 (0.084) and
    # eta = 1 (0.055); route_moves = 10 passes but its delivery share is too small (Γ ≈ 1 at N = 64
    # for eta = 0) and the exponents fall further from (1 − eta)/2. hub_spacing_ref = 16 as in the pilot.
    "poly_main": dict(
        description="PAPER 1, E9: c = 1, eta ∈ {0, .25, .5, .75, 1}, N = 64–32768. route_moves = 5 "
                    "chosen from poly_pilot (2026-10-05).",
        base=dict(P1, area_exponent=1.0, rho0=0.0386, exclusion=0.5, coordination=False,
                  centralization=1.0, hub_spacing_ref=16.0),
        grid=dict(N=N_GRID, hub_count_exponent=[0.0, 0.25, 0.5, 0.75, 1.0], route_moves=[5]),
        seeds=3,
    ),
    "poly_main_base": dict(
        description="PAPER 1, E9: c = 0 baseline for poly_main (same route_moves).",
        base=dict(P1, area_exponent=1.0, rho0=0.0386, exclusion=0.5, coordination=False,
                  centralization=0.0),
        grid=dict(N=N_GRID, route_moves=[5]),
        seeds=3,
    ),
    # E10 (NOT recommended for Paper 1, see report): coordination with enough seeds to separate
    # jammed from flowing runs.
    "coord_jam_seeds": dict(
        description="(optional) paper_coord_value_endogenous + no/8% fixed coordinators, 12 seeds.",
        base=dict(P1, area_exponent=1.0, rho0=0.4, exclusion=1.0, coordination=True,
                  coord_relocate_prob=0.1),
        grid=dict(N=[1024, 4096, 16384, 65536], centralization=[0.0, 0.5, 1.0],
                  coord_mode=["endogenous", "fixed"], coord_fixed_frac=[0.0, 0.08]),
        seeds=12,
    ),
})
# E3 ablations: one experiment per ingredient, short Fig. 2 (3 values of a) and short Fig. 3
# (c ∈ {0, 0.2, 1}); seeds 0–2 / 0–3 pair with the same seeds of the full experiments (common
# random numbers). λ = 0 runs the centralization version up to N = 131072 (hub saturation lives there).
for _name, _ov in ABLATIONS.items():
    EXPERIMENTS[f"abl_pl_{_name}"] = dict(
        description=f"PAPER 1, E3: Fig. 2 short version, ablation '{_name}' ({_ov}).",
        base=dict(PL_BASE, **{k: v for k, v in _ov.items() if k != "route_moves"}),
        grid=dict(N=N_GRID, area_exponent=PL_AVALS_SHORT, rho_at_Nref=[0.0386],
                  **({"route_moves": [_ov["route_moves"]]} if "route_moves" in _ov else {})),
        seeds=ABL_SEEDS_PL,
    )
    EXPERIMENTS[f"abl_c_{_name}"] = dict(
        description=f"PAPER 1, E3: Fig. 3 short version, ablation '{_name}' ({_ov}).",
        base=dict(CITY_BASE, **{k: v for k, v in _ov.items() if k != "route_moves"}),
        grid=dict(N=N_WIDE if _name == "lambda0" else N_GRID, centralization=C_SHORT,
                  **({"route_moves": [_ov["route_moves"]]} if "route_moves" in _ov else {})),
        seeds=ABL_SEEDS_C,
    )
# E12 (optional): is the centralization crossover sensitive to λ? (λ = 0 is abl_c_lambda0)
EXPERIMENTS["abl_c_lambda1"] = dict(
    description="PAPER 1, E12: Fig. 3 short version with λ = 1 (hard crossings).",
    base=dict(CITY_BASE, exclusion=1.0),
    grid=dict(N=N_GRID, centralization=C_SHORT),
    seeds=ABL_SEEDS_C,
)


# λ dependence (Paper 1, SI): how β and output per agent depend on the crossing-block probability.
# λ = 0, 0.5 and 1 already exist on the city path (abl_c_lambda0, paper_centralization_sweep,
# abl_c_lambda1, seeds 0–3); these runs add the intermediate values with the SAME seeds, so every
# λ is paired by seed. N = 64–32768 (the range of abl_c_lambda1).
EXPERIMENTS["lambda_city"] = dict(
    description="PAPER 1, λ dependence: city path, c ∈ {0, 0.2, 1}, λ ∈ {0.25, 0.75}, N = 64–32768. "
                "λ = 0, 0.5, 1 come from abl_c_lambda0, paper_centralization_sweep, abl_c_lambda1.",
    base=dict(CITY_BASE),
    grid=dict(N=N_GRID, centralization=C_SHORT, exclusion=[0.25, 0.75]),
    seeds=ABL_SEEDS_C,
)
# ε(ρ) at constant density for λ = 0 and λ = 1 (λ = 0.5 is density_window_scan): same densities,
# same N, same seeds 0–1, same physics apart from λ.
for _lam, _tag in ((0.0, "lam0"), (1.0, "lam1")):
    EXPERIMENTS[f"density_window_scan_{_tag}"] = dict(
        description=f"PAPER 1, λ dependence: density_window_scan with λ = {_lam:g}.",
        base=dict(P1, exclusion=_lam, coordination=False, centralization=0.0, area_exponent=1.0, N=4096),
        grid=dict(rho0=[0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
        seeds=2,
    )


def _to_py(v):
    """Convert numpy types to plain Python types (for JSON and stable hashing)."""
    if hasattr(v, "item"):
        return v.item()
    return v


def job_id(params_dict: dict, seed: int) -> str:
    blob = json.dumps({"p": params_dict, "seed": seed}, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def build_jobs(name: str, experiment_datetime: str, extra_base: dict | None = None, seeds: int | None = None,
               root: str | None = None) -> list:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment: {name}. Available: {sorted(EXPERIMENTS)}")
    exp = EXPERIMENTS[name]
    base = {**exp.get("base", {}), **(extra_base or {})}
    grid = exp["grid"]
    keys = list(grid)
    n_seeds = seeds if seeds is not None else exp.get("seeds", 1)
    s_off = int(exp.get("seed_offset", 0))     # independent seed blocks (E7); 0 = as before
    jobs = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = {**base, **{k: _to_py(v) for k, v in zip(keys, combo)}}
        # derived keys (not Params fields): resolved into real parameters here
        if "phi_over_xi" in kw:
            ratio = kw.pop("phi_over_xi")
            xi = kw.get("xi", Params().xi)
            kw["phi"] = round(ratio * xi, 6)
        if "route_moves" in kw:
            # job length in MOVES of a typically skilled agent (one same-task neighbour):
            # D = route_moves · w_ref,  w_ref = ((g + κ·1) / ρ_E)^b
            moves = kw.pop("route_moves")
            d0 = Params()
            g = kw.get("practice_gain", d0.practice_gain)
            k = kw.get("spillover", d0.spillover)
            rho_e = kw.get("exp_decay", d0.exp_decay)
            b = kw.get("learning_exponent", d0.learning_exponent)
            kw["route_length"] = round(moves * ((g + k * 1.0) / rho_e) ** b, 3)
        if "rho_at_Nref" in kw:
            # same density rho_ref at N = rho_ref_N (default N_REF) for every area exponent a:
            # density ≈ rho0 · N^(1−a)  ->  rho0 = rho_ref / rho_ref_N^(1−a)
            rho_ref = kw.pop("rho_at_Nref")
            n_ref = kw.pop("rho_ref_N", N_REF)
            a = kw.get("area_exponent", Params().area_exponent)
            kw["rho0"] = round(rho_ref / n_ref ** (1.0 - a), 8)
        p = Params(**kw)
        p.validate()
        pd = p.to_dict()
        for s in range(s_off, s_off + n_seeds):
            jobs.append({
                "job_id": job_id(pd, s),
                "experiment": name,
                "params": pd,
                "seed": s,
                "experiment_datetime": experiment_datetime,
                # keyword arguments: robust to the parameter order of paths.experiment_run_dir
                "run_dir": paths.experiment_run_dir(name, experiment_datetime=experiment_datetime,
                                                    p=p, seed=s, r=root),
                "record_timeseries": bool(exp.get("timeseries", False)),
                "snapshot_every": int(exp.get("snapshot_every", 0)),
            })
    return jobs


def estimated_cost(job: dict) -> float:
    """Relative cost ≈ steps × (N·m + m·L²). Used to launch the most expensive jobs first."""
    p = Params.from_dict(job["params"])
    return (p.T_burn + p.T_measure) * (p.N * (p.m + 1) + (p.m + 2) * p.L ** 2)