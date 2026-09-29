# FlowMergence — specialization versus coordination: from a "city" regime (β > 1) to a "company" regime (β < 1)

A minimal agent-based model to study whether total output scales as `Y ∝ N^β` with
β > 1 or β < 1 depending on how costly coordination is, and how the bureaucracy that
emerges scales (`M ∝ N^γ`).

## The idea in one paragraph

Identical agents specialize through **threshold reinforcement** (Theraulaz et al. 1998).
They become more efficient by **learning by doing** and from their neighbours
(spillover). They work on a **BML traffic substrate** (Biham et al. 1992) whose density
grows with N. That density produces both a benefit (spillovers, pushing towards β > 1)
and a cost (blocking, pushing towards β < 1). The parameter **λ (crossing exclusion)**
tunes the cost. When congestion appears, some agents become **coordinators**: they do
not produce, but they reduce blocking in their zone. How many there are is not imposed;
it emerges.

## Where results are stored

Everything goes under **`/data/samuel_lozano/FlowMergence/`**. You can change this with
the `FLOWMERGENCE_DATA` environment variable or `--root` on any command.

```
/data/samuel_lozano/FlowMergence/
├── experiments/<experiment>/
│   ├── config.txt          experiment definition, overrides, machine info
│   ├── manifest.json
│   ├── results.jsonl       one line per finished run (used by analysis.py)
│   ├── errors.jsonl
│   ├── runs/<run name>/seedXXX/
│   │   ├── config.txt      full parameter list of this run (* = non-default)
│   │   ├── result.json     summary (+ time series if recorded)
│   │   ├── snapshots.npz   if recorded
│   │   └── error.txt       only on failure
│   └── analysis/by_<columns>/   figures, CSV tables, analysis_config.txt
├── single_runs/<timestamp>_<run name>_seedXXX/   runs launched with runner.py
└── checks/<timestamp>/                           report.txt + runs used by checks.py
```

A **run name** always starts with the core parameters and then lists every parameter
that differs from its default, abbreviated, for example
`N0001024_lam0.5_a0.85_kap2_coordon_xi0.5_phi0.05`. Two runs that differ in any
parameter never share a folder. The abbreviations are listed in `paths.py`, and the
full, unabbreviated list is always in each run's `config.txt`.

## What comes from each paper and what is new

| Component | Origin |
|---|---|
| `T(s,θ) = s²/(s²+θ²)`, quitting with probability p | Bonabeau et al. 1996 |
| `θ -= ξ` when performing the task, `θ += φ` otherwise | Theraulaz et al. 1998 |
| Two species (→, ↑), exclusion, even/odd traffic light | BML 1992, Model I |
| Weakened exclusion between species (λ < 1) | inspired by BML Model III |
| Efficiency `w = (1+e)^b`, experience with forgetting and spillover | **new** (T98 suggests it but keeps α fixed) |
| Composition stimulus (not normalized by N) | **new** (replaces `∂s = δ − α/N Σx`) |
| Area `A ∝ N^a` (growing density) | **new**, inspired by West/Bettencourt |
| Coordination as one more task, zones, active/inactive | **new** |

**Why the B96/T98 demand equation is not used.** In those papers demand is scaled by N
on purpose ("we expect results to be largely independent of colony size", B96). That
caps total output at ~δN/α and forces β = 1 by construction. Here the stimulus only
**allocates** work across tasks and does not cap the total.

**Time scale.** T98 uses ξ = 10, φ = 1 with tasks lasting ~5 steps. Here jobs last 40–400
steps, so with those values a single job would wipe the whole threshold profile. The
defaults keep T98's ratio ξ/φ = 10 with values ten times smaller (ξ = 1, φ = 0.1). The
window where specialization emerges must be recalibrated (`calibration_xi_phi`).

## Equations (one step, Δt = 1)

- **Movement**: a worker on task j moves in direction j mod 2. It is blocked if the target
  cell in its own lane is occupied. If the target is occupied in the other lane, it is
  blocked with probability `b(k) = λ[(1−q)^k + η·max(k−1,0)]`, where k is the coordinator
  coverage of that cell.
- **Progress**: each successful move adds `w_ij` to the job's progress; the job ends when
  progress ≥ D_j. Output Y is the sum of D_j over completed jobs.
- **Experience**: `e ← (1−ρ)e + [works on j](g + κ·n_ij)`, with n_ij the number of peers
  on the same task within radius R.
- **Stimuli**: `s_j = clip(s_base·(1 + δ(π_j − share_j)/π_j))` (proportional; a pure
  integrator oscillated), and `s_C = g_C · crossing-block rate`.
- **Recruitment**: each idle agent encounters a random role (m tasks + coordinating) and
  takes it with probability `T(s, θ)`.
- **Thresholds**: T98 reinforcement using the role performed during the step.

Cost: **O(N·m + m·L²) per step**, and since `L² ∝ N^a` with a ≤ 1, the cost is linear in
N. Nothing is exponential or quadratic in N.

## Files

| File | Role |
|---|---|
| `params.py` | **Tunable variables**, documented with their origin |
| `paths.py` | **Storage layout**, run-folder naming, `config.txt` writers |
| `mechanism.py` | **Mechanism**: microscopic rules (pure functions) |
| `traffic.py` | BML substrate: movement, entry, exit, spillover counts |
| `bureaucracy.py` | **Bureaucracy**: coverage, active/inactive, placement |
| `dynamics.py` | **Dynamics**: state and simulation step |
| `grid.py` | O(L²) periodic box sums |
| `metrics.py` | Observables, Gorelick DOL index, summaries |
| `runner.py` | **Global runner**: one run (also from the command line) |
| `config.py` | **Experiment configuration** (sweeps) |
| `launcher.py` | Parallel **launcher** with resumption |
| `analysis.py` | β and γ fits with bootstrap, local exponents, calibration maps |
| `plots.py` | Figures |
| `visualize.py` | GIF animation of the lattice |
| `checks.py` | Sanity checks |
| `backend.py` | NumPy (CPU) or CuPy (GPU) |

Dependencies: `numpy`, `pandas`, `matplotlib`, `pillow` (for GIFs). Optional: `cupy>=12`.

## Recommended workflow

```bash
cd /home/samuel_lozano/FlowMergence
python checks.py                                          # 1. sanity (minutes)
python launcher.py --exp smoke --workers 16               # 2. whole pipeline end to end
python analysis.py smoke --by exclusion

# 3. calibrate the specialization window (ξ, φ)
python launcher.py --exp calibration_xi_phi --workers 190
python analysis.py calibration_xi_phi --by xi phi         # -> map_DOL_tasks.png

# 4. inspect one run: does specialization settle before T_burn?
python runner.py --set N=4096 exclusion=0.8 --timeseries --snapshots 25
python plots.py     /data/samuel_lozano/FlowMergence/single_runs/<folder>
python visualize.py /data/samuel_lozano/FlowMergence/single_runs/<folder>

# 5. control: β ≈ 1 is expected
python launcher.py --exp control_density --workers 190
python analysis.py control_density --by exclusion

# 6. main experiment and ablation
python launcher.py --exp lambda_sweep --workers 190
python launcher.py --exp lambda_sweep_nocoord --workers 190
python analysis.py lambda_sweep --by exclusion

# 7. phase diagram (benefit κ versus cost λ)
python launcher.py --exp phase_diagram --workers 190
python analysis.py phase_diagram --by spillover exclusion

# 8. large N on GPU
python launcher.py --exp large_N --backend cupy --workers 8
```

If the calibration shows a better (ξ, φ), pass it to every later experiment with
`--set xi=... phi=...`. It will appear in the run names and in every `config.txt`.

**CPU or GPU.** With N ≤ 3·10⁴ the efficient choice is one NumPy run per core (190
workers). The GPU only pays off when a single run is large (N ≳ 10⁵); in that case use
few workers and `--min-N` / `--max-N` to split the jobs between CPU and GPU.

## What to look at and what to expect (hypotheses, not results)

1. **Control (a = 1)**: β ≈ 1. If not, there is a bug or a finite-size effect.
2. **`lambda_sweep`**: β(λ) should decrease as λ grows. The interesting questions are
   whether it crosses β = 1, where, and whether the crossover is smooth or abrupt. BML
   Model I has an abrupt jamming transition and Model III a continuous one, so the
   shape of β(λ) is not decided in advance.
3. **`local_beta.png`**: the exponent may change with N (a regime crossover). This is the
   analogue of the transition from sublinear to linear expenses that West describes for
   companies.
4. **Bureaucracy**: γ > 1 would mean administration grows faster than the workforce. A high
   `bureau_inactive_frac` signals ossification.
5. **Ablations**: no coordination, useless coordinators (q = 0) and no payoff from
   specializing (b = 0) separate the contribution of each mechanism.

## Known limitations

- Coordinators appear **in response to a stimulus** (there is congestion), not because it
  is globally optimal, as in social insects. This is a modelling hypothesis worth stating.
- Defaults are NOT calibrated. Check T_burn (has specialization converged?) and the
  (ξ, φ) window where specialization exists.
- ξ and φ are PER-STEP rates: they must be small relative to the duration of a job.
- Three decades of N are the minimum to talk about exponents; use them with care.
- Exponents are fitted over a finite range of N, as with real data on cities and companies.
