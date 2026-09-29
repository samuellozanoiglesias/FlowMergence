"""
params.py — Every tunable variable of the model.

Each field states its meaning and its ORIGIN:
  [B96]  Bonabeau, Theraulaz & Deneubourg 1996 (fixed response thresholds)
  [T98]  Theraulaz, Bonabeau & Deneubourg 1998 (reinforced thresholds)
  [BML]  Biham, Middleton & Levine 1992 (2D traffic with two species)
  [NEW]  added in this study (not in the papers)

Defaults are a reasonable starting point, NOT a calibration. The first job of the
study is to explore parameter space (see config.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Params:
    # ------------------------------------------------------------------ population and space
    N: int = 1024
    """Total number of agents (workers + coordinators + idle). Coordinators COUNT in N,
    just as managers count as employees."""

    area_exponent: float = 0.85
    """[NEW] a: lattice area scales as A = N**a / rho0.
    a = 1 -> constant density: CONTROL, no mechanism can make β ≠ 1.
    a < 1 -> density grows as N**(1-a), as in cities (sublinear land area).
    This is the ONLY way N enters the local physics: more density means more
    spillovers (benefit) and more blocking (cost)."""

    rho0: float = 0.05
    """Reference density: A = N**a / rho0."""

    L_min: int = 8
    """Minimum lattice side."""

    # ------------------------------------------------------------------ tasks
    m: int = 4
    """Number of productive tasks (routes). Task j moves in direction j % 2
    (0 = horizontal/right, 1 = vertical/up), like the two species of [BML]."""

    route_length: float = 200.0
    """[NEW] Work needed to complete one job (route units). An agent with efficiency w
    advances w units per successful move ("it learns the way")."""

    route_length_spread: float = 0.0
    """Heterogeneous lengths: D_j = D·(1 + spread·(j/(m-1) − 0.5)). Must be < 2."""

    demand_skew: float = 0.0
    """Demand composition: π_j ∝ exp(−skew·j). 0 = uniform demand."""

    # ------------------------------------------------------------------ traffic substrate
    exclusion: float = 1.0
    """λ, MAIN CONTROL PARAMETER. Probability that an agent is blocked when trying to
    enter a cell occupied by an agent of the OTHER direction.
    λ = 1 -> hard exclusion, as in [BML] Model I (abrupt jamming transition).
    λ -> 0 -> flows can cross, similar to [BML] Model III (continuous transition).
    Exclusion within the same lane is always hard."""

    traffic_light: bool = True
    """[BML] True = Model I: horizontal agents move on even steps, vertical agents on odd
    steps (no target conflicts). False = both directions move every step and crossing
    is evaluated against the occupancy at the start of the step."""

    # ------------------------------------------------------------------ learning by doing
    practice_gain: float = 1.0
    """[NEW] g: experience gained per step spent working on the task."""

    spillover: float = 2.0
    """[NEW] κ: extra experience per OTHER agent doing the same task within the
    spillover radius. This is the source of SUPERLINEARITY (the analogue of social
    interactions in cities, West/Bettencourt)."""

    spillover_radius: int = 2
    """[NEW] R: Chebyshev radius of the spillover neighbourhood, (2R+1)^2 cells."""

    exp_decay: float = 0.002
    """[NEW] ρ: per-step forgetting of experience, e <- (1-ρ)e. Bounds e ≲ (g+κn)/ρ.
    The memory (~1/ρ = 500 steps) must exceed the duration of a job; otherwise
    specializing barely pays off."""

    learning_exponent: float = 0.5
    """[NEW] b: efficiency w = (1+e)**b (Wright's law). b = 0 removes the payoff of
    specializing (ablation)."""

    w_max: float = 200.0
    """[NEW] Efficiency cap. Too low a cap kills superlinearity."""

    # ------------------------------------------------------------------ response thresholds
    theta_min: float = 1.0
    """[T98] Lower bound of the threshold."""
    theta_max: float = 4000.0
    """[T98] Upper bound of the threshold. HIGHER is more selective: at the ceiling a
    specialist takes an unwanted task with probability s²/(s²+θ_max²) (4% at θ_max=1000,
    20% at 400 for s=200). Calibration: DOL 0.07 -> 0.20 -> 0.42 for θ_max 400/600/1000."""
    theta_init: float = 200.0
    """[T98] Initial threshold (identical agents at start). Keep it ≈ s_init: starting on
    the flat part of the response curve (e.g. 500) traps agents in the DEAF regime,
    because thresholds reach the ceiling before any preference can form."""
    theta_init_noise: float = 0.0
    """[T98 fig. 1c] Initial heterogeneity: uniform noise of ±noise·(θmax−θmin)."""
    xi: float = 3.0
    """[T98] ξ: learning; θ drops by ξ per step spent doing the task.
    At fixed φ/ξ, SMALLER ξ gives cleaner specialization (long memory: one unlucky job
    cannot erase a specialty) but slower convergence; check with time series that it
    settles within T_burn. Calibration (φ/ξ=0.5, θ_max=1000): DOL 0.42, 0.40, 0.21 for
    ξ = 1, 3, 10."""
    phi: float = 1.5
    """[T98] φ: forgetting; θ rises by φ per step NOT doing the task.
    THE MASTER KNOB IS THE RATIO φ/ξ. A task done a fraction f of the active time
    becomes a specialty if f > f* = φ/(ξ+φ). Specialization needs a WINDOW:
        1/m  <  f*  <  (share a specialist can actually devote to its task)
    too low  (φ/ξ ≲ 0.2): generalists, every task is "practised enough";
    too high (φ/ξ ≳ 1.5): deaf, no task can hold its share, all thresholds hit θ_max.
    Fine calibration (calibration_fine, N=1024, λ=0.5, 864 runs): with ξ=3, θ_max=4000,
    DOL ≥ 0.84 for 0.4 ≤ φ/ξ ≤ 1.0; output per agent peaks at φ/ξ = 0.4–0.5.
    Chosen default: ξ=3, φ=1.5 (φ/ξ=0.5), θ_max=4000, θ_init=200 -> DOL 0.916 ± 0.002.
    ξ=0.3 did not converge within T_burn (identical results for every θ_max)."""

    # ------------------------------------------------------------------ stimuli
    s_init: float = 200.0
    """BASE stimulus of each task (its value when production composition is balanced)."""
    s_max: float = 1000.0
    """Stimulus cap."""
    s_gain: float = 2.0
    """[NEW] δ: proportional gain, s_j = s_base·(1 + δ·(π_j − share_j)/π_j).
    With δ=2, a task 25% below its demanded share gets 1.5·s_base.
    Replaces the [B96]/[T98] demand equation (∂s = δ − α/N Σx), which normalizes by N
    and therefore CAPS total output at ~δN/α, forcing β = 1 by construction."""
    share_ema: float = 0.02
    """Exponential smoothing of the observed production composition."""
    p_quit_work: float = 0.0
    """[B96] p: per-step probability of abandoning an unfinished job (0 = never)."""
    encounter: str = "stimulus"
    """How an idle agent picks the ONE role it considers this step (it then accepts with
    probability T(s, θ)).
    'stimulus' (default): role k is encountered with probability ∝ s_k. A silent role
       (s = 0, e.g. coordination when there is no congestion) is never encountered, so
       its mere existence costs nothing. Closer to B96, where the stimulus sets "the
       probability of being exposed" to a task.
    'uniform': every role equally likely. Legacy behaviour: with coordination enabled
       but unused, 1/(m+1) of all encounters are wasted (−6–15% output in density_scan)."""

    # ------------------------------------------------------------------ coordination (bureaucracy)
    coordination: bool = True
    """[NEW] If False, the coordinator role does not exist (ablation)."""
    coord_radius: int = 3
    """[NEW] Chebyshev radius of the zone a coordinator covers ("span of control")."""
    coord_strength: float = 0.5
    """[NEW] q: each coordinator multiplies the crossing-block probability by (1−q).
    q = 0 -> a bureaucracy that does not help at all (pure overhead control)."""
    coord_conflict: float = 0.0
    """[NEW] η: contradictory orders when zones overlap:
    b(k) = λ[(1−q)^k + η·max(k−1,0)]."""
    coord_stimulus_gain: float = 1000.0
    """[NEW] g_C: coordination stimulus s_C = min(g_C · signal_rate, s_max)."""
    coord_signal: str = "cross"
    """'cross' = the signal is crossing blocks (what a coordinator can fix).
    'all' = any block (a bureaucracy reacting to problems it cannot solve)."""
    p_quit_coord: float = 0.01
    """[B96] p for coordinators: per-step probability of leaving the role."""
    coord_placement: str = "congestion"
    """'congestion' = a new coordinator is placed where a signal event happened this
    step; 'random' = at a random cell."""
    coord_relocate_prob: float = 0.0
    """[NEW] Per-step probability that an INACTIVE coordinator moves to a congested
    spot. 0 = rigid bureaucracy."""
    blocked_ema: float = 0.05
    """Smoothing of the blocking rates."""

    # ------------------------------------------------------------------ time
    T_burn: int = 6000
    """Transient steps. With φ = 0.1 a threshold needs ~5000 steps to rise from 500 to
    1000: check with time series (runner --timeseries) before trusting results."""
    T_measure: int = 2000
    """Measurement steps."""
    record_every: int = 10
    """Averaging window of the time series, if recorded."""

    # ------------------------------------------------------------------ execution
    backend: str = "numpy"
    """'numpy' (CPU) or 'cupy' (GPU; only worth it for N ≳ 1e5)."""

    # ================================================================== derived quantities
    @property
    def L(self) -> int:
        area = (self.N ** self.area_exponent) / self.rho0
        return max(self.L_min, int(math.ceil(math.sqrt(area))))

    @property
    def density(self) -> float:
        return self.N / float(self.L ** 2)

    def route_lengths(self) -> list:
        if self.m == 1:
            return [float(self.route_length)]
        return [float(self.route_length * (1.0 + self.route_length_spread * (j / (self.m - 1) - 0.5)))
                for j in range(self.m)]

    def demand_shares(self) -> list:
        w = [math.exp(-self.demand_skew * j) for j in range(self.m)]
        s = sum(w)
        return [x / s for x in w]

    # ================================================================== utilities
    def validate(self) -> list:
        errs = []
        if self.N < 2:
            errs.append("N must be >= 2")
        if self.m < 1:
            errs.append("m must be >= 1")
        if not 0.0 <= self.exclusion <= 1.0:
            errs.append("exclusion must be in [0,1]")
        if not 0.0 <= self.coord_strength <= 1.0:
            errs.append("coord_strength must be in [0,1]")
        if self.coord_conflict < 0:
            errs.append("coord_conflict must be >= 0")
        if not (0 < self.theta_min < self.theta_max):
            errs.append("require 0 < theta_min < theta_max")
        if not (self.theta_min <= self.theta_init <= self.theta_max):
            errs.append("theta_init outside [theta_min, theta_max]")
        if not (0 <= self.route_length_spread < 2):
            errs.append("route_length_spread must be in [0,2)")
        if self.coord_placement not in ("congestion", "random"):
            errs.append("coord_placement ∈ {congestion, random}")
        if self.coord_signal not in ("cross", "all"):
            errs.append("coord_signal ∈ {cross, all}")
        if self.encounter not in ("stimulus", "uniform"):
            errs.append("encounter ∈ {stimulus, uniform}")
        if self.backend not in ("numpy", "cupy"):
            errs.append("backend ∈ {numpy, cupy}")
        if self.T_measure < 1 or self.T_burn < 0:
            errs.append("need T_measure >= 1 and T_burn >= 0")
        if not 0 <= self.exp_decay < 1:
            errs.append("exp_decay must be in [0,1)")
        if errs:
            raise ValueError("; ".join(errs))
        warns = []
        if self.density > 1.5:
            warns.append(f"total density {self.density:.2f} > 1.5: many agents will not be able to enter "
                         "the lattice (2 layers => at most 2 per cell).")
        if 2 * self.coord_radius + 1 > self.L:
            warns.append("a coordinator zone is larger than the lattice; the radius will be clipped.")
        for w in warns:
            warnings.warn(w)
        return warns

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            warnings.warn(f"ignoring unknown keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


DEFAULTS = Params()


def coerce_value(name: str, value: str):
    """Convert a command-line string to the type of field `name`."""
    if not hasattr(DEFAULTS, name):
        raise KeyError(f"unknown parameter: {name}")
    default = getattr(DEFAULTS, name)
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


def parse_overrides(pairs) -> dict:
    """['N=512', 'exclusion=0.3'] -> {'N': 512, 'exclusion': 0.3}"""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"malformed override (use key=value): {pair}")
        k, v = pair.split("=", 1)
        out[k.strip()] = coerce_value(k.strip(), v)
    return out