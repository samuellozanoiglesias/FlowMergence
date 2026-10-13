#!/bin/bash
# =============================================================================
# cuenca/run_paper.sh — ONLY the experiments needed for the final paper, with the final
# ("best") configuration applied explicitly (config.py: BEST = b 1, κ 8, R 4, 50-move jobs),
# followed by the analyses and the final figures (paper_figures.py).
#
# Start it in the background (from the FlowMergence code folder):
#     CPUS=40 nohup bash cuenca/run_paper.sh > run_paper.out 2>&1 &
#     tail -f run_paper.out
#
#  1 checks                      sanity (current defaults; stops everything if it fails)
#  2 paper_centralization_sweep  Fig 1–2: one knob c, city density path          (384 runs)
#  3 paper_constant_density      control: extensivity + pure organizational effect (200 runs)
#  4 paper_densification         Fig 4: β − 1 ≈ (1 − a)·ε                         (240 runs)
#  5 paper_coord_value           Fig 5: value of coordination, fixed bureaucracy   (216 runs)
#  6 paper_coord_value_endogenous Fig 5: emergent bureaucracy + comparison         (36 runs)
#  7 paper_gamma                 Fig 3 data: Γ collapse over steps 2–4 (analysis only)
#  8 paper_figures               all final figures + paper_numbers.txt (analysis only)
#  9 paper_choice_fixed          option B (only with INCLUDE_B=1): planner benchmark (324 runs)
# 10 paper_choice_sweep          option B: emergent centralization               (336 runs)
# 11 paper_choice_lockin         option B: lock-in                               (96 runs)
# 12 paper_figures_B             figures again, now including Fig 6 (option B)
#
# Environment variables (optional):
#     CPUS=40              CPU cores you may use  -> workers = CPUS − 2
#     WORKERS=38           or set the workers directly
#     INCLUDE_B=1          also run the option-B steps 9–12 with the best config
#     START_AT / ONLY / SKIP   step selection, as in run_all.sh
#     STOP_ON_CHECK_FAIL=0 continue even if checks.py fails
#     FORCE=1              run even if the pre-flight dry runs fail
# Logs: <data root>/logs/run_paper_<stamp>/   Figures: <data root>/paper_figures/<stamp>/
# =============================================================================
set -uo pipefail

# ------------------------------------------------------------------ setup
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$HERE")"
cd "$CODE_DIR" || exit 2
CONDA_ENV="${CONDA_ENV:-FlowMergence}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

NPROC=$(nproc)                       # cores the MACHINE has (not necessarily yours)
CPUS="${CPUS:-$NPROC}"               # cores YOU may use: set CPUS=40 on a shared machine
# one simulation per core, 2 cores left for the launcher process and the system
WORKERS="${WORKERS:-$(( CPUS > 3 ? CPUS - 2 : 1 ))}"
if (( WORKERS > NPROC )); then
    echo "WARNING: $WORKERS workers but only $NPROC cores on this machine (oversubscription)" >&2
fi
START_AT="${START_AT:-1}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
[[ "${INCLUDE_B:-0}" == "1" ]] || SKIP="$SKIP 9 10 11 12"   # option B only on request
STOP_ON_CHECK_FAIL="${STOP_ON_CHECK_FAIL:-1}"
SET_ARGS=()
[[ -n "${EXTRA_SET:-}" ]] && { read -r -a _s <<< "$EXTRA_SET"; SET_ARGS=(--set "${_s[@]}"); }

DATA="${FLOWMERGENCE_DATA:-$(python -c "import paths; print(paths.root() if hasattr(paths, 'root') else paths.DATA_ROOT)")}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$DATA/logs/run_paper_$STAMP"

# ------------------------------------------------------------------ the steps (RUN_GUIDE.md)
# format: "name|description|command ; command ; ..."   (%W = workers, %S = --set args)
STEPS=(
"checks|sanity checks|python checks.py"
"paper_centralization_sweep|Fig 1-2: one knob c on the city density path|L paper_centralization_sweep ; A paper_centralization_sweep --by centralization"
"paper_constant_density|control: extensivity at constant density|L paper_constant_density ; A paper_constant_density --by centralization"
"paper_densification|Fig 4: densification rate x centralization|L paper_densification ; A paper_densification --by area_exponent centralization"
"paper_coord_value|Fig 5: value of coordination (fixed bureaucracy)|L paper_coord_value"
"paper_coord_value_endogenous|Fig 5: emergent bureaucracy + comparison|L paper_coord_value_endogenous ; B paper_coord_value paper_coord_value_endogenous"
"paper_gamma|Fig 3 data: Gamma collapse + beta decomposition|G paper_centralization_sweep paper_constant_density paper_densification"
"paper_figures|final figures + paper_numbers.txt|python paper_figures.py"
#"paper_choice_fixed|option B: planner benchmark + baseline|L paper_choice_fixed"
#"paper_choice_sweep|option B: emergent centralization|L paper_choice_sweep ; C --choice paper_choice_sweep --fixed paper_choice_fixed ; G paper_choice_sweep --baseline paper_choice_fixed"
#"paper_choice_lockin|option B: lock-in|L paper_choice_lockin ; C --choice paper_choice_sweep --fixed paper_choice_fixed --lockin paper_choice_lockin"
#"paper_figures_B|final figures including option B|python paper_figures.py"
)

if [[ "${1:-}" == "--list" ]]; then
    for i in "${!STEPS[@]}"; do IFS='|' read -r n d _ <<< "${STEPS[$i]}"; printf "%2d  %-34s %s\n" $((i+1)) "$n" "$d"; done
    exit 0
fi

mkdir -p "$LOGDIR" || { echo "ERROR: cannot create $LOGDIR" >&2; exit 2; }
SUMMARY="$LOGDIR/SUMMARY.txt"

# ------------------------------------------------------------------ helpers
ts () { date '+%F %T'; }
say () { echo "[$(ts)] $*"; }
L () { python launcher.py --exp "$1" --workers "$WORKERS" ${SET_ARGS[@]+"${SET_ARGS[@]}"}; }          # launch a whole grid
A () { local e="$1"; shift; python analysis.py "$e" "$@"; }                            # scaling analysis
B () { python analysis_bureaucracy.py --fixed "$1" --endo "$2"; }                      # bureaucracy analysis
T () { python analysis_transition.py "$1"; }                                           # transition analysis
G () { python analysis_gamma.py "$@"; }                                                # Γ collapse (option A)
C () { python analysis_choice.py "$@"; }                                               # emergent vs planner (option B)
COMBINE () {   # COMBINE <new name> <exp1> <exp2> ... : newest launch of each -> experiments/<new name>/results.jsonl
    local new="$1"; shift; local dst="$DATA/experiments/$new"; mkdir -p "$dst"; : > "$dst/results.jsonl"
    local e d
    for e in "$@"; do
        d=$(ls -1dt "$DATA/experiments/$e"/exp_*/ 2>/dev/null | head -n 1)
        [[ -n "$d" && -f "$d/results.jsonl" ]] || { echo "COMBINE: no results for $e"; return 1; }
        cat "$d/results.jsonl" >> "$dst/results.jsonl"; echo "COMBINE: + $d"
    done
}
BPATH () { python analysis_bureaucracy.py --fixed "$DATA/experiments/$1/results.jsonl" --endo "$2"; }
PLOT_EXAMPLES () {                                                                     # time-series plots of 1a
    local latest; latest=$(ls -1dt "$DATA/experiments/centralization_examples"/exp_*/ 2>/dev/null | head -n 1)
    [[ -z "$latest" ]] && { echo "no launch folder found"; return 1; }
    local n=0
    while IFS= read -r f; do python plots.py "$(dirname "$f")" && n=$((n+1)); done \
        < <(find "$latest" -name result.json | sort)
    echo "time-series plots written: $n (timeseries.png inside each run folder of $latest)"
}
selected () {   # is step number $1 selected by START_AT / ONLY / SKIP ?
    local k="$1"
    [[ "$k" -lt "$START_AT" ]] && return 1
    [[ -n "$ONLY" && " $ONLY " != *" $k "* ]] && return 1
    [[ -n "$SKIP" && " $SKIP " == *" $k "* ]] && return 1
    return 0
}
fmt_dur () { local s=$1; printf "%dh%02dm%02ds" $((s/3600)) $((s%3600/60)) $((s%60)); }

declare -a RES_NUM=() RES_NAME=() RES_STATUS=() RES_TIME=()
write_summary () {
    {
        echo "FlowMergence run_paper — started $START_TS on $(hostname)"
        echo "code: $CODE_DIR   data: $DATA   workers: $WORKERS   extra --set: ${EXTRA_SET:-none}"
        echo
        printf "%-4s %-34s %-9s %s\n" "step" "name" "status" "duration"
        for i in "${!RES_NAME[@]}"; do printf "%-4s %-34s %-9s %s\n" "${RES_NUM[$i]}" "${RES_NAME[$i]}" "${RES_STATUS[$i]}" "${RES_TIME[$i]}"; done
    } > "$SUMMARY"
}
trap 'say "INTERRUPTED (signal) during step ${CURRENT:-?}"; RES_NUM+=("${CURRENT_NUM:-?}"); RES_NAME+=("${CURRENT:-?}"); RES_STATUS+=("KILLED"); RES_TIME+=("-"); write_summary; exit 130' INT TERM

# ------------------------------------------------------------------ banner + dry runs
START_TS="$(ts)"; T0=$(date +%s)
say "FlowMergence run_paper on $(hostname): machine has $NPROC cores, using CPUS=$CPUS -> $WORKERS workers"
say "code folder: $CODE_DIR"
say "data root  : $DATA"
say "logs       : $LOGDIR"
say "python     : $(command -v python)  (conda env: ${CONDA_DEFAULT_ENV:-none})"
say "dry runs (job counts and relative cost) — also a pre-flight check of the installed code:"
PREFLIGHT_FAIL=0
for i in "${!STEPS[@]}"; do
    k=$((i+1)); selected "$k" || continue
    IFS='|' read -r name _ cmds <<< "${STEPS[$i]}"
    for e in $(grep -o 'L [a-z_]*' <<< "$cmds" | awk '{print $2}'); do
        if dry=$(python launcher.py --exp "$e" --dry-run ${SET_ARGS[@]+"${SET_ARGS[@]}"} 2>&1); then
            grep "total=" <<< "$dry" | sed "s/^/    /"
        else
            PREFLIGHT_FAIL=1
            say "PRE-FLIGHT FAILED for experiment '$e' (step $k):"
            tail -n 4 <<< "$dry" | sed 's/^/      | /'
        fi
    done
done
if [[ $PREFLIGHT_FAIL -eq 1 && "${FORCE:-0}" != "1" ]]; then
    say "Stopping before running anything: the installed code does not match config.py"
    say "(usually an old params.py / dynamics.py / metrics.py). Install the missing files, or set FORCE=1."
    exit 1
fi

# ------------------------------------------------------------------ run
for i in "${!STEPS[@]}"; do
    k=$((i+1))
    IFS='|' read -r name desc cmds <<< "${STEPS[$i]}"
    selected "$k" || { say "skip  $k $name"; continue; }
    CURRENT="$name"; CURRENT_NUM="$k"
    LOG="$LOGDIR/$(printf '%02d' "$k")_${name}.log"
    say "START $k/${#STEPS[@]}  $name — $desc"
    say "      log: $LOG"
    t1=$(date +%s); status="OK"
    IFS=';' read -r -a parts <<< "$cmds"
    for c in "${parts[@]}"; do
        c="$(echo "$c" | sed 's/^ *//; s/ *$//')"
        [[ -z "$c" ]] && continue
        echo "[$(ts)] \$ $c" >> "$LOG"
        # shellcheck disable=SC2086
        if ! eval "$c" >> "$LOG" 2>&1; then status="FAILED"; echo "[$(ts)] command failed: $c" >> "$LOG"; break; fi
    done
    dt=$(( $(date +%s) - t1 ))
    RES_NUM+=("$k"); RES_NAME+=("$name"); RES_STATUS+=("$status"); RES_TIME+=("$(fmt_dur $dt)")
    write_summary
    say "END   $k $name: $status in $(fmt_dur $dt)"
    if [[ "$status" != "OK" ]]; then
        say "      last lines of the log:"; tail -n 15 "$LOG" | sed 's/^/      | /'
        if [[ "$name" == "checks" && "$STOP_ON_CHECK_FAIL" == "1" ]]; then
            say "checks.py failed: stopping (everything else depends on it). Set STOP_ON_CHECK_FAIL=0 to ignore."
            write_summary; exit 1
        fi
    fi
done

say "ALL DONE in $(fmt_dur $(( $(date +%s) - T0 )))"
echo; cat "$SUMMARY"
