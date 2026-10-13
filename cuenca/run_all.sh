#!/bin/bash
# =============================================================================
# cuenca/run_all.sh — PAPER 1 (scaling) campaign: tests, cost calibration, analysis of the
# existing data, then every new experiment in order of priority, with analyses in between.
#
# Start it in the background (from the FlowMergence code folder):
#     nohup bash cuenca/run_all.sh > run_all.out 2>&1 &
#     tail -f run_all.out                      # overall progress
#     bash cuenca/run_all.sh --list            # numbered steps
#
# Per-step logs:  <data root>/logs/run_all/run_all_<stamp>/NN_<step>.log
# Summary:        <data root>/logs/run_all/run_all_<stamp>/SUMMARY.txt (updated after every step)
# Only ONE run_all.sh can run at a time (lock file <data root>/logs/run_all/.lock); a second one
# exits at once with a message.
# Figures/numbers: <data root>/paper_figures/paper1_<stage>/  (paper_numbers.txt, paper_macros.tex,
#                  paper_switches.tex, figures, tables/), one folder per analysis stage.
#                  The last step also packs paper1_final into paper1_final_<stamp>.tar.gz.
#
# Environment variables (optional):
#     CPUS=40              CPU cores you are allowed to use [nproc]
#     WORKERS=38           parallel simulations            [CPUS − 2]
#     START_AT=5           skip steps before this number   [1]
#     ONLY="7 9"           run only these step numbers
#     SKIP="18"            skip these step numbers
#     NMIN=256             main N range of Fig. 2 (E1 decision rule)          [64]
#     POLY_READY=1         run poly_main: set route_moves in config.py (poly_main AND
#                          poly_main_base) from the pilot first               [0 = wait]
#     STOP_ON_TEST_FAIL=0  continue even if the tests fail [1 = stop]
#     EXTRA_SET="..."      --set overrides added to EVERY launch. NOT for Paper 1: it changes the
#                          physics of every experiment (all parameters are pinned in config.py)
#     CONDA_ENV=FlowMergence
# A failed step is logged and the script goes on with the next one (except the tests and the
# seed check). Relaunching a step starts a NEW launch folder (no resumption); analyses always
# read the newest launch of each experiment.
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

NPROC=$(nproc --all 2>/dev/null || nproc)   # cores the MACHINE has (nproc alone can report 1 under nohup)
CPUS="${CPUS:-$NPROC}"               # cores YOU may use: set CPUS=40 on a shared machine
# one simulation per core, 2 cores left for the launcher process and the system
WORKERS="${WORKERS:-$(( CPUS > 3 ? CPUS - 2 : 1 ))}"
if (( WORKERS > NPROC )); then
    echo "WARNING: $WORKERS workers but only $NPROC cores on this machine (oversubscription)" >&2
fi
START_AT="${START_AT:-1}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
NMIN="${NMIN:-64}"
POLY_READY="${POLY_READY:-0}"
STOP_ON_TEST_FAIL="${STOP_ON_TEST_FAIL:-1}"
SET_ARGS=()
if [[ -n "${EXTRA_SET:-}" ]]; then
    read -r -a _s <<< "$EXTRA_SET"; SET_ARGS=(--set "${_s[@]}")
    echo "WARNING: EXTRA_SET='$EXTRA_SET' changes EVERY launch; Paper 1 pins all parameters in config.py" >&2
fi

DATA="${FLOWMERGENCE_DATA:-$(python -c "import paths; print(paths.root())")}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$DATA/logs/run_all/run_all_$STAMP"
FIGROOT="$DATA/paper_figures"

# ------------------------------------------------------------------ the steps
# format: "name|description|command ; command ; ..."
#   L <exp>      launch the whole grid            PF <stage> [args]  paper_figures.py -> paper1_<stage>
#   TESTS        Paper 1 unit tests               CALIB              hours per experiment
#   FIG1 <stage> Fig. 1 (schematic)               SEEDCHECK <exp>    1 cheap run, check seed_offset
#   POLYGUARD    wait for POLY_READY=1            PACK               tar.gz of paper1_final
# Priority order: decisive and cheap first; the most expensive (seed block) and the run that
# needs a human choice (poly_main) last.
STEPS=(
"tests|Unit tests of the Paper 1 changes (stop if they fail)|TESTS"
"calibrate|Cost calibration: hours per experiment at \$WORKERS workers|CALIB"
"analysis_existing|Analysis of the EXISTING data: E1, E2, E5, E6 (Fig. 3), E11, Fig. 1|PF existing ; FIG1 existing"
"deliver_learning_check|E5b: no learning while delivering (decides the p < 1 explanation)|L deliver_learning_check"
"transient_check_pl|E6: Fig. 2 cells with doubled transient|L transient_check_pl"
"abl_pl_kappa0|E3: Fig. 2 short, no spillovers|L abl_pl_kappa0"
"abl_pl_nodol|E3: Fig. 2 short, no division of labour|L abl_pl_nodol"
"abl_pl_lambda0|E3: Fig. 2 short, crossings never block|L abl_pl_lambda0"
"abl_pl_b0|E3: Fig. 2 short, no learning|L abl_pl_b0"
"abl_c_kappa0|E3: Fig. 3 short, no spillovers|L abl_c_kappa0"
"abl_c_nodol|E3: Fig. 3 short, no division of labour|L abl_c_nodol"
"abl_c_lambda0|E3: Fig. 3 short, crossings never block (to N = 131072)|L abl_c_lambda0"
"abl_c_b0|E3: Fig. 3 short, no learning|L abl_c_b0"
"paper_density_exponent_b075|E4: Fig. 2 with b = 0.75, kappa = 4, R = 2 in its own window|L paper_density_exponent_b075"
"poly_pilot|E9 pilot: job length for polycentric delivery (table in the log)|L poly_pilot ; L poly_pilot_base ; PF pilot --only poly"
"analysis_mid|Analysis after the decisive experiments (DECISION lines)|PF mid"
"paper_density_exponent_seedsB|E7: independent seed block 100-104 (most expensive step)|SEEDCHECK paper_density_exponent_seedsB ; L paper_density_exponent_seedsB"
"abl_c_lambda1|E12 (optional): Fig. 3 short with hard crossings|L abl_c_lambda1"
"poly_main|E9 main grid (needs route_moves from the pilot and POLY_READY=1)|POLYGUARD ; L poly_main ; L poly_main_base"
"lambda_city|λ dependence on the city path: λ = 0.25, 0.75 (0, 0.5, 1 already exist)|L lambda_city"
"lambda_density|λ dependence at constant density: ε(ρ) for λ = 0 and λ = 1|L density_window_scan_lam0 ; L density_window_scan_lam1"
"analysis_final|Final figures, numbers, tables and switches (+ .tar.gz)|PF final ; FIG1 final ; PACK"
)
# Previous campaign (superlinear_tuning … choice_with_coordination): see git history of this file.

if [[ "${1:-}" == "--list" ]]; then
    for i in "${!STEPS[@]}"; do IFS='|' read -r n d _ <<< "${STEPS[$i]}"; printf "%2d  %-30s %s\n" $((i+1)) "$n" "$d"; done
    exit 0
fi

mkdir -p "$LOGDIR" || { echo "ERROR: cannot create $LOGDIR" >&2; exit 2; }
SUMMARY="$LOGDIR/SUMMARY.txt"
# one run_all.sh at a time: two copies would write into the same output and launch twice
if command -v flock >/dev/null 2>&1; then
    exec 9>"$DATA/logs/run_all/.lock"
    if ! flock -n 9; then
        echo "ERROR: another run_all.sh is already running (lock $DATA/logs/run_all/.lock)." >&2
        echo "       Wait for it to finish, or stop it:  pkill -f cuenca/run_all.sh" >&2
        rmdir "$LOGDIR" 2>/dev/null; exit 3
    fi
fi

# ------------------------------------------------------------------ helpers
ts () { date '+%F %T'; }
say () { echo "[$(ts)] $*"; }
STEP_WAIT=0
L () { python launcher.py --exp "$1" --workers "$WORKERS" ${SET_ARGS[@]+"${SET_ARGS[@]}"}; }   # whole grid
PF () {                                    # paper_figures.py -> FIGROOT/paper1_<stage>; extra args passed on
    local stage="$1"; shift
    python paper_figures.py --out "$FIGROOT/paper1_$stage" --nmin "$NMIN" "$@" || return 1
    echo "--- decisions (paper_numbers.txt) ---"
    grep -h "DECISION" "$FIGROOT/paper1_$stage/paper_numbers.txt" 2>/dev/null || echo "(none yet)"
}
FIG1 () { python fig0_model_schematic.py --out "$FIGROOT/paper1_$1/fig1_model"; }
CALIB () { python cost_calibration.py --workers "$WORKERS"; }
TESTS () {       # test files live in the code folder
    local t ok=0
    for t in test_hubs test_meanfield test_dynamics; do
        echo "== $t"; python "$t.py" || ok=1
    done
    return $ok
}
SEEDCHECK () {   # one cheap run (N = 64) of an experiment with seed_offset; check the seed written
    local exp="$1" off
    off=$(python -c "import config; print(config.EXPERIMENTS['$exp'].get('seed_offset', 0))")
    python launcher.py --exp "$exp" --workers 1 --max-N 64 --max-jobs 1 || return 1
    local d; d=$(ls -1dt "$DATA/experiments/$exp"/exp_*/ 2>/dev/null | head -n 1)
    python - "$d/results.jsonl" "$off" <<'PY' || return 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
seeds = sorted({r["seed"] for r in rows if "error" not in r})
off = int(sys.argv[2])
ok = bool(seeds) and all(s >= off for s in seeds)
print(f"SEEDCHECK: seeds written {seeds}, expected >= {off}: {'OK' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
PY
    echo "(the one-run launch folder stays; analyses read the newest launch, i.e. the full one)"
}
POLYGUARD () {
    if [[ "$POLY_READY" != "1" ]]; then
        echo "poly_main waits for a human choice: read the poly_pilot table in the log of step"
        echo "'poly_pilot' (or $FIGROOT/paper1_pilot/paper_numbers.txt), set route_moves in config.py"
        echo "for poly_main AND poly_main_base, then:  ONLY=\"19 22\" POLY_READY=1 bash cuenca/run_all.sh"
        STEP_WAIT=1
    fi
    return 0
}
PACK () {        # one file with everything the manuscript needs
    local src="$FIGROOT/paper1_final" tarball="$FIGROOT/paper1_final_$STAMP.tar.gz"
    [[ -f "$src/paper_macros.tex" ]] || { echo "PACK: $src/paper_macros.tex not found"; return 1; }
    cp "$SUMMARY" "$src/run_all_SUMMARY.txt" 2>/dev/null
    tar -czf "$tarball" -C "$FIGROOT" paper1_final && echo "send this file: $tarball ($(du -h "$tarball" | cut -f1))"
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
    mkdir -p "$LOGDIR"
    {
        echo "FlowMergence run_all (Paper 1) — started $START_TS on $(hostname)"
        echo "code: $CODE_DIR   data: $DATA   workers: $WORKERS   NMIN: $NMIN   extra --set: ${EXTRA_SET:-none}"
        echo
        printf "%-4s %-30s %-9s %s\n" "step" "name" "status" "duration"
        for i in "${!RES_NAME[@]}"; do printf "%-4s %-30s %-9s %s\n" "${RES_NUM[$i]}" "${RES_NAME[$i]}" "${RES_STATUS[$i]}" "${RES_TIME[$i]}"; done
    } > "$SUMMARY"
}
trap 'say "INTERRUPTED (signal) during step ${CURRENT:-?}"; RES_NUM+=("${CURRENT_NUM:-?}"); RES_NAME+=("${CURRENT:-?}"); RES_STATUS+=("KILLED"); RES_TIME+=("-"); write_summary; exit 130' INT TERM

# ------------------------------------------------------------------ banner + dry runs
START_TS="$(ts)"; T0=$(date +%s)
say "FlowMergence run_all (Paper 1) on $(hostname): machine has $NPROC cores, using CPUS=$CPUS -> $WORKERS workers"
say "code folder: $CODE_DIR"
say "data root  : $DATA"
say "logs       : $LOGDIR"
say "python     : $(command -v python)  (conda env: ${CONDA_DEFAULT_ENV:-none})"
say "dry runs (job counts and relative cost):"
for i in "${!STEPS[@]}"; do
    k=$((i+1)); selected "$k" || continue
    IFS='|' read -r name _ cmds <<< "${STEPS[$i]}"
    for e in $(grep -o '\bL [a-z_0-9]*' <<< "$cmds" | awk '{print $2}'); do
        python launcher.py --exp "$e" --dry-run ${SET_ARGS[@]+"${SET_ARGS[@]}"} 2>/dev/null | grep "total=" | sed "s/^/    /"
    done
done

# ------------------------------------------------------------------ run
for i in "${!STEPS[@]}"; do
    k=$((i+1))
    IFS='|' read -r name desc cmds <<< "${STEPS[$i]}"
    selected "$k" || { say "skip  $k $name"; continue; }
    CURRENT="$name"; CURRENT_NUM="$k"
    mkdir -p "$LOGDIR"            # re-create it if it was moved or deleted while running
    LOG="$LOGDIR/$(printf '%02d' "$k")_${name}.log"
    say "START $k/${#STEPS[@]}  $name — $desc"
    say "      log: $LOG"
    t1=$(date +%s); status="OK"; STEP_WAIT=0
    IFS=';' read -r -a parts <<< "$cmds"
    for c in "${parts[@]}"; do
        c="$(echo "$c" | sed 's/^ *//; s/ *$//')"
        [[ -z "$c" ]] && continue
        echo "[$(ts)] \$ $c" >> "$LOG"
        # shellcheck disable=SC2086
        if ! eval "$c" >> "$LOG" 2>&1; then status="FAILED"; echo "[$(ts)] command failed: $c" >> "$LOG"; break; fi
        if [[ "$STEP_WAIT" == "1" ]]; then status="WAITING"; break; fi
    done
    dt=$(( $(date +%s) - t1 ))
    RES_NUM+=("$k"); RES_NAME+=("$name"); RES_STATUS+=("$status"); RES_TIME+=("$(fmt_dur $dt)")
    write_summary
    say "END   $k $name: $status in $(fmt_dur $dt)"
    if [[ "$status" == "WAITING" ]]; then
        tail -n 4 "$LOG" | sed 's/^/      | /'
    elif [[ "$status" != "OK" ]]; then
        say "      last lines of the log:"; tail -n 15 "$LOG" | sed 's/^/      | /'
        if [[ "$name" == "tests" && "$STOP_ON_TEST_FAIL" == "1" ]]; then
            say "tests failed: stopping (nothing should run on broken code). STOP_ON_TEST_FAIL=0 ignores this."
            write_summary; exit 1
        fi
        if [[ "$name" == "paper_density_exponent_seedsB" ]] && grep -q "SEEDCHECK: .*FAILED" "$LOG"; then
            say "seed check failed: the runner does not use job['seed']; fix runner.py before the seed block."
        fi
    fi
    # decisions so far, after every analysis step
    if [[ "$cmds" == *"PF "* ]]; then
        grep -h "DECISION" "$LOG" 2>/dev/null | awk '!seen[$0]++' | sed 's/^/      /'
    fi
done

say "ALL DONE in $(fmt_dur $(( $(date +%s) - T0 )))"
echo; cat "$SUMMARY"