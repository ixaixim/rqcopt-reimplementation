#!/usr/bin/env bash
# Run a command in the background while *not* hogging the machine.
# Tries to use systemd cgroups (best), otherwise falls back to nice/ionice/cpulimit/taskset + nohup.
#
# Defaults: 50% CPU quota, low IO priority, mild CPU nice.
#
# Usage examples:
#   ./run_bg_throttled.sh -- python3 myscript.py
#   ./run_bg_throttled.sh -c 35 -- python3 train.py --epochs 50
#   ./run_bg_throttled.sh -c 50 -a "2-3" -m 2G -- python3 heavy_job.py --opt x
#   ./run_bg_throttled.sh -n 15 -i be:7 -l out.log -- python3 myscript.py

set -euo pipefail

CPU_PERCENT=50        # -- CPU quota target (0<..<=100)
NICE_LEVEL=10         # -- CPU scheduling priority (higher = "nicer"/less priority)
IO_CLASS="be"         # -- IO class: idle|be|rt  (best-effort = "be")
IO_LEVEL=7            # -- IO class level: 0(fast) .. 7(slowest) for best-effort
CPUSET=""             # -- CPU affinity (e.g. "0-1" or "0,2,4")
MEM_MAX=""            # -- Memory cap (e.g. "2G", "800M") if systemd available
LOGFILE="output.log"  # -- Only used in fallback mode (nohup)
USE_SYSTEMD=auto      # -- auto|yes|no

print_help() {
  cat <<EOF
run_bg_throttled.sh - throttle a background process

Options:
  -c PCT       CPU percent quota (default: ${CPU_PERCENT})
  -n NICE      CPU nice level (default: ${NICE_LEVEL})
  -i CLASS:LVL IO class and level (default: ${IO_CLASS}:${IO_LEVEL})
               CLASS in {idle,be,rt}; LVL in 0..7 (only for 'be')
  -a CPUSET    CPU affinity (e.g. "0-1", "0,2,4")
  -m MEM       Memory cap (systemd mode only), e.g. "2G", "800M"
  -l FILE      Log file (fallback mode), default: ${LOGFILE}
  -S MODE      systemd usage: auto|yes|no (default: auto)
  -h           Help

Command must follow after -- . Examples:
  $0 -- python3 myscript.py
  $0 -c 35 -a 2-3 -m 1G -- python3 train.py
EOF
}

# --- Parse options ---
while getopts ":c:n:i:a:m:l:S:h" opt; do
  case "$opt" in
    c) CPU_PERCENT="$OPTARG" ;;
    n) NICE_LEVEL="$OPTARG" ;;
    i) IO_CLASS="${OPTARG%%:*}"; IO_LEVEL="${OPTARG##*:}" ;;
    a) CPUSET="$OPTARG" ;;
    m) MEM_MAX="$OPTARG" ;;
    l) LOGFILE="$OPTARG" ;;
    S) USE_SYSTEMD="$OPTARG" ;;
    h) print_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; print_help; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
  esac
done
shift $((OPTIND-1))

# After option parsing, the next token may still be the -- separator.
if [[ "${1:-}" == "--" ]]; then
  shift 1
fi

if [[ $# -eq 0 ]]; then
  echo "Error: missing command to run." >&2
  print_help
  exit 2
fi

CMD=( "$@" )

VENV_ACTIVATE="/home/neelmiscia/data/environments/rqcopt/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  CMD=( bash -lc "source \"$VENV_ACTIVATE\" && exec \"\${@:1}\"" _ "${CMD[@]}" )
else
  echo "Warning: Python environment not found at $VENV_ACTIVATE; running without activation." >&2
fi

# --- Helpers & detection ---
have() { command -v "$1" >/dev/null 2>&1; }

# Normalize IO class
case "$IO_CLASS" in
  idle) IO_C=3; IO_N=7 ;;            # idle class ignores level, but we keep 7
  be)   IO_C=2; IO_N="${IO_LEVEL}" ;;# best-effort with level 0..7
  rt)   IO_C=1; IO_N=0 ;;            # real-time (avoid unless you know what you're doing)
  *)    echo "Invalid IO class: $IO_CLASS (use idle|be|rt)"; exit 2 ;;
esac

# --- Try systemd transient unit (best control: CPUQuota, MemoryMax, CPUAffinity) ---
SYSTEMD_OK=false
if [[ "$USE_SYSTEMD" != "no" ]] && have systemd-run; then
  # user mode is best for non-root; if not available, system mode may work under sudo
  if systemctl --user show-environment >/dev/null 2>&1; then
    SYSTEMD_MODE=(--user)
  else
    SYSTEMD_MODE=()
  fi
  UNIT="bgjob-$(date +%Y%m%d%H%M%S)-$$"

  PROPS=( "-p" "CPUQuota=${CPU_PERCENT}%" "-p" "WorkingDirectory=$(pwd)" )
  # CPU affinity (optional)
  if [[ -n "$CPUSET" ]]; then
    PROPS+=( "-p" "CPUAffinity=${CPUSET//,/ }" )
  fi
  # Memory cap (optional)
  if [[ -n "$MEM_MAX" ]]; then
    PROPS+=( "-p" "MemoryMax=${MEM_MAX}" )
  fi
  # IO: try to map to IOSchedulingClass/IOSchedulingPriority
  PROPS+=( "-p" "IOSchedulingClass=${IO_C}" "-p" "IOSchedulingPriority=${IO_N}" )

  # Build wrapped command with nice/ionice as extra hint inside the cgroup
  WRAPPED=( bash -lc "exec nice -n ${NICE_LEVEL} ionice -c ${IO_C} -n ${IO_N} ${CPUSET:+taskset -c ${CPUSET}} \"\${@:1}\"" _ ) # '_' sentinel
  # Run as a transient service (survives logout), quiet output -> see journal
  if systemd-run "${SYSTEMD_MODE[@]}" --unit="$UNIT" --quiet "${PROPS[@]}" -- "${WRAPPED[@]}" "${CMD[@]}"; then
    SYSTEMD_OK=true
    echo "Started as systemd unit: ${UNIT}"
    if [[ "${SYSTEMD_MODE[*]}" == *"--user"* ]]; then
      echo "Check status:  systemctl --user status ${UNIT}"
      echo "View logs:    journalctl --user -u ${UNIT} -f"
    else
      echo "Check status:  sudo systemctl status ${UNIT}"
      echo "View logs:    sudo journalctl -u ${UNIT} -f"
    fi
  fi
fi

if ! $SYSTEMD_OK; then
  # --- Fallback path: nohup + nice + ionice + (optional) taskset + (optional) cpulimit ---
  echo "Using fallback mode (nohup). Output -> ${LOGFILE}"
  PREFIX=( nice -n "${NICE_LEVEL}" ionice -c "${IO_C}" -n "${IO_N}" )
  if [[ -n "$CPUSET" ]] && have taskset; then
    PREFIX+=( taskset -c "${CPUSET}" )
  fi

  # If cpulimit is available and CPU_PERCENT<100, use it to cap CPU usage.
  if have cpulimit && [[ "${CPU_PERCENT}" -lt 100 ]]; then
    PREFIX+=( cpulimit -l "${CPU_PERCENT}" -- )
  fi

  # Run in background and survive logout
  nohup "${PREFIX[@]}" "${CMD[@]}" >> "${LOGFILE}" 2>&1 &
  PID=$!
  disown "$PID"
  echo "Started PID ${PID} with nice=${NICE_LEVEL}, io=${IO_CLASS}:${IO_LEVEL}${CPUSET:+, cpus=${CPUSET}}${MEM_MAX:+ (mem cap not available in fallback)}"
  echo "Tail logs:     tail -f ${LOGFILE}"
fi
