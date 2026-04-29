#!/usr/bin/env bash
# =============================================================================
# profile_remote.sh
#
# Connects to a remote server, pulls, builds (CMake Profile), profiles
# with Valgrind/callgrind (CPU) and ncu (GPU), scps results back, then
# annotates copies of your source files with per-line timing data.
#
# Usage:
#   ./profile_remote.sh [options]
#
# Options:
#   -h, --host HOST          SSH host (required if not in config)
#   -u, --user USER          SSH user (required if not in config)
#   -p, --port PORT          SSH port (default: 22)
#   --password PASSWORD      SSH password (or leave blank to be prompted securely)
#   -r, --repo-path PATH     Path to repo root on server (required)
#   -b, --binary PATH        Path to built binary relative to build dir (required)
#   -a, --args ARGS          Arguments to pass to the binary (quote as one string)
#   -c, --cmake-args ARGS    Extra CMake arguments (quote as one string)
#   -o, --output-dir DIR     Local output directory (default: ./profile_output)
#   -t, --hot-threshold PCT  % threshold for * marker (default: 1.0)
#   -w, --col-width N        Column prefix width in chars (default: 45)
#       --no-gpu             Skip GPU profiling even if ncu is available
#       --no-cpu             Skip CPU profiling (callgrind)
#       --branch BRANCH      Git branch to checkout (default: current)
#       --build-dir NAME     Build subdirectory name (default: build_profile)
#       --config FILE        Load options from a config file (see below)
#
# Config file format (shell key=value, one per line, # comments ok):
#   HOST=myserver.example.com
#   USER=myuser
#   PORT=22

#   REPO_PATH=/home/myuser/myproject
#   BINARY=src/myapp
#   ARGS="--input data.bin --iterations 1000"
#   CMAKE_EXTRA_ARGS="-DWITH_CUDA=ON"
#   OUTPUT_DIR=./profile_output
#   HOT_THRESHOLD=1.0
#   COL_WIDTH=45
#   BRANCH=main
#   BUILD_DIR=build_profile
#
# Requirements (local):
#   - ssh, scp
#   - python3 (for annotation script, generated alongside this script)
#
# Requirements (remote):
#   - git, cmake, make/ninja
#   - valgrind (for CPU profiling)
#   - ncu (Nsight Compute, for GPU profiling)
#   - callgrind_annotate (part of valgrind suite)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HOST=""
USER_NAME=""
PORT=22
SSH_PASSWORD=""
REPO_PATH=""
BINARY=""
BINARY_ARGS=""
CMAKE_EXTRA_ARGS=""
OUTPUT_DIR="./profile_output"
HOT_THRESHOLD=1.0
COL_WIDTH=45
NO_GPU=0
NO_CPU=0
BRANCH=""
BUILD_DIR="build_profile"
CONFIG_FILE=""

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[profile]${RESET} $*"; }
ok()   { echo -e "${GREEN}[  ok  ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[ warn ]${RESET} $*"; }
die()  { echo -e "${RED}[ FAIL ]${RESET} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)       CONFIG_FILE="$2"; shift 2 ;;
    -h|--host)      HOST="$2"; shift 2 ;;
    -u|--user)      USER_NAME="$2"; shift 2 ;;
    -p|--port)      PORT="$2"; shift 2 ;;
    --password)     SSH_PASSWORD="$2"; shift 2 ;;
    -r|--repo-path) REPO_PATH="$2"; shift 2 ;;
    -b|--binary)    BINARY="$2"; shift 2 ;;
    -a|--args)      BINARY_ARGS="$2"; shift 2 ;;
    -c|--cmake-args) CMAKE_EXTRA_ARGS="$2"; shift 2 ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -t|--hot-threshold) HOT_THRESHOLD="$2"; shift 2 ;;
    -w|--col-width) COL_WIDTH="$2"; shift 2 ;;
    --no-gpu)       NO_GPU=1; shift ;;
    --no-cpu)       NO_CPU=1; shift ;;
    --branch)       BRANCH="$2"; shift 2 ;;
    --build-dir)    BUILD_DIR="$2"; shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Load config file if given (values only fill in unset vars)
if [[ -n "$CONFIG_FILE" ]]; then
  [[ -f "$CONFIG_FILE" ]] || die "Config file not found: $CONFIG_FILE"
  log "Loading config from $CONFIG_FILE"
  # Source with restricted scope: only load known keys
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    # Skip blank lines and full-line comments
    [[ "$raw_line" =~ ^[[:space:]]*# || -z "${raw_line// }" ]] && continue
    # Split on first = only
    key="${raw_line%%=*}"
    val="${raw_line#*=}"
    # Strip all leading/trailing whitespace from key
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}" 
    [[ -z "$key" ]] && continue
    # Strip inline comments, quotes, and all surrounding whitespace from val
    val="${val%%#*}"
    val="${val//\"/}"
    val="${val//\'/}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}" 
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}" 
    case "$key" in
      HOST)             [[ -z "$HOST" ]]             && HOST="$val" ;;
      USER)             [[ -z "$USER_NAME" ]]        && USER_NAME="$val" ;;
      PORT)             PORT="$val" ;;
      REPO_PATH)        [[ -z "$REPO_PATH" ]]        && REPO_PATH="$val" ;;
      BINARY)           [[ -z "$BINARY" ]]           && BINARY="$val" ;;
      ARGS)             [[ -z "$BINARY_ARGS" ]]      && BINARY_ARGS="$val" ;;
      CMAKE_EXTRA_ARGS) [[ -z "$CMAKE_EXTRA_ARGS" ]] && CMAKE_EXTRA_ARGS="$val" ;;
      OUTPUT_DIR)       [[ "$OUTPUT_DIR" == "./profile_output" ]] && OUTPUT_DIR="$val" ;;
      HOT_THRESHOLD)    HOT_THRESHOLD="$val" ;;
      COL_WIDTH)        COL_WIDTH="$val" ;;
      BRANCH)           [[ -z "$BRANCH" ]]           && BRANCH="$val" ;;
      BUILD_DIR)        BUILD_DIR="$val" ;;
    esac
  done < "$CONFIG_FILE"
fi

# Validate required
[[ -z "$HOST" ]]      && die "SSH host is required (--host or config HOST=)"
[[ -z "$USER_NAME" ]] && die "SSH user is required (--user or config USER=)"
[[ -z "$REPO_PATH" ]] && die "Repo path is required (--repo-path or config REPO_PATH=)"
[[ -z "$BINARY" ]]    && die "Binary path is required (--binary or config BINARY=)"

# Build SSH options string
SSH_OPTS="-p $PORT -o StrictHostKeyChecking=accept-new"
SCP_OPTS="-P $PORT -o StrictHostKeyChecking=accept-new"
SSH_TARGET="${USER_NAME}@${HOST}"

# ---------------------------------------------------------------------------
# Password handling via sshpass
# ---------------------------------------------------------------------------
if ! command -v sshpass &>/dev/null; then
  die "sshpass is not installed locally. Install it with:\n  macOS:  brew install sshpass\n  Ubuntu: sudo apt install sshpass"
fi

if [[ -z "$SSH_PASSWORD" ]]; then
  # Prompt securely (input not echoed)
  echo -n "SSH password for ${SSH_TARGET}: "
  read -rs SSH_PASSWORD
  echo ""
fi

[[ -z "$SSH_PASSWORD" ]] && die "Password cannot be empty."

# Wrap ssh and scp with sshpass
SSHPASS_ENV="SSHPASS=${SSH_PASSWORD}"
SSH_CMD="sshpass -e ssh"
SCP_CMD="sshpass -e scp"

# ---------------------------------------------------------------------------
# Prepare local output directory
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
ANNOTATED_DIR="${RUN_DIR}/annotated_source"
RAW_DIR="${RUN_DIR}/raw"
mkdir -p "$ANNOTATED_DIR" "$RAW_DIR"
LOG_FILE="${RUN_DIR}/profile_run.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo -e "${BOLD}=== C++ Profile Run: $TIMESTAMP ===${RESET}"
log "Host:       $SSH_TARGET:$PORT"
log "Repo:       $REPO_PATH"
log "Binary:     $BINARY"
log "Build dir:  $BUILD_DIR"
log "Output:     $RUN_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helper: run command on remote
# ---------------------------------------------------------------------------
remote() {
  env $SSHPASS_ENV $SSH_CMD $SSH_OPTS "$SSH_TARGET" "$@"
}

remote_script() {
  # Pass a heredoc as a script to run remotely
  env $SSHPASS_ENV $SSH_CMD $SSH_OPTS "$SSH_TARGET" bash -s
}

# ---------------------------------------------------------------------------
# Step 1: Git pull and build on remote
# ---------------------------------------------------------------------------
log "Step 1/5: Pulling and building on remote..."

REMOTE_BUILD="${REPO_PATH}/${BUILD_DIR}"

remote_script << REMOTE_EOF
set -euo pipefail
cd "${REPO_PATH}"

echo "--- Git status ---"
git fetch --all

if [[ -n "${BRANCH}" ]]; then
  git checkout "${BRANCH}"
fi

git stash

git pull --rebase
echo "--- At commit: \$(git rev-parse --short HEAD) \$(git log -1 --format='%s') ---"

echo ""
echo "--- CMake configure ---"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
  -DCMAKE_BUILD_TYPE=Profile \
  ${CMAKE_EXTRA_ARGS} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "--- Build ---"
# Use all available cores
cmake --build . --parallel \$(nproc)

echo ""
echo "--- Build complete ---"
REMOTE_EOF

ok "Build succeeded"

# ---------------------------------------------------------------------------
# Step 2: CPU profiling with callgrind
# ---------------------------------------------------------------------------
if [[ $NO_CPU -eq 0 ]]; then
  log "Step 2/5: Running callgrind CPU profile (this may take a while)..."

  REMOTE_CALLGRIND_OUT="${REMOTE_BUILD}/callgrind.out"
  REMOTE_CALLGRIND_ANNOTATED="${REMOTE_BUILD}/callgrind_annotated.txt"

  remote_script << REMOTE_EOF
set -euo pipefail
cd "${REMOTE_BUILD}"

BINARY_FULL="./${BINARY}"
[[ -x "\$BINARY_FULL" ]] || { echo "ERROR: Binary not found or not executable: \$BINARY_FULL"; exit 1; }

echo "--- Running valgrind callgrind ---"
{ time valgrind \
  --tool=callgrind \
  --callgrind-out-file="${REMOTE_CALLGRIND_OUT}" \
  --collect-atstart=yes \
  --instr-atstart=yes \
  --cache-sim=yes \
  --branch-sim=yes \
  --collect-jumps=yes \
  --dump-line=yes \
  --dump-instr=yes \
  --compress-strings=no \
  "\$BINARY_FULL" ${BINARY_ARGS} ; } 2>&1

echo "--- Generating callgrind annotation ---"
callgrind_annotate \
  --auto=yes \
  --threshold=0 \
  --context=0 \
  "${REMOTE_CALLGRIND_OUT}" \
  > "${REMOTE_CALLGRIND_ANNOTATED}" 2>&1

echo "--- callgrind done ---"
REMOTE_EOF

  log "Fetching callgrind results..."
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_CALLGRIND_ANNOTATED}" "${RAW_DIR}/callgrind_annotated.txt"
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_CALLGRIND_OUT}" "${RAW_DIR}/callgrind.out" 2>/dev/null || true
  ok "CPU profiling complete"
else
  warn "CPU profiling skipped (--no-cpu)"
fi

# ---------------------------------------------------------------------------
# Step 3: GPU profiling with ncu
# ---------------------------------------------------------------------------
if [[ $NO_GPU -eq 0 ]]; then
  log "Step 3/5: Running ncu GPU profile..."

  REMOTE_NCU_OUT="${REMOTE_BUILD}/ncu_report.ncu-rep"
  REMOTE_NCU_CSV="${REMOTE_BUILD}/ncu_report.csv"
  REMOTE_NCU_SOURCE="${REMOTE_BUILD}/ncu_source.txt"

  remote_script << REMOTE_EOF
set -euo pipefail
cd "${REMOTE_BUILD}"

BINARY_FULL="./${BINARY}"
[[ -x "\$BINARY_FULL" ]] || { echo "ERROR: Binary not found: \$BINARY_FULL"; exit 1; }

# Check if ncu is available
if ! command -v ncu &>/dev/null; then
  echo "WARN: ncu not found on PATH, skipping GPU profiling"
  exit 0
fi

echo "--- Running ncu ---"
ncu \
  --set full \
  --import-source yes \
  --export "${REMOTE_NCU_OUT}" \
  --force-overwrite \
  --call-stack \
  "\$BINARY_FULL" ${BINARY_ARGS} 2>&1 || true

# Export CSV for parsing
if [[ -f "${REMOTE_NCU_OUT}" ]]; then
  echo "--- Exporting ncu CSV ---"
  ncu \
    --import "${REMOTE_NCU_OUT}" \
    --csv \
    --page raw \
    > "${REMOTE_NCU_CSV}" 2>&1 || true

  echo "--- Exporting ncu source view ---"
  ncu \
    --import "${REMOTE_NCU_OUT}" \
    --page source \
    --print-source cuda,sass \
    > "${REMOTE_NCU_SOURCE}" 2>&1 || true
fi

echo "--- ncu done ---"
REMOTE_EOF

  log "Fetching ncu results..."
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_NCU_CSV}" "${RAW_DIR}/ncu_report.csv" 2>/dev/null || warn "ncu CSV not found (GPU may not have been exercised)"
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_NCU_SOURCE}" "${RAW_DIR}/ncu_source.txt" 2>/dev/null || true
  # Optionally fetch the full .ncu-rep for local Nsight UI
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_NCU_OUT}" "${RAW_DIR}/ncu_report.ncu-rep" 2>/dev/null || true
  ok "GPU profiling complete (or skipped if no CUDA kernels ran)"
else
  warn "GPU profiling skipped (--no-gpu)"
fi

# ---------------------------------------------------------------------------
# Step 4: Fetch source files from remote
# ---------------------------------------------------------------------------
log "Step 4/5: Fetching source files for annotation..."

REMOTE_SOURCE_LIST="${REMOTE_BUILD}/source_files.txt"

remote_script << REMOTE_EOF
set -euo pipefail
cd "${REPO_PATH}"
# Find all C/C++/CUDA source files tracked by git
git ls-files | grep -E '\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx)$' > "${REMOTE_SOURCE_LIST}" || true
echo "Found \$(wc -l < ${REMOTE_SOURCE_LIST}) source files"
REMOTE_EOF

env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:${REMOTE_SOURCE_LIST}" "${RAW_DIR}/source_files.txt" 2>/dev/null || true

SOURCE_FILES_LIST="${RAW_DIR}/source_files.txt"
SOURCE_FETCH_DIR="${RAW_DIR}"
mkdir -p "$SOURCE_FETCH_DIR"

if [[ -f "$SOURCE_FILES_LIST" && -s "$SOURCE_FILES_LIST" ]]; then
  # Tar the source tree on remote and fetch it
  remote_script << REMOTE_EOF
set -euo pipefail
cd "${REPO_PATH}"
tar czf /tmp/profile_src.tar.gz \$(cat "${REMOTE_SOURCE_LIST}" 2>/dev/null || echo ".")
REMOTE_EOF
  env $SSHPASS_ENV $SCP_CMD $SCP_OPTS "${SSH_TARGET}:/tmp/profile_src.tar.gz" "${RAW_DIR}/profile_src.tar.gz"
  tar xzf "${RAW_DIR}/profile_src.tar.gz" -C "$SOURCE_FETCH_DIR"
  ok "Source files fetched"
else
  warn "Could not list source files; annotation will work on whatever is referenced in profile data"
fi

# ---------------------------------------------------------------------------
# Step 5: Annotate source files locally (Python)
# ---------------------------------------------------------------------------
log "Step 5/5: Annotating source files..."

# Write the annotation Python script alongside the output
ANNOTATE_SCRIPT="${RUN_DIR}/annotate.py"

cat > "$ANNOTATE_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
annotate.py  —  Merges callgrind + ncu data into annotated source file copies.

Column layout (prefix of COL_WIDTH chars before the original source line):
  char 0      : '*' if self% >= HOT_THRESHOLD, else ' '
  chars 1-7   : self%  (e.g. "  3.21%") right-aligned, or blanks
  char 8      : '|'
  chars 9-15  : total% (inclusive: self + children)
  char 16     : '|'
  chars 17-25 : calls  (integer, right-aligned)
  char 26     : '|'
  chars 27-35 : gpu_ms (GPU kernel ms attributed, or blanks)
  char 36-    : padding to COL_WIDTH, then original source line
"""

import sys, os, re, csv, argparse
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--callgrind",  default="")
    p.add_argument("--ncu-csv",    default="")
    p.add_argument("--ncu-source", default="")
    p.add_argument("--src-dir",    required=True)
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--col-width",  type=int,   default=45)
    p.add_argument("--hot-thresh", type=float, default=1.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parse callgrind raw .out file
# ---------------------------------------------------------------------------
def parse_callgrind(path):
    """
    Parses callgrind.out directly (more reliable than callgrind_annotate,
    which requires source files present on the profiling machine).

    Raw format key lines:
      summary: <ir> ...     — program-wide total instruction count
      fl=<path>             — sets current source file
      fn=<name>             — sets current function (resets line cursor)
      fi=<path>             — inline file change (treat same as fl)
      fe=<path>             — end of inlined file (treat same as fl)
      calls=<n> <addr>      — next cost line is call-site inclusive cost
      <hex> <line> <ir> ... — absolute instruction record
      +<delta> <ir> ...     — line relative to previous record

    Returns:
      file_line_data : dict[abs_path] -> dict[lineno] ->
                         { self_ir, total_ir, calls }
      total_ir       : program-wide instruction count (denominator for %)
    """
    if not path or not os.path.exists(path):
        print(f"[annotate] Warning: callgrind file not found: {path}", file=sys.stderr)
        return {}, 1

    # Three ints per line: self instructions, inclusive call cost, call count
    file_line_data = defaultdict(
        lambda: defaultdict(lambda: {"self_ir": 0, "total_ir": 0, "calls": 0})
    )
    total_ir = 1
    cur_file = None
    cur_line = 0
    next_is_call_cost = False

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip()

            # ---- program total ----
            if line.startswith("summary:"):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        total_ir = int(parts[1])
                    except ValueError:
                        pass
                continue

            # ---- file context ----
            if line.startswith(("fl=", "fi=", "fe=")):
                val = line.split("=", 1)[1].strip()
                cur_file = None if val == "???" else val
                cur_line = 0
                next_is_call_cost = False
                continue

            # ---- function context ----
            if line.startswith("fn="):
                cur_line = 0
                next_is_call_cost = False
                continue

            # ---- call site marker ----
            # The data line immediately following carries the inclusive cost
            # of the callee attributed to cur_line in cur_file.
            if line.startswith("calls="):
                parts = line.split()
                # calls=<count> <addr> [<line_delta_to_callee>]
                # The count is embedded in parts[0] after the '='
                try:
                    n_calls = int(parts[0].split("=")[1])
                except (ValueError, IndexError):
                    n_calls = 0
                if cur_file and cur_line > 0:
                    file_line_data[cur_file][cur_line]["calls"] += n_calls
                next_is_call_cost = True
                continue

            # ---- skip jump/object records ----
            if line.startswith(("jump=", "jmp=", "ob=")):
                continue

            # ---- skip comments and blanks ----
            if not line or line.startswith("#"):
                continue

            # ---- skip if no file context ----
            if cur_file is None:
                continue

            # ---- parse data line ----
            parts = line.split()
            if not parts or parts[0].startswith(("c", "#", "j")):
                continue

            # Field 0: address (hex, *, or +delta) — we only care to detect the pattern
            # Field 1: line number (int, *, or +delta)
            # Field 2+: cost columns (first is Ir)
            if len(parts) < 2:
                continue

            addr_field = parts[0]
            line_field = parts[1] if len(parts) > 1 else "*"

            # Update cur_line from field 1
            if line_field == "*":
                pass  # keep cur_line
            elif line_field.startswith("+"):
                try:
                    cur_line += int(line_field[1:])
                except ValueError:
                    continue
            else:
                try:
                    cur_line = int(line_field)
                except ValueError:
                    continue

            # Cost starts at field 2
            if len(parts) < 3:
                continue
            try:
                ir_val = int(parts[2])
            except (ValueError, IndexError):
                continue

            if cur_line <= 0:
                continue
            if next_is_call_cost:
                # Inclusive cost of the call — add to total_ir for this line
                next_is_call_cost = False
                file_line_data[cur_file][cur_line]["total_ir"] += ir_val
            else:
                # Self cost
                file_line_data[cur_file][cur_line]["self_ir"] += ir_val

    if total_ir == 0:
        total_ir = 1

    print(f"[annotate] callgrind: {total_ir:,} total instructions, "
          f"{len(file_line_data)} source files referenced")
    return file_line_data, total_ir


def compute_totals(file_line_data, total_ir):
    """
    Convert raw IR counts to percentages.
    self%  = self_ir / total_ir
    total% = (self_ir + total_ir_from_calls) / total_ir
             i.e. self + inclusive cost of all calls made from this line
    """
    result = {}
    for fpath, lines in file_line_data.items():
        result[fpath] = {}
        for lno, d in lines.items():
            self_ir      = d["self_ir"]
            inclusive_ir = self_ir + d["total_ir"]   # total_ir field = call costs
            self_pct     = (self_ir      / total_ir) * 100.0
            total_pct    = (inclusive_ir / total_ir) * 100.0
            result[fpath][lno] = {
                "self_pct":  self_pct,
                "total_pct": total_pct,
                "calls":     d["calls"],
            }
    return result


# ---------------------------------------------------------------------------
# Parse ncu CSV output
# ---------------------------------------------------------------------------
def parse_ncu_csv(path):
    """
    Returns: dict[kernel_name] -> {duration_ms, file, line}
    """
    if not path or not os.path.exists(path):
        return {}

    kernels = defaultdict(lambda: {"duration_ms": 0.0, "file": None, "line": None})
    try:
        with open(path, "r", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Kernel Name", row.get("kernel_name", "unknown"))
                dur_str = (row.get("Duration") or
                           row.get("gpu__time_duration.sum") or
                           row.get("elapsed_cycles_sm") or "0")
                unit = row.get("Duration:unit", "nsecond")
                try:
                    dur_raw = float(dur_str.replace(",", ""))
                except ValueError:
                    dur_raw = 0.0
                if "nsec" in unit or unit == "ns":
                    dur_ms = dur_raw / 1e6
                elif "usec" in unit or unit == "us":
                    dur_ms = dur_raw / 1e3
                elif "msec" in unit or unit == "ms":
                    dur_ms = dur_raw
                else:
                    dur_ms = dur_raw / 1e6  # assume ns

                kernels[name]["duration_ms"] += dur_ms
                src_file = row.get("Source File", row.get("source_file", ""))
                src_line = row.get("Source Line", row.get("source_line", ""))
                if src_file and not kernels[name]["file"]:
                    kernels[name]["file"] = src_file
                if src_line and not kernels[name]["line"]:
                    try:
                        kernels[name]["line"] = int(src_line)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[annotate] Warning: could not parse ncu CSV: {e}", file=sys.stderr)
    return kernels


def parse_ncu_source(path):
    """
    Parse ncu --page source output for per-line GPU metrics.
    Returns: dict[src_file] -> dict[lineno] -> relative_cost
    """
    if not path or not os.path.exists(path):
        return {}

    result = defaultdict(lambda: defaultdict(float))
    cur_file = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = re.match(r"Source File:\s*(.+)", line)
            if m:
                cur_file = m.group(1).strip()
                continue
            if cur_file:
                m2 = re.match(r"^\s*(\d+)\s*\|", line)
                if m2:
                    lno = int(m2.group(1))
                    nums = re.findall(r"[\d.]+", line[m2.end():])
                    if nums:
                        try:
                            result[cur_file][lno] += float(nums[0])
                        except ValueError:
                            pass
    return result


# ---------------------------------------------------------------------------
# Formatting helpers  (all return fixed-width strings)
# ---------------------------------------------------------------------------
def fmt_pct(val):
    """7 chars: '  3.21%' or '       '"""
    if val is None or val == 0.0:
        return "       "
    return f"{val:6.2f}%"

def fmt_calls(val):
    """9 chars right-aligned integer, or blanks"""
    if not val:
        return "         "
    return str(int(val)).rjust(9)

def fmt_gpu(val):
    """9 chars: ' 142.33m' or '         '"""
    if not val:
        return "         "
    return f"{val:8.3f}m".rjust(9)


# ---------------------------------------------------------------------------
# Path matching helper
# ---------------------------------------------------------------------------
def match_path(candidate_paths, rel_str, abs_str):
    """
    Find the best key in candidate_paths that matches the given source file.
    Tries suffix matching on both the relative and absolute local path.
    """
    # Normalise separators
    rel_str = rel_str.replace("\\", "/")
    abs_str = abs_str.replace("\\", "/")
    for k in candidate_paths:
        k_norm = k.replace("\\", "/")
        if k_norm.endswith("/" + rel_str) or k_norm == rel_str:
            return k
        if abs_str.endswith("/" + k_norm.lstrip("/")) or abs_str == k_norm:
            return k
    return None


# ---------------------------------------------------------------------------
# Annotate a single source file
# ---------------------------------------------------------------------------
def annotate_file(src_path, out_path, line_data, gpu_data, col_width, hot_thresh):
    try:
        with open(src_path, "r", errors="replace") as f:
            src_lines = f.readlines()
    except Exception as e:
        print(f"[annotate] Cannot read {src_path}: {e}", file=sys.stderr)
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    header = (
        "// " + "=" * (col_width + 60) + "\n"
        "// PROFILING ANNOTATION\n"
        "//   self%  : % of total CPU instructions on this line (exclusive)\n"
        "//   total% : self% + inclusive cost of all calls from this line\n"
        "//   calls  : number of calls made from this line\n"
        "//   gpu_ms : GPU kernel time in ms attributed to this line (ncu)\n"
        "//   *      : self% >= hot threshold (" + str(hot_thresh) + "%)\n"
        "// Columns: [*] self%  | total% |    calls | gpu_ms   | <source>\n"
        "// " + "=" * (col_width + 60) + "\n"
    )

    with open(out_path, "w") as out:
        out.write(header)
        for i, src_line in enumerate(src_lines, start=1):
            d = line_data.get(i)
            g = gpu_data.get(i, 0.0)

            if d:
                self_pct  = d["self_pct"]
                total_pct = d["total_pct"]
                calls     = d["calls"]
                hot       = "*" if self_pct >= hot_thresh else " "
                prefix    = (f"{hot}{fmt_pct(self_pct)}"
                             f"|{fmt_pct(total_pct)}"
                             f"|{fmt_calls(calls)}"
                             f"|{fmt_gpu(g or None)} ")
            elif g > 0:
                prefix    = (f" {fmt_pct(None)}"
                             f"|{fmt_pct(None)}"
                             f"|{fmt_calls(None)}"
                             f"|{fmt_gpu(g)} ")
            else:
                prefix = " " * col_width

            # Pad or trim prefix to exactly col_width chars
            prefix = (prefix + " " * col_width)[:col_width]
            out.write(prefix + src_line.rstrip("\n") + "\n")

    print(f"[annotate] Written: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- CPU data ----
    print("[annotate] Parsing callgrind data...")
    raw_file_data, total_ir = parse_callgrind(args.callgrind)
    line_data_by_file = compute_totals(raw_file_data, total_ir)

    # ---- GPU data ----
    print("[annotate] Parsing ncu data...")
    ncu_kernels  = parse_ncu_csv(args.ncu_csv)
    ncu_src_data = parse_ncu_source(args.ncu_source)

    gpu_by_file = defaultdict(lambda: defaultdict(float))
    for kdata in ncu_kernels.values():
        if kdata["file"] and kdata["line"]:
            gpu_by_file[kdata["file"]][kdata["line"]] += kdata["duration_ms"]
    for fpath, lmap in ncu_src_data.items():
        for lno, val in lmap.items():
            gpu_by_file[fpath][lno] += val

    # ---- Walk source tree and annotate ----
    src_root = Path(args.src_dir)
    out_root = Path(args.out_dir)
    annotated_count = 0

    cpu_keys = list(line_data_by_file.keys())
    gpu_keys = list(gpu_by_file.keys())

    for src_file in sorted(src_root.rglob("*")):
        if not src_file.is_file():
            continue
        if src_file.suffix.lower() not in {
            ".c", ".cc", ".cpp", ".cxx", ".cu",
            ".h", ".hh", ".hpp", ".hxx"
        }:
            continue

        rel     = src_file.relative_to(src_root)
        out_file = out_root / rel
        rel_str  = str(rel)
        abs_str  = str(src_file.resolve())

        cpu_key = match_path(cpu_keys, rel_str, abs_str)
        gpu_key = match_path(gpu_keys, rel_str, abs_str)

        file_line_data = line_data_by_file.get(cpu_key, {})
        gpu_line_data  = gpu_by_file.get(gpu_key, {})

        annotate_file(
            src_path  = str(src_file),
            out_path  = str(out_file),
            line_data = file_line_data,
            gpu_data  = gpu_line_data,
            col_width = args.col_width,
            hot_thresh= args.hot_thresh,
        )
        annotated_count += 1

    print(f"[annotate] Done. {annotated_count} source files annotated.")


if __name__ == "__main__":
    main()
PYEOF

python3 "$ANNOTATE_SCRIPT" \
  --callgrind  "${RAW_DIR}/callgrind.out" \
  --ncu-csv    "${RAW_DIR}/ncu_report.csv" \
  --ncu-source "${RAW_DIR}/ncu_source.txt" \
  --src-dir    "${SOURCE_FETCH_DIR}" \
  --out-dir    "${ANNOTATED_DIR}" \
  --col-width  "${COL_WIDTH}" \
  --hot-thresh "${HOT_THRESHOLD}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=== Profile run complete ===${RESET}"
echo ""
ok "Raw profiler output:      ${RAW_DIR}/"
ok "Annotated source copies:  ${ANNOTATED_DIR}/"
ok "Run log:                  ${LOG_FILE}"
if [[ -f "${RAW_DIR}/ncu_report.ncu-rep" ]]; then
  ok "Nsight Compute report:    ${RAW_DIR}/ncu_report.ncu-rep  (open with: ncu-ui)"
fi
echo ""
log "Hottest lines across all files:"
echo ""

# Quick grep of annotated files to show top hot lines
grep -rh "^\*" "${ANNOTATED_DIR}" 2>/dev/null \
  | sort -t'%' -k1 -rn \
  | head -30 \
  || echo "(no hot lines found — program may have run too fast for callgrind to sample)"

echo ""
echo -e "${BOLD}Tip:${RESET} Open annotated files in VSCode:"
echo "  code ${ANNOTATED_DIR}"
echo ""
echo -e "${BOLD}Tip:${RESET} To view Nsight Compute report locally (if ncu-ui installed):"
echo "  ncu-ui ${RAW_DIR}/ncu_report.ncu-rep"
echo ""