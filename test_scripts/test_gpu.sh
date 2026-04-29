#!/usr/bin/env bash
# test_identify.sh — builds identify in both CPU and CUDA mode, runs both on
# the same inputs, compares outputs, and reports timing for each.
#
# Usage:
#   ./test_identify.sh <sequence-directory> <taxonomy-file> <input-fasta>
#
# Requirements: cmake, make, nvidia-smi (for CUDA build), diff, bc

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <sequence-directory> <taxonomy-file> <input-fasta>"
    exit 1
fi

SEQ_DIR="$1"
TAX_FILE="$2"
FASTA="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

BUILD_CPU="$REPO_ROOT/build_test_cpu"
BUILD_CUDA="$REPO_ROOT/build_test_cuda"
OUT_CPU="$REPO_ROOT/out_cpu.txt"
OUT_CUDA="$REPO_ROOT/out_cuda.txt"

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { echo "[test] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

check_dep() {
    command -v "$1" &>/dev/null || fail "Required tool not found: $1"
}

check_dep cmake
check_dep make
check_dep diff
check_dep bc

# ── Check CUDA availability ───────────────────────────────────────────────────

CUDA_AVAILABLE=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_AVAILABLE=1
    log "CUDA device detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    log "WARNING: no CUDA device found — skipping CUDA build"
fi

# ── Build function ────────────────────────────────────────────────────────────
# build <build_dir> <HDC_BACKEND>

build() {
    local build_dir="$1"
    local backend="$2"

    log "Building with HDC_BACKEND=$backend ..."
    mkdir -p "$build_dir"
    cmake -S "$REPO_ROOT" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDC_BACKEND="$backend" \
        -DCMAKE_CXX_COMPILER_LAUNCHER="" \
        > "$build_dir/cmake.log" 2>&1 \
        || fail "cmake configure failed for $backend — see $build_dir/cmake.log"

    cmake --build "$build_dir" --target identify -j"$(nproc)" \
        > "$build_dir/build.log" 2>&1 \
        || fail "build failed for $backend — see $build_dir/build.log"

    log "Build OK: $build_dir/targets/identify"
}

# ── Run + time function ───────────────────────────────────────────────────────
# run_timed <binary> <output_file> — prints elapsed seconds, returns via TIME_SEC

TIME_SEC=""
run_timed() {
    local bin="$1"
    local outfile="$2"

    local start end elapsed
    start=$(date +%s%N)

    "$bin" "$SEQ_DIR" "$TAX_FILE" "$FASTA" "$outfile" \
        || fail "identify exited with error (binary: $bin)"

    end=$(date +%s%N)
    # nanoseconds → seconds with 3 decimal places
    elapsed=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    TIME_SEC="$elapsed"
}

# ── CPU build + run ───────────────────────────────────────────────────────────

build "$BUILD_CPU" "CPU"
CPU_BIN="$BUILD_CPU/targets/identify"

log "Running CPU mode..."
run_timed "$CPU_BIN" "$OUT_CPU"
CPU_TIME="$TIME_SEC"
log "CPU finished in ${CPU_TIME}s"

# ── CUDA build + run ──────────────────────────────────────────────────────────

if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    build "$BUILD_CUDA" "CUDA"
    CUDA_BIN="$BUILD_CUDA/targets/identify"

    log "Running CUDA mode..."
    run_timed "$CUDA_BIN" "$OUT_CUDA"
    CUDA_TIME="$TIME_SEC"
    log "CUDA finished in ${CUDA_TIME}s"
else
    log "Skipping CUDA run."
    CUDA_TIME=""
fi

# ── Output comparison ─────────────────────────────────────────────────────────

log "Comparing outputs..."

if [ -z "$CUDA_TIME" ]; then
    log "Only CPU output available — no comparison to perform."
else
    # Sort both outputs before diffing because map iteration order may differ
    sort "$OUT_CPU"  > "$OUT_CPU.sorted"
    sort "$OUT_CUDA" > "$OUT_CUDA.sorted"

    if diff -u "$OUT_CPU.sorted" "$OUT_CUDA.sorted" > "$REPO_ROOT/output_diff.txt" 2>&1; then
        log "Outputs match exactly."
        rm -f "$REPO_ROOT/output_diff.txt"
    else
        echo ""
        echo "┌─────────────────────────────────────────────────┐"
        echo "│  WARNING: CPU and CUDA outputs differ!          │"
        echo "│  See output_diff.txt for details.               │"
        echo "└─────────────────────────────────────────────────┘"
        echo ""
        head -40 "$REPO_ROOT/output_diff.txt"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  CPU time :  %ss\n" "$CPU_TIME"

if [ -n "$CUDA_TIME" ]; then
    printf "  CUDA time:  %ss\n" "$CUDA_TIME"
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $CUDA_TIME" | bc)
    printf "  Speedup  :  %sx\n" "$SPEEDUP"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  CPU output  : $OUT_CPU"
[ -n "$CUDA_TIME" ] && echo "  CUDA output : $OUT_CUDA"
echo ""