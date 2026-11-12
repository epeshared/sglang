#!/usr/bin/env bash
# Compare 32-layer CPU-offload runs with next-layer L3 prefetch disabled/enabled.
# The enabled case is considered functional only when daemon METRICS report a
# non-zero [PREFETCH] exec_count; throughput alone cannot establish that.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/dual-loop-benchmark.sh"
[[ -f "${BENCH_SCRIPT}" ]] || { echo "missing ${BENCH_SCRIPT}" >&2; exit 1; }

# The standalone daemon invokes `python` itself. Put the same SGLang runtime
# first on PATH so it and the benchmark resolve torch/safetensors identically.
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/sglang_v0.5.5_heto/bin/python}
[[ -x "${PYTHON_BIN}" ]] || { echo "missing PYTHON_BIN=${PYTHON_BIN}" >&2; exit 2; }
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
"${PYTHON_BIN}" -c 'import moe_cpu_engine, safetensors, sglang, torch' || {
    echo "PYTHON_BIN is missing a required benchmark dependency" >&2
    exit 2
}

MOE_NUM_LAYERS=${MOE_NUM_LAYERS:-32}
[[ "${MOE_NUM_LAYERS}" == "32" ]] || {
    echo "This A/B is defined for 32 layers; got MOE_NUM_LAYERS=${MOE_NUM_LAYERS}" >&2
    exit 2
}

RUNS=${RUNS:-1}
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS must be a positive integer" >&2; exit 2; }
RESULT_ROOT=${RESULT_ROOT:-/tmp/cpu_offload_prefetch_ab_32layers_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${RESULT_ROOT}"

summarize_case() {
    local label=$1 log_file=$2 summary_file="${RESULT_ROOT}/summary.tsv"
    local prefetch_lines exec_total
    prefetch_lines=$(grep -E '^  \[PREFETCH\]' "${log_file}" || true)
    exec_total=$(printf '%s\n' "${prefetch_lines}" |
        grep -oE 'exec_count=[0-9]+' | cut -d= -f2 |
        awk '{sum += $1} END {print sum + 0}')

    printf '%s\t%s\t%s\n' "${label}" "prefetch_exec_total" "${exec_total}" >> "${summary_file}"
    echo ""
    echo "===== ${label}: daemon metrics ====="
    grep -E '^\[cpu_offload\]\[rank=.*\] METRICS|^  \[(ROUTING|COMBINE|PREFETCH|PREFETCH-DIAG)\]' \
        "${log_file}" | tail -80 || true
    echo "===== ${label}: benchmark metrics ====="
    grep -Ei 'prefill|decode|throughput|latency|[Tt]ok/s' "${log_file}" | tail -25 || true
    echo "${label}: aggregated prefetch_exec_count=${exec_total}"
}

run_case() {
    local enabled=$1 iteration=$2 label log_file shm_prefix
    if [[ "${enabled}" == "1" ]]; then
        label="on_${iteration}"
    else
        label="off_${iteration}"
    fi
    log_file="${RESULT_ROOT}/${label}.log"
    shm_prefix="/dev/shm/sglang_cpu_offload_prefetch_ab_${label}_$$"

    echo ""
    echo "===== A/B case ${label}: prefetch=${enabled} ====="
    CPU_OFFLOAD=1 \
    GPU_ONLY=0 \
    MOE_NUM_LAYERS=32 \
    CPU_OFFLOAD_PREFETCH_NEXT_LAYER="${enabled}" \
    CPU_OFFLOAD_SHM_PREFIX="${shm_prefix}" \
    bash "${BENCH_SCRIPT}" 2>&1 | tee "${log_file}"

    summarize_case "${label}" "${log_file}"
}

printf 'case\tmetric\tvalue\n' > "${RESULT_ROOT}/summary.tsv"
for ((iteration = 1; iteration <= RUNS; ++iteration)); do
    run_case 0 "${iteration}"
    run_case 1 "${iteration}"
done

on_exec=$(awk -F '\t' '$1 ~ /^on_/ && $2 == "prefetch_exec_total" {sum += $3} END {print sum + 0}' \
    "${RESULT_ROOT}/summary.tsv")
echo ""
echo "===== A/B verdict ====="
if (( on_exec > 0 )); then
    echo "PASS: enabled runs touched next-layer weight bytes (total exec_count=${on_exec})."
else
    echo "FAIL: enabled runs have no executed prefetch. Check [PREFETCH-DIAG], MOE_KERNEL_BACKEND=torch, and rebuild moe_cpu_engine."
fi
echo "Artifacts: ${RESULT_ROOT}"
echo "Summary:   ${RESULT_ROOT}/summary.tsv"