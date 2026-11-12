#!/usr/bin/env bash
# dual-loop-benchark.sh
#
# Two-pass benchmark for Qwen3-235B-A22B-Thinking-2507-FP8 (94 MoE layers).
#   Pass 1 — GPU-only  (no offload)
#   Pass 2 — CPU offload  (dual-loop EPLB, daemon preloads weights to /dev/shm)
#
# Usage:
#   bash dual-loop-benchark.sh             # node 0 (default)
#   NODE_RANK=1 bash dual-loop-benchark.sh # node 1
#
# Key tuning knobs — full list documented inline per §§ section below:
#   MODEL_PATH  MOE_NUM_LAYERS  BATCH_SIZE  INPUT_LEN  OUTPUT_LEN
#   DIST_INIT_ADDR  NODE_RANK  MEM_FRACTION_STATIC
#   CPU_OFFLOAD_ALLOC_MODE  CPU_WORKERS_PER_NUMA  MOE_DTYPE
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# Resolve workspace root relative to this script's location.
# Works from both the legacy location (<workspace>/) and the
# target location (<workspace>/sglang/hetero-test/).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${_SCRIPT_DIR}/sglang_env" ]]; then
    WORKSPACE_ROOT="${_SCRIPT_DIR}"                          # script at workspace root
elif [[ -d "${_SCRIPT_DIR}/../../sglang_env" ]]; then
    WORKSPACE_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"    # script at sglang/hetero-test/
elif [[ -d "${_SCRIPT_DIR}/../../moe-cpu-engine" ]]; then
    WORKSPACE_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"    # script at sglang/hetero-test/ (no sglang_env yet)
else
    WORKSPACE_ROOT="${_SCRIPT_DIR}"                          # fallback: same dir
fi

# ════════════════════════════════════════════════════════════
# §1  RUNTIME ENV  (xFasterTransformer + Intel OpenMP)
# ════════════════════════════════════════════════════════════
XFT_DIR="${WORKSPACE_ROOT}/sglang_env/lib/python3.12/site-packages/xfastertransformer-devel"
if [[ -f "${XFT_DIR}/lib/libiomp5.so" ]]; then
    IOMP_SO="${XFT_DIR}/lib/libiomp5.so"
elif [[ -f "${WORKSPACE_ROOT}/sglang_env/lib/libiomp5.so" ]]; then
    IOMP_SO="${WORKSPACE_ROOT}/sglang_env/lib/libiomp5.so"
else
    IOMP_SO=$(python -c "import sysconfig,os,glob; p=glob.glob(os.path.join(sysconfig.get_path('purelib'),'..','..','libiomp5.so')); print(p[0] if p else '')" 2>/dev/null || true)
fi
[[ -n "${IOMP_SO:-}" ]] && export LD_PRELOAD="${IOMP_SO}"
export LD_LIBRARY_PATH="${XFT_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1

# ════════════════════════════════════════════════════════════
# §2  MODEL & BENCHMARK CONFIG
# ════════════════════════════════════════════════════════════
MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-R1/}

# MOE_NUM_LAYERS: 0=full model (94 layers); >0=truncated smoke-test
MOE_NUM_LAYERS=${MOE_NUM_LAYERS:-20}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
# Truncated model needs less static VRAM (KV cache otherwise crowds FlashInfer workspace)
if [[ "${MOE_NUM_LAYERS}" -gt 0 ]] && [[ -z "${MEM_FRACTION_STATIC+x}" || "${MEM_FRACTION_STATIC}" == "0.7" ]]; then
    MEM_FRACTION_STATIC=0.4
fi

BATCH_SIZE=${BATCH_SIZE:-384}
INPUT_LEN=${INPUT_LEN:-32}
OUTPUT_LEN=${OUTPUT_LEN:-384}

# CPU offload is submitted only during CUDA-graph replay. SGLang captures
# batch sizes above 512 in steps of 32, so round up to guarantee one graph can
# serve the requested decode batch (for example, 1000 requires 1024, not 1000).
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$(( ((BATCH_SIZE + 31) / 32) * 32 ))}

# CORRECTNESS_TEST=1: greedy decode on 3 fixed prompts instead of throughput bench
CORRECTNESS_TEST=${CORRECTNESS_TEST:-0}
# DeepSeek-R1 is an instruction model; format correctness prompts as user turns.
CORRECTNESS_APPLY_CHAT_TEMPLATE=${CORRECTNESS_APPLY_CHAT_TEMPLATE:-1}
# GPU_ONLY=1: skip Pass 2;  CPU_OFFLOAD=1: skip Pass 1
GPU_ONLY=${GPU_ONLY:-0}
CPU_OFFLOAD=${CPU_OFFLOAD:-1}

# BENCH_MODE: bench_one_batch (offline single-batch) | launch_server (HTTP + bench_serving)
BENCH_MODE=${BENCH_MODE:-bench_one_batch}
SERVER_PORT=${SERVER_PORT:-30000}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-100}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
SERVER_READY_TIMEOUT_S=${SERVER_READY_TIMEOUT_S:-900}
# RUN_CLIENT: 1 (default) = this script runs bench_serving itself once the server is ready.
#             0 = only start the server (launch_server mode); run your own client
#                 against it separately. Script blocks holding the server up until
#                 you Ctrl+C (or kill it), then cleanup still runs as usual.
RUN_CLIENT=${RUN_CLIENT:-0}

# ════════════════════════════════════════════════════════════
# §3  DISTRIBUTED & NETWORK
# ════════════════════════════════════════════════════════════
_NNODES=1                         # total nodes; drives rank-range math + SGLANG_NNODES
NODE_RANK=${NODE_RANK:-0}         # 0=this machine, 1=peer
export SGLANG_NNODES=${_NNODES}   # CpuOffloadConfig reads this for socket-index mapping
# TP_SIZE: total tensor-parallel/GPU rank count (propagated to --tp/--ep-size).
# e.g. TP_SIZE=2 CPU_NUM_HOSTS=2 -> exactly 1 GPU rank per physical socket.
_TP_SIZE=${TP_SIZE:-8}
if (( _TP_SIZE % _NNODES )); then
    echo "[full94] ERROR: TP_SIZE=${_TP_SIZE} not divisible by nnodes=${_NNODES}" >&2
    exit 1
fi

# Validate NODE_RANK immediately — fail before any expensive setup
if (( NODE_RANK >= _NNODES )); then
    echo "[full94] ERROR: NODE_RANK=${NODE_RANK} >= nnodes=${_NNODES}." \
         "Valid: 0 .. $(( _NNODES - 1 ))" >&2
    exit 1
fi

# torch.distributed rendezvous (TCPStore).
# Override if node 0's IP differs: DIST_INIT_ADDR=<ip>:8000 bash dual-loop-benchark.sh
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
_DIST_INIT_HOST="${DIST_INIT_ADDR%%:*}"
_DIST_INIT_PORT="${DIST_INIT_ADDR##*:}"
# Seconds torch.distributed waits for all nodes; raise only if startup is very slow
DIST_TIMEOUT_S=${DIST_TIMEOUT_S:-99999}

# ── NCCL ─────────────────────────────────────────────────────────────────────
# Auto-detect the network interface matching DIST_INIT_HOST (three fallbacks).
# Override: NCCL_SOCKET_IFNAME=enp205s0 bash dual-loop-benchark.sh
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    _nccl_iface=$(ip -o addr show 2>/dev/null \
        | awk -v ip="${_DIST_INIT_HOST}" '$0 ~ ip {print $2; exit}' || true)
    if [[ -z "${_nccl_iface}" ]]; then
        _nccl_iface=$(awk -v ip="${_DIST_INIT_HOST}" \
            'NR>1 && $1==ip {print $NF; exit}' /proc/net/arp 2>/dev/null || true)
    fi
    if [[ -z "${_nccl_iface}" ]]; then
        _nccl_iface=$(awk 'NR>1 {print $NF}' /proc/net/arp 2>/dev/null \
            | sort | uniq -c | sort -rn | awk '{print $2; exit}' || true)
    fi
    [[ -n "${_nccl_iface}" ]] && export NCCL_SOCKET_IFNAME="${_nccl_iface}"
fi
# irdma (Intel iWARP) causes ncclCommInitRank "unhandled system error" → disable IB by default
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# DeepGEMM JIT has no sm_120 (Blackwell) recipe → disable, fall back to triton FP8
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-1}

# ════════════════════════════════════════════════════════════
# §4  CPU OFFLOAD CONFIG
# ════════════════════════════════════════════════════════════

# ── 4a  Worker placement ──────────────────────────────────────────────────────
CPU_NUM_HOSTS=${CPU_NUM_HOSTS:-2}           # physical sockets / daemon instances
# CPU_ACTIVE_RANKS_PER_SOCKET: how many TP ranks per socket participate in CPU offload.
#   1 (default) — only the first rank of each socket (minimal daemon load)
#   2            — first two ranks of each socket share one daemon worker
#   N            — must not exceed _ranks_per_node / CPU_NUM_HOSTS
CPU_ACTIVE_RANKS_PER_SOCKET=${CPU_ACTIVE_RANKS_PER_SOCKET:-2}


# CPU_OFFLOAD_ALLOC_MODE: how CPU cores are assigned to workers
#   core      — reserve first CPU_RESERVED_CORES per socket for OS; split the rest
#               intended for non-SNC machines (1 Linux NUMA per physical socket)
#   numa-node — each worker owns one non-reserved sub-NUMA node (default)
#               works on both non-SNC and SNC machines
CPU_OFFLOAD_ALLOC_MODE=${CPU_OFFLOAD_ALLOC_MODE:-numa-node}
case "${CPU_OFFLOAD_ALLOC_MODE}" in
    core)
        export CPU_OFFLOAD_USE_SUBNUMA_ALLOCATION=0
        export CPU_OFFLOAD_USE_PHYSICAL_SOCKET_SUBNUMA=0 ;;
    numa-node)
        export CPU_OFFLOAD_USE_SUBNUMA_ALLOCATION=1
        export CPU_OFFLOAD_USE_PHYSICAL_SOCKET_SUBNUMA=1 ;;
    *)
        echo "[full94] ERROR: unknown CPU_OFFLOAD_ALLOC_MODE='${CPU_OFFLOAD_ALLOC_MODE}'" \
             "(valid: core | numa-node)" >&2
        exit 1 ;;
esac

# NUMA nodes reserved per socket for OS scheduling. only used when CPU_OFFLOAD_ALLOC_MODE=numa-node.
# Default to the first sub-NUMA node discovered in each physical socket, rather
# than assuming SNC3 numbering (socket0=0, socket1=3).
if [[ -z "${CPU_OFFLOAD_RESERVED_SUBNUMA_IDS+x}" ]]; then
    CPU_OFFLOAD_RESERVED_SUBNUMA_IDS=$(python - <<'PY'
import subprocess

nodes_by_socket = {}
for line in subprocess.check_output(["lscpu", "-p=NODE,SOCKET"], text=True).splitlines():
    if line.startswith("#"):
        continue
    node, socket = map(int, line.split(","))
    nodes_by_socket.setdefault(socket, set()).add(node)

print(";".join(
    f"socket{socket}={min(nodes)}"
    for socket, nodes in sorted(nodes_by_socket.items())
))
PY
)
    export CPU_OFFLOAD_RESERVED_SUBNUMA_IDS
fi
# CPU_RESERVED_CORES: when CPU_OFFLOAD_ALLOC_MODE=core, reserve this many cores per socket for OS.
CPU_RESERVED_CORES=${CPU_RESERVED_CORES:-8}
# Daemon --socket N IDs passed as --socket N to cpu_expertd.
# numa-node: physical socket sequential index (always "0,1", SNC-agnostic).
# core:      Linux NUMA node ID of first NUMA per physical socket. Derive this
#            from lscpu instead of assuming contiguous socket/NUMA numbering.
if [[ "${CPU_OFFLOAD_ALLOC_MODE}" == "numa-node" ]]; then
    export CPU_NUMA_NODE_IDS=${CPU_NUMA_NODE_IDS:-"0,1"}
else
    if [[ -z "${CPU_NUMA_NODE_IDS+x}" ]]; then
        CPU_NUMA_NODE_IDS=$(python - <<'PY'
import subprocess

nodes_by_socket = {}
for line in subprocess.check_output(["lscpu", "-p=NODE,SOCKET"], text=True).splitlines():
    if line.startswith("#"):
        continue
    node, socket = map(int, line.split(","))
    nodes_by_socket.setdefault(socket, set()).add(node)

print(",".join(
    str(min(nodes_by_socket[socket]))
    for socket in sorted(nodes_by_socket)
))
PY
)
        export CPU_NUMA_NODE_IDS
    fi
fi

# Rank-to-worker assignment: order (sequential 1:1, default) | custom (explicit map)
export CPU_OFFLOAD_WORKER_ASSIGN_STRATEGY=${CPU_OFFLOAD_WORKER_ASSIGN_STRATEGY:-order}
# Custom map — only used when strategy=custom. Format: "rank:slot,rank:slot,..."
# Active ranks with CPU_ACTIVE_RANKS_PER_SOCKET=2: 0,1 (socket0) and 4,5 (socket1).
# SNC3: socket0 reserved NUMA=0, workers at slot0=NUMA1 slot1=NUMA2;
#        socket1 reserved NUMA=3, workers at slot0=NUMA4 slot1=NUMA5.
# Each rank gets its own slot within the socket (no two ranks share a slot).
export CPU_OFFLOAD_RANK_TO_WORKER_MAP="0:0,1:1,4:0,5:1"
#export CPU_OFFLOAD_RANK_TO_WORKER_MAP="0:0,1:0"

# ── 4b  Daemon process ────────────────────────────────────────────────────────
export CPU_OFFLOAD_SHM_PREFIX=${CPU_OFFLOAD_SHM_PREFIX:-/dev/shm/sglang_cpu_offload_b1}
CPU_SLOT_UPDATE_MS=${CPU_SLOT_UPDATE_MS:-500}
export CPU_EXPERTD_READY_TIMEOUT_S=${CPU_EXPERTD_READY_TIMEOUT_S:-1800}
export CPU_OFFLOAD_BACKEND=daemon
export CPU_OFFLOAD_DAEMON_LOADS_WEIGHTS=1
export CPU_OFFLOAD_MODEL_PATH=${CPU_OFFLOAD_MODEL_PATH:-${MODEL_PATH}}
export CPU_OFFLOAD_DAEMON_LAUNCHER=           # intentionally empty (auto-managed)
export CPU_OFFLOAD_DAEMON_PROC_NAME=${CPU_OFFLOAD_DAEMON_PROC_NAME:-cpu_expertd}
export CPU_ZERO_COPY=${CPU_ZERO_COPY:-1}
export CPU_OFFLOAD_KEEP_WEIGHTS=${CPU_OFFLOAD_KEEP_WEIGHTS:-0}
# CPU_OFFLOAD_PRELOAD_THREADS: parallel threads for the one-time direct-load
# FP8->INT8 convert+VNNI-pack pass (_direct_load_weights). Each thread gets
# torch_threads/this-value OMP threads, so raising it trades per-layer
# convert speed for more layers in flight -- re-measure total load time
# (see /tmp/cpu_expertd_logs/cpu_expertd.socket*.log) after changing.
export CPU_OFFLOAD_PRELOAD_THREADS=${CPU_OFFLOAD_PRELOAD_THREADS:-4}

# ── 4c  Engine tuning ─────────────────────────────────────────────────────────
# MOE_DTYPE: bf16 | int8 (default) | fp8
export MOE_DTYPE=${MOE_DTYPE:-int8}
# MOE_KERNEL_BACKEND: torch (default) | xft
export MOE_KERNEL_BACKEND=${MOE_KERNEL_BACKEND:-torch}
# 单位: spin-iteration count (NOT us), backoff_ns=200/spin by default.
# MUST cover real CPU compute latency (avg_gemm_us) now that D2H/submit/wait
# run on their own CUDA stream (see cuda_graph_offload.py's _offload_stream):
# previously wait_kernel was queued BEHIND this layer's local-GPU-compute
# kernels on one shared stream, so it only started executing after ~500-600us
# had already elapsed for free -- MAX_SPIN=1 "worked" only because of that
# accidental delay, not because 1 spin is enough. Decoupling the streams
# removed that free head start (wait_kernel now starts polling almost
# immediately after submit), so a tiny MAX_SPIN now times out on nearly
# every call. 20000 spins * 200ns = 4ms of real poll budget (>> avg_gemm_us
# ~220-280us), costs nothing on a hit (returns the instant done_seq flips).
export CPU_OFFLOAD_MAX_SPIN=${CPU_OFFLOAD_MAX_SPIN:-10000}
# CPU_OFFLOAD_MAX_HITS: compact D2H/H2D slot count; 0 = auto (min(max_tokens,128)).
# NOTE: raising this does NOT increase GPU-side compute savings -- masked_topk_ids
# (what local fused_moe_kernel actually computes) is built from the FULL branch_mask
# (100% of true cpu_slot demand) BEFORE compaction/max_hits even applies, so GPU
# already skips all of it regardless of this value. Raising MAX_HITS only recovers
# more of the OVERFLOW-dropped tokens' correctness (fewer silently-zeroed expert
# contributions) at the cost of bigger D2H/H2D + longer real daemon compute -- a
# correctness/overhead tradeoff, not a throughput lever. Do not raise expecting
# a throughput win.
export CPU_OFFLOAD_MAX_HITS=${CPU_OFFLOAD_MAX_HITS:-64}
# CPU_EXPERTD_OPTIMAL_CORES: performance sweet spot — beyond this thread count
# OMP scaling saturates (cache contention, NUMA bandwidth, etc.).
# If auto-allocated cores > optimal, clamp down to optimal.
# 0 = disabled (use all allocated cores).  Typical range: 16-64.
CPU_EXPERTD_OPTIMAL_CORES=${CPU_EXPERTD_OPTIMAL_CORES:-32}
# CPU_OFFLOAD_PREFETCH_NEXT_LAYER: fire-and-forget prefetch of the one
# CPU-assigned expert in the next layer while the current layer computes.
# It is targeted (~18 MiB), not a whole-layer cache fill.  1 = enabled,
# 0 = disabled for A/B measurement.
# HISTORY: an initial A/B found this to be a net regression (latency
# +11.7%, throughput -10.5%, COMBINE wait_timeout 6-60x worse) because
# MOE::prefetchExpert() (moe.cpp) burst ~18MB/expert through DRAM
# unthrottled, starving the real FORWARD_EXPERTS GEMM. Fixed by pacing
# the touch loop (CPU_OFFLOAD_PREFETCH_CHUNK_BYTES / _PACE_US) and using
# an L3-only prefetch hint. Re-enabled by default after that fix --
# re-run this same benchmark on/off if you touch prefetchExpert() again.
export CPU_OFFLOAD_PREFETCH_NEXT_LAYER=${CPU_OFFLOAD_PREFETCH_NEXT_LAYER:-1}
# CPU_OFFLOAD_WARMUP_LEVEL: strategy used for the next-layer warm above
# (no effect if CPU_OFFLOAD_PREFETCH_NEXT_LAYER=0).
#   1 (default) — L3 cache warm only (read-only, hard-abortable mid-flight).
#   2           — additionally runs a real forwardExperts() GEMM on the
#                 target expert with fixed dummy hidden states, discarding
#                 the output. Cannot be aborted once started, so a real
#                 request arriving mid-compute must wait for it to finish.
export CPU_OFFLOAD_WARMUP_LEVEL=${CPU_OFFLOAD_WARMUP_LEVEL:-2}
# Token count used for the level-2 dummy forwardExperts() call.
export CPU_OFFLOAD_WARMUP_TOKENS=${CPU_OFFLOAD_WARMUP_TOKENS:-64}
# Each next-layer warm is capped at: observed attention window x this
# sustainable per-prefetch-thread DDR rate. Do not set this to peak socket BW.
export CPU_OFFLOAD_PREFETCH_DDR_GBPS=${CPU_OFFLOAD_PREFETCH_DDR_GBPS:-200}
# Used only before the daemon has observed one real next-layer attention gap.
export CPU_OFFLOAD_PREFETCH_BOOTSTRAP_ATTENTION_US=${CPU_OFFLOAD_PREFETCH_BOOTSTRAP_ATTENTION_US:-500}
# Pace the L3 touch loop only while a real CPU GEMM is active. These defaults
# trade prefetch completion time for lower instantaneous DRAM-bandwidth use.
export CPU_OFFLOAD_PREFETCH_CHUNK_BYTES=${CPU_OFFLOAD_PREFETCH_CHUNK_BYTES:-262144}
export CPU_OFFLOAD_PREFETCH_PACE_US=${CPU_OFFLOAD_PREFETCH_PACE_US:-150}
export CPU_OFFLOAD_IMPACT_DIAG=0        # 详细 hidden-state 诊断日志
export CPU_OFFLOAD_IMPACT_DIAG_MAX=80   # 最大打印次数
# CPU_OFFLOAD_VERIFY_PROBE: pinned-host status/error readback used only by
# read_last_status_blocking (timeout detection on the non-graph combine path
# + debugging). Each forward call pays 2 extra memcpy_async + a host stream
# sync to check it. 1 (default) = on; 0 = off (skip the pinned-host buffers
# entirely — cheaper, but read_last_status_blocking always reports (0,0)).
export CPU_OFFLOAD_VERIFY_PROBE=${CPU_OFFLOAD_VERIFY_PROBE:-0}

# ── 4d  Process binding & NUMA topology ──────────────────────────────────────
export SGLANG_SET_CPU_AFFINITY=${SGLANG_SET_CPU_AFFINITY:-1}
# In numa-node mode, bind GPU service processes to their reserved NUMA nodes.
# In core mode, the daemon and GPU ranks share a NUMA node. Build explicit GPU
# CPU sets from the reserved physical cores (including SMT siblings), leaving
# daemon worker cores unavailable to scheduler ranks. Select the first NUMA
# node of each physical socket: on SNC3, nodes 0/1/2 are socket0 and 3/4/5 are
# socket1, so rank 4 must use node3 rather than node1.
if [[ "${CPU_OFFLOAD_ALLOC_MODE}" == "core" && -z "${SGLANG_GPU_PROC_CPUSET+x}" ]]; then
    SGLANG_GPU_PROC_CPUSET=$(CPU_RESERVED_CORES="${CPU_RESERVED_CORES}" TP_SIZE="${_TP_SIZE}" python - <<'PY'
import os
import subprocess

reserved_cores = int(os.environ["CPU_RESERVED_CORES"])
cores_by_node = {}
nodes_by_socket = {}
for line in subprocess.check_output(
    ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True
).splitlines():
    if line.startswith("#"):
        continue
    cpu, core, socket, node = map(int, line.split(","))
    cores_by_node.setdefault(node, {}).setdefault(core, []).append(cpu)
    nodes_by_socket.setdefault(socket, set()).add(node)

cpu_sets_by_socket = []
for socket in sorted(nodes_by_socket):
    reserved_node = min(nodes_by_socket[socket])
    core_map = cores_by_node[reserved_node]
    selected = sorted(core_map)[:reserved_cores]
    cpus = sorted(cpu for core in selected for cpu in core_map[core])
    cpu_sets_by_socket.append(",".join(str(cpu) for cpu in cpus))

# GPU ranks are split evenly across physical sockets (TP_SIZE / num_sockets each).
if len(cpu_sets_by_socket) < 2:
    raise SystemExit("need at least two physical sockets for core-mode binding")
tp_size = int(os.environ["TP_SIZE"])
if tp_size % len(cpu_sets_by_socket):
    raise SystemExit("TP_SIZE not divisible by number of physical sockets")
ranks_per_socket = tp_size // len(cpu_sets_by_socket)
print(";".join(
    cpu_set
    for cpu_set in cpu_sets_by_socket
    for _ in range(ranks_per_socket)
))
PY
)
    export SGLANG_GPU_PROC_CPUSET
fi
# In numa-node mode, bind each rank to the same first sub-NUMA node reserved
# from cpu_expertd. This keeps every non-reserved sub-NUMA node exclusive to
# daemon workers. Ranks are split evenly across the CPU_NUM_HOSTS sockets.
if [[ "${CPU_OFFLOAD_ALLOC_MODE}" == "numa-node" && -z "${SGLANG_GPU_PROC_CPUNODEBIND+x}" ]]; then
    SGLANG_GPU_PROC_CPUNODEBIND=$(CPU_NUM_HOSTS="${CPU_NUM_HOSTS}" TP_SIZE="${_TP_SIZE}" \
        CPU_OFFLOAD_RESERVED_SUBNUMA_IDS="${CPU_OFFLOAD_RESERVED_SUBNUMA_IDS}" python - <<'PY'
import os

num_sockets = int(os.environ["CPU_NUM_HOSTS"])
reserved = {}
for item in os.environ["CPU_OFFLOAD_RESERVED_SUBNUMA_IDS"].replace(";", ",").split(","):
    socket, node = item.strip().split("=", 1)
    reserved[int(socket.removeprefix("socket"))] = int(node)

if any(socket not in reserved for socket in range(num_sockets)):
    raise SystemExit("missing reserved sub-NUMA ID for a physical socket")
tp_size = int(os.environ["TP_SIZE"])
if tp_size % num_sockets:
    raise SystemExit("TP_SIZE not divisible by CPU_NUM_HOSTS")

ranks_per_socket = tp_size // num_sockets
print(",".join(
    str(reserved[socket])
    for socket in range(num_sockets)
    for _ in range(ranks_per_socket)
))
PY
)
    export SGLANG_GPU_PROC_CPUNODEBIND
fi
export ENABLE_NUMA=${ENABLE_NUMA:-0}
export SGLANG_CPU_OFFLOAD_RECORD_LOAD=${SGLANG_CPU_OFFLOAD_RECORD_LOAD:-1}
export SGLANG_CPU_OFFLOAD_RECORD_LOAD_SAMPLE=${SGLANG_CPU_OFFLOAD_RECORD_LOAD_SAMPLE:-32}

# ════════════════════════════════════════════════════════════
# §5  DERIVED: ACTIVE RANKS + CORE BUDGET  (do not edit directly)
# ════════════════════════════════════════════════════════════
# Each node owns _TP_SIZE/_NNODES ranks (_TP_SIZE set in §3 via TP_SIZE env var), split
# evenly across CPU_NUM_HOSTS physical sockets, e.g. TP_SIZE=8/_NNODES=2:
#   node 0: ranks  0-7  (socket 0: 0-3,  socket 1: 4-7)
#   node 1: ranks 8-15  (socket 0: 8-11, socket 1: 12-15)
_ranks_per_node=$(( _TP_SIZE / _NNODES ))                     # ranks owned by this node
_node_rank_start=$(( NODE_RANK * _ranks_per_node ))           # first global TP rank on this node
_node_socket_stride=$(( _ranks_per_node / CPU_NUM_HOSTS ))    # ranks per socket

# Build active-ranks list: CPU_ACTIVE_RANKS_PER_SOCKET ranks from each socket.
_cap_per_sock=$(( CPU_ACTIVE_RANKS_PER_SOCKET > _node_socket_stride ? _node_socket_stride : CPU_ACTIVE_RANKS_PER_SOCKET ))
(( _cap_per_sock < 1 )) && _cap_per_sock=1
if [[ -z "${CPU_OFFLOAD_ACTIVE_RANKS+x}" ]]; then
    _ar_list=""
    for (( _sock=0; _sock<CPU_NUM_HOSTS; _sock++ )); do
        _sock_start=$(( _node_rank_start + _sock * _node_socket_stride ))
        for (( _ri=0; _ri<_cap_per_sock; _ri++ )); do
            _ar_list+="${_ar_list:+,}$(( _sock_start + _ri ))"
        done
    done
    CPU_OFFLOAD_ACTIVE_RANKS="${_ar_list}"
fi
#CPU_OFFLOAD_ACTIVE_RANKS="2,3,6,7" 
#CPU_OFFLOAD_RANK_TO_WORKER_MAP="2:0,3:1,6:0,7:1"
# Derive CPU_WORKERS_PER_NUMA from how many active ranks fall in each socket
_socket_boundary=$(( _node_rank_start + _node_socket_stride ))
_ranks_per_socket_arr=(0 0)
IFS=',' read -ra _active_rank_array <<< "${CPU_OFFLOAD_ACTIVE_RANKS}"
for _r in "${_active_rank_array[@]}"; do
    _r=$(echo "${_r}" | xargs)
    if [[ ${_r} -lt ${_socket_boundary} ]]; then
        : $(( ++_ranks_per_socket_arr[0] ))
    else
        : $(( ++_ranks_per_socket_arr[1] ))
    fi
done
_min_workers_needed=$(( (_ranks_per_socket_arr[0] > _ranks_per_socket_arr[1]) \
                         ? _ranks_per_socket_arr[0] : _ranks_per_socket_arr[1] ))
(( _min_workers_needed < 1 )) && _min_workers_needed=1
CPU_WORKERS_PER_NUMA=${CPU_WORKERS_PER_NUMA:-${_min_workers_needed}}

# Detect physical core count for NUMA node 0 (fallback: nproc / sockets)
_CORES_PER_NUMA=$(python - <<'PY' 2>/dev/null || echo 0
import subprocess
try:
    out = subprocess.check_output(["lscpu", "-p=CPU,Core,Socket,Node"], text=True)
except Exception:
    print(0); raise SystemExit
node0 = set()
for line in out.splitlines():
    if line.startswith("#"): continue
    parts = line.split(",")
    if len(parts) >= 4 and parts[3] == "0":
        node0.add(parts[1])
print(len(node0))
PY
)
if [[ -z "${_CORES_PER_NUMA}" || "${_CORES_PER_NUMA}" -lt 2 ]]; then
    _CORES_PER_NUMA=$(( $(nproc) / CPU_NUM_HOSTS ))
fi

if [[ "${CPU_OFFLOAD_ALLOC_MODE}" == "core" ]]; then
    # Mode A — core: split daemon cores evenly among workers
    _DAEMON_CORES=$(( _CORES_PER_NUMA > CPU_RESERVED_CORES \
                      ? _CORES_PER_NUMA - CPU_RESERVED_CORES \
                      : _CORES_PER_NUMA ))
    _auto_cpw=$(( CPU_WORKERS_PER_NUMA > 0 \
                  ? _DAEMON_CORES / CPU_WORKERS_PER_NUMA \
                  : _DAEMON_CORES ))
    (( _auto_cpw < 1 )) && _auto_cpw=1
    export CPU_OFFLOAD_RESERVED_CORES_PER_NUMA=${CPU_OFFLOAD_RESERVED_CORES_PER_NUMA:-${CPU_RESERVED_CORES}}
else
    # Mode B — numa-node: worker owns the entire sub-NUMA node, use all its cores
    _DAEMON_CORES=${_CORES_PER_NUMA}
    _auto_cpw=${_CORES_PER_NUMA}
    export CPU_OFFLOAD_CORES_PER_NUMA=${CPU_OFFLOAD_CORES_PER_NUMA:-${_CORES_PER_NUMA}}
    export CPU_OFFLOAD_RESERVED_CORES_PER_NUMA=${CPU_OFFLOAD_RESERVED_CORES_PER_NUMA:-0}
fi

# Apply optimal-core cap: if auto allocation > sweet spot, clamp down.
# Rationale: OMP scaling saturates; using fewer threads reduces cache/NUMA contention.
_effective_cpw=${_auto_cpw}
if [[ ${CPU_EXPERTD_OPTIMAL_CORES} -gt 0 && ${_auto_cpw} -gt ${CPU_EXPERTD_OPTIMAL_CORES} ]]; then
    echo "[full94] INFO: cores_per_worker: allocated=${_auto_cpw} > optimal=${CPU_EXPERTD_OPTIMAL_CORES}; clamping to ${CPU_EXPERTD_OPTIMAL_CORES}"
    _effective_cpw=${CPU_EXPERTD_OPTIMAL_CORES}
fi

# Honor explicit user override (CPU_EXPERTD_CORES_PER_WORKER), hard-capped to _auto_cpw.
if [[ -n "${CPU_EXPERTD_CORES_PER_WORKER+x}" && ${CPU_EXPERTD_CORES_PER_WORKER} -gt 0 ]]; then
    if [[ ${CPU_EXPERTD_CORES_PER_WORKER} -gt ${_auto_cpw} ]]; then
        echo "[full94] WARN: CPU_EXPERTD_CORES_PER_WORKER=${CPU_EXPERTD_CORES_PER_WORKER} > allocated=${_auto_cpw}" \
             "(${CPU_OFFLOAD_ALLOC_MODE} mode); capping to ${_auto_cpw}"
        _effective_cpw=${_auto_cpw}
    else
        _effective_cpw=${CPU_EXPERTD_CORES_PER_WORKER}
    fi
fi

export CPU_EXPERTD_CORES_PER_WORKER=${_effective_cpw}

# In core mode cpu_expertd builds worker affinity masks from
# --cores-per-numa; CPU_EXPERTD_CORES_PER_WORKER only controls OpenMP threads.
# Keep the daemon core budget consistent with the effective per-worker cap so
# each worker receives a contiguous CPU range instead of the entire socket.
if [[ "${CPU_OFFLOAD_ALLOC_MODE}" == "core" ]]; then
    _effective_daemon_cores=$(( CPU_WORKERS_PER_NUMA * CPU_EXPERTD_CORES_PER_WORKER ))
    export CPU_OFFLOAD_CORES_PER_NUMA=${CPU_OFFLOAD_CORES_PER_NUMA:-${_effective_daemon_cores}}
fi

_expertd_cores=${CPU_EXPERTD_CORES_PER_WORKER}
export CPU_EXPERTD_OMP_NUM_THREADS=${CPU_EXPERTD_OMP_NUM_THREADS:-${_expertd_cores}}
export CPU_EXPERTD_KMP_HW_SUBSET=${CPU_EXPERTD_KMP_HW_SUBSET:-${_expertd_cores}c,1t}
export CPU_EXPERTD_KMP_AFFINITY=${CPU_EXPERTD_KMP_AFFINITY:-granularity=fine,compact,1,0}
export CPU_OFFLOAD_NUM_WORKERS_PER_NUMA=${CPU_OFFLOAD_NUM_WORKERS_PER_NUMA:-${CPU_WORKERS_PER_NUMA}}
export CPU_OFFLOAD_MIN_CORES_PER_WORKER=${CPU_OFFLOAD_MIN_CORES_PER_WORKER:-1}

# ════════════════════════════════════════════════════════════
# §6  PATHS
# ════════════════════════════════════════════════════════════
LAUNCHER="${WORKSPACE_ROOT}/moe-cpu-engine/launch_cpu_expertd.sh"
LOG_GPU=/tmp/bench_94layer_gpu_only.log
LOG_CPU=/tmp/bench_94layer_cpu_offload.log

# ════════════════════════════════════════════════════════════
# §7  SERVER & BENCH ARGS
# ════════════════════════════════════════════════════════════
COMMON_SERVER_ARGS=(
    --model "${MODEL_PATH}"
    --trust-remote-code
    --tp ${_TP_SIZE} --ep-size ${_TP_SIZE}
    --deepep-mode normal
    --attention-backend flashinfer
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --dist-init-addr "${DIST_INIT_ADDR}"
    --nnodes ${_NNODES}
    --node-rank ${NODE_RANK}
    --dist-timeout ${DIST_TIMEOUT_S}
    #--disable-cuda-graph
)
if [[ "${MOE_NUM_LAYERS}" -gt 0 ]]; then
    COMMON_SERVER_ARGS+=(--json-model-override-args "{\"num_hidden_layers\": ${MOE_NUM_LAYERS}}")
fi

# bench_one_batch args — extends COMMON_SERVER_ARGS (only used in bench_one_batch mode)
BENCH_ONE_BATCH_ARGS=("${COMMON_SERVER_ARGS[@]}")
if [[ "${BENCH_MODE}" == "bench_one_batch" ]]; then
    if [[ "${CORRECTNESS_TEST}" == "1" ]]; then
        # batch-size 4: covers 3 correctness prompts without OOM from large CUDA graph
        BENCH_ONE_BATCH_ARGS+=(--correctness-test --output-len "${OUTPUT_LEN}" --batch-size 4)
        if [[ "${CORRECTNESS_APPLY_CHAT_TEMPLATE}" == "1" ]]; then
            BENCH_ONE_BATCH_ARGS+=(--apply-chat-template)
        fi
    else
        # BATCH_SIZE may hold several space-separated values ("1 8 32"):
        # bench_one_batch takes nargs="+" and sweeps them in ONE process, so a
        # batch sweep costs one model load instead of one per batch size.
        # Callers passing several values must also set CUDA_GRAPH_MAX_BS
        # explicitly; the arithmetic default above only handles one value.
        # shellcheck disable=SC2206
        BENCH_ONE_BATCH_ARGS+=(
            --batch-size ${BATCH_SIZE}
            --input-len  "${INPUT_LEN}"
            --output-len "${OUTPUT_LEN}"
        )
    fi
    # BENCH_EXTRA_ARGS: verbatim extra bench_one_batch flags, e.g.
    #   BENCH_EXTRA_ARGS="--log-decode-step 1 --result-filename /tmp/r.jsonl"
    # Without --log-decode-step only the first 5 decode steps are printed,
    # which is too little to see the per-step effect of a slot re-selection.
    if [[ -n "${BENCH_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        BENCH_ONE_BATCH_ARGS+=(${BENCH_EXTRA_ARGS})
    fi
fi

# CPU offload extra args (Pass 2 only)
CPU_OFFLOAD_EXTRA_ARGS=(
    --enable-cpu-offload-fast-loop
    --cpu-active-ranks "${CPU_OFFLOAD_ACTIVE_RANKS}"
    --cpu-offload-decode-only true
    --cpu-offload-slot-update-interval-ms "${CPU_SLOT_UPDATE_MS}"
    --cpu-offload-standalone-daemon 1
    --cpu-offload-num-workers-per-numa "${CPU_WORKERS_PER_NUMA}"
    --cpu-offload-cores-per-numa "${CPU_OFFLOAD_CORES_PER_NUMA}"
    --cpu-offload-reserved-cores-per-numa "${CPU_OFFLOAD_RESERVED_CORES_PER_NUMA}"
    --cpu-offload-min-cores-per-worker "${CPU_OFFLOAD_MIN_CORES_PER_WORKER}"
    --cpu-offload-shm-prefix "${CPU_OFFLOAD_SHM_PREFIX}"
    --cpu-offload-daemon-proc-name "${CPU_OFFLOAD_DAEMON_PROC_NAME}"
    --cpu-offload-max-spin "${CPU_OFFLOAD_MAX_SPIN}"
    --cpu-offload-max-hits "${CPU_OFFLOAD_MAX_HITS}"
)

# Shared bench_serving client — called from node 0 in launch_server mode
_run_bench_serving() {
    local log_file=$1
    python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "${SERVER_PORT}" \
        --model "${MODEL_PATH}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --request-rate "${BENCH_REQUEST_RATE}" \
        2>&1 | tee -a "${log_file}"
}

# ════════════════════════════════════════════════════════════
# §8  STARTUP SUMMARY
# ════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════"
echo "[full94] Qwen3-235B-A22B-Thinking-2507-FP8 — full 94 layers"
echo "[full94] model=${MODEL_PATH}"
echo "[full94] batch=${BATCH_SIZE}  input_len=${INPUT_LEN}  output_len=${OUTPUT_LEN}"
if [[ "${CPU_OFFLOAD_ALLOC_MODE}" != "core" ]]; then
    _subnuma_reservation="reserved_ids=${CPU_OFFLOAD_RESERVED_SUBNUMA_IDS}"
    echo "[full94] cpu_worker_mode=${CPU_OFFLOAD_ALLOC_MODE}  workers_requested=${CPU_WORKERS_PER_NUMA}" \
         " cores_per_worker=${CPU_EXPERTD_CORES_PER_WORKER}  ${_subnuma_reservation}" \
         " OMP(daemon)=${CPU_EXPERTD_OMP_NUM_THREADS}"
else
    echo "[full94] cpu_worker_mode=core  cores/NUMA=${_CORES_PER_NUMA}" \
         " reserved=${CPU_RESERVED_CORES} daemon_cores=${_DAEMON_CORES}" \
         " workers=${CPU_WORKERS_PER_NUMA} per_worker=${CPU_EXPERTD_CORES_PER_WORKER}" \
         " OMP(daemon)=${CPU_EXPERTD_OMP_NUM_THREADS}"
fi
echo "[full94] cpu_active_ranks(node${NODE_RANK})=${CPU_OFFLOAD_ACTIVE_RANKS}  (ranks_per_sock=${_cap_per_sock}/${_node_socket_stride})"
echo "[full94] shm_prefix=${CPU_OFFLOAD_SHM_PREFIX}"
echo "[full94] cpu_engine: dtype=${MOE_DTYPE}  backend=${MOE_KERNEL_BACKEND}"
echo "[full94] nccl: SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>}  IB_DISABLE=${NCCL_IB_DISABLE}  DEBUG=${NCCL_DEBUG}"
echo "[full94] deepgemm: SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM}  (0=禁用DeepGEMM JIT，回退triton FP8)"
echo "[full94] logs: ${LOG_GPU}  ${LOG_CPU}"
echo "════════════════════════════════════════════════════════"

# ════════════════════════════════════════════════════════════
# §9  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════

# Kill all processes listening on a TCP port via /proc/net/tcp (no fuser/ss/lsof needed)
_release_tcp_port() {
    local _port=$1
    python - <<PYEOF 2>/dev/null || true
import os
port = ${_port}
port_hex = '%04X' % port
inodes = set()
for path in ('/proc/net/tcp', '/proc/net/tcp6'):
    try:
        for line in open(path).readlines()[1:]:
            parts = line.split()
            if len(parts) > 9 and parts[1].split(':')[1].upper() == port_hex and parts[3] == '0A':
                inodes.add(parts[9])
    except Exception:
        pass
for inode in inodes:
    for pid in os.listdir('/proc'):
        if not pid.isdigit(): continue
        try:
            for fd in os.listdir('/proc/' + pid + '/fd'):
                try:
                    if os.readlink('/proc/' + pid + '/fd/' + fd) == 'socket:[' + inode + ']':
                        os.kill(int(pid), 9)
                except OSError:
                    pass
        except OSError:
            pass
PYEOF
}

# Return 0 if something IS listening on the TCP port (no ss/netstat needed)
_tcp_port_listening() {
    python -c "
import socket as _s, sys
s = _s.socket()
s.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
try:
    s.bind(('', ${1})); s.listen(1); s.close(); sys.exit(1)  # free → not listening
except OSError:
    s.close(); sys.exit(0)  # in use → IS listening
" 2>/dev/null
}

# ── Lifecycle ─────────────────────────────────────────────────────────────────
preflight_cleanup() {
    local self_pid=$$
    echo "[full94] preflight: memory=$(free -g | awk '/^Mem:/ {print $3}')Gi used"
    local stale
    stale=$(pgrep -f 'sglang\.(bench_one_batch|launch_server|bench_serving)|cpu_expertd' 2>/dev/null \
            | grep -vw "${self_pid}" || true)
    if [[ -n "${stale}" ]]; then
        echo "[full94] preflight: killing stale PIDs: ${stale//$'\n'/ }"
        kill -TERM ${stale} 2>/dev/null || true; sleep 2
        kill -KILL ${stale} 2>/dev/null || true; sleep 1
    fi
    echo "[full94] preflight: releasing ports ${_DIST_INIT_PORT} / ${SERVER_PORT} (if held)..."
    for _stale_port in "${_DIST_INIT_PORT}" "${SERVER_PORT}"; do
        _release_tcp_port "${_stale_port}"
    done
    sleep 1
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
               | tr -d ' ' | grep -v "^$" || true)
    if [[ -n "${gpu_pids}" ]]; then
        local orphans
        orphans=$(echo "${gpu_pids}" | grep -vw "${self_pid}" || true)
        if [[ -n "${orphans}" ]]; then
            echo "[full94] preflight: killing orphaned GPU PIDs: ${orphans//$'\n'/ }"
            kill -KILL ${orphans} 2>/dev/null || true; sleep 2
        fi
    fi
    rm -f "${CPU_OFFLOAD_SHM_PREFIX}".socket*.{ready,slot*} 2>/dev/null || true
    find /dev/shm -maxdepth 1 -name '__KMP_REGISTERED_LIB_*' -delete 2>/dev/null || true
    local live_daemon
    live_daemon=$(pgrep -af 'cpu_expertd --socket|cpu_expertd_main --socket' 2>/dev/null \
                  | grep -vw "${self_pid}" || true)
    if [[ -z "${live_daemon}" && "${CPU_OFFLOAD_KEEP_WEIGHTS}" != "1" ]]; then
        local wc
        wc=$(find /dev/shm -maxdepth 1 -name "$(basename "${CPU_OFFLOAD_SHM_PREFIX}").socket*.layer*.full.weights" 2>/dev/null | wc -l || echo 0)
        if [[ "${wc}" -gt 0 ]]; then
            echo "[full94] preflight: purging ${wc} orphaned *.full.weights (~$((wc*23/10))GiB)"
            rm -f "${CPU_OFFLOAD_SHM_PREFIX}".socket*.layer*.full.weights 2>/dev/null || true
        fi
    fi
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo "[full94] preflight: memory=$(free -g | awk '/^Mem:/ {print $3}')Gi used (after)"
}
preflight_cleanup

kill_gpu_workers() {
    local self_pid=$$ gpu_pids orphans
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
               | tr -d ' ' | grep -v "^$" || true)
    [[ -z "${gpu_pids}" ]] && return 0
    orphans=$(echo "${gpu_pids}" | grep -vw "${self_pid}" || true)
    [[ -z "${orphans}" ]] && return 0
    echo "[full94] killing GPU workers: ${orphans//$'\n'/ }"
    kill -KILL ${orphans} 2>/dev/null || true; sleep 2
}

cleanup_exit() {
    echo "[full94] cleaning up daemons..."
    pkill -f "${CPU_OFFLOAD_DAEMON_PROC_NAME} --socket" 2>/dev/null || true
    pkill -f "cpu_expertd_main --socket" 2>/dev/null || true
    sleep 1
    if [[ -n "${CPU_NUMA_NODE_IDS:-}" ]]; then
        IFS=',' read -ra _cleanup_nodes <<< "${CPU_NUMA_NODE_IDS}"
    else
        _cleanup_nodes=()
        for _ci in $(seq 0 $((CPU_NUM_HOSTS - 1))); do _cleanup_nodes+=("${_ci}"); done
    fi
    for i in "${_cleanup_nodes[@]}"; do
        rm -f "${CPU_OFFLOAD_SHM_PREFIX}.socket${i}.ready"
    done
    if [[ "${CPU_OFFLOAD_KEEP_WEIGHTS}" != "1" ]]; then
        local wc
        wc=$(find /dev/shm -maxdepth 1 -name "$(basename "${CPU_OFFLOAD_SHM_PREFIX}").socket*.layer*.full.weights" 2>/dev/null | wc -l || echo 0)
        if [[ "${wc}" -gt 0 ]]; then
            echo "[full94] purging ${wc} *.full.weights files (~$((wc*23/10))GiB)"
            rm -f "${CPU_OFFLOAD_SHM_PREFIX}".socket*.layer*.full.weights 2>/dev/null || true
        fi
    fi
}
trap cleanup_exit EXIT

# ── Server helpers ────────────────────────────────────────────────────────────
_SERVER_PID=
_LOG_TAIL_PID=

_wait_server_ready() {
    local port=${1} log_file=${2:-} timeout=${SERVER_READY_TIMEOUT_S} elapsed=0
    local _dist_stuck_at=0 _dist_hint_shown=0
    echo "[full94] waiting for server on port ${port} (timeout ${timeout}s)..."
    echo "[full94] server log → ${log_file:-<no log>}  (tail it to watch progress)"
    while ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        sleep 5; elapsed=$(( elapsed + 5 ))
        if (( elapsed % 30 == 0 )); then
            local _last_line
            _last_line=$(tail -1 "${log_file}" 2>/dev/null || true)
            echo "[full94] still waiting... ${elapsed}s / ${timeout}s elapsed"
            echo "[full94] server tail: ${_last_line}"
            if [[ "${_last_line}" == *"Init torch distributed"* ]]; then
                if [[ ${_dist_stuck_at} -eq 0 ]]; then
                    _dist_stuck_at=${elapsed}
                elif (( elapsed - _dist_stuck_at >= 60 && !_dist_hint_shown )); then
                    _dist_hint_shown=1
                    echo "" >&2
                    echo "[full94] ⚠⚠  DIST INIT HANG: stuck at 'Init torch distributed' for $(( elapsed - _dist_stuck_at ))s" >&2
                    echo "[full94]  The server is waiting for ALL ${_NNODES} nodes to join the rendezvous." >&2
                    echo "[full94]  Common causes:" >&2
                    echo "[full94]    1. NODE_RANK=1 script is NOT running on the peer node" >&2
                    echo "[full94]       → Run on peer: NODE_RANK=1 bash dual-loop-benchark.sh" >&2
                    echo "[full94]    2. Firewall blocking port ${_DIST_INIT_PORT} between nodes" >&2
                    echo "[full94]       → Test from peer: python -c \"import socket; socket.create_connection(('${_DIST_INIT_HOST}',${_DIST_INIT_PORT}),5)\"" >&2
                    echo "[full94]    3. dist-init-addr wrong (override: DIST_INIT_ADDR=<node0-ip>:${_DIST_INIT_PORT})" >&2
                    if _tcp_port_listening "${_DIST_INIT_PORT}"; then
                        echo "[full94]  [node0] port ${_DIST_INIT_PORT} IS listening → network/firewall issue on node 1" >&2
                    else
                        echo "[full94]  [node0] port ${_DIST_INIT_PORT} NOT listening → rank 0 TCPStore not started yet" >&2
                    fi
                    echo "[full94]  Server will auto-exit after DIST_TIMEOUT_S=${DIST_TIMEOUT_S}s" >&2
                    echo "" >&2
                fi
            fi
        fi
        if [[ -n "${_SERVER_PID:-}" ]] && ! kill -0 "${_SERVER_PID}" 2>/dev/null; then
            echo "[full94] ERROR: server process ${_SERVER_PID} died unexpectedly" >&2
            if [[ -n "${log_file}" && -f "${log_file}" ]]; then
                echo "[full94] last 10 lines of server log:" >&2
                tail -10 "${log_file}" >&2 2>/dev/null || true
            fi
            return 1
        fi
        if (( elapsed >= timeout )); then
            echo "[full94] ERROR: server not ready after ${timeout}s" >&2
            if [[ -n "${log_file}" && -f "${log_file}" ]]; then
                echo "[full94] last 10 lines of server log:" >&2
                tail -10 "${log_file}" >&2 2>/dev/null || true
            fi
            return 1
        fi
    done
    echo "[full94] server ready on port ${port} (elapsed ${elapsed}s)"
}

_kill_server() {
    if [[ -n "${_LOG_TAIL_PID:-}" ]]; then
        kill "${_LOG_TAIL_PID}" 2>/dev/null || true
        wait "${_LOG_TAIL_PID}" 2>/dev/null || true
        _LOG_TAIL_PID=
    fi
    if [[ -n "${_SERVER_PID:-}" ]]; then
        local spid=${_SERVER_PID}
        echo "[full94] stopping server PID ${spid}..."
        kill -TERM "${spid}" 2>/dev/null || true
        local waited=0
        while kill -0 "${spid}" 2>/dev/null && (( waited < 15 )); do
            sleep 1; (( ++waited ))
        done
        if kill -0 "${spid}" 2>/dev/null; then
            echo "[full94] server did not exit after ${waited}s, sending SIGKILL"
            kill -KILL "${spid}" 2>/dev/null || true
        fi
        wait "${spid}" 2>/dev/null || true
        _SERVER_PID=
    fi
    kill_gpu_workers
}

# Node 1 is a pure worker — wait for its process to exit naturally.
# When node 0 finishes and kills its server, NCCL detects the disconnect
# and node 1's worker exits automatically.
_wait_worker_exit() {
    echo "[full94] node ${NODE_RANK}: worker started (PID ${_SERVER_PID}), waiting for it to exit..."
    local _n1_waited=0
    while kill -0 "${_SERVER_PID}" 2>/dev/null; do
        sleep 5; (( _n1_waited += 5 ))
        # At 30 s probe TCP connectivity to node 0 — diagnoses firewall / wrong IP
        if (( _n1_waited == 30 )); then
            echo "[full94] node ${NODE_RANK}: testing TCP to ${_DIST_INIT_HOST}:${_DIST_INIT_PORT}..."
            if python -c "import socket; socket.create_connection(('${_DIST_INIT_HOST}',${_DIST_INIT_PORT}),5)" 2>/dev/null; then
                echo "[full94] node ${NODE_RANK}: dist-init port REACHABLE ✓"
            else
                echo "[full94] ⚠ node ${NODE_RANK}: ${_DIST_INIT_HOST}:${_DIST_INIT_PORT} UNREACHABLE!" >&2
                echo "[full94]   → Check firewall, or set DIST_INIT_ADDR=<correct-ip>:${_DIST_INIT_PORT}" >&2
            fi
        fi
        (( _n1_waited % 60 == 0 )) && echo "[full94] node ${NODE_RANK}: worker still running (${_n1_waited}s)"
    done
    echo "[full94] node ${NODE_RANK}: worker process exited (${_n1_waited}s)"
    kill "${_LOG_TAIL_PID:-}" 2>/dev/null || true; wait "${_LOG_TAIL_PID:-}" 2>/dev/null || true
    _LOG_TAIL_PID=; _SERVER_PID=
}

# ════════════════════════════════════════════════════════════
# §10  NETWORK SANITY CHECK (node 0 only)
# ════════════════════════════════════════════════════════════
echo "[full94] node_rank=${NODE_RANK}/${_NNODES}  dist_init_addr=${DIST_INIT_ADDR}"
_my_ips=$(hostname -I 2>/dev/null | tr -s ' ' ' ' | sed 's/ $//' || true)
echo "[full94] local IPs: ${_my_ips}"
if [[ "${NODE_RANK}" == "0" ]] && [[ " ${_my_ips} " != *" ${_DIST_INIT_HOST} "* ]]; then
    echo "" >&2
    echo "[full94] ⚠⚠  NETWORK MISMATCH on node 0:" >&2
    echo "[full94]   DIST_INIT_HOST=${_DIST_INIT_HOST} is NOT in local IPs: ${_my_ips}" >&2
    echo "[full94]   Docker bridge detected — port ${_DIST_INIT_PORT} inside the container is" >&2
    echo "[full94]   NOT exposed on the host, so node 1 gets 'Connection refused'." >&2
    echo "[full94]   Fix (choose one):" >&2
    echo "[full94]     A) Restart both containers with --network=host (recommended)" >&2
    echo "[full94]        docker run --network=host --gpus all ..." >&2
    echo "[full94]     B) Add port mappings:" >&2
    echo "[full94]        docker run -p ${_DIST_INIT_PORT}:${_DIST_INIT_PORT} -p ${SERVER_PORT}:${SERVER_PORT} ..." >&2
    echo "[full94]     C) Set overlay IP: DIST_INIT_ADDR=<container-ip>:${_DIST_INIT_PORT}" >&2
    echo "" >&2
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — GPU ONLY
# ══════════════════════════════════════════════════════════════════════════════
if [[ "${CPU_OFFLOAD}" != "1" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "[full94] PASS 1: GPU-only (no CPU offload)"
    echo "[full94] log → ${LOG_GPU}"
    echo "════════════════════════════════════════════════════════"

    if [[ "${BENCH_MODE}" == "launch_server" ]]; then
        echo "[full94] mode=launch_server  port=${SERVER_PORT}  prompts=${BENCH_NUM_PROMPTS}  rate=${BENCH_REQUEST_RATE}"
        > "${LOG_GPU}"
        python -m sglang.launch_server "${COMMON_SERVER_ARGS[@]}" --port "${SERVER_PORT}" \
            >> "${LOG_GPU}" 2>&1 &
        _SERVER_PID=$!
        tail -F "${LOG_GPU}" & _LOG_TAIL_PID=$!
        if [[ "${NODE_RANK}" == "0" ]]; then
            _wait_server_ready "${SERVER_PORT}" "${LOG_GPU}"
            kill "${_LOG_TAIL_PID}" 2>/dev/null || true; wait "${_LOG_TAIL_PID}" 2>/dev/null || true; _LOG_TAIL_PID=
            if [[ "${RUN_CLIENT}" == "1" ]]; then
                _run_bench_serving "${LOG_GPU}"
            else
                echo "[full94] RUN_CLIENT=0: server is up on port ${SERVER_PORT}."
                echo "[full94] Run your own client against it now; Ctrl+C here to stop the server."
                wait "${_SERVER_PID}" || true
            fi
        else
            _wait_worker_exit
        fi
        _kill_server
    else
        python -m sglang.bench_one_batch "${BENCH_ONE_BATCH_ARGS[@]}" 2>&1 | tee "${LOG_GPU}"
        kill_gpu_workers
    fi

    echo ""
    echo "[full94] ── Pass 1 summary ──────────────────────────────"
    grep -E 'Prefill|Decode|throughput|latency|median|mean|[Tt]ok/s' "${LOG_GPU}" | tail -20 || true
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — CPU OFFLOAD
# ══════════════════════════════════════════════════════════════════════════════
if [[ "${GPU_ONLY}" != "1" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "[full94] PASS 2: CPU offload (daemon, phase-2 weights preload)"
    if [[ "${MOE_NUM_LAYERS}" -gt 0 ]]; then
        echo "[full94] Daemon will preload ${MOE_NUM_LAYERS}-layer expert weights (truncated)"
        export SGLANG_NUM_HIDDEN_LAYERS_OVERRIDE="${MOE_NUM_LAYERS}"
    else
        echo "[full94] Daemon will preload ~426 GiB expert weights to /dev/shm"
    fi
    echo "[full94] First cold run: up to ${CPU_EXPERTD_READY_TIMEOUT_S}s"
    echo "[full94] log → ${LOG_CPU}"
    echo "════════════════════════════════════════════════════════"

    echo "[full94] launching cpu_expertd daemons (sockets=${CPU_NUM_HOSTS})..."
    "${LAUNCHER}" "${CPU_NUM_HOSTS}"
    echo "[full94] daemons ready"
    ps -eo pid,ppid,comm | awk 'NR==1 || /cpu_expertd/' || true

    # cpu_slot = last GPU-owned expert of each rank → combine is non-aliased (shadow_state strategy 2.5)
    export CPU_OFFLOAD_USE_LAST_GPU_SLOT=0

    if [[ "${BENCH_MODE}" == "launch_server" ]]; then
        echo "[full94] mode=launch_server  port=${SERVER_PORT}  prompts=${BENCH_NUM_PROMPTS}  rate=${BENCH_REQUEST_RATE}"
        > "${LOG_CPU}"
        python -m sglang.launch_server "${COMMON_SERVER_ARGS[@]}" --port "${SERVER_PORT}" \
            "${CPU_OFFLOAD_EXTRA_ARGS[@]}" \
            >> "${LOG_CPU}" 2>&1 &
        _SERVER_PID=$!
        tail -F "${LOG_CPU}" & _LOG_TAIL_PID=$!
        if [[ "${NODE_RANK}" == "0" ]]; then
            _wait_server_ready "${SERVER_PORT}" "${LOG_CPU}"
            kill "${_LOG_TAIL_PID}" 2>/dev/null || true; wait "${_LOG_TAIL_PID}" 2>/dev/null || true; _LOG_TAIL_PID=
            if [[ "${RUN_CLIENT}" == "1" ]]; then
                _run_bench_serving "${LOG_CPU}"
            else
                echo "[full94] RUN_CLIENT=0: server is up on port ${SERVER_PORT}."
                echo "[full94] Run your own client against it now; Ctrl+C here to stop the server."
                wait "${_SERVER_PID}" || true
            fi
        else
            _wait_worker_exit
        fi
        _kill_server
    else
        python -m sglang.bench_one_batch "${BENCH_ONE_BATCH_ARGS[@]}" \
            "${CPU_OFFLOAD_EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_CPU}"
        kill_gpu_workers
    fi

    echo ""
    echo "[full94] ── Pass 2 summary ──────────────────────────────"
    grep -E 'Prefill|Decode|throughput|latency|median|mean|[Tt]ok/s|METRICS|combine_success' "${LOG_CPU}" | tail -30 || true
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if [[ "${GPU_ONLY}" != "1" && "${CPU_OFFLOAD}" != "1" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "[full94] COMPARISON SUMMARY"
    echo "════════════════════════════════════════════════════════"
    echo "[full94] GPU-only  key metrics:"
    grep -E 'Prefill|Decode|median|[Tt]ok/s' "${LOG_GPU}" | grep -v 'Using\|config\|not found' | tail -10 || echo "  (see ${LOG_GPU})"
    echo ""
    echo "[full94] CPU-offload key metrics:"
    grep -E 'Prefill|Decode|median|[Tt]ok/s' "${LOG_CPU}" | grep -v 'Using\|config\|not found' | tail -10 || echo "  (see ${LOG_CPU})"
    echo ""
    echo "[full94] CPU offload routing (from all ranks):"
    grep 'METRICS' "${LOG_CPU}" | grep -v 'token_expert_pairs total=0' | tail -8 || true
fi

echo ""
echo "[full94] done."
