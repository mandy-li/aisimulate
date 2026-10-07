# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) DeepEP high-throughput all-to-all collector (op ``moe_a2a``).

XPU port of ``collector/wideep/vllm/collect_moe_a2a.py`` for single-node CRI
hosts without RDMA NICs. Launch one rank per local XPU with torchrun::

    ZE_AFFINITY_MASK=0,1 torchrun --standalone --nproc-per-node 2 \\
        collector/vllm/collect_moe_a2a_xpu.py --gpus-per-node 2 --output-path <dir>

Add ``--debug [--guard-bdfs 0000:05:00.0,0000:09:00.0]`` while investigating
device failures (see "Debug mode" below).

Rows land in the unified ``moe_a2a_perf`` table with the CUDA collector's
row builder and key (``_build_moe_a2a_row``), so the two backends cannot
drift. Differences from the CUDA collector:

* Only ``deepep_ht`` within one node. DeepSymm's ``deep_ep_xpu`` intranode
  path moves data with PCIe peer-to-peer writes into IPC-mapped buffers and
  needs no RDMA; LL, internode HT and V2 need iSHMEM/IBGDA plus a NIC per GPU.
* DeepSymm's ``deep_ep_xpu.Buffer`` is driven directly (layout -> dispatch ->
  combine), not through vLLM's ``DeepEPHTPrepareAndFinalize``: the CRI image's
  vLLM fork has not been verified to wire DeepEP HT to ``deep_ep_xpu``. The
  timed dispatch includes ``get_dispatch_layout``, as vLLM's HT ``prepare``
  does. ``runtime_meta.transport.path`` records ``deepsymm_direct_buffer``.
* A fixed, conservative channel config (``XPU_HT_NUM_EUS`` work-group pairs,
  send/recv chunk tokens below) persisted as ``sms``. On 2026-10-06 a tuner
  candidate with 12 channels x 512 ring slots (6144 in-flight slots) wedged a
  CRI card; every completed candidate was <= 4096 slots. The fixed config
  stays far below that (``MAX_INFLIGHT_SLOTS``).
* World sizes 2, 4 and 8 (one node, at most ``NUM_MAX_PCIE_PEERS`` = 8). EP=2
  is for development only: it stages the CSV but never finalizes a parquet or
  sidecar. Remove it before publishing.
* Shapes whose local experts exceed DeepSymm's notify-kernel limit (128) are
  skipped at plan time, and token counts are capped by ``--max-tokens``
  (default: the largest count validated on CRI).

Safety, because a hung DeepEP kernel can wedge a CRI card and take a shared
host's GPU offline:

* Every case runs an int32 correctness round with the same bytes per row
  before timing. DeepSymm's spin-waits give up silently after 1e6 polls, so
  a broken run can otherwise exit cleanly with plausible timings.
* A per-case watchdog ends the process (SIGTERM, letting DeepSymm drain its
  queues, then a hard exit) instead of letting a hang continue.
* The first failure stops the whole run on every rank; later cases are not
  attempted.
* The DeepEP buffer is sized from ``Config.get_pcie_buffer_size_hint``
  (DeepSymm does not bounds-check intranode buffers) and destroyed explicitly
  after a barrier.
* A failed case is written to the rank's error file before teardown, and
  teardown after a failure does not touch the (possibly lost) device.

Debug mode (``--debug``), for diagnosing device hangs and faults:

* Rank 0 prints a timestamped, flushed line when each case starts, finishes
  or fails, so a crash or hang always names the case.
* With ``--guard-bdfs`` (comma-separated PCI BDFs of this run's cards, or
  ``AIC_A2A_GUARD_BDFS``): each rank's device BDF must match the list, and the
  kernel log is checked for new xe error/fault/reset/wedge/timeout lines on
  those cards around buffer setup and every case. Needs a readable ``dmesg``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector import provenance
from collector.framework_manifest import get_collector_runtime
from collector.helper import finalize_perf_files, log_perf, stale_output_artifacts
from collector.registry_types import PerfFile
from collector.wideep.distributed_lifecycle import StageAgreement, agree_stage, raise_for_stage
from collector.wideep.sglang.collect_moe_a2a import (
    DistIdentity,
    MoeA2AShape,
    PhaseTiming,
    _build_moe_a2a_row,
    derive_dist_identity,
    get_moe_a2a_workload_grid,
)
from collector.wideep.vllm.collect_moe_a2a import (
    BenchmarkResult,
    CaseFailure,
    VllmMoeA2ABenchmarkError,
    VllmMoeA2ACase,
    VllmMoeA2ADeclarationError,
    VllmMoeA2APeerError,
    _git_collector_ref,
    _row_key,
    get_vllm_moe_a2a_shapes,
)

MODULE_NAME = "collector.vllm.collect_moe_a2a_xpu"
OP_NAME = "moe_a2a"
FRAMEWORK = "vLLM"
MANIFEST_FRAMEWORK = "vllm_xpu"
KERNEL_SOURCE = "deepep"
COMM_BACKEND = "deepep_ht"
INFERENCE_PHASE = "context"
COMM_DTYPE = "default"
PHASES = ("combine", "dispatch")
TRANSPORT_PATH = "deepsymm_direct_buffer"

# EP=2 is development-only (stages CSV, never finalizes). Remove before publishing.
DEV_ONLY_WORLD_SIZES = (2,)
SUPPORTED_WORLD_SIZES = (2, 4, 8)
SUPPORTED_NODE_COUNTS = (1,)

# Fixed conservative HT channel config (deep_ep_xpu.Config). num_channels = num_eus / 2.
XPU_HT_NUM_EUS = 12
XPU_HT_SEND_TOKENS = 32
XPU_HT_RECV_TOKENS = 64
# In-flight ring slots per peer = channels * recv tokens. 6144 wedged a CRI card on
# 2026-10-06; every completed tuner candidate was <= 4096.
MAX_INFLIGHT_SLOTS = 3072
# DeepSymm NotifyDispatchKernel asserts num_experts / num_ranks <= 128.
MAX_LOCAL_EXPERTS = 128
# Largest per-rank token count validated (int32 verify, no driver faults) on CRI.
DEFAULT_MAX_TOKENS = 4096
NUM_BUFFER_ALIGNMENT_BYTES = 128
# Size the buffer above the hint; DeepSymm does not bounds-check intranode rings.
BUFFER_HEADROOM = 2
# int32 verify values are rank * stride + token; combine reduces in fp32, so the
# largest sum, (world_size * stride) * world_size, must stay below 2**24.
VERIFY_RANK_STRIDE = 100_000
DEFAULT_CASE_TIMEOUT_S = 120.0
WATCHDOG_GRACE_S = 30.0
DRIVER_EVENT_PATTERN = re.compile(r"error|fault|reset|wedge|timeout", re.IGNORECASE)
ERRORS_FILENAME_TEMPLATE = "errors_moe_a2a_vllm_xpu.rank{rank}.json"
STATS_FILENAME_TEMPLATE = "moe_a2a_vllm_xpu_stats.rank{rank}.json"

assert XPU_HT_NUM_EUS % 2 == 0
assert XPU_HT_SEND_TOKENS < XPU_HT_RECV_TOKENS
assert XPU_HT_NUM_EUS // 2 * XPU_HT_RECV_TOKENS <= MAX_INFLIGHT_SLOTS
assert (max(SUPPORTED_WORLD_SIZES) * VERIFY_RANK_STRIDE) * max(SUPPORTED_WORLD_SIZES) < 2**24


class DriverFaultError(VllmMoeA2ABenchmarkError):
    """The kernel log reported new device errors on a guarded card."""


class VerificationError(VllmMoeA2ABenchmarkError):
    """The int32 dispatch/combine round returned wrong data."""


@dataclass(frozen=True)
class PlanSkip:
    shape: MoeA2AShape
    reason: str


@dataclass(frozen=True)
class XpuCaseStats:
    """Per-rank transfer volume for one case (informational, not a parquet column)."""

    num_tokens: int
    hidden_size: int
    remote_tokens: int
    remote_bytes: int
    dispatch_gbps: float
    combine_gbps: float


@dataclass
class XpuCollectionResult:
    rows: list[dict[str, Any]]
    failures: list[CaseFailure]
    resolved_cases: list[VllmMoeA2ACase]
    not_run: list[VllmMoeA2ACase] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def build_xpu_case_plan(
    *,
    shapes: list[MoeA2AShape],
    grid: dict[str, list[int]],
    world_size: int,
    node_num: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[VllmMoeA2ACase], list[PlanSkip]]:
    """Build the deterministic single-node HT plan and the shapes it skips."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(
            f"vLLM-XPU moe_a2a supports world sizes {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if node_num not in SUPPORTED_NODE_COUNTS:
        raise VllmMoeA2ADeclarationError(
            f"vLLM-XPU moe_a2a is intranode only (nodes {SUPPORTED_NODE_COUNTS}), got {node_num}"
        )
    if max_tokens <= 0:
        raise VllmMoeA2ADeclarationError(f"--max-tokens must be positive, got {max_tokens}")
    raw_tokens = grid.get("ht_token_counts")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise VllmMoeA2ADeclarationError("ht_token_counts must be a non-empty list")
    tokens = sorted({int(value) for value in raw_tokens if 0 < int(value) <= max_tokens})
    if not tokens:
        raise VllmMoeA2ADeclarationError(f"no ht_token_counts <= --max-tokens={max_tokens}")
    if tokens[-1] >= VERIFY_RANK_STRIDE:
        raise VllmMoeA2ADeclarationError(f"token counts must stay below {VERIFY_RANK_STRIDE} for int32 verification")

    cases: list[VllmMoeA2ACase] = []
    skipped: list[PlanSkip] = []
    for shape in shapes:
        if shape.num_experts % world_size:
            raise VllmMoeA2ADeclarationError(
                f"shape {shape} is not divisible by world_size={world_size}; request the EP constraint "
                "from get_vllm_moe_a2a_shapes instead of filtering a generated plan"
            )
        local_experts = shape.num_experts // world_size
        if local_experts > MAX_LOCAL_EXPERTS:
            skipped.append(PlanSkip(shape, f"{local_experts} local experts > DeepSymm limit {MAX_LOCAL_EXPERTS}"))
            continue
        if shape.hidden_size % 8:
            skipped.append(PlanSkip(shape, f"hidden_size={shape.hidden_size} is not a multiple of 8 (int4 rows)"))
            continue
        cases.extend(
            VllmMoeA2ACase(COMM_BACKEND, INFERENCE_PHASE, shape, num_tokens, XPU_HT_NUM_EUS, 0) for num_tokens in tokens
        )
    cases.sort(key=VllmMoeA2ACase.sort_key)
    keys = [case.persisted_key(ep_size=world_size, node_num=node_num) for case in cases]
    if len(keys) != len(set(keys)):
        raise VllmMoeA2ADeclarationError("duplicate vLLM-XPU DeepEP persisted key")
    return cases, skipped


def case_plan_ids(cases: list[VllmMoeA2ACase], *, world_size: int, node_num: int) -> list[str]:
    ids = []
    for case in cases:
        payload = {
            "comm_backend": case.comm_backend,
            "config": {
                "num_eus": XPU_HT_NUM_EUS,
                "recv_tokens": XPU_HT_RECV_TOKENS,
                "send_tokens": XPU_HT_SEND_TOKENS,
            },
            "ep_size": world_size,
            "hidden_size": case.shape.hidden_size,
            "node_num": node_num,
            "num_experts": case.shape.num_experts,
            "num_tokens": case.num_tokens,
            "routing": {
                "has_correction_bias": case.shape.routing.has_correction_bias,
                "method_type": case.shape.routing.method_type,
                "num_expert_group": case.shape.routing.num_expert_group,
                "renormalize": case.shape.routing.renormalize,
                "routed_scaling_factor": case.shape.routing.routed_scaling_factor,
                "scoring_func": case.shape.routing.scoring_func,
                "topk_group": case.shape.routing.topk_group,
            },
            "sms": case.sms,
            "topk": case.shape.topk,
            "transport": TRANSPORT_PATH,
        }
        ids.append(f"{MODULE_NAME}:benchmark:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ids


def select_canary_cases(cases: list[VllmMoeA2ACase]) -> list[VllmMoeA2ACase]:
    """Smallest, middle and largest token count of the first shape."""
    if not cases:
        return []
    first = [case for case in cases if case.shape == cases[0].shape]
    picks = {first[0], first[len(first) // 2], first[-1]}
    return sorted(picks, key=VllmMoeA2ACase.sort_key)


def effective_gbps(remote_bytes: int, latency_us: float) -> float:
    """Per-rank, one-direction effective bandwidth: remote bytes / stage time."""
    if latency_us <= 0:
        return 0.0
    return remote_bytes / (latency_us * 1e-6) / 1e9


# ---------------------------------------------------------------------------
# Safety: kernel-log guard and per-case watchdog
# ---------------------------------------------------------------------------


def read_kernel_log() -> list[str]:
    result = subprocess.run(["dmesg"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


class DriverFaultGuard:
    """Detect new xe error/fault/reset/wedge/timeout lines for the guarded cards."""

    def __init__(self, bdfs: list[str], reader: Callable[[], list[str]] = read_kernel_log):
        if not bdfs:
            raise VllmMoeA2ADeclarationError("DriverFaultGuard needs at least one BDF")
        self.bdfs = tuple(bdfs)
        self.reader = reader

    def events(self) -> list[str]:
        return [
            line
            for line in self.reader()
            if any(bdf in line for bdf in self.bdfs) and DRIVER_EVENT_PATTERN.search(line)
        ]

    def snapshot(self) -> int:
        return len(self.events())

    def check(self, baseline: int, *, context: str) -> None:
        events = self.events()
        if len(events) != baseline:
            tail = "\n".join(events[-10:])
            raise DriverFaultError(
                f"{len(events) - baseline:+d} kernel-log device events on {list(self.bdfs)} during {context}:\n{tail}"
            )


def parse_bdfs(raw: str | None) -> list[str]:
    values = [value.strip().lower() for value in (raw or "").split(",") if value.strip()]
    pattern = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
    bad = [value for value in values if not pattern.match(value)]
    if bad:
        raise VllmMoeA2ADeclarationError(f"invalid --guard-bdfs entries {bad}; expected e.g. 0000:05:00.0")
    return values


def local_device_bdf(device_index: int) -> str:
    """PCI address of visible XPU ``device_index`` (after ZE_AFFINITY_MASK), via DeepSymm's SYCL query."""
    from deep_symm.tools.topology import device_bdf_from_sycl

    return str(device_bdf_from_sycl(device_index)).lower()


def check_device_bdfs(rank_bdfs: list[Any], guard_bdfs: list[str]) -> None:
    """The ranks' devices must be exactly the guarded cards, or the guard watches the wrong log lines."""
    observed = [str(bdf or "").lower() for bdf in rank_bdfs]
    if any(not bdf for bdf in observed):
        raise VllmMoeA2ADeclarationError(f"could not resolve every rank's device BDF: {observed}")
    if len(set(observed)) != len(observed):
        raise VllmMoeA2ADeclarationError(f"ranks share a device: {observed}")
    if set(observed) != set(guard_bdfs):
        raise VllmMoeA2ADeclarationError(
            f"rank devices {observed} do not match --guard-bdfs {sorted(guard_bdfs)}; check ZE_AFFINITY_MASK"
        )


class CaseWatchdog:
    """End the process if one case runs past its deadline.

    SIGTERM first so deep_ep_xpu's Python signal handler can drain the GPU
    queues (it only runs once the main thread returns from native code), then
    a hard exit after a grace period if the main thread never returns.
    """

    def __init__(self, timeout_s: float, on_expire: Callable[[str], None] | None = None):
        self.timeout_s = timeout_s
        self.on_expire = on_expire

    @contextmanager
    def arm(self, label: str):
        if self.timeout_s <= 0:
            yield
            return
        timer = threading.Timer(self.timeout_s, self._expire, args=(label,))
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _expire(self, label: str) -> None:
        message = f"[vllm-xpu moe_a2a] watchdog: {label} exceeded {self.timeout_s:.0f}s; terminating"
        print(message, file=sys.stderr, flush=True)
        if self.on_expire is not None:
            try:
                self.on_expire(message)
            except Exception as error:
                print(f"[vllm-xpu moe_a2a] watchdog record failed: {error}", file=sys.stderr, flush=True)
        hard = threading.Timer(WATCHDOG_GRACE_S, os._exit, args=(124,))
        hard.daemon = True
        hard.start()
        os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Collection loop (pure; GPU-free tests drive it with a fake adapter)
# ---------------------------------------------------------------------------


def collect_with_adapter(
    cases: list[VllmMoeA2ACase],
    *,
    adapter: Any,
    world_size: int,
    node_num: int,
    agreement: StageAgreement,
    guard: DriverFaultGuard | None = None,
    watchdog: CaseWatchdog | None = None,
    on_failure: Callable[[CaseFailure], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> XpuCollectionResult:
    """Verify then time each case; stop every rank at the first failure.

    ``on_failure`` persists a failed case before teardown, so a teardown error
    on a lost device cannot hide which case failed.
    """
    result = XpuCollectionResult([], [], [])
    prepare_error: BaseException | None = None
    try:
        baseline = guard.snapshot() if guard is not None else 0
        adapter.prepare(cases)
        if guard is not None:
            guard.check(baseline, context="buffer setup")
    except BaseException as error:
        prepare_error = error
    raise_for_stage(
        agree_stage("adapter_prepare", prepare_error, agreement=agreement, peer_error_type=VllmMoeA2APeerError)
    )
    for case_index, case in enumerate(cases):
        case_rows: list[dict[str, Any]] = []
        local_error: BaseException | None = None
        shape = case.shape
        label = f"case {case_index} ({shape.hidden_size}/{shape.topk}/{shape.num_experts}, {case.num_tokens} tok)"
        if log is not None:
            log(f"[{case_index + 1}/{len(cases)}] start {label}")
        try:
            baseline = guard.snapshot() if guard is not None else 0
            arm = watchdog.arm(label) if watchdog is not None else _null_context()
            with arm:
                adapter.verify(case)
                bench = adapter.benchmark(case)
            if guard is not None:
                guard.check(baseline, context=label)
            case_rows = _rows_for(case, bench, world_size=world_size, node_num=node_num)
        except Exception as error:
            local_error = error
        outcome = agree_stage(
            f"case:{case_index}:benchmark", local_error, agreement=agreement, peer_error_type=VllmMoeA2APeerError
        )
        if outcome.failed:
            assert outcome.error is not None
            failure = CaseFailure(case, type(outcome.error).__name__, str(outcome.error))
            result.failures.append(failure)
            result.not_run = cases[case_index + 1 :]
            if log is not None:
                log(f"[{case_index + 1}/{len(cases)}] FAILED {label}: {failure.error_type}: {failure.error}")
            if on_failure is not None:
                on_failure(failure)
            break
        if log is not None:
            log(f"[{case_index + 1}/{len(cases)}] done {label}")
        result.resolved_cases.append(case)
        result.rows.extend(case_rows)
    close_error: BaseException | None = None
    try:
        adapter.close(failed=bool(result.failures))
    except BaseException as error:
        close_error = error
    raise_for_stage(
        agree_stage("adapter_close", close_error, agreement=agreement, peer_error_type=VllmMoeA2APeerError)
    )
    return result


@contextmanager
def _null_context():
    yield


def _rows_for(case: VllmMoeA2ACase, bench: BenchmarkResult, *, world_size: int, node_num: int) -> list[dict[str, Any]]:
    if bench.sms != XPU_HT_NUM_EUS:
        raise VllmMoeA2ABenchmarkError(f"adapter returned sms={bench.sms}, expected {XPU_HT_NUM_EUS}")
    if set(bench.timings) != set(PHASES):
        raise VllmMoeA2ABenchmarkError(f"adapter returned phases {sorted(bench.timings)}, expected {list(PHASES)}")
    return [
        _build_moe_a2a_row(
            comm_backend=case.persisted_backend,
            phase=phase,
            ep_size=world_size,
            node_num=node_num,
            shape=case.shape,
            num_tokens=case.num_tokens,
            sms=bench.sms,
            transmit_us=bench.timings[phase].transmit_us,
            notify_us=bench.timings[phase].notify_us,
            comm_dtype=COMM_DTYPE,
        )
        for phase in PHASES
    ]


# ---------------------------------------------------------------------------
# XPU runtime (imports torch / deep_ep_xpu lazily)
# ---------------------------------------------------------------------------


def build_topk_ids(torch: Any, case: VllmMoeA2ACase, *, rank: int, device: str) -> Any:
    """Deterministic unique routes; mirrors ``VllmBenchmarkAdapter._build_topk_ids`` on ``device``."""
    routing = case.shape.routing
    seed_payload = (rank, case.shape.hidden_size, case.shape.topk, case.shape.num_experts, case.num_tokens, routing)
    seed = int.from_bytes(hashlib.sha256(repr(seed_payload).encode()).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(
        (case.num_tokens, case.shape.num_experts), device=device, dtype=torch.float32, generator=generator
    )
    if routing.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif routing.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits).sqrt()
    else:
        scores = torch.softmax(logits, dim=-1)
    selection_scores = scores
    if routing.has_correction_bias:
        correction = torch.randn((case.shape.num_experts,), device=device, dtype=torch.float32, generator=generator)
        selection_scores = scores + correction
    if routing.num_expert_group > 1:
        experts_per_group = case.shape.num_experts // routing.num_expert_group
        grouped = scores.view(case.num_tokens, routing.num_expert_group, experts_per_group)
        if routing.has_correction_bias:
            grouped_for_selection = selection_scores.view(case.num_tokens, routing.num_expert_group, experts_per_group)
            group_scores = grouped_for_selection.topk(2, dim=-1).values.sum(dim=-1)
        else:
            group_scores = grouped.max(dim=-1).values
        selected_groups = group_scores.topk(routing.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        selection_scores = selection_scores.masked_fill(
            ~group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape_as(scores),
            float("-inf"),
        )
    topk_ids = selection_scores.topk(case.shape.topk, dim=-1).indices.to(torch.int64)
    ordered = topk_ids.sort(dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise VllmMoeA2ABenchmarkError("routing generator produced duplicate experts for one token")
    return topk_ids


def expected_int32_combine(torch: Any, x: Any, is_token_in_rank: Any) -> Any:
    """Identity experts: combine returns each token times the ranks it was sent to."""
    copies = is_token_in_rank.sum(dim=1).to(x.dtype)
    return x * copies.unsqueeze(1)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class XpuDeepEPDirectAdapter:
    """Drive ``deep_ep_xpu.Buffer`` HT dispatch/combine directly on XPU."""

    def __init__(
        self,
        group,
        cpu_group,
        identity: DistIdentity,
        *,
        warmups: int = 5,
        runs: int = 20,
    ):
        self.group = group
        self.cpu_group = cpu_group
        self.identity = identity
        self.warmups = warmups
        self.runs = runs
        self.device = f"xpu:{identity.local_rank}"
        self._buffer = None
        self.pcie_bytes: int | None = None
        self.stats: list[XpuCaseStats] = []
        self.runtime_capability: dict[str, str] | None = None

    def _configs(self):
        import deep_ep_xpu

        dispatch = deep_ep_xpu.Config(XPU_HT_NUM_EUS, XPU_HT_SEND_TOKENS, XPU_HT_RECV_TOKENS)
        combine = deep_ep_xpu.Config(XPU_HT_NUM_EUS, XPU_HT_SEND_TOKENS, XPU_HT_RECV_TOKENS)
        return dispatch, combine

    def prepare(self, cases: list[VllmMoeA2ACase]) -> None:
        """Allocate one long-lived buffer sized for the largest declared row."""
        import deep_ep_xpu

        if not cases:
            raise VllmMoeA2ADeclarationError("cannot prepare an empty vLLM-XPU moe_a2a case plan")
        world = self.identity.world_size
        max_hidden_bytes = max(case.shape.hidden_size for case in cases) * 2  # bf16 == int32 at hidden/2
        hint = max(int(config.get_pcie_buffer_size_hint(max_hidden_bytes, world)) for config in self._configs())
        self.pcie_bytes = _align_up(hint * BUFFER_HEADROOM, NUM_BUFFER_ALIGNMENT_BYTES)
        deep_ep_xpu.Buffer.set_num_eus(XPU_HT_NUM_EUS)
        self._buffer = deep_ep_xpu.Buffer(
            self.group,
            num_pcie_bytes=self.pcie_bytes,
            num_rdma_bytes=0,
            low_latency_mode=False,
            explicitly_destroy=True,
        )
        num_rdma_ranks = int(self._buffer.runtime.get_num_rdma_ranks())
        if num_rdma_ranks != 1:
            raise VllmMoeA2ABenchmarkError(
                f"expected an intranode buffer, DeepSymm reports {num_rdma_ranks} RDMA ranks"
            )
        self.runtime_capability = {
            "backend": COMM_BACKEND,
            "topology_source": "deepsymm_intranode_pcie",
            "num_scaleout_ranks": "1",
            "num_scaleup_ranks": str(world),
            "num_pcie_bytes": str(self.pcie_bytes),
        }

    def close(self, *, failed: bool = False) -> None:
        if self._buffer is None:
            return
        if failed:
            # After a failed case the device may be lost: any synchronize or
            # destroy would raise again and mask the original error. The
            # process exits non-zero right after, which releases the buffer.
            self._buffer = None
            return
        import torch
        import torch.distributed as dist

        torch.xpu.synchronize()
        dist.barrier(group=self.cpu_group)
        self._buffer.destroy()
        self._buffer = None

    def _round(self, torch: Any, x: Any, topk_ids: Any, topk_weights: Any, num_experts: int, *, timed: bool):
        import torch.distributed as dist

        dispatch_config, combine_config = self._configs()
        torch.xpu.synchronize()
        dist.barrier(group=self.cpu_group)
        if timed:
            start_dispatch = torch.xpu.Event(enable_timing=True)
            end_dispatch = torch.xpu.Event(enable_timing=True)
            start_combine = torch.xpu.Event(enable_timing=True)
            end_combine = torch.xpu.Event(enable_timing=True)
            start_dispatch.record()
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
            self._buffer.get_dispatch_layout(topk_ids, num_experts)
        )
        recv_x, _, recv_topk_weights, _, handle, _ = self._buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            config=dispatch_config,
        )
        if timed:
            end_dispatch.record()
        # Stand in for expert output without charging the copy to combine.
        expert_output = recv_x.clone()
        if timed:
            start_combine.record()
        combined_x, _, _ = self._buffer.combine(
            x=expert_output, handle=handle, topk_weights=recv_topk_weights, config=combine_config
        )
        if timed:
            end_combine.record()
        torch.xpu.synchronize()
        elapsed = None
        if timed:
            elapsed = (start_dispatch.elapsed_time(end_dispatch), start_combine.elapsed_time(end_combine))
        return combined_x, is_token_in_rank, num_tokens_per_rank, elapsed

    def _routes(self, torch: Any, case: VllmMoeA2ACase):
        topk_ids = build_topk_ids(torch, case, rank=self.identity.rank, device=self.device)
        topk_weights = torch.full(topk_ids.shape, 1.0 / case.shape.topk, device=self.device, dtype=torch.float32)
        return topk_ids, topk_weights

    def verify(self, case: VllmMoeA2ACase) -> None:
        """One int32 round with the bf16 row's byte size; exact match required."""
        import torch

        topk_ids, topk_weights = self._routes(torch, case)
        hidden_int32 = case.shape.hidden_size // 2
        values = self.identity.rank * VERIFY_RANK_STRIDE + torch.arange(
            case.num_tokens, device=self.device, dtype=torch.int32
        )
        x = values.unsqueeze(1).expand(case.num_tokens, hidden_int32).contiguous()
        combined, is_token_in_rank, _, _ = self._round(
            torch, x, topk_ids, topk_weights, case.shape.num_experts, timed=False
        )
        expected = expected_int32_combine(torch, x, is_token_in_rank)
        if combined.shape != expected.shape or not torch.equal(combined, expected):
            bad = int((combined != expected).any(dim=1).sum().item()) if combined.shape == expected.shape else -1
            raise VerificationError(
                f"int32 dispatch/combine mismatch: {bad} of {case.num_tokens} tokens wrong "
                f"(shape {tuple(combined.shape)} vs {tuple(expected.shape)})"
            )

    def benchmark(self, case: VllmMoeA2ACase) -> BenchmarkResult:
        import torch

        torch.manual_seed(17 + self.identity.rank)
        tokens = torch.randn((case.num_tokens, case.shape.hidden_size), device=self.device, dtype=torch.bfloat16)
        topk_ids, topk_weights = self._routes(torch, case)
        for _ in range(self.warmups):
            self._round(torch, tokens, topk_ids, topk_weights, case.shape.num_experts, timed=False)
        dispatch_ms = combine_ms = 0.0
        num_tokens_per_rank = None
        for _ in range(self.runs):
            _, _, num_tokens_per_rank, elapsed = self._round(
                torch, tokens, topk_ids, topk_weights, case.shape.num_experts, timed=True
            )
            dispatch_ms += elapsed[0]
            combine_ms += elapsed[1]
        dispatch_us = dispatch_ms * 1000.0 / self.runs
        combine_us = combine_ms * 1000.0 / self.runs
        per_rank = num_tokens_per_rank.cpu().tolist()
        remote_tokens = sum(count for rank, count in enumerate(per_rank) if rank != self.identity.rank)
        remote_bytes = remote_tokens * case.shape.hidden_size * 2
        self.stats.append(
            XpuCaseStats(
                num_tokens=case.num_tokens,
                hidden_size=case.shape.hidden_size,
                remote_tokens=remote_tokens,
                remote_bytes=remote_bytes,
                dispatch_gbps=effective_gbps(remote_bytes, dispatch_us),
                combine_gbps=effective_gbps(remote_bytes, combine_us),
            )
        )
        return BenchmarkResult(
            timings={"dispatch": PhaseTiming(dispatch_us, 0.0), "combine": PhaseTiming(combine_us, 0.0)},
            sms=XPU_HT_NUM_EUS,
            capacity=case.capacity,
        )


# ---------------------------------------------------------------------------
# Runtime provenance and persistence
# ---------------------------------------------------------------------------


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd, timeout=10)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def observe_xpu_runtime(bdfs: list[str]) -> dict[str, str]:
    """Best-effort live runtime facts (no attestation: the CRI image is a fork build)."""
    import deep_ep_xpu
    import torch

    deepsymm_root = Path(deep_ep_xpu.__file__).resolve().parents[1]
    observed = {
        "torch": str(torch.__version__),
        "deep_ep_xpu": str(Path(deep_ep_xpu.__file__).resolve()),
        "deepsymm_commit": _command_output(["git", "rev-parse", "HEAD"], cwd=deepsymm_root),
        "kernel": platform.release(),
        "xe_module": _command_output(["modinfo", "-F", "version", "xe"]),
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", ""),
        "guard_bdfs": ",".join(bdfs),
    }
    try:
        from vllm.version import __version__ as vllm_version

        observed["vllm"] = str(vllm_version)
    except ImportError:
        observed["vllm"] = "not installed"
    return observed


def resolve_runtime_meta(live_abi: dict[str, str], pcie_bytes: int | None) -> dict[str, Any]:
    runtime = get_collector_runtime(MANIFEST_FRAMEWORK)
    image, _, digest = runtime.image().partition("@")
    meta: dict[str, Any] = {"framework": runtime.framework, "version": runtime.version, "image": image}
    if digest:
        meta["image_digest"] = digest
    meta["live_abi"] = live_abi
    meta["transport"] = {
        "path": TRANSPORT_PATH,
        "num_eus": XPU_HT_NUM_EUS,
        "send_tokens": XPU_HT_SEND_TOKENS,
        "recv_tokens": XPU_HT_RECV_TOKENS,
        "num_pcie_bytes": pcie_bytes,
        "failure_agreement": "gloo_cpu",
    }
    return meta


def _write_rows(rows: list[dict[str, Any]], *, perf_path: Path, version: str, device_name: str) -> None:
    if perf_path.exists():
        raise VllmMoeA2ABenchmarkError(f"stale staging file exists at {perf_path}; resume/merge must be explicit")
    for row in rows:
        if not log_perf(
            item_list=[row],
            framework=FRAMEWORK,
            version=version,
            device_name=device_name,
            op_name=OP_NAME,
            kernel_source=KERNEL_SOURCE,
            perf_filename=str(perf_path),
        ):
            raise VllmMoeA2ABenchmarkError(f"write loss: log_perf rejected row key {_row_key(row)}")
    with perf_path.open(newline="") as handle:
        persisted = list(csv.DictReader(handle))
    if len(persisted) != len(rows):
        raise VllmMoeA2ABenchmarkError(f"write loss: emitted {len(rows)} rows but staging file has {len(persisted)}")


def _failure_record(failure: CaseFailure, identity: DistIdentity) -> dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "op": OP_NAME,
        "classification": "unexpected",
        "error_type": failure.error_type,
        "error": failure.error,
        "rank": identity.rank,
        "case": {
            "comm_backend": failure.case.comm_backend,
            "ep_size": identity.world_size,
            "node_num": identity.node_num,
            "hidden_size": failure.case.shape.hidden_size,
            "topk": failure.case.shape.topk,
            "num_experts": failure.case.shape.num_experts,
            "num_tokens": failure.case.num_tokens,
            "sms": failure.case.sms,
        },
    }


def _stage_record(identity: DistIdentity, stage: str, error_type: str, error: str) -> dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "op": OP_NAME,
        "classification": "unexpected",
        "stage": stage,
        "error_type": error_type,
        "error": error,
        "rank": identity.rank,
        "case": None,
    }


def _append_rank_error(output_dir: Path, identity: DistIdentity, record: dict[str, Any]) -> Path:
    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=identity.rank)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, indent=2))
    return path


def _write_sidecar(
    output_dir: Path,
    *,
    runtime_meta: dict[str, Any],
    case_ids: list[str],
    parquet_path: Path,
    failure_count: int,
) -> Path:
    import pyarrow.parquet as pq

    closures = provenance.load_closures(_REPO_ROOT / "collector" / "hash_closures.yaml")
    table = {
        "collector_ref": _git_collector_ref(_REPO_ROOT),
        "collector_hash": provenance.collector_hash(MODULE_NAME, _REPO_ROOT, closures),
        "case_plan_hash": provenance.case_plan_hash(case_ids),
        "collected_at": date.today().isoformat(),
        "rows": pq.read_metadata(parquet_path).num_rows,
        "classified_failures": failure_count,
        "status": provenance.STATUS_PARTIAL if failure_count else provenance.STATUS_COMPLETE,
    }
    return provenance.write_collection_meta(output_dir, runtime_meta, {Path(PerfFile.MOE_A2A.value).stem: table})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--debug", action="store_true", help="per-case progress lines; enables --guard-bdfs")
    parser.add_argument(
        "--guard-bdfs",
        default=os.environ.get("AIC_A2A_GUARD_BDFS", ""),
        help="comma-separated PCI BDFs of this run's cards; their new kernel-log errors stop the run",
    )
    parser.add_argument("--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT_S)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--world-size", type=int, help="world size for --plan-only")
    return parser.parse_args(argv)


def _plan(args: argparse.Namespace, identity: DistIdentity):
    cases, skipped = build_xpu_case_plan(
        shapes=get_vllm_moe_a2a_shapes(required_expert_parallel_size=identity.world_size),
        grid=get_moe_a2a_workload_grid(),
        world_size=identity.world_size,
        node_num=identity.node_num,
        max_tokens=args.max_tokens,
    )
    if args.canary:
        cases = select_canary_cases(cases)
    if not cases:
        raise VllmMoeA2ADeclarationError("vLLM-XPU moe_a2a plan is empty after skips")
    return cases, skipped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env = dict(os.environ)
    if args.plan_only and args.world_size is not None:
        env["WORLD_SIZE"] = str(args.world_size)
    identity = derive_dist_identity(env, gpus_per_node=args.gpus_per_node)
    cases, skipped = _plan(args, identity)
    ids = case_plan_ids(cases, world_size=identity.world_size, node_num=identity.node_num)
    if identity.rank == 0:
        for skip in skipped:
            print(f"[vllm-xpu moe_a2a] skip shape {skip.shape}: {skip.reason}", flush=True)
    if args.plan_only:
        summary = {
            "cases": len(cases),
            "shapes": sorted({(c.shape.hidden_size, c.shape.topk, c.shape.num_experts) for c in cases}),
            "token_counts": sorted({c.num_tokens for c in cases}),
            "skipped_shapes": [
                f"{s.shape.hidden_size}/{s.shape.topk}/{s.shape.num_experts}: {s.reason}" for s in skipped
            ],
            "dev_only": identity.world_size in DEV_ONLY_WORLD_SIZES,
            "case_plan_hash": provenance.case_plan_hash(ids),
        }
        print(json.dumps(summary, indent=2))
        return

    bdfs = parse_bdfs(args.guard_bdfs)
    if bdfs and not args.debug:
        raise VllmMoeA2ADeclarationError("--guard-bdfs (or AIC_A2A_GUARD_BDFS) is a debug-mode option; add --debug")
    guard = DriverFaultGuard(bdfs) if bdfs else None
    if guard is not None:
        try:
            guard.snapshot()
        except (OSError, subprocess.CalledProcessError) as error:
            raise VllmMoeA2ADeclarationError(
                f"--guard-bdfs needs a readable kernel log (`dmesg` failed: {error}); "
                "drop --guard-bdfs to run without it"
            ) from error

    # Runtime indices must follow BDF order so ZE_AFFINITY_MASK matches --guard-bdfs.
    os.environ.setdefault("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "1")
    import torch
    import torch.distributed as dist

    identity = derive_dist_identity(
        dict(os.environ), gpus_per_node=args.gpus_per_node, visible_device_count=torch.xpu.device_count()
    )
    if identity.node_num not in SUPPORTED_NODE_COUNTS or identity.world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(
            f"vLLM-XPU collector requires one node and world size {SUPPORTED_WORLD_SIZES}; "
            f"got nodes={identity.node_num}, world={identity.world_size}"
        )
    dev_only = identity.world_size in DEV_ONLY_WORLD_SIZES
    torch.xpu.set_device(identity.local_rank)
    # env:// reuses torchrun's agent store (an explicit tcp:// server would collide with it).
    dist.init_process_group(backend="xccl", world_size=identity.world_size, rank=identity.rank)
    ranks = list(range(identity.world_size))
    group = dist.new_group(ranks, backend="xccl")
    cpu_group = dist.new_group(ranks, backend="gloo")
    output_dir = Path(args.output_path)
    if guard is not None:
        local_bdf = local_device_bdf(identity.local_rank)
        bdf_list: list[Any] = [None] * identity.world_size
        dist.all_gather_object(bdf_list, local_bdf, group=cpu_group)
        check_device_bdfs(bdf_list, bdfs)
    print(
        f"[vllm-xpu moe_a2a] host={socket.gethostname()} rank={identity.rank}/{identity.world_size} "
        f"cases={len(cases)} dev_only={dev_only}",
        flush=True,
    )

    def agree(stage: str, failed: bool) -> bool:
        failure = torch.tensor([int(failed)], device="cpu", dtype=torch.int64)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=cpu_group)
        return bool(failure.item())

    def log_progress(message: str) -> None:
        if identity.rank == 0:
            print(f"[vllm-xpu moe_a2a] {datetime.now():%H:%M:%S} {message}", flush=True)

    def record_watchdog(message: str) -> None:
        _append_rank_error(output_dir, identity, _stage_record(identity, "watchdog", "Timeout", message))

    run_failed = False
    try:
        preflight_error: BaseException | None = None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            stale = stale_output_artifacts(output_dir, PerfFile.MOE_A2A.value)
            if stale:
                raise VllmMoeA2ABenchmarkError(f"refusing stale output artifacts in {output_dir}: {', '.join(stale)}")
        except BaseException as error:
            preflight_error = error
        raise_for_stage(agree_stage("preflight", preflight_error, agreement=agree, peer_error_type=VllmMoeA2APeerError))

        adapter = XpuDeepEPDirectAdapter(group, cpu_group, identity, warmups=args.warmups, runs=args.runs)
        result = collect_with_adapter(
            cases,
            adapter=adapter,
            world_size=identity.world_size,
            node_num=identity.node_num,
            agreement=agree,
            guard=guard,
            watchdog=CaseWatchdog(args.case_timeout, on_expire=record_watchdog),
            on_failure=lambda failure: _append_rank_error(output_dir, identity, _failure_record(failure, identity)),
            log=log_progress if args.debug else None,
        )
        (output_dir / STATS_FILENAME_TEMPLATE.format(rank=identity.rank)).write_text(
            json.dumps([stat.__dict__ for stat in adapter.stats], indent=2)
        )
        if identity.rank == 0:
            for failure in result.failures:
                print(
                    f"[vllm-xpu moe_a2a] STOPPED at {failure.case}: {failure.error_type}: {failure.error}",
                    file=sys.stderr,
                    flush=True,
                )
            if result.not_run:
                print(f"[vllm-xpu moe_a2a] {len(result.not_run)} later cases not run", file=sys.stderr, flush=True)
            for stat in adapter.stats:
                print(
                    f"[vllm-xpu moe_a2a] hidden={stat.hidden_size} tokens={stat.num_tokens} "
                    f"remote={stat.remote_bytes / 1e6:.1f}MB dispatch={stat.dispatch_gbps:.1f}GB/s "
                    f"combine={stat.combine_gbps:.1f}GB/s",
                    flush=True,
                )
        if not result.rows:
            raise VllmMoeA2ABenchmarkError("no case completed; nothing to write")

        runtime_meta = resolve_runtime_meta(observe_xpu_runtime(bdfs), adapter.pcie_bytes)
        runtime_meta["backend_capability"] = adapter.runtime_capability
        perf_path = output_dir / PerfFile.MOE_A2A.value
        write_error: BaseException | None = None
        parquet_path: Path | None = None
        sidecar: Path | None = None
        if identity.rank == 0:
            try:
                _write_rows(
                    result.rows,
                    perf_path=perf_path,
                    version=runtime_meta["version"],
                    device_name=torch.xpu.get_device_name(identity.local_rank),
                )
                failure_count = len(result.failures) + len(result.not_run)
                if not dev_only:
                    converted = finalize_perf_files([perf_path], merge_existing=False)
                    if len(converted) != 1:
                        raise VllmMoeA2ABenchmarkError("finalization did not produce exactly one parquet")
                    parquet_path = Path(converted[0])
                    sidecar = _write_sidecar(
                        output_dir,
                        runtime_meta=runtime_meta,
                        case_ids=ids,
                        parquet_path=parquet_path,
                        failure_count=failure_count,
                    )
                (output_dir / "runtime_meta_moe_a2a_vllm_xpu.json").write_text(json.dumps(runtime_meta, indent=2))
            except BaseException as error:
                write_error = error
        raise_for_stage(agree_stage("row_write", write_error, agreement=agree, peer_error_type=VllmMoeA2APeerError))
        dist.barrier(group=cpu_group)
        if identity.rank == 0:
            if dev_only:
                print(
                    f"[vllm-xpu moe_a2a] dev world size {identity.world_size}: staged {len(result.rows)} rows "
                    f"at {perf_path}; parquet and collection_meta.yaml are not finalized",
                    flush=True,
                )
            else:
                print(f"[vllm-xpu moe_a2a] wrote {parquet_path} and {sidecar}", flush=True)
        if result.failures:
            raise VllmMoeA2ABenchmarkError(f"run stopped at the first failure: {result.failures[0].error_type}")
    except BaseException as error:
        run_failed = True
        try:
            recorded = _append_rank_error(
                output_dir, identity, _stage_record(identity, "fatal_runtime", type(error).__name__, str(error))
            )
            print(f"[vllm-xpu moe_a2a] recorded failure in {recorded}: {error}", file=sys.stderr, flush=True)
        except Exception as record_error:
            print(f"[vllm-xpu moe_a2a] failed to record {error!r}: {record_error}", file=sys.stderr, flush=True)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as error:
                if not run_failed:
                    raise
                print(f"[vllm-xpu moe_a2a] process-group destroy after failure also failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
