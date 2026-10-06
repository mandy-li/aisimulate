# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) large-EP MoE expert-compute collector (op ``moe_ep``).

XPU port of ``collector/wideep/vllm/collect_moe_ep.py`` (WideEP
``wideep_context_moe`` / ``wideep_generation_moe``). Simulates an EP world of
``moe_ep_size`` ranks on one XPU by allocating only rank 0's expert shard
(block-scaled FP8) and timing the fused-experts kernel vLLM-XPU serving runs,
``vllm_xpu_kernels.fused_moe_interface.XpuFusedMoe`` -- the same op the CRI
``moe`` collector (``collect_moe_xpu.py``) drives.

Rows land in the unified ``moe_expert_compute_perf`` table with the CUDA
collector's exact payload (``_build_moe_ep_row``) and ``kernel_source`` label,
imported rather than re-declared so the two backends cannot drift. The sweep
(model allowlist, EP sizes, per-phase token counts, distributions) is declared
in ``cases/base_ops/moe.yaml`` under ``common_case_values.moe_ep``; model
dimensions come from the models' ``vllm_xpu_cri`` MoE rows. Token counts are
GLOBAL (the persisted ``num_tokens`` key, = per-rank * ep) and are benchmarked
as-is.

Differences from the CUDA bench, all mirroring vLLM's DeepEP high-throughput
serving path into ``XPUExperts`` (``fused_moe/experts/xpu_moe.py``):

* Only tokens with at least one rank-0 expert reach the kernel -- DeepEP HT
  dispatch delivers only those to a rank. ``XpuFusedMoe`` sizes every
  intermediate (remapped input, GEMM1/act/GEMM2 outputs) by ``rows * topk``
  regardless of locality, so feeding all global rows (as the CUDA bench does)
  would inflate memory and elementwise time ~ep-fold. The persisted
  ``num_tokens`` stays the GLOBAL count.
* Expert ids are global; unrouted slots get ``num_experts - 1`` (a non-local
  expert for rank 0), as upstream ``prepare_finalize/deepep_ht.py`` rewrites
  DeepEP's ``-1``. ``XpuFusedMoe``'s expert map (``init_expert_map``, linear
  placement = vLLM's default) maps any non-local id to local ``-1``.
* Weights and activations follow the CRI ``moe`` collector
  (``collect_moe_xpu.py``), verified against the installed vLLM-XPU image
  (a fork build, not upstream v0.28.0):

  - Block-FP8 weights are passed in vLLM's loaded ``[E, N, K]`` layout with
    ``[E, N/128, K/128]`` scales, untransposed. The image's ``XpuFusedMoe``
    reads ``inter_size`` from ``w13.shape[-2]`` for this input.
  - Activations are block-quantized to FP8 (``quant_fp8_block_act``) outside
    the timed region and passed with ``a1q_scale``: the image's
    ``XPUExperts`` sets ``expects_unquantized_inputs = not is_xe3p``, so on
    CRI (Xe3P) vLLM quantizes inputs in prepare, before DeepEP dispatch.
* Points whose estimated ``XpuFusedMoe`` intermediates exceed a fraction of
  device memory are skipped before launch. On CRI such points do not raise a
  catchable OOM: they end in ``UR_RESULT_ERROR_DEVICE_LOST``, which kills the
  worker and fails every later case on it. Measured: DSv4-Pro ep=128 power_law
  context ran at 376,656 kernel inputs and was lost at 2,394,090. Like the
  GDN collector's grid-y pre-skip, these are printed and skipped (with the
  larger token counts of that phase/distribution series), not raised. A real
  OOM means the estimate was too low: it also skips the rest of its series,
  but the case raises at the end listing those points.
"""

__compat__ = "vllm==0.28.0"

import gc
import os

import torch
from vllm.version import __version__ as vllm_version
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
from vllm_xpu_kernels.moe_utils import quant_fp8_block_act

from collector.case_generator import get_xpu_moe_ep_test_cases
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    power_law_deepep_decode,
    power_law_deepep_prefill,
    xpu_graph_measure_enabled,
)
from collector.vllm.collect_moe_xpu import create_fp8_block_weights_xpu, resolve_moe_activation
from collector.wideep.vllm.collect_moe_ep import (
    MOE_EP_KERNEL_SOURCE,
    MOE_EP_OP_NAME,
    MOE_EP_QUANT_MODE,
    MoeEpBenchmarkError,
    _build_moe_ep_row,
    _moe_expert_compute_perf_path,
    _power_columns,
)

_DISTRIBUTIONS = ("uniform", "power_law")

# Fraction of device memory the estimated per-launch intermediates may use.
# Headroom covers weights, allocator slack and the second buffer set an
# XPU-graph capture holds.
_MEM_FRACTION = float(os.getenv("aic_moe_ep_mem_fraction", "0.5"))  # noqa: SIM112


class _PointExceedsMemoryBudget(Exception):
    """A token point is pre-skipped: launching it would exhaust device memory."""


def get_moe_ep_test_cases():
    """CRI large-EP MoE compute cases as positional per-case lists.

    Returns:
        list[list]: ``[num_local_experts, moe_ep_size, hidden_size, inter_size,
        topk, num_experts, num_slots, moe_dtype, model_name,
        context_token_counts, generation_token_counts, distributions]`` per
        case, ``distributions`` being ``[name, power_law_alpha]`` pairs.
    """
    return [
        [
            case.num_experts // case.ep,
            case.ep,
            case.hidden_size,
            case.inter_size,
            case.topk,
            case.num_experts,
            case.num_experts,  # num_slots: no EPLB redundancy axis on vllm
            MOE_EP_QUANT_MODE,
            case.model_name,
            case.context_token_counts,
            case.generation_token_counts,
            [[name, alpha] for name, alpha in case.token_expert_distributions],
        ]
        for case in get_xpu_moe_ep_test_cases()
    ]


def _phase_points(token_counts, distributions):
    """(distributed, power_law_alpha, global_num_tokens) points, sorted like the CUDA sweep (D5)."""
    points = []
    for name, alpha in distributions:
        if name not in _DISTRIBUTIONS:
            raise MoeEpBenchmarkError(f"moe_ep[vllm_xpu] distribution {name!r} not in {_DISTRIBUTIONS}")
        points.extend((name, alpha, int(num_tokens)) for num_tokens in token_counts)
    return sorted(points, key=lambda p: (p[0], p[1] if p[1] is not None else -1.0, p[2]))


def _make_xpu_fused_moe_bench(
    *,
    num_local_experts: int,
    moe_ep_size: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    model_name: str,
    device,
):
    """Build the ``bench(phase, global_num_tokens, distributed, alpha)`` callable for one case."""
    # [E, N, K] fp8 + [E, N/128, K/128] scales, passed as-is (the CRI image's
    # XpuFusedMoe takes this layout for block-FP8; see the module docstring).
    w13, w2, w13_scales, w2_scales, local_num_experts, padded_hidden = create_fp8_block_weights_xpu(
        num_experts, hidden_size, inter_size, 1, moe_ep_size, device
    )
    if local_num_experts != num_local_experts:
        raise MoeEpBenchmarkError(
            f"determine_expert_map gave {local_num_experts} local experts, case declares {num_local_experts}"
        )
    fused_moe_impl = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=None,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=None,
        n_experts_per_token=topk,
        activation=resolve_moe_activation(model_name),
        num_experts=num_local_experts,
        ep_rank=0,
        ep_size=moe_ep_size,
    )
    def _to_kernel_routing(topk_ids, topk_weights):
        """Rank-local routing -> what ``XPUExperts`` receives after DeepEP HT dispatch.

        Keeps only rows routed to a rank-0 expert, and rewrites ``-1`` to
        ``num_experts - 1`` as ``deepep_ht.py`` does for rank 0 (rank 0's
        local ids already equal its global ids, so no offset is added).
        """
        topk_ids = topk_ids.to(device=device, dtype=torch.int64)
        topk_weights = topk_weights.to(device=device, dtype=torch.float32)
        recv_rows = (topk_ids >= 0).any(dim=1).nonzero().flatten()
        if recv_rows.numel() == 0:
            # Nothing routed here: keep one unrouted row (zero-token input is not exercised).
            recv_rows = recv_rows.new_zeros(1)
        topk_ids = topk_ids[recv_rows]
        topk_ids = torch.where(topk_ids == -1, num_experts - 1, topk_ids).contiguous()
        return topk_ids, topk_weights[recv_rows].contiguous()

    def _counts_to_topk_ids(tokens_per_local_expert, global_num_tokens: int):
        """Spread per-local-expert token counts over a [global, topk] id grid (CUDA twin)."""
        flat = torch.repeat_interleave(
            torch.arange(num_local_experts, dtype=torch.int64), tokens_per_local_expert.to(torch.int64)
        )[: global_num_tokens * topk]
        if flat.numel() < global_num_tokens * topk:
            flat = torch.nn.functional.pad(flat, (0, global_num_tokens * topk - flat.numel()), value=-1)
        topk_ids = flat.reshape(global_num_tokens, topk)
        topk_weights = torch.where(topk_ids >= 0, 1.0 / topk, 0.0).to(torch.float32)
        return _to_kernel_routing(topk_ids, topk_weights)

    def _routing(inference_phase: str, global_num_tokens: int, distributed: str, power_law_alpha):
        """Rank-local routing over the GLOBAL token count; same arithmetic as the CUDA collector."""
        if distributed == "uniform":
            tokens_per_local_expert = global_num_tokens * topk // num_experts
            counts = torch.full((num_local_experts,), tokens_per_local_expert, dtype=torch.int64)
            if tokens_per_local_expert == 0:
                counts[: max(global_num_tokens * topk // moe_ep_size, 1)] = 1
            return _counts_to_topk_ids(counts, global_num_tokens)
        if inference_phase == "context":
            topk_idx, topk_weights, _ = power_law_deepep_prefill(
                global_num_tokens, num_experts, topk, moe_ep_size, power_law_alpha
            )
            return _to_kernel_routing(topk_idx, topk_weights)
        tokens_per_local_expert = power_law_deepep_decode(
            global_num_tokens, num_experts, topk, moe_ep_size, power_law_alpha
        )
        return _counts_to_topk_ids(tokens_per_local_expert, global_num_tokens)

    mem_budget = _MEM_FRACTION * get_device_module().get_device_properties(device).total_memory

    def _estimated_bytes(num_recv_tokens: int) -> int:
        """Upper estimate of what one apply allocates; every buffer is sized by rows * topk."""
        num_moe_inputs = num_recv_tokens * topk
        per_input = (
            padded_hidden * (1 + 2 + 2)  # fp8 remapped input, its bf16 dequant, bf16 GEMM2 output
            + inter_size * (2 * 2 + 2 + 1 + 2)  # bf16 GEMM1 output, bf16 act, fp8 requant + bf16 dequant
        )
        return num_moe_inputs * per_input + num_recv_tokens * padded_hidden * (1 + 2)  # fp8 input, bf16 output

    def bench(inference_phase: str, global_num_tokens: int, distributed: str, power_law_alpha):
        # Sampling and timing match the CRI moe collector (collect_moe_xpu.py):
        # power_law times 5 independent routing draws back to back in one call
        # (1 warmup / 1 run) and reports the per-draw average; uniform is
        # deterministic, so 1 draw with 3 warmups / 6 runs.
        num_draws = 5 if distributed == "power_law" else 1
        num_warmups, num_runs = (1, 1) if distributed == "power_law" else (3, 6)
        draws = [_routing(inference_phase, global_num_tokens, distributed, power_law_alpha) for _ in range(num_draws)]
        max_recv_tokens = max(topk_ids.shape[0] for topk_ids, _ in draws)
        estimated = _estimated_bytes(max_recv_tokens)
        if estimated > mem_budget:
            raise _PointExceedsMemoryBudget(
                f"{max_recv_tokens} rows x topk {topk} -> ~{estimated / 1e9:.1f} GB > "
                f"budget {mem_budget / 1e9:.1f} GB (aic_moe_ep_mem_fraction={_MEM_FRACTION})"
            )
        hidden_states = torch.randn(max_recv_tokens, padded_hidden, dtype=torch.bfloat16, device=device)
        # Xe3P serving quantizes in prepare (before dispatch), so outside the timed region.
        act_hidden_states, a1q_scale = quant_fp8_block_act(hidden_states)
        del hidden_states
        outputs = [
            torch.empty(topk_ids.shape[0], padded_hidden, dtype=torch.bfloat16, device=device) for topk_ids, _ in draws
        ]

        def kernel_func():
            for (topk_ids, topk_weights), output in zip(draws, outputs, strict=True):
                num_recv_tokens = topk_ids.shape[0]
                fused_moe_impl.apply(
                    output=output,
                    hidden_states=act_hidden_states[:num_recv_tokens],
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    a1q_scale=a1q_scale[:num_recv_tokens],
                )

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=num_warmups,
            num_runs=num_runs,
            repeat_n=1,
            use_cuda_graph=xpu_graph_measure_enabled(),
            allow_graph_fail=False,  # graph mandatory when enabled; capture failure fails the case
        ) as results:
            pass
        return results["latency_ms"] / num_draws, results["power_stats"]

    return bench


def run_moe_ep_torch(
    num_local_experts,
    moe_ep_size,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    num_slots,
    moe_dtype,
    model_name,
    context_token_counts,
    generation_token_counts,
    distributions,
    *,
    perf_filename,
    device="xpu:0",
    bench=None,
    output_path=None,
):
    """Run one declared CRI large-EP MoE compute case through ``XpuFusedMoe``.

    A failing token point raises ``MoeEpBenchmarkError`` with the case
    parameters. The only skips are printed memory-budget pre-skips and
    OOM'd series, which are raised at the end of the case.
    """
    if moe_dtype != MOE_EP_QUANT_MODE:
        raise MoeEpBenchmarkError(
            f"moe_ep[vllm_xpu] benchmarks {MOE_EP_QUANT_MODE!r} only; case declared moe_dtype={moe_dtype!r}"
        )
    device = torch.device(device)
    get_device_module().set_device(device)
    # Free the prior case's cached device memory: global token counts reach
    # 524288, so fragmentation from the previous case can OOM.
    gc.collect()
    get_device_module().empty_cache()

    if bench is None:
        bench = _make_xpu_fused_moe_bench(
            num_local_experts=num_local_experts,
            moe_ep_size=moe_ep_size,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            model_name=model_name,
            device=device,
        )

    oom_points = []
    for inference_phase, token_counts in (("context", context_token_counts), ("generation", generation_token_counts)):
        # (distributed, alpha) -> "budget" | "OOM". Points are sorted by token
        # count within a series: once one is too large, every larger one is too.
        skipped_series = {}
        for distributed, power_law_alpha, global_num_tokens in _phase_points(token_counts, distributions):
            point = (inference_phase, distributed, power_law_alpha, global_num_tokens)
            label = (
                f"moe_ep {inference_phase} skip {model_name} ep={moe_ep_size} "
                f"{distributed} alpha={power_law_alpha} global_num_tokens={global_num_tokens}"
            )
            series_skip = skipped_series.get((distributed, power_law_alpha))
            if series_skip == "OOM":
                oom_points.append(point)
                continue
            if series_skip == "budget":
                print(f"{label} (larger than a memory-budget skip)")
                continue
            try:
                latency_ms, power_stats = bench(inference_phase, global_num_tokens, distributed, power_law_alpha)
            except MoeEpBenchmarkError:
                raise
            except _PointExceedsMemoryBudget as e:
                # skip: an over-budget launch is an uncatchable DEVICE_LOST (see
                # the module docstring), so this is a deliberate pre-launch skip.
                skipped_series[(distributed, power_law_alpha)] = "budget"
                print(f"{label} ({e})")
                continue
            except torch.OutOfMemoryError:
                skipped_series[(distributed, power_law_alpha)] = "OOM"
                oom_points.append(point)
                gc.collect()
                get_device_module().empty_cache()
                continue
            except Exception as e:
                raise MoeEpBenchmarkError(
                    f"moe_ep {inference_phase} case failed (global_num_tokens={global_num_tokens}, "
                    f"distribution={distributed}, alpha={power_law_alpha}, "
                    f"moe_ep_size={moe_ep_size}, num_experts={num_experts}, model={model_name}): {e}"
                ) from e
            row = _build_moe_ep_row(
                moe_dtype=MOE_EP_QUANT_MODE,
                distribution=f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed,
                inference_phase=inference_phase,
                num_tokens=global_num_tokens,
                hidden_size=hidden_size,
                inter_size=inter_size,
                topk=topk,
                num_experts=num_experts,
                num_slots=num_slots,
                moe_tp_size=1,
                moe_ep_size=moe_ep_size,
                latency_ms=latency_ms,
            )
            if not log_perf(
                item_list=[row],
                framework="VLLM",
                version=vllm_version,
                device_name=get_device_module().get_device_name(),
                op_name=MOE_EP_OP_NAME,
                kernel_source=MOE_EP_KERNEL_SOURCE,
                perf_filename=_moe_expert_compute_perf_path(output_path, perf_filename),
                power_stats=_power_columns(power_stats),
            ):
                raise MoeEpBenchmarkError(
                    f"helper.log_perf failed to persist the measured {inference_phase} row "
                    f"(ep={moe_ep_size}, num_experts={num_experts})"
                )

    if oom_points:
        raise MoeEpBenchmarkError(
            f"moe_ep OOM (model={model_name}, moe_ep_size={moe_ep_size}) despite the memory-budget guard; "
            f"lower aic_moe_ep_mem_fraction (now {_MEM_FRACTION}). {len(oom_points)} point(s) not collected as "
            f"(phase, distribution, alpha, global_num_tokens): {oom_points}"
        )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_ep_test_cases()
    print(f"Total test cases: {len(test_cases)}")
    for test_case in test_cases[:1]:
        print(f"Running test case: {test_case}")
        run_moe_ep_torch(*test_case, perf_filename=PerfFile.MOE_EXPERT_COMPUTE)
