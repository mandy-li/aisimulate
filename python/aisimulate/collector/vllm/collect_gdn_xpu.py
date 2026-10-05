# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) Qwen3.5 Gated DeltaNet collector.

XPU exposes GDN through a single fused kernel, ``gdn_attention`` in
``vllm_xpu_kernels._xpu_C``, that applies the packed-QKV depthwise convolution
and the recurrent/chunked scan in one SYCL launch and dispatches prefill vs
decode from the ``num_prefills``/``num_decodes`` counts. This is exactly the op
vLLM-XPU serving runs (``torch.ops.vllm.gdn_attention_core_xpu`` wraps it), so
the collector times that same fused op to record the true production latency --
rather than summing the separately-exposed ``causal_conv1d`` and
``gated_delta_rule`` stages. Input/output projection GEMMs stay with the GEMM
collector.
"""

__compat__ = "vllm==0.28.0"

import gc
import math
import os

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_xpu_gdn_test_cases
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112

# Kernel grid-y = (context) num_tokens/128 * num_v_heads, (decode) batch * num_v_heads.
# Hardware caps grid-y at 65535 (NV collect_gdn.py raises; XPU overflow = uncatchable
# DEVICE_LOST that kills the worker), so pre-skip offending shapes.
_GRID_Y_LIMIT = int(os.getenv("aic_gdn_grid_y_limit", str(65535)))  # noqa: SIM112
_CTX_CHUNK = 128  # chunk length of the context (prefill) scan kernel

# Per COLLECTOR_SYSTEM: CRI needs the larger window, b60 keeps the smaller default.
_LAUNCH_AMORTIZE_BYTES = {"cri": 340_000_000}
_LAUNCH_AMORTIZE_BYTES_DEFAULT = 64_000_000
_PACK_N_MAX = 64


def get_gdn_test_cases():
    """CRI GDN cases (curated model set) as positional per-case lists."""
    return [
        [
            case.phase,
            case.d_model,
            case.d_conv,
            case.num_k_heads,
            case.head_k_dim,
            case.num_v_heads,
            case.head_v_dim,
            case.batch_size_list,
            case.seq_len_list,
            case.model_name,
        ]
        for case in get_xpu_gdn_test_cases()
    ]


def _mixed_sizes(num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    """Packed projection widths the XPU kernel expects (tp-local, tp_size=1)."""
    vpk = num_v_heads // num_k_heads  # v-heads per k-head
    mixed_qkvz = num_k_heads * (2 * head_k_dim + 2 * head_v_dim * vpk)
    mixed_ba = num_k_heads * (2 * vpk)
    mixed_qkv = num_k_heads * (2 * head_k_dim + head_v_dim * vpk)  # conv dim
    return mixed_qkvz, mixed_ba, mixed_qkv


def _grid_y(phase, batch_size, num_tokens, num_v_heads):
    """Kernel grid-y launch extent for this shape (see _GRID_Y_LIMIT note)."""
    if phase == "context":
        # chunk scan: one grid-y row per (chunk, v-head)
        return (num_tokens // _CTX_CHUNK) * num_v_heads
    # packed recurrent decode: one grid-y row per (request, v-head)
    return batch_size * num_v_heads


def _decode_pack_n(common_log_data):
    """Kernels to pack per decode graph: fp32 ssm-state bytes vs amortize threshold, pow2 1..64."""
    system = os.environ.get("COLLECTOR_SYSTEM") or None
    amortize = int(
        os.getenv(
            "aic_gdn_launch_amortize_bytes",
            str(_LAUNCH_AMORTIZE_BYTES.get(system, _LAUNCH_AMORTIZE_BYTES_DEFAULT)),
        )
    )
    state_bytes = (
        common_log_data["batch_size"]
        * common_log_data["num_v_heads"]
        * common_log_data["head_v_dim"]
        * common_log_data["head_k_dim"]
        * 4  # ssm_state is fp32
    )
    raw = min(float(_PACK_N_MAX), max(1.0, amortize / max(state_bytes, 1)))
    return min(_PACK_N_MAX, max(1, 1 << round(math.log2(raw))))


def _benchmark(kernel_func, device, common_log_data, kernel_source, perf_filename, *, phase):
    """Time one op and log a row. Context: eager, repeat_n=10 (NV parity). Generation: graph, pack_n per replay."""
    use_graph = xpu_graph_measure_enabled() and phase != "context"

    if use_graph:
        pack_n = _decode_pack_n(common_log_data)
        repeat_n = 1

        def run():
            for _ in range(pack_n):
                kernel_func()
    else:
        # eager context: NV-style back-to-back amortization, no graph packing
        pack_n = 1
        repeat_n = 10
        run = kernel_func

    kernel_func()
    get_device_module().synchronize()
    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=3,
        num_runs=10,
        repeat_n=repeat_n,
        use_cuda_graph=use_graph,
        allow_graph_fail=False,  # when graphing (generation), capture failure fails the case
    ) as results:
        log_perf(
            item_list=[
                {
                    **common_log_data,
                    "latency": results["latency_ms"] / pack_n,
                    "used_cuda_graph": results["used_cuda_graph"],
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=get_device_module().get_device_name(),
            op_name="gdn",
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )


def _run_phase(
    *,
    phase,
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    device,
):
    """Shared driver: build inputs then time conv + scan for each shape."""
    device = torch.device(device)
    get_device_module().set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    mixed_qkvz, mixed_ba, mixed_qkv = _mixed_sizes(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    # tp-local head counts already; pass tp_size=1 so the kernel keeps them.
    tp_size = 1
    conv_weights = torch.randn(mixed_qkv, d_conv, dtype=dtype, device=device)
    conv_bias = None  # Qwen3.5 depthwise conv is bias=False
    a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(num_v_heads, dtype=dtype, device=device)

    # context: seq_len sub-sweep; generation: single token per request.
    seq_iter = seq_len_list if phase == "context" else [1]

    if aic_debug:
        print(
            f"GDN {phase}: d_model={d_model}, mixed_qkv={mixed_qkv}, "
            f"num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}"
        )

    for batch_size in batch_size_list:
        for seq_len in seq_iter:
            num_tokens = batch_size * seq_len
            grid_y = _grid_y(phase, batch_size, num_tokens, num_v_heads)
            if grid_y > _GRID_Y_LIMIT:
                # skip: kernel grid-y exceeds the hardware limit -> uncatchable
                # DEVICE_LOST (see _GRID_Y_LIMIT note). Not a memory issue.
                print(
                    f"GDN {phase} skip {model_name} bs={batch_size} seq={seq_len} "
                    f"(grid_y={grid_y} > {_GRID_Y_LIMIT})"
                )
                continue
            if phase == "context":
                num_prefills, num_decodes = batch_size, 0
                # fresh prompts: no prior conv/ssm state
                conv_state = torch.zeros(batch_size + 1, d_conv - 1, mixed_qkv, dtype=dtype, device=device)
                ssm_state = torch.zeros(
                    batch_size + 1, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
                )
                has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                num_prefills, num_decodes = 0, batch_size
                # decode continues from cached state
                conv_state = torch.randn(batch_size + 1, d_conv - 1, mixed_qkv, dtype=dtype, device=device)
                ssm_state = torch.randn(
                    batch_size + 1, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
                )
                has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=device)

            # equal token split per request; slot 0 is vLLM's null block
            query_start_loc = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=device)
            state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)

            projected_states_qkvz = torch.randn(num_tokens, mixed_qkvz, dtype=dtype, device=device)
            projected_states_ba = torch.randn(num_tokens, mixed_ba, dtype=dtype, device=device)
            core_attn_out = torch.zeros(num_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)
            z = torch.empty_like(core_attn_out)

            common_log_data = {
                "phase": phase,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_tokens": num_tokens,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }

            # Mirror serving: one fused gdn_attention launch (conv + scan in a
            # single SYCL kernel), exactly as vLLM-XPU calls it at
            # vllm/_xpu_ops.py (torch.ops.vllm.gdn_attention_core_xpu wraps this
            # op). Timing the fused op gives the true production latency rather
            # than a sum of two separately-launched stages.
            def run_gdn():
                torch.ops._xpu_C.gdn_attention(
                    core_attn_out,
                    z,
                    projected_states_qkvz,
                    projected_states_ba,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    conv_weights=conv_weights,
                    conv_bias=conv_bias,
                    activation="silu",
                    A_log=a_log,
                    dt_bias=dt_bias,
                    num_prefills=num_prefills,
                    num_decodes=num_decodes,
                    num_spec_decodes=0,
                    has_initial_state=has_initial_state,
                    non_spec_query_start_loc=query_start_loc,
                    non_spec_token_indx=None,
                    non_spec_state_indices_tensor=state_indices,
                    spec_query_start_loc=None,
                    spec_token_indx=None,
                    spec_state_indices_tensor=None,
                    num_accepted_tokens=None,
                    num_actual_tokens=num_tokens,
                    tp_size=tp_size,
                    reorder_input=False,
                )

            _benchmark(run_gdn, device, common_log_data, "gdn_attention", perf_filename, phase=phase)

            del projected_states_qkvz, projected_states_ba
            del core_attn_out, z, conv_state, ssm_state
            gc.collect()
            get_device_module().empty_cache()


def run_gdn_torch(
    phase: str,
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int] | None,
    model_name: str,
    *,
    perf_filename: str,
    device: str = "xpu:0",
):
    """Route one collector-v2 GDN case to the shared XPU driver."""
    if phase == "context":
        if seq_len_list is None:
            raise ValueError("context GDN cases require seq_len_list")
    elif phase != "generation":
        raise ValueError(f"Unknown phase: {phase}")

    _run_phase(
        phase=phase,
        d_model=d_model,
        d_conv=d_conv,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        batch_size_list=batch_size_list,
        seq_len_list=seq_len_list,
        model_name=model_name,
        perf_filename=perf_filename,
        device=device,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    print(f"GDN XPU Collector - vLLM {vllm_version}")
    print(f"Device: {get_device_module().get_device_name()}")

    test_cases = get_gdn_test_cases()
    print(f"Total test cases: {len(test_cases)}")
    for i, test_case in enumerate(test_cases):
        (
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            batch_size_list,
            seq_len_list,
            model_name,
        ) = test_case
        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        run_gdn_torch(
            phase=phase,
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=PerfFile.GDN,
        )
