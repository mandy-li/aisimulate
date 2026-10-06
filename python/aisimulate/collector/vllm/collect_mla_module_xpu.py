# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-XPU (CRI) full MLA/DSA module collector.

XPU port of ``collect_mla_module.py``. Builds a single ``DeepseekV2MLAAttention``
module from vLLM's own modeling code with dummy weights and times the complete
forward (projections + attention + output), mirroring serving. MLA vs DSA is
selected by ``index_topk`` in the HF config; the DSA path routes the sparse
indexer through the deepklox XE3P kernels (``has_deepklox()`` true).

Test cases are the curated GLM-5.3 DSA ctx+gen subset declared under
``common_case_values.mla_module.xpu*``
"""

__compat__ = "vllm==0.28.0"

import gc
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_xpu_mla_module_test_cases
from collector.helper import (
    benchmark_with_power,
    get_device_module,
    log_perf,
    xpu_graph_measure_enabled,
)
from collector.vllm.utils_xpu import (
    BatchSpec,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_vllm_config,
    setup_distributed,
    with_exit_stack,
)

# vLLM registers GlmMoeDsaForCausalLM but omits the "glm_moe_dsa" config-type
# mapping; the layout matches DeepSeek-V3, so reuse DeepseekV3Config.
if "glm_moe_dsa" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"


_MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local dir with config.json for *model_name* if cached, else the name."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]
    config_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_config.json"
    if not config_file.exists():
        return model_name
    tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
    os.symlink(config_file, os.path.join(tmp_dir, "config.json"))
    # Strip auto_map: vLLM natively supports these architectures and only needs
    # the JSON fields; the auto_map import target is absent from the temp dir.
    with open(config_file) as f:
        config_data = json.load(f)
    if "auto_map" in config_data:
        config_data.pop("auto_map")
        os.remove(os.path.join(tmp_dir, "config.json"))
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_data, f)
    quant_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_hf_quant_config.json"
    if quant_file.exists():
        os.symlink(quant_file, os.path.join(tmp_dir, "hf_quant_config.json"))
    _local_config_cache[model_name] = tmp_dir
    return tmp_dir


def _create_gemm_quant_config(gemm_type: str):
    """vLLM QuantizationConfig for a gemm_type. None for bf16; Fp8Config for fp8_block."""
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        # Block-scaled FP8 requires is_checkpoint_fp8_serialized=True (fp8.py
        # raises otherwise); dynamic activation scales, 128x128 weight blocks.
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    raise ValueError(f"unsupported XPU MLA gemm_type: {gemm_type!r}")


def _create_attention_module(
    model_path: str,
    attn_type: str,
    num_heads: int,
    use_fp8_kv_cache: bool,
    gemm_type: str,
    max_seq_len: int,
    max_batch_size: int,
    device: str,
    is_context: bool,
):
    """Build a DeepseekV2MLAAttention module with dummy weights from vLLM modeling code."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype

    local_model_path = _resolve_model_path(model_path)
    block_size = 64
    max_model_len = max(max_seq_len, 4096)
    num_kv_cache_blocks = max(1 + math.ceil((max_seq_len + 1) / block_size) * max_batch_size, 8192)

    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=max(max_batch_size * max_seq_len, 131072) if is_context else max_batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
        trust_remote_code=True,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    # Control linear-layer GEMM precision: None for bf16, Fp8Config for fp8_block.
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    hf_config = vllm_config.model_config.hf_text_config
    hf_config.num_hidden_layers = 1
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_heads

    topk_indices_buffer = None
    if attn_type == "dsa" and hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(max_tokens, hf_config.index_topk, dtype=torch.int32, device=device)

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        attn_module = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_config,
            hidden_size=hf_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            q_lora_rank=hf_config.q_lora_rank if hasattr(hf_config, "q_lora_rank") else None,
            kv_lora_rank=hf_config.kv_lora_rank,
            max_position_embeddings=hf_config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

    # Serialized block-scaled FP8 creates params on meta device; to() can't copy them.
    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # fill_() dummy weights (latency is value-invariant; overwritten by
    # process_weights_after_loading). Scales -> 0.5 to avoid NaN.
    with torch.no_grad():
        for name, tensor in list(attn_module.named_parameters()) + list(attn_module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.data.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.data.fill_(0.5)
            else:
                tensor.data.fill_(0.01)
    return attn_module, vllm_config


def _process_module_weights(attn_module, vllm_config):
    """Run process_weights_after_loading (FP8 quant + MLAAttention W_UK_T / W_UV)."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)
        for _, module in attn_module.named_modules():
            if isinstance(module, MLAAttention):
                module.process_weights_after_loading(vllm_config.model_config.dtype)


def _create_context_kv_inputs(batch_spec: BatchSpec, kv_lora_rank: int, qk_rope_head_dim: int, device: str):
    """Cached KV tensors for tokens already in the paged cache before this forward."""
    kv_c_contexts, k_pe_contexts = [], []
    for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens, strict=True):
        context_len = max(0, int(seq_len) - int(query_len))
        kv_c_contexts.append(torch.full((context_len, kv_lora_rank), 0.01, dtype=torch.bfloat16, device=device))
        k_pe_contexts.append(torch.full((context_len, 1, qk_rope_head_dim), 0.01, dtype=torch.bfloat16, device=device))
    return kv_c_contexts, k_pe_contexts


def _populate_indexer_kv_cache(indexer_kv_cache, common_attn_metadata, context_lens):
    """Fill the DSA indexer cache so decode sees realistic historical K, not zeros."""
    block_table = common_attn_metadata.block_table_tensor
    block_size = indexer_kv_cache.shape[1]
    entry_dim = indexer_kv_cache.shape[2]
    device = indexer_kv_cache.device
    for i, context_len in enumerate(context_lens):
        if context_len <= 0:
            continue
        token_offsets = torch.arange(context_len, dtype=torch.long, device=device)
        block_indices = token_offsets // block_size
        intra = token_offsets % block_size
        block_ids = block_table[i, block_indices]
        indexer_kv_cache[block_ids, intra, :] = torch.full(
            (context_len, entry_dim), 42, dtype=torch.uint8, device=device
        )


def _create_kv_cache_and_metadata(vllm_config, attn_type, batch_size, seq_len, is_context, prefix_len, device):
    """Build the paged KV cache + attention/indexer metadata for one case."""
    hf_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    block_size = vllm_config.cache_config.block_size
    is_dsa = attn_type == "dsa"
    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(seq_lens=[prefix_len + seq_len] * batch_size, query_lens=[seq_len] * batch_size)
    else:
        batch_spec = BatchSpec(seq_lens=[seq_len] * batch_size, query_lens=[1] * batch_size)

    num_kv_cache_blocks = max(1 + math.ceil((prefix_len + seq_len + 1) / block_size) * batch_size, 8192)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, torch.device(device), arange_block_indices=True
    )

    # Pad the page table's block dim to a multiple of 128/block_size (kernels
    # that require it otherwise raise on single-block short sequences).
    required_divisor = max(1, 128 // block_size)
    current_block_num = common_attn_metadata.block_table_tensor.shape[1]
    if current_block_num % required_divisor != 0:
        padded = ((current_block_num + required_divisor - 1) // required_divisor) * required_divisor
        padding = torch.zeros(
            (common_attn_metadata.block_table_tensor.shape[0], padded - current_block_num),
            dtype=common_attn_metadata.block_table_tensor.dtype,
            device=common_attn_metadata.block_table_tensor.device,
        )
        common_attn_metadata.block_table_tensor = torch.cat([common_attn_metadata.block_table_tensor, padding], dim=1)

    attn_layer_name = "model.layers.0.self_attn.attn"
    attn_layer = vllm_config.compilation_config.static_forward_context[attn_layer_name]
    backend_cls = attn_layer.get_attn_backend()
    kv_cache_spec = attn_layer.get_kv_cache_spec(vllm_config)
    cache_dtype = kv_cache_spec.dtype
    kv_cache_dtype_str = kv_cache_spec.cache_dtype_str

    kv_c_contexts, k_pe_contexts = _create_context_kv_inputs(batch_spec, kv_lora_rank, qk_rope_head_dim, device)
    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=kv_cache_spec.head_size,
        dtype=cache_dtype,
        device=torch.device(device),
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype_str,
        scale=attn_layer._k_scale,
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, [attn_layer_name], vllm_config, torch.device(device))
    attn_metadata = builder.build(common_prefix_len=prefix_len, common_attn_metadata=common_attn_metadata)

    # XPUMLASparseMetadata omits the decode/prefill-split fields the shared
    # forward_impl asserts; the XPU sparse impl is MQA-only, so route all tokens MQA.
    if backend_cls.get_name() == "XPU_MLA_SPARSE" and not hasattr(attn_metadata, "num_decode_tokens"):
        attn_metadata.num_decode_tokens = attn_metadata.num_actual_tokens
        attn_metadata.num_decodes = attn_metadata.num_reqs
        attn_metadata.num_prefills = 0
        attn_metadata.prefill_max_seq_len = attn_metadata.max_seq_len
        attn_metadata.prefill = None

    indexer_kv_cache = None
    indexer_metadata = None
    if is_dsa:
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_layer = vllm_config.compilation_config.static_forward_context[indexer_layer_name]
        indexer_spec = indexer_layer.get_kv_cache_spec(vllm_config)
        indexer_kv_cache = torch.zeros(
            num_kv_cache_blocks, block_size, indexer_spec.head_size, dtype=indexer_spec.dtype, device=device
        )
        indexer_builder_cls = indexer_layer.get_attn_backend().get_builder_cls()
        builder_kwargs = {}
        if getattr(indexer_builder_cls, "requires_block_table_width", False):
            builder_kwargs["block_table_width"] = common_attn_metadata.block_table_tensor.shape[1]
        indexer_builder = indexer_builder_cls(
            indexer_spec, [indexer_layer_name], vllm_config, torch.device(device), **builder_kwargs
        )
        indexer_metadata = indexer_builder.build(common_prefix_len=prefix_len, common_attn_metadata=common_attn_metadata)
        _populate_indexer_kv_cache(indexer_kv_cache, common_attn_metadata, [t.shape[0] for t in kv_c_contexts])

    return kv_cache, attn_metadata, indexer_kv_cache, indexer_metadata


def _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata):
    """Ground-truth backend for the perf row (DSA/decode -> attn_backend)."""
    if attn_type == "dsa" or not is_context or attn_metadata.num_prefills == 0:
        return mla_layer.attn_backend.get_name()
    return mla_layer.prefill_backend.get_name()


@with_exit_stack
def run_mla_module(
    exit_stack,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "xpu:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single vLLM-XPU MLA/DSA module-level benchmark point."""
    if attn_type not in {"mla", "dsa"}:
        raise ValueError(f"unsupported vLLM attention type: {attn_type!r}")
    if kv_cache_dtype not in {"bfloat16", "fp8"}:
        raise ValueError(f"unsupported vLLM MLA KV-cache dtype: {kv_cache_dtype!r}")
    if compute_dtype != "bfloat16":
        raise ValueError(f"XPU MLA query compute is bfloat16-only; got {compute_dtype!r}")

    setup_distributed(device)
    get_device_module().set_device(device)
    # DSA's sparse_attn_indexer requires a WorkspaceManager.
    init_workspace_manager(torch.device(device))

    use_fp8_kv_cache = kv_cache_dtype == "fp8"
    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"

    # ctx-dsa at bs>=8 + isl>=16384 triggers an uncorrectable GPU compute fault
    # (device-lost / wedged). Skipped by default; set SKIP_HW_FAULT_CONFIGS=0 to
    # force-run (will likely wedge the GPU and need a reboot).
    _hw_fault_config = (
        attn_type == "dsa" and is_context and batch_size >= 8 and seq_len >= 16384
    )
    if _hw_fault_config and os.environ.get("SKIP_HW_FAULT_CONFIGS", "1") not in (
        "0",
        "false",
        "False",
        "no",
    ):
        print(
            f"  [SKIP] ctx-dsa b={batch_size}, s={seq_len}: known HW-fault config "
            f"(bs>=8 + isl>=16384). Set SKIP_HW_FAULT_CONFIGS=0 to force-run."
        )
        return None

    # deepklox fused prefill uses a 32-bit token offset that wraps at
    # s_q*h_q*512 >= 2^31; disable the fused path for those long-context cases.
    if attn_type == "dsa" and is_context and seq_len * num_heads * 512 >= 2**31:
        os.environ["DEEPKLOX_SPARSE_PREFILL_DISABLE_FUSED"] = "1"
        print(
            f"  [WARN] ctx-dsa s={seq_len}, heads={num_heads}: fused deepklox "
            f"prefill disabled (32-bit offset would overflow at s*heads*512>=2^31); "
            f"using the slower non-fused path - latency is higher than fused."
        )

    print(
        f"\n[{attn_type.upper()} module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    attn_module, vllm_config = _create_attention_module(
        model_path=model_path,
        attn_type=attn_type,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        gemm_type=gemm_type,
        max_seq_len=prefix_len + seq_len,
        max_batch_size=batch_size,
        device=device,
        is_context=is_context,
    )
    _process_module_weights(attn_module, vllm_config)

    with set_current_vllm_config(vllm_config):
        kv_cache, attn_metadata, indexer_kv_cache, indexer_metadata = _create_kv_cache_and_metadata(
            vllm_config, attn_type, batch_size, seq_len, is_context, prefix_len, device
        )

    attn_layer_name = "model.layers.0.self_attn.attn"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx[attn_layer_name].kv_cache = kv_cache
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    if indexer_kv_cache is not None and indexer_layer_name in forward_ctx:
        forward_ctx[indexer_layer_name].kv_cache = indexer_kv_cache

    hidden_size = vllm_config.model_config.hf_text_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        positions = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
    hidden_states = torch.full((num_tokens, hidden_size), 0.01, dtype=torch.bfloat16, device=device)

    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    attn_metadata_dict = {attn_layer_name: attn_metadata}
    if indexer_metadata is not None:
        attn_metadata_dict[indexer_layer_name] = indexer_metadata
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))

    try:
        with torch.inference_mode():
            attn_module.forward(positions, hidden_states, None)
    except Exception as e:
        print(f"  Dry run failed: {e}")
        traceback.print_exc()
        _cleanup()
        raise

    def kernel_func():
        attn_module.forward(positions, hidden_states, None)

    # Generation -> graph (matches decode serving); context -> eager (DSA
    # prefill is heavy and its captured scratch would retain across tasks).
    use_graph = xpu_graph_measure_enabled() and not is_context

    # Optional overrides to reduce sustained load on heavy shapes (RAS stress).
    warming_up = int(os.environ.get("DSA_WARMUP", warming_up))
    test_ite = int(os.environ.get("DSA_ITERS", test_ite))

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
        use_cuda_graph=use_graph,
    ) as results:
        pass

    latency = results["latency_ms"]

    if is_context:
        isl, step = seq_len, prefix_len
    else:
        isl, step = 1, seq_len

    hf_cfg = vllm_config.model_config.hf_config
    architecture = getattr(hf_cfg, "architectures", [getattr(hf_cfg, "model_type", "unknown")])[0]
    mla_layer = attn_module.mla_attn.mla_attn
    backend_name = _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata)
    actual_kv_cache_dtype = "fp8" if mla_layer.kv_cache_dtype.startswith("fp8") else "bfloat16"

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": actual_kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": 1,
                "step": step,
                "latency": f"{latency:.4f}",
                "used_cuda_graph": results["used_cuda_graph"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=get_device_module().get_device_name(device),
        op_name=f"{attn_type}_{phase}_module",
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, prefix={prefix_len}, "
        f"gemm={gemm_type}, kv={kv_cache_dtype}, backend={backend_name}: {latency:.4f} ms"
    )
    _cleanup()
    return latency


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "xpu:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    # Serving runs forward under inference mode; keep init, dry run, warmup and
    # timing in the SAME mode for consistent workspace/kernel caching.
    with torch.inference_mode():
        return run_mla_module(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            model_path=model_path,
            attn_type=attn_type,
            prefix_len=prefix_len,
            perf_filename=perf_filename,
            device=device,
        )


def _cleanup():
    # Drop vLLM's WorkspaceManager singleton so its scratch is freed; it only
    # grows, so a large task would otherwise pin memory for the worker's life.
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    gc.collect()
    get_device_module().empty_cache()


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection (XPU)."""
    return get_xpu_mla_module_test_cases("context")


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection (XPU)."""
    return get_xpu_mla_module_test_cases("generation")
