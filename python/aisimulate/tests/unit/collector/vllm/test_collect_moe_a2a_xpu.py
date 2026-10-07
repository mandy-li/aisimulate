# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the vLLM-XPU (CRI) DeepEP HT moe_a2a collector."""

from __future__ import annotations

import json

import pytest
from collector import provenance
from collector.vllm import collect_moe_a2a_xpu as a2a
from collector.wideep.sglang.collect_moe_a2a import MoeA2AShape, PhaseTiming
from collector.wideep.vllm.collect_moe_a2a import BenchmarkResult, VllmMoeA2ADeclarationError

pytestmark = pytest.mark.unit

SHAPE = MoeA2AShape(7168, 8, 256)
GRID = {"ht_token_counts": [16, 512, 4096, 8192], "ll_token_counts": [1], "sms": [20]}
BDF = "0000:05:00.0"
OTHER_BDF = "0000:79:00.0"


def _agree(stage: str, failed: bool) -> bool:
    del stage
    return failed


class FakeAdapter:
    def __init__(self, *, fail_verify=(), fail_bench=(), sms=a2a.XPU_HT_NUM_EUS, on_case=None):
        self.fail_verify = set(fail_verify)
        self.fail_bench = set(fail_bench)
        self.sms = sms
        self.on_case = on_case
        self.prepared = None
        self.verified = []
        self.benched = []
        self.closed = False
        self.closed_failed = None

    def prepare(self, cases):
        self.prepared = list(cases)

    def verify(self, case):
        self.verified.append(case.num_tokens)
        if case.num_tokens in self.fail_verify:
            raise a2a.VerificationError(f"synthetic mismatch at {case.num_tokens}")

    def benchmark(self, case):
        self.benched.append(case.num_tokens)
        if self.on_case is not None:
            self.on_case(case)
        if case.num_tokens in self.fail_bench:
            raise RuntimeError(f"synthetic failure at {case.num_tokens}")
        return BenchmarkResult(
            timings={"dispatch": PhaseTiming(11.0, 0.0), "combine": PhaseTiming(7.0, 0.0)},
            sms=self.sms,
            capacity=case.capacity,
        )

    def close(self, *, failed=False):
        self.closed = True
        self.closed_failed = failed


def _plan(world_size=2, max_tokens=4096, shapes=(SHAPE,)):
    cases, skipped = a2a.build_xpu_case_plan(
        shapes=list(shapes), grid=GRID, world_size=world_size, node_num=1, max_tokens=max_tokens
    )
    return cases, skipped


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_plan_supports_single_node_world_sizes(world_size):
    cases, skipped = _plan(world_size=world_size)
    assert skipped == []
    assert [case.num_tokens for case in cases] == [16, 512, 4096]
    assert {case.comm_backend for case in cases} == {"deepep_ht"}
    assert {case.sms for case in cases} == {a2a.XPU_HT_NUM_EUS}


@pytest.mark.parametrize("world_size", [1, 16, 32])
def test_plan_rejects_unsupported_world_sizes(world_size):
    with pytest.raises(VllmMoeA2ADeclarationError, match="world sizes"):
        _plan(world_size=world_size)


def test_plan_rejects_multi_node():
    with pytest.raises(VllmMoeA2ADeclarationError, match="intranode"):
        a2a.build_xpu_case_plan(shapes=[SHAPE], grid=GRID, world_size=8, node_num=2)


def test_ep2_is_dev_only():
    assert a2a.DEV_ONLY_WORLD_SIZES == (2,)
    assert set(a2a.DEV_ONLY_WORLD_SIZES) <= set(a2a.SUPPORTED_WORLD_SIZES)


def test_token_counts_are_capped():
    cases, _ = _plan(max_tokens=512)
    assert [case.num_tokens for case in cases] == [16, 512]
    with pytest.raises(VllmMoeA2ADeclarationError, match="no ht_token_counts"):
        _plan(max_tokens=8)


def test_shapes_over_local_expert_limit_are_skipped():
    big = MoeA2AShape(7168, 8, 384)
    cases, skipped = _plan(world_size=2, shapes=(SHAPE, big))
    assert {case.shape for case in cases} == {SHAPE}
    assert [skip.shape for skip in skipped] == [big]
    assert "192 local experts" in skipped[0].reason
    cases, skipped = _plan(world_size=4, shapes=(SHAPE, big))
    assert {case.shape for case in cases} == {SHAPE, big}
    assert skipped == []


def test_hidden_not_multiple_of_8_is_skipped():
    odd = MoeA2AShape(7170, 8, 256)
    cases, skipped = _plan(shapes=(odd,) + (SHAPE,))
    assert {case.shape for case in cases} == {SHAPE}
    assert [skip.shape for skip in skipped] == [odd]


def test_indivisible_experts_are_a_declaration_error():
    with pytest.raises(VllmMoeA2ADeclarationError, match="not divisible"):
        _plan(world_size=8, shapes=(MoeA2AShape(7168, 8, 100),))


def test_fixed_config_stays_inside_the_inflight_guardrail():
    channels = a2a.XPU_HT_NUM_EUS // 2
    assert channels * a2a.XPU_HT_RECV_TOKENS <= a2a.MAX_INFLIGHT_SLOTS
    assert a2a.MAX_INFLIGHT_SLOTS < 6144  # the footprint that wedged a CRI card
    assert a2a.XPU_HT_SEND_TOKENS < a2a.XPU_HT_RECV_TOKENS


def test_persisted_key_matches_cuda_schema_with_xpu_sms():
    cases, _ = _plan(world_size=4)
    key = cases[0].persisted_key(ep_size=4, node_num=1)
    assert key == ("deepep_ht", "default", 4, 1, 7168, 8, 256, 16, a2a.XPU_HT_NUM_EUS)


def test_case_plan_ids_are_deterministic_and_name_the_transport():
    cases, _ = _plan()
    ids = a2a.case_plan_ids(cases, world_size=2, node_num=1)
    assert ids == a2a.case_plan_ids(cases, world_size=2, node_num=1)
    assert all(case_id.startswith(f"{a2a.MODULE_NAME}:benchmark:") for case_id in ids)
    payload = json.loads(ids[0].split(":benchmark:", 1)[1])
    assert payload["transport"] == a2a.TRANSPORT_PATH
    assert payload["config"]["num_eus"] == a2a.XPU_HT_NUM_EUS


def test_canary_picks_small_middle_large_of_first_shape():
    cases, _ = _plan(shapes=(SHAPE, MoeA2AShape(4096, 8, 128)))
    picks = a2a.select_canary_cases(cases)
    assert len({case.shape for case in picks}) == 1
    assert [case.num_tokens for case in picks] == [16, 512, 4096]


def test_effective_bandwidth_formula():
    # 4096 tokens x 7168 bf16 x p_hit 0.99651 ~= 58.5 MB in 5.0 ms ~= 11.7 GB/s (CRI, 2026-10-06).
    assert a2a.effective_gbps(58_515_326, 5000.0) == pytest.approx(11.70, abs=0.01)
    assert a2a.effective_gbps(1, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Safety: guard, BDF parsing, watchdog
# ---------------------------------------------------------------------------


def test_parse_bdfs():
    assert a2a.parse_bdfs(" 0000:05:00.0,0000:8A:00.0 ") == ["0000:05:00.0", "0000:8a:00.0"]
    assert a2a.parse_bdfs("") == []
    with pytest.raises(VllmMoeA2ADeclarationError, match="invalid --guard-bdfs"):
        a2a.parse_bdfs("card3")


def _kernel_log(*lines):
    log = list(lines)
    return log, (lambda: list(log))


def test_guard_counts_only_guarded_cards_and_error_lines():
    log, reader = _kernel_log(
        f"[1.0] xe {BDF}: [drm] GT0: whitelist REG[0x3949d0]: allow rw access",
        f"[2.0] xe {BDF}: [drm] Tile0: GT0: Fault response: Unsuccessful -ENOENT",
        f"[3.0] xe {OTHER_BDF}: [drm] device wedged, needs recovery",
    )
    guard = a2a.DriverFaultGuard([BDF], reader=reader)
    assert guard.snapshot() == 1
    log.append(f"[4.0] xe {OTHER_BDF}: [drm] exec queue reset detected")
    guard.check(1, context="other tenant")  # someone else's card: no trip
    log.append(f"[5.0] xe {BDF}: [drm] *ERROR* TLB invalidation fence timeout, seqno=2 recv=1")
    with pytest.raises(a2a.DriverFaultError, match="TLB invalidation"):
        guard.check(1, context="case 0")


def test_guard_requires_bdfs():
    with pytest.raises(VllmMoeA2ADeclarationError):
        a2a.DriverFaultGuard([])


def test_watchdog_terminates_then_hard_exits(monkeypatch):
    killed, exited, recorded = [], [], []
    monkeypatch.setattr(a2a.os, "kill", lambda pid, sig: killed.append(sig))
    monkeypatch.setattr(a2a.os, "_exit", lambda code: exited.append(code))
    monkeypatch.setattr(a2a, "WATCHDOG_GRACE_S", 0.01)
    watchdog = a2a.CaseWatchdog(5.0, on_expire=recorded.append)
    watchdog._expire("case 3")
    import time

    deadline = time.monotonic() + 2.0
    while not exited and time.monotonic() < deadline:
        time.sleep(0.01)
    assert killed == [a2a.signal.SIGTERM]
    assert exited == [124]
    assert "case 3" in recorded[0]


def test_watchdog_disarms_after_a_fast_case(monkeypatch):
    monkeypatch.setattr(a2a.os, "kill", lambda pid, sig: pytest.fail("watchdog fired"))
    with a2a.CaseWatchdog(0.5).arm("fast"):
        pass
    with a2a.CaseWatchdog(0).arm("disabled"):
        pass


def test_check_device_bdfs():
    a2a.check_device_bdfs(["0000:05:00.0", "0000:09:00.0"], ["0000:09:00.0", "0000:05:00.0"])
    with pytest.raises(VllmMoeA2ADeclarationError, match="do not match"):
        a2a.check_device_bdfs(["0000:05:00.0", "0000:09:00.0"], ["0000:05:00.0", "0000:8a:00.0"])
    with pytest.raises(VllmMoeA2ADeclarationError, match="share a device"):
        a2a.check_device_bdfs(["0000:05:00.0", "0000:05:00.0"], ["0000:05:00.0"])
    with pytest.raises(VllmMoeA2ADeclarationError, match="resolve"):
        a2a.check_device_bdfs(["0000:05:00.0", ""], ["0000:05:00.0", "0000:09:00.0"])


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------


def test_collect_writes_two_rows_per_case():
    cases, _ = _plan()
    adapter = FakeAdapter()
    result = a2a.collect_with_adapter(cases, adapter=adapter, world_size=2, node_num=1, agreement=_agree)
    assert adapter.prepared == cases and adapter.closed
    assert adapter.closed_failed is False
    assert adapter.verified == adapter.benched == [16, 512, 4096]
    assert len(result.rows) == 2 * len(cases)
    assert {row["phase"] for row in result.rows} == {"dispatch", "combine"}
    assert {row["sms"] for row in result.rows} == {a2a.XPU_HT_NUM_EUS}
    assert {row["ep_size"] for row in result.rows} == {2}
    assert result.failures == [] and result.not_run == []


def test_collect_stops_at_first_verify_failure():
    cases, _ = _plan()
    adapter = FakeAdapter(fail_verify={512})
    recorded, logged = [], []
    result = a2a.collect_with_adapter(
        cases,
        adapter=adapter,
        world_size=2,
        node_num=1,
        agreement=_agree,
        on_failure=recorded.append,
        log=logged.append,
    )
    assert [failure.case.num_tokens for failure in recorded] == [512]  # persisted before teardown
    assert adapter.closed_failed is True
    assert any("start case 1" in line for line in logged)
    assert any("FAILED case 1" in line for line in logged)
    assert adapter.benched == [16]
    assert [failure.case.num_tokens for failure in result.failures] == [512]
    assert result.failures[0].error_type == "VerificationError"
    assert [case.num_tokens for case in result.not_run] == [4096]
    assert len(result.rows) == 2 and adapter.closed


def test_collect_stops_when_a_peer_fails():
    cases, _ = _plan()
    calls = []

    def peer_fails_second_case(stage, failed):
        calls.append(stage)
        return failed or stage == "case:1:benchmark"

    result = a2a.collect_with_adapter(
        cases, adapter=FakeAdapter(), world_size=2, node_num=1, agreement=peer_fails_second_case
    )
    assert [failure.case.num_tokens for failure in result.failures] == [512]
    assert result.failures[0].error_type == "VllmMoeA2APeerError"
    assert [case.num_tokens for case in result.not_run] == [4096]


def test_collect_stops_on_new_driver_fault():
    cases, _ = _plan()
    log, reader = _kernel_log()

    def fault_during_512(case):
        if case.num_tokens == 512:
            log.append(f"[9.0] xe {BDF}: [drm] Tile0: GT0: Engine memory CAT error [18]: class=ccs")

    adapter = FakeAdapter(on_case=fault_during_512)
    result = a2a.collect_with_adapter(
        cases,
        adapter=adapter,
        world_size=2,
        node_num=1,
        agreement=_agree,
        guard=a2a.DriverFaultGuard([BDF], reader=reader),
    )
    assert [failure.case.num_tokens for failure in result.failures] == [512]
    assert result.failures[0].error_type == "DriverFaultError"
    assert len(result.rows) == 2  # only the 16-token case


def test_collect_rejects_unexpected_sms():
    cases, _ = _plan()
    result = a2a.collect_with_adapter(cases, adapter=FakeAdapter(sms=20), world_size=2, node_num=1, agreement=_agree)
    assert result.failures and "sms=20" in result.failures[0].error
    assert result.rows == []


# ---------------------------------------------------------------------------
# Tensor helpers (CPU torch)
# ---------------------------------------------------------------------------


def test_expected_int32_combine_multiplies_by_copies():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[5, 5], [7, 7], [9, 9]], dtype=torch.int32)
    in_rank = torch.tensor([[True, False], [True, True], [False, True]])
    expected = a2a.expected_int32_combine(torch, x, in_rank)
    assert expected.tolist() == [[5, 5], [14, 14], [9, 9]]


def test_build_topk_ids_is_deterministic_and_unique():
    torch = pytest.importorskip("torch")
    (case, *_), _ = _plan()
    first = a2a.build_topk_ids(torch, case, rank=0, device="cpu")
    again = a2a.build_topk_ids(torch, case, rank=0, device="cpu")
    other_rank = a2a.build_topk_ids(torch, case, rank=1, device="cpu")
    assert first.shape == (case.num_tokens, SHAPE.topk)
    assert torch.equal(first, again)
    assert not torch.equal(first, other_rank)
    ordered = first.sort(dim=-1).values
    assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())


# ---------------------------------------------------------------------------
# Registration and plan-only entry point
# ---------------------------------------------------------------------------


def test_module_is_a_registered_standalone_provenance_producer():
    assert a2a.MODULE_NAME in provenance.STANDALONE_COLLECTOR_MODULES
    assert a2a.MODULE_NAME in provenance.enumerate_provenance_modules()


def test_plan_only_needs_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(a2a, "get_vllm_moe_a2a_shapes", lambda **_: [SHAPE, MoeA2AShape(7168, 8, 384)])
    monkeypatch.setattr(a2a, "get_moe_a2a_workload_grid", lambda: GRID)
    monkeypatch.delenv("RANK", raising=False)
    a2a.main(["--gpus-per-node", "2", "--plan-only", "--world-size", "2"])
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{") :])
    assert summary["cases"] == 3
    assert summary["token_counts"] == [16, 512, 4096]
    assert summary["dev_only"] is True
    assert len(summary["skipped_shapes"]) == 1


def test_guard_bdfs_requires_debug_mode(monkeypatch):
    monkeypatch.setattr(a2a, "get_vllm_moe_a2a_shapes", lambda **_: [SHAPE])
    monkeypatch.setattr(a2a, "get_moe_a2a_workload_grid", lambda: GRID)
    monkeypatch.delenv("AIC_A2A_GUARD_BDFS", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(VllmMoeA2ADeclarationError, match="add --debug"):
        a2a.main(["--gpus-per-node", "2", "--guard-bdfs", BDF])
    monkeypatch.setenv("AIC_A2A_GUARD_BDFS", BDF)
    with pytest.raises(VllmMoeA2ADeclarationError, match="add --debug"):
        a2a.main(["--gpus-per-node", "2"])


def test_debug_flag_parses():
    args = a2a.parse_args(["--gpus-per-node", "2", "--debug", "--guard-bdfs", BDF])
    assert args.debug and args.guard_bdfs == BDF
    assert not a2a.parse_args(["--gpus-per-node", "2"]).debug
