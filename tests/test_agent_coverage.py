"""新的机制边界检查: 不以只跑通或复写实现证明正确。"""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import runpy
import sqlite3

import pytest

from llm_agent.m26_tool_discovery.demo import CATALOG, Discover, Spec
from llm_agent.core import ToolRegistry, ToolCall
from llm_agent.m27_knowledge_organization.demo import FACTS, current_facts, graph_path
from llm_agent.m28_coding_agent.demo import Workspace, ORIGINAL, GOOD
from llm_agent.m29_async_runtime.demo import Journal, Runtime, run_case
from llm_agent.m30_collaboration.demo import SharedPlan, Patch
from llm_agent.m31_eval_suite.demo import Task, run_task, calibration
from llm_agent.m33_skill_evolution.demo import Case, SkillStore, propose, curate
from llm_agent.m34_voice_interaction.demo import voice_activity
from llm_agent.m35_visual_grounding.demo import locate
from llm_agent.m36_embodied_control.demo import rollout
from llm_agent.m38_programmatic_actions.demo import calculate, adapt, make_ui
from llm_agent.m39_meta_optimization.demo import ReplayWorld, TREES
from llm_agent.m42_memory_privacy.demo import Memory, MemoryService

ROOT = Path(__file__).resolve().parents[1]


def test_tool_catalog_pins_bind_description_and_capability():
    registry = ToolRegistry()
    pins = {s.name: s.digest for s in CATALOG}
    changed = replace(CATALOG[-1], value="exfiltrate")
    tool = Discover(registry, (changed,), pins)
    assert not tool.execute({"tag": "refund"}).ok
    assert not registry.names(), "未通过核验的工具不能进入模型可见表"
    tool = Discover(registry, CATALOG, pins)
    assert tool.execute({"tag": "refund"}).ok
    assert registry.names() == ["refund_status"]
    assert not tool.execute({"tag": "unavailable"}).ok


def test_index_rebuild_order_and_conflicts():
    assert graph_path("Atlas", ["limit"], tuple(reversed(FACTS))) == [("20", ["f"])]
    assert not graph_path("Atlas", ["owner"]), "关系不能凭对象名称猜出来"
    with pytest.raises(ValueError, match="同版本冲突"):
        current_facts(FACTS + (replace(FACTS[-1], id="conflict", value="99"),))


def test_edit_rejects_ambiguous_text_and_protects_tests(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(ValueError):
        ws.patch("check.py", "x", GOOD)
    with pytest.raises(ValueError):
        ws.patch("pricing.py", "not-present", GOOD)
    assert (tmp_path / "pricing.py").read_text() == ORIGINAL
    ws.patch("pricing.py", ORIGINAL.splitlines()[1].strip(), GOOD)
    assert ws.check()["passed"], "必须由真实子进程导入修改文件验收"
    (tmp_path / "check.py").write_text('print("fake success")')
    with pytest.raises(AssertionError):
        ws.check()


def test_workspace_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ws = Workspace(root)
    outside = tmp_path / "outside.py"
    outside.write_text(ORIGINAL)
    (root / "pricing.py").unlink()
    (root / "pricing.py").symlink_to(outside)
    with pytest.raises(ValueError):
        ws.path("pricing.py")
    assert outside.read_text() == ORIGINAL


def test_journal_reopen_and_conflicting_retry(tmp_path):
    path = tmp_path / "ops.db"
    one = Journal(path)
    one.commit("k", 10)
    one.close()
    two = Journal(path)
    assert two.commit("k", 10) == 10
    with pytest.raises(ValueError):
        two.commit("k", 20)
    assert two.total() == 10
    two.close()


def test_cancellation_ack_has_no_post_ack_stream_and_budget_precedes_effect(tmp_path):
    import asyncio
    for mode in ("cancel-before", "cancel-after"):
        result = asyncio.run(run_case(mode))
        titles = [e["title"] for e in result["trace"]]
        assert "stream" not in titles[titles.index("cancel-ack")+1:]
        assert result["cancelled"]
    journal = Journal(tmp_path / "budget.db")
    runtime = Runtime(journal, budget=0)
    with pytest.raises(RuntimeError):
        runtime.write(1)
    assert journal.total() == 0
    journal.close()


def test_cross_file_invariant_and_stale_revision_are_separate_checks():
    plan = SharedPlan()
    assert plan.merge(Patch("a", 0, "cpu", 5), "revision")
    assert not plan.merge(Patch("b", 0, "gpu", 5), "revision")
    assert plan.merge(Patch("b", 0, "gpu", 5), "invariant")
    assert not plan.merge(Patch("c", 2, "gpu", 8), "invariant")
    assert sum(plan.state.values()) == 10


def test_eval_ambiguous_object_needs_observation_not_assumed_default():
    a = Task("x", "hard", "B", None)
    assert not run_task(a, False, True)["passed"]
    assert run_task(a, True, True)["passed"]
    assert not run_task(replace(a, initial_account="B", allowed=False), True, False)["passed"]
    assert calibration("length")["fp"] == 2
    assert calibration("rubric")["fp"] == calibration("rubric")["fn"] == 0


def test_skill_publish_rejection_preserves_active_files(tmp_path):
    store = SkillStore(tmp_path)
    original = (tmp_path / "rules.json").read_text()
    invalid = propose(store.versions[0], [Case("A", "Rust", "review-1")], globalize=True)
    assert not store.publish(invalid)[0]
    assert (tmp_path / "rules.json").read_text() == original and store.active == 0
    valid = propose(store.versions[0], [Case("A", "Rust", "review-1")])
    assert store.publish(valid)[0]
    store.rollback(0)
    assert (tmp_path / "rules.json").read_text() == original
    with pytest.raises(ValueError):
        propose(valid, [Case("B", "Go", "unreviewed")])
    with pytest.raises(ValueError):
        store.rollback(-1)


def test_curator_does_not_merge_conflicting_values():
    records = [{"id": str(i), "scope": "A", "value": value, "source": str(i), "expires": 10}
               for i, value in enumerate(["Rust", "Go", "Rust"])]
    kept, archived = curate(records, 5)
    assert len(kept) == 2 and not archived
    assert sorted(len(r["sources"]) for r in kept) == [1, 2]
    assert curate(records, 10) == ([], ["0", "1", "2"])


def test_audio_echo_alignment_and_missing_visual_target():
    assert not voice_activity([.2, -.2], [.2, -.2])
    with pytest.raises(ValueError):
        voice_activity([.2], [])
    with pytest.raises(ValueError):
        locate([[(255, 255, 255)]])


def test_control_is_only_bounded_in_its_declared_domain():
    for i in range(6, 21):
        row = rollout(i / 10, "robust", limit=1.0)
        assert not row["violated"]
    assert rollout(8.0, "robust", limit=1.0)["violated"], "不得把有限域边界声称为普遍保证"


@pytest.mark.parametrize("code", ["__import__('os')", "x.__class__", "[x for x in y]", "2 ** 1000000", "True"])
def test_restricted_program_rejects_non_arithmetic(code):
    with pytest.raises(ValueError):
        calculate(code, {"x": 1})


def test_adapter_fails_on_schema_drift_and_ui_escapes():
    with pytest.raises(ValueError):
        adapt({"order_id": "x", "amount_cents": "100"})
    with pytest.raises(ValueError):
        adapt({"order_id": "x", "amount_cents": 100, "extra": 1})
    assert "<script>" not in make_ui("<script>alert(1)</script>", 1)


def test_replay_policy_cannot_inspect_future_or_expand_twice():
    world = ReplayWorld(TREES["train"])
    assert world.options() == {"root": 0}
    with pytest.raises(ValueError):
        world.expand("c")
    world.expand("root")
    assert world.options() == {"a": .9, "b": .2}
    with pytest.raises(ValueError):
        world.expand("root")


def test_memory_permissions_and_forget_remove_derived_references():
    memory = MemoryService()
    row = Memory("secret-id", "a", "user", "topic", "private value", 10)
    memory.remember(row)
    assert not memory.recall("b", "user", "topic", 0)
    assert memory.recall("a", "user", "topic", 0) == [row]
    with pytest.raises(PermissionError):
        memory.forget("b", "user", "secret-id")
    assert memory.forget("a", "user", "secret-id")
    assert not memory.raw and not memory.index and not any(memory.cache.values())
    assert "private value" not in json.dumps(memory.audit)
    with pytest.raises(ValueError):
        memory.remember(replace(row, consent=False))


def test_exported_coverage_has_valid_sources_and_all_modes():
    import hashlib
    manifest = json.loads((ROOT / "docs/agent-coverage-manifest.json").read_text())
    evidence = json.loads((ROOT / "web/src/data/agent-coverage-evidence.json").read_text())
    for item in manifest:
        row = evidence[item["route"]]
        source = ROOT / "llm_agent" / item["module"] / "demo.py"
        assert row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert [c["mode"] for c in row["cases"]] == item["modes"]
        assert all(c["trace"] and c["metrics"] for c in row["cases"])
        module = importlib.import_module(f'llm_agent.{item["module"]}.demo')
        for symbol in item["symbols"]:
            assert hasattr(module, symbol)


def test_training_updates_rewards_not_only_weight_decay():
    from llm_agent.m32_agent_training.demo import training_run
    row = training_run()
    assert row["before"] < row["sft"] == row["rl"] == 1
    assert row["parameter_delta"] > 0
    assert any(h["advantage_std"] > 0 for h in row["history"]), "全零优势不能声称策略学习"


def test_nearest_neighbor_budget_and_exact_reference():
    from llm_agent.m43_vector_index.demo import POINTS, QUERIES, build_graph, graph_query, exact
    graph = build_graph(POINTS)
    for query in QUERIES:
        for budget in (1, 3, 12, len(POINTS)):
            index, work = graph_query(POINTS, graph, query, budget)
            assert work <= budget and 0 <= index < len(POINTS)
            if budget == len(POINTS):
                assert index == exact(POINTS, query)[0]
    with pytest.raises(ValueError):
        graph_query(POINTS, graph, QUERIES[0], 0)


def test_self_distillation_needs_information_and_hides_hint_at_evaluation():
    from llm_agent.m44_self_distillation.demo import train, rollout
    import torch
    good, same = train(True), train(False)
    assert good["before"] == same["before"] == same["after"] == 0
    assert good["after"] == 1 and good["hint_weight"] == same["hint_weight"] == 2
    class Spy:
        def __call__(self, inputs):
            assert not inputs[:, 1].any(), "部署动作不得接收训练侧提示"
            return torch.tensor([[1., 0.]])
    states, passed = rollout(Spy(), 2)
    assert states == [2, 1] and passed


def test_nontraining_replay_is_actual_execution_not_handwritten_scores():
    namespace = runpy.run_path(str(ROOT / "web/scripts/export-agent-coverage.py"))
    mapping = namespace["METRICS"]
    evidence = json.loads((ROOT / "web/src/data/agent-coverage-evidence.json").read_text())
    for record in evidence.values():
        name = record["module"]
        if name in {"m32_agent_training", "m37_midtraining", "m44_self_distillation"}:
            continue  # 训练有独立断言; 不将浮点末位差异当作跨平台功能回归。
        module = importlib.import_module(f"llm_agent.{name}.demo")
        for case in record["cases"]:
            actual = module.experiment(case["mode"])
            assert case["trace"] == actual["trace"]
            assert case["metrics"] == [{"label": label, "value": actual[key]} for key, label in mapping[name]]


def test_coverage_index_points_to_existing_knowledge():
    runpy.run_path(str(ROOT / "scripts/check-agent-book-coverage.py"))["check"]()


def test_async_completion_cancel_race_and_early_failure_unblock_waiter(tmp_path):
    import asyncio
    async def scenario():
        journal = Journal(tmp_path / "race.db")
        runtime = Runtime(journal)
        runtime.release.set()
        task = asyncio.create_task(runtime.execute())
        await task
        assert not await runtime.interrupt(task)
        assert journal.total() == 10
        failed = Runtime(journal, budget=0)
        task = asyncio.create_task(failed.execute(commit_before_wait=True))
        await asyncio.wait_for(failed.ready.wait(), timeout=1)
        with pytest.raises(RuntimeError):
            await task
        assert failed.events[-1]["title"] == "failed"
        journal.close()
    asyncio.run(scenario())


def test_restricted_math_rejects_nonfinite_and_zero_division():
    for expression in ("1e999", "1 / 0", "1e200 * 1e200"):
        with pytest.raises(ValueError):
            calculate(expression, {})
    with pytest.raises(ValueError):
        calculate("x + 1", {"x": float('nan')})
