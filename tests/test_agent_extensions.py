"""跨模块边界与网页数值对照。demo 自己的断言仍由 test_smoke 自动发现。"""
from dataclasses import replace
import json
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest

from llm_agent.m20_memory_lifecycle.demo import Evidence, VersionedMemory, comparison
from llm_agent.m21_completion_verification.demo import experiment
from llm_agent.m22_runtime_reliability.demo import retry_write, schedule
from llm_agent.m23_context_cache.demo import simulate
from llm_agent.m24_agentic_rag.demo import QUESTIONS, DOCS, answer, supported
from llm_agent.m25_trajectory_learning.demo import Task, candidates, execute
from llm_agent.full_learning_loop.demo import TASKS, evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_memory_rejects_stale_and_cross_scope_updates():
    memory = VersionedMemory()
    for e in [Evidence("global", "alice", "*", "Python", 1), Evidence("local", "alice", "A", "Rust", 3)]:
        memory.register(e)
        memory.publish(memory.propose(e.id))
    memory.register(Evidence("stale", "alice", "A", "Go", 2))
    with pytest.raises(ValueError):
        memory.publish(memory.propose("stale"))
    with pytest.raises(ValueError):
        memory.register(Evidence("local", "alice", "*", "Rust", 3))
    with pytest.raises(ValueError):
        memory.rollback("bob", "A", 2)
    assert memory.recall("alice", "B").value == "Python"
    assert memory.recall("bob", "A") is None
    memory.index.clear()
    memory.rebuild()
    assert memory.recall("alice", "A").value == "Rust"


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stubborn,broken", [(False, False), (True, False), (False, True)])
def test_completion_budget_and_failure_are_not_success(budget, stubborn, broken):
    row = experiment(max_calls=budget, max_repairs=3, stubborn=stubborn, broken=broken)
    assert row["calls"] <= budget
    assert row["passed"] == row["actual"]
    assert row["notes"]["existing"] == "keep"
    if stubborn or broken:
        assert not row["passed"]


def test_rag_needs_link_and_current_source_even_when_order_changes():
    docs = tuple(reversed(DOCS))
    result = answer(QUESTIONS[1], docs=docs)
    assert supported(QUESTIONS[1], result, docs)
    assert not supported(QUESTIONS[1], {**result, "sources": ["atlas-v2"]}, docs)
    assert not supported(QUESTIONS[1], {**result, "sources": ["alias-comet", "atlas-v1"]}, docs)
    assert answer(QUESTIONS[1], budget=1, docs=docs)["answer"] is None


def test_heldout_feedback_is_rejected_and_future_does_not_leak():
    with pytest.raises(ValueError):
        evaluate("versioned", (replace(TASKS[0], phase="transfer"),))
    full = evaluate("versioned")
    for stop in range(1, len(TASKS)):
        assert evaluate("versioned", TASKS[:stop])["rows"] == full["rows"][:stop]


def test_exported_diagnostic_evidence_is_current():
    generate = runpy.run_path(str(ROOT / "web/scripts/export-agent-evidence.py"))["records"]
    assert json.loads((ROOT / "web/src/data/agent-diagnostics.json").read_text()) == generate()


def test_browser_models_match_python_for_all_controls():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for Python / browser numerical parity")
    script = """
import * as lab from './web/src/utils/agent-reliability.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const cases = JSON.parse(input);
console.log(JSON.stringify(cases.map(([fn, args]) => lab[fn](...args))));
"""
    calls, expected = [], []
    def add(fn, args, result):
        calls.append([fn, args])
        expected.append(result)
    for mode, values in comparison().items():
        add("memoryAnswers", [mode], values)
    for mode in ("direct", "text", "verified"):
        for repairs in range(3):
            for stubborn in (False, True):
                for budget in range(1, 6):
                    row = experiment(mode, repairs, stubborn, max_calls=budget)
                    add("completion", [mode, repairs, stubborn, budget], {k: row[k] for k in ("passed", "actual", "calls", "notes")})
    for mode in ("blind", "idempotent", "query"):
        for attempts in range(1, 4):
            for allowed in (False, True):
                add("retryWrite", [mode, allowed, attempts], retry_write(mode, attempts, allowed))
    for mode in ("all", "first-valid"):
        for budget in range(13):
            add("schedule", [mode, budget], schedule(mode, budget))
    for turns in range(2, 11):
        for placement in ("front", "tail"):
            for threshold in (0, 2, 4, 6):
                for keep in (False, True):
                    add("contextCache", [turns, placement, threshold, keep], simulate(turns, placement, threshold, keep))
    for index, q in enumerate(QUESTIONS):
        for mode in ("plain", "contextual", "agentic"):
            for budget in range(4):
                add("evidenceChain", [index, mode, budget], answer(q, mode, budget))
    for budget in range(1, 6):
        task = Task("test", "customer", 10, budget)
        for trajectory in candidates(task):
            add("trajectoryReport", [trajectory.actions, budget], execute(task, trajectory))
    for mode in ("static", "append", "versioned"):
        add("learning", [mode], evaluate(mode))
    proc = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(calls),
                          text=True, capture_output=True, cwd=ROOT, timeout=30, check=True)
    actual = json.loads(proc.stdout)
    assert len(actual) == len(expected)
    for call, js, python in zip(calls, actual, expected):
        if isinstance(python, dict):
            js = {key: js[key] for key in python}
        assert js == python, f"Python / browser disagreement: {call}"
