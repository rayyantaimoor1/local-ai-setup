"""The orchestration graph engine and the two graphs built on it (chat pipeline, deep-thinking refine loop)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import router_engine  # noqa: E402
from orchestrator import END, Graph, GraphError, State  # noqa: E402


def collect(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_state_reads_and_writes_as_attributes():
    s = State(a=1)
    s.b = 2
    assert s.a == 1 and s["b"] == 2 and dict(s) == {"a": 1, "b": 2}
    with pytest.raises(AttributeError):
        s.missing


def test_nodes_edges_branches_cycles_and_events():
    def count(s):
        s.n += 1
        yield {"type": "tick", "n": s.n}

    g = (Graph("loop").add_node("start", lambda s: None).add_node("count", count)
         .set_entry("start").add_edge("start", "count")
         .add_branch("count", lambda s: "count" if s.n < 3 else END, ("count", END)).validate())
    events, state = collect(g.run(State(n=0)))
    assert [e["n"] for e in events] == [1, 2, 3]  # a generator node's events pass straight through
    assert state.trace == ["start", "count", "count", "count"]  # the path taken, cycle included


def test_stop_ends_the_run_early():
    g = (Graph("g").add_node("a", lambda s: None).add_node("b", lambda s: None)
         .set_entry("a").add_edge("a", "b").add_edge("b", END).validate())
    events, state = collect(g.run(State(), stop=lambda s: True))
    assert state.trace == ["a"]  # the user pressed Stop: nothing after the current node runs


def test_mistakes_fail_when_the_graph_is_built_not_mid_chat():
    with pytest.raises(GraphError):
        Graph("g").add_node("a", lambda s: None).set_entry("a").validate()  # no edge out of a
    with pytest.raises(GraphError):
        Graph("g").add_node("a", lambda s: None).set_entry("a").add_edge("a", "nowhere").validate()
    with pytest.raises(GraphError):
        Graph("g").add_node("a", lambda s: None).add_node("a", lambda s: None)  # defined twice
    g = Graph("g").add_node("a", lambda s: None).set_entry("a").add_branch("a", lambda s: "zzz", ("a",)).validate()
    with pytest.raises(GraphError):
        collect(g.run(State()))  # a chooser may only pick a declared target
    endless = Graph("g", max_steps=5).add_node("a", lambda s: None).set_entry("a").add_edge("a", "a").validate()
    with pytest.raises(GraphError):
        collect(endless.run(State()))  # a cycle without an exit is stopped


def test_the_pipeline_and_refine_graphs_are_complete():
    pipe = router_engine.PIPELINE.describe()
    assert pipe["entry"] == "intake" and [n["id"] for n in pipe["nodes"]] == [
        "intake", "vision", "attachments", "route", "load", "prompt", "answer", "finish"]
    loop = router_engine.REFINE_GRAPH.describe()
    targets = {(e["from"], e["to"]) for e in loop["edges"]}
    assert ("revise", "check") in targets and ("check", END) in targets  # a real cycle with an exit
    assert "flowchart TD" in router_engine.REFINE_GRAPH.mermaid()


def test_the_tool_loop_graph_runs_tools_then_answers(monkeypatch):
    import ollama_client
    replies = iter(["TOOL_CALL: get_system_info()", "You have enough RAM."])
    monkeypatch.setattr(ollama_client, "stream_generate", lambda *a, **k: iter([("response", next(replies))]))
    events, (reply, used, created, generations) = collect(router_engine._answer(
        "m", "sys", "How much free RAM does this computer have?", "How much free RAM does this computer have?",
        True, None, False, False, False))
    assert reply == "You have enough RAM." and used == ["get_system_info"] and generations == 2
    loop = router_engine.TOOL_LOOP.describe()
    assert {(e["from"], e["to"]) for e in loop["edges"]} >= {("generate", "tools"), ("tools", "generate"),
                                                              ("generate", "finalize")}


def test_an_answer_records_the_path_through_the_pipeline(monkeypatch):
    import ollama_client
    monkeypatch.setattr(ollama_client, "generate", lambda *a, **k: "WRITING_TASK")
    monkeypatch.setattr(ollama_client, "stream_generate", lambda *a, **k: iter([("response", "Hi.")]))
    monkeypatch.setattr(router_engine, "_trim", lambda: {"freed_mb": 0})
    done = list(router_engine.process_request("Write a haiku"))[-1]
    assert done["meta"]["trace"] == ["intake", "attachments", "route", "load", "prompt", "answer", "finish"]


def test_api_graph_describes_both_graphs():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app, base_url="http://127.0.0.1:8420") as c:
        data = c.get("/api/graph").json()
    assert data["pipeline"]["name"] == "chat pipeline" and data["refine"]["name"] == "deep thinking refine loop"
