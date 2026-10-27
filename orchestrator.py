"""A small orchestration graph engine: named nodes over a shared state, static and conditional edges,
cycles, and live events.

The same model as LangGraph's StateGraph, without its dependency tree (LangChain core and dozens of
packages), which would add start-up time and RAM on a 16 GB laptop for no speed gain. A node is a
function of the state; it may be a generator, in which case its events are passed straight through to
the caller (the chat streams them live) and its return value is ignored. After each node the graph
follows the node's edge: a fixed next node, or a chooser function that picks one by name.

    g = Graph("example")
    g.add_node("draft", draft_fn)
    g.add_node("check", check_fn)
    g.add_edge("draft", "check")
    g.add_branch("check", lambda s: "draft" if s.problems else END)
    g.set_entry("draft")
    final_state = yield from g.run(state)
"""
import inspect
from typing import Callable, Iterator, Optional

END = "__end__"


class State(dict):
    """The shared state: a dict whose keys also read and write as attributes (state.model)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


class GraphError(RuntimeError):
    pass


class Graph:
    def __init__(self, name: str, max_steps: int = 200) -> None:
        self.name, self.max_steps = name, max_steps
        self.nodes: dict = {}
        self.labels: dict = {}
        self.edges: dict = {}  # node -> next node name, or (chooser, possible targets)
        self.entry: Optional[str] = None

    # --- building ----------------------------------------------------------------------------------------
    def add_node(self, name: str, fn: Callable, label: str = "") -> "Graph":
        if name in self.nodes or name == END:
            raise GraphError(f"{self.name}: node {name!r} defined twice or reserved")
        self.nodes[name], self.labels[name] = fn, label or name
        return self

    def add_edge(self, src: str, dst: str) -> "Graph":
        self.edges[src] = dst
        return self

    def add_branch(self, src: str, chooser: Callable, targets: tuple) -> "Graph":
        """After `src`, go to chooser(state), which must be one of `targets`. List END among the targets
        when the branch can finish the run, so describe() and mermaid() show that exit."""
        self.edges[src] = (chooser, tuple(targets))
        return self

    def set_entry(self, name: str) -> "Graph":
        self.entry = name
        return self

    def validate(self) -> "Graph":
        """Every node has an edge and every edge points at a node: mistakes fail at import, not mid-chat."""
        if self.entry not in self.nodes:
            raise GraphError(f"{self.name}: entry {self.entry!r} is not a node")
        for name in self.nodes:
            edge = self.edges.get(name)
            if edge is None:
                raise GraphError(f"{self.name}: node {name!r} has no outgoing edge")
            for target in (edge[1] if isinstance(edge, tuple) else (edge,)):
                if target != END and target not in self.nodes:
                    raise GraphError(f"{self.name}: edge {name!r} -> {target!r} points nowhere")
        return self

    # --- running ---------------------------------------------------------------------------------------------
    def run(self, state: State, stop: Optional[Callable] = None) -> Iterator[dict]:
        """Run from the entry node until END (or until stop(state) is true, e.g. the user pressed Stop).
        Yields the nodes' events; returns the final state via `yield from`. The path taken is recorded
        in state.trace."""
        state.setdefault("trace", [])
        current, steps = self.entry, 0
        while current != END:
            steps += 1
            if steps > self.max_steps:
                raise GraphError(f"{self.name}: no END after {self.max_steps} steps (a cycle without an exit?)")
            state.trace.append(current)
            result = self.nodes[current](state)
            if inspect.isgenerator(result):
                yield from result
            if stop is not None and stop(state):
                break
            current = self._next(current, state)
        return state

    def _next(self, current: str, state: State) -> str:
        edge = self.edges[current]
        if not isinstance(edge, tuple):
            return edge
        chooser, targets = edge
        target = chooser(state)
        if target not in targets:
            raise GraphError(f"{self.name}: {current!r} chose {target!r}, not one of {targets}")
        return target

    # --- describing -------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        """Nodes and edges as plain data, for the UI and the docs."""
        edges = []
        for src, edge in self.edges.items():
            if isinstance(edge, tuple):
                edges.extend({"from": src, "to": t, "conditional": True} for t in edge[1])
            else:
                edges.append({"from": src, "to": edge, "conditional": False})
        return {"name": self.name, "entry": self.entry,
                "nodes": [{"id": n, "label": self.labels[n]} for n in self.nodes], "edges": edges}

    def mermaid(self) -> str:
        """The graph as a Mermaid flowchart (rendered by GitHub in the README)."""
        lines = ["flowchart TD"]
        for n in self.nodes:
            lines.append(f'    {n}["{self.labels[n]}"]')
        lines.append(f"    {END}((end))")
        for e in self.describe()["edges"]:
            arrow = "-.->" if e["conditional"] else "-->"
            lines.append(f"    {e['from']} {arrow} {e['to']}")
        return "\n".join(lines)
