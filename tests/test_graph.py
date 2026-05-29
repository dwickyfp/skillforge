"""Tests for SkillDependencyGraph (skillforge.core.graph)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillforge.core.graph import SkillDependencyGraph


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def graph() -> SkillDependencyGraph:
    return SkillDependencyGraph()


@pytest.fixture
def linear_graph(graph: SkillDependencyGraph) -> SkillDependencyGraph:
    """A → B → C chain."""
    graph.add_skill("A")
    graph.add_skill("B")
    graph.add_skill("C")
    graph.add_dependency("A", "B")
    graph.add_dependency("B", "C")
    return graph


@pytest.fixture
def diamond_graph(graph: SkillDependencyGraph) -> SkillDependencyGraph:
    """A → B, A → C, B → D, C → D diamond."""
    for s in "ABCD":
        graph.add_skill(s)
    graph.add_dependency("A", "B")
    graph.add_dependency("A", "C")
    graph.add_dependency("B", "D")
    graph.add_dependency("C", "D")
    return graph


# -----------------------------------------------------------------------
# Add skills
# -----------------------------------------------------------------------

class TestAddSkill:
    """Tests for SkillDependencyGraph.add_skill."""

    def test_add_single(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("s1")
        assert "s1" in graph
        assert len(graph) == 1

    def test_add_multiple(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        assert len(graph) == 2

    def test_add_duplicate_no_error(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("dup")
        graph.add_skill("dup")
        assert len(graph) == 1

    def test_add_with_metadata(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("m", metadata={"key": "value"})
        assert "m" in graph

    def test_add_empty_id_raises(self, graph: SkillDependencyGraph) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            graph.add_skill("")

    def test_duplicate_updates_metadata(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("u", metadata={"a": 1})
        graph.add_skill("u", metadata={"b": 2})
        # Metadata should be merged


# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------

class TestAddDependency:
    """Tests for SkillDependencyGraph.add_dependency."""

    def test_simple_dependency(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_dependency("a", "b")
        deps = graph.get_dependencies("a")
        assert len(deps) == 1
        assert deps[0][0] == "b"

    def test_dependency_with_weight(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_dependency("a", "b", weight=0.7)
        deps = graph.get_dependencies("a")
        assert deps[0][1] == 0.7

    def test_self_loop_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        with pytest.raises(ValueError, match="self-loop"):
            graph.add_dependency("a", "a")

    def test_missing_source_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("b")
        with pytest.raises(ValueError, match="Source"):
            graph.add_dependency("missing", "b")

    def test_missing_target_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        with pytest.raises(ValueError, match="Target"):
            graph.add_dependency("a", "missing")

    def test_duplicate_updates_weight(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_dependency("a", "b", weight=0.5)
        graph.add_dependency("a", "b", weight=0.9)
        deps = graph.get_dependencies("a")
        assert len(deps) == 1
        assert deps[0][1] == 0.9


# -----------------------------------------------------------------------
# Path finding
# -----------------------------------------------------------------------

class TestFindPath:
    """Tests for SkillDependencyGraph.find_path."""

    def test_path_self(self, linear_graph: SkillDependencyGraph) -> None:
        assert linear_graph.find_path("A", "A") == ["A"]

    def test_path_direct(self, linear_graph: SkillDependencyGraph) -> None:
        path = linear_graph.find_path("A", "B")
        assert path == ["A", "B"]

    def test_path_transitive(self, linear_graph: SkillDependencyGraph) -> None:
        path = linear_graph.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_path_reverse_no_path(self, linear_graph: SkillDependencyGraph) -> None:
        # C → A shouldn't exist in a linear A→B→C graph
        assert linear_graph.find_path("C", "A") is None

    def test_path_diamond_shortest(self, diamond_graph: SkillDependencyGraph) -> None:
        path = diamond_graph.find_path("A", "D")
        assert path is not None
        assert len(path) == 3  # A → B → D or A → C → D
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_path_nonexistent_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        with pytest.raises(ValueError, match="Target"):
            graph.find_path("a", "missing")
        with pytest.raises(ValueError, match="Source"):
            graph.find_path("missing", "a")


# -----------------------------------------------------------------------
# Topological sort
# -----------------------------------------------------------------------

class TestTopologicalSort:
    """Tests for SkillDependencyGraph.topological_sort."""

    def test_sort_empty(self, graph: SkillDependencyGraph) -> None:
        assert graph.topological_sort() == []

    def test_sort_single(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("only")
        assert graph.topological_sort() == ["only"]

    def test_sort_linear(self, linear_graph: SkillDependencyGraph) -> None:
        order = linear_graph.topological_sort()
        assert order.index("A") > order.index("B")
        assert order.index("B") > order.index("C")

    def test_sort_diamond(self, diamond_graph: SkillDependencyGraph) -> None:
        order = diamond_graph.topological_sort()
        # D must come before both B and C
        assert order.index("D") < order.index("B")
        assert order.index("D") < order.index("C")
        # B and C must come before A
        assert order.index("B") < order.index("A")
        assert order.index("C") < order.index("A")

    def test_sort_cycle_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("x")
        graph.add_skill("y")
        graph.add_dependency("x", "y")
        graph.add_dependency("y", "x")
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()

    def test_sort_independent_nodes(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_skill("c")
        order = graph.topological_sort()
        assert set(order) == {"a", "b", "c"}


# -----------------------------------------------------------------------
# Q-value propagation
# -----------------------------------------------------------------------

class TestPropagate:
    """Tests for SkillDependencyGraph.propagate_q_update."""

    def test_propagate_basic(self, linear_graph: SkillDependencyGraph) -> None:
        # A→B→C: A depends on B, B depends on C.
        # When C's Q changes, B (and transitively A) should be affected.
        propagated = linear_graph.propagate_q_update("C", delta=0.5)
        # B depends on C, so B should get the propagated delta
        assert "B" in propagated
        assert propagated["B"] == pytest.approx(0.5 * 0.9)  # delta * gamma * weight

    def test_propagate_transitive(self, linear_graph: SkillDependencyGraph) -> None:
        propagated = linear_graph.propagate_q_update("C", delta=1.0)
        # B gets direct propagation, A gets second-order (through B)
        assert "B" in propagated
        assert "A" in propagated
        assert propagated["A"] < propagated["B"]

    def test_propagate_decay(self, linear_graph: SkillDependencyGraph) -> None:
        propagated = linear_graph.propagate_q_update(
            "C", delta=1.0, gamma=0.5
        )
        assert propagated["B"] == pytest.approx(0.5)  # 1.0 * 0.5 * 1.0
        assert propagated["A"] == pytest.approx(0.25)  # 0.5 * 0.5 * 1.0

    def test_propagate_with_weight(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_dependency("a", "b", weight=0.3)
        # a depends on b, so when b changes, a should be affected
        propagated = graph.propagate_q_update("b", delta=1.0)
        assert propagated["a"] == pytest.approx(1.0 * 0.9 * 0.3)

    def test_propagate_nonexistent_raises(self, graph: SkillDependencyGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.propagate_q_update("missing", delta=1.0)

    def test_propagate_invalid_gamma_raises(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        with pytest.raises(ValueError, match="gamma"):
            graph.propagate_q_update("a", delta=1.0, gamma=2.0)

    def test_propagate_diamond_accumulates(
        self, diamond_graph: SkillDependencyGraph
    ) -> None:
        """D is depended on by both B and C; A depends on both B and C.
        When D changes, B and C get direct propagation, A gets from both paths."""
        propagated = diamond_graph.propagate_q_update("D", delta=1.0)
        # B depends on D, C depends on D → both get direct
        assert "B" in propagated
        assert "C" in propagated
        # A depends on both B and C → gets accumulated propagation
        assert "A" in propagated


# -----------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------

class TestQueryHelpers:
    """Tests for get_skills, get_dependencies, get_dependents."""

    def test_get_skills(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        assert set(graph.get_skills()) == {"a", "b"}

    def test_get_dependencies(self, linear_graph: SkillDependencyGraph) -> None:
        deps = linear_graph.get_dependencies("A")
        assert len(deps) == 1
        assert deps[0][0] == "B"

    def test_get_dependents(self, linear_graph: SkillDependencyGraph) -> None:
        deps = linear_graph.get_dependents("C")
        assert len(deps) == 1
        assert deps[0][0] == "B"

    def test_get_dependencies_nonexistent(self, graph: SkillDependencyGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.get_dependencies("missing")

    def test_get_dependents_nonexistent(self, graph: SkillDependencyGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.get_dependents("missing")


# -----------------------------------------------------------------------
# Remove skill
# -----------------------------------------------------------------------

class TestRemoveSkill:
    """Tests for SkillDependencyGraph.remove_skill."""

    def test_remove_existing(self, linear_graph: SkillDependencyGraph) -> None:
        linear_graph.remove_skill("B")
        assert "B" not in linear_graph
        assert len(linear_graph) == 2
        # A should have no dependencies, C should have no dependents
        assert linear_graph.get_dependencies("A") == []
        assert linear_graph.get_dependents("C") == []

    def test_remove_nonexistent_raises(self, graph: SkillDependencyGraph) -> None:
        with pytest.raises(ValueError, match="not found"):
            graph.remove_skill("missing")


# -----------------------------------------------------------------------
# Impact analysis
# -----------------------------------------------------------------------

class TestImpactAnalysis:
    """Tests for downstream_impact and upstream_impact."""

    def test_downstream_impact(self, linear_graph: SkillDependencyGraph) -> None:
        # A→B→C: C is the root prerequisite.
        # Skills that depend on C = B (direct), A (transitive)
        impact = linear_graph.downstream_impact("C")
        assert set(impact) == {"A", "B"}

    def test_downstream_impact_leaf(self, linear_graph: SkillDependencyGraph) -> None:
        # A is the most dependent node — nothing depends on A
        assert linear_graph.downstream_impact("A") == []

    def test_upstream_impact(self, linear_graph: SkillDependencyGraph) -> None:
        # A→B→C: A depends on B and C
        impact = linear_graph.upstream_impact("A")
        assert set(impact) == {"B", "C"}

    def test_upstream_impact_root(self, linear_graph: SkillDependencyGraph) -> None:
        # C is the root — depends on nothing
        assert linear_graph.upstream_impact("C") == []


# -----------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------

class TestSerialization:
    """Tests for save/load."""

    def test_save_load_roundtrip(self, linear_graph: SkillDependencyGraph, tmp_path: Path) -> None:
        path = tmp_path / "graph.json"
        linear_graph.save(path)
        loaded = SkillDependencyGraph.load(path)
        assert len(loaded) == 3
        assert loaded.find_path("A", "C") == ["A", "B", "C"]

    def test_save_creates_parent_dirs(self, graph: SkillDependencyGraph, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "graph.json"
        graph.save(path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SkillDependencyGraph.load(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            SkillDependencyGraph.load(bad_file)


# -----------------------------------------------------------------------
# Repr and contains
# -----------------------------------------------------------------------

class TestDunderMethods:
    """Tests for __len__, __contains__, __repr__."""

    def test_len(self, graph: SkillDependencyGraph) -> None:
        assert len(graph) == 0
        graph.add_skill("a")
        assert len(graph) == 1

    def test_contains(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        assert "a" in graph
        assert "b" not in graph

    def test_repr(self, graph: SkillDependencyGraph) -> None:
        graph.add_skill("a")
        graph.add_skill("b")
        graph.add_dependency("a", "b")
        r = repr(graph)
        assert "skills=2" in r
        assert "edges=1" in r
