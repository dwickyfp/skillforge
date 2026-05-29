"""
Skill Dependency Graph Module.

Provides an in-memory directed graph using adjacency lists for managing
skill dependencies, impact analysis, Q-value propagation, and topological ordering.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SkillNode:
    """Represents a node (skill) in the dependency graph."""

    skill_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillDependencyGraph:
    """
    Directed graph for skill dependencies using in-memory adjacency lists.

    Each skill is a node, and edges represent dependencies (from_id depends on to_id).
    Supports BFS pathfinding, impact analysis, Q-value propagation, and topological sort.

    Attributes:
        _adjacency: Forward adjacency list mapping skill_id -> list of (dependent_id, weight).
        _reverse: Reverse adjacency list mapping skill_id -> list of (dependency_id, weight).
        _nodes: Registry of all skill nodes.
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._adjacency: dict[str, list[tuple[str, float]]] = {}
        self._reverse: dict[str, list[tuple[str, float]]] = {}
        self._nodes: dict[str, SkillNode] = {}

    def add_skill(self, skill_id: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        Add a skill node to the graph.

        Args:
            skill_id: Unique identifier for the skill.
            metadata: Optional metadata dictionary for the skill.

        Raises:
            ValueError: If skill_id is empty or None.
        """
        if not skill_id:
            raise ValueError("skill_id must be a non-empty string")

        if skill_id not in self._nodes:
            self._nodes[skill_id] = SkillNode(skill_id=skill_id, metadata=metadata or {})
            self._adjacency[skill_id] = []
            self._reverse[skill_id] = []
        else:
            # Update metadata if already exists
            if metadata:
                self._nodes[skill_id].metadata.update(metadata)

    def add_dependency(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """
        Add a directed dependency edge: from_id depends on to_id.

        Args:
            from_id: The skill that has the dependency.
            to_id: The skill being depended upon.
            weight: Edge weight (0.0 to 1.0+), representing dependency strength.

        Raises:
            ValueError: If either skill_id is not in the graph.
        """
        if from_id not in self._nodes:
            raise ValueError(f"Source skill '{from_id}' not found in graph")
        if to_id not in self._nodes:
            raise ValueError(f"Target skill '{to_id}' not found in graph")
        if from_id == to_id:
            raise ValueError("Cannot create self-loop dependency")

        # Check for duplicate edge
        existing = [edge for edge in self._adjacency[from_id] if edge[0] == to_id]
        if existing:
            # Update weight of existing edge
            self._adjacency[from_id] = [
                (tid, weight) if tid == to_id else (tid, w)
                for tid, w in self._adjacency[from_id]
            ]
            self._reverse[to_id] = [
                (fid, weight) if fid == from_id else (fid, w)
                for fid, w in self._reverse[to_id]
            ]
        else:
            self._adjacency[from_id].append((to_id, weight))
            self._reverse[to_id].append((from_id, weight))

    def find_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """
        Find the shortest path between two skills using BFS.

        Args:
            from_id: Starting skill.
            to_id: Target skill.

        Returns:
            List of skill IDs representing the path, or None if no path exists.

        Raises:
            ValueError: If either skill_id is not in the graph.
        """
        if from_id not in self._nodes:
            raise ValueError(f"Source skill '{from_id}' not found in graph")
        if to_id not in self._nodes:
            raise ValueError(f"Target skill '{to_id}' not found in graph")

        if from_id == to_id:
            return [from_id]

        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((from_id, [from_id]))
        visited.add(from_id)

        while queue:
            current, path = queue.popleft()

            for neighbor, _ in self._adjacency.get(current, []):
                if neighbor == to_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def downstream_impact(self, skill_id: str) -> list[str]:
        """
        Find all skills that depend on the given skill (downstream dependents).

        Uses BFS traversal through the reverse adjacency list to find all skills
        that would be affected if this skill changes.

        Args:
            skill_id: The skill to analyze.

        Returns:
            List of all downstream dependent skill IDs.

        Raises:
            ValueError: If skill_id is not in the graph.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")

        impacted: list[str] = []
        visited: set[str] = {skill_id}
        queue: deque[str] = deque()

        # Start with direct dependents (reverse edges)
        for dependent, _ in self._reverse.get(skill_id, []):
            if dependent not in visited:
                visited.add(dependent)
                queue.append(dependent)
                impacted.append(dependent)

        while queue:
            current = queue.popleft()
            for dependent, _ in self._reverse.get(current, []):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)
                    impacted.append(dependent)

        return impacted

    def upstream_impact(self, skill_id: str) -> list[str]:
        """
        Find all skills that the given skill depends on (upstream dependencies).

        Uses BFS traversal through the forward adjacency list to find all skills
        that this skill depends upon.

        Args:
            skill_id: The skill to analyze.

        Returns:
            List of all upstream dependency skill IDs.

        Raises:
            ValueError: If skill_id is not in the graph.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")

        impacted: list[str] = []
        visited: set[str] = {skill_id}
        queue: deque[str] = deque()

        # Start with direct dependencies (forward edges)
        for dependency, _ in self._adjacency.get(skill_id, []):
            if dependency not in visited:
                visited.add(dependency)
                queue.append(dependency)
                impacted.append(dependency)

        while queue:
            current = queue.popleft()
            for dependency, _ in self._adjacency.get(current, []):
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append(dependency)
                    impacted.append(dependency)

        return impacted

    def propagate_q_update(
        self, skill_id: str, delta: float, gamma: float = 0.9
    ) -> dict[str, float]:
        """
        Propagate a Q-value update through the dependency graph.

        When a skill's Q-value changes, dependent skills are affected with
        diminishing impact based on distance and the decay factor (gamma).

        Args:
            skill_id: The skill whose Q-value changed.
            delta: The change in Q-value.
            gamma: Decay factor for propagation (0.0 to 1.0). Default 0.9.

        Returns:
            Dictionary mapping affected skill IDs to their propagated delta values.

        Raises:
            ValueError: If skill_id is not in the graph or gamma is out of range.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be between 0.0 and 1.0, got {gamma}")

        propagated: dict[str, float] = {}
        visited: set[str] = {skill_id}
        queue: deque[tuple[str, float]] = deque()

        # Initialize with direct dependents
        for dependent, weight in self._reverse.get(skill_id, []):
            if dependent not in visited:
                propagated_delta = delta * gamma * weight
                visited.add(dependent)
                queue.append((dependent, propagated_delta))
                propagated[dependent] = propagated_delta

        while queue:
            current, current_delta = queue.popleft()
            for dependent, weight in self._reverse.get(current, []):
                if dependent not in visited:
                    propagated_delta = current_delta * gamma * weight
                    visited.add(dependent)
                    queue.append((dependent, propagated_delta))
                    propagated[dependent] = propagated[dependent] = (
                        propagated.get(dependent, 0.0) + propagated_delta
                    )

        return propagated

    def topological_sort(self) -> list[str]:
        """
        Perform topological sort on the dependency graph using Kahn's algorithm.

        Returns skills in an order where dependencies come before dependents.

        Returns:
            List of skill IDs in topological order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        # Calculate in-degrees (number of dependencies for each node)
        in_degree: dict[str, int] = {sid: 0 for sid in self._nodes}
        for sid in self._nodes:
            for dep, _ in self._adjacency.get(sid, []):
                # sid depends on dep, so sid has an incoming conceptual edge from dep
                in_degree[sid] = in_degree.get(sid, 0) + 1

        # Start with nodes that have no dependencies
        queue: deque[str] = deque(
            sid for sid, degree in in_degree.items() if degree == 0
        )
        result: list[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # For each skill that depends on current, reduce their in-degree
            for dependent, _ in self._reverse.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            raise ValueError(
                "Graph contains a cycle; topological sort is not possible"
            )

        return result

    def get_skills(self) -> list[str]:
        """Return all skill IDs in the graph."""
        return list(self._nodes.keys())

    def get_dependencies(self, skill_id: str) -> list[tuple[str, float]]:
        """
        Get direct dependencies of a skill.

        Args:
            skill_id: The skill to query.

        Returns:
            List of (dependency_id, weight) tuples.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")
        return list(self._adjacency.get(skill_id, []))

    def get_dependents(self, skill_id: str) -> list[tuple[str, float]]:
        """
        Get direct dependents of a skill.

        Args:
            skill_id: The skill to query.

        Returns:
            List of (dependent_id, weight) tuples.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")
        return list(self._reverse.get(skill_id, []))

    def remove_skill(self, skill_id: str) -> None:
        """
        Remove a skill and all its edges from the graph.

        Args:
            skill_id: The skill to remove.

        Raises:
            ValueError: If skill_id is not in the graph.
        """
        if skill_id not in self._nodes:
            raise ValueError(f"Skill '{skill_id}' not found in graph")

        # Remove from adjacency lists of other nodes
        for dep, _ in self._adjacency.get(skill_id, []):
            self._reverse[dep] = [
                (fid, w) for fid, w in self._reverse[dep] if fid != skill_id
            ]

        for dep, _ in self._reverse.get(skill_id, []):
            self._adjacency[dep] = [
                (tid, w) for tid, w in self._adjacency[dep] if tid != skill_id
            ]

        # Remove the node itself
        del self._nodes[skill_id]
        del self._adjacency[skill_id]
        del self._reverse[skill_id]

    def save(self, path: str | Path) -> None:
        """
        Serialize the graph to a JSON file.

        Args:
            path: File path to save the graph to.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": {
                sid: {"metadata": node.metadata}
                for sid, node in self._nodes.items()
            },
            "edges": [
                {"from": sid, "to": tid, "weight": weight}
                for sid, edges in self._adjacency.items()
                for tid, weight in edges
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SkillDependencyGraph":
        """
        Deserialize a graph from a JSON file.

        Args:
            path: File path to load the graph from.

        Returns:
            Reconstructed SkillDependencyGraph instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls()

        # Restore nodes
        for sid, node_data in data.get("nodes", {}).items():
            graph.add_skill(sid, metadata=node_data.get("metadata", {}))

        # Restore edges
        for edge in data.get("edges", []):
            from_id = edge["from"]
            to_id = edge["to"]
            weight = edge.get("weight", 1.0)
            graph.add_dependency(from_id, to_id, weight=weight)

        return graph

    def __len__(self) -> int:
        """Return the number of skills in the graph."""
        return len(self._nodes)

    def __contains__(self, skill_id: str) -> bool:
        """Check if a skill exists in the graph."""
        return skill_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"SkillDependencyGraph(skills={len(self._nodes)}, "
            f"edges={sum(len(edges) for edges in self._adjacency.values())})"
        )
