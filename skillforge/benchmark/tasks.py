"""Benchmark task definitions and suite generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskCategory(Enum):
    CODING = "coding"
    RESEARCH = "research"
    FILE_OPS = "file_ops"
    SKILL_INTENSIVE = "skill_intensive"


@dataclass
class Task:
    """A single benchmark task."""

    id: str
    name: str
    description: str
    expected: Any
    difficulty: TaskDifficulty
    category: TaskCategory
    verifier_fn: Callable[[Any, Any], float]  # (expected, actual) -> 0-1
    optimal_steps: int = 1

    def verify(self, actual: Any) -> float:
        return self.verifier_fn(self.expected, actual)


def _default_verifier(expected: Any, actual: Any) -> float:
    return 1.0 if expected == actual else 0.0


def _partial_match_verifier(expected: str, actual: str) -> float:
    if not actual:
        return 0.0
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 0.0
    overlap = expected_words & actual_words
    return len(overlap) / len(expected_words)


def _contains_verifier(expected: str, actual: str) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


_TASK_TEMPLATES: list[dict[str, Any]] = [
    # CODING - EASY
    {
        "category": TaskCategory.CODING,
        "difficulty": TaskDifficulty.EASY,
        "templates": [
            ("Write a Python function to reverse a string", "reverse", 2),
            ("Write a function to check if a number is prime", "prime", 3),
            ("Write a function to find the max in a list", "max_value", 2),
            ("Write a function to count vowels in a string", "count_vowels", 2),
            ("Write a function to convert Celsius to Fahrenheit", "celsius_to_f", 2),
        ],
    },
    # CODING - MEDIUM
    {
        "category": TaskCategory.CODING,
        "difficulty": TaskDifficulty.MEDIUM,
        "templates": [
            ("Implement a linked list with append and reverse", "linked_list", 5),
            ("Write a function to flatten nested lists", "flatten", 3),
            ("Implement binary search on a sorted array", "binary_search", 4),
            ("Write a function to detect cycles in a graph", "cycle_detect", 5),
            ("Implement a simple LRU cache", "lru_cache", 6),
        ],
    },
    # CODING - HARD
    {
        "category": TaskCategory.CODING,
        "difficulty": TaskDifficulty.HARD,
        "templates": [
            ("Implement a concurrent task scheduler", "scheduler", 10),
            ("Write a mini regex engine", "regex_engine", 12),
            ("Implement a B-tree with insert and search", "btree", 10),
            ("Build a simple HTTP parser", "http_parser", 8),
            ("Implement A* pathfinding", "astar", 8),
        ],
    },
    # RESEARCH - EASY
    {
        "category": TaskCategory.RESEARCH,
        "difficulty": TaskDifficulty.EASY,
        "templates": [
            ("Summarize the main features of Python 3.12", "python312", 2),
            ("What are the SOLID principles?", "solid", 2),
            ("Explain REST vs GraphQL", "rest_graphql", 2),
            ("List 5 common sorting algorithms", "sorting_algos", 2),
            ("What is Docker and why use it?", "docker", 2),
        ],
    },
    # RESEARCH - MEDIUM
    {
        "category": TaskCategory.RESEARCH,
        "difficulty": TaskDifficulty.MEDIUM,
        "templates": [
            ("Compare React, Vue, and Angular frameworks", "frontend_fw", 4),
            ("Explain CAP theorem with examples", "cap_theorem", 3),
            ("How does garbage collection work in Go?", "go_gc", 4),
            ("Compare SQL vs NoSQL databases", "sql_nosql", 3),
            ("Explain OAuth2 flow types", "oauth2", 4),
        ],
    },
    # RESEARCH - HARD
    {
        "category": TaskCategory.RESEARCH,
        "difficulty": TaskDifficulty.HARD,
        "templates": [
            ("Analyze tradeoffs of microservices vs monolith", "microservices", 6),
            ("Compare consensus algorithms: Raft vs Paxos", "consensus", 8),
            ("Deep dive into Linux kernel scheduling", "kernel_sched", 8),
            ("Explain distributed tracing architecture", "dist_tracing", 6),
            ("Analyze WebAssembly runtime internals", "wasm_runtime", 7),
        ],
    },
    # FILE_OPS - EASY
    {
        "category": TaskCategory.FILE_OPS,
        "difficulty": TaskDifficulty.EASY,
        "templates": [
            ("Read a CSV file and count rows", "count_csv", 2),
            ("Create a directory structure with 3 folders", "mkdir_tree", 2),
            ("Find all .py files in a directory", "find_py", 2),
            ("Replace all occurrences of a word in a file", "replace_word", 2),
            ("Concatenate multiple text files into one", "concat_files", 3),
        ],
    },
    # FILE_OPS - MEDIUM
    {
        "category": TaskCategory.FILE_OPS,
        "difficulty": TaskDifficulty.MEDIUM,
        "templates": [
            ("Parse a JSON config and validate schema", "json_validate", 4),
            ("Watch a directory for changes and log them", "file_watch", 5),
            ("Generate a file tree report", "tree_report", 4),
            ("Merge multiple CSV files with dedup", "merge_csv", 5),
            ("Build a simple file search index", "file_index", 6),
        ],
    },
    # FILE_OPS - HARD
    {
        "category": TaskCategory.FILE_OPS,
        "difficulty": TaskDifficulty.HARD,
        "templates": [
            ("Implement a virtual filesystem in memory", "vfs", 10),
            ("Build a log rotation system", "log_rotate", 8),
            ("Create a file sync tool between directories", "file_sync", 10),
            ("Implement a simple key-value store backed by files", "kv_store", 8),
            ("Build a git-like diff tool", "diff_tool", 10),
        ],
    },
    # SKILL_INTENSIVE - EASY
    {
        "category": TaskCategory.SKILL_INTENSIVE,
        "difficulty": TaskDifficulty.EASY,
        "templates": [
            ("Use a skill to format code with black", "format_black", 2),
            ("Use a skill to run tests with pytest", "run_pytest", 2),
            ("Use a skill to lint code with ruff", "lint_ruff", 2),
            ("Use a skill to generate type stubs", "type_stubs", 3),
            ("Use a skill to create a README", "gen_readme", 2),
        ],
    },
    # SKILL_INTENSIVE - MEDIUM
    {
        "category": TaskCategory.SKILL_INTENSIVE,
        "difficulty": TaskDifficulty.MEDIUM,
        "templates": [
            ("Use skills to set up a full dev environment", "dev_setup", 6),
            ("Use skills to scaffold a REST API project", "scaffold_api", 5),
            ("Use skills to add CI/CD pipeline config", "cicd_setup", 5),
            ("Use skills to refactor codebase to use types", "add_types", 6),
            ("Use skills to generate and run benchmarks", "gen_benchmarks", 5),
        ],
    },
    # SKILL_INTENSIVE - HARD
    {
        "category": TaskCategory.SKILL_INTENSIVE,
        "difficulty": TaskDifficulty.HARD,
        "templates": [
            ("Use skills to build a full-stack app with tests", "fullstack_app", 15),
            ("Use skills to set up monitoring and observability", "monitoring", 10),
            ("Use skills to migrate database schema", "db_migration", 12),
            ("Use skills to implement auth system end-to-end", "auth_system", 14),
            ("Use skills to containerize and deploy an app", "containerize", 10),
        ],
    },
]


def _build_tasks_from_templates() -> list[Task]:
    tasks: list[Task] = []
    for group in _TASK_TEMPLATES:
        cat = group["category"]
        diff = group["difficulty"]
        for desc, key, steps in group["templates"]:
            tid = f"{cat.value}_{diff.value}_{key}"
            ver = _default_verifier if cat == TaskCategory.CODING else _partial_match_verifier
            tasks.append(
                Task(
                    id=tid,
                    name=key.replace("_", " ").title(),
                    description=desc,
                    expected=key,
                    difficulty=diff,
                    category=cat,
                    verifier_fn=ver,
                    optimal_steps=steps,
                )
            )
    return tasks


class TaskSuite:
    """A collection of benchmark tasks."""

    def __init__(self, tasks: list[Task] | None = None) -> None:
        self.tasks: list[Task] = tasks or []

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> TaskSuite:
        """Load tasks from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML task suites")
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        tasks: list[Task] = []
        for item in data.get("tasks", []):
            tasks.append(
                Task(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    expected=item.get("expected", ""),
                    difficulty=TaskDifficulty(item["difficulty"]),
                    category=TaskCategory(item["category"]),
                    verifier_fn=_default_verifier,
                    optimal_steps=item.get("optimal_steps", 1),
                )
            )
        return cls(tasks)

    @classmethod
    def generate_standard_suite(cls, count: int = 50) -> TaskSuite:
        """Generate a standard benchmark suite with tasks across all categories and difficulties."""
        all_tasks = _build_tasks_from_templates()
        if count >= len(all_tasks):
            return cls(all_tasks)
        # Proportional sampling across categories/difficulties
        import random

        random.seed(42)
        selected = random.sample(all_tasks, count)
        selected.sort(key=lambda t: (t.category.value, t.difficulty.value))
        return cls(selected)

    def filter_by_category(self, category: TaskCategory) -> TaskSuite:
        return TaskSuite([t for t in self.tasks if t.category == category])

    def filter_by_difficulty(self, difficulty: TaskDifficulty) -> TaskSuite:
        return TaskSuite([t for t in self.tasks if t.difficulty == difficulty])

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
