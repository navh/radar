from typing import List

from task import Task


def summary_stats(tasks: List[Task], name: str, max_cost: float):
    drop_count = sum(1 for task in tasks if task.dropped)
    n = len(tasks)
    print(
        f"     Executed tasks count: {n-drop_count}/{n} ({name}) | Dropped tasks count: {drop_count}/{n} ({name})"
    )
    normalized_cost = sum(task.calculate_cost() for task in tasks) / max_cost
    print(f"     Normalized cost calculated by {name}: {normalized_cost} (0~1)")
