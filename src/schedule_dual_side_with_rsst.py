import random
from copy import deepcopy
from typing import List

from pack_tasks_head_to_tail import (pack_tasks_head_to_tail,
                                     pack_tasks_tail_to_head)
from task import Task


def dual_side_with_rsst(
    tasks: List[Task], permutations: int, borders: int
) -> List[Task]:
    shuffles = (
        shuffle_and_dual_side_schedule(deepcopy(tasks), borders)
        for _ in range(permutations)
    )
    return min(shuffles, key=lambda tasks: sum(task.calculate_cost() for task in tasks))


def shuffle_and_dual_side_schedule(tasks: List[Task], borders: int) -> List[Task]:
    for task in tasks:
        task.t_scheduled = random.uniform(task.t_earliest, task.t_latest)
    tasks_sorted_by_start = sorted(tasks, key=lambda task: task.t_scheduled)
    return lowest_cost_border(deepcopy(tasks_sorted_by_start), borders)


def lowest_cost_border(tasks: List[Task], borders: int) -> List[Task]:
    border_schedules = (
        dual_side_schedule(tasks, border / borders) for border in range(borders)
    )
    return min(
        border_schedules, key=lambda tasks: sum(task.calculate_cost() for task in tasks)
    )


def dual_side_schedule(tasks: List[Task], border: float) -> List[Task]:
    # If more than half of the task is before the border, go to the left
    # tasks_before_border = [
    #     task for task in tasks if ((task.t_scheduled + task.t_dwell / 2)) < border
    # ]
    # TODO: change this so that tasks on border fall onto correct side of border
    tasks_before_border = [task for task in tasks if ((task.t_scheduled)) < border]

    tasks_after_border = [task for task in tasks if task not in tasks_before_border]

    scheduled_tasks_before_border = pack_tasks_tail_to_head(tasks_before_border, border)

    scheduled_tasks_after_border = pack_tasks_head_to_tail(tasks_after_border, border)

    return scheduled_tasks_before_border + scheduled_tasks_after_border
