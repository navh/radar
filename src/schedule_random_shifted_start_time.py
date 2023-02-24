import random
from copy import deepcopy
from typing import List

from pack_tasks_head_to_tail import pack_tasks_head_to_tail
from task import Task


def random_shifted_start_time(tasks: List[Task], permutations: int) -> List[Task]:
    shuffles = (shuffle_and_schedule(deepcopy(tasks)) for _ in range(permutations))
    return min(shuffles, key=lambda tasks: sum(task.calculate_cost() for task in tasks))


def shuffle_and_schedule(tasks: List[Task]) -> List[Task]:
    # avoiding repeats feels useful here
    for task in tasks:
        task.t_scheduled = random.uniform(task.t_earliest, task.t_latest)
    return pack_tasks_head_to_tail(sorted(tasks, key=lambda task: task.t_scheduled))
