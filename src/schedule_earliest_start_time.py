from typing import List

from pack_tasks_head_to_tail import pack_tasks_head_to_tail
from task import Task


def earliest_start_time(tasks: List[Task]) -> List[Task]:
    return pack_tasks_head_to_tail(sorted(tasks, key=lambda task: task.t_start))
