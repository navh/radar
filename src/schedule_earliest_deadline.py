from typing import List

from pack_tasks_head_to_tail import pack_tasks_head_to_tail
from task import Task

# TODO: This 'start + dwell' doesn't really strike me as 'deadline'
# In my head, t_latest is the deadline. For now I am implementing this
# to be faithful to the reference implementations.


def schedule_earliest_deadline(tasks: List[Task]) -> List[Task]:
    return pack_tasks_head_to_tail(
        sorted(tasks, key=lambda task: task.t_start + task.t_dwell)
    )
