from typing import List

from globals import TIME_WINDOW_END, TIME_WINDOW_START
from task import Task

EPSILON = 0.00000000000000000000000000001


def pack_tasks_head_to_tail(
    sorted_tasks: List[Task], t_head: float = TIME_WINDOW_START
) -> List[Task]:
    for task in sorted_tasks:
        t_proposed = max(t_head, task.t_earliest)

        if t_proposed > task.t_latest or t_proposed + task.t_dwell > TIME_WINDOW_END:
            task.drop_task()
        else:
            task.schedule_at(t_proposed)
            t_head = t_proposed + task.t_dwell

    return sorted_tasks


def pack_tasks_tail_to_head(
    sorted_tasks: List[Task], t_tail: float = TIME_WINDOW_END
) -> List[Task]:
    for task in sorted_tasks:
        # minus epsilon as the t_dwell was causing rounding conflicts
        t_proposed = min(t_tail - task.t_dwell - EPSILON, task.t_latest)

        if t_proposed < TIME_WINDOW_START:
            task.drop_task()
        else:
            task.schedule_at(t_proposed)
            t_tail = t_proposed

    return sorted_tasks
