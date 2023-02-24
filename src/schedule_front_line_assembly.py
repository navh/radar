from typing import List

from globals import TIME_WINDOW_END, TIME_WINDOW_START
from task import Task


class Multitask:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
        self.tasks = sorted(tasks, key=lambda task: task.t_start)
        self.t_dwell = sum(task.t_dwell for task in self.tasks)
        self.t_scheduled = self.t_desired()

    def t_desired(self) -> float:
        # arithmetic mean weighted by priority
        sum_of_t_start = 0
        sum_of_weights = 0
        t_offset = 0  # Tasks vote for t_start of supertask
        for task in self.tasks:
            sum_of_t_start += (task.t_start - t_offset) * task.priority
            sum_of_weights += task.priority
            t_offset += task.t_dwell
        ideal_start = sum_of_t_start / sum_of_weights
        last_possible_start = TIME_WINDOW_END - self.t_dwell
        return max(TIME_WINDOW_START, min(ideal_start, last_possible_start))

    def schedule_at(self, t_scheduled):
        self.t_scheduled = t_scheduled

    def schedule_tasks(self) -> List[Task]:
        t_offset = self.t_scheduled
        for task in self.tasks:
            task.schedule_at(t_offset)
            t_offset += task.t_dwell
        return self.tasks


def front_line_assembly(tasks: List[Task]) -> List[Task]:
    # candidate_tasks = tasks  # TODO: think about pass by copy vs pass by ref
    candidates = [task for task in tasks if task.priority > 0]

    # list of dequeues of tasks
    assembled_tasks: List[Multitask] = []

    # dropped_tasks = []
    dropped_tasks = [task for task in tasks if task not in candidates]

    while candidates:
        best_task = select_best_task(candidates)
        candidates.remove(best_task)

        new_multitask = Multitask([best_task])
        assembled_tasks = add_task_to_assembly(assembled_tasks, new_multitask)

        # drop unschedulables
        schedulable_tasks = get_schedulables(candidates, assembled_tasks)
        dropped_tasks += [task for task in candidates if task not in schedulable_tasks]
        candidates = schedulable_tasks

    scheduled_tasks = []
    for multitask in assembled_tasks:
        scheduled_tasks += multitask.schedule_tasks()

    for task in dropped_tasks:
        task.drop_task()

    return scheduled_tasks + dropped_tasks


def select_best_task(tasks: List[Task]) -> Task:
    # epsilon = 0.000_000_1  # All tasks cost 0 when t_scheduled == t_start
    # return min(tasks, key=lambda task: (task.calculate_cost() + epsilon) * task.t_dwell)
    return min(tasks, key=lambda task: task.t_dwell / task.priority)


def get_schedulables(candidate_tasks: List[Task], assembled_tasks: List[Multitask]):
    window_time = TIME_WINDOW_END - TIME_WINDOW_START
    time_to_fill = window_time - sum(multitask.t_dwell for multitask in assembled_tasks)
    return [task for task in candidate_tasks if task.t_dwell < time_to_fill]


def add_task_to_assembly(
    assembly: List[Multitask], new_task: Multitask
) -> List[Multitask]:
    new_head = new_task.t_scheduled
    new_tail = new_task.t_scheduled + new_task.t_dwell

    for task in assembly:
        task_head = task.t_scheduled
        task_tail = task.t_scheduled + task.t_dwell

        # If there is any overlap
        if (new_head <= task_tail) and (new_tail >= task_head):
            assembly.remove(task)
            if new_head < task.t_scheduled:
                combined_task = Multitask(new_task.tasks + task.tasks)
            else:
                combined_task = Multitask(task.tasks + new_task.tasks)
            return add_task_to_assembly(assembly, combined_task)

    return assembly + [new_task]
