from typing import List

import matplotlib.pyplot as plt

from task import Task


def pretty_plot(tasks: List[Task], fig_title: str, window_length: float):
    non_dropped_tasks = (task for task in tasks if not task.dropped)
    non_overlapping_task_lists = []
    for task in sorted(non_dropped_tasks, key=lambda task: task.t_scheduled):
        for task_list in non_overlapping_task_lists:
            last_task = task_list[-1]
            if task.t_scheduled >= last_task.t_scheduled + last_task.t_dwell:
                task_list.append(task)
                break
        else:
            non_overlapping_task_lists.append([task])

    plt.figure(fig_title, figsize=(6, len(non_overlapping_task_lists)))
    for i, task_list in enumerate(non_overlapping_task_lists):
        x_ranges = list((task.t_scheduled, task.t_dwell) for task in task_list)
        max_priority = max(task.priority for task in task_list)
        x_alpha = list(task.priority / max_priority for task in task_list)
        y_ranges = (i * 0.1, 0.095)
        facecolors = list(task.color for task in task_list)
        plt.broken_barh(
            x_ranges, y_ranges, facecolors=facecolors, edgecolor="black", alpha=x_alpha
        )
    plt.yticks([])
    plt.xlim(0, window_length)
    plt.title(fig_title)
