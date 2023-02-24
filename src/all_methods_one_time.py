from copy import deepcopy
from math import ceil
from time import time_ns
from typing import List

import distinctipy
import matplotlib.pyplot as plt
import numpy as np

from schedule_dual_side_with_rsst import dual_side_with_rsst
from schedule_earliest_deadline import schedule_earliest_deadline
from schedule_earliest_start_time import earliest_start_time
from schedule_front_line_assembly import front_line_assembly
from schedule_random_shifted_start_time import random_shifted_start_time
from task import Task

# --------------------------------------------------------------------------------
# Create the Task Sequence
# --------------------------------------------------------------------------------

WINDOW_LENGTH = 1
N = 10
LOADING_RATE = 200  # 50% underloaded to 200% overloaded
N_ACTUAL = int(LOADING_RATE * N / 100)  # TODO: should this be ceil? uncast in matlab
TAU = 1
ITR_RSST = int(float("1e3"))  # 1e3 to match matlab, could we invest in a couple zeros?
# ITR_RSST = 1_000_000_000 # I find this easier to parse than 1e9, either works though

TOTAL_BORDERS = 10
BORDER = np.linspace(1 / N, WINDOW_LENGTH - 1 / N, TOTAL_BORDERS)
ITR_RSST_DUALSIDE = 50

# parameters for reinforcement learning scheduling -------------------------------
ITR_RS_RSST = 150  # total iterations for RL, RSST tryouts
ITR_RL_GD = 30  # total tryouts for  gradient descent
REWARD_RL = 10  # initial reward value for reinforcement learning
GOOD_ACTION_REWARD = 1
BAD_ACTION_REWARD = -2
# --------------------------------------------------------------------------------
# parameters for task selection --------------------------------------------------
# task selection has totally K iterations (select K times, K = 4*N)
# ITR_TS=N*4
ITR_TS = ceil(10 * N / N_ACTUAL)

# in each group shuffle S (= N_actual/4) times (randomly select) at first
SHUFFLE_TIMES = ceil(N_ACTUAL / 4)

AWARD = 10  # reward = reward + award
PUNISH = 1  # reward = reward - punish

CROWDEDNESS_OPTION = "Probability"
# CROWDEDNESS_OPTION = "pdf"

ITR_TSRSST = 100  # 200X RSST after the task selection
# --------------------------------------------------------------------------------

# note: I just did a `t_latest = min(1, t_latest)` cap based on comment in matlab.
# in matlab there is `t_latest(t_latest+t_dwell>1)=1-t_dwell(t_latest+t_dwell>1);`
# which I can't can't make sense of.

task_list = list(Task() for _ in range(N_ACTUAL))

for task in task_list:
    task.uniform_randomize(
        n=N,
        minimum_t=0,
        maximum_t=WINDOW_LENGTH,
        tau=TAU,
        minimum_priority=1,  # TODO: minimum priority 0? All priority zero tasks sholud be dropped
        maximum_priority=9,
    )

distinct_colors = distinctipy.get_colors(N_ACTUAL)
for task, color in zip(task_list, distinct_colors):
    task.color = color


most_cost = sum(task.drop_cost() for task in task_list)


# -----------------------------------------------------------------------------------
# helper functions


def pretty_plot(tasks: List[Task], fig_title: str):
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
    plt.xlim(0, WINDOW_LENGTH)
    plt.title(fig_title)


def summary_stats(tasks: List[Task], schedule_name):
    drop_count = sum(1 for task in tasks if task.dropped)
    print(
        f"     Executed tasks count: {N_ACTUAL-drop_count}/{N_ACTUAL} ({schedule_name}) | Dropped tasks count: {drop_count}/{N_ACTUAL} ({schedule_name})"
    )
    normalized_cost = sum(task.calculate_cost() for task in tasks) / most_cost
    print(
        f"     Normalized cost calculated by {schedule_name}: {normalized_cost} (0~1)"
    )


# ----------------------------------------------------------------------------------

print(f"--------------Designed for {N} Tasks-----------------")

# --- show start time of each task

pretty_plot(task_list, "Original Task Sequence, without Scheduling")

# -------------------------------a1. Earliest Start Time (EST) Scheduling ----------

tic = time_ns()

earliest_start_time_tasks_scheduled = earliest_start_time(deepcopy(task_list))

print(f"Time elapsed for EST: {(time_ns()-tic)/1_000_000} ms (EST)")

pretty_plot(earliest_start_time_tasks_scheduled, "Scheduled by EST")
summary_stats(earliest_start_time_tasks_scheduled, "EST Scheduling")

# -------------------------------a2. Earliest Deadline (ED) Scheduling -------------

tic = time_ns()
earliest_deadline_tasks_scheduled = schedule_earliest_deadline(deepcopy(task_list))

print(f"Time elapsed for ED: {(time_ns()-tic)/1_000_000} ms (ED)")

pretty_plot(earliest_deadline_tasks_scheduled, "Earliest Deadline Scheduling")
summary_stats(earliest_deadline_tasks_scheduled, "ED Scheduling")

# -------------------------------b. Random Shifted Start Time (RSST) Scheduling ----

tic = time_ns()
rsst_tasks_scheduled = random_shifted_start_time(deepcopy(task_list), ITR_RSST)

print(f"Time elapsed for RSST: {(time_ns()-tic)/1_000_000} ms (ED)")

pretty_plot(rsst_tasks_scheduled, "RSST Scheduling")
summary_stats(rsst_tasks_scheduled, "RSST Scheduling")

# -------------------------------c. Dual-Side Scheduling with RSST (DSS) -----------

tic = time_ns()
dss_tasks_scheduled = dual_side_with_rsst(
    deepcopy(task_list), ITR_RSST, ITR_RSST_DUALSIDE
)

print(f"Time elapsed for DSS: {(time_ns()-tic)/1_000_000} ms (DSS)")

pretty_plot(dss_tasks_scheduled, "DSS Scheduling")
summary_stats(dss_tasks_scheduled, "DSS Scheduling")

# d. front line assembly

tic = time_ns()
fla_scheduled = front_line_assembly(deepcopy(task_list))


print(f"Time elapsed for FLA: {(time_ns()-tic)/1_000_000} ms (FLA)")

pretty_plot(fla_scheduled, "FLA Scheduling")
summary_stats(fla_scheduled, "FLA Scheduling")


plt.show()
