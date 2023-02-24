from copy import deepcopy
from math import ceil
from time import time_ns

import distinctipy
import matplotlib.pyplot as plt
import numpy as np

from pretty_plot import pretty_plot
from schedule_dual_side_with_rsst import dual_side_with_rsst
from schedule_earliest_deadline import schedule_earliest_deadline
from schedule_earliest_start_time import earliest_start_time
from schedule_front_line_assembly import front_line_assembly
from schedule_random_shifted_start_time import random_shifted_start_time
from summary_stats import summary_stats
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
        minimum_priority=1,
        maximum_priority=9,
    )

distinct_colors = distinctipy.get_colors(N_ACTUAL)
for task, color in zip(task_list, distinct_colors):
    task.color = color


most_cost = sum(task.drop_cost() for task in task_list)

# ----------------------------------------------------------------------------------

print(f"--------------Designed for {N} Tasks-----------------")

# --- show start time of each task

pretty_plot(task_list, "Original Task Sequence, without Scheduling", WINDOW_LENGTH)

# -------------------------------a1. Earliest Start Time (EST) Scheduling ----------

tic = time_ns()

earliest_start_time_tasks_scheduled = earliest_start_time(deepcopy(task_list))

print(f"Time elapsed for EST: {(time_ns()-tic)/1_000_000} ms (EST)")

pretty_plot(earliest_start_time_tasks_scheduled, "Scheduled by EST", WINDOW_LENGTH)
summary_stats(earliest_start_time_tasks_scheduled, "EST Scheduling", most_cost)

# -------------------------------a2. Earliest Deadline (ED) Scheduling -------------

tic = time_ns()
earliest_deadline_tasks_scheduled = schedule_earliest_deadline(deepcopy(task_list))

print(f"Time elapsed for ED: {(time_ns()-tic)/1_000_000} ms (ED)")

pretty_plot(
    earliest_deadline_tasks_scheduled, "Earliest Deadline Scheduling", WINDOW_LENGTH
)
summary_stats(earliest_deadline_tasks_scheduled, "ED Scheduling", most_cost)

# -------------------------------b. Random Shifted Start Time (RSST) Scheduling ----

tic = time_ns()
rsst_tasks_scheduled = random_shifted_start_time(deepcopy(task_list), ITR_RSST)

print(f"Time elapsed for RSST: {(time_ns()-tic)/1_000_000} ms (ED)")

pretty_plot(rsst_tasks_scheduled, "RSST Scheduling", WINDOW_LENGTH)
summary_stats(rsst_tasks_scheduled, "RSST Scheduling", most_cost)

# -------------------------------c. Dual-Side Scheduling with RSST (DSS) -----------

tic = time_ns()
dss_tasks_scheduled = dual_side_with_rsst(
    deepcopy(task_list), ITR_RSST, ITR_RSST_DUALSIDE
)

print(f"Time elapsed for DSS: {(time_ns()-tic)/1_000_000} ms (DSS)")

pretty_plot(dss_tasks_scheduled, "DSS Scheduling", WINDOW_LENGTH)
summary_stats(dss_tasks_scheduled, "DSS Scheduling", most_cost)

# d. front line assembly

tic = time_ns()
fla_scheduled = front_line_assembly(deepcopy(task_list))


print(f"Time elapsed for FLA: {(time_ns()-tic)/1_000_000} ms (FLA)")

pretty_plot(fla_scheduled, "FLA Scheduling", WINDOW_LENGTH)
summary_stats(fla_scheduled, "FLA Scheduling", most_cost)

plt.show()
