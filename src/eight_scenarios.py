import json
from copy import deepcopy
from pathlib import Path
from time import time_ns
from typing import List

import numpy as np

from scenario import Scenario
from schedule_dual_side_with_rsst import dual_side_with_rsst
from schedule_earliest_deadline import schedule_earliest_deadline
from schedule_earliest_start_time import earliest_start_time
from schedule_front_line_assembly import front_line_assembly
from schedule_random_shifted_start_time import random_shifted_start_time
from task import Task

ITR_RSST = int(float("1e3"))  # 1e3 to match matlab, could we invest in a couple zeros?
# ITR_RSST = 1_000_000_000 # I find this easier to parse than 1e9, either works though

TOTAL_BORDERS = 10
ITR_RSST_DUALSIDE = 50


def summary_stats(
    tasks: List[Task], schedule_name: str, n_actual: int, all_drop_cost: float
):
    drop_count = sum(1 for task in tasks if task.dropped)
    print(
        f"     Executed tasks count: {n_actual-drop_count}/{n_actual} ({schedule_name}) | Dropped tasks count: {drop_count}/{n_actual} ({schedule_name})"
    )
    normalized_cost = sum(task.calculate_cost() for task in tasks) / all_drop_cost
    print(
        f"     Normalized cost calculated by {schedule_name}: {normalized_cost} (0~1)"
    )


run_path = "./eight_scenarios/run_1677256728178111000"

states = [
    "state_1.json",
    "state_2.json",
    "state_3.json",
    "state_4.json",
    "state_5.json",
    "state_6.json",
    "state_7.json",
    "state_8.json",
]

for state in states:
    file = Path(run_path) / state
    with file.open() as f:
        scene_dict = json.load(f)
    assert scene_dict is not None

    scenario = Scenario()
    scenario.tasks_from_json(scene_dict)

    task_list = scenario.tasks
    max_drop_cost = sum(task.drop_cost() for task in task_list)

    print(f"--------------Designed for {len(task_list)} Tasks-----------------")
    print(state)

    # -------------------------------a1. Earliest Start Time (EST) Scheduling ----------

    tic = time_ns()

    earliest_start_time_tasks_scheduled = earliest_start_time(deepcopy(task_list))

    print(f"Time elapsed for EST: {(time_ns()-tic)/1_000_000} ms (EST)")

    summary_stats(
        earliest_start_time_tasks_scheduled,
        "EST Scheduling",
        len(task_list),
        max_drop_cost,
    )

    # -------------------------------a2. Earliest Deadline (ED) Scheduling -------------

    tic = time_ns()
    earliest_deadline_tasks_scheduled = schedule_earliest_deadline(deepcopy(task_list))

    print(f"Time elapsed for ED: {(time_ns()-tic)/1_000_000} ms (ED)")

    summary_stats(
        earliest_deadline_tasks_scheduled,
        "ED Scheduling",
        len(task_list),
        max_drop_cost,
    )

    # -------------------------------b. Random Shifted Start Time (RSST) Scheduling ----

    tic = time_ns()
    rsst_tasks_scheduled = random_shifted_start_time(deepcopy(task_list), ITR_RSST)

    print(f"Time elapsed for RSST: {(time_ns()-tic)/1_000_000} ms (ED)")

    summary_stats(
        rsst_tasks_scheduled, "RSST Scheduling", len(task_list), max_drop_cost
    )

    # -------------------------------c. Dual-Side Scheduling with RSST (DSS) -----------

    tic = time_ns()
    dss_tasks_scheduled = dual_side_with_rsst(
        deepcopy(task_list), ITR_RSST, ITR_RSST_DUALSIDE
    )

    print(f"Time elapsed for DSS: {(time_ns()-tic)/1_000_000} ms (DSS)")

    summary_stats(dss_tasks_scheduled, "DSS Scheduling", len(task_list), max_drop_cost)

    # d. front line assembly

    tic = time_ns()
    fla_scheduled = front_line_assembly(deepcopy(task_list))

    print(f"Time elapsed for FLA: {(time_ns()-tic)/1_000_000} ms (FLA)")

    summary_stats(fla_scheduled, "FLA Scheduling", len(task_list), max_drop_cost)
