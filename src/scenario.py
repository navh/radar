import json
from typing import List

from task import Task


class Scenario:
    def __init__(self):
        self.tasks: List[Task] = []

    # def generate_random_tasks(self, n_tasks: int):
    #     pass  # TODO

    def tasks_from_tasklist(self, tasklist: List[Task]):
        self.tasks = tasklist

    def tasks_from_json(self, scene_spec_dict: str):
        # scene_spec_dict = json.loads(scenario_specification)
        for task_spec in scene_spec_dict["tasks"]:
            self.tasks.append(
                Task(
                    t_dwell=float(task_spec["t_dwell"]),
                    t_start=float(task_spec["t_start"]),
                    t_earliest=float(task_spec["t_earliest"]),
                    t_latest=float(task_spec["t_latest"]),
                    priority=float(task_spec["priority"]),
                    color=task_spec["color"],
                )
            )

    def tasks_to_scenario_specification(self) -> str:
        # TODO: should probably just implement __repr__ on Task
        # don't want to break other things, so for now, here it is
        lodo_tasks = [
            {
                "t_dwell": task.t_dwell,
                "t_start": task.t_start,
                "t_earliest": task.t_earliest,
                "t_latest": task.t_latest,
                "priority": task.priority,
                "color": task.color,
            }
            for task in self.tasks
        ]

        scene_spec_dict = {"tasks": lodo_tasks}

        return json.dumps(scene_spec_dict)
