import random

from globals import DROP_COST_DERIVATIVE, DROP_PENALTY, N_TASKS


class Task:
    def __init__(
        self,
        t_dwell=float("nan"),
        t_start=float("nan"),
        t_earliest=float("nan"),
        t_latest=float("nan"),
        priority=float("nan"),
        t_scheduled=float("nan"),
        dropped=False,
        color="black",
    ):
        self.t_dwell = t_dwell
        self.t_start = t_start
        self.t_earliest = t_earliest
        self.t_latest = t_latest
        self.priority = priority
        self.t_scheduled = t_scheduled
        self.unscheduled = True
        self.dropped = dropped
        self.color = color

    def uniform_randomize(
        self,
        n: int,  # TODO: should this be n_actual?
        minimum_t: float,
        maximum_t: float,
        tau: float,
        minimum_priority: int,
        maximum_priority: int,
    ):
        self.t_dwell = random.uniform(0, 2 / n)  # (on average, 1/n seconds)
        self.t_start = random.uniform(0, maximum_t - self.t_dwell)
        self.t_earliest = max(minimum_t, self.t_start - tau)
        self.t_latest = min(maximum_t, self.t_start + tau)
        self.priority = 0.1 * random.randint(minimum_priority, maximum_priority)
        self.t_scheduled = self.t_start  # hack to allow plotting initial layout
        self.dropped = False

    def drop_task(self) -> None:
        self.unscheduled = False
        self.dropped = True
        self.t_scheduled = float("nan")

    def schedule_at(self, t_scheduled: float) -> None:
        self.unscheduled = False
        self.dropped = False
        self.t_scheduled = t_scheduled

    def get_tau(self, t_scheduled) -> float:
        assert self.dropped is False, "Task dropped, has no t_scheduled"
        if t_scheduled <= self.t_start:
            return self.t_start - self.t_earliest
        else:
            return self.t_latest - self.t_start

    def drop_cost(self) -> float:
        return (self.priority * DROP_PENALTY) ** 2

    def calculate_cost(self, t_scheduled=None) -> float:
        if self.dropped:
            return self.drop_cost()
        if t_scheduled is None:
            t_scheduled = self.t_scheduled
        return (
            self.priority
            * ((self.t_start - t_scheduled) / self.get_tau(t_scheduled)) ** 2
        )

    def calculate_cost_derivative(self, t_scheduled=None) -> float:
        if self.dropped:
            return DROP_COST_DERIVATIVE
        if t_scheduled is None:
            t_scheduled = self.t_scheduled
        return (
            2
            * (self.t_scheduled - self.t_start)
            / N_TASKS
            * self.priority**2
            / self.get_tau(t_scheduled) ** 2
        )
