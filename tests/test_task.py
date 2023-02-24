import unittest

from src.task import Task


class TestTask(unittest.TestCase):
    def test_drop_task(self):
        t = Task(1, 2, 3, 4, 5)
        self.assertEqual(t.dropped, False)
        t.drop_task()
        self.assertEqual(t.dropped, True)

    def test_schedule_at(self):
        t = Task(1, 2, 3, 4, 5)
        t.schedule_at(42)
        self.assertEqual(t.t_scheduled, 42)
