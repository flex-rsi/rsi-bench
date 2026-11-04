"""Starter: X-VLA passthrough. Queries X-VLA when the queue is empty and executes all 30 actions.

It uses neither the run's model (model_client.py) nor geometry (geometry.py).

A fresh CPU process seeds Python and NumPy with zero, creates Controller, and
calls reset once per episode. Deterministic controllers must make the same
decisions for the same public observations and policy replies during replay.
"""

from collections import deque

import numpy as np


class Controller:
    def reset(self):
        self.actions = deque()

    def step(self, observation, query_policy):
        if not self.actions:
            self.actions.extend(query_policy())
        return np.array(self.actions.popleft(), dtype=np.float32, copy=True)
