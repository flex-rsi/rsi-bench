"""Starter: holds the arm still. Replace with a program that completes the task (task.md)."""


def solve(env) -> None:
    import torch

    action = torch.zeros(env.num_envs, env.robot.action_dim, device=env.device)
    for _ in range(10):
        env.step(action)
