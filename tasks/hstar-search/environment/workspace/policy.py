"""Starter policy: the H*Bench baseline ("free_think") without a written thought process. One model
call per step on the whole conversation: the model sees every view so far with the environment's
feedback and answers <answer>rotate(dyaw,dpitch)</answer> or <answer>submit(yaw,pitch)</answer>.
Prompts are those of the H*Bench environment, except that the model is not asked to write out its
reasoning in <think> tags (models reason on their own; Claude Opus 5.5 refuses requests that ask for
its reasoning in the reply).

The evaluator calls search(env) once per episode, several episodes at a time (WORKERS):
    env.kind        "hos" (find an object) or "hps" (find the direction to move in)
    env.task        the instruction
    env.view        path of the current view: JPEG, 960 x 720, 100-degree field of view, green cross
                    at the center
    env.yaw, env.pitch    current direction in degrees (yaw in [0, 360), pitch in [-90, 90])
    env.steps_left, env.done
    env.rotate(dyaw, dpitch) -> path of the new view (None after the last step)
    env.submit(yaw, pitch)    ends the episode with that absolute direction
"""

import re

from model import chat, image_part

WORKERS = 8
MAX_TOKENS = 8192

_ACTIONS = """Actions you can take: rotate(yaw:int,pitch:int), submit(yaw:int,pitch:int).
rotate(yaw:int,pitch:int): rotate the camera in the yaw and pitch direction relative to the current direction. Yaw is the rotation angle in the x-y plane, pitch is the rotation angle in the y-z plane. Yaw angle > 0 means rotate to the right, yaw angle < 0 means rotate to the left. Pitch angle > 0 means look up, pitch angle < 0 means look down.
"""
_EXAMPLE = """Example:
Round 1:
image_1
<answer>rotate(-45,0)</answer>
Round 2:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (315,0).
image_2
<answer>rotate(-30,-5)</answer>
Round 3:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (285,-5).
image_3
<answer>submit(285,-5)</answer>
Round 4:
Env_feedback: Success
"""
FORMAT = """You can take 1 action(s) at a time.
Your response should be in the format of:
<answer>...</answer>"""
SYSTEM = {
    "hos": ("You are a robot and perform object searching tasks according to instructions. Your goal "
            "is to rotate the camera to center the target object in the camera view, and then submit "
            "the task. The camera center is presented as a green cross in the picture.\n" + _ACTIONS
            + "submit(yaw:int,pitch:int): submit the task with the current camera view with the target "
            "object at the center, yaw and pitch are the angles of the current camera view, which is "
            "reported by the environment.\n\nYou can only take one action at a time. The instruction "
            "will be provided with each observation. Look at the image carefully to complete the "
            "instruction.\n\n" + _EXAMPLE),
    "hps": ("You are a robot and perform navigation tasks according to instructions. Your goal is to "
            "turn your camera center to the target direction you need to move towards to reach the "
            "target location. The camera center is presented as a green cross in the picture. Don't "
            "move in the inavailable direction, such as obstacles or gaps.\n" + _ACTIONS
            + "submit(yaw:int,pitch:int): submit the task with the current direction to move torwards, "
            "yaw and pitch are the angles of the current camera view, which is reported by the "
            "environment.\nYou can only take one action at a time. The instruction will be provided "
            "with each observation. Look at the image carefully to complete the instruction.\n\n"
            + _EXAMPLE),
}
_EXAMPLE_LINE = "e.g. <answer>rotate(-45,0)</answer>"
ANSWER = re.compile(r"<answer>\s*(rotate|submit)\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
                    r"\s*</answer>", re.S)


def parse(text: str):
    found = ANSWER.findall(text or "")
    if not found:
        return None
    name, a, b = found[-1]
    return name, round(float(a)), round(float(b))


def turn(head: str, view: str, task: str) -> list:
    return [{"type": "text", "text": head}, image_part(view),
            {"type": "text", "text": f"\nHuman Instruction: {task}\nDecide your next action.\n{FORMAT}"}]


def search(env) -> None:
    system = SYSTEM[env.kind] + FORMAT + "\n" + _EXAMPLE_LINE
    messages = [{"role": "user", "content": turn("[initial observation]:\n", env.view, env.task)}]
    for _ in range(env.max_steps):
        text, reply = chat(messages, system=system, max_tokens=MAX_TOKENS)
        messages.append(reply)
        action = parse(text)
        if action and action[0] == "submit":
            env.submit(action[1], action[2])
            return
        if action:
            view = env.rotate(action[1], action[2])
            if view is None:  # that was the last step
                return
            valid = f"rotate({action[1]},{action[2]})"
            feedback = (f"Last action executed successfully, your current direction (yaw,pitch) is "
                        f"({env.yaw},{env.pitch})")
        else:
            view, valid, feedback = env.view, "None", "Last action is not executed successfully."
        head = (f"After your answer, the extracted valid action is {valid}.\nThe environment feedback "
                f"is: {feedback}\ndone: False\nAfter that, the observation is:\n")
        messages.append({"role": "user", "content": turn(head, view, env.task)})
