"""Evaluator-owned dev evidence; no diagnostic data enters the CPU controller."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def command_tracking(before, action, after, observed_step):
    """Compare an absolute command with the next public robot observation.

    Positive progress means displacement toward the commanded position. This
    describes robot motion and gripper joints, not object contact or possession.
    """
    action = np.asarray(action, dtype=np.float64)
    initial = np.asarray(before["eef_pos"], dtype=np.float64)
    observed = np.asarray(after["eef_pos"], dtype=np.float64)
    requested = action[:3] - initial
    moved = observed - initial
    requested_norm = float(np.linalg.norm(requested))
    rot6d = np.asarray(after["eef_rot6d"], dtype=np.float64)
    matrix = np.column_stack((rot6d[:3], rot6d[3:], np.cross(rot6d[:3], rot6d[3:])))
    target = Rotation.from_rotvec(action[3:6])
    rotation_error = (target * Rotation.from_matrix(matrix).inv()).magnitude()
    result = {
        "observed_step": observed_step,
        "observed_proprio": json_value(after),
        "requested_delta_m": requested.tolist(),
        "observed_delta_m": moved.tolist(),
        "position_error_m": (action[:3] - observed).tolist(),
        "position_error_norm_m": float(np.linalg.norm(action[:3] - observed)),
        "progress_along_command_m": float(np.dot(moved, requested) / requested_norm)
            if requested_norm > 0 else 0.0,
        "orientation_error_rad": float(rotation_error),
        "gripper_command": float(action[6]),
    }
    if "gripper_qpos" in before and "gripper_qpos" in after:
        result["gripper_qpos_delta_m"] = (
            np.asarray(after["gripper_qpos"]) - np.asarray(before["gripper_qpos"])).tolist()
    return result


def tracking_summary(steps):
    transitions = [step["command_tracking"] for step in steps if "command_tracking" in step]
    if not transitions:
        return {"observed_transitions": 0}
    errors = [item["position_error_norm_m"] for item in transitions]
    return {"observed_transitions": len(transitions),
            "mean_position_error_m": float(np.mean(errors)),
            "max_position_error_m": max(errors),
            "max_orientation_error_rad": max(item["orientation_error_rad"] for item in transitions)}


class DevEvidence:
    """Record public decisions and separate trusted physical observations.

    Query/action provenance describes exact equality with the latest policy
    reply. A changed controller action is kept as a controller action; duplicate
    chunk rows expose all matching indices rather than inventing an index.
    """

    def __init__(self, root, trusted_sample=None):
        self.root = Path(root)
        self.trusted_sample = trusted_sample
        self.steps, self.queries, self.stage_steps = [], [], []
        self.active_query = None
        self.first_reached, self.stage_clips = {}, {}
        self.frames, self.sparse_steps = {}, []
        self.recent = collections.deque(maxlen=3)
        self.last_frame = None

    def record_post_action(self, public):
        if self.steps:
            previous = self.steps[-1]
            previous["command_tracking"] = command_tracking(
                previous["proprio"], previous["action"], public["proprio"], public["step"])

    def record_pre_action(self, public):
        self.record_post_action(public)
        self.steps.append({"policy_step": public["step"],
                           "instruction": public["instruction"],
                           "proprio": json_value(public["proprio"]),
                           "last_action": json_value(public["last_action"])})

    def record_query(self, step, source, chunk):
        self.active_query = {"policy_step": step, "source": source,
                             "action_chunk_shape": [30, 7], "actions": chunk.tolist(),
                             "executed_action_count": 0}
        self.queries.append(self.active_query)

    def record_executed(self, action):
        source = {"kind": "controller_action"}
        if self.active_query is not None:
            chunk = np.asarray(self.active_query["actions"], dtype=np.float32)
            matches = np.flatnonzero(np.all(chunk == action, axis=1)).tolist()
            if matches:
                source = {"kind": "exact_query_match",
                          "query_step": self.active_query["policy_step"]}
                if len(matches) == 1:
                    source["chunk_index"] = matches[0]
                else:
                    source["candidate_chunk_indices"] = matches
                self.active_query["executed_action_count"] += 1
        self.steps[-1].update(action=action.tolist(), action_source=source)

    @staticmethod
    def preview(raw):
        from PIL import Image
        views = [Image.fromarray(np.asarray(raw[key], dtype=np.uint8)).resize((180, 180))
                 for key in ("agentview_image", "robot0_eye_in_hand_image")]
        sheet = Image.new("RGB", (360, 180))
        for index, view in enumerate(views):
            sheet.paste(view, (180 * index, 0))
        return sheet

    def observe_state(self, raw, env, step):
        if self.trusted_sample is not None:
            sample = json_value(self.trusted_sample(env, raw))
            self.stage_steps.append({"policy_step": step, **sample})
            reached = [key for key, value in sample.items()
                       if value is True and key not in self.first_reached]
            for key in reached:
                self.first_reached[key] = step
                self.stage_clips[key] = list(range(max(0, step - 2), step + 3))
        frame = (step, self.preview(raw))
        self.last_frame = frame
        self.recent.append(frame)
        if step % 30 == 0:
            self.sparse_steps.append(step)
            self.frames[step] = frame[1]
        wanted = {index for indices in self.stage_clips.values() for index in indices}
        for index, image in self.recent:
            if index in wanted:
                self.frames[index] = image

    def finish(self, native_row):
        self.root.mkdir(parents=True, exist_ok=True)
        queries = native_row.get("policy_queries", [])
        events = native_row.get("intervention_events", [])
        event_step = events[0]["cosmos_query_boundary_step"] if events else None
        query_steps = [query["policy_step"] for query in queries if query["source"] == "model"]
        next_query = next((step for step in query_steps
                           if event_step is not None and step >= event_step), None)
        public = {"timing": {"event_step": event_step, "next_query_step": next_query,
                            "query_delay_steps": None if next_query is None else next_query - event_step},
                  "query_steps": query_steps, "queries": self.queries, "steps": self.steps,
                  "tracking_basis": "action at t compared with public robot proprio at t+1; no object-state inference",
                  "tracking_summary": tracking_summary(self.steps),
                  "action_source_basis": "exact equality with latest returned policy chunk; not controller intent",
                  "frame_sequence_path": "dev_frames.json"}
        (self.root / "dev_trajectory.json").write_text(json.dumps(public, indent=2))
        if self.stage_steps:
            stages = {"timing_basis": "0 is settled initial state; k is the observation after k policy actions",
                      "first_reached_steps": self.first_reached, "steps": self.stage_steps}
            (self.root / "dev_stage_diagnostics.json").write_text(json.dumps(stages, indent=2))
        if self.last_frame is not None:
            step, image = self.last_frame
            if step not in self.sparse_steps:
                self.sparse_steps.append(step)
            self.frames[step] = image
        folder = self.root / "dev_frames"
        folder.mkdir(exist_ok=True)
        images = []
        for step, image in sorted(self.frames.items()):
            path = folder / f"step-{step:04d}.png"
            image.save(path)
            images.append(str(path.relative_to(self.root)))
        sequence = {"observation_step_basis": "0 is settled initial state; k follows k policy actions",
                    "sparse_steps": self.sparse_steps,
                    "stage_clips": {key: [step for step in steps if step in self.frames]
                                    for key, steps in self.stage_clips.items()},
                    "frame_steps": sorted(self.frames), "image_files": images}
        (self.root / "dev_frames.json").write_text(json.dumps(sequence, indent=2))
