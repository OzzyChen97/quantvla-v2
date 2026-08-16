#!/usr/bin/env python3
"""GR00T policy adapter for the RoboCerebra evaluation (v1.4, Stage D bridge).

RoboCerebra runs on the LIBERO fork, so the observation maps 1:1 onto the
GR00T LIBERO obs format (2 cameras + 8-dim state = eef_pos(3) + axisangle(3) +
gripper_qpos(2)). Talks to the GR00T inference service (ZMQ, same protocol as
run_libero_eval.py); self-contained (zmq + msgpack + numpy) so the
robocerebra_test conda env needs no gr00t data-stack deps.
"""

from collections import deque

import numpy as np


class _Gr00TZMQClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        import zmq
        import msgpack

        self.msgpack = msgpack
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            import io

            out = io.BytesIO()
            np.save(out, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": out.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            import io

            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def get_action(self, obs: dict) -> dict:
        import zmq

        request = {"endpoint": "get_action", "data": obs}
        self.sock.send(self.msgpack.packb(request, default=self._encode))
        try:
            resp = self.msgpack.unpackb(self.sock.recv(), object_hook=self._decode)
        except zmq.Again:
            raise RuntimeError("GR00T server timeout on get_action") from None
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp


def create_policy_client(host: str, port: int):
    return _Gr00TZMQClient(host, port)


def _to_gr00t_obs(observation: dict, desc: str) -> dict:
    state = np.asarray(observation["state"], dtype=np.float32)  # 8-dim
    return {
        "video.image": np.expand_dims(observation["full_image"], axis=0),
        "video.wrist_image": np.expand_dims(observation["wrist_image"], axis=0),
        "state.x": np.array([[state[0]]], dtype=np.float32),
        "state.y": np.array([[state[1]]], dtype=np.float32),
        "state.z": np.array([[state[2]]], dtype=np.float32),
        "state.roll": np.array([[state[3]]], dtype=np.float32),
        "state.pitch": np.array([[state[4]]], dtype=np.float32),
        "state.yaw": np.array([[state[5]]], dtype=np.float32),
        "state.gripper": np.expand_dims(state[6:8], axis=0),
        "annotation.human.action.task_description": [desc],
    }


def infer_chunk(client, observation, desc: str) -> np.ndarray:
    """Full GR00T action chunk (50, 7), gripper binarized."""
    out = client.get_action(_to_gr00t_obs(observation, desc))
    chunk = np.stack([
        np.atleast_1d(out[f"action.{k}"])
        for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    ], axis=-1)  # (H, 7)
    chunk = chunk.copy()
    # binarize the gripper like run_libero_eval's normalize_gripper_action
    chunk[..., -1] = np.where(chunk[..., -1] > 0.5, 1.0, -1.0)
    return chunk


def execute_policy_step(cfg, client, observation, desc, action_queue: deque) -> np.ndarray:
    """Mirrors the pi0.5 runner: replan every cfg.replan_steps env steps."""
    if not action_queue:
        chunk = infer_chunk(client, observation, desc)
        assert len(chunk) >= cfg.replan_steps, (
            f"replan every {cfg.replan_steps} steps but chunk has {len(chunk)}"
        )
        action_queue.extend(chunk[: cfg.replan_steps])
    return action_queue.popleft()
