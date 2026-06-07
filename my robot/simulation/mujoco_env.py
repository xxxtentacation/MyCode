import os
import numpy as np
import mujoco
from mujoco import viewer


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCENE_PATH = os.path.join(_SCRIPT_DIR, "assets", "ur5e", "scene.xml")


class MujocoEnv:
    """MuJoCo simulation environment for UR5 manipulation.

    Provides a Gymnasium-like API for step/reset/render, plus
    convenience methods for joint state and end-effector queries.
    """

    # Joint indices for the 6 UR5 arm joints
    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist1", "wrist2", "wrist3"]
    GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]

    def __init__(self, scene_path: str | None = None, render_mode: str = "offscreen"):
        path = scene_path or _SCENE_PATH
        self._model = mujoco.MjModel.from_xml_path(path)
        self._data = mujoco.MjData(self._model)

        self.render_mode = render_mode
        self._renderer = None
        if render_mode == "offscreen":
            self._renderer = mujoco.Renderer(self._model, 640, 480)

        # Joint ID lookups
        self._arm_joint_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
                               for j in self.ARM_JOINTS]
        self._gripper_joint_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
                                   for j in self.GRIPPER_JOINTS]
        self._arm_actuator_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                                  for a in [f"j{i}_act" for i in range(1, 7)]]
        self._gripper_actuator_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                                      for a in ["left_grip_act", "right_grip_act"]]

        # Body ID for end-effector
        self._ee_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "flange")

        self._timestep = 0

    # ---- Core API ----

    def step(self) -> None:
        """Advance simulation by one timestep."""
        mujoco.mj_step(self._model, self._data)
        self._timestep += 1

    def reset(self) -> dict:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        self._timestep = 0
        return self.get_obs()

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- Rendering ----

    def render(self) -> np.ndarray | None:
        """Render RGB image. Returns (H, W, 3) uint8 array or None if no renderer."""
        if self._renderer is None:
            return None
        self._renderer.update_scene(self._data, camera="wrist_cam")
        return self._renderer.render()

    def get_depth(self) -> np.ndarray | None:
        """Render depth image. Returns (H, W) float32 array or None."""
        if self._renderer is None:
            return None
        self._renderer.update_scene(self._data, camera="wrist_cam")
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return depth

    # ---- Joint State ----

    def get_joint_angles(self) -> np.ndarray:
        """Return joint positions for 6 arm joints (rad)."""
        return np.array([self._data.qpos[jid] for jid in self._arm_joint_ids])

    def get_joint_velocities(self) -> np.ndarray:
        """Return joint velocities for 6 arm joints (rad/s)."""
        return np.array([self._data.qvel[self._model.jnt_dofadr[jid]]
                         for jid in self._arm_joint_ids])

    def set_joint_angles(self, q: np.ndarray, gripper_width: float | None = None) -> None:
        """Set target joint positions for the arm and optionally the gripper."""
        for i, qid in enumerate(self._arm_actuator_ids):
            self._data.ctrl[qid] = q[i]
        if gripper_width is not None:
            half = gripper_width / 2.0
            for gid in self._gripper_actuator_ids:
                self._data.ctrl[gid] = -half

    # ---- End-Effector ----

    def get_ee_pose(self) -> np.ndarray:
        """Return end-effector pose as 4x4 homogeneous matrix."""
        pos = self._data.xpos[self._ee_body_id].copy()
        xmat = self._data.xmat[self._ee_body_id].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = xmat
        T[:3, 3] = pos
        return T

    def get_ee_position(self) -> np.ndarray:
        """Return end-effector position (x, y, z)."""
        return self._data.xpos[self._ee_body_id].copy()

    # ---- Gripper ----

    def get_gripper_state(self) -> dict:
        """Return gripper state: width (m) and touch forces."""
        left_pos = self._data.qpos[self._gripper_joint_ids[0]]
        right_pos = self._data.qpos[self._gripper_joint_ids[1]]
        width = - (left_pos + right_pos)
        left_force = self._data.sensor("left_touch").data[0] if self._model.nsensor > 0 else 0.0
        right_force = self._data.sensor("right_touch").data[0] if self._model.nsensor > 0 else 0.0
        return {"width": float(width), "left_force": float(left_force), "right_force": float(right_force)}

    # ---- Misc ----

    def get_obs(self) -> dict:
        """Return observation dict with standardized keys."""
        ee_pose = self.get_ee_pose()
        gripper = self.get_gripper_state()
        return {
            "joint_angles": self.get_joint_angles(),
            "joint_velocities": self.get_joint_velocities(),
            "ee_pose": ee_pose,
            "ee_position": ee_pose[:3, 3],
            "ee_orientation": ee_pose[:3, :3],
            "gripper_width": gripper["width"],
            "gripper_force": max(gripper["left_force"], gripper["right_force"]),
            "gripper": gripper,
            "timestep": self._timestep,
        }

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def dt(self) -> float:
        return self._model.opt.timestep

    @staticmethod
    def launch_passive_viewer(scene_path: str | None = None):
        """Launch MuJoCo's built-in passive viewer (GUI mode). Blocks until window closes."""
        path = scene_path or _SCENE_PATH
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        with viewer.launch_passive(m, d) as v:
            while v.is_running():
                mujoco.mj_step(m, d)
                v.sync()
