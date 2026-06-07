"""Phase 0 E2E test: sensor data stream integrity.

Verifies that the MuJoCo environment provides complete and valid sensor
data: joint states, end-effector pose, camera images, and gripper feedback.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestSensorStream:
    """Validate sensor data from the MuJoCo environment."""

    def test_obs_structure(self, mujoco_env, home_joint_angles):
        """Observation dictionary has all required keys."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        obs = env.get_obs()
        required_keys = [
            "joint_angles", "joint_velocities",
            "ee_position", "ee_orientation",
            "gripper_width", "gripper_force",
        ]
        for key in required_keys:
            assert key in obs, f"Missing observation key: {key}"

    def test_joint_angles_range(self, mujoco_env, home_joint_angles):
        """Joint angles are within UR5 limits."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        q = env.get_joint_angles()
        assert len(q) == 6
        assert np.all(np.abs(q) <= np.pi * 2)  # within ±2π

    def test_ee_pose_is_valid(self, mujoco_env, home_joint_angles):
        """End-effector pose matrix is valid SE(3)."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        T = env.get_ee_pose()
        assert T.shape == (4, 4)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-6
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    def test_gripper_state(self, mujoco_env, home_joint_angles):
        """Gripper state is physically valid."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.0)  # closed
        for _ in range(200):
            env.step()

        state = env.get_gripper_state()
        width = state["width"]
        assert 0.0 <= width <= 0.1  # gripper width in meters
        assert state["left_force"] >= 0
        assert state["right_force"] >= 0

    def test_depth_image(self, mujoco_env, home_joint_angles):
        """Depth rendering produces valid output when enabled."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        depth = env.get_depth()
        if depth is not None:
            assert depth.ndim == 2
            assert depth.shape[0] > 0 and depth.shape[1] > 0
            assert np.all(depth >= 0)  # depth values non-negative

    def test_rgb_frame(self, mujoco_env, home_joint_angles):
        """RGB rendering produces valid image."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        rgb = env.render()
        if rgb is not None:
            assert rgb.ndim == 3
            assert rgb.shape[2] == 3  # RGB channels
            assert rgb.dtype == np.uint8

    def test_step_state_consistency(self, mujoco_env, home_joint_angles):
        """Consecutive steps without action maintain consistent state."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        q1 = env.get_joint_angles().copy()
        for _ in range(50):
            env.step()
        q2 = env.get_joint_angles()

        # Without external forces, joint angles should be stable
        diff = np.linalg.norm(q2 - q1)
        assert diff < 0.01, f"Joint drift too large: {diff:.4f} rad"


@pytest.mark.slow
@pytest.mark.e2e
class TestSensorStreamExtended:
    """Extended sensor checks — run less frequently."""

    def test_sensor_frame_rate(self, mujoco_env, home_joint_angles):
        """Camera rendering can sustain reasonable frame rate."""
        import time
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(100):
            env.step()

        n_frames = 30
        t0 = time.perf_counter()
        for _ in range(n_frames):
            env.render()
            env.step()
        elapsed = time.perf_counter() - t0

        fps = n_frames / elapsed
        # With hardware rendering, should be at least 10 FPS
        assert fps > 5, f"Frame rate too low: {fps:.1f} FPS"
