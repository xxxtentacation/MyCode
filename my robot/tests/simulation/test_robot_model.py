"""Phase 0 simulation tests: robot model and physics validation.

Verifies the MuJoCo scene loads correctly and the UR5 model has the
expected physical properties (joint limits, DOF count, sensors).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestRobotModel:
    """Validate the UR5 + Robotiq scene in MuJoCo."""

    def test_model_loads(self, mujoco_env):
        """Scene loads without MuJoCo errors."""
        assert mujoco_env.model is not None
        assert mujoco_env.data is not None

    def test_dof_count(self, mujoco_env):
        """Model has correct number of degrees of freedom."""
        nq = mujoco_env.model.nq
        nv = mujoco_env.model.nv
        # UR5 (6) + gripper (2 joints) + object free joint (7)
        assert nq >= 8, f"Expected >= 8 position DOFs, got {nq}"
        assert nv >= 8, f"Expected >= 8 velocity DOFs, got {nv}"

    def test_joint_limits(self, mujoco_env):
        """Arm joints have reasonable limits."""
        # Joint limits for the 6 arm joints
        for i in range(6):
            jid = mujoco_env.model.jnt_qposadr[i]
            low = mujoco_env.model.jnt_range[i, 0] if i < mujoco_env.model.jnt_range.shape[0] else -np.pi
            high = mujoco_env.model.jnt_range[i, 1] if i < mujoco_env.model.jnt_range.shape[0] else np.pi
            assert low <= high
            assert low >= -2 * np.pi
            assert high <= 2 * np.pi

    def test_sensors_exist(self, mujoco_env):
        """End-effector and force sensors are defined."""
        nsensor = mujoco_env.model.nsensor
        assert nsensor >= 2, f"Expected >= 2 sensors, got {nsensor}"

    def test_actuators_exist(self, mujoco_env):
        """Position actuators are defined for arm and gripper."""
        nu = mujoco_env.model.nu
        assert nu >= 8, f"Expected >= 8 actuators (6 arm + 2 gripper), got {nu}"

    def test_camera_exists(self, mujoco_env):
        """The wrist_cam camera is defined in the scene."""
        ncam = mujoco_env.model.ncam
        assert ncam >= 1, f"Expected >= 1 camera, got {ncam}"

    def test_initial_reset_state(self, mujoco_env, home_joint_angles):
        """After reset + home pose, EE is above the table."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(200):
            env.step()

        ee_pos = env.get_ee_pose()[:3, 3]
        # EE should be above the ground (z > 0)
        assert ee_pos[2] > 0.1, f"EE too low: z={ee_pos[2]:.3f}m"
        # EE should be above the table (table height ~0.4m)
        assert ee_pos[2] > 0.3, f"EE below table: z={ee_pos[2]:.3f}m"

    def test_gripper_actuators_work(self, mujoco_env, home_joint_angles):
        """Gripper position commands change gripper state."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(200):
            env.step()
        width_open = env.get_gripper_state()["width"]

        env.set_joint_angles(home_joint_angles, 0.0)
        for _ in range(200):
            env.step()
        width_closed = env.get_gripper_state()["width"]

        assert width_open > width_closed, (
            f"Gripper open ({width_open:.4f}) should be > closed ({width_closed:.4f})"
        )


class TestPhysics:
    """Basic physics sanity checks."""

    def test_gravity(self, mujoco_env):
        """Gravity is enabled and points downward."""
        grav = mujoco_env.model.opt.gravity
        assert grav[2] < 0, f"Gravity should point down, got {grav}"

    def test_timestep(self, mujoco_env):
        """Simulation timestep is reasonable for manipulation."""
        dt = mujoco_env.model.opt.timestep
        assert 0.001 <= dt <= 0.01, f"Timestep {dt:.4f}s out of reasonable range"

    def test_no_nan_in_state(self, mujoco_env, home_joint_angles):
        """After settling, no NaN values in joint state."""
        env = mujoco_env
        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(200):
            env.step()

        q = env.get_joint_angles()
        qd = env.get_joint_velocities()
        assert not np.any(np.isnan(q)), "NaN in joint angles"
        assert not np.any(np.isnan(qd)), "NaN in joint velocities"
        # Also check for inf
        assert not np.any(np.isinf(q)), "Inf in joint angles"
