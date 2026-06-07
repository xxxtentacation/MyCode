"""Phase 0 E2E test: keyboard teleop smoke test.

Verifies that the teleop module can be imported, that the IK-driven EE
motion produces valid joint targets, and that basic key mappings work.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestTeleopSmoke:
    """Smoke tests for the teleop subsystem — no viewer required."""

    def test_import_teleop(self):
        """Teleop module imports without errors."""
        import scripts.teleop  # noqa: F401

    def test_ee_delta_computation(self):
        """Key combinations produce expected 6-DoF deltas."""
        from scripts.teleop import _compute_ee_delta

        # No keys pressed → zero delta
        delta = _compute_ee_delta(set())
        np.testing.assert_array_equal(delta, np.zeros(6))

        # Single key: W → +x translation
        delta = _compute_ee_delta({"w"})
        assert delta[0] > 0
        assert delta[1] == 0
        assert delta[2] == 0

        # S → -x translation
        delta = _compute_ee_delta({"s"})
        assert delta[0] < 0

        # All keys pressed simultaneously
        all_keys = set("wasdqejlikuo")
        delta = _compute_ee_delta(all_keys)
        assert np.any(delta != 0)

    def test_home_pose_valid(self):
        """Home pose is a valid 6-DoF joint configuration."""
        from scripts.teleop import HOME_Q
        assert len(HOME_Q) == 6
        assert np.all(np.abs(HOME_Q) <= np.pi * 2)

    def test_euler_rot_produces_so3(self):
        """_euler_to_rot returns a valid SO(3) matrix."""
        from scripts.teleop import _euler_to_rot

        R = _euler_to_rot(0.1, -0.2, 0.3)
        # R @ R.T should be identity
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # det(R) should be 1
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-10


@pytest.mark.slow
@pytest.mark.e2e
class TestTeleopIntegration:
    """Full teleop loop with MuJoCo viewer — requires display."""

    def test_ik_solve_for_home(self, ik_solver, home_joint_angles):
        """IK can recover home pose from FK of home pose."""
        T_home = ik_solver.forward_kinematics(home_joint_angles)
        q_sol = ik_solver.solve(T_home, home_joint_angles)
        assert q_sol is not None
        from control.ik_solver import _angle_diff
        diff = np.linalg.norm(_angle_diff(q_sol, home_joint_angles))
        assert diff < 0.1, f"Home IK error: {diff:.4f} rad"

    def test_teleop_ee_step(self, mujoco_env, ik_solver, home_joint_angles):
        """A small EE translation step produces valid IK solution."""
        env = mujoco_env
        ik = ik_solver

        env.set_joint_angles(home_joint_angles, 0.08)
        for _ in range(200):
            env.step()

        T_current = env.get_ee_pose()
        T_target = T_current.copy()
        T_target[2, 3] += 0.01  # move up 1 cm

        q_sol = ik.solve(T_target, env.get_joint_angles())
        assert q_sol is not None, "IK failed for small EE translation"

        # Verify FK of solution matches target
        T_result = ik.forward_kinematics(q_sol)
        pos_err = np.linalg.norm(T_result[:3, 3] - T_target[:3, 3])
        assert pos_err < 0.03, f"EE position error {pos_err:.4f}m"
