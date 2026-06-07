import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from control.ik_solver import UR5IK, _angle_diff


def _random_ee_pose_in_workspace(rng: np.random.Generator) -> np.ndarray:
    """Generate a random reachable end-effector pose via FK."""
    # Sample joint angles in valid range, compute FK, then perturb slightly
    q = rng.uniform(-pi_deg(170), pi_deg(170), size=6)
    T = UR5IK.forward_kinematics(q)
    # Small perturbation to orientation
    rx = rng.uniform(-0.3, 0.3)
    ry = rng.uniform(-0.3, 0.3)
    rz = rng.uniform(-0.3, 0.3)
    R_delta = _euler_to_rot(rx, ry, rz)
    T[:3, :3] = T[:3, :3] @ R_delta
    return T


def pi_deg(deg):
    return deg * np.pi / 180.0


def _euler_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class TestUR5IK:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ik = UR5IK()

    def test_forward_kinematics_structure(self):
        T = self.ik.forward_kinematics(np.zeros(6))
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])

    def test_ik_fk_consistency(self):
        """IK solution -> FK should recover target pose within tolerance."""
        rng = np.random.default_rng(42)
        passed = 0
        for _ in range(30):
            T_target = _random_ee_pose_in_workspace(rng)
            q_near = rng.uniform(-pi_deg(170), pi_deg(170), size=6)
            q_sol = self.ik.solve(T_target, q_near)
            if q_sol is not None:
                T_result = self.ik.forward_kinematics(q_sol)
                pos_err = np.linalg.norm(T_result[:3, 3] - T_target[:3, 3])
                assert pos_err < 0.02, f"Position error {pos_err:.4f} too large"
                passed += 1
        assert passed >= 20, f"Only {passed}/30 poses solved"

    def test_nearest_solution_selected(self):
        """IK returns solution closest to current config."""
        rng = np.random.default_rng(1)
        T_target = _random_ee_pose_in_workspace(rng)
        q_far = np.array([2.0, -0.5, 0.5, -0.5, 0.5, -2.0])
        q_near = np.array([0.3, -1.5, 1.5, 0.1, 1.5, 0.1])

        sol_far = self.ik.solve(T_target, q_far)
        sol_near = self.ik.solve(T_target, q_near)

        if sol_far is not None and sol_near is not None:
            dist_far = float(np.linalg.norm(_angle_diff(sol_far, q_far)))
            dist_near = float(np.linalg.norm(_angle_diff(sol_near, q_near)))
            assert dist_far <= dist_near + 0.2  # tolerance for workspace edge

    def test_unreachable_pose_returns_none(self):
        """Pose far outside workspace returns None."""
        T = np.eye(4)
        T[:3, 3] = [10.0, 10.0, 10.0]
        result = self.ik.solve(T, np.zeros(6))
        assert result is None

    def test_home_ik_roundtrip(self):
        """IK on FK of home pose should recover home angles."""
        q_home = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
        T_home = self.ik.forward_kinematics(q_home)
        q_sol = self.ik.solve(T_home, q_home)
        assert q_sol is not None
        diff = np.linalg.norm(_angle_diff(q_sol, q_home))
        assert diff < 0.1, f"Home IK error: {diff:.4f}"
