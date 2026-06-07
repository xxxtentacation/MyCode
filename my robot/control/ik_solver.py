import numpy as np
from numpy import pi, sin, cos, arctan2, arccos, sqrt


class UR5IK:
    """UR5 analytical inverse kinematics.

    6-DoF with offset wrist → closed-form solution.
    Uses modified DH parameters (standard UR5 values in meters).

    The key insight for the offset wrist: in modified DH convention,
    the wrist axis z4 is exactly the negative of the end-effector y-axis.
    This gives an exact expression for the wrist center p_04 (frame 4 origin)
    that depends only on the target pose, decoupling position from orientation.
    """

    _d1 = 0.089159
    _a2 = -0.425
    _a3 = -0.39225
    _d4 = 0.10915
    _d5 = 0.09465
    _d6 = 0.0823

    _ARM_REACH = abs(_a2) + sqrt(_a3**2 + _d4**2)  # ≈ 0.832 m max reach

    def solve(self, T_target: np.ndarray, q_current: np.ndarray) -> np.ndarray | None:
        """Return the IK solution closest to q_current, or None if unreachable."""
        solutions = self._ik_analytical(T_target)
        if not solutions:
            return None
        best = min(solutions,
                   key=lambda s: float(np.linalg.norm(_angle_diff(s, q_current))))
        return best

    def _ik_analytical(self, T: np.ndarray) -> list[np.ndarray]:
        d1, a2, a3, d4, d5, d6 = self._d1, self._a2, self._a3, self._d4, self._d5, self._d6

        p = T[:3, 3]
        R = T[:3, :3]

        # ---- Wrist center (origin of frame 4) ----
        # In modified DH: z4 = -y_ee exactly (y_ee is end-effector y-axis).
        # p_04 = p_ee - d5*z4 - d6*z5
        #       = p_ee + d5*R[:,1] - d6*R[:,2]
        pw = p + d5 * R[:, 1] - d6 * R[:, 2]

        # ---- Workspace check ----
        r_horiz = sqrt(pw[0]**2 + pw[1]**2)
        if r_horiz < abs(d4) - 1e-6:
            return []
        dist_2d = sqrt(max(r_horiz**2 - d4**2, 0))
        vec_to_wrist = np.array([dist_2d, pw[2] - d1])
        wrist_dist = np.linalg.norm(vec_to_wrist)
        if wrist_dist > abs(a2) + sqrt(a3**2 + d4**2) + 0.01:
            return []

        solutions = []

        # ---- θ1: base rotation (2 solutions) ----
        for sign1 in [+1, -1]:
            t1 = arctan2(pw[1], pw[0]) - arctan2(d4, sign1 * dist_2d)

            c1, s1 = cos(t1), sin(t1)

            # Wrist center projected onto shoulder-elbow plane
            x = c1 * pw[0] + s1 * pw[1]
            y = pw[2] - d1

            # ---- θ3: elbow (2 solutions) ----
            c3 = (x**2 + y**2 - a2**2 - a3**2) / (2 * a2 * a3)
            c3 = max(-1.0, min(1.0, c3))

            for sign3 in [+1, -1]:
                t3 = sign3 * arccos(c3)
                s3 = sin(t3)

                # ---- θ2: shoulder ----
                t2 = arctan2(-y, x) - arctan2(-a3 * s3, a2 + a3 * cos(t3))

                # ---- Orientation: solve θ4, θ5, θ6 from R03^T * R ----
                R03 = self._rot_03(t1, t2, t3)
                R36 = R03.T @ R

                r02, r12, r22 = R36[0, 2], R36[1, 2], R36[2, 2]
                c5 = r22
                r_s5 = sqrt(r02**2 + r12**2)

                for sign5 in [+1, -1]:
                    s5 = sign5 * r_s5
                    t5 = arctan2(s5, c5)

                    if abs(s5) < 1e-6:
                        t4 = 0.0
                        t6 = arctan2(-R36[1, 0], R36[0, 0])
                    else:
                        t4 = arctan2(-r12 / s5, -r02 / s5)
                        t6 = arctan2(-R36[2, 1] / s5, R36[2, 0] / s5)

                    q = _norm_angles(np.array([t1, t2, t3, t4, t5, t6]))
                    solutions.append(q)

        return solutions

    def _rot_03(self, t1: float, t2: float, t3: float) -> np.ndarray:
        """R_0_3 = R_z(t1) * R_x(pi/2) * R_z(t2+t3)"""
        c1, s1 = cos(t1), sin(t1)
        c23, s23 = cos(t2 + t3), sin(t2 + t3)
        return np.array([
            [c1 * c23, -c1 * s23,  s1],
            [s1 * c23, -s1 * s23, -c1],
            [s23,       c23,       0],
        ])

    @staticmethod
    def forward_kinematics(q: np.ndarray) -> np.ndarray:
        d1, a2, a3, d4, d5, d6 = (
            UR5IK._d1, UR5IK._a2, UR5IK._a3, UR5IK._d4, UR5IK._d5, UR5IK._d6)
        t = q
        T01 = _dh(t[0], d1, 0, pi / 2)
        T12 = _dh(t[1], 0, a2, 0)
        T23 = _dh(t[2], 0, a3, 0)
        T34 = _dh(t[3], d4, 0, pi / 2)
        T45 = _dh(t[4], d5, 0, -pi / 2)
        T56 = _dh(t[5], d6, 0, 0)
        return T01 @ T12 @ T23 @ T34 @ T45 @ T56


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + pi) % (2 * pi) - pi


def _norm_angles(q: np.ndarray) -> np.ndarray:
    return (q + pi) % (2 * pi) - pi
