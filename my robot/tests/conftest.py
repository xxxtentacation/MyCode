"""Global test fixtures for the embodied robot project.

Provides shared fixtures: MuJoCo environment, IK solver, camera, test data.
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def ik_solver():
    """Reusable UR5 IK solver instance."""
    from control.ik_solver import UR5IK
    return UR5IK()


@pytest.fixture(scope="session")
def home_joint_angles():
    """Nominal home pose for UR5."""
    return np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])


@pytest.fixture(scope="function")
def mujoco_env():
    """Create a fresh MuJoCo environment for each test.

    Skips if MuJoCo or the scene file is unavailable (e.g., headless CI).
    """
    mujoco = pytest.importorskip("mujoco")
    from simulation.mujoco_env import MujocoEnv
    env = MujocoEnv()
    env.reset()
    yield env
    env.close()


@pytest.fixture(scope="session")
def scene_path():
    """Path to the UR5 scene XML."""
    return os.path.join(
        os.path.dirname(__file__), "..", "simulation", "assets", "ur5e", "scene.xml"
    )


@pytest.fixture(scope="session")
def sim_camera():
    """Reusable SimCamera instance."""
    from perception.camera import SimCamera
    return SimCamera()


@pytest.fixture(scope="session")
def gripper():
    """Reusable RobotiqGripper instance."""
    from control.gripper import RobotiqGripper
    return RobotiqGripper()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "e2e: end-to-end scenario tests")
