"""Keyboard teleop for UR5 end-effector control.

Uses IK to convert EE displacement commands to joint targets.
Runs inside a MuJoCo active viewer loop.

Controls:
  W/S      EE x +/-      (forward/back)
  A/D      EE y +/-      (left/right)
  Q/E      EE z +/-      (up/down)
  J/L      rotate about z (yaw)
  I/K      rotate about y (pitch)
  U/O      rotate about x (roll)
  G        toggle gripper open/close
  R        reset arm to home pose
  ESC      quit
"""

import sys
import os
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.mujoco_env import MujocoEnv
from control.ik_solver import UR5IK

# ---- Key state ----
_keys_pressed: set[str] = set()

TRANSLATION_STEP = 0.005   # m per step
ROTATION_STEP = 0.05       # rad per step

HOME_Q = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
GRIPPER_WIDTHS = {"open": 0.08, "close": 0.0}


def _on_press(key):
    try:
        _keys_pressed.add(key.char.lower())
    except AttributeError:
        _keys_pressed.add(str(key))


def _on_release(key):
    try:
        _keys_pressed.discard(key.char.lower())
    except AttributeError:
        _keys_pressed.discard(str(key))


def _start_key_listener():
    from pynput.keyboard import Listener
    listener = Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()


def _compute_ee_delta(keys: set[str]) -> np.ndarray:
    """Convert key presses to 6-DoF twist [dx, dy, dz, drx, dry, drz]."""
    dx = dy = dz = 0.0
    drx = dry = drz = 0.0

    if "w" in keys: dx += TRANSLATION_STEP
    if "s" in keys: dx -= TRANSLATION_STEP
    if "a" in keys: dy += TRANSLATION_STEP
    if "d" in keys: dy -= TRANSLATION_STEP
    if "q" in keys: dz -= TRANSLATION_STEP
    if "e" in keys: dz += TRANSLATION_STEP
    if "j" in keys: drz += ROTATION_STEP
    if "l" in keys: drz -= ROTATION_STEP
    if "i" in keys: dry += ROTATION_STEP
    if "k" in keys: dry -= ROTATION_STEP
    if "u" in keys: drx += ROTATION_STEP
    if "o" in keys: drx -= ROTATION_STEP

    return np.array([dx, dy, dz, drx, dry, drz])


def run_teleop():
    """Run interactive teleop with MuJoCo viewer."""
    import mujoco
    from mujoco import viewer

    env = MujocoEnv()
    ik = UR5IK()

    env.reset()
    env.set_joint_angles(HOME_Q, GRIPPER_WIDTHS["open"])
    # Step to apply initial pose
    for _ in range(500):
        env.step()

    q_current = env.get_joint_angles()
    gripper_open = True

    _start_key_listener()

    print("=" * 50)
    print("UR5 Teleop Controls:")
    print("  W/S:   EE x +/-    A/D:   EE y +/-")
    print("  Q/E:   EE z +/-    J/L:   rot z (yaw)")
    print("  I/K:   rot y       U/O:   rot x")
    print("  G:     toggle gripper")
    print("  R:     reset home")
    print("  ESC:   quit")
    print("=" * 50)

    with viewer.launch(env.model, env.data) as v:
        while v.is_running():
            # Handle special keys
            if "g" in _keys_pressed:
                _keys_pressed.discard("g")
                gripper_open = not gripper_open
                width = GRIPPER_WIDTHS["open"] if gripper_open else GRIPPER_WIDTHS["close"]
                env.set_joint_angles(q_current, width)
                print(f"Gripper: {'OPEN' if gripper_open else 'CLOSE'}")

            if "r" in _keys_pressed:
                _keys_pressed.discard("r")
                env.set_joint_angles(HOME_Q, GRIPPER_WIDTHS["open"])
                gripper_open = True
                print("Reset to home")

            if "esc" in _keys_pressed:
                break

            # Compute EE delta from active keys
            delta = _compute_ee_delta(_keys_pressed)
            if np.any(delta != 0):
                T_current = env.get_ee_pose()
                T_target = T_current.copy()

                # Apply translation in base frame
                T_target[:3, 3] += delta[:3]

                # Apply rotation (approximate, in world frame)
                drx, dry, drz = delta[3], delta[4], delta[5]
                R_delta = _euler_to_rot(drx, dry, drz)
                T_target[:3, :3] = T_target[:3, :3] @ R_delta

                q_sol = ik.solve(T_target, q_current)
                if q_sol is not None:
                    q_current = q_sol
                    width = GRIPPER_WIDTHS["open"] if gripper_open else GRIPPER_WIDTHS["close"]
                    env.set_joint_angles(q_current, width)

            env.step()
            v.sync()

    env.close()
    print("Teleop ended.")


def _euler_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ Euler angles to rotation matrix."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


if __name__ == "__main__":
    run_teleop()
