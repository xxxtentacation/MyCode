"""Verify sensor data stream from MuJoCo simulation.

Runs the simulation for N seconds, collecting:
- Joint angles and velocities (6 arm joints)
- End-effector pose
- Gripper state
- RGB / Depth images (if renderer available)

Prints statistics at the end.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.mujoco_env import MujocoEnv
from perception.camera import SimCamera


def run_sensor_check(duration: float = 3.0):
    print("=" * 50)
    print("Sensor Data Stream Check")
    print("=" * 50)

    env = MujocoEnv(render_mode="offscreen")
    cam = SimCamera(width=640, height=480, fov=45)

    env.reset()

    # Move to a visible pose
    home_q = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
    env.set_joint_angles(home_q, 0.08)
    for _ in range(500):
        env.step()

    dt = env.dt
    steps = int(duration / dt)
    print(f"Duration: {duration}s, Steps: {steps}, dt: {dt:.4f}s")

    # Data collectors
    timestamps: list[float] = []
    joint_positions: list[np.ndarray] = []
    joint_velocities: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    gripper_widths: list[float] = []
    frame_count = 0
    render_times: list[float] = []

    start = time.time()
    for i in range(steps):
        env.step()
        obs = env.get_obs()

        timestamps.append(obs["timestep"] * dt)
        joint_positions.append(obs["joint_angles"])
        joint_velocities.append(obs["joint_velocities"])
        ee_positions.append(obs["ee_position"])
        gripper_widths.append(obs["gripper_width"])

        # Render every 10 steps
        if i % 10 == 0:
            t0 = time.perf_counter()
            rgb = env.render()
            t1 = time.perf_counter()
            if rgb is not None:
                frame_count += 1
                render_times.append(t1 - t0)
    elapsed = time.time() - start

    jp = np.array(joint_positions)
    jv = np.array(joint_velocities)
    ep = np.array(ee_positions)

    # ---- Print report ----
    print(f"\nElapsed wall time: {elapsed:.2f}s")
    print(f"Real-time factor: {duration / elapsed:.2f}x\n")

    print("--- Joint Angles (rad) ---")
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist1", "wrist2", "wrist3"]
    for i, name in enumerate(names):
        print(f"  {name:>15s}: mean={jp[:, i].mean():+.3f}, "
              f"std={jp[:, i].std():.4f}, "
              f"min={jp[:, i].min():+.3f}, max={jp[:, i].max():+.3f}")

    print("\n--- Joint Velocities (rad/s) ---")
    for i, name in enumerate(names):
        print(f"  {name:>15s}: mean={jv[:, i].mean():+.4f}, "
              f"std={jv[:, i].std():.4f}, "
              f"max_abs={np.abs(jv[:, i]).max():.4f}")

    print("\n--- End-Effector Position (m) ---")
    for i, axis in enumerate(["x", "y", "z"]):
        print(f"  {axis}: mean={ep[:, i].mean():+.4f}, "
              f"std={ep[:, i].std():.4f}, "
              f"range=[{ep[:, i].min():+.4f}, {ep[:, i].max():+.4f}]")

    print(f"\n--- Gripper ---")
    gw = np.array(gripper_widths)
    print(f"  width: mean={gw.mean():.4f}m, std={gw.std():.4f}m")

    print(f"\n--- Rendering ---")
    print(f"  Frames captured: {frame_count}")
    if render_times:
        rt = np.array(render_times)
        print(f"  Render time: mean={rt.mean()*1000:.1f}ms, "
              f"max={rt.max()*1000:.1f}ms, fps={1/rt.mean():.1f}")

    print(f"\n--- Camera Model ---")
    print(f"  Resolution: {cam.width}x{cam.height}")
    print(f"  FOV: {cam.fov} deg")
    print(f"  Intrinsic:\n{cam.intrinsic_matrix}")

    env.close()
    print("\nSensor check complete.")


if __name__ == "__main__":
    run_sensor_check()
