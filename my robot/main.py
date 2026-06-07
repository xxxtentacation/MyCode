"""UR5 Embodied Manipulation Robot — Entry Point.

Usage:
  python main.py              Launch MuJoCo passive viewer (requires display)
  python main.py --demo       Run a simple demo trajectory (headless, saves frames)
  python main.py --teleop     Run interactive keyboard teleop
  python main.py --sensor     Run sensor data stream check
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def _ensure_deps():
    try:
        import mujoco
    except ImportError:
        print("Error: mujoco not installed. Run: pip install mujoco")
        sys.exit(1)
    try:
        import numpy
    except ImportError:
        print("Error: numpy not installed. Run: pip install numpy")
        sys.exit(1)


def run_viewer():
    """Launch MuJoCo passive viewer (requires display)."""
    from simulation.mujoco_env import MujocoEnv
    print("Launching MuJoCo passive viewer...")
    print("Close the viewer window to exit.")
    MujocoEnv.launch_passive_viewer()


def run_demo():
    """Run a simple joint-space demo trajectory headlessly, saving frames."""
    import numpy as np
    from simulation.mujoco_env import MujocoEnv
    from perception.camera import SimCamera

    print("Running demo trajectory (headless)...")
    env = MujocoEnv(render_mode="offscreen")
    cam = SimCamera()

    env.reset()
    home_q = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
    env.set_joint_angles(home_q, 0.08)

    # Settle
    for _ in range(300):
        env.step()

    # Simple joint-space trajectory: oscillate elbow and wrist
    total_steps = 600
    for i in range(total_steps):
        t = i / total_steps
        q = home_q.copy()
        q[2] += 0.3 * np.sin(t * 2 * np.pi * 2)  # oscillate elbow
        q[4] -= 0.2 * np.sin(t * 2 * np.pi * 2)  # oscillate wrist
        env.set_joint_angles(q, 0.08)
        env.step()

        if i % 50 == 0:
            rgb = env.render()
            if rgb is not None:
                import cv2
                frame_path = f"frame_{i:04d}.png"
                cv2.imwrite(frame_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            obs = env.get_obs()
            ee = obs["ee_pose"][:3, 3]
            print(f"  Step {i}: EE pos = [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")

    env.close()
    print("Demo complete. Frames saved as frame_*.png")


def main():
    _ensure_deps()

    parser = argparse.ArgumentParser(description="UR5 Embodied Robot")
    parser.add_argument("--demo", action="store_true", help="Run demo trajectory")
    parser.add_argument("--teleop", action="store_true", help="Run keyboard teleop")
    parser.add_argument("--sensor", action="store_true", help="Run sensor check")
    args = parser.parse_args()

    if args.sensor:
        from scripts.sensor_check import run_sensor_check
        run_sensor_check()
    elif args.teleop:
        from scripts.teleop import run_teleop
        run_teleop()
    elif args.demo:
        run_demo()
    else:
        run_viewer()


if __name__ == "__main__":
    main()
