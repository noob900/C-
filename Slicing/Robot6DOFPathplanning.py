from __future__ import annotations
import argparse
import queue
import time
import types
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import swift
from spatialmath import SE3

SCRIPT_DIR = Path(__file__).resolve().parent

# WCS is the fixed world coordinate system. For now this file only verifies that
# the robot and simulation libraries can be loaded relative to this origin.
WCS_XYZ_MM = (0.0, 0.0, 0.0)
WCS_ABC_DEG = (0.0, 0.0, 0.0)
ROBOT_HOME_Q_DEG = (0.0, -90.0, 90.0, 0.0, 0.0, 0.0)

ROBOT_URDF = (
    SCRIPT_DIR
    / "robot_models"
    / "kuka_experimental"
    / "kuka_kr10_support"
    / "urdf"
    / "kr10r1100sixx_pybullet.urdf"
)

def load_robot(urdf_path: Path):
    if not urdf_path.exists():
        raise FileNotFoundError(f"Robot URDF was not found: {urdf_path}")
    return rtb.ERobot.URDF(str(urdf_path))


class LightweightRobot:
    def __init__(self, env, robot, home_q_deg=ROBOT_HOME_Q_DEG):
        self.env = env
        self.robot = robot
        self.home_q_deg = tuple(home_q_deg)
        self.joint_spheres = []
        self.link_cylinders = []
        self.end_frame = None
        self.q = self.home_q()

    def home_q(self):
        if len(self.home_q_deg) == self.robot.n:
            return np.radians(np.asarray(self.home_q_deg, dtype=float))
        return np.zeros(self.robot.n)

    def link_points(self, q):
        points = []
        for link in self.robot.links:
            try:
                point = self.robot.fkine(q, end=link.name).t
            except (ValueError, TypeError):
                continue
            if points and np.linalg.norm(point - points[-1]) < 1e-5:
                continue
            points.append(point)
        return points

    def frame_from_z_axis(self, start, end):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        z_axis = end - start
        length = np.linalg.norm(z_axis)
        if length < 1e-9:
            return SE3(start)

        z_axis /= length
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, z_axis))) > 0.95:
            reference = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(reference, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        midpoint = (start + end) * 0.5
        return SE3.Rt(rotation, midpoint)

    def add_to_scene(self) -> None:
        self.set_joints(self.q)
        points = self.link_points(self.q)
        if not points:
            return

        for index, point in enumerate(points):
            radius = 0.035 if index in (0, len(points) - 1) else 0.022
            color = [0.05, 0.05, 0.05, 1.0] if index == 0 else [1.0, 0.52, 0.0, 1.0]
            sphere = sg.Sphere(radius, pose=SE3(point), color=color)
            self.env.add(sphere, readonly=True)
            self.joint_spheres.append(sphere)

        for start_index, end_index in zip(range(len(points) - 1), range(1, len(points))):
            start = points[start_index]
            end = points[end_index]
            length = float(np.linalg.norm(end - start))
            if length < 1e-4:
                continue
            cylinder = sg.Cylinder(
                radius=0.015,
                length=length,
                pose=self.frame_from_z_axis(start, end),
                color=[1.0, 0.45, 0.0, 1.0],
            )
            self.env.add(cylinder, readonly=True)
            self.link_cylinders.append((cylinder, start_index, end_index))

        self.end_frame = sg.Axes(0.16, pose=SE3(points[-1]))
        self.env.add(self.end_frame, readonly=True)

    def set_joints(self, q) -> None:
        self.q = np.asarray(q, dtype=float)
        self.robot.q = self.q

        points = self.link_points(self.q)
        for sphere, point in zip(self.joint_spheres, points):
            sphere.T = SE3(point).A

        for cylinder, start_index, end_index in self.link_cylinders:
            start = points[start_index]
            end = points[end_index]
            cylinder.T = self.frame_from_z_axis(start, end).A

        if self.end_frame is not None and points:
            self.end_frame.T = SE3(points[-1]).A


class SwiftScene:
    def __init__(self, *, browser: str | None = None, rate: int = 30):
        self.browser = browser
        self.rate = rate
        self.env = swift.Swift()

    def launch(self) -> None:
        self.env.launch(realtime=True, headless=False, rate=self.rate, browser=self.browser)
        self.drain_handshakes()
        self.install_response_filter()

    def build(self, *, show_robot: bool = True) -> None:
        self.launch()
        self.add_wcs()
        self.add_floor()

        if show_robot:
            lightweight_robot = LightweightRobot(self.env, load_robot(ROBOT_URDF))
            lightweight_robot.add_to_scene()

        self.set_default_camera()
        self.step()
        self.hold()

    def add_wcs(self) -> None:
        self.env.add(sg.Axes(0.35, pose=SE3()), readonly=True)

    def add_floor(self) -> None:
        self.env.add(
            sg.Cuboid(
                [1.6, 1.2, 0.015],
                pose=SE3(0.45, 0.0, -0.015),
                color=[0.72, 0.72, 0.72, 1.0],
            ),
            readonly=True,
        )

    def set_default_camera(self) -> None:
        self.env.set_camera_pose([1.8, -1.6, 1.1], [0.45, 0.0, 0.35])

    def step(self, dt: float = 0.05) -> None:
        self.env.step(dt)

    def hold(self) -> None:
        print("\nSwift visual environment is open.")
        print("WCS is the large XYZ axis at 0,0,0. Press Ctrl+C in this terminal to close it.")
        while True:
            time.sleep(1.0)

    def drain_handshakes(self) -> None:
        while True:
            try:
                message = self.env.inq.get_nowait()
            except queue.Empty:
                return
            if message != "Connected":
                self.env.inq.put(message)
                return

    def install_response_filter(self) -> None:
        def filtered_send_socket(env, code, data=None, expected=True):
            env.outq.put([expected, [code, data]])
            if not expected:
                return "0"
            while True:
                response = env.inq.get()
                if response == "Connected":
                    continue
                return response

        self.env._send_socket = types.MethodType(filtered_send_socket, self.env)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 environment setup check for the RTB + Swift robot workflow."
    )
    parser.add_argument(
        "--show-swift",
        action="store_true",
        help="Open the visual Swift environment with WCS axes.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="With --show-swift, show only the WCS environment and no robot.",
    )
    parser.add_argument(
        "--browser",
        default=None,
        help="With --show-swift, choose a browser controller such as chrome or windows-default.",
    )
    args = parser.parse_args()

    print("Robot6DOFPathplanning environment setup")
    print("=" * 42)
    print(f"WCS XYZ mm:  {WCS_XYZ_MM}")
    print(f"WCS ABC deg: {WCS_ABC_DEG}")
    print()

    if args.show_swift:
        scene = SwiftScene(browser=args.browser)
        scene.build(show_robot=not args.no_robot)

    print("Environment setup check complete.")
    print("Next step will be adding the WCS robot scene in Swift, without path planning yet.")


if __name__ == "__main__":
    main()
