from __future__ import annotations

import argparse
import math
import queue
import time
import types
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import swift
import swift.SwiftRoute as swift_route
from spatialmath import SE3


SCRIPT_DIR = Path(__file__).resolve().parent
KR10_URDF = (
    SCRIPT_DIR
    / "robot_models"
    / "kuka_experimental"
    / "kuka_kr10_support"
    / "urdf"
    / "kr10r1100sixx_pybullet.urdf"
)

END_EFFECTOR_LINK = "flange"
HOME_Q_DEG = np.array([0.0, -90.0, 90.0, 0.0, 0.0, 0.0], dtype=float)
STICK_LINK_NAMES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "flange"]


@dataclass
class IkResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    rotation_error_deg: float


@dataclass
class StickRobot:
    joint_spheres: list[sg.Sphere]
    link_cylinders: list[tuple[sg.Cylinder, int, int]]


def rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    delta = actual.T @ target
    value = (np.trace(delta) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def joint_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    delta = (q_a - q_b + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.linalg.norm(delta))


def frame_from_z_axis(start: np.ndarray, end: np.ndarray) -> SE3:
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


def load_robot() -> rtb.ERobot:
    if not KR10_URDF.exists():
        raise FileNotFoundError(f"URDF not found: {KR10_URDF}")

    robot = rtb.ERobot.URDF(str(KR10_URDF))
    link_names = [link.name for link in robot.links]
    if END_EFFECTOR_LINK not in link_names:
        raise ValueError(f"Link '{END_EFFECTOR_LINK}' not found. Links: {link_names}")
    return robot


def print_robot_summary(robot: rtb.ERobot) -> None:
    print(f"Robot: {robot.name}")
    print(f"DOF: {robot.n}")
    print(f"End-effector link: {END_EFFECTOR_LINK}")
    print("Links:", ", ".join(link.name for link in robot.links))
    print("Joint limits deg:")
    for index, (low, high) in enumerate(np.degrees(robot.qlim).T, start=1):
        print(f"  J{index}: {low:8.2f} to {high:8.2f}")


def fk(robot: rtb.ERobot, q: np.ndarray) -> SE3:
    return robot.fkine(q, end=END_EFFECTOR_LINK)


def stick_points(robot: rtb.ERobot, q: np.ndarray) -> list[np.ndarray]:
    return [robot.fkine(q, end=link_name).t for link_name in STICK_LINK_NAMES]


def solve_ik(robot: rtb.ERobot, target: SE3, seed: np.ndarray) -> IkResult:
    seed_candidates = [
        seed,
        np.radians(HOME_Q_DEG),
        np.zeros(robot.n),
        np.radians([0.0, -80.0, 80.0, 0.0, 0.0, 0.0]),
        np.radians([0.0, -105.0, 105.0, 0.0, 0.0, 0.0]),
    ]
    results: list[IkResult] = []

    for candidate in seed_candidates:
        solution = robot.ikine_LM(
            target,
            end=END_EFFECTOR_LINK,
            q0=candidate,
            ilimit=200,
            slimit=50,
            tol=1e-8,
            joint_limits=True,
            method="sugihara",
        )
        q = np.asarray(solution.q, dtype=float)
        actual = fk(robot, q)
        pos_error = float(np.linalg.norm(actual.t - target.t))
        rot_error = rotation_error_deg(actual.R, target.R)
        success = bool(pos_error < 1e-3 and rot_error < 0.1)
        results.append(
            IkResult(
                q=q,
                success=success,
                position_error_m=pos_error,
                rotation_error_deg=rot_error,
            )
        )

    good_results = [result for result in results if result.success]
    if good_results:
        return min(good_results, key=lambda result: joint_distance(result.q, seed))

    return min(
        results,
        key=lambda result: result.position_error_m * 1000.0 + result.rotation_error_deg,
    )


def make_straight_line_targets() -> list[SE3]:
    z = 0.995
    y = 0.0
    return [SE3(float(x), y, z) for x in np.linspace(0.50, 0.75, 9)]


def solve_target_path(robot: rtb.ERobot, targets: list[SE3], q_home: np.ndarray) -> list[IkResult]:
    results: list[IkResult] = []
    seed = q_home.copy()
    for target in targets:
        result = solve_ik(robot, target, seed)
        results.append(result)
        seed = result.q.copy()
    return results


def print_path_report(targets: list[SE3], results: list[IkResult]) -> None:
    print("\nIK path report:")
    print("idx | target xyz m                  | success | pos err mm | rot err deg | q deg")
    print("-" * 102)
    for index, (target, result) in enumerate(zip(targets, results), start=1):
        xyz = target.t
        q_deg = np.degrees(result.q)
        print(
            f"{index:3d} | "
            f"({xyz[0]: .3f}, {xyz[1]: .3f}, {xyz[2]: .3f}) | "
            f"{str(result.success):7s} | "
            f"{result.position_error_m * 1000.0:10.4f} | "
            f"{result.rotation_error_deg:11.5f} | "
            f"{np.array2string(q_deg, precision=2, suppress_small=True)}"
        )


def joint_trajectory(q_points: list[np.ndarray], samples_per_segment: int) -> list[np.ndarray]:
    if len(q_points) < 2:
        return q_points

    trajectory: list[np.ndarray] = []
    for start, end in zip(q_points[:-1], q_points[1:]):
        segment = rtb.jtraj(start, end, samples_per_segment).q
        if trajectory:
            segment = segment[1:]
        trajectory.extend(np.asarray(q, dtype=float) for q in segment)
    return trajectory


def add_target_markers(env: swift.Swift, targets: list[SE3]) -> None:
    env.add(sg.Axes(0.08, pose=SE3()), readonly=True)
    for index, target in enumerate(targets):
        color = [0.1, 0.7, 1.0, 1.0] if index not in (0, len(targets) - 1) else [1.0, 0.2, 0.1, 1.0]
        env.add(sg.Sphere(0.012, pose=target, color=color), readonly=True)
        if index in (0, len(targets) // 2, len(targets) - 1):
            env.add(sg.Axes(0.06, pose=target), readonly=True)


def add_stick_robot(env: swift.Swift, robot: rtb.ERobot, q: np.ndarray) -> StickRobot:
    points = stick_points(robot, q)
    joint_spheres: list[sg.Sphere] = []
    link_cylinders: list[tuple[sg.Cylinder, int, int]] = []

    for index, point in enumerate(points):
        radius = 0.025 if index in (0, len(points) - 1) else 0.018
        color = [1.0, 0.45, 0.0, 1.0] if index else [0.05, 0.05, 0.05, 1.0]
        sphere = sg.Sphere(radius, pose=SE3(point), color=color)
        env.add(sphere, readonly=True)
        joint_spheres.append(sphere)

    for start_index, end_index in zip(range(len(points) - 1), range(1, len(points))):
        start = points[start_index]
        end = points[end_index]
        length = float(np.linalg.norm(end - start))
        if length < 1e-4:
            continue

        cylinder = sg.Cylinder(0.012, length, pose=frame_from_z_axis(start, end), color=[1.0, 0.55, 0.0, 1.0])
        env.add(cylinder, readonly=True)
        link_cylinders.append((cylinder, start_index, end_index))

    return StickRobot(joint_spheres=joint_spheres, link_cylinders=link_cylinders)


def update_stick_robot(stick_robot: StickRobot, robot: rtb.ERobot, q: np.ndarray) -> None:
    points = stick_points(robot, q)
    for sphere, point in zip(stick_robot.joint_spheres, points):
        sphere.T = SE3(point).A
    for cylinder, start_index, end_index in stick_robot.link_cylinders:
        cylinder.T = frame_from_z_axis(points[start_index], points[end_index]).A


def drain_swift_handshakes(env: swift.Swift) -> None:
    drained = 0
    while True:
        try:
            message = env.inq.get_nowait()
        except queue.Empty:
            break

        if message == "Connected":
            drained += 1
            continue

        env.inq.put(message)
        break

    if drained:
        print(f"Drained {drained} extra Swift browser handshake message(s).")


def install_swift_response_filter(env: swift.Swift) -> None:
    def filtered_send_socket(self, code, data=None, expected=True):
        self.outq.put([expected, [code, data]])
        if not expected:
            return "0"

        while True:
            response = self.inq.get()
            if response == "Connected":
                print("Ignored extra Swift browser handshake message.")
                continue
            return response

    env._send_socket = types.MethodType(filtered_send_socket, env)


def animate_swift(
    robot: rtb.ERobot,
    targets: list[SE3],
    q_points: list[np.ndarray],
    *,
    headless: bool,
    hold: bool,
    fps: int,
    browser: str | None,
    show_robot: bool,
    show_markers: bool,
    manual_open: bool,
    urdf_mesh: bool,
) -> None:
    original_open_new_tab = swift_route.wb.open_new_tab
    original_get = swift_route.wb.get

    def print_and_open(url: str) -> bool:
        print(f"\nSwift URL: {url}")
        print("If the browser page crashes, copy this exact URL into Chrome or Edge.")
        if manual_open:
            return True
        return original_open_new_tab(url)

    class PrintingBrowser:
        def __init__(self, controller):
            self.controller = controller

        def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
            print(f"\nSwift URL: {url}")
            print("If the browser page crashes, copy this exact URL into Chrome or Edge.")
            if manual_open:
                return True
            return self.controller.open(url, new=new, autoraise=autoraise)

        def open_new_tab(self, url: str) -> bool:
            print(f"\nSwift URL: {url}")
            print("If the browser page crashes, copy this exact URL into Chrome or Edge.")
            if manual_open:
                return True
            if hasattr(self.controller, "open_new_tab"):
                return self.controller.open_new_tab(url)
            return self.controller.open(url, new=2)

    def print_and_get(name: str | None = None):
        controller = original_get(name) if name else webbrowser.get()
        return PrintingBrowser(controller)

    if not headless and (browser is None or manual_open):
        swift_route.wb.open_new_tab = print_and_open
    if not headless and browser is not None:
        swift_route.wb.get = print_and_get

    env = swift.Swift()
    try:
        env.launch(realtime=True, headless=headless, rate=fps, browser=browser)
    finally:
        swift_route.wb.open_new_tab = original_open_new_tab
        swift_route.wb.get = original_get

    drain_swift_handshakes(env)
    install_swift_response_filter(env)

    if manual_open and not headless:
        input("Open the Swift URL in Chrome or Edge, wait for the page to load, then press Enter here...")

    if not show_robot and not show_markers:
        print("Empty Swift page test is running. Press Ctrl+C to stop." if hold and not headless else "Empty Swift page test complete.")
        if hold and not headless:
            env.hold()
        return

    stick_robot = None
    if show_robot and urdf_mesh:
        env.add(robot, robot_alpha=1.0, collision_alpha=0.0, readonly=True)
    elif show_robot:
        stick_robot = add_stick_robot(env, robot, q_points[0])
    if show_markers:
        add_target_markers(env, targets)
    env.set_camera_pose([1.8, -1.4, 1.2], [0.62, 0.0, 0.75])

    q_path = joint_trajectory(q_points, samples_per_segment=30)
    for q in q_path:
        robot.q = q
        if stick_robot is not None:
            update_stick_robot(stick_robot, robot, q)
        env.step(1.0 / fps)

    if hold and not headless:
        print("\nSwift is still open. Press Ctrl+C in this terminal to close it.")
        env.hold()

    time.sleep(0.5)
    if not headless:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small Robotics Toolbox + Swift sandbox for KR10 FK/IK."
    )
    parser.add_argument("--no-swift", action="store_true", help="Run FK/IK checks only.")
    parser.add_argument("--headless", action="store_true", help="Run Swift without opening the browser.")
    parser.add_argument("--no-hold", action="store_true", help="Close Swift after one animation pass.")
    parser.add_argument("--browser", default=None, help="Browser name for Swift, for example chrome, firefox, or windows-default.")
    parser.add_argument("--no-robot", action="store_true", help="Launch Swift without adding the robot.")
    parser.add_argument("--no-markers", action="store_true", help="Launch Swift without target frame markers.")
    parser.add_argument("--manual-open", action="store_true", help="Print the Swift URL but do not open a browser automatically.")
    parser.add_argument("--urdf-mesh", action="store_true", help="Use the URDF STL mesh in Swift instead of the lightweight stick robot.")
    parser.add_argument("--fps", type=int, default=60, help="Swift animation rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = load_robot()
    q_home = np.radians(HOME_Q_DEG)
    robot.q = q_home

    print_robot_summary(robot)
    home_fk = fk(robot, q_home)
    print("\nHome FK at flange:")
    print(home_fk)

    targets = make_straight_line_targets()
    results = solve_target_path(robot, targets, q_home)
    print_path_report(targets, results)

    failures = [index for index, result in enumerate(results, start=1) if not result.success]
    if failures:
        print(f"\nIK failures at target indexes: {failures}")
    else:
        print("\nAll IK targets solved inside tolerance.")

    if not args.no_swift:
        q_points = [q_home] + [result.q for result in results]
        animate_swift(
            robot,
            targets,
            q_points,
            headless=args.headless,
            hold=not args.no_hold,
            fps=args.fps,
            browser=args.browser,
            show_robot=not args.no_robot,
            show_markers=not args.no_markers,
            manual_open=args.manual_open,
            urdf_mesh=args.urdf_mesh,
        )


if __name__ == "__main__":
    main()
