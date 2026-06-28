import heapq
import itertools
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pygame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from algorithms.Mazegeneration import MazeGenerator


class MazeWorld:
    def __init__(self):
        self.maze_generator = MazeGenerator()
        self.grid = self.maze_generator.generate_maze()
        self.rows = self.maze_generator.MAX_SIZE
        self.cols = self.maze_generator.MAX_SIZE
        self.extra_path_probability = 0.12
        self.add_extra_paths()
        self.tile_size = 18
        self.width = self.cols * self.tile_size
        self.height = self.rows * self.tile_size
        self.panel_height = 82

        self.wall_color = (0, 0, 0)
        self.path_color = (245, 245, 245)
        self.start_color = (0, 90, 255)
        self.goal_color = (0, 210, 60)
        self.lidar_color = (255, 80, 80)
        self.track_color = (0, 120, 255)

        self.start_cell = (self.maze_generator.START_ROW, self.maze_generator.START_COL)
        self.goal_cell = self.find_goal_cell()
        self.start_position = self.cell_center(self.start_cell)
        self.goal_position = self.cell_center(self.goal_cell)
        self._dijkstra_counter = itertools.count()

    def add_extra_paths(self):
        for row in range(1, self.rows - 1):
            for col in range(1, self.cols - 1):
                if self.grid[row][col] == 0:
                    continue
                if random.random() > self.extra_path_probability:
                    continue

                horizontal_path = self.grid[row][col - 1] == 0 and self.grid[row][col + 1] == 0
                vertical_path = self.grid[row - 1][col] == 0 and self.grid[row + 1][col] == 0
                if horizontal_path or vertical_path:
                    self.grid[row][col] = 0

    def find_goal_cell(self):
        for row in range(self.rows - 2, 0, -1):
            for col in range(self.cols - 2, 0, -1):
                if self.grid[row][col] == 0:
                    return row, col
        return self.start_cell

    def cell_center(self, cell):
        row, col = cell
        return (
            col * self.tile_size + self.tile_size // 2,
            row * self.tile_size + self.tile_size // 2,
        )

    def pixel_to_cell(self, x, y):
        return int(y // self.tile_size), int(x // self.tile_size)

    def is_wall_pixel(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        row, col = self.pixel_to_cell(x, y)
        return self.grid[row][col] == 1

    def has_clear_line(self, start, end, step=2):
        sx, sy = start
        ex, ey = end
        distance = max(1, int(math.hypot(ex - sx, ey - sy)))
        for length in range(0, distance + 1, step):
            u = length / distance
            x = sx + (ex - sx) * u
            y = sy + (ey - sy) * u
            if self.is_wall_pixel(x, y):
                return False
        return True

    def open_neighbors(self, cell):
        row, col = cell
        for next_cell in ((row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)):
            next_row, next_col = next_cell
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                if self.grid[next_row][next_col] == 0:
                    yield next_cell

    def dijkstra_shortest_path(self):
        queue = [(0, next(self._dijkstra_counter), self.start_cell)]
        distances = {self.start_cell: 0}
        parents = {}

        while queue:
            distance, _, cell = heapq.heappop(queue)
            if cell == self.goal_cell:
                return self.reconstruct_path(parents), distance
            if distance > distances.get(cell, float("inf")):
                continue

            for next_cell in self.open_neighbors(cell):
                next_distance = distance + 1
                if next_distance < distances.get(next_cell, float("inf")):
                    distances[next_cell] = next_distance
                    parents[next_cell] = cell
                    heapq.heappush(
                        queue,
                        (next_distance, next(self._dijkstra_counter), next_cell),
                    )

        return [], float("inf")

    def reconstruct_path(self, parents):
        path = [self.goal_cell]
        cell = self.goal_cell
        while cell != self.start_cell:
            cell = parents[cell]
            path.append(cell)
        path.reverse()
        return path

    def cells_to_points(self, cells):
        return [self.cell_center(cell) for cell in cells]

    def draw(self, screen, goal_seen):
        screen.fill((225, 225, 225))
        for row in range(self.rows):
            for col in range(self.cols):
                color = self.wall_color if self.grid[row][col] == 1 else self.path_color
                pygame.draw.rect(
                    screen,
                    color,
                    (
                        col * self.tile_size,
                        row * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    ),
                )

        pygame.draw.circle(screen, self.start_color, self.start_position, 5)
        goal_color = self.goal_color if goal_seen else (0, 130, 45)
        pygame.draw.circle(screen, goal_color, self.goal_position, 7)


class MazeLidar:
    def __init__(self, world, ray_count=72, max_range=110):
        self.world = world
        self.ray_count = ray_count
        self.max_range = max_range
        self.relative_angles = [
            -math.pi + i * (2 * math.pi / ray_count) for i in range(ray_count)
        ]

    def scan(self, robot):
        rays = []
        for relative_angle in self.relative_angles:
            absolute_angle = robot.theta + relative_angle
            hit_point = (
                robot.x + self.max_range * math.cos(absolute_angle),
                robot.y - self.max_range * math.sin(absolute_angle),
            )
            hit_distance = self.max_range

            for distance in range(1, self.max_range + 1):
                x = robot.x + distance * math.cos(absolute_angle)
                y = robot.y - distance * math.sin(absolute_angle)
                if self.world.is_wall_pixel(x, y):
                    hit_point = (x, y)
                    hit_distance = distance
                    break

            rays.append(
                {
                    "angle": relative_angle,
                    "distance": hit_distance,
                    "point": hit_point,
                }
            )
        return rays

    def draw(self, screen, robot, rays):
        for index, ray in enumerate(rays):
            if index % 4 != 0:
                continue
            pygame.draw.line(
                screen,
                self.world.lidar_color,
                (int(robot.x), int(robot.y)),
                (int(ray["point"][0]), int(ray["point"][1])),
                1,
            )


class MazeRobot:
    HEADINGS = [0.0, math.pi / 2, math.pi, -math.pi / 2]

    def __init__(self, start_position, tile_size):
        self.x, self.y = start_position
        self.theta = 0.0
        self.tile_size = tile_size
        self.radius = max(5, tile_size // 3)
        self.speed = 200
        self.turn_speed = 5.5
        self.target = (self.x, self.y)
        self.mode = "explore"
        self.track = [(int(self.x), int(self.y))]
        self.replay_path = []
        self.replay_index = 0
        start_cell = (int(start_position[1] // tile_size), int(start_position[0] // tile_size))
        self.visited_cells = {start_cell}
        self.route_stack = [start_cell]

    def nearest_heading(self):
        return min(
            self.HEADINGS,
            key=lambda heading: abs(self.normalize_angle(heading - self.theta)),
        )

    def choose_next_move(self, lidar_data, world, goal_seen):
        if self.is_moving():
            return

        current_cell = world.pixel_to_cell(self.x, self.y)
        self.visited_cells.add(current_cell)
        if not self.route_stack or self.route_stack[-1] != current_cell:
            self.route_stack.append(current_cell)
        current_heading = self.nearest_heading()

        if goal_seen:
            goal_angle = math.atan2(
                -(world.goal_position[1] - self.y),
                world.goal_position[0] - self.x,
            )
            current_heading = min(
                self.HEADINGS,
                key=lambda heading: abs(self.normalize_angle(heading - goal_angle)),
            )
            self.mode = "target visible"
        else:
            open_options = self.open_move_options(lidar_data, world, current_cell, current_heading)
            unexplored_options = [
                option for option in open_options if option["cell"] not in self.visited_cells
            ]

            if unexplored_options:
                chosen = unexplored_options[0]
                current_heading = chosen["heading"]
                self.route_stack.append(chosen["cell"])
                self.mode = "exploring new branch"
            elif len(self.route_stack) > 1:
                self.route_stack.pop()
                backtrack_cell = self.route_stack[-1]
                current_heading = self.heading_to_cell(current_cell, backtrack_cell)
                self.mode = "backtracking"
            else:
                self.mode = "fully explored"

        next_cell = self.cell_in_heading(current_cell, current_heading)
        if world.grid[next_cell[0]][next_cell[1]] == 0:
            self.theta = current_heading
            self.target = world.cell_center(next_cell)

    def open_move_options(self, lidar_data, world, current_cell, current_heading):
        options = []
        for relative_angle in (math.pi / 2, 0.0, -math.pi / 2, math.pi):
            if not self.open_in_direction(lidar_data, relative_angle):
                continue

            heading = self.normalize_angle(current_heading + relative_angle)
            next_cell = self.cell_in_heading(current_cell, heading)
            row, col = next_cell
            if 0 <= row < world.rows and 0 <= col < world.cols and world.grid[row][col] == 0:
                options.append({"heading": heading, "cell": next_cell})
        return options

    def open_in_direction(self, lidar_data, relative_angle):
        nearest = min(
            lidar_data,
            key=lambda ray: abs(self.normalize_angle(ray["angle"] - relative_angle)),
        )
        return nearest["distance"] > self.tile_size * 0.75

    def cell_in_heading(self, cell, heading):
        row, col = cell
        if abs(self.normalize_angle(heading - 0.0)) < 0.1:
            return row, col + 1
        if abs(self.normalize_angle(heading - math.pi / 2)) < 0.1:
            return row - 1, col
        if abs(self.normalize_angle(heading - math.pi)) < 0.1:
            return row, col - 1
        return row + 1, col

    def heading_to_cell(self, current_cell, next_cell):
        row, col = current_cell
        next_row, next_col = next_cell
        if next_col > col:
            return 0.0
        if next_row < row:
            return math.pi / 2
        if next_col < col:
            return math.pi
        return -math.pi / 2

    def move(self, dt):
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        distance = math.hypot(dx, dy)
        if distance <= 1:
            self.x, self.y = self.target
            return

        step = min(distance, self.speed * dt)
        self.x += dx / distance * step
        self.y += dy / distance * step
        point = (int(self.x), int(self.y))
        if math.hypot(point[0] - self.track[-1][0], point[1] - self.track[-1][1]) > 3:
            self.track.append(point)

    def set_replay_path(self, path_points):
        self.replay_path = path_points
        self.replay_index = 1 if len(path_points) > 1 else 0
        self.target = path_points[self.replay_index] if path_points else (self.x, self.y)
        self.mode = "rerun learned path"

    def follow_replay_path(self, dt):
        if not self.replay_path:
            return True

        if self.is_moving():
            self.move(dt)
            return False

        if self.replay_index >= len(self.replay_path) - 1:
            return True

        self.replay_index += 1
        self.target = self.replay_path[self.replay_index]
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        if dx or dy:
            self.theta = math.atan2(-dy, dx)
        self.move(dt)
        return False

    def learned_path_cells(self, world):
        cells = []
        for x, y in self.track:
            cell = world.pixel_to_cell(x, y)
            if not cells or cells[-1] != cell:
                cells.append(cell)

        if not cells or cells[0] != world.start_cell:
            cells.insert(0, world.start_cell)
        if cells[-1] != world.goal_cell:
            cells.append(world.goal_cell)

        loop_erased = []
        cell_indexes = {}
        for cell in cells:
            if cell in cell_indexes:
                loop_start = cell_indexes[cell]
                for removed_cell in loop_erased[loop_start + 1:]:
                    cell_indexes.pop(removed_cell, None)
                loop_erased = loop_erased[:loop_start + 1]
            else:
                cell_indexes[cell] = len(loop_erased)
                loop_erased.append(cell)

        return loop_erased

    def is_moving(self):
        return math.hypot(self.target[0] - self.x, self.target[1] - self.y) > 1

    def at_goal(self, goal_position):
        return math.hypot(goal_position[0] - self.x, goal_position[1] - self.y) < self.radius

    def draw(self, screen, track_color=(0, 120, 255), planned_path=None):
        if planned_path and len(planned_path) > 1:
            pygame.draw.lines(screen, (220, 0, 0), False, planned_path, 4)

        if len(self.track) > 1:
            pygame.draw.lines(screen, track_color, False, self.track, 2)

        pygame.draw.circle(screen, (255, 160, 0), (int(self.x), int(self.y)), self.radius)
        heading_end = (
            self.x + self.radius * 1.7 * math.cos(self.theta),
            self.y - self.radius * 1.7 * math.sin(self.theta),
        )
        pygame.draw.line(
            screen,
            (30, 30, 30),
            (int(self.x), int(self.y)),
            (int(heading_end[0]), int(heading_end[1])),
            3,
        )

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


def path_length(points):
    return sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(points[:-1], points[1:])
    )


def plot_path_comparison(world, learned_points, dijkstra_points):
    plt.ion()
    fig, ax = plt.subplots(num="Autonomous2 Path Comparison")
    ax.set_title("Dijkstra shortest path vs discovered rerun path")
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal", adjustable="box")

    for row in range(world.rows):
        for col in range(world.cols):
            if world.grid[row][col] == 1:
                ax.add_patch(
                    plt.Rectangle(
                        (col * world.tile_size, row * world.tile_size),
                        world.tile_size,
                        world.tile_size,
                        color="black",
                    )
                )

    if dijkstra_points:
        xs = [point[0] for point in dijkstra_points]
        ys = [point[1] for point in dijkstra_points]
        label = f"Dijkstra shortest: {path_length(dijkstra_points):.0f}px"
        ax.plot(xs, ys, color="green", linewidth=3, label=label)

    if learned_points:
        xs = [point[0] for point in learned_points]
        ys = [point[1] for point in learned_points]
        label = f"Discovered rerun: {path_length(learned_points):.0f}px"
        ax.plot(xs, ys, color="red", linewidth=2, linestyle="--", label=label)

    ax.scatter([world.start_position[0]], [world.start_position[1]], color="blue", s=45, label="Start")
    ax.scatter([world.goal_position[0]], [world.goal_position[1]], color="lime", s=55, label="Target")
    ax.legend(loc="upper right")
    fig.canvas.draw()
    plt.show(block=False)


def draw_status(screen, world, robot, goal_seen, found, rerunning):
    font = pygame.font.SysFont("Arial", 18)
    panel_top = world.height
    pygame.draw.rect(screen, (35, 35, 35), (0, panel_top, world.width, world.panel_height))

    status = "RERUN COMPLETE" if found and rerunning else "FOUND TARGET" if found else robot.mode
    sight = "visible" if goal_seen else "not visible"
    prefix = "Autonomous2 | replaying learned path" if rerunning else "Autonomous2 | no path planning"
    text = f"{prefix} | LIDAR target: {sight} | mode: {status}"
    image = font.render(text, True, (255, 255, 255))
    screen.blit(image, (12, panel_top + 12))

    hint = font.render("N: new maze    Esc: quit", True, (200, 200, 200))
    screen.blit(hint, (12, panel_top + 42))

    button_rect = None
    if found and not rerunning:
        button_rect = pygame.Rect(world.width - 150, panel_top + 24, 128, 34)
        pygame.draw.rect(screen, (230, 230, 230), button_rect, border_radius=4)
        pygame.draw.rect(screen, (20, 20, 20), button_rect, 2, border_radius=4)
        button_text = font.render("Rerun", True, (10, 10, 10))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, button_text_rect)

    return button_rect


def run_simulation():
    pygame.init()
    pygame.display.set_caption("Autonomous2 - Maze LIDAR Search")

    world = MazeWorld()
    screen = pygame.display.set_mode((world.width, world.height + world.panel_height))
    clock = pygame.time.Clock()
    robot = MazeRobot(world.start_position, world.tile_size)
    lidar = MazeLidar(world)

    found = False
    rerunning = False
    rerun_button = None
    learned_points = []
    dijkstra_points = []
    running = True

    def reset_world():
        next_world = MazeWorld()
        next_screen = pygame.display.set_mode(
            (next_world.width, next_world.height + next_world.panel_height)
        )
        next_robot = MazeRobot(next_world.start_position, next_world.tile_size)
        next_lidar = MazeLidar(next_world)
        return next_world, next_screen, next_robot, next_lidar

    def start_rerun():
        learned_cells = robot.learned_path_cells(world)
        next_learned_points = world.cells_to_points(learned_cells)
        dijkstra_cells, _ = world.dijkstra_shortest_path()
        next_dijkstra_points = world.cells_to_points(dijkstra_cells)
        plot_path_comparison(world, next_learned_points, next_dijkstra_points)

        replay_robot = MazeRobot(world.start_position, world.tile_size)
        replay_robot.set_replay_path(next_learned_points)
        return replay_robot, next_learned_points, next_dijkstra_points

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if rerun_button and rerun_button.collidepoint(event.pos):
                    robot, learned_points, dijkstra_points = start_rerun()
                    found = False
                    rerunning = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    world, screen, robot, lidar = reset_world()
                    found = False
                    rerunning = False
                    learned_points = []
                    dijkstra_points = []
                elif event.key == pygame.K_r and found and not rerunning:
                    robot, learned_points, dijkstra_points = start_rerun()
                    found = False
                    rerunning = True

        rays = lidar.scan(robot)
        goal_distance = math.hypot(world.goal_position[0] - robot.x, world.goal_position[1] - robot.y)
        goal_seen = (
            goal_distance <= lidar.max_range
            and world.has_clear_line((robot.x, robot.y), world.goal_position)
        )

        if rerunning:
            found = robot.follow_replay_path(dt)
            if found:
                rerunning = False
        elif not found:
            robot.choose_next_move(rays, world, goal_seen)
            robot.move(dt)
            found = robot.at_goal(world.goal_position)

        world.draw(screen, goal_seen)
        if not rerunning:
            lidar.draw(screen, robot, rays)
        track_color = (220, 0, 0) if learned_points else (0, 120, 255)
        replay_path = learned_points if learned_points else None
        robot.draw(screen, track_color=track_color, planned_path=replay_path)
        rerun_button = draw_status(screen, world, robot, goal_seen, found, rerunning)
        pygame.display.flip()
        plt.pause(0.001)

    pygame.quit()


if __name__ == "__main__":
    run_simulation()
