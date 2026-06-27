import random
import pygame
import numpy as np
import math
import heapq
import itertools
from pathlib import Path

import matplotlib.pyplot as plt

linear_velocity_mps=0.01  #linear velocity in meters per second
angular_velocity_mps=0.01  #angular velocity in radians per second

class LIDAR:
    
    def __init__(self,robot_position,robot_theta,number_of_rays, Range, surface):
        self.robot_position=robot_position
        self.robot_theta=robot_theta
        self.number_of_rays=number_of_rays
        self.Range=Range
        self.screen=pygame.display.get_surface()
        self.w,self.h=self.screen.get_size()
        self.angles=np.linspace(-np.pi,np.pi,self.number_of_rays,endpoint=False)
        
    def sense_obstacle(self):
        data=[]
        Rx, Ry=self.robot_position[0], self.robot_position[1]
        for angle in self.angles:
            hit_point = None
            hit_distance = self.Range
            absolute_angle = angle + self.robot_theta
            Rx1,Ry1= Rx + self.Range*np.cos(absolute_angle), Ry - self.Range*np.sin(absolute_angle)
            for i in range(1, self.Range + 1):
                u= i/self.Range
                x=int(Rx + u*(Rx1 - Rx))
                y=int(Ry + u*(Ry1 - Ry)) 
                if not (0 <= x < self.w and 0 <= y < self.h):
                    hit_point = (max(0, min(x, self.w - 1)), max(0, min(y, self.h - 1)))
                    hit_distance = i
                    break

                color= self.screen.get_at((x,y))
                if color[:3]==(0,0,0): #black obstacle
                    hit_point = (x,y)
                    hit_distance = i
                    break

            data.append({
                "angle": angle,
                "distance": hit_distance,
                "point": hit_point,
            })

        return data
        
    def Sensor_rays(self,screen,Position,rotation):
        n=30
        Rx, Ry=Position
        Center_x, Center_y=Position
        X_axis=(Center_x + n*np.cos(rotation-45), Center_y- n*np.sin(rotation-45))
        Y_axis=(Center_x - n*np.sin(rotation-45), Center_y - n*np.cos(rotation-45))
        pygame.draw.line(screen,(255,0,0),(Center_x,Center_y),X_axis,2) 
        pygame.draw.line(screen,(255,0,0),(Center_x,Center_y),Y_axis,2)
        
class Robot:
    def __init__(self, start_position,width):
        self.m2p= 3779.52 #meters to pixels conversion factor
        self.width=width*self.m2p  #robot width in pixels
        self.x=start_position[0]
        self.y=start_position[1]
        
        self.Vr=0.0  #right wheel velocity
        self.Vl=0.0  #left wheel velocity
        self.theta=0.0  
        
        #graphical representation
        image_path = Path(__file__).with_name("Drone.png")
        self.robot_image=pygame.image.load(image_path)
        self.robot_image=pygame.transform.scale(self.robot_image,(int(self.width),int(self.width))) #scale the image to the robot width
        self.rotated=self.robot_image
        self.rect=self.rotated.get_rect(center=(self.x,self.y))

        # store previous pose for collision recovery
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
        self.collided = False
        self.recovery_steps = 0
        self.recovery_phase = None
        self.recovery_turn_direction = 1
        self.track_points = [(int(self.x), int(self.y))]
        self.path_waypoints = []
        self.path_index = 0
            
    def robot_frame(self,screen,Position,rotation):
        n=30
        Center_x, Center_y=Position
        X_axis=(Center_x + n*np.cos(rotation-45), Center_y- n*np.sin(rotation-45))
        Y_axis=(Center_x - n*np.sin(rotation-45), Center_y - n*np.cos(rotation-45))
        pygame.draw.line(screen,(255,0,0),(Center_x,Center_y),X_axis,2)
        pygame.draw.line(screen,(0,255,0),(Center_x,Center_y),Y_axis,2)
        
        
    def draw(self,screen):
        screen.blit(self.rotated,self.rect)
        # draw a red ring when in collision state
        if self.collided:
            radius = max(8, int(self.width/4))
            pygame.draw.circle(screen, (255,0,0), (int(self.x), int(self.y)), radius, 2)

    def record_track_point(self):
        current_point = (int(self.x), int(self.y))
        last_point = self.track_points[-1]
        if math.hypot(current_point[0] - last_point[0], current_point[1] - last_point[1]) >= 2:
            self.track_points.append(current_point)

    def draw_track(self, screen):
        if len(self.track_points) < 2:
            return
        pygame.draw.lines(screen, (0, 120, 255), False, self.track_points, 2)

    def draw_planned_path(self, screen):
        if len(self.path_waypoints) < 2:
            return
        pygame.draw.lines(screen, (0, 180, 0), False, self.path_waypoints, 2)
     
    def _legacy_autonomous_control(self, lidar_data, goal_pos):
        """Smart sensor-based navigation + crash recovery"""
    
        # CRASH RECOVERY
        
        # GOAL REACHED
        dx = goal_pos[0] - self.x
        dy = goal_pos[1] - self.y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 10:
            self.Vl = 0
            self.Vr = 0
            return
    
        # SMART OBSTACLE AVOIDANCE
        front_clear = True
        if lidar_data:
            front_rays = lidar_data[:len(lidar_data)//3]
            for hit in front_rays:
                if math.hypot(hit[0]-self.x-10, hit[1]-self.y-10) < 80:
                    front_clear = False
                    break
    
        if front_clear:
            # NORMAL GOAL SEEKING (your PID controller)
            K_linear = 0.1
            K_angular = 1
        
            goal_angle = math.atan2(-dy, dx)
            angle_diff = (goal_angle - self.theta + np.pi) % (2 * np.pi) - np.pi
        
            linear_velocity = min(K_linear * goal_dist, 30)
            angular_velocity = K_angular * angle_diff
        
            self.Vl = linear_velocity - (angular_velocity * self.width / 2.0)
            self.Vr = linear_velocity + (angular_velocity * self.width / 2.0)
        else:
            # SMART TURNING BASED ON SENSOR DATA
            if lidar_data:
                n = len(lidar_data)
                left_rays = lidar_data[n//3:2*n//3]
                right_rays = lidar_data[2*n//3:]
            
                left_dist = 999 if not left_rays else min(math.hypot(hit[0]-self.x, hit[1]-self.y) for hit in left_rays)
                right_dist = 999 if not right_rays else min(math.hypot(hit[0]-self.x, hit[1]-self.y) for hit in right_rays)

                if left_dist < right_dist:
                    # LEFT more blocked → TURN RIGHT
                    self.Vl = 0
                    self.Vr = 15
                                            
                if right_dist < left_dist:
                    # RIGHT more blocked → TURN LEFT
                    self.Vl = 15
                    self.Vr = 0
                
            else:
                self.Vl = 5
                self.Vr = 5     
        
    def autonomous_control(self, lidar_data, goal_pos):
        """Navigate to the goal, avoid sensed obstacles, and recover after contact."""

        if self.recovery_phase == "reverse":
            self.Vl = -18
            self.Vr = -18
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_phase = "turn"
                self.recovery_steps = 15
            return False

        if self.recovery_phase == "turn":
            turn_speed = 5 * self.recovery_turn_direction
            self.Vl = -turn_speed
            self.Vr = turn_speed
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_phase = None
                self.Vl = 0
                self.Vr = 0
            return False

        dx = goal_pos[0] - self.x
        dy = goal_pos[1] - self.y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 10:
            self.Vl = 0
            self.Vr = 0
            return True

        front_limit = np.pi / 12
        side_limit = np.pi / 3
        safe_distance = max(32, self.width * 1.2)
        front_rays = [ray for ray in lidar_data if abs(ray["angle"]) <= front_limit]
        left_rays = [ray for ray in lidar_data if front_limit < ray["angle"] <= side_limit]
        right_rays = [ray for ray in lidar_data if -side_limit <= ray["angle"] < -front_limit]

        front_distance = min((ray["distance"] for ray in front_rays), default=999)
        left_distance = min((ray["distance"] for ray in left_rays), default=999)
        right_distance = min((ray["distance"] for ray in right_rays), default=999)

        if front_distance <= safe_distance:
            if left_distance >= right_distance:
                self.Vl = 2
                self.Vr = 18
            else:
                self.Vl = 18
                self.Vr = 2
            return False

        K_linear = 0.1
        K_angular = 1
        goal_angle = math.atan2(-dy, dx)
        angle_diff = (goal_angle - self.theta + np.pi) % (2 * np.pi) - np.pi

        linear_velocity = min(K_linear * goal_dist, 30)
        angular_velocity = K_angular * angle_diff

        self.Vl = linear_velocity - (angular_velocity * self.width / 2.0)
        self.Vr = linear_velocity + (angular_velocity * self.width / 2.0)
        return False

    def set_planned_path(self, path, final_goal):
        if not path:
            self.path_waypoints = [final_goal]
        else:
            self.path_waypoints = [(int(x), int(y)) for x, y in path]
            self.path_waypoints.append((int(final_goal[0]), int(final_goal[1])))
        self.path_index = 1 if len(self.path_waypoints) > 1 else 0

    def follow_planned_path(self):
        if self.recovery_phase == "reverse":
            self.Vl = -18
            self.Vr = -18
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_phase = "turn"
                self.recovery_steps = 15
            return False

        if self.recovery_phase == "turn":
            turn_speed = 5 * self.recovery_turn_direction
            self.Vl = -turn_speed
            self.Vr = turn_speed
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_phase = None
                self.Vl = 0
                self.Vr = 0
            return False

        if not self.path_waypoints or self.path_index >= len(self.path_waypoints):
            self.Vl = 0
            self.Vr = 0
            return True

        target = self.path_waypoints[self.path_index]
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_dist = math.hypot(dx, dy)

        while target_dist < 10 and self.path_index < len(self.path_waypoints) - 1:
            self.path_index += 1
            target = self.path_waypoints[self.path_index]
            dx = target[0] - self.x
            dy = target[1] - self.y
            target_dist = math.hypot(dx, dy)

        if target_dist < 10 and self.path_index == len(self.path_waypoints) - 1:
            self.Vl = 0
            self.Vr = 0
            return True

        target_angle = math.atan2(-dy, dx)
        angle_diff = (target_angle - self.theta + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_diff) > 0.45:
            turn_speed = 9 if angle_diff > 0 else -9
            self.Vl = -turn_speed
            self.Vr = turn_speed
            return False

        linear_velocity = min(22, max(8, target_dist * 0.8))
        angular_velocity = angle_diff * 1.8
        self.Vl = linear_velocity - (angular_velocity * self.width / 2.0)
        self.Vr = linear_velocity + (angular_velocity * self.width / 2.0)
        return False

    def move(self, dt, event=None):
        self.dt=dt
        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.Vl += 10
                    self.Vr += 10
                elif event.key == pygame.K_LEFT:
                    self.Vl -= 10
                    self.Vr -= 10
                    
                elif event.key == pygame.K_UP:
                    # Spin right motor faster and left motor slower to pivot left
                    self.Vr += 1
                    self.Vl -= 0.1
                    
                elif event.key == pygame.K_DOWN:
                    # Spin left motor faster and right motor slower to pivot right
                    self.Vl += 1
                    self.Vr -= 0.1
                elif event.key == pygame.K_KP_ENTER: 
                    # Cut power to both motors
                    self.Vl, self.Vr = 0, 0

        # DIFFERENTIAL DRIVE FORMULAS                 
        # Save previous pose in case of collision
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta

        #Update position using basic integration
        self.theta += (self.Vr - self.Vl) / self.width * self.dt
        self.x +=((self.Vl + self.Vr) / 2.0) * np.cos(self.theta) * (self.dt)
        self.y -=((self.Vl + self.Vr)/ 2.0) * np.sin(self.theta) * (self.dt)
        
        self.rotated = pygame.transform.rotozoom(self.robot_image, np.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(int(self.x), int(self.y)))
        
    def collision(self,obstacle):
        self.obstacle = obstacle
        if self.recovery_phase is not None:
            return

        self.collided = False  # reset collision state
        for i in obstacle:
            rect = pygame.Rect(i)
            if self.rect.colliderect(rect):
                obstacle_center = rect.center
                obstacle_angle = math.atan2(-(obstacle_center[1] - self.y), obstacle_center[0] - self.x)
                relative_angle = (obstacle_angle - self.theta + np.pi) % (2 * np.pi) - np.pi
                self.recovery_turn_direction = -1 if relative_angle > 0 else 1
                self.recovery_phase = "reverse"
                self.recovery_steps = 24

                # revert to previous pose and stop both wheels
                self.x = self.prev_x
                self.y = self.prev_y
                self.theta = self.prev_theta
                self.collided = True
                # update rotated image and rect to reverted pose
                self.rotated = pygame.transform.rotozoom(self.robot_image, np.degrees(self.theta), 1)
                self.rect = self.rotated.get_rect(center=(int(self.x), int(self.y)))
                break
            
                     
class Map:
    
    def __init__(self):
        self.Map_height= 500
        self.Map_width= 500
        
        self.white= (255,255,255)
        self.black= (0,0,0)
        pygame.font.init()
        self.Boundary_clearance= 50
        self.Rectangle_limits=(10,20) #Min and max limits for rectangle width and height
        self.map=pygame.display.set_mode((self.Map_width,self.Map_height))  
        
        self.font = pygame.font.SysFont('Arial', 20)
        self.text=self.font.render('Default', True, self.white, self.black)
        self.textbox=self.text.get_rect()
        self.textbox.center=(300,400) 
        self.point_cloud=[]  # Store all LIDAR hits here
     
     
    def goal_position_circle(self,screen,goal_position):
        pygame.draw.circle(screen,(0,255,0),goal_position,5)

    def draw_checkpoints(self, screen, checkpoints, urgent_index):
        font = pygame.font.SysFont('Arial', 16)
        for index, checkpoint in enumerate(checkpoints):
            color = (255, 0, 255) if index == urgent_index else (255, 165, 0)
            pygame.draw.circle(screen, color, checkpoint, 6)
            label = font.render(str(index + 1), True, self.black)
            screen.blit(label, (checkpoint[0] + 8, checkpoint[1] - 8))
            
    def add_lidar_hits(self, hits):
        for hit in hits:
            point = hit["point"]
            if point is not None and point not in self.point_cloud:  # Avoid duplicates
                self.point_cloud.append(point)     
                
    def draw_point_cloud(self, screen):
            for point in self.point_cloud[-1000:]:  # Last 1000 points
                color = (100, 100)
                pygame.draw.circle(screen, (255, 0, 0), (int(point[0]), int(point[1])), 1) 
        
    def Legend(self,linear_velocity,angular_velocity, theta):
        txt = f"V: {linear_velocity:.2f} W: {angular_velocity:.2f} theta: {theta:.2f}"
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textbox)        
    
    def Random_origin_point(self): #Generates a random origin point for rectangle
        x=int(random.uniform(0,self.Map_width-self.Boundary_clearance))
        y=int(random.uniform(0,self.Map_height-self.Boundary_clearance))
        return (x,y)
    
    def Generate_rectangle(self): #Generates a random rectangle within the map boundaries
        origin=self.Random_origin_point()
        width=int(random.uniform(self.Rectangle_limits[0],self.Rectangle_limits[1]))
        height=int(random.uniform(self.Rectangle_limits[0],self.Rectangle_limits[1]))
        return (origin[0],origin[1],width,height)

    def Generate_obstacles(self,number_of_obstacles): #Generates a list of random rectangles as obstacles
        obstacles=[]
        for i in range(number_of_obstacles):
            rectangle=self.Generate_rectangle()
            obstacles.append(rectangle)
        return obstacles

    def Generate_goal_position(self, obstacles, robot_position, existing_positions=None, margin=25):
        return self.Generate_random_position(
            obstacles,
            robot_position,
            existing_positions=existing_positions,
            margin=margin,
            min_distance=120,
        )

    def Generate_random_position(self, obstacles, robot_position, existing_positions=None, margin=25, min_distance=70):
        existing_positions = existing_positions or []
        while True:
            x = random.randint(margin, self.Map_width - margin)
            y = random.randint(margin, self.Map_height - margin)
            point_rect = pygame.Rect(x - 6, y - 6, 12, 12)
            too_close_to_robot = math.hypot(x - robot_position[0], y - robot_position[1]) < min_distance
            too_close_to_existing = any(math.hypot(x - px, y - py) < min_distance for px, py in existing_positions)
            inside_obstacle = any(point_rect.colliderect(pygame.Rect(obstacle)) for obstacle in obstacles)
            if not too_close_to_robot and not too_close_to_existing and not inside_obstacle:
                return (x, y)

    def Generate_checkpoints(self, obstacles, robot_position, count=3):
        checkpoints = []
        for _ in range(count):
            checkpoint = self.Generate_random_position(
                obstacles,
                robot_position,
                existing_positions=checkpoints,
                min_distance=80,
            )
            checkpoints.append(checkpoint)
        return checkpoints


class DijkstraPathPlanner:
    def __init__(self, map_width, map_height, obstacles, grid_step=10, clearance=16):
        self.map_width = map_width
        self.map_height = map_height
        self.grid_step = grid_step
        self.blocked_rects = [pygame.Rect(obstacle).inflate(clearance * 2, clearance * 2) for obstacle in obstacles]
        self.nodes = self._build_free_nodes()
        self.counter = itertools.count()

    def _build_free_nodes(self):
        nodes = set()
        for x in range(0, self.map_width + 1, self.grid_step):
            for y in range(0, self.map_height + 1, self.grid_step):
                node = (x, y)
                if self._is_free(node):
                    nodes.add(node)
        return nodes

    def _is_free(self, point):
        x, y = point
        if x < 0 or x > self.map_width or y < 0 or y > self.map_height:
            return False
        return not any(rect.collidepoint(point) for rect in self.blocked_rects)

    def nearest_free_node(self, point):
        px, py = point
        return min(self.nodes, key=lambda node: math.hypot(node[0] - px, node[1] - py))

    def neighbors(self, node, banned_nodes=None, banned_edges=None):
        banned_nodes = banned_nodes or set()
        banned_edges = banned_edges or set()
        x, y = node
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        for dx, dy in directions:
            next_node = (x + dx * self.grid_step, y + dy * self.grid_step)
            if next_node not in self.nodes or next_node in banned_nodes:
                continue
            if (node, next_node) in banned_edges:
                continue
            yield next_node, math.hypot(next_node[0] - x, next_node[1] - y)

    def dijkstra(self, start, goal, banned_nodes=None, banned_edges=None):
        banned_nodes = banned_nodes or set()
        banned_edges = banned_edges or set()
        queue = [(0.0, next(self.counter), start)]
        distances = {start: 0.0}
        parents = {}

        while queue:
            distance, _, node = heapq.heappop(queue)
            if node == goal:
                return self._reconstruct_path(parents, start, goal), distance
            if distance > distances.get(node, float("inf")):
                continue

            for next_node, edge_cost in self.neighbors(node, banned_nodes, banned_edges):
                next_distance = distance + edge_cost
                if next_distance < distances.get(next_node, float("inf")):
                    distances[next_node] = next_distance
                    parents[next_node] = node
                    heapq.heappush(queue, (next_distance, next(self.counter), next_node))

        return None, float("inf")

    def _reconstruct_path(self, parents, start, goal):
        path = [goal]
        node = goal
        while node != start:
            node = parents[node]
            path.append(node)
        path.reverse()
        return path

    def top_k_paths(self, start_point, goal_point, k=5):
        start = self.nearest_free_node(start_point)
        goal = self.nearest_free_node(goal_point)
        first_path, first_cost = self.dijkstra(start, goal)
        if first_path is None:
            return []

        paths = [(first_path, first_cost)]
        candidates = []
        seen_paths = {tuple(first_path)}

        for path_index in range(1, k):
            previous_path = paths[path_index - 1][0]
            for spur_index in range(len(previous_path) - 1):
                spur_node = previous_path[spur_index]
                root_path = previous_path[:spur_index + 1]
                banned_edges = set()

                for path, _ in paths:
                    if len(path) > spur_index and path[:spur_index + 1] == root_path:
                        banned_edges.add((path[spur_index], path[spur_index + 1]))

                banned_nodes = set(root_path[:-1])
                spur_path, _ = self.dijkstra(spur_node, goal, banned_nodes, banned_edges)
                if spur_path is None:
                    continue

                total_path = root_path[:-1] + spur_path
                path_key = tuple(total_path)
                if path_key in seen_paths:
                    continue

                seen_paths.add(path_key)
                total_cost = self.path_length(total_path)
                heapq.heappush(candidates, (total_cost, next(self.counter), total_path))

            if not candidates:
                break

            cost, _, next_path = heapq.heappop(candidates)
            paths.append((next_path, cost))

        return paths

    @staticmethod
    def path_length(path):
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(path[:-1], path[1:])
        )


def plot_top_paths(map_width, map_height, obstacles, start_position, goal_position, paths):
    plt.ion()
    fig, ax = plt.subplots(num="Dijkstra Top 5 Shortest Paths")
    ax.set_title("Dijkstra Top 5 Shortest Paths")
    ax.set_xlim(0, map_width)
    ax.set_ylim(map_height, 0)
    ax.set_aspect("equal", adjustable="box")

    for obstacle in obstacles:
        x, y, width, height = obstacle
        ax.add_patch(plt.Rectangle((x, y), width, height, color="black"))

    for index, (path, cost) in enumerate(paths):
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        color = "green" if index == 0 else "red"
        width = 3 if index == 0 else 1.5
        label = f"#{index + 1}: {cost:.1f}px"
        ax.plot(xs, ys, color=color, linewidth=width, label=label)

    ax.scatter([start_position[0]], [start_position[1]], color="blue", s=45, label="Start")
    ax.scatter([goal_position[0]], [goal_position[1]], color="lime", s=55, label="Goal")
    ax.legend(loc="upper right")
    fig.canvas.draw()
    plt.show(block=False)


def choose_urgent_checkpoint(checkpoints):
    print("\nGenerated checkpoints:")
    for index, checkpoint in enumerate(checkpoints):
        print(f"  {index + 1}: {checkpoint}")

    try:
        choice = int(input("Choose urgent checkpoint first (1-3): ").strip())
    except ValueError:
        choice = 1

    if choice < 1 or choice > len(checkpoints):
        choice = 1
    return choice - 1


def build_checkpoint_route(planner, start_position, checkpoints, urgent_index, final_goal):
    remaining = list(enumerate(checkpoints))
    urgent_checkpoint = remaining.pop(urgent_index)
    visit_order = [urgent_checkpoint]
    current_position = urgent_checkpoint[1]

    while remaining:
        best_option = None
        for option in remaining:
            path, cost = planner.dijkstra(
                planner.nearest_free_node(current_position),
                planner.nearest_free_node(option[1]),
            )
            if path is not None and (best_option is None or cost < best_option[0]):
                best_option = (cost, option)

        if best_option is None:
            visit_order.extend(remaining)
            break

        next_checkpoint = best_option[1]
        visit_order.append(next_checkpoint)
        remaining.remove(next_checkpoint)
        current_position = next_checkpoint[1]

    route_targets = [checkpoint for _, checkpoint in visit_order] + [final_goal]
    route_path = []
    segment_start = start_position

    for target in route_targets:
        segment_path, _ = planner.dijkstra(
            planner.nearest_free_node(segment_start),
            planner.nearest_free_node(target),
        )
        if segment_path is None:
            continue
        if route_path:
            route_path.extend(segment_path[1:])
        else:
            route_path.extend(segment_path)
        route_path.append(target)
        segment_start = target

    return route_path, visit_order


def plot_checkpoint_route(map_width, map_height, obstacles, start_position, goal_position, checkpoints, urgent_index, route_path, visit_order):
    plt.ion()
    fig, ax = plt.subplots(num="Urgent Checkpoint Dijkstra Route")
    ax.set_title("Urgent Checkpoint Dijkstra Route")
    ax.set_xlim(0, map_width)
    ax.set_ylim(map_height, 0)
    ax.set_aspect("equal", adjustable="box")

    for obstacle in obstacles:
        x, y, width, height = obstacle
        ax.add_patch(plt.Rectangle((x, y), width, height, color="black"))

    if route_path:
        xs = [point[0] for point in route_path]
        ys = [point[1] for point in route_path]
        ax.plot(xs, ys, color="green", linewidth=3, label="Planned route")

    for index, checkpoint in enumerate(checkpoints):
        color = "magenta" if index == urgent_index else "orange"
        ax.scatter([checkpoint[0]], [checkpoint[1]], color=color, s=70)
        ax.text(checkpoint[0] + 6, checkpoint[1] - 6, f"C{index + 1}", color=color, weight="bold")

    order_text = " -> ".join(f"C{index + 1}" for index, _ in visit_order)
    if order_text:
        order_text += " -> Goal"
        ax.text(10, map_height - 14, f"Visit order: {order_text}", color="black")

    ax.scatter([start_position[0]], [start_position[1]], color="blue", s=45, label="Start")
    ax.scatter([goal_position[0]], [goal_position[1]], color="lime", s=55, label="Goal")
    ax.legend(loc="upper right")
    fig.canvas.draw()
    plt.show(block=False)

if __name__ == "__main__":
    
    objects=Map()
    pygame.init()     
    screen=pygame.display.set_mode((500,500))
    pygame.display.set_caption("Drone Simulation")

    robot = Robot((30,30),0.005)
    obstacles_list = objects.Generate_obstacles(50)  # Store the list of obstacles in a variable
    checkpoints = objects.Generate_checkpoints(obstacles_list, (robot.x, robot.y), count=3)
    Goal_Pos = objects.Generate_goal_position(obstacles_list, (robot.x, robot.y), existing_positions=checkpoints)
    urgent_checkpoint_index = choose_urgent_checkpoint(checkpoints)
    planner = DijkstraPathPlanner(objects.Map_width, objects.Map_height, obstacles_list)
    route_path, visit_order = build_checkpoint_route(
        planner,
        (robot.x, robot.y),
        checkpoints,
        urgent_checkpoint_index,
        Goal_Pos,
    )
    robot.set_planned_path(route_path, Goal_Pos)
    plot_checkpoint_route(
        objects.Map_width,
        objects.Map_height,
        obstacles_list,
        (robot.x, robot.y),
        Goal_Pos,
        checkpoints,
        urgent_checkpoint_index,
        route_path,
        visit_order,
    )
    lasttime = pygame.time.get_ticks()
    running = True

    while running:
        
        # CLEAR THE SCREEN FIRST
        screen.fill((255, 255, 255)) 
    
        # UPDATE TIME DELTA
        Currenttime = pygame.time.get_ticks()
        dt = (pygame.time.get_ticks() - lasttime) / 1000.0
        lasttime = Currenttime
        if dt > 0.1: dt = 0.016 # Cap dt to prevent jump crashes
            
        # DRAW THE STORED OBSTACLES (Inside the loop)
        for obstacle in obstacles_list:
            color = (0, 0, 0)  # Default color for obstacles
            if robot.rect.colliderect(pygame.Rect(obstacle)):
                color = (255, 0, 0)  # Change color to red on collision 
            pygame.draw.rect(screen, color, obstacle)
        
        # HANDLE EVENTS
        event=None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
         
        objects.goal_position_circle(screen, Goal_Pos)
        objects.draw_checkpoints(screen, checkpoints, urgent_checkpoint_index)
        sensor=LIDAR((robot.x,robot.y),robot.theta,72,80,screen.get_size())
        Lidar_data=sensor.sense_obstacle()
        goal_reached = robot.follow_planned_path()
        if goal_reached:
            running = False
        else:
            robot.move(dt, event)
            robot.collision(obstacles_list)
        robot.record_track_point()
        robot.draw_planned_path(screen)
        robot.draw_track(screen)
        robot.draw(screen)
        sensor.Sensor_rays(screen,(robot.x,robot.y),robot.theta)
        if Lidar_data:
            objects.add_lidar_hits(Lidar_data)
            objects.draw_point_cloud(screen)
        
         
    
        # Update THE DISPLAY
        pygame.display.update()
        plt.pause(0.001)
        

    pygame.quit()
    
        
