import random
import pygame
import numpy as np
import math

linear_velocity_mps=0.01  #linear velocity in meters per second
angular_velocity_mps=0.01  #angular velocity in radians per second

class LIDAR:
    
    def __init__(self,robot_position,robot_theta,number_of_rays, Range, surface):
        self.robot_position=robot_position
        self.robot_theta=robot_theta
        self.number_of_rays=number_of_rays
        self.Range=Range
        self.surface=pygame.display.get_surface().get_size()
        self.w,self.h=self.surface
        self.angles=np.linspace(0,2*np.pi,self.number_of_rays,endpoint=False)
        
    def sense_obstacle(self):
        data=[]
        Rx, Ry=self.robot_position[0], self.robot_position[1]
        for angle in self.angles:
            Rx1,Ry1= Rx + self.Range*np.cos(angle + self.robot_theta), Ry - self.Range*np.sin(angle + self.robot_theta)
            for i in range(self.Range):
                u= i/self.Range
                x=int(Rx + u*(Rx1 - Rx))
                y=int(Ry + u*(Ry1 - Ry)) 
                if 0<x<self.w and 0<y<self.h:
                    color= self.surface=pygame.display.get_surface().get_at((x,y))
                    if color==(0,0,0,255): #black obstacle
                        Distance=((x,y))
                        data.append(Distance)
                        break
        if len(data) > 0:
            return data
        else:
            return False
        
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
        self.robot_image=pygame.image.load('Drone.png')
        self.robot_image=pygame.transform.scale(self.robot_image,(int(self.width),int(self.width))) #scale the image to the robot width
        self.rotated=self.robot_image
        self.rect=self.rotated.get_rect(center=(self.x,self.y))

        # store previous pose for collision recovery
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
        self.collided = False
        

        
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
     
    def autonomous_control(self, lidar_data, goal_pos):
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
                if math.hypot(hit[0]-self.x, hit[1]-self.y) < 80:
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
        self.collided = False  # reset collision state
        for i in obstacle:
            rect = pygame.Rect(i)
            if self.rect.colliderect(rect):
                # revert to previous pose and stop both wheels
                self.x = self.prev_x
                self.y = self.prev_y
                self.theta = self.prev_theta
                self.Vl = 0
                self.Vr = 0
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
            
    def add_lidar_hits(self, hits):
        for hit in hits:
            if hit not in self.point_cloud:  # Avoid duplicates
                self.point_cloud.append(hit)     
                
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

if __name__ == "__main__":
    
    objects=Map()
    pygame.init()     
    screen=pygame.display.set_mode((500,500))
    pygame.display.set_caption("Drone Simulation")

    robot = Robot((30,30),0.005)
    obstacles_list = objects.Generate_obstacles(20)  # Store the list of obstacles in a variable
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
         
        Goal_Pos = (100, 450)  # Define a fixed goal position
        objects.goal_position_circle(screen, Goal_Pos)
        sensor=LIDAR((robot.x,robot.y),robot.theta,16,100,screen.get_size())
        Lidar_data=sensor.sense_obstacle()
        robot.collision(obstacles_list)
        robot.draw(screen)
        robot.autonomous_control(Lidar_data, Goal_Pos)
        robot.move(dt, event)
        sensor.Sensor_rays(screen,(robot.x,robot.y),robot.theta)
        if Lidar_data:
            objects.add_lidar_hits(Lidar_data)
            objects.draw_point_cloud(screen)
        
         
    
        # Update THE DISPLAY
        pygame.display.update()
        

    pygame.quit()
    
        
