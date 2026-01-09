import random
import pygame
import numpy as np

linear_velocity_mps=0.01  #linear velocity in meters per second
angular_velocity_mps=0.01  #angular velocity in radians per second

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
        X_axis=(Center_x + n*np.cos(rotation), Center_y- n*np.sin(rotation))
        Y_axis=(Center_x - n*np.sin(rotation), Center_y - n*np.cos(rotation))
        pygame.draw.line(screen,(255,0,0),(Center_x,Center_y),X_axis,2)
        pygame.draw.line(screen,(0,255,0),(Center_x,Center_y),Y_axis,2)
        
        
    def draw(self,screen):
        screen.blit(self.rotated,self.rect)
        # draw a red ring when in collision state
        if self.collided:
            radius = max(8, int(self.width/4))
            pygame.draw.circle(screen, (255,0,0), (int(self.x), int(self.y)), radius, 2)
        
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
                    self.Vr += 0.3
                    self.Vl -= 0.1
                    
                elif event.key == pygame.K_DOWN:
                    # Spin left motor faster and right motor slower to pivot right
                    self.Vl += 0.3
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

    robot = Robot((50,50),0.01)
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
    
        robot.move(dt, event)
        robot.collision(obstacles_list)
        robot.draw(screen)
        
        robot.robot_frame(screen, (robot.x, robot.y), robot.theta)
        objects.Legend(robot.Vl, robot.Vr, robot.theta)   
    
        # Update THE DISPLAY
        pygame.display.update()
        

    pygame.quit()
    
        
