import random 
import pygame
import math
import numpy as np


class RRTMAP: 

    def __init__(self,start,goal,Mapdimensions,obstacles_dimensions,obstacle_number):
        self.start=start
        self.goal=goal
        self.Mapdimensions=[self.mapheight,self.mapwidth]=Mapdimensions
        self.obstacles_dimensions=obstacles_dimensions
        self.obstacle_number=obstacle_number
        
        
        #window settings
        self.window_name="RRT Path Planning"
        pygame.display.set_caption(self.window_name)
        self.map=pygame.display.set_mode((self.mapwidth,self.mapheight))
        self.map.fill((255,255,255)) #White background
        
        #nodes
        self.noderadius=3
        self.nodethickness=0
        self.edgethickness=1
        
        #obstacles
        self.obstacles= []
        self.obstacles_dimensions=obstacles_dimensions
        self.obstacle_number=obstacle_number
        
        #colors
        self.white= (255,255,255)
        self.black= (0,0,0)
        self.red= (255,0,0)
        self.green= (0,255,0)
        self.blue= (0,0,255)
        self.grey= (70,70,70)
        
        
        

    def  drawmap(self):
        pass
    
    def drawpath(self):
        pass
    
    def drawobstacle(self,node):
        pass  
    

class RRTGraph:
    
    def __init__(self,start,goal,Mapdimensions,obstacles_dimensions,obstacle_number):
        (x,y)=start
        self.start=start
        self.goal=goal
        self.mapheight,self.mapwidth=Mapdimensions
        
        #inintial node list
        self.x=[]
        self.y=[]
        self.parent=[]
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
        
        #obstacles
        self.obstacles= []
        self.obstacles_dimensions=obstacles_dimensions
        self.obstacle_number=obstacle_number
        
        #path
        self.goalstate=None
        self.path=[]
        self.path_points=[]
        self.path_length=0
    
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
    
    def Add_node(self,node):
        pass
    
    def remove_node(self,node):
        pass
    
    def add_edge(self,node1,node2):
        pass
    
    def remove_edge(self,node1,node2):
        pass
    
    def number_of_nodes(self):
        pass
    
    def distance(self,node1,node2):
        pass
    
    def nearest_node(self,node):
        pass
    
    def isfree(self,node1,node2):
        pass
    
    def crossobstacle(self,node1,node2):
        pass
    
    def connect(self)
        pass    
    
    def step(self)
        pass
    
    def path_to_goal(self)
        pass
    
    def get_path_points(self)
        pass
    
    def bias(self)
        pass
    
    def expand(self)
        pass
    
    def cost(self)
        pass
    
    
                