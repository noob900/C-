import pygame

# Initialize pygame
pygame.joystick.init()


class player(object):
    def __init__(self):
        self.player = pygame.rect.Rect(100, 100, 50, 50)
        self.color = "Red"
        
    def move(self, dx, dy):
        self.player.move_ip(dx, dy)
        
    def change_color(self, color):
        self.color = color
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.player)
    
    
pygame.init()
player1 = player()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Joystick Test")

joysticks=[]



while True:
    
    player1.draw(screen)
    pygame.display.update()
    clock.tick(60)
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False       
        if event.type == pygame.JOYDEVICEADDED:
           print(event)


                
    

        