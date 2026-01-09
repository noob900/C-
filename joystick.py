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

while True:
    
    screen.fill((0,0,0))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break
        if event.type == pygame.JOYBUTTONDOWN:
            if pygame.joystick.joystick(0).get_button(0):
                player1.change_color("Green")
            elif pygame.joystick.joystick(0).get_button(1):
                player1.change_color("Blue")
            elif pygame.joystick.joystick(0).get_button(2):
                player1.change_color("white")
            elif pygame.joystick.joystick(0).get_button(3):
                player1.change_color("Black")
            print(event)
             
    
    player1.draw(screen)
    pygame.display.update()
    clock.tick(60)
        