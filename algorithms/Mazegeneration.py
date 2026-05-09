import pygame
import random


class MazeGenerator:
    def __init__(self):
        self.MAX_SIZE = 41
        self.TILE_SIZE = 15
        self.SCREEN_SIZE = self.MAX_SIZE * self.TILE_SIZE
        self.START_ROW, self.START_COL = 1,1

# Colors
        self.COLOR_WALL = (0, 0, 0)      # Black
        self.COLOR_PATH = (255, 255, 255) # White
        self.COLOR_BTN = (100, 100, 100)

# Directions: [row_offset, col_offset] (Steps of 2 to jump over walls)
        self.DIRECTIONS = [(0, -2), (2, 0), (0, 2), (-2, 0)]

    def is_valid_wall(self, grid, row, col):
        return (0 < row < self.MAX_SIZE - 1 and 
                0 < col < self.MAX_SIZE - 1 and 
                grid[row][col] == 1)

    def generate_maze(self):
        # 1 = Wall, 0 = Path
        grid = [[1 for _ in range((self.MAX_SIZE))] for _ in range(self.MAX_SIZE)]
        
        # Starting point
        grid[self.START_ROW][self.START_COL] = 0
    
    # frontier stores (row, col, parent_row, parent_col)
        frontier = set()

        for r, c in self.DIRECTIONS:
            new_row, new_col = self.START_ROW + r, self.START_COL + c
            if self.is_valid_wall(grid, new_row, new_col):
                frontier.add((new_row, new_col, self.START_ROW, self.START_COL))

        while frontier:
        # Pick a random frontier cell
            row, col, p_row, p_col = random.choice(list(frontier))
            frontier.remove((row, col, p_row, p_col))

            if grid[row][col] == 1:
            # Carve path to the new cell
               grid[row][col] = 0
            # Carve the wall between the new cell and its parent
               grid[(row + p_row) // 2][(col + p_col) // 2] = 0

            # Add new neighbors to frontier
            for r, c in self.DIRECTIONS:
                n_row, n_col = row + r, col + c
                if self.is_valid_wall(grid, n_row, n_col):
                    frontier.add((n_row, n_col, row, col))
    
        return grid

def main():
    Maze=MazeGenerator()
    pygame.init()
    screen = pygame.display.set_mode((Maze.SCREEN_SIZE, Maze.SCREEN_SIZE + 50))
    pygame.display.set_caption("Maze Generator (Prim's Algorithm)")
    clock = pygame.time.Clock()

    current_maze = Maze.generate_maze()
    
    running = True
    while running:
        screen.fill((200, 200, 200)) # Background for the "button" area
        # 1. Draw the Maze
        for r in range(Maze.MAX_SIZE):
            for c in range(Maze.MAX_SIZE):
                color = Maze.COLOR_WALL if current_maze[r][c] == 1 else Maze.COLOR_PATH
                pygame.draw.rect(screen, color, (c * Maze.TILE_SIZE ,
                                                 r * Maze.TILE_SIZE ,
                                                 Maze.TILE_SIZE,
                                                 Maze.TILE_SIZE))

        # 2. Draw "Button" UI
        button_rect = pygame.Rect(10, Maze.SCREEN_SIZE + 10, 150, 30)
        pygame.draw.rect(screen, Maze.COLOR_BTN, button_rect)
        font = pygame.font.SysFont(None, 24)
        img = font.render('Generate Maze', True, (255, 255, 255))
        screen.blit(img, (20, Maze.SCREEN_SIZE + 15))
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    current_maze = Maze.generate_maze()

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()