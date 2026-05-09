import turtle
import time
import sys

# Set up the screen
wn = turtle.Screen()
wn.bgcolor("black")
wn.setup(1300, 700)
wn.title("Draw Your Own Maze")

# Class for the Maze turtle (white square)
class Maze(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("white")
        self.penup()
        self.speed(0)

# Class for the End marker turtle (green square)
class End(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("green")
        self.penup()
        self.speed(0)

# Class for the sprite turtle (red turtle)
class Sprite(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("turtle")
        self.color("red")
        self.setheading(270)  # Point turtle downward
        self.penup()
        self.speed(0)

    def spriteDown(self):
        if self.heading() == 270:
            x_walls = round(self.xcor(), 0)
            y_walls = round(self.ycor(), 0)
            if (x_walls, y_walls) in finish:
                print("Finished")
                endProgram()
            if (x_walls + 24, y_walls) in walls:
                if (x_walls, y_walls - 24) not in walls:
                    self.forward(24)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(24)

    def spriteLeft(self):
        if self.heading() == 0:
            x_walls = round(self.xcor(), 0)
            y_walls = round(self.ycor(), 0)
            if (x_walls, y_walls) in finish:
                print("Finished")
                endProgram()
            if (x_walls, y_walls + 24) in walls:
                if (x_walls + 24, y_walls) not in walls:
                    self.forward(24)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(24)

    def spriteUp(self):
        if self.heading() == 90:
            x_walls = round(self.xcor(), 0)
            y_walls = round(self.ycor(), 0)
            if (x_walls, y_walls) in finish:
                print("Finished")
                endProgram()
            if (x_walls - 24, y_walls) in walls:
                if (x_walls, y_walls + 24) not in walls:
                    self.forward(24)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(24)

    def spriteRight(self):
        if self.heading() == 180:
            x_walls = round(self.xcor(), 0)
            y_walls = round(self.ycor(), 0)
            if (x_walls, y_walls) in finish:
                print("Finished")
                endProgram()
            if (x_walls, y_walls - 24) in walls:
                if (x_walls - 24, y_walls) not in walls:
                    self.forward(24)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(24)


def endProgram():
    wn.exitonclick()
    sys.exit()


# Function to handle mouse clicks
def draw_wall(x, y):
    x = round(x // 24) * 24
    y = round(y // 24) * 24
    if (x, y) not in walls and (x, y) not in finish and (x, y) != sprite.pos():
        maze.goto(x, y)
        maze.stamp()
        walls.append((x, y))


def draw_start(x, y):
    x = round(x // 24) * 24
    y = round(y // 24) * 24
    if (x, y) not in walls and (x, y) not in finish:
        sprite.goto(x, y)


def draw_end(x, y):
    x = round(x // 24) * 24
    y = round(y // 24) * 24
    if (x, y) not in walls and (x, y) != sprite.pos():
        end.goto(x, y)
        end.stamp()
        finish.append((x, y))


# Function to start the turtle's movement
def start_movement(x, y):
    global movement_started
    if not movement_started:
        movement_started = True
        while movement_started:
            sprite.spriteRight()
            sprite.spriteDown()
            sprite.spriteLeft()
            sprite.spriteUp()

            if (round(sprite.xcor(), 0), round(sprite.ycor(), 0)) in finish:
                endProgram()

            time.sleep(0.02)


# Initialize turtles and lists
maze = Maze()
sprite = Sprite()
end = End()
walls = []
finish = []

# Movement flag
movement_started = False

# Bind mouse clicks to functions
wn.onclick(draw_wall, btn=1)  # Left-click to draw walls
wn.onclick(draw_start, btn=2)  # Middle-click to set start position
wn.onclick(draw_end, btn=3)    # Right-click to set end position

# Create a Start button
start_button = turtle.Turtle()
start_button.shape("square")
start_button.color("blue")
start_button.penup()
start_button.goto(0, -300)  # Position the button at the bottom center
start_button.write("Start", align="center", font=("Arial", 16, "bold"))
start_button.goto(0, -330)
start_button.onclick(start_movement)

# Position the turtle just above the button
sprite.goto(0, -250)  # Adjust the y-coordinate to place the turtle just above the button

# Instructions for the user
print("Left-click to draw walls (+)")
print("Middle-click to set start position (s)")
print("Right-click to set end position (e)")
print("Click the Start button to begin navigation")

# Main loop
wn.mainloop()