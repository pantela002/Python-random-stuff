
import pygame
import random
from circle import Circle, Coordinates
import math
pygame.init()
pygame.display.set_caption("My First Pygame")
screen_size = 900
screen = pygame.display.set_mode((screen_size, screen_size))



def draw_grid(screen, screen_size, grid_size):
    for x in range(1, screen_size, grid_size):
        pygame.draw.line(screen, (0, 0, 0), (x, 0), (x, screen_size))
    for y in range(1, screen_size, grid_size):
        pygame.draw.line(screen, (0, 0, 0), (0, y), (screen_size, y))

def intersect(c1,c2):
    return math.sqrt((c1.get_x()-c2.get_x())**2+(c1.get_y()-c2.get_y())**2)<=c1.get_radius()+c2.get_radius()+1

if __name__ == "__main__":
    
    circle_size=40
    clock = pygame.time.Clock()
    running = True
    dt = 0
    dt = clock.tick(60) / 1000
    #player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    circles = []
    p1=random.random()
    p2=random.random()
    p3=random.random()

    circles.append(Circle(Coordinates(screen_size // 3, screen_size // 3), circle_size,math.sin(p1),math.cos(p1)))
    circles.append(Circle(Coordinates(2*screen_size // 3, 2*screen_size // 3), circle_size,math.sin(p2),math.cos(p2)))
    circles.append(Circle(Coordinates(3*screen_size // 5, 3*screen_size // 6), circle_size,math.sin(p3),math.cos(p3)))
    while running:
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("purple")

        draw_grid(screen, screen_size, 20)

        for c in circles:
            #pygame.draw.circle(screen, "red",c.get_pos() , c.get_radius())
            pygame.draw.circle(screen, "red", (c.get_x(),c.get_y()) , c.get_radius())

        for c in circles:
            if c.get_x() > screen_size-circle_size or c.get_x() < circle_size:
                c.set_dx(-c.get_dx())
            if c.get_y() > screen_size-circle_size or c.get_y() < circle_size:
                c.set_dy(-c.get_dy())
        
        for i in range(len(circles)):
            for j in range(len(circles)):
                if i!=j:
                    if intersect(circles[i],circles[j]):
                        vector_ij=(circles[j].get_x()-circles[i].get_x(),circles[j].get_y()-circles[i].get_y())
                        circles[i].set_dx(-circles[i].get_dx())
                        circles[i].set_dy(-circles[i].get_dy())



        for c in circles:
            c.move(700,dt)
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()