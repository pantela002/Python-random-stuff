import pygame
import random
import time

GREY = (128, 128, 128)
GREEN = (0, 255, 0)


BLOCK_SIZE = 40
SPEED = 20
pygame.init()


class Display:
    def __init__(self) -> None:
        self.WINDOW_SIZE = (640,480)
        self.display = pygame.display.set_mode((640,480))
        pygame.display.set_caption('FloodFill')
        self.clock = pygame.time.Clock()
        self.islands = []
        self.positions=[]
        self.make_positions()
        self.game_started = False
        self.playyer1 = 0
        self.playyer2 = 0


    def make_positions(self):
        for x in range(0, self.WINDOW_SIZE[0], BLOCK_SIZE):
            for y in range(0, self.WINDOW_SIZE[1], BLOCK_SIZE):
                center = (x + BLOCK_SIZE // 2, y + BLOCK_SIZE // 2)
                self.positions.append(center)


    def add_island(self):
        random.shuffle(self.positions)
        print(len(self.positions),end=' ')
        for i in range(5):
            self.islands.append(self.positions[i])
        self.positions=self.positions[5:]
        print(len(self.positions))
        self.draw()
        self.clock.tick(SPEED)

    def draw(self):
        self.display.fill((0,0,0))
        for x in range(0, self.WINDOW_SIZE[0], BLOCK_SIZE):
            for y in range(0, self.WINDOW_SIZE[1], BLOCK_SIZE):
                center = (x + BLOCK_SIZE // 2, y + BLOCK_SIZE // 2)
                pygame.draw.circle(self.display, GREY, center, BLOCK_SIZE // 2)

        for center in self.islands:
            pygame.draw.circle(self.display, GREEN, center, BLOCK_SIZE // 2)

        pygame.display.flip()

    def start(self):
        self.display.fill((0,0,0))

        over=False
        while True:
            if over == True:
                break
            
            if self.game_started == False:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        quit()
                    elif e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_RETURN:
                            self.game_started = True
                            break
                        elif e.key == pygame.K_m:
                            over=True
                            break
            else:
                self.add_island()
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        quit()
                    elif e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_p or e.key == pygame.K_SPACE:
                            if e.key == pygame.K_p:
                                self.playyer1 += 1
                            else:
                                self.playyer2 += 1
                            self.positions=[]
                            self.make_positions()
                            self.islands = []
                            self.game_started = False
                            self.display.fill((0,0,0))
                            pygame.display.flip()
                            break
                        elif e.key == pygame.K_m:
                            over=True
                            break
                            


        





if __name__ == '__main__':
    dis = Display()
    dis.start()
    print("player1 : "+dis.playyer1)
    print("player2 : "+dis.playyer2)
    
    
