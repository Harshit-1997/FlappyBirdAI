import pygame
class Bird:

    def __init__(self, screen_width, screen_height, size):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 100
        self.y = self.screen_height/2
        self.size = size
        self.velocity = 0
        self.acceleration = 0.2

    def move(self):
        self.velocity+=self.acceleration
        self.y+=self.velocity
        if self.y-self.size<0 or self.y+self.size>self.screen_height:
            return True
        else:
            return False

    def jump(self):
        self.velocity = -6


    def draw(self,gamescreen,color):
        pygame.draw.circle(gamescreen,color,[self.x,self.y],self.size)
