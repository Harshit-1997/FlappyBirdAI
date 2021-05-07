import random
import math
import pygame

class Obstacles:

    def __init__(self,screen_width,screen_height, obstacle_width, min_height,gap_size, obstacle_distance, obstacle_vel):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obstacle_width = obstacle_width
        self.min_height = min_height
        self.gap_size = gap_size
        self.obstacle_vel = obstacle_vel
        self.obstacle_distance = obstacle_distance
        self.obstacles = []

########################################################################################################################

    def getNewObstacle(self):
        y = random.randint(self.min_height,self.screen_height-self.min_height-self.gap_size)
        obstacle = dict()
        obstacle['top'] = dict()
        obstacle['top']['x1'] = self.screen_width
        obstacle['top']['y1'] = 0
        obstacle['top']['x2'] = self.screen_width+self.obstacle_width
        obstacle['top']['y2'] = y
        obstacle['bottom'] = dict()
        obstacle['bottom']['x1'] = self.screen_width
        obstacle['bottom']['y1'] = y+self.gap_size
        obstacle['bottom']['x2'] = self.screen_width+self.obstacle_width
        obstacle['bottom']['y2'] = self.screen_height
        return obstacle

########################################################################################################################

    def updateObstacles(self):
        if len(self.obstacles)==0:
            self.obstacles.append(self.getNewObstacle())
        elif self.obstacles[-1]['top']['x2']==self.screen_width-self.obstacle_distance:
            self.obstacles.append(self.getNewObstacle())
        elif self.obstacles[0]['top']['x2']+self.obstacle_width==0:
            self.obstacles.pop(0)

########################################################################################################################

    def moveObstacles(self):
        for x in self.obstacles:
            x['top']['x1']-=self.obstacle_vel
            x['top']['x2']-=self.obstacle_vel
            x['bottom']['x1']-=self.obstacle_vel
            x['bottom']['x2']-=self.obstacle_vel

        self.updateObstacles()

########################################################################################################################

    def draw(self,surface):
        BLUE = (0,0,255)
        obs_wdt = self.obstacle_width
        scr_hgt = self.screen_height
        for obs in self.obstacles:
            pygame.draw.rect(surface, BLUE, [obs['top']['x1'],obs['top']['y1'],obs_wdt,obs['top']['y2']])
            pygame.draw.rect(surface, BLUE, [obs['bottom']['x1'],obs['bottom']['y1'],obs_wdt,scr_hgt-obs['bottom']['y1']])

########################################################################################################################

    def detectCollision(self,x,y,r):
        for obs in self.obstacles:
            tx1 = obs['top']['x1'] 
            ty1 = obs['top']['y1']
            tx2 = obs['top']['x2']
            ty2 = obs['top']['y2']
            bx1 = obs['bottom']['x1']
            by1 = obs['bottom']['y1']
            bx2 = obs['bottom']['x2']
            by2 = obs['bottom']['y2']

            if y < ty2 and tx1 <= x+r <= tx2:
                return True, -10

            if y > by1 and bx1 <= x+r <= bx2:
                return True, -10

            if tx2 > x > tx1 and (ty2>y-r or by1<y+r):
                return True, -10

            if math.sqrt( (tx1-x)**2 + (ty2-y)**2) < r:  # top right corner
                return True, -10

            if math.sqrt( (tx2-x)**2 + (ty2-y)**2) < r:  # top left corner
                return True, -10
            
            if math.sqrt( (bx1-x)**2 + (by1-y)**2) < r:  # bottom right corner
                return True, -10

            if math.sqrt( (bx2-x)**2 + (by1-y)**2) < r:  # bottom left corner
                return True, -10

            if tx1<x<tx2 and ty2<y<by1:
                return False, 20

        return False, 0

########################################################################################################################

    def getClosestObsticle(self,birdx,sz):
        for obs in self.obstacles:
            if obs['top']['x2']>birdx-sz:
                return obs['top']['x1']+self.obstacle_width/2,obs['top']['y2']+self.gap_size/2
        return birdx+10,self.screen_height/2
