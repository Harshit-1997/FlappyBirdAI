import Population
import Obstacles
import Bird
import pygame

WIDTH = 800
HEIGHT = 600

def run_agent(pos,player,queue,train=True):
    global WIDTH,HEIGHT
    if not train:
        pygame.init()
        surface = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("FlappyBirdAI")
        clock = pygame.time.Clock()
    score = 0 
    bird = Bird.Bird(WIDTH,HEIGHT,8)
    obs = Obstacles.Obstacles(WIDTH,HEIGHT,30,50,125,200,2)
    end = False
    while not end:
        if not train:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    end = True

                if event.type == pygame.MOUSEBUTTONDOWN:
                    end = True  

        obsx,obsy = obs.getClosestObsticle(bird.x,bird.size)
        data = [bird.x/WIDTH,bird.y/HEIGHT,obsx/WIDTH,obsy/HEIGHT,bird.velocity/100]
        
        jump = player.forward(data,'s')
        if jump==1:
            bird.jump()

        out = bird.move()
        obs.moveObstacles()

        hit, rev = obs.detectCollision(bird.x,bird.y,bird.size)

        if hit or out or (score>50000 and train):
            end = True
            if train:
                print(pos,score)
            if not queue is None:
                queue.put((pos,score))
        else:
            score+=1+rev

            
        if not train:
            if hit or out:
                col = (255,0,0)
            else:
                col = (0,255,0)
            surface.fill((0,0,0))
            bird.draw(surface,col)
            obs.draw(surface)
            pygame.display.flip()
            clock.tick(60)



p = Population.Population(5,2,5,250)
for gen in range(1000):
    p.evaluateGeneration(run_agent,50)
    b = max([brain for brain in p.getPopulation().values()],key= lambda x : x.fitness)
    print(f"best in genenration {gen} = {b.fitness}")
    run_agent(0,b,None,False)
    p.createNextGeneration(10)
