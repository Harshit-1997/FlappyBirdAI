from multiprocessing.context import Process
from NeuralNetwork import Brain
import random
import multiprocessing as mp

class Population:

    def __init__(self, inputs, outputs, init_connections, population_size):
        self.population_size = population_size
        self.population = dict()
        for i in range(self.population_size):
            self.population[i]=Brain(inputs,outputs,init_connections)

########################################################################################################################

    def getPopulation(self):
        return self.population

########################################################################################################################

    def createNextGeneration(self,keep=5):
        next_gen =[]
        current_gen = list(self.population.values())
        current_gen.sort(key=lambda brain:brain.fitness, reverse=True)
        current_gen = current_gen[:self.population_size//2]
        add = 0
        worst = current_gen[-1].fitness
        if worst<0:
            add = abs(worst)
        fitnesses = [brain.fitness+add+0.1 for brain in current_gen]
        next_gen+=current_gen[:keep]
        for brain in next_gen:
            print(brain.fitness)
        for _ in range(self.population_size-keep):
            parent1,parent2 = random.choices(current_gen,fitnesses,k=2)
            next_gen.append(Brain(parent1,parent2))

        del self.population
        self.population = dict()
        for i in range(self.population_size):
            self.population[i] = next_gen[i]

########################################################################################################################

    def evaluateGeneration(self, func, process_count):
        queue = mp.Queue()
        sema = mp.Semaphore(process_count)
        processes = [ mp.Process(target=func,args=(pos,player,queue,sema)) for pos,player in self.population.items()]

        for p in processes:
            sema.acquire()
            p.start()

        for p in processes:
            p.join()
            p.terminate()
    
        fitnesses = []
        for _ in range(self.population_size):
            fit = queue.get()
            fitnesses.append(fit)
    
        for i,fitness in fitnesses:
            self.population[i].fitness = fitness

########################################################################################################################
if __name__ == '__main__':
    p = Population(5,5,15,5)
    for fit,model in enumerate(p.getPopulation()):
        model.fitness = fit 
        model.drawNetwork()

    p.createNextGeneration(keep=2)

    for model in p.getPopulation():
        model.drawNetwork()
