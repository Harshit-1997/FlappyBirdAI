from NeuralNetwork import Brain
import random

class Population:

    def __init__(self, inputs, outputs, init_connections, population_size):
        self.population_size = population_size
        self.population = [ Brain(inputs,outputs,init_connections) for _ in range(self.population_size)]

########################################################################################################################

    def getPopulation(self):
        return self.population

########################################################################################################################

    def createNextGeneration(self,keep=5):
        next_gen =[]
        self.population.sort(key=lambda brain:brain.fitness)
        fitnesses = [brain.fitness for brain in self.population]
        next_gen+=self.population[:keep]
        for _ in range(self.population_size-keep):
            parent1,parent2 = random.choices(self.population,fitnesses,k=2)
            next_gen.append(Brain(parent1,parent2))

        del self.population
        self.population = next_gen

########################################################################################################################

if __name__ == '__main__':
    p = Population(5,5,15,5)
    for fit,model in enumerate(p.getPopulation()):
        model.fitness = fit 
        model.drawNetwork()

    p.createNextGeneration(keep=2)

    for model in p.getPopulation():
        model.drawNetwork()
