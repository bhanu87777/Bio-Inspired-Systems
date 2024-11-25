import numpy as np
import random
import math

class Particle:
    def __init__(self, n, cities):
        self.position = [random.randint(0, n - 1) for _ in range(n)]
        self.velocity = [random.uniform(-1, 1) for _ in range(n)]
        self.best_position = self.position
        self.best_fitness = float('inf')
        self.cities = cities
        self.n = n

    def fitness(self):
        distance = 0
        for i in range(self.n - 1):
            distance += self.distance(self.position[i], self.position[i + 1])
        distance += self.distance(self.position[-1], self.position[0])
        return distance

    def distance(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update_best(self):
        fitness_value = self.fitness()
        if fitness_value < self.best_fitness:
            self.best_fitness = fitness_value
            self.best_position = self.position

    def update_position(self):
        for i in range(self.n):
            self.position[i] = int(self.position[i] + self.velocity[i])
            if self.position[i] >= self.n:
                self.position[i] = self.n - 1
            elif self.position[i] < 0:
                self.position[i] = 0

    def update_velocity(self, global_best_position):
        w = 0.5
        c1 = 1.5
        c2 = 1.5
        for i in range(self.n):
            r1 = random.random()
            r2 = random.random()
            self.velocity[i] = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) + c2 * r2 * (global_best_position[i] - self.position[i])

def pso_vrp(cities, population_size, generations):
    n = len(cities)
    particles = [Particle(n, cities) for _ in range(population_size)]
    global_best_position = particles[0].position
    global_best_fitness = float('inf')

    for generation in range(generations):
        for particle in particles:
            particle.update_best()
            if particle.best_fitness < global_best_fitness:
                global_best_fitness = particle.best_fitness
                global_best_position = particle.best_position

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

    return global_best_position

cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0), (2, 4), (5, 2)]
population_size = 100
generations = 500

best_route = pso_vrp(cities, population_size, generations)

print("\nBest Route:", best_route)
print("Total Distance:", sum([math.sqrt((cities[best_route[i]][0] - cities[best_route[i + 1]][0])**2 + (cities[best_route[i]][1] - cities[best_route[i + 1]][1])**2) for i in range(len(best_route) - 1)]) + math.sqrt((cities[best_route[-1]][0] - cities[best_route[0]][0])**2 + (cities[best_route[-1]][1] - cities[best_route[0]][1])**2))