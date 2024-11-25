import numpy as np
import random
import math

def calculate_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def create_population(cities, population_size):
    population = []
    for _ in range(population_size):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_fitness(individual):
    distance = 0
    for i in range(len(individual) - 1):
        distance += calculate_distance(individual[i], individual[i + 1])
    distance += calculate_distance(individual[-1], individual[0])
    return 1 / distance

def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [fitness / total_fitness for fitness in fitness_values]
    
    parent1 = random.choices(population, selection_probs)[0]
    parent2 = random.choices(population, selection_probs)[0]
    
    return parent1, parent2

def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])

    for i in range(start, end + 1):
        child[i] = parent1[i]
    
    pointer = 0
    for i in range(size):
        if child[i] is None:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(child, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(child)), 2)
        child[idx1], child[idx2] = child[idx2], child[idx1]
    return child

def genetic_algorithm(cities, population_size, generations, mutation_rate):
    population = create_population(cities, population_size)
    for generation in range(generations):
        fitness_values = [calculate_fitness(individual) for individual in population]
        
        best_individual = population[np.argmax(fitness_values)]
        best_fitness = max(fitness_values)
        print(f"Generation {generation + 1}, Best Distance: {1/best_fitness:.2f}")
        
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_values)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population
    
    best_individual = population[np.argmax(fitness_values)]
    return best_individual

cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0), (2, 4), (5, 2)]

population_size = 100
generations = 500
mutation_rate = 0.01

best_path = genetic_algorithm(cities, population_size, generations, mutation_rate)

print("\nBest Path:", best_path)
print("Total Distance:", sum([calculate_distance(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]) + calculate_distance(best_path[-1], best_path[0]))

