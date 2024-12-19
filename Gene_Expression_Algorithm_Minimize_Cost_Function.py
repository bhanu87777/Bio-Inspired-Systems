import numpy as np

def objective_function(x):
    return sum(x**2)

population_size = 20
num_genes = 10
mutation_rate = 0.1
crossover_rate = 0.7
num_generations = 100
bounds = (-10, 10)

def initialize_population(pop_size, num_genes, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, num_genes))

def evaluate_fitness(population):
    return np.array([objective_function(individual) for individual in population])

def selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, mutation_rate, bounds):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(bounds[0], bounds[1])
    return individual

population = initialize_population(population_size, num_genes, bounds)
best_solution = None
best_fitness = float("inf")

for generation in range(num_generations):
    fitness = evaluate_fitness(population)
    min_index = np.argmin(fitness)
    if fitness[min_index] < best_fitness:
        best_fitness = fitness[min_index]
        best_solution = population[min_index]
    selected_population = selection(population, fitness)
    next_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_population[i], selected_population[i+1]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate, bounds)
        child2 = mutate(child2, mutation_rate, bounds)
        next_population.append(child1)
        next_population.append(child2)
    population = np.array(next_population)

print("Best solution:", *best_solution, sep='\n')
print(f"\nBest fitness: {best_fitness:.3f}")
