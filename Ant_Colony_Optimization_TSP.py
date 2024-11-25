import numpy as np
import random

class AntColony:
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        self.distance_matrix = distance_matrix
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_cities = len(distance_matrix)
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

    def _calc_total_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i], path[i + 1]]
        distance += self.distance_matrix[path[-1], path[0]]  # Return to starting point
        return distance

    def _choose_next_city(self, current_city, visited, pheromone, distance):
        pheromone[current_city] = 0
        pheromone = pheromone ** self.alpha
        distance = 1.0 / (distance + 1e-10)  # Avoid division by zero
        distance[current_city] = 0
        distance = distance ** self.beta
        probabilities = pheromone * distance
        for city in visited:
            probabilities[city] = 0  # No visit to already visited cities
        total = sum(probabilities)
        probabilities /= total
        return np.random.choice(range(self.n_cities), p=probabilities)

    def run(self):
        best_path = None
        best_distance = float('inf')
        
        for _ in range(self.n_iterations):
            all_paths = []
            all_distances = []
            for _ in range(self.n_ants):
                path = self._find_path()
                distance = self._calc_total_distance(path)
                all_paths.append(path)
                all_distances.append(distance)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path

            self._update_pheromone(all_paths, all_distances)
        return best_path, best_distance

    def _find_path(self):
        path = [random.randint(0, self.n_cities - 1)]
        visited = set(path)
        while len(path) < self.n_cities:
            current_city = path[-1]
            pheromone = self.pheromone[current_city]
            distance = self.distance_matrix[current_city]
            next_city = self._choose_next_city(current_city, visited, pheromone, distance)
            path.append(next_city)
            visited.add(next_city)
        return path

    def _update_pheromone(self, all_paths, all_distances):
        pheromone_delta = np.zeros_like(self.pheromone)
        for path, distance in zip(all_paths, all_distances):
            for i in range(len(path) - 1):
                pheromone_delta[path[i], path[i + 1]] += 1.0 / distance
            pheromone_delta[path[-1], path[0]] += 1.0 / distance
        self.pheromone = (1 - self.decay) * self.pheromone + pheromone_delta
        self.pheromone = np.clip(self.pheromone, 0, 1)

def generate_distance_matrix(n):
    return np.random.rand(n, n)

# Parameters for ACO
n_cities = 10
distance_matrix = generate_distance_matrix(n_cities)
n_ants = 50
n_best = 20
n_iterations = 100
decay = 0.95
alpha = 1
beta = 2

aco = AntColony(distance_matrix, n_ants, n_best, n_iterations, decay, alpha, beta)
best_path, best_distance = aco.run()

print("Best Path:", best_path)
print("Best Distance:", best_distance)
