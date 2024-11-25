import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

class CuckooSearch:
    def __init__(self, objective_function, n_solutions, n_iterations, alpha=0.01, beta=1.5, pa=0.25):
        self.objective_function = objective_function
        self.n_solutions = n_solutions
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.pa = pa
        self.solutions = np.random.uniform(-10, 10, (n_solutions, 2))
        self.fitness = np.array([self.objective_function(solution) for solution in self.solutions])

    def run(self):
        for _ in range(self.n_iterations):
            new_solutions = self.solutions + self.alpha * np.random.randn(self.n_solutions, 2)
            new_fitness = np.array([self.objective_function(solution) for solution in new_solutions])
            for i in range(self.n_solutions):
                if new_fitness[i] < self.fitness[i] and random.random() < self.pa:
                    self.solutions[i] = new_solutions[i]
                    self.fitness[i] = new_fitness[i]
        best_solution_index = np.argmin(self.fitness)
        return self.solutions[best_solution_index], self.fitness[best_solution_index]

def linear_regression_mse(params):
    X, y = generate_data()
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

def generate_data():
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    return X, y

n_solutions = 20
n_iterations = 100
cuckoo = CuckooSearch(objective_function=linear_regression_mse, n_solutions=n_solutions, n_iterations=n_iterations)
best_params, best_fitness = cuckoo.run()

print("Best Parameters:", best_params)
print("Best MSE:", best_fitness)
