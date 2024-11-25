import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GreyWolfOptimizer:
    def __init__(self, objective_function, n_wolves, n_iterations, lb, ub):
        self.objective_function = objective_function
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.lb = lb
        self.ub = ub
        self.alpha = np.inf
        self.beta = np.inf
        self.delta = np.inf
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_wolves, 2))

    def update_position(self, A, C, D, X, i):
        return X[i] + A[i] * D[i]

    def run(self):
        for _ in range(self.n_iterations):
            for i in range(self.n_wolves):
                fitness = self.objective_function(self.positions[i])
                if fitness < self.alpha:
                    self.delta = self.beta
                    self.beta = self.alpha
                    self.alpha = fitness
                    self.alpha_pos = self.positions[i]
                elif fitness < self.beta:
                    self.delta = self.beta
                    self.beta = fitness
                    self.beta_pos = self.positions[i]
                elif fitness < self.delta:
                    self.delta = fitness
                    self.delta_pos = self.positions[i]

            A = 2 * np.random.rand(self.n_wolves) - 1
            C = 2 * np.random.rand(self.n_wolves, 2)
            D = np.abs(C * self.alpha_pos - self.positions)

            for i in range(self.n_wolves):
                self.positions[i] = self.update_position(A, C, D, self.positions, i)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.alpha_pos, self.alpha

def objective_function(params):
    max_depth, min_samples_split = int(params[0]), int(params[1])
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return -accuracy_score(y_test, y_pred)

X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lb = [1, 2]  
ub = [20, 10]  
n_wolves = 20
n_iterations = 50

gwo = GreyWolfOptimizer(objective_function, n_wolves, n_iterations, lb, ub)
best_params, best_fitness = gwo.run()

print("Best Parameters:", best_params)
print("Best Accuracy:", -best_fitness)
