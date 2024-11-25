import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.special import gamma

class CuckooSearch:
    def __init__(self, objective_function, n_cuckoos, n_iterations, lb, ub):
        self.objective_function = objective_function
        self.n_cuckoos = n_cuckoos
        self.n_iterations = n_iterations
        self.lb = lb
        self.ub = ub
        self.best_solution = None
        self.best_score = np.inf
        self.positions = np.random.uniform(self.lb, self.ub, self.n_cuckoos)

    def levy_flight(self, step_size):
        beta = 3 / 2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.n_cuckoos)
        v = np.random.normal(0, 1, size=self.n_cuckoos)
        step = u / (np.abs(v) ** (1 / beta)) * step_size
        return step

    def update_position(self, position, best_position, step_size):
        new_position = position + step_size * (position - best_position)
        return np.clip(new_position, self.lb, self.ub)

    def run(self):
        for _ in range(self.n_iterations):
            for i in range(self.n_cuckoos):
                fitness = self.objective_function(self.positions[i])

                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_solution = self.positions[i]

            step_size = 0.01
            step = self.levy_flight(step_size)

            for i in range(self.n_cuckoos):
                self.positions[i] = self.update_position(self.positions[i], self.best_solution, step[i])

        return self.best_solution, self.best_score

def objective_function(threshold):
    ret, thresholded_image = cv2.threshold(image, int(threshold), 255, cv2.THRESH_BINARY)
    
    non_zero_count = np.count_nonzero(thresholded_image)
    return -non_zero_count

image = cv2.imread('Untitled.png', cv2.IMREAD_GRAYSCALE)

lb = 0 
ub = 255 
n_cuckoos = 20 
n_iterations = 100 

cs = CuckooSearch(objective_function, n_cuckoos, n_iterations, lb, ub)
best_threshold, best_score = cs.run()

ret, thresholded_image = cv2.threshold(image, int(best_threshold), 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Thresholded Image (Threshold = {best_threshold})")
plt.imshow(thresholded_image, cmap='gray')
plt.show()

print(f"Best Threshold: {best_threshold}")
