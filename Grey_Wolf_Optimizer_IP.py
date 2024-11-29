import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def objective_function(params, image):
    alpha = params[0]
    beta = params[1]
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    edges = cv2.Canny(enhanced_image, 100, 200)
    fitness = np.sum(edges)
    return fitness, enhanced_image

def grey_wolf_optimizer(image, dim, lb, ub, max_iter, population_size):
    Alpha_pos = np.zeros(dim)
    Beta_pos = np.zeros(dim)
    Delta_pos = np.zeros(dim)
    Alpha_score = float('-inf')
    Beta_score = float('-inf')
    Delta_score = float('-inf')
    wolves = np.random.uniform(lb, ub, (population_size, dim))
    best_image = None
    
    for t in range(max_iter):
        for i in range(population_size):
            fitness, enhanced_image = objective_function(wolves[i], image)
            if fitness > Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                Alpha_pos = wolves[i].copy()
                best_image = enhanced_image.copy()
            elif fitness > Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = wolves[i].copy()
            elif fitness > Delta_score:
                Delta_score = fitness
                Delta_pos = wolves[i].copy()
        
        a = 2 - t * (2 / max_iter)
        for i in range(population_size):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - wolves[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                r1 = random.random()
                r2 = random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - wolves[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                r1 = random.random()
                r2 = random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - wolves[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                wolves[i, j] = (X1 + X2 + X3) / 3.0
                wolves[i, j] = np.clip(wolves[i, j], lb[j], ub[j])
        print(f"Iteration {t+1}/{max_iter}, Best Fitness: {Alpha_score}")
    
    return Alpha_pos, Alpha_score, best_image

image_path = "Untitled.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

dim = 2
lb = [1.0, 0.0]
ub = [3.0, 100.0]
max_iter = 20
population_size = 10

best_params, best_fitness, best_image = grey_wolf_optimizer(image, dim, lb, ub, max_iter, population_size)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Enhanced Image\nAlpha: {best_params[0]:.2f}, Beta: {best_params[1]:.2f}")
plt.imshow(best_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
