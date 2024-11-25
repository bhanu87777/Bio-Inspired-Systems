import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

def objective_function(x):
    return sum(x**2)

def pso(objective_function, dim, bounds, num_particles, max_iter):
    particles = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best_position = np.copy(particles[0].position)
    global_best_value = float('inf')
    
    for _ in range(max_iter):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)
            
            if value < global_best_value:
                global_best_value = value
                global_best_position = np.copy(particle.position)
        
        for particle in particles:
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            r1, r2 = np.random.random(2)
            particle.velocity = inertia * particle.velocity + cognitive * r1 * (particle.best_position - particle.position) + social * r2 * (global_best_position - particle.position)
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, bounds[0], bounds[1])
    
    return global_best_position, global_best_value

dim = 30
bounds = [-5.12, 5.12]
num_particles = 100
max_iter = 1000

best_position, best_value = pso(objective_function, dim, bounds, num_particles, max_iter)

print("Best Position:", best_position)
print("Best Value:", best_value)
