import numpy as np
import matplotlib.pyplot as plt
import random
 
np.random.seed(42)
random.seed(42)
 
# 1. Objective Function (Fitness)
def fitness_function(x):
    return 1 + (np.exp(x) * np.sin(np.pi * x) - 1) / x
 
# 2. Population Initialization
def initialize_population(pop_size, bounds):
    return np.random.uniform(bounds[0], bounds[1], pop_size)
 
# 3. Selection Method (Tournament Selection)
def tournament_selection(population, fitnesses, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[np.argmax(fitnesses[selected_indices])]
    return population[best_index]
 
# 3.1 Alternative Selection Method (Roulette Wheel Selection)
def roulette_wheel_selection(population, fitnesses):
    # Determine the lowest fitness to handle negative or near-zero values safely
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        adjusted_fitnesses = fitnesses - min_fitness + 0.0001
    else:
        adjusted_fitnesses = fitnesses
 
    total_fitness = np.sum(adjusted_fitnesses)
    
    if total_fitness == 0:
        return random.choice(population)
        
    selection_probs = adjusted_fitnesses / total_fitness
    selected_index = np.random.choice(len(population), p=selection_probs)
    return population[selected_index]
 
# 4. Crossover Operator (Arithmetic Crossover)
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    return parent1, parent2
 
# 5. Mutation Operator (Gaussian Mutation)
def mutate(child, mutation_rate, bounds, sigma=0.1):
    if random.random() < mutation_rate:
        child += random.gauss(0, sigma)
        # Ensure the mutated child stays within the specified interval
        child = max(bounds[0], min(child, bounds[1]))
    return child
 
#6. Main Genetic Algorithm Engine
def run_genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, bounds):
    population = initialize_population(pop_size, bounds)
    
    best_fitness_history = []
    avg_fitness_history = []
    
    global_best_solution = None
    global_best_fitness = -float('inf')
 
    for generation in range(generations):
        # Evaluate fitness for all chromosomes in the population
        fitnesses = np.array([fitness_function(chromosome) for chromosome in population])
        
        # Track the best solution in the current generation
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > global_best_fitness:
            global_best_fitness = fitnesses[current_best_idx]
            global_best_solution = population[current_best_idx]
            
        # Record statistics for the convergence plot
        best_fitness_history.append(global_best_fitness)
        avg_fitness_history.append(np.mean(fitnesses))
        
        new_population = []
        
        # Elitism: Automatically pass the absolute best chromosome to the next generation
        new_population.append(population[current_best_idx])
        
        # Generate the rest of the new population
        while len(new_population) < pop_size:
            # Select parents
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
 
            # Select parents with Roulette Wheel (Stage II Test)
            #p1 = roulette_wheel_selection(population, fitnesses)
            #p2 = roulette_wheel_selection(population, fitnesses)
            
            # Crossover
            c1, c2 = crossover(p1, p2, crossover_rate)
            
            # Mutation
            c1 = mutate(c1, mutation_rate, bounds)
            c2 = mutate(c2, mutation_rate, bounds)
            
            new_population.extend([c1, c2])
            
        # Ensure the population size remains strictly constant
        population = np.array(new_population[:pop_size])
 
    return global_best_solution, global_best_fitness, best_fitness_history, avg_fitness_history
 
# --- EXPERIMENTAL SETUP PARAMETERS ---
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
BOUNDS = [1.5, 3.5]  
 
# Execute the Algorithm
best_x, best_fit, best_hist, avg_hist = run_genetic_algorithm(
    pop_size=POPULATION_SIZE,
    generations=GENERATIONS,
    crossover_rate=CROSSOVER_PROBABILITY,
    mutation_rate=MUTATION_PROBABILITY,
    bounds=BOUNDS
)
 
# Output Results
print("--- Genetic Algorithm Results ---")
print(f"Optimal x value found : {best_x:.5f}")
print(f"Maximum Fitness value : {best_fit:.5f}")
print("---------------------------------")
 
# Plot the Convergence Graph
plt.figure(figsize=(10, 6))
plt.plot(best_hist, label='Best Fitness', color='green', linewidth=2)
plt.plot(avg_hist, label='Average Fitness', color='orange', linestyle='--')
plt.title('Genetic Algorithm Convergence')
plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.legend()
plt.grid(True)
plt.show()
 
 
# =====================================================================
# STAGE II: 2D SPACE EXTENSION
# =====================================================================
 
# 1. 2D Fitness Function
def fitness_function_2d(chromosome):
    x, y = chromosome[0], chromosome[1]
    return 20 - (x**2 + y**2)
 
# 2. 2D Population Initialization
def initialize_population_2d(pop_size, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
 
# 3. 2D Mutation
def mutate_2d(child, mutation_rate, bounds, sigma=0.1):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += random.gauss(0, sigma)
            child[i] = max(bounds[0], min(child[i], bounds[1]))
    return child
 
# 4. 2D Genetic Algorithm Engine
def run_genetic_algorithm_2d(pop_size, generations, crossover_rate, mutation_rate, bounds):
    population = initialize_population_2d(pop_size, bounds)
    global_best_solution = None
    global_best_fitness = -float('inf')
 
    for generation in range(generations):
        fitnesses = np.array([fitness_function_2d(chromosome) for chromosome in population])
        
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > global_best_fitness:
            global_best_fitness = fitnesses[current_best_idx]
            global_best_solution = population[current_best_idx].copy()
            
        new_population = [population[current_best_idx]] # Elitism
        
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            
            c1, c2 = crossover(p1.copy(), p2.copy(), crossover_rate)
            
            c1 = mutate_2d(c1, mutation_rate, bounds)
            c2 = mutate_2d(c2, mutation_rate, bounds)
            
            new_population.extend([c1, c2])
            
        population = np.array(new_population[:pop_size])
 
    return global_best_solution, global_best_fitness
 
# --- EXECUTE 2D TEST ---
BOUNDS_2D = [-5.0, 5.0]
best_2d_sol, best_2d_fit = run_genetic_algorithm_2d(
    pop_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    bounds=BOUNDS_2D
)
 
print("\n--- Stage II: 2D Space Optimization Results ---")
print(f"Optimal [x, y] coordinates : [{best_2d_sol[0]:.5f}, {best_2d_sol[1]:.5f}]")
print(f"Maximum Fitness value      : {best_2d_fit:.5f}")
print("Expected value             : 20.00000 at [0.00000, 0.00000]")
print("-----------------------------------------------")