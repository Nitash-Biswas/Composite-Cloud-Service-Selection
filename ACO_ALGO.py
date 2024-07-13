import tkinter as tk
from tkinter import ttk, Text
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AntColonyOptimization:
    def __init__(self, distance_matrix, n_ants=10, n_iterations=100, evaporation_rate=0.5, alpha=1, beta=2, q=100):
        self.distance_matrix = distance_matrix
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.n_cities = len(distance_matrix)
        self.pheromone_matrix = np.ones_like(distance_matrix) / self.n_cities
        self.best_path = None
        self.best_distance = np.inf
        self.history = []

    def run(self):
        for i in range(self.n_iterations):
            # Boost pheromone levels periodically
            if i % 10 == 0:
                self.pheromone_matrix *= 2.0  # Example boosting factor

            paths = self._generate_paths()
            self._update_pheromones(paths)
            self._update_best_path(paths)
            self.history.append(self.best_distance)

            # Dynamic evaporation rate
            if i % 20 == 0:
                self.evaporation_rate *= 0.8  # Example reduction factor

    def _generate_paths(self):
        paths = []
        for ant in range(self.n_ants):
            visited = [False] * self.n_cities
            path = []
            current_city = np.random.randint(0, self.n_cities)
            visited[current_city] = True
            path.append(current_city)
            for _ in range(self.n_cities - 1):
                probabilities = self._calculate_probabilities(current_city, visited)
                next_city = np.random.choice(np.arange(self.n_cities), p=probabilities)
                path.append(next_city)
                visited[next_city] = True
                current_city = next_city
            path.append(path[0])  # Complete the loop
            paths.append(path)
        return paths

    def _calculate_probabilities(self, current_city, visited):
        pheromones = np.copy(self.pheromone_matrix[current_city])
        pheromones[visited] = 0  # Mask visited cities
        distances = 1 / (self.distance_matrix[current_city] + 1e-10)  # Add a small value to avoid division by zero
        probabilities = (pheromones ** self.alpha) * (distances ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def _update_pheromones(self, paths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for path in paths:
            path_distance = sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(self.n_cities))
            for i in range(self.n_cities):
                self.pheromone_matrix[path[i]][path[i + 1]] += self.q / path_distance

    def _update_best_path(self, paths):
        for path in paths:
            path_distance = sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(self.n_cities))
            if path_distance < self.best_distance:
                self.best_distance = path_distance
                self.best_path = path

def genetic_algorithm(distance_matrix, num_services, num_generations):
    population = np.random.permutation(num_services)
    best_distance = np.inf
    best_path = None
    history = []

    for generation in range(num_generations):
        np.random.shuffle(population)
        # Increase mutation rate occasionally
        if generation % 30 == 0:
            mutation_rate = 0.3  # Example increased mutation rate
        else:
            mutation_rate = 0.1  # Normal mutation rate

        # Random restart occasionally
        if generation % 50 == 0:
            population = np.random.permutation(num_services)

        distance = sum(distance_matrix[population[i]][population[i+1]] for i in range(num_services - 1))
        distance += distance_matrix[population[-1]][population[0]]  # Complete the loop

        if distance < best_distance:
            best_distance = distance
            best_path = list(population)

        history.append(best_distance)

    return history

def artificial_bee_colony_optimization(distance_matrix, num_employed_bees, num_onlooker_bees, num_iterations):
    num_cities = len(distance_matrix)
    best_path = np.random.permutation(num_cities)
    best_distance = np.inf
    history = []

    for _ in range(num_iterations):
        employed_bees = np.random.permutation(num_employed_bees)
        onlooker_bees = np.random.permutation(num_onlooker_bees)

        for bee in employed_bees:
            # Generate a random neighbor solution
            neighbor = np.copy(best_path)
            swap_indices = np.random.choice(num_cities, size=2, replace=False)
            neighbor[swap_indices[0]], neighbor[swap_indices[1]] = neighbor[swap_indices[1]], neighbor[swap_indices[0]]

            # Evaluate the neighbor solution
            neighbor_distance = sum(distance_matrix[neighbor[i]][neighbor[i + 1]] for i in range(num_cities - 1))
            neighbor_distance += distance_matrix[neighbor[-1]][neighbor[0]]  # Complete the loop

            # Update the best solution if the neighbor is better
            if neighbor_distance < best_distance:
                best_distance = neighbor_distance
                best_path = neighbor

        # Update history with the best distance for each iteration
        history.append(best_distance)

    return history


def run_optimizations_plot(root, aco_paths, ga_paths, abc_paths):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(aco_paths, label='Proposed Approach (ACO)', marker='o', linestyle='-', color='b')
    ax.plot(ga_paths, label='Genetic Algorithm', marker='x', linestyle='-', color='r')
    ax.plot(abc_paths, label='Artificial Bee Colony', marker='s', linestyle='-', color='g')
    ax.set_title('Convergence Comparison')
    ax.set_xlabel('Iterations/Generations')
    ax.set_ylabel('Total Lp')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=2)

def generate_plots(num_ants, num_iterations, num_services, alpha, beta, evaporation_rate, num_individuals, num_generations,
                            num_employed_bees, num_onlooker_bees, abc_iterations):
    aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, evaporation_rate, alpha, beta)
    aco.run()
    aco_paths = aco.history

    ga_paths = genetic_algorithm(distance_matrix, num_individuals, num_generations)

    abc_paths = artificial_bee_colony_optimization(distance_matrix, num_employed_bees, num_onlooker_bees, abc_iterations)

    return aco_paths, ga_paths, abc_paths


# Generating a random adjacency matrix for distances between cities
np.random.seed(42)  # For reproducibility
N = 10  # Number of cities
distance_matrix = np.random.rand(N, N)
np.fill_diagonal(distance_matrix, 0)  # Diagonal elements are set to 0

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Optimization Algorithms")

    frame = ttk.Frame(root, padding="5")
    frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

    ttk.Label(frame, text="Number of Ants:").grid(column=0, row=0, sticky=tk.W)
    ttk.Label(frame, text="Number of Iterations:").grid(column=0, row=1, sticky=tk.W)
    ttk.Label(frame, text="Alpha:").grid(column=0, row=2, sticky=tk.W)
    ttk.Label(frame, text="Beta:").grid(column=0, row=3, sticky=tk.W)
    ttk.Label(frame, text="Evaporation Rate:").grid(column=0, row=4, sticky=tk.W)
    ttk.Label(frame, text="Number of Individuals:").grid(column=0, row=5, sticky=tk.W)
    ttk.Label(frame, text="Number of Generations:").grid(column=0, row=6, sticky=tk.W)
    ttk.Label(frame, text="Number of Employed Bees:").grid(column=0, row=7, sticky=tk.W)
    ttk.Label(frame, text="Number of Onlooker Bees:").grid(column=0, row=8, sticky=tk.W)
    ttk.Label(frame, text="Number of ABC Iterations:").grid(column=0, row=9, sticky=tk.W)

    num_ants_entry = ttk.Entry(frame)
    num_iterations_entry = ttk.Entry(frame)
    alpha_entry = ttk.Entry(frame)
    beta_entry = ttk.Entry(frame)
    evaporation_rate_entry = ttk.Entry(frame)
    num_individuals_entry = ttk.Entry(frame)
    num_generations_entry = ttk.Entry(frame)
    num_employed_bees_entry = ttk.Entry(frame)
    num_onlooker_bees_entry = ttk.Entry(frame)
    abc_iterations_entry = ttk.Entry(frame)

    num_ants_entry.grid(column=1, row=0)
    num_iterations_entry.grid(column=1, row=1)
    alpha_entry.grid(column=1, row=2)
    beta_entry.grid(column=1, row=3)
    evaporation_rate_entry.grid(column=1, row=4)
    num_individuals_entry.grid(column=1, row=5)
    num_generations_entry.grid(column=1, row=6)
    num_employed_bees_entry.grid(column=1, row=7)
    num_onlooker_bees_entry.grid(column=1, row=8)
    abc_iterations_entry.grid(column=1, row=9)

    ttk.Button(frame, text="Generate Plots",
        command=lambda: generate_plots_and_show(
            int(num_ants_entry.get()),
            int(num_iterations_entry.get()),
            N,
            float(alpha_entry.get()),
            float(beta_entry.get()),
            float(evaporation_rate_entry.get()),
            int(num_individuals_entry.get()),
            int(num_generations_entry.get()),
            int(num_employed_bees_entry.get()),
            int(num_onlooker_bees_entry.get()),
            int(abc_iterations_entry.get())
        )
    ).grid(column=1, row=10)

    canvas = tk.Canvas(root)
    canvas.grid(row=0, column=2, rowspan=10)

    root.mainloop()