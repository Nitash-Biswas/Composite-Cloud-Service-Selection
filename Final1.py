import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Canvas, Text
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import *
from ACO_ALGO import generate_plots, run_optimizations_plot






# Define the CSV file path
csv_file = "dataset.csv"  # Replace with the path to your CSV file

# Initialize an empty 3D matrix (a list of lists of lists)
three_d_matrix = []

# Initialize a dictionary to store row names and a list to store column names
row_names = {}
col_names = []

# Open the CSV file and read its contents
with open(csv_file, newline='') as file:
    csv_reader = csv.reader(file)

    # Read the header row to get column names
    header = next(csv_reader)

    # Determine the index of the "Name" column
    name_column_index = header.index("Name")

    # Store the column names in the col_names list
    for i, col_name in enumerate(header):
        if i != name_column_index:
            col_names.append(col_name)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Store the row name in the dictionary
        row_names[row[0]] = len(three_d_matrix)

        # Initialize a 2D matrix (a list of lists) for each row
        row_matrix = []

        # Convert each cell's content (string representing a list) into a list
        for cell in row[1:]:
            cell_data = ast.literal_eval(cell)  # Convert the string to a list
            row_matrix.append(cell_data)

        three_d_matrix.append(row_matrix)

# Print the 3D matrix
print("Original Matrix")
for row_matrix in three_d_matrix:
    for i, row_list in enumerate(row_matrix):
        print(row_list, end="")
        if i < len(row_matrix) - 1:
            print(" , ", end="")
    print()  # Move to a new line
print()



# Calculate A(max) and A(min) for each column and each element index ----------------------------------------------------------------------
n = len(three_d_matrix)  # Number of rows (0-1)
m = len(col_names)  # Number of columns (0-1)
num_elements = len(three_d_matrix[0][0])  # Number of elements (0-3)

a_max_values = []
a_min_values = []

for col_index in range(m):
    for el_index in range(num_elements):
        # Extract values from the 3D matrix for the current column and element index
        column_values = [three_d_matrix[row_index][col_index][el_index] for row_index in range(n) if three_d_matrix[row_index][col_index][el_index] != 0]


        # Calculate A(max) and A(min)
        a_max = max(column_values)
        a_min = min(column_values)

        a_max_values.append(a_max)
        a_min_values.append(a_min)

#LIST OF MAX VALUES ----------------------------------------------------------------------------------------------------------------------
# Group values in a_max_values by elements
a_max_grouped = []
for el_index in range(num_elements):
    element_values = a_max_values[el_index::num_elements]
    a_max_grouped.append(element_values)


# Transpose the 2D list to get the desired format
a_max_2d_list = list(map(list, zip(*a_max_grouped)))

# Print the A(max) values as separate sublists
print(f"A(max) values for all {len(three_d_matrix)} services:")
for element_values in a_max_2d_list:
    print(element_values)
print()


#LIST OF MIN VALUES --------------------------------------------------------------------------------------------------------------------
# Group values in a_max_values by elements
a_min_grouped = []
for el_index in range(num_elements):
    element_values = a_min_values[el_index::num_elements]
    a_min_grouped.append(element_values)


# Transpose the 2D list to get the desired format
a_min_2d_list = list(map(list, zip(*a_min_grouped)))

# Print the A(max) values as separate sublists
print(f"A(min) values for all {len(three_d_matrix)} services:")
for element_values in a_min_2d_list:
    print(element_values)
print()


# Initialize a new 3D matrix for normalized values (Matrix_Normal) --------------------------------------------------------------------------
matrix_normal = []

# Normalize each element in the original matrix and store it in Matrix_Normal
for row_index in range(n):
    row_normal = []  # Initialize a new row in Matrix_Normal
    for col_index in range(m):
        el_normal = []  # Initialize a new element in the row
        for el_index in range(num_elements):
            # Calculate the normalized value using the given formulas
            a_i = three_d_matrix[row_index][col_index][el_index]
            a_max = a_max_values[col_index * num_elements + el_index]
            a_min = a_min_values[col_index * num_elements + el_index]

            if a_max == a_min:
                normalized_value = 1  # Set to 1 if A(max) = A(min)
            else:
                if el_index < 2:
                    normalized_value = (a_max - a_i) / (a_max - a_min)  # min_normal
                else:
                    normalized_value = (a_i - a_min) / (a_max - a_min)  # max_normal

            # Ensure the value is clamped within the range of 0 to 1
            normalized_value = max(0, min(1, normalized_value))

            # Round the value to two decimal places
            normalized_value = round(normalized_value, 2)

            el_normal.append(normalized_value)

        row_normal.append(el_normal)  # Add the element to the row
    matrix_normal.append(row_normal)  # Add the row to Matrix_Normal




# Print the Matrix_Normal
print("Normalised Matrix")
for row in matrix_normal:
    for i, el in enumerate(row):
        print(el, end="")
        if i < len(row) - 1:
            print(" , ", end="")
    print()  # Move to a new line
print()





# Creating UTILITY MATRIX ------------------------------------------------------------------------------------------------
utility_matrix = []

# Calculate and store the sums of values from the Matrix_Normal
for row in matrix_normal:
    row_sums = []  # Initialize a new row for sums
    for el in row:
        # Calculate the sum of the values in the list and append it to the row
        el_sum = sum(el)
        row_sums.append(round(el_sum,2))
    utility_matrix.append(row_sums)  # Add the row of sums to the new matrix

# Print the new matrix containing sums
print("Utility Matrix of Sums:")
for element_values in utility_matrix:

    print(element_values)











#IDEAL F0 FOR ALL 4 QoS ATTRIBUTES: --------------------------------------------------------------------------------------------------------
# Calculate the sum of values from a_min_grouped for the first two values
sum_first_two = [sum(row[i] for row in a_min_2d_list) for i in range(2)]

# Calculate the product of values from a_max_grouped for the last two values
product_last_two = [1.0] * 2  # Assuming there are 2 values in each sublist
for values in zip(*a_max_grouped[2:]):
    product_last_two = [x * y for x, y in zip(product_last_two, values)]
    product_last_two = [round(value, 3) for value in product_last_two]

# Combine the calculated values into a new list
f0_list = sum_first_two + product_last_two

# Print the F0 list
print("Ideal F0 for all 4 QoS attributes :")
print(f0_list,"\n")




# ACCESS ----------------------------------------------------------------------------------------------------------------------------------

#Greedy algo for first iteration
# Initialize a list to store the row indices of the highest values in each column
highest_value_row_indices = []

# Iterate through the columns
for col in range(len(utility_matrix[0])):
    max_value = float('-inf')  # Initialize with negative infinity to find the maximum
    max_row_index = -1  # Initialize with -1 to keep track of the row index with the max value
    for row_index, row in enumerate(utility_matrix):
        if row[col] > max_value:
            max_value = row[col]
            max_row_index = row_index
    highest_value_row_indices.append(max_row_index + 1)  # Add 1 to convert to 1-based indexing

# Print the list of row indices with the highest values in each column
formatted_labels = [f"C{index}" for index in highest_value_row_indices]
print(f"Best set (GREEDY ALGORITHM): {' -> '.join(formatted_labels)}")
print()












#ANT COLONY OPTIMISATION ----------------------------------------------------------------------------
import numpy as np
import random

def calculate_lp(f0_values, rt_values, c_values, a_values, m_values):
    # Calculate Lp using the provided formula
    f0rt, f0c, f0a, f0m = f0_values
    frt = sum(rt_values)
    fc = sum(c_values)

    # Replace 0 values with 1 before taking the product
    a_values = [1 if value == 0 else value for value in a_values]
    m_values = [1 if value == 0 else value for value in m_values]
    fa = np.prod(a_values)
    fm = np.prod(m_values)

    lp = np.sqrt(((f0rt - frt) / f0rt) ** 2 + ((f0c - fc) / f0c) ** 2 + ((f0a - fa) / fa) ** 2 + ((f0m - fm) / fm) ** 2)

    return lp


import random

def ant_colony_optimization(three_d_matrix, utility_matrix, f0_list, num_ants, num_services, alpha, beta, num_iterations, evaporation_rate):
    num_clouds = len(three_d_matrix)
    num_services = len(three_d_matrix[0])

    paths = []  # Store all paths found

    for _ in range(num_iterations):
        pheromone = np.full((num_clouds, num_clouds), 1.0)  # Initialize pheromone levels with a smaller value

        for _ in range(num_ants):
            current_service = 0
            path = []
            f0_values = f0_list
            rt_values = [0.0] * num_services
            c_values = [0.0] * num_services
            a_values = [1.0] * num_services
            m_values = [1.0] * num_services

            while current_service < num_services:
                # Calculate Lp for each cloud
                lp_values = []
                for i in range(num_clouds):
                    rt_values[current_service] = three_d_matrix[i][current_service][0]
                    c_values[current_service] = three_d_matrix[i][current_service][1]
                    a_values[current_service] = three_d_matrix[i][current_service][2]
                    m_values[current_service] = three_d_matrix[i][current_service][3]
                    lp = calculate_lp(f0_values, rt_values, c_values, a_values, m_values)
                    lp_values.append(lp)

                if current_service == 0:
                    # Choose the first cloud randomly without pheromone information
                    next_cloud = random.randint(0, num_clouds - 1)
                else:
                    # Calculate probabilities for choosing the next cloud
                    probabilities = [(utility_matrix[i][current_service] ** beta) * (pheromone[path[-1]][i]) ** alpha for i in range(num_clouds)]
                    total_probability = sum(probabilities)
                    probabilities = [p / total_probability for p in probabilities]

                    # Choose the next cloud based on probabilities
                    next_cloud = random.choices(range(num_clouds), probabilities)[0]

                path.append(next_cloud)
                current_service += 1
                f0_values = f0_list  # Reset f0_values for the next service

            paths.append(path)

            # Deposit pheromone on the path edges based on the quality of the path (1/Lp)
            path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)
            pheromone_deposit = 1.0 / (1.0 + path_lp)
            for i in range(num_services - 1):
                pheromone[path[i]][path[i+1]] += pheromone_deposit

        # Evaporate pheromone on all edges
        pheromone *= (1.0 - evaporation_rate)

    return paths






def count_paths(paths):
    path_counts = {}
    total_paths = len(paths)

    for path in paths:
        # Convert the path list to a tuple for dictionary key
        path_tuple = tuple(path)
        if path_tuple in path_counts:
            path_counts[path_tuple] += 1
        else:
            path_counts[path_tuple] = 1

    return path_counts

def calculate_percentages(path_counts, total_paths):
    path_percentages = {path: (count / total_paths) * 100.0 for path, count in path_counts.items()}
    return path_percentages


#GENETIC ALGO OPTIMISATION ----------------------------------------------------------------------------

def genetic_algorithm(three_d_matrix, utility_matrix, f0_list,beta, num_individuals, num_services, num_generations):
    num_clouds = len(three_d_matrix)
    num_services = len(three_d_matrix[0])

    paths = []  # Store all paths found

    for _ in range(num_generations):
        individuals = []

        # Initialize a population of individuals
        for _ in range(num_individuals):
            individual = []
            f0_values = f0_list.copy()
            rt_values = [0.0] * num_services
            c_values = [0.0] * num_services
            a_values = [1.0] * num_services
            m_values = [1.0] * num_services

            for _ in range(num_services):
                # Calculate Lp for each cloud
                lp_values = []
                for i in range(num_clouds):
                    rt_values[_] = three_d_matrix[i][_][0]
                    c_values[_] = three_d_matrix[i][_][1]
                    a_values[_] = three_d_matrix[i][_][2]
                    m_values[_] = three_d_matrix[i][_][3]
                    lp = calculate_lp(f0_values, rt_values, c_values, a_values, m_values)
                    lp_values.append(lp)

                if _ == 0:
                    # Choose the first cloud randomly without pheromone information
                    next_cloud = random.randint(0, num_clouds - 1)
                else:
                    # Calculate probabilities for choosing the next cloud
                    probabilities = [(utility_matrix[i][_] ** beta) for i in range(num_clouds)]
                    total_probability = sum(probabilities)
                    probabilities = [p / total_probability for p in probabilities]

                    # Choose the next cloud based on probabilities
                    next_cloud = random.choices(range(num_clouds), probabilities)[0]

                individual.append(next_cloud)
                f0_values = f0_list  # Reset f0_values for the next service

            individuals.append(individual)

        # Evaluate individuals and select the best ones
        fitness_scores = []

        for individual in individuals:
            cloud_indices = list(individual)
            rt_values = []
            c_values = []
            a_values = []
            m_values = []

            for cloud_index, service_index in enumerate(cloud_indices):
                rt, c, a, m = three_d_matrix[service_index][cloud_index]
                rt_values.append(rt)
                c_values.append(c)
                a_values.append(a)
                m_values.append(m)

            path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)
            fitness_scores.append(1 / (1 + path_lp))  # Invert LP as fitness

        # Select the top individuals based on fitness
        selected_indices = np.argsort(fitness_scores)[-num_individuals:]

        # Create a new generation of individuals through crossover and mutation
        new_individuals = []

        for _ in range(num_individuals):
            parent1 = individuals[random.choice(selected_indices)]
            parent2 = individuals[random.choice(selected_indices)]
            crossover_point = random.randint(1, num_services - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            # Apply mutation with a small probability
            mutation_prob = 0.1
            if random.random() < mutation_prob:
                mutation_point = random.randint(0, num_services - 1)
                child[mutation_point] = random.randint(0, num_clouds - 1)

            new_individuals.append(child)

        individuals = new_individuals
        best_individual = individuals[np.argmax(fitness_scores)]
        paths.append(best_individual)

    return paths









def run_optimizations(paths,ga_paths,num_iterations,num_generations):
    # Clear previous results
    result_text.delete(1.0, tk.END)

    path_counts = count_paths(paths)
    total_paths = len(paths)
    path_percentages = calculate_percentages(path_counts, total_paths)
    sorted_paths = sorted(path_percentages.items(), key=lambda x: x[1], reverse=True)

    # Get the first path in the sorted list
    first_path, _ = sorted_paths[0]
    # Extract the cloud indices from the path
    cloud_indices = list(first_path)

    # Initialize lists to store the values for each cloud in the path
    rt_values = []
    c_values = []
    a_values = []
    m_values = []

    # Loop through the cloud indices and extract the corresponding values
    for cloud_index, service_index in enumerate(cloud_indices):
        rt, c, a, m = three_d_matrix[service_index][cloud_index]
        rt_values.append(rt)
        c_values.append(c)
        a_values.append(a)
        m_values.append(m)



    # Calculate the Lp value for the first path
    first_path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)

    # Update the Text widget to display results
    result_text.insert(tk.END, "Best Set (ACO):\n")
    formatted_labels = [f"C{index + 1}" for index in first_path]  # Add 1 to convert to 1-based indexing
    result_text.insert(tk.END, " -> ".join(formatted_labels))
    result_text.insert(tk.END, "\n")

    # Print the Lp value for the first path
    # result_text.insert(tk.END, f"\nLp Value of the Path: {first_path_lp:.2f}\n")

    # Find the top 10 unique paths and their occurrences
    unique_paths = set(tuple(path) for path, _ in sorted_paths)
    path_occurrences = {path: paths.count(list(path)) for path in unique_paths}
    sorted_paths_by_occurrences = sorted(path_occurrences.items(), key=lambda x: x[1], reverse=True)[0:10]

    # Calculate the percentage of each top path out of all paths
    total_paths = len(paths)
    path_percentages = [(path, occurrences / total_paths * 100) for path, occurrences in sorted_paths_by_occurrences]

    # Print the list of paths with highest percentages
    result_text.insert(tk.END, "Top paths:\n")
    lp_values_iterations = [0.0] * num_iterations




    for i, (path, percentage) in enumerate(path_percentages):
        # Extract the cloud indices from the path
        cloud_indices = list(path)

        # Initialize lists to store the values for each cloud in the path
        rt_values = []
        c_values = []
        a_values = []
        m_values = []

        # Loop through the cloud indices and extract the corresponding values
        for cloud_index, service_index in enumerate(cloud_indices):
            rt, c, a, m = three_d_matrix[service_index][cloud_index]
            rt_values.append(rt)
            c_values.append(c)
            a_values.append(a)
            m_values.append(m)

        # Calculate the Lp value for the path
        path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)
        lp_values_iterations[i] = path_lp

        # Print the path, percentage, and Lp value
        result_text.insert(tk.END, f"Path: {path}\n")
    result_text.insert(tk.END, "\n")



# GA ALGORITHM
    gpath_counts = count_paths(ga_paths)
    total_gpaths = len(ga_paths)

    path_percentages = calculate_percentages(gpath_counts, total_gpaths)
    sorted_gpaths = sorted(path_percentages.items(), key=lambda x: x[1], reverse=True)

    # Get the first path in the sorted list
    first_path, _ = sorted_gpaths[0]

    # Extract the cloud indices from the path
    cloud_indices = list(first_path)

    # Initialize lists to store the values for each cloud in the path
    rt_values = []
    c_values = []
    a_values = []
    m_values = []

    # Loop through the cloud indices and extract the corresponding values
    for cloud_index, service_index in enumerate(cloud_indices):
        rt, c, a, m = three_d_matrix[service_index][cloud_index]
        rt_values.append(rt)
        c_values.append(c)
        a_values.append(a)
        m_values.append(m)


    # Print the first path as an example
    result_text.insert(tk.END, "Best Set (GA):\n")
    formatted_labels = [f"C{index + 1}" for index in first_path]  # Add 1 to convert to 1-based indexing
    result_text.insert(tk.END, " -> ".join(formatted_labels))
    # Calculate the Lp value for the first path
    first_path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)
    # result_text.insert(tk.END, f"\nLp Value of the Path: {first_path_lp:.2f}\n")
    result_text.insert(tk.END, "\n")


    # Find the top 10 unique paths and their occurrences
    unique_paths = set(tuple(path) for path, _ in sorted_gpaths)
    path_occurrences = {path: ga_paths.count(list(path)) for path in unique_paths}
    sorted_paths_by_occurrences = sorted(path_occurrences.items(), key=lambda x: x[1], reverse=True)[0:10]

    # Calculate the percentage of each top path out of all paths
    total_gpaths = len(ga_paths)
    path_percentages = [(path, occurrences / total_gpaths * 100) for path, occurrences in sorted_paths_by_occurrences]

    # Print the list of paths with highest percentages
    result_text.insert(tk.END, "Top paths:\n")
    ga_lp_values_iterations = [0.0] * num_generations

    for i, (path, percentage) in enumerate(path_percentages):
        # Extract the cloud indices from the path
        cloud_indices = list(path)

        # Initialize lists to store the values for each cloud in the path
        rt_values = []
        c_values = []
        a_values = []
        m_values = []

        # Loop through the cloud indices and extract the corresponding values
        for cloud_index, service_index in enumerate(cloud_indices):
            rt, c, a, m = three_d_matrix[service_index][cloud_index]
            rt_values.append(rt)
            c_values.append(c)
            a_values.append(a)
            m_values.append(m)

        # Calculate the Lp value for the path
        path_lp = calculate_lp(f0_list, rt_values, c_values, a_values, m_values)
        ga_lp_values_iterations[i] = path_lp

        # Print the path, percentage, and Lp value
        result_text.insert(tk.END, f"Path: {path}\n")

#PLOTTING THE GRAPH ----------------------------------------------------------------------------











def generate_text(num_ants,num_iterations,num_services,alpha,beta,evaporation_rate,num_individuals,num_generations):


    # Call ant_colony_optimization() and genetic_algorithm() with the provided parameters
    aco_paths = ant_colony_optimization(three_d_matrix, utility_matrix, f0_list, num_ants, num_services, alpha, beta, num_iterations, evaporation_rate)
    ga__paths = genetic_algorithm(three_d_matrix, utility_matrix, f0_list,beta, num_individuals, num_services, num_generations)
    run_optimizations(aco_paths,ga__paths,num_iterations,num_generations)

def generate_plot_text():
    num_ants = int(num_ants_entry.get())
    num_iterations = int(num_iterations_entry.get())
    num_services = len(three_d_matrix[0])
    alpha = float(alpha_entry.get())
    beta = float(beta_entry.get())
    evaporation_rate = float(evaporation_rate_entry.get())
    num_individuals = int(num_individuals_entry.get())
    num_generations = int(num_iterations_entry.get())
    num_employed_bees = int(num_employed_bees_entry.get())
    num_onlooker_bees = int(num_onlooker_bees_entry.get())
    abc_iterations = int(num_iterations_entry.get())

    # generate_text(num_ants,num_iterations,num_services,alpha,beta,evaporation_rate,num_individuals,num_generations)
    aco_paths, ga_paths,abc_paths = generate_plots(num_ants, num_iterations, num_services, alpha, beta, evaporation_rate, num_individuals, num_generations, num_employed_bees, num_onlooker_bees, abc_iterations)
    run_optimizations_plot(root, aco_paths, ga_paths, abc_paths)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Optimization Algorithms")

# Create and set up the input fields and labels
frame = ttk.Frame(root, padding="5")
frame.grid(column=0, row=0, sticky=(N, W, E, S))

# Labels
ttk.Label(frame, text="Number of Ants:").grid(column=0, row=0, sticky=W)
ttk.Label(frame, text="Number of Iterations:").grid(column=0, row=1, sticky=W)
ttk.Label(frame, text="Alpha:").grid(column=0, row=2, sticky=W)
ttk.Label(frame, text="Beta:").grid(column=0, row=3, sticky=W)
ttk.Label(frame, text="Evaporation Rate:").grid(column=0, row=4, sticky=W)
ttk.Label(frame, text="Number of Individuals:").grid(column=0, row=5, sticky=W)
# ttk.Label(frame, text="Number of Generations:").grid(column=0, row=6, sticky=W)
ttk.Label(frame, text="Number of Employed Bees").grid(column=0, row=7, sticky=W)
ttk.Label(frame, text="Number of Onlooker Bees:").grid(column=0, row=8, sticky=W)
# ttk.Label(frame, text="Number of Iterations:").grid(column=0, row=9, sticky=W)

# Entry fields
num_ants_entry = ttk.Entry(frame)
num_iterations_entry = ttk.Entry(frame)
alpha_entry = ttk.Entry(frame)
beta_entry = ttk.Entry(frame)
evaporation_rate_entry = ttk.Entry(frame)
num_individuals_entry = ttk.Entry(frame)
# num_generations_entry = ttk.Entry(frame)
num_employed_bees_entry = ttk.Entry(frame)
num_onlooker_bees_entry = ttk.Entry(frame)
# abc_iterations_entry = ttk.Entry(frame)

# Place entry fields
num_ants_entry.grid(column=1, row=0)
num_iterations_entry.grid(column=1, row=1)
alpha_entry.grid(column=1, row=2)
beta_entry.grid(column=1, row=3)
evaporation_rate_entry.grid(column=1, row=4)
num_individuals_entry.grid(column=1, row=5)
# num_generations_entry.grid(column=1, row=6)
num_employed_bees_entry.grid(column=1, row=7)
num_onlooker_bees_entry.grid(column=1, row=8)
# abc_iterations_entry.grid(column=1, row=9)

# Create and set up the "Generate Plots" button
ttk.Button(frame, text="Generate Plots", command=generate_plot_text).grid(column=1, row=10)


# Create a Text widget for displaying results
# result_text = Text(frame, width=57, height=2)
# result_text.grid(row=9, columnspan=2)

# Create a GUI loop
root.mainloop()







