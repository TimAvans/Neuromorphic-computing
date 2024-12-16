'''
I want to try supervised learning since the hand tuned params and functions did not seem to want to work
Multiple matrices for training
'''

import nengo
import numpy as np
import matplotlib.pyplot as plt

# Training Data: Known distance matrices
train_distances = [
    np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]),  # Matrix 1
    np.array([[0, 8, 12, 18], [8, 0, 25, 15], [12, 25, 0, 20], [18, 15, 20, 0]]),   # Matrix 2
    np.array([[0, 5, 10, 15], [5, 0, 20, 10], [10, 20, 0, 25], [15, 10, 25, 0]])    # Matrix 3
]

# Prepare a new unseen matrix for testing
test_matrix = np.array([[0, 8, 14, 18],
                        [8, 0, 40, 22],
                        [14, 40, 0, 28],
                        [18, 22, 28, 0]])

# Normalize all matrices
def normalize_matrix(matrix):
    inv = 1 / (matrix + 1e-5)  # Invert to get higher weights for shorter distances
    return inv / inv.max()

train_matrices_norm = [normalize_matrix(m) for m in train_distances]

# Number of cities
n_cities = len(test_matrix)

train_targets = [
    {0: [0, 1, 0, 0], 1: [0, 0, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 0, 0]},  # Targets for Matrix 1
    {0: [0, 1, 0, 0], 1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [0, 0, 0, 0]},  # Targets for Matrix 2
    {0: [0, 1, 0, 0], 1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [0, 0, 0, 0]}   # Targets for Matrix 3
]

# Global variables to track training state
current_matrix_index = 0  # Tracks which matrix we're training on
current_city = 0          # Tracks the current city

# Dynamic input function: supplies the current row of the active matrix
def dynamic_input(t):
    global current_city, current_matrix_index
    return train_matrices_norm[current_matrix_index][current_city]

# Dynamic target function: supplies the correct target for the active city
def dynamic_target(t):
    global current_city, current_matrix_index
    row = train_matrices_norm[current_matrix_index][current_city]
    next_city = np.argmin(row + (np.eye(n_cities)[current_city] * 1000))  # Ignore current city
    target = np.eye(n_cities)[next_city]
    return target

# Nengo model training
with nengo.Network() as model:
    current = nengo.Ensemble(500, dimensions=n_cities)
    next_city = nengo.Ensemble(500, dimensions=n_cities)

    # Input and target nodes
    input_node = nengo.Node(output=dynamic_input)
    target_node = nengo.Node(output=dynamic_target)

    # Connections
    nengo.Connection(input_node, current)  # Input to the 'current' ensemble
    conn = nengo.Connection(
        current, next_city, 
        transform=np.zeros((n_cities, n_cities)),  # Start with zero weights
        learning_rule_type=nengo.PES()  # Enable PES learning
    )
    nengo.Connection(target_node, conn.learning_rule)  # Target guides learning

    # Probes
    next_city_probe = nengo.Probe(next_city, synapse=0.1)

# Train the model with multiple matrices
with nengo.Simulator(model) as sim:
    for epoch in range(3):  # Number of epochs
        for matrix_index in range(len(train_matrices_norm)):
            current_matrix_index = matrix_index  # Switch to current matrix
            for city in range(n_cities):
                current_city = city  # Set the current city
                print(f"Epoch {epoch+1}, Matrix {matrix_index+1}, City {city}")
                sim.run(0.5)  # Train for a short interval




#Testing model
# Global variables to track the current city and visited cities
current_city = 0  # Start at City 0
visited_cities = [current_city]

# Simulation for testing
with nengo.Simulator(model) as sim:
    print("Testing on unseen matrix...")
    for step in range(n_cities):
        # Run for a short duration to stabilize the winner
        sim.run(0.5)

        # Decode the winner city
        predicted_city = np.argmax(sim.data[next_city_probe][-1])
        print(f"Step {step+1}: Current City = {current_city}, Next City = {predicted_city}")
        
        # Append the winner city to visited list
        visited_cities.append(predicted_city)

        # Update the current city
        current_city = predicted_city

        # Prevent revisiting cities (optional): You can mask the visited rows here

# Print the visited sequence
print("Visited Cities:", visited_cities)

# Plot the activity of next_city
plt.figure()
plt.plot(sim.trange(), sim.data[next_city_probe])
plt.title("Test Phase: Next City Neural Activity")
plt.xlabel("Time (s)")
plt.ylabel("Activity")
plt.legend([f"City {i}" for i in range(n_cities)])
plt.show()