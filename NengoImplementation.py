import nengo
import nengo.ensemble
import numpy as np
import matplotlib.pyplot as plt

'''
BELANGRIJKE INFO:
Relationship Between n_neurons and dimensions
n_neurons: Determines the number of neurons in the ensemble. More neurons allow for finer, more accurate representations.
dimensions: Determines the structure of the data that the ensemble can represent.
You can think of it this way:

Each neuron in an ensemble contributes to representing the signal in the specified dimensional space.
The more neurons you have relative to the number of dimensions, the better the ensemble will be at accurately representing the signal.

dimensions=1: The ensemble represents a scalar value (e.g., a single number, like position or temperature).
dimensions=2: The ensemble represents a 2D vector (e.g., a point in a 2D plane, like (x, y)).
dimensions=3: The ensemble represents a 3D vector (e.g., spatial coordinates (x, y, z)).

'''
'''
t: is Time from the simulation of nengo
x: represents the input vector in this case the activity from the next ensemble
'''

distances = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])
#Invert the distances so they can be used as weights
distances_inverted = 1 / (distances + 0.00001) #Add small number to all distances so we dont divide by 0
#Normalize the distances for less noisy data
distances_norm = distances_inverted / distances_inverted.max()
#!TODO: We can add a scale to the distances to make them weigh more heavy on the outcome? We can scale nonlinearly to amplify the difference in the non winning weights?
#This is a test because it might be needed to amplify the differences in the weights
nonlin_trans = lambda x: x**3
distances_transformed = nonlin_trans(distances_norm) * 10
#Calculate the amount of cities from the distances matrix
n_cities = len(distances)

visited_cities = []

#We need a function to track all visited cities to record the shortest path
def track_visited(t, x):
    global visited_cities
    winner = np.argmax(x)  # Find the city with the highest activity
    if len(visited_cities) == 0 or visited_cities[-1] != winner:
        visited_cities.append(winner)  # Add the winner to the visited list
    return 0  # Return dummy value (Nengo nodes require an output)


model = nengo.Network()
with model:

    def WTA(t, x):
        winning = np.argmax(x) #Get winning index
        inhibitions = np.zeros(n_cities)  #create inhibition vector with -3 on all spots
        inhibitions[winning] = 1 #Place no inhibition on the winning index
        return inhibitions
    
    # Suppress visited cities
    def inhibit_visited(t):
        inhibition = np.zeros(n_cities)
        for city in visited_cities:
            inhibition[city] = -3.0  # Inhibition for visited cities
        return inhibition
    
    #We represent the city as a cluster of neurons with dimension of n_cities because we represent them as a vector of sorts 
    #e.g. city 1 = [1, 0, 0 , 0]
    #Cluster of neurons representing our current city high neuron count for stable results
    current = nengo.Ensemble(n_neurons=1000, dimensions=n_cities)
    #Cluster of neurons representing the next city high neuron count for stable results
    next = nengo.Ensemble(n_neurons=1000, dimensions=n_cities)
    #Inhibitory ensemble because lateral inhibition did not produce consistently good results
    inhibitory = nengo.Ensemble(n_neurons=1000, dimensions=n_cities)

    #Create a node that selects the winner
    winner = nengo.Node(size_in=n_cities, output=WTA)
    #Node to track all visited cities
    visited = nengo.Node(size_in=n_cities)
    #node for tracking all visited cities
    tracker = nengo.Node(track_visited, size_in=n_cities)
    #Node to inhibit the already visited cities
    inhibit_node = nengo.Node(output=inhibit_visited)

    #Represent the distances matrix as neural connections
    #we do this by looping over the matrix and representing each distance as a weighted connection
    for start_city in range(n_cities):
        for end_city in range(n_cities):
            if start_city != end_city:
                nengo.Connection(
                    current[start_city], 
                    next[end_city], 
                    transform=distances_transformed[start_city, end_city])
                
    nengo.Connection(next, winner, synapse=0.01)
    nengo.Connection(winner, next, transform=4 * np.eye(n_cities), synapse=0.01)
    nengo.Connection(next, tracker, synapse=0.01)
    nengo.Connection(inhibit_node, next, synapse=0.01)

#region Old winner takes all mechanism instead now we use wta node and function
    # #There is need for a decisionmaking mechanism, 
    # #will take winner takes all (WTA) because we need a clear choice in city with shortest distance
    # #We will build it by inhibiting all other cities that arent the shortest
    # for i in range(n_cities):
    #     for j in range(n_cities):
    #         if i != j:  # Inhibit other cities
    #             #!TODO: This is also something we can scale nonstaticly, so scale the inhibition with the weight of the city?
    #             strength = -2.0 * (1 - distances_norm[i, j])
    #             nengo.Connection(next[i], next[j], transform=strength)  # Inhibit the city high number for more stable results

    # #To amplify the winner even more add a recurrent connection that boosts its activity
    # nengo.Connection(next, next, transform=1.05 * np.eye(n_cities), synapse=0.02)

    # #Also suppress the current city in the next city ensemble so we dont travel to ourselves 
    # for i in range(n_cities):
    #     nengo.Connection(current[i], next[i], transform=-1.5)  # Suppress current city activity
#endregion

    # Input node to specify the starting city (e.g., city 0)
    input_node = nengo.Node([1, 0, 0, 0])  # Start at city 0
    nengo.Connection(input_node, current)

    # Probe the next city ensemble to observe activity
    next_city_probe = nengo.Probe(next, synapse=0.02) #an filter is necessary due to noisy activity data

with nengo.Simulator(model) as sim:
    sim.run(1.0)   


# Print the visited cities (shortest route)
print("Visited Cities (Shortest Route):", visited_cities)

# Plot the activity of next city
plt.figure()
plt.plot(sim.trange(), sim.data[next_city_probe])
plt.title("Next City Neural Activity (Decoded)")
plt.xlabel("Time (s)")
plt.ylabel("Activity")
plt.legend([f"City {i}" for i in range(n_cities)])
plt.show()