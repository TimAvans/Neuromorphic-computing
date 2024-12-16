import numpy as np
import heapq
import math

#Estimate the cost to visit all remaining unvisited nodes
def heuristic(current_node, visited_nodes, graph_length):
    unvisited = []
    for i in range(graph_length):
        if i not in visited_nodes:
            unvisited.append(i)
    #If there are no unvisited nodes we already reached the end
    if len(unvisited) == 0:
        return 0
    
    #Now estimate remaining costs
    cost_nearest_city = float('inf')

#Check to find the lowest cost to any unvisited city/node
    for city in unvisited:
        cost_city = graph[current_node][city]
        if cost_city < cost_nearest_city:
            cost_nearest_city = cost_city

    cost_nearest_start = float('inf')
#Check to find lowest cost from any unvisited node/city to the start node/city
    for city in unvisited:
        cost_start = graph[city][0]
        if cost_start < cost_nearest_start:
            cost_nearest_start = cost_start

    return cost_nearest_city + cost_nearest_start

def run_astar(graph):
    graph_length = len(graph)
    queue = []
    heapq.heappush(queue, (0,0,0, [0], set([0])))
    min = math.inf
    best = None
    
    #While the queue isnt empty we continue the algo
    while queue:
        f_score, cost, node, path, visited_nodes = heapq.heappop(queue)
        #If we have visited all nodes
        if len(visited_nodes) == graph_length:
            total = cost + graph[node][0] #Costs to return to the start
            if total < min:
                min = total
                best = path + [0]
            continue

        #Check each neighbour in the graph
        for neigh in range(graph_length):
            #If it isnt visited yet
            if neigh not in visited_nodes:
                #Calc all variables for the algorithm
                n_cost = cost + graph[node][neigh]
                n_visited_nodes = visited_nodes.union({neigh}) 
                n_path = path + [neigh]
                h_score = heuristic(neigh, n_visited_nodes, graph_length)
                f_score = n_cost + h_score
                heapq.heappush(queue, (f_score, n_cost, neigh, n_path, n_visited_nodes))
    return min, best

graph = [
[0, 10, 15, 20], 
[10, 0, 35, 25], 
[15, 35, 0, 30],
[20, 25, 30, 0]
]

cost, path = run_astar(graph)
print(f"Minimum cost: {cost}")
print(f"Best path: {path}")