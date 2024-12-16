import numpy
from queue import PriorityQueue

class city:
    def __init__(self, name, paths):
        self.name = name
        self.paths = paths #Dict of paths with weight costs to travel there

#Define a state class for use in algorithm
class state:
    def __init__(self, current_city, visited, heuristic_cost, weight_cost):
        self.current_city = current_city,
        self.visited = visited, 
        self.heuristic_cost = heuristic_cost,
        self.weight_cost = weight_cost,
        self.combined_cost = heuristic_cost + weight_cost

#Dictionary of all cities and their city objects
cities = {
    "city1" : city("city1", {"city1":0, "city2":20, "city3":42, "city4":35}),
    "city2" : city("city2", {"city1":20, "city2":0, "city3":30, "city4":34}), 
    "city3" : city("city3", {"city1":42, "city2":30, "city3":0, "city4":12}), 
    "city4" : city("city4", {"city1":35, "city2":34, "city3":12, "city4":0}) 
    }

# Initialize a queue
q = PriorityQueue(maxsize = 0)

# initialize a starting state
first_city = cities["city1"]
q.put(state(current_city=first_city, visited=[first_city], heuristic_cost=0, weight_cost=0))

best_solution = None
best_cost = float('inf')

while not q.empty():
    current_state = q.get()

#Each state in A* should represent current city, set of visited and current path cost

#Goal state is all cities visited exactly once and salesman is back in start city

#Use of a heuristc function (Lowerbound or MST)

#Algorithm
#1: Pick the state with lowest cost (f(n)=g(n)+h(n)) from set of states 
# g(n) = cost h(n) is heuristic cost
#2: If picked state is the goal state return path as optimal
#3: Expand picked state by making successors for all cities that havent been visited
# from the current city
#4: for each successor calculate the cost and heuristic costs 
# and add it to the set of states
#5: remove all states with a higher cost than previously encountered paths 
# to the same state
#6: Start from the start and compute until optimal solution has been found

