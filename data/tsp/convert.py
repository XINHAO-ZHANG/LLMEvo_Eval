import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# 读取CSV文件
matrix = np.loadtxt('tsp_dismat_60.csv', delimiter=',')

def create_data_model():
    data = {}
    data['distance_matrix'] = matrix.tolist()
    return data

data = create_data_model()
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 1, 0)
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(data['distance_matrix'][from_node][to_node])

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.time_limit.seconds = 300  # 可选：限制最大搜索时间

solution = routing.SolveWithParameters(search_parameters)

if solution:
    index = routing.Start(0)
    route = []
    route_distance = 0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        next_index = solution.Value(routing.NextVar(index))
        route_distance += data['distance_matrix'][manager.IndexToNode(index)][manager.IndexToNode(next_index)]
        index = next_index
    route.append(manager.IndexToNode(index))
    print("Optimal route:", route)
    print("Total distance:", int(route_distance))
else:
    print("No solution found!")