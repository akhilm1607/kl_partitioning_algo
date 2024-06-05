import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(5000)

inf = np.inf

# This definition is to convert all the entries with inf in adjacency matrix to 0. 
# This will ease the plotting function as networkx consider no edge as 0
def convert_inf_zero(adjacency_matrix_inf):
    adj_mat_zero = adjacency_matrix_inf
    for i in range(len(adjacency_matrix_inf)):
        for j in range(len(adjacency_matrix_inf)):
            if i != j and adjacency_matrix_inf[i, j] == inf:
                adj_mat_zero[i, j] = 0
    
    return adj_mat_zero

# This definition uses networkx library and plots the graphs with circular layout. 
# This plots either Directed graph or Undirected graph as per the variable dir_or_undir
def graph_plotter(adjacency_matrix, dir_or_undir, visited_list, nodes_dict, nodes):
    if dir_or_undir == '1':
        G = nx.from_numpy_matrix(adjacency_matrix, create_using = nx.DiGraph)
    if dir_or_undir == '2':
        G = nx.from_numpy_matrix(adjacency_matrix, create_using = nx.Graph)
    pos = nx.circular_layout(G)

# Visited and not visited nodes are seperated in order to differentiate the color of nodes while plotting.
    nodes_visited = []
    nodes_not_visited = []
    for i in range(len(nodes)):
        if visited_list[i] == True:
            nodes_visited.append(i)
        else:
            nodes_not_visited.append(i)
    
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_visited, node_color = 'black', alpha = 0.5, node_size = 500)
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_not_visited, node_color = 'skyblue', node_size = 500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, nodes_dict, font_size = 8)

# To display edge weights or interconnect names, uncomment this section
    # edge_weights = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_weights, font_size = 8)
    plt.show()

# This definition is to get the numerical index of a particular node.
def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

# This definition performs Depth First Search
def depth_first_search(adjacency_matrix, node_u, nodes, nodes_dict, visited, stack, dfs_path):

# As the adjacency matrix works on numerical indeces, it has to be first extracted.
    node_u_num = get_key_by_value(nodes_dict, node_u)
    visited[node_u_num] = True # Making the corresponding node visited
    dfs_path.append(node_u) # Adding the node to the dfs list
    for node in range(len(adjacency_matrix[node_u_num])):
        if(node != node_u_num):  # Adding the adjacent node into stack if not present and not visited.
            if(adjacency_matrix[node_u_num, node] != 0 and visited[node] == False):
                if nodes[node] not in stack:
                    stack.append(nodes_dict[node])
    
    if False not in visited:
        print("Visited list: ", visited)
        print("Stack of nodes: ", stack)
        print("Depth First Search Path: ", dfs_path)
        print("----------------------------------------------------------------------------------")

# Uncomment this to plot the graph with color distinctions for each iteration
    # graph_plotter(adjacency_matrix, visited, nodes)

# Starting recursion on the last entry of stack
    if (len(stack) != 0):
        node_v = stack.pop()
        depth_first_search(adjacency_matrix, node_v, nodes, nodes_dict, visited, stack, dfs_path)
    
    return dfs_path

# This definition performs Breadth First Search
def breadth_first_search(adjacency_matrix, node_u, nodes_dict):

# Initialzing the data structures needed for BFS
    visited = [False] * len(adjacency_matrix[0])
    bfs_added = [False] * len(adjacency_matrix[0])
    queue = []
    bfs_path = []
    node_u_num = get_key_by_value(nodes_dict, node_u)
    visited[node_u_num] = True
    bfs_added[node_u_num] = True
    queue.append(node_u)
    # print("Visited list: ", visited)
    # print("Queue of nodes: ", queue)
    # print("Breadth First Search Path: ", bfs_path)
    # graph_plotter(adjacency_matrix, bfs_added, nodes)

    while(len(queue)!=0):           # Continuing until the queue is empty
        cur_node = get_key_by_value(nodes_dict, queue.pop(0))
        for node_adj in range(len(adjacency_matrix[cur_node])): 
            if adjacency_matrix[cur_node, node_adj] != 0 and visited[node_adj] == False and nodes_dict[node_adj] not in queue:
                visited[node_adj] = True
                queue.append(nodes_dict[node_adj])
        bfs_path.append(nodes_dict[cur_node])
        bfs_added[cur_node] = True
    print("Visited list: ", visited)
    print("Queue of nodes: ", queue)
    print("Breadth First Search Path: ", bfs_path)
    print("----------------------------------------------------------------------------------")
        # graph_plotter(adjacency_matrix, bfs_added, nodes)
