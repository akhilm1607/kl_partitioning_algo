import graph_search_algo
import netlist_graph
import numpy as np
import partitioning 

inf = np.inf

# Taking the inputs from user
inputs, outputs, gate_io = netlist_graph.extract_gate_io(r"C:\Users\akhil\OneDrive\Documents\SEMESTER_2\VDA\Assignments\ISCAS89\s27.bench")
# dir_or_undir = input("Choose one of the following: \n1. Directed Graph\n2. Undirected Graph\nEnter your option: ")
dir_or_undir = '2'

# Creating the list of nodes from netlist
node_list = []
node_list.append('SRC')
node_list.extend(gate_io.keys())
node_list.append('SNK')
# print('Node List: \n', node_list)

# To extract the edges from the netlist by considering the nodes as Gates and edges as interconnects
edge_list = netlist_graph.gate_io_edges(gate_io, inputs, outputs, node_list)
adjacency_matrix_inf = netlist_graph.edge_list_adj_mat(edge_list, node_list, dir_or_undir)

adjacency_matrix_inf_arr = np.array(adjacency_matrix_inf)

# Mapping the nodes to numerical indeces to form adjacency matrix
nodes_dict = {}
for i in range(len(node_list)):
    nodes_dict[i] = node_list[i]
adjacency_matrix = graph_search_algo.convert_inf_zero(adjacency_matrix_inf_arr)

visited_list = [False] * len(adjacency_matrix_inf)
graph_search_algo.graph_plotter(adjacency_matrix, dir_or_undir, visited_list, nodes_dict, node_list)

print('----------------------------------------------------------------------------------')
print("\nStarting partitionin of the circuit: ")
partitioning.kl_algorithm(node_list, adjacency_matrix_inf)