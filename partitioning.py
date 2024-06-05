import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graph_search_algo
import math

inf = np.inf

# This function adds a dummy node if the total no. of nodes is odd. It also updates the 
# adjacency matrix by adding a row and col filled with inf
def kl_upd_node_list_adj_mat(adjacency_matrix_inf, node_list):
    adjacency_matrix_dmy = adjacency_matrix_inf
    dmy_rw = [inf] * (len(adjacency_matrix_inf) + 1)
    if(len(adjacency_matrix_inf)%2 != 0):
        for row in adjacency_matrix_dmy:
            row.append(inf)
        adjacency_matrix_dmy.append(dmy_rw)
        node_list.append('DMY')

    return adjacency_matrix_dmy, node_list

def kl_algorithm(node_list, adjacency_matrix_inf):
    adjacency_matrix_upd_kl, node_list = kl_upd_node_list_adj_mat(adjacency_matrix_inf, node_list)
    partition_1 = []
    partition_2 = []
    fixed = []
    # Obtaining the intial partition.
    for node in range(0, int(len(node_list)/2)):
        partition_1.append(node_list[node])
        partition_2.append(node_list[node + int(len(node_list)/2)])
    print("Initial partition\n")
    print("Partition 1: \n")
    print(partition_1)
    print("Partition 2: \n")
    print(partition_2)
    adjacency_matrix_upd_kl = np.array(adjacency_matrix_upd_kl)
    adjacency_matrix_upd_kl = graph_search_algo.convert_inf_zero(adjacency_matrix_upd_kl)

    kl_graph_plotter(adjacency_matrix_upd_kl, partition_1, partition_2, node_list)

    # computing the initial cut size of the partition
    cut_size_init = cut_size_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list)
    print("Initial cut size: ", cut_size_init)
    # computing initial D values for all nodes
    init_d_value = d_value_init(adjacency_matrix_upd_kl, partition_1, partition_2, node_list)

    # computing initial gains for all the possible node pairs
    init_gain = gain_compute(adjacency_matrix_upd_kl, node_list, partition_1, partition_2, init_d_value, fixed)
    # print(init_gain)

    # selecting the pair with highest gain.
    sel_pair = list(init_gain.items())[0]

    # After computing the initial swap, it will be continued until the there is no positive gain.
    while sel_pair[1] > 0:

        # Making the nodes of the pair as fixed.
        fixed.append(sel_pair[0][0])
        fixed.append(sel_pair[0][1])

        # swapping the nodes between partitions.
        partition_1, partition_2 = node_swap(partition_1, partition_2, sel_pair[0])

        # computing new D values and gains and selecting the pair with maximum gain.
        d_value = d_value_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list, init_d_value, fixed, sel_pair[0])
        gain = gain_compute(adjacency_matrix_upd_kl, node_list, partition_1, partition_2, d_value, fixed)
        # print(gain)
        sel_pair = list(gain.items())[0]

    # Printing the final partitions
    print("Final partition: \n")
    print("Partition 1: \n")
    print(partition_1)
    print("Partition 2: \n")
    print(partition_2)

    kl_graph_plotter(adjacency_matrix_upd_kl, partition_1, partition_2, node_list)

    cut_size_final = cut_size_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list)
    print("Final cut size: ", cut_size_final)

# This definition is to calculate the initial D values of all the nodes as per the formula 
# Da = Ea − Ia
def d_value_init(adjacency_matrix_upd_kl, partition_1, partition_2, node_list):
    d_value_in = [0] * len(node_list)
    for node in range(len(node_list)):
        for node_v in range(len(adjacency_matrix_upd_kl[node])):
            if adjacency_matrix_upd_kl[node][node_v] != 0 and node != node_v:
                if (node_list[node] in partition_1 and node_list[node_v] in partition_2) or (node_list[node_v] in partition_1 and node_list[node] in partition_2):
                    d_value_in[node] = d_value_in[node] + adjacency_matrix_upd_kl[node][node_v]
                elif (node_list[node] in partition_1 and node_list[node_v] in partition_1) or (node_list[node_v] in partition_2 and node_list[node] in partition_2):  
                    d_value_in[node] = d_value_in[node] - adjacency_matrix_upd_kl[node][node_v]
    return d_value_in

# This definition is to compute the updated D values after a swap has been done as per the formula
# D′x = Dx + 2cxa − 2cxb, ∀x ∈ A − {a}
# D′y = Dy + 2cyb − 2cya, ∀y ∈ B − {b}.
def d_value_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list, d_value, fixed, pair):
    pair_index_0 = node_list.index(pair[0])
    pair_index_1 = node_list.index(pair[1])
    for node in partition_1:
        node_index = node_list.index(node)
        if node not in fixed:
            d_value[node_index] = d_value[node_index] + (2*adjacency_matrix_upd_kl[node_index][pair_index_0]) - (2*adjacency_matrix_upd_kl[node_index][pair_index_1])
        else:
            continue

    for node in partition_2:
        node_index = node_list.index(node)
        if node not in fixed:
            d_value[node_index] = d_value[node_index] + (2*adjacency_matrix_upd_kl[node_index][pair_index_1]) - (2*adjacency_matrix_upd_kl[node_index][pair_index_0])
        else:
            continue

    return d_value

# This definition is to compute the gain that can be acheived after swapping a pair of nodes as per
# formula gxy = Dx + Dy − 2cxy
def gain_compute(adjacency_matrix_upd_kl, node_list, partition_1, partition_2, d_value, fixed):
    gain = {}
    for node_u in partition_1:
        node_u_index = node_list.index(node_u)
        for node_v in partition_2:
            if node_u not in fixed and node_v not in fixed:
                node_v_index = node_list.index(node_v)
                pair = (node_u, node_v)
                gain[pair] = d_value[node_u_index] + d_value[node_v_index] - (2 * adjacency_matrix_upd_kl[node_u_index][node_v_index])
                gain = dict(sorted(gain.items(), key=lambda item: item[1], reverse=True))

    return gain
    
# This definition is to swap the nodes of a selected pair.
def node_swap(partition_1, partition_2, pair):
    partition_1.remove(pair[0])
    partition_2.remove(pair[1])
    partition_1.append(pair[1])
    partition_2.append(pair[0])

    return partition_1, partition_2

# This definition is to compute the cut size of the partition.
def cut_size_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list):
    cut_size = 0
    for node_u in partition_1:
        node_u_index = node_list.index(node_u)
        for node_v in partition_2:
            node_v_index = node_list.index(node_v)
            cut_size = cut_size + adjacency_matrix_upd_kl[node_u_index][node_v_index]
    
    return cut_size

# To plot the partitions on the same graph.
def kl_graph_plotter(adjacency_matrix, partition_1, partition_2, node_list):

    # removing the matrix entries of dmy node
    adjacency_matrix = adjacency_matrix[:, :-1]
    adjacency_matrix = adjacency_matrix[:-1, :]
    G = nx.from_numpy_matrix(adjacency_matrix, create_using = nx.Graph)

    # calculating the angle for positioning the nodes
    count = len(partition_1)
    angle = (2*math.pi)/count
    pos = {}
    for node in partition_1:
        value = partition_1.index(node) + 1
        pos[node_list.index(node)] = (-count - count + (count* math.cos(value*angle)), count*(math.sin(value*angle)))
    for node in partition_2:
        value = partition_2.index(node) + 1
        pos[node_list.index(node)] = (count + count + (count* math.cos(value*angle)), count*(math.sin(value*angle)))

    # creating a node list for each partition separately and also removing dmy
    nodes_partition_1 = []
    nodes_partition_2 = []
    nodes_dict = {}

    for i in range(len(partition_1)):
        if partition_1[i] != 'DMY':
            nodes_partition_1.append(node_list.index(partition_1[i]))
            nodes_dict[node_list.index(partition_1[i])] = partition_1[i]
        if partition_2[i] != 'DMY':
            nodes_partition_2.append(node_list.index(partition_2[i]))
            nodes_dict[node_list.index(partition_2[i])] = partition_2[i]
    
    # Adding all the nodes, edges and labels to the graph. Each partition is of different color.
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_partition_1, node_color = 'yellow', node_size = 500)
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_partition_2, node_color = 'skyblue', node_size = 500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, nodes_dict, font_size = 8)
    plt.show()
