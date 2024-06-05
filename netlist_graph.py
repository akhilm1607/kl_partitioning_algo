import re
import numpy as np

inf = np.inf

def extract_gate_io(file_path):
    gate_io = {}
    inputs = []
    outputs = []
    gate_count = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # print(lines)
            for line in lines:
                pattern_input = re.compile(f'INPUT\s*\(\s*(.*?)\s*\)')
                pattern_output = re.compile(f'OUTPUT\s*\(\s*(.*?)\s*\)')
                match_input = re.search(pattern_input, line)
                match_output = re.search(pattern_output, line)
                match_gate = re.search(r'\s*=\s*(\w+)', line)
                if match_input:
                    ip = match_input.group(1).split()
                    inputs.append(ip[0])
                if match_output:
                    op = match_output.group(1).split()
                    outputs.append(op[0])                
                if match_gate:
                    gate_name = match_gate.group(1)
                    if gate_name in gate_count:
                        gate_count[gate_name] += 1
                    else:
                        gate_count[gate_name] = 1
                    gate_number = gate_count[gate_name] - 1
                    gate_label = gate_name + f"{gate_number}"
                    io = extract_io(line)
                    gate_io[gate_label] = []
                    gate_io[gate_label].append(io[0])
                    io.pop(0)
                    gate_io[gate_label].append(io)
        return inputs, outputs, gate_io
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    
def extract_io(input_string):
    # Extract data from the start of the string till ' ' and from '( ' till ')'
    start_data = re.match(r'^\S+', input_string).group()
    inside_parentheses = re.findall(r'\((.*?)\)', input_string)
    # print(inside_parentheses)
    # Combine the results into a list
    result = [start_data] + [item.strip() for sublist in inside_parentheses for item in sublist.split(',')]
    # print(result)

    return result

def gate_io_edges(gate_io, inputs, outputs, node_list):
    edge_list = []
    for gate_v in gate_io:
        output = gate_io[gate_v][0]
        # if gate_io[gate_v][1][0] in inputs:
        #     edge = (node_list.index('SRC'), node_list.index(gate_v), gate_io[gate_v][1][0])
        #     edge_list.append(edge)
        # if len(gate_io[gate_v][1]) == 2 and gate_io[gate_v][1][1] in inputs:
        #     edge = (node_list.index('SRC'), node_list.index(gate_v), gate_io[gate_v][1][1])
        #     edge_list.append(edge)
        for in_v in range(len(gate_io[gate_v][1])):
            if gate_io[gate_v][1][in_v] in inputs:
                edge = (node_list.index('SRC'), node_list.index(gate_v), gate_io[gate_v][1][in_v])
                # print(edge)
                edge_list.append(edge)
        if gate_io[gate_v][0] in outputs:
            edge = (node_list.index(gate_v), node_list.index('SNK'), gate_io[gate_v][0])
            edge_list.append(edge)
        for gate_u in gate_io:
            if output in gate_io[gate_u][1]:
                edge = (node_list.index(gate_v), node_list.index(gate_u), output)
                edge_list.append(edge)
    # print("Edge List: \n", sorted(edge_list, key = lambda x : x[2]))
    return sorted(edge_list, key = lambda x : x[2])

def edge_list_adj_mat(edge_list, node_list, dir_or_undir):
    no_of_nodes = len(node_list)
    adjacency_matrix = [[inf] * no_of_nodes for _ in range(no_of_nodes)]

    # Populate the adjacency matrix based on the edge list
    if dir_or_undir == '2':
        for edge in edge_list:
            adjacency_matrix[edge[0]][edge[0]] = 0
            adjacency_matrix[edge[1]][edge[1]] = 0
            adjacency_matrix[edge[0]][edge[1]] = 1
            adjacency_matrix[edge[1]][edge[0]] = 1
    
    elif dir_or_undir == '1':
        for edge in edge_list:
            adjacency_matrix[edge[0]][edge[0]] = 0
            adjacency_matrix[edge[1]][edge[1]] = 0
            adjacency_matrix[edge[0]][edge[1]] = 1
            # adjacency_matrix[edge[1]][edge[0]] = 1

    return adjacency_matrix

def adj_mat_adj_lst(adjacency_matrix):
    adj_lst = {}
    for node_u in range(len(adjacency_matrix)):
        adj_lst[node_u] = []
        for node_v in range(len(adjacency_matrix)):
            if adjacency_matrix[node_u, node_v] != 0 and adjacency_matrix[node_u, node_v] != inf:
                adj_lst[node_u].append(node_v)
    
    return adj_lst