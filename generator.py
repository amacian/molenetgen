import random
import numpy as np

from network import gen_topology, count_degree_freqs, calculate_edge_distances, \
    write_network_xls, backbone_assign_types, rename_nodes, metro_assign_types, color_nodes, \
    connect_sub_metro_regions, write_network_xls_plus
import networkconstants as nc
import networkx as nx
from matplotlib import pyplot as plt

import time
import pandas as pd

# Global variables
types_excluded = [nc.TRANSIT_CODE]


# Filename for the Backbone network
# degrees: A list with the possible node degrees
# weights: The frequency % of the node degrees in the total number of COs
# nodes: Total number of COs
# upper_limits: upper length limits for each range
# types: available types of nodes and % per type
def backbone(degrees, weights, nodes, upper_limits, types, algo="spectral", dict_colors={}):
    # Sheet names in the Excel file for nodes and links
    node_sheet = "Nodes"
    link_sheet = "Links"

    # Repeat
    while True:
        # Call the function to generate the topology
        topo = gen_topology(degrees, weights, nodes)
        # Calculate the actual degrees for each node
        dist_degrees = [val for (node, val) in topo.degree()]
        # Topology generation removes parallel links and self-loops, so
        # it is possible that the number of degrees of some nodes is
        # smaller than the smallest allowed degree.
        if min(dist_degrees) >= degrees[0]:
            # Finish only if the degrees of all nodes are >= minimum accepted degree
            break
        # Otherwise, repeat the node generation
        print("Nodes with less links than lower edge threshold")

    # Compare expected degree weights and actual frequencies
    print("Expected degree frequency:", weights)
    print("Actual degree frequency:", count_degree_freqs(dist_degrees))
    # Represent the topology
    # print(topo.edges)

    # Assign types to nodes
    assigned_types = backbone_assign_types(topo, types)

    # Modify the node labels to name them as a combination of the type and an index
    name_nodes = rename_nodes(assigned_types, types)
    topo = nx.relabel_nodes(topo, dict(zip(topo.nodes, name_nodes)))

    # Generate positions of the nodes based on the spectral distribution
    pos = None

    # print(algo)
    match algo:
        case nc.KAMADA_ALGO:
            pos = nx.kamada_kawai_layout(topo)
        case nc.SPRING_ALGO:
            pos = nx.spring_layout(topo)
        case nc.SPIRAL_ALGO:
            pos = nx.spiral_layout(topo)
        case nc.SHELL_ALGO:
            pos = nx.shell_layout(topo)
        case _:
            pos = nx.spectral_layout(topo)

    # Other options for the positions
    # pos = nx.kamada_kawai_layout(topo)
    # pos = nx.spring_layout(topo)
    # print(pos)

    # Assign the positions to the topology
    # plt.rcParams["figure.figsize"] = [3 * val for val in plt.rcParamsDefault["figure.figsize"]]
    # nx.draw(topo, pos=pos, with_labels=True, font_weight='bold')
    # plt.savefig(filename + ".png")
    # plt.figure().clear()

    # Calculate distance limits
    max_upper = upper_limits[len(upper_limits) - 1]
    # Modify the limits to approximate to the expected percentages per distance range
    corrected_max_upper = max_upper - (max_upper - upper_limits[len(upper_limits) - 2]) / 2
    # modify distances from the ones in the graph to the actual expected scale
    distances = calculate_edge_distances(topo, pos, corrected_max_upper)
    # print("Count distance per range:", count_distance_ranges(distances, upper_limits))

    # Generate a sequence of colors for each node depending on the type
    colors = color_nodes(assigned_types, dict_colors)

    # Write Excel file
    # write_network_xls(filename, topo, distances, assigned_types, node_sheet, link_sheet)
    return topo, distances, assigned_types, node_sheet, link_sheet, pos, colors


def write_backbone(filename, topo, distances, assigned_types, figure, node_sheet="nodes", link_sheet="links",
                   clusters=None, pos=None, reference_nodes=None):
    figure.savefig(filename + ".png")
    x_coord = [0] * len(topo.nodes)
    y_coord = [0] * len(topo.nodes)
    if pos is not None:
        x_coord, y_coord = zip(*pos.values())
    write_network_xls(filename, topo, distances, assigned_types, node_sheet, link_sheet, clusters, x_coord,
                      y_coord, reference_nodes)


# Generate a region of a Metro Core Network
def metro_core_split(filename, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                     national_nodes=[]):
    node_sheet = "Nodes"
    link_sheet = "Links"

    # Split the generation into subnets so the connections are balanced for each region
    # With a NCO
    num_subnets = len(national_nodes)
    if num_subnets == 0:
        num_subnets = types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'].values[0]
        national_nodes = [nc.NATIONAL_CO_CODE + str(i + 1) for i in range(num_subnets)]

    if filename is not None:
        # Actual filename
        filename = str(time.time()) + "_" + filename

    # Pending number of each type of node to be generated
    pending_number = types.number

    # Initialize the topology to empty value
    topo = None
    # Variable that will hold the type assigned to each node
    assigned_types = []
    # Generate as many subnets as defined above
    for i in range(num_subnets):
        # Define the number of nodes of each type that will take part in the sub topology
        subnet_numbers = [round(val / (num_subnets - i)) for val in pending_number]
        # Update the pending number per type of node
        pending_number = [orig - spent for orig, spent in zip(pending_number, subnet_numbers)]

        # Generate one of the sub topologies
        sub_topo, assigned_types_sub = sub_metro(degrees, weights,
                                                 sum(subnet_numbers), types, subnet_numbers)

        # plt.rcParams["figure.figsize"] = [val for val in plt.rcParamsDefault["figure.figsize"]]

        # If it is the first sub topology, create the topology assigning it
        if topo is None:
            topo = sub_topo
        # Otherwise connect the subtopology by selecting 2 random elements of the
        # topology and 2 of the subtopology and connect them.
        else:
            topo = connect_sub_metro_regions(topo, sub_topo, degrees[0])
        # print("Total topo: ", len(topo.nodes))
        # Extend the types of the nodes with the new types obtained
        assigned_types.extend(assigned_types_sub)

    # Generate a sequence of colors for each node depending on the type
    colors = color_nodes(assigned_types, dict_colors)

    # Modify the node labels to name them as a combination of the type and an index
    name_nodes = rename_nodes(assigned_types, types, nc.NATIONAL_CO_CODE, national_nodes)
    topo = nx.relabel_nodes(topo, dict(zip(topo.nodes, name_nodes)))

    # Generate positions of the nodes based on the spectral/spring distribution
    # print(algo)
    match algo:
        case nc.KAMADA_ALGO:
            pos = nx.kamada_kawai_layout(topo)
        case nc.SPRING_ALGO:
            pos = nx.spring_layout(topo)
        case nc.SPIRAL_ALGO:
            pos = nx.spiral_layout(topo)
        case nc.SHELL_ALGO:
            pos = nx.shell_layout(topo)
        case _:
            pos = nx.spectral_layout(topo)
    # Draw and save the figure
    plt.rcParams["figure.figsize"] = [3 * val for val in plt.rcParamsDefault["figure.figsize"]]

    nx.draw(topo, pos=pos, with_labels=True, font_weight='bold', node_color=colors)
    # plt.show()
    if filename is not None:
        plt.savefig(filename + ".png")
        # Reinitialize the figure for future use
        # plt.figure().clear()

    # Define the upper bound for the distances. Could be the maximum limit
    # or something below that (e.g. 1/2 of the last range)
    corrected_max_upper = upper_limits[-1]
    # Scale the distances from the topology to the defined values
    distances = calculate_edge_distances(topo, pos, corrected_max_upper)
    # Check the generate percentages for each range
    # print(count_distance_ranges(distances, upper_limits))

    # Write Excel file
    if filename is not None:
        write_network_xls(filename, topo, distances, assigned_types, node_sheet, link_sheet)

    return topo, distances, assigned_types, pos, colors


# Generate the subtopologies for the metro topology
def sub_metro(degrees, weights, nodes, types, subnet_number):
    while True:
        # Call the function to generate the topology
        topo = gen_topology(degrees, weights, nodes)
        # Calculate the actual degrees for each node
        dist_degrees = [val for (node, val) in topo.degree()]
        # Topology generation removes parallel links and self-loops, so
        # it is possible that the number of degrees of some nodes is
        # smaller than the smallest allowed degree.
        if min(dist_degrees) >= degrees[0]:
            # Finish only if the degrees of all nodes are >= minimum accepted degree
            break
        # Otherwise, repeat the node generation
        # print("Nodes with fewer links than lower edge threshold")

    # Assign types to nodes
    assigned_types = metro_assign_types(pd.DataFrame({'code': types.code,
                                                      'number': subnet_number}))

    return topo, assigned_types


def format_node_list(nodes):
    result = ""
    process = 0
    for i in nodes:
        if process != 0:
            result = result + ","
        process = process+1
        result = result + i
        if process == 10:
            result = result + "\n"
            process = 1
    return result


# Generate the Metro aggregation Horseshoes
def metro_aggregation_horseshoe(initial_rco_name,
                                initial_lco_idx, final_rco_name,
                                hops, length_ranges,
                                perc_horse_length, prefix=nc.LOCAL_CO_CODE, dict_colors={},
                                filename=None):
    # Sheet names in the Excel file for nodes and links
    node_sheet = "Nodes"
    link_sheet = "Links"

    # The number of local offices is 1 element less than number of hops
    n_local_offices = hops - 1
    # select one of the ranges from the distances based on the weight defined by percentage
    range_idx = 1 + random.choices(range(len(length_ranges) - 1),
                                   weights=perc_horse_length, k=1)[0]
    # Select a length from the range using a uniform distribution
    total_length = int(random.uniform(length_ranges[range_idx - 1] + 1, length_ranges[range_idx]))

    # Select as many cut points as local COs to define the place in the line where they
    # will be located
    cut_points = [element / 100 for element in random.sample(range(1, (total_length * 100)), hops - 1)]
    # Sort in ascending order
    cut_points.sort()
    # Create a graph
    horseshoe = nx.Graph()
    # Add the initial RCO and its type in the list of node types
    horseshoe.add_node(initial_rco_name)
    previous_node = initial_rco_name
    types = [nc.REGIONAL_CO_CODE]
    # Generate the Local COs with the initial index
    for i in range(initial_lco_idx, n_local_offices + initial_lco_idx):
        # Create the name concatenating the prefix (LCO) and the index
        name = prefix + str(i)
        # Add the node to the graph
        horseshoe.add_node(name)
        # Add an edge to the previous node
        horseshoe.add_edge(previous_node, name)
        # Set this node as the previous node for the next iteration
        previous_node = name
        # Append the type to the list of node types
        types.append(prefix)

    # Add the final RCO
    horseshoe.add_node(final_rco_name)
    # And link it to the last LCO
    horseshoe.add_edge(previous_node, final_rco_name)
    # Add its type to the list of node types
    types.append(nc.REGIONAL_CO_CODE)

    # Define the colors based on the node types
    colors = color_nodes(types, dict_colors)

    # Define the positions. First node is located at 0, 0
    positions = [0]
    # Local COs are located at the cut points
    positions.extend(cut_points)
    # Final RCO is located at total_length
    positions.append(total_length)
    # Calculate distances based on positions
    distances = [round(positions[i + 1] - positions[i], 2) for i in range(len(positions) - 1)]
    # print(distances)

    # Draw the nodes in the defined positions
    nx.draw(horseshoe, node_color=colors, with_labels=True, font_weight='bold',
            pos={key: np.array([value, 0]) for key, value in zip(horseshoe.nodes, positions)})
    # Add the distance as a label for the nodes
    nx.draw_networkx_edge_labels(horseshoe, edge_labels=dict(zip(horseshoe.edges, distances)),
                                 pos={key: np.array([value, 0]) for key, value in zip(horseshoe.nodes, positions)})
    # And save the file
    if filename is not None:
        plt.savefig(filename + ".png")
    # plt.figure().clear()

    # Add the distances as an edge attribute for the nodes
    nx.set_edge_attributes(horseshoe, dict(zip(horseshoe.edges, distances)), 'weight')

    # Create a Dummy node connected to the horseshoe ends to calculate which end is
    # close to each node
    dummy_node = 'Dummy'
    horseshoe.add_node(dummy_node)
    horseshoe.add_edge(dummy_node, initial_rco_name)
    horseshoe.add_edge(dummy_node, final_rco_name)
    # Set distance 1 for both edges between Dummy and the ends
    nx.set_edge_attributes(horseshoe, dict(zip(nx.edges(horseshoe, [dummy_node]), [1, 1])), 'weight')
    # print(nx.get_edge_attributes(horseshoe, 'weight'))

    # Generate all the shortest paths between any node and the Dummy one
    results = nx.shortest_path(horseshoe, source=dummy_node, weight='weight')
    reference_node = [results[i][1] for i in horseshoe.nodes if i != dummy_node]
    # Remove the dummy node (and the edges automatically)
    horseshoe.remove_node('Dummy')
    # Write Excel file
    if filename is not None:
        write_network_xls_plus(filename, horseshoe, distances, types, node_sheet,
                               link_sheet, reference_node)
    pos = {key: np.array([value, 0]) for key, value in zip(horseshoe.nodes, positions)}
    return horseshoe, distances, types, pos, colors, reference_node


# Function to define the parameters for the ring Metro-regional networks
# n_rings number of rings in the structure (1, 2, 3, 4 or 6)
# end_1 and end_2 name of both ends of the ring.
# prefix to prepend to the regional nodes
# initial index for the regional nodes
# Variability the actual length of the links will be between avg_length [+- var * avg_length]
def ring_structure_tel(n_rings, end_1, end_2, prefix='R',
                       initial_index=1, variability=0.1, filename=None):
    # Definition of parameters for each type of ring structure, the lengths of each subrings
    # number of offices per subring, length ranges and max length without amplifier.
    ring_config = pd.DataFrame({"r_numbers": [1, 2, 3, 4, 6],
                                "avg_lengths": [[380], [225, 235], [145, 280, 350],
                                                [120, 160, 240, 275],
                                                [55, 85, 95, 185, 200, 245]],
                                "offices": [[5], [4, 4], [3, 5, 6],
                                            [3, 5, 9, 6],
                                            [1, 1, 3, 6, 6, 8]],
                                "ranges": [[[35, 65]], [[4, 55], [5, 70]],
                                           [[12, 60], [30, 60], [24, 60]],
                                           [[13, 45], [10, 46], [4, 36], [16, 65]],
                                           [[18, 38], [11, 50], [4, 38],
                                            [8, 41], [14, 44], [2, 50]]
                                           ],
                                "amplifiers": [[1], [0, 2], [0, 1, 2], [1, 0, 0, 2],
                                               [0, 1, 1, 0, 0, 0]],
                                "dist_amplifiers": [70, 70, 60, 60, 55]
                                })
    # Actual creation of rings
    return create_ring_network_tel(n_rings, ring_config, end_1, end_2,
                                   prefix, initial_index, variability, filename)


# Function to create ring Metro-regional networks
# n_rings number of rings in the structure (1, 2, 3, 4 or 6)
# ring_config DataFrame structure formed by number of rings
def create_ring_network_tel(n_rings, ring_config, end_1, end_2,
                            prefix, initial_index, variability, filename=None):
    # Generation of an empty graph
    ring = nx.Graph()
    # Retrieving the information related to the ring structure for the number of rings
    info_ring = ring_config.loc[ring_config['r_numbers'] == n_rings]
    # Index to be used to retrieve the information from the DataFrame
    idx = info_ring.index.to_list()[0]

    # Start with the initial index and increase it if needed
    next_index = initial_index
    # Suffixes that will be used for each of the rings
    suffixes = ['A', 'B', 'C', 'D', 'E', 'F']

    # Build each sub_ring
    for i in range(n_rings):
        # Take the corresponding suffix and the appropriate parameters for that ring
        # in the ring structure (idx)
        suffix = suffixes[i]
        avg_length = info_ring['avg_lengths'][idx][i]
        offices = info_ring['offices'][idx][i]
        range_ring = info_ring['ranges'][idx][i]
        dist_amplifiers = info_ring['dist_amplifiers'][idx]

        # Generate a random uniform value for the actual total length of the ring
        length = random.uniform(avg_length - variability * avg_length,
                                avg_length + variability * avg_length)

        # Generate random values for the lengths between offices using the uniform
        # distribution except for the last link
        link_lengths = [random.uniform(range_ring[0], range_ring[1]) for i in range(offices)]
        # Calculate the length of the final link
        pending_length = length - sum(link_lengths)
        # pending_length might be negative due to the way the lengths are built
        # correct it by reducing proportionally from the other lengths
        link_lengths = correct_negative_pending(link_lengths, pending_length, offices)
        # Shuffle the lengths so their distribution is random
        random.shuffle(link_lengths)
        # Calculate if amplifiers are needed and place them in the proper positions
        # splitting the links
        link_lengths, pos_amplifiers = place_amplifiers(link_lengths, dist_amplifiers)
        # Round to 2 decimal digits
        link_lengths = [round(val, 2) for val in link_lengths]

        # Generate the names of the offices using the prefix, the number of rings
        # in the structure, an increasing index starting at next index and covering
        # the number of offices, and the appropriate suffix
        office_names = [prefix + str(n_rings) + str(my_index) + suffix
                        for my_index in range(next_index, next_index + offices)]

        # Insert the name of the amplifiers in the proper position
        for pos in pos_amplifiers:
            office_names.insert((pos - 1), (prefix + 'AMP' + str(pos) + suffix))

        # Place the first and last end of the rings into the office names
        office_names.insert(0, end_1)
        office_names.append(end_2)
        # Printing to screen
        # print_ring_to_screen(link_lengths, office_names)
        # Create a graph for the subring using the names for the nodes and the
        # link_lengths as an attribute for weight
        sub_ring = create_graph_rings(office_names, link_lengths)
        # merge the ring and the sub_ring
        ring = nx.compose(ring, sub_ring)

    # Retrieve the distances from the weight parameter of the edges
    distances = [val for key, val in nx.get_edge_attributes(ring, 'weight').items()]

    # Generate positions of the nodes based on the spectral/spring distribution
    # pos = nx.spectral_layout(topo)
    pos_loc = nx.spring_layout(ring)
    # pos_loc = nx.kamada_kawai_layout(ring)
    # Draw and save the figure
    color_map = ['y' if node == end_1 or node == end_2 else 'c' for node in ring.nodes]
    # print(nx.to_numpy_array(ring))
    nx.draw(ring, pos=pos_loc, with_labels=True, font_weight='bold', node_color=color_map)

    if filename is not None:
        plt.savefig(filename)
    # Reinitialize the figure for future use
    # plt.show()
    # plt.figure().clear()
    # Types will be NCO for the ends and RCO for the rest of the nodes
    # TODO pass it as an argument
    types = [nc.NATIONAL_CO_CODE if node == end_1 or node == end_2 else nc.REGIONAL_CO_CODE for node in ring.nodes]

    # Create a Dummy node connected to the ring ends to calculate which end is
    # close to each node
    dummy_node = 'Dummy'
    ring.add_node(dummy_node)
    ring.add_edge(dummy_node, end_1)
    ring.add_edge(dummy_node, end_2)
    # Set distance 1 for both edges between Dummy and the ends
    nx.set_edge_attributes(ring, dict(zip(nx.edges(ring, [dummy_node]), [1, 1])), 'weight')

    # Generate all the shortest paths between any node and the Dummy one
    results = nx.shortest_path(ring, source=dummy_node, weight='weight')
    # Locate the reference node
    reference_node = [results[i][1] for i in ring.nodes if i != dummy_node]

    # Remove the dummy node (and the edges automatically)
    ring.remove_node('Dummy')
    # Write Excel file
    if filename is not None:
        write_network_xls_plus("test_ring_Telefonica.xlsx", ring, distances, types,
                               "nodes", "links", reference_node)

    return ring, distances, types, pos_loc, color_map, reference_node


# Method to place amplifiers in between nodes when the length is over the max allowed length
def place_amplifiers(lengths, max_length_without):
    # To be used as final result
    actual_lengths = []
    # Positions of the amplifiers
    pos_amplifiers = []
    # Index to identify the position
    i = 0
    # Check every length
    for length in lengths:
        i += 1
        # If the length is below the max length without amplifier
        if length < max_length_without:
            # add the length as is
            actual_lengths.append(length)
        else:
            # otherwise, select a factor to divide the link in two parts
            # as a random value from a uniform variable (min 40%, max 60%)
            factor = random.uniform(0.4, 0.6)
            # Add both parts as 2 links
            actual_lengths.extend([length * factor, length * (1 - factor)])
            # Add the position as a place to locate an amplifier
            pos_amplifiers.append(i)
    return actual_lengths, pos_amplifiers


# Correct negative length of the final link by reducing it propotionally
# from the rest of the links
def correct_negative_pending(link_lengths, pending_length, offices):
    if pending_length < 0:
        # print("Negative value detected for pending length.")
        # print(link_lengths, pending_length)
        updated_link_lengths = [val - abs(2 * pending_length) / offices for val in link_lengths]
        link_lengths = updated_link_lengths
        link_lengths.append(abs(2 * pending_length))
        # print(link_lengths, pending_length)
    else:
        link_lengths.append(pending_length)
    return link_lengths


def print_ring_to_screen(link_lengths, office_names):
    for i in range(len(link_lengths)):
        print("[" + office_names[i] + "]--- ", end="")
        print(str(link_lengths[i]) + " ---", end="")
    print("[" + office_names[-1] + "]")
    return


def create_graph_rings(office_names, link_lengths):
    g = nx.Graph()
    g.add_node(office_names[0])
    for i in range(1, len(office_names)):
        g.add_node(office_names[i])
        g.add_edge(office_names[i - 1], office_names[i])

    # Add the distances as an edge attribute for the nodes
    nx.set_edge_attributes(g, dict(zip(g.edges, link_lengths)), 'weight')

    return g
