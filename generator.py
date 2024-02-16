import random
import numpy as np

from network import write_network_xls, color_nodes, write_network_xls_plus
import networkconstants as nc
import networkx as nx
from matplotlib import pyplot as plt


def write_network(filename, topo, distances, assigned_types, figure, node_sheet="nodes", link_sheet="links",
                  clusters=None, pos=None, reference_nodes=None, type_nw=nc.BACKBONE):
    figure.savefig(filename + ".png")
    x_coord = [0] * len(topo.nodes)
    y_coord = [0] * len(topo.nodes)
    if pos is not None:
        x_coord, y_coord = zip(*pos.values())
    return write_network_xls(filename, topo, distances, assigned_types, node_sheet, link_sheet, clusters, x_coord,
                             y_coord, reference_nodes, coord_type=type_nw)


def format_node_list(nodes):
    result = ""
    process = 0
    for i in nodes:
        if process != 0:
            result = result + ","
        process = process + 1
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


def print_ring_to_screen(link_lengths, office_names):
    for i in range(len(link_lengths)):
        print("[" + office_names[i] + "]--- ", end="")
        print(str(link_lengths[i]) + " ---", end="")
    print("[" + office_names[-1] + "]")
    return
