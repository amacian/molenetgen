import collections

import numpy as np
from scipy import spatial
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

import networkconstants as nc
import networkx as nx
import random
from scipy.spatial import distance
import pandas as pd
import os.path
import Texts_EN as texts
import math
from itertools import combinations, product


# Generate topology with the specific degrees and weights and the specified number of nodes
def gen_topology(degrees, weights, nodes):
    # Generate k random values with the weight frequencies of degrees
    # sequence = random.choices(degrees, weights=weights, k=nodes)
    float_per_degree = [weight * nodes / sum(weights) for weight in weights]
    nodes_per_degree = [int(round(float_degree)) for float_degree in float_per_degree]
    if sum(nodes_per_degree) < nodes: #TODO turn if into while
        decimal_part = [f % 1 for f in float_per_degree if f % 1 < 0.5]
        inc_index = decimal_part.index(max(decimal_part))
        nodes_per_degree[inc_index] += 1
    elif sum(nodes_per_degree) > nodes: #TODO turn if into while
        decimal_part = [f % 1 for f in float_per_degree if f % 1 > 0.5]
        red_index = decimal_part.index(min(decimal_part))
        nodes_per_degree[red_index] -= 1
    sequence = [degree for degree, val in zip(degrees, nodes_per_degree) for _ in range(val)]
    random.shuffle(sequence)
    # The sum of values must be even. If not, increase one of the smallest degree instances
    # print("Sum sequence: ", sum(sequence))
    if (sum(sequence) % 2) != 0:
        # print("Odd sequence")
        min_degree_idx = 0
        min_degree = 1000
        updated = False

        for i in range(len(sequence)):
            # Look for the first element with the smallest degree value and increase it, then break
            if sequence[i] == degrees[0]:
                sequence[i] += 1
                updated = True
                break
            elif min_degree > sequence[i]:
                min_degree = sequence[i]
                min_degree_idx = i
        if not updated:
            sequence[min_degree_idx] += 1
    # create the bidirectional network topology using the configuration model
    # it may create parallel edges and self-loops.
    g = nx.configuration_model(sequence)
    # g = nx.random_degree_sequence_graph(sequence)
    # remove parallel edges
    g = nx.Graph(g)
    # Remove self-loops
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


# Calculate the actual frequency of the different degrees in the list of degree_values.
def count_degree_freqs(degree_values):
    counter = collections.Counter(degree_values)
    sorted_values = [counter[x] for x in sorted(counter.keys())]
    frequencies = [100 * val // sum(sorted_values) for val in sorted_values]
    return frequencies


# Calculate distances and scale them to the expected range of values
def calculate_edge_distances(graph, positions, max_limit):
    distances = [distance.euclidean(positions[u], positions[v]) for u, v in graph.edges]
    factor = max_limit / max(distances)
    actual_distances = [int(dist * factor) for dist in distances] # TODO: Round to 2 dec
    return actual_distances


# Rename nodes combining Type and index
def rename_nodes(assigned_types, types, special_rename_type=None, explicit_values=None, add_prefix=""):
    indexes = dict(zip(types.code, [1] * len(types)))
    names, indexes = rename_nodes_idx(assigned_types, indexes, special_rename_type, explicit_values, add_prefix)
    return names


# Rename nodes combining Type and index and receiving initial index per type
def rename_nodes_idx(assigned_types, indexes, special_rename_type=None, explicit_values=None, add_prefix=""):
    names = [''] * len(assigned_types)
    for i in range(len(assigned_types)):
        if assigned_types[i] == special_rename_type:
            names[i] = explicit_values[indexes[assigned_types[i]] - 1]
        else:
            names[i] = assigned_types[i] + add_prefix + str(indexes[assigned_types[i]])
        indexes[assigned_types[i]] += 1
    return names, indexes


# Write Excel file with two sheets
def write_network_xls(file, graph, distances, types, node_sheet, link_sheet, macro_region=None, x_coord=None,
                      y_coord=None, reference_nodes=None, coord_type=nc.BACKBONE, households=None, macro_cells=None,
                      small_cells=None, regional_twins=None, national_twins=None, link_capacities=None):
    x_coord_back = [0] * len(graph.nodes)
    y_coord_back = [0] * len(graph.nodes)
    x_coord_m_core = [0] * len(graph.nodes)
    y_coord_m_core = [0] * len(graph.nodes)

    regional_ref_nodes = [val for val in graph.nodes]

    match coord_type:
        case nc.METRO_CORE:
            x_coord_m_core = x_coord
            y_coord_m_core = y_coord
        case nc.METRO_AGGREGATION:
            if reference_nodes is not None:
                regional_ref_nodes = reference_nodes
                reference_nodes = [''] * len(graph.nodes)
        case _:
            x_coord_back = x_coord
            y_coord_back = y_coord
            if reference_nodes is None:
                reference_nodes = [val for val in graph.nodes]

    if macro_region is None:
        macro_region = [-1] * len(graph.nodes)

    if reference_nodes is None:
        reference_nodes = [''] * len(graph.nodes)

    if national_twins is None:
        national_twins = [''] * len(graph.nodes)

    if regional_twins is None:
        regional_twins = [''] * len(graph.nodes)

    if households is None:
        households = [12000] * len(graph.nodes)

    if macro_cells is None:
        macro_cells = [8] * len(graph.nodes)

    if small_cells is None:
        small_cells = [24] * len(graph.nodes)

    if link_capacities is None:
        link_capacities = [100] * len(graph.edges)

    nodes_df = pd.DataFrame({nc.XLS_NODE_NAME: [val for val in graph.nodes],
                             nc.XLS_CO_TYPE: types,
                             nc.XLS_REF_RCO: regional_ref_nodes,  # [val for val in graph.nodes],
                             nc.XLS_REF_NCO: reference_nodes,
                             nc.XLS_HOUSE_H: households,
                             nc.XLS_MACRO_C: macro_cells,
                             nc.XLS_SMALL_C: small_cells,
                             nc.XLS_TWIN_RCO: regional_twins,
                             nc.XLS_TWIN_NCO: national_twins,
                             nc.XLS_CLUSTER: macro_region,
                             nc.XLS_X_BACK: x_coord_back,
                             nc.XLS_Y_BACK: y_coord_back,
                             nc.XLS_X_MCORE: x_coord_m_core,
                             nc.XLS_Y_MCORE: y_coord_m_core,
                             })
    links_df = pd.DataFrame({nc.XLS_SOURCE_ID: [u for u, v in graph.edges],
                             nc.XLS_DEST_ID: [v for u, v in graph.edges],
                             nc.XLS_DISTANCE: distances,
                             nc.XLS_CAPACITY_GBPS: link_capacities}
                            )
    if os.path.isfile(file):
        res, message, nodes_df, links_df = merge_data_frames_to_xls(file, node_sheet, link_sheet, nodes_df, links_df,
                                                                    coord_type)
        if not res:
            return res, message
        with pd.ExcelWriter(file, engine='openpyxl') as writer:
            nodes_df.to_excel(writer, sheet_name=node_sheet, index=False)
            links_df.to_excel(writer, sheet_name=link_sheet, index=False)
            # with pd.ExcelWriter(file, engine='openpyxl', mode="a", if_sheet_exists='overlay') as writer:
            '''book = load_workbook(file)
            # writer.book=book
            writer.sheets.update(dict((ws.title, ws) for ws in book.worksheets))'''

            '''nodes_df.to_excel(writer, sheet_name=node_sheet, index=False, header=False,
                              startrow=writer.sheets[node_sheet].max_row)'''
            '''links_df.to_excel(writer, sheet_name=link_sheet, index=False, header=False,
                              startrow=writer.sheets[link_sheet].max_row)'''
    else:
        with pd.ExcelWriter(file, engine='openpyxl') as writer:
            nodes_df.to_excel(writer, sheet_name=node_sheet, index=False)
            links_df.to_excel(writer, sheet_name=link_sheet, index=False)

    return True, ""


# When creating s network, a node might belong to backbone, metro core and metro aggregation networks
# Read and update the information already available at the file so it can be written altogether
# file: filename to be read
# node_sheet: name of the excel sheet for nodes
# link_sheet: name of the excel sheet for links
# nodes_df: data frame with nodes
# links_df: data frame with links
# coord_type: type of segment being created (backbone, metro core, metro aggregation).
def merge_data_frames_to_xls(file, node_sheet, link_sheet, nodes_df, links_df, coord_type):
    ex_nodes = pd.read_excel(file, node_sheet, engine="openpyxl")
    existing_links = pd.read_excel(file, link_sheet, engine="openpyxl")
    node_names = nodes_df[nc.XLS_NODE_NAME]
    same = pd.merge(node_names, ex_nodes, on=[nc.XLS_NODE_NAME])
    link_ids = existing_links[[nc.XLS_SOURCE_ID, nc.XLS_DEST_ID]]
    same_links = pd.merge(link_ids, links_df, on=[nc.XLS_SOURCE_ID, nc.XLS_DEST_ID])

    if len(same_links) > 0:
        return False, texts.LINK_EXISTS, nodes_df, links_df

    # We are generating a backbone network, but there is already one defined
    if coord_type == nc.BACKBONE and max(ex_nodes[nc.XLS_X_BACK]) > 0:
        return False, texts.BACKBONE_EXISTS, nodes_df, links_df

    # We are generating a metro core network but the node was already used for another macro region
    if coord_type == nc.METRO_CORE and max(same[nc.XLS_X_MCORE]) > 0:
        return False, texts.METRO_CORE_CONFLICT, nodes_df, links_df

    if coord_type == nc.METRO_AGGREGATION:
        first_lco = nodes_df[nc.XLS_NODE_NAME][1]
        if first_lco in same[nc.XLS_NODE_NAME]:
            return False, texts.METRO_AGG_CONFLICT, nodes_df, links_df
        first_regional = nodes_df[nc.XLS_NODE_NAME][0]
        last_regional = nodes_df[nc.XLS_NODE_NAME][len(nodes_df) - 1]
        if first_regional in list(same[nc.XLS_NODE_NAME]):
            first_regional_ref_nco = ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == first_regional, nc.XLS_REF_NCO].iloc[0]
            nodes_df.loc[nodes_df[nc.XLS_REF_RCO] == first_regional, nc.XLS_REF_NCO] = first_regional_ref_nco
            nodes_df = nodes_df[nodes_df[nc.XLS_NODE_NAME] != first_regional]
        if last_regional in list(same[nc.XLS_NODE_NAME]):
            last_regional_ref_nco = ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == last_regional, nc.XLS_REF_NCO].iloc[0]
            nodes_df.loc[nodes_df[nc.XLS_REF_RCO] == last_regional, nc.XLS_REF_NCO] = last_regional_ref_nco
            nodes_df = nodes_df[nodes_df[nc.XLS_NODE_NAME] != last_regional]
    # We are creating a metro core network and one of the nodes already exists (probably mapped from
    # the backbone network). Let's add information related to the coordinates and update other fields
    if coord_type == nc.METRO_CORE and len(same) > 0:
        # nc.XLS_X_MCORE, nc.XLS_Y_MCORE]
        for name in same[nc.XLS_NODE_NAME]:
            # TODO read all at once and write them together
            hh = nodes_df.loc[nodes_df[nc.XLS_NODE_NAME] == name, nc.XLS_HOUSE_H].iloc[0]
            mc = nodes_df.loc[nodes_df[nc.XLS_NODE_NAME] == name, nc.XLS_MACRO_C].iloc[0]
            sc = nodes_df.loc[nodes_df[nc.XLS_NODE_NAME] == name, nc.XLS_SMALL_C].iloc[0]
            x_coord = nodes_df.loc[nodes_df[nc.XLS_NODE_NAME] == name, nc.XLS_X_MCORE].iloc[0]
            y_coord = nodes_df.loc[nodes_df[nc.XLS_NODE_NAME] == name, nc.XLS_Y_MCORE].iloc[0]

            ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == name, nc.XLS_HOUSE_H] = hh
            ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == name, nc.XLS_MACRO_C] = mc
            ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == name, nc.XLS_SMALL_C] = sc
            ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == name, nc.XLS_X_MCORE] = x_coord
            ex_nodes.loc[ex_nodes[nc.XLS_NODE_NAME] == name, nc.XLS_Y_MCORE] = y_coord

            nodes_df = nodes_df[nodes_df[nc.XLS_NODE_NAME] != name]

        ex_nodes.update(ex_nodes, overwrite=True)

    nodes_df = pd.concat([ex_nodes, nodes_df])
    links_df = pd.concat([existing_links, links_df])
    return True, texts.NO_ERROR, nodes_df, links_df


# Write Excel file with two sheets
def write_network_xls_plus(file, graph, distances, types, node_sheet, link_sheet, reference_nodes):
    nodes_df = pd.DataFrame({'Nodes': [val for val in graph.nodes],
                             'Degrees': [val for u, val in graph.degree],
                             'Type': types,
                             'Reference': reference_nodes})
    links_df = pd.DataFrame({'Node1': [u for u, v in graph.edges],
                             'Node2': [v for u, v in graph.edges],
                             'Distance': distances}
                            )
    with pd.ExcelWriter(file, engine='openpyxl') as writer:
        nodes_df.to_excel(writer, sheet_name=node_sheet, index=False)
        links_df.to_excel(writer, sheet_name=link_sheet, index=False)
    return True


# Extract proportion of distances in each defined range
def count_distance_ranges(distances, upper_limits):
    occurrences = [0] * len(upper_limits)
    for dist in distances:
        i = 0
        for limit in upper_limits:
            if dist < limit:
                occurrences[i] += 1
                break
            i += 1
    percentages = [round(100 * occurrence / sum(occurrences), 2) for occurrence in occurrences]
    return percentages


def set_missing_types_colors(assigned_types, dict_colors: dict):
    unique_types = set(assigned_types)
    for a_type in unique_types:
        color = dict_colors.get(a_type, None)
        if color is None:
            dict_colors[a_type] = 'b'
    return dict_colors


def color_nodes(assigned_types, dict_colors):
    dict_colors = set_missing_types_colors(assigned_types, dict_colors)
    return [dict_colors[assigned_type] for assigned_type in assigned_types]


def format_distance_limits(distances, upper_limits, req_perc=None):
    init_dist = 0
    text = "% of distances per range (kms):\n"
    if req_perc is None:
        req_perc = count_distance_ranges(distances, upper_limits)
    else:
        req_perc = [round(req * 100, 1) for req in req_perc]
    for upper, perc, req in zip(upper_limits, (count_distance_ranges(distances, upper_limits)), req_perc):
        text += (str(init_dist) + "-" + str(upper) + ":\t" + str(perc) + "% (" + str(req) + ")\n")
        init_dist = upper
    text += "Max length = " + str(max(distances)) + "\n"
    return text


def relocate_problematic(problematic, final_coordinates, dist_min):
    res = True
    # for each pair in problematic, try to relocate the first element. If fails, try with the second.
    for a, b in problematic:
        # One of them was relocated in a different iteration so it is no longer a problem
        if a not in final_coordinates or b not in final_coordinates:
            continue
        check_coordinates = [coor for coor in final_coordinates if coor != a and coor != b]
        a_x, a_y = a
        b_x, b_y = b
        diff = np.subtract(a, b)

        if abs(diff[0]) < dist_min:
            a_x = int(a[0] + math.copysign(1, diff[0]) * (dist_min - abs(diff[0])))
            b_x = int(b[0] - math.copysign(1, diff[0]) * (dist_min - abs(diff[0])))
        if abs(diff[1]) < dist_min:
            a_y = int(a[1] + math.copysign(1, diff[1]) * (dist_min - abs(diff[1])))
            b_y = int(b[1] - math.copysign(1, diff[1]) * (dist_min - abs(diff[1])))
        all_combinations = [(a_x, a[1]), (a[0], a_y), (a_x, a_y),
                            (b_x, b[1]), (b[0], b_y), (b_x, b_y)]
        new_problem = [(s, t) for s, t in product(check_coordinates, all_combinations) if
                       abs(s[0] - t[0]) < dist_min and abs(s[1] - t[1]) < dist_min]
        no_problem = [coor for coor in all_combinations if coor not in [tup[1] for tup in new_problem]]
        if len(no_problem) == 0:
            return final_coordinates, False
        result = no_problem[0]
        result_was_a = np.linalg.norm(np.subtract(result, a)) < np.linalg.norm(np.subtract(result, b))
        old_value = a if result_was_a else b
        final_coordinates[final_coordinates.index(old_value)] = result

    final_validation = [(a, b) for a, b in combinations(final_coordinates, 2) if
                        abs(a[0] - b[0]) < dist_min and abs(a[1] - b[1]) < dist_min]
    if len(final_validation) > 0:
        print(final_validation)
        res = False

    return final_coordinates, res


# Generate a graph trying to imitate the work from
# C. Pavan, R. M. Morais, J. R. Ferreira da Rocha and A. N. Pinto,
# "Generating Realistic Optical Transport Network Topologies,"
# in Journal of Optical Communications and Networking, vol. 2, no. 1, pp. 80-90,
# January 2010, doi: 10.1364/JOCN.2.000080.
# We adapt it slightly as we are receiving different parameters
# Part of the process is based on Waxman work
def gen_waxman_paven_topology(nodes, regions, beta=0.4, alpha=0.4,
                              squared_factor=4, dist_factor=0.63):
    # The graph was succesfully generated
    res = True

    # Distributing regions in rows and columns following the Paven et al. description
    actual_regions = regions
    rows = find_max_prime(actual_regions)
    if rows == actual_regions:
        actual_regions += 1
        rows = find_max_prime(actual_regions)
    columns = int(actual_regions / rows)

    # Instead of defining a global area and calculating area, rows and columns per region
    # a factor (a squared integer) is used to define the area of each region and the
    # total area
    region_area = rows * columns * squared_factor
    total_area = region_area * actual_regions
    # Rows and columns for the regions are calculated based on the columns and the squared factor
    region_rows = int(columns * math.sqrt(squared_factor))
    region_columns = int(rows * math.sqrt(squared_factor))

    # Instead of receiving the minimum distance between nodes, we calculate it based on a
    # dist_factor parameter and the area available per node
    # minimum distance^2 must be lower than total_area/nodes
    dist_min = math.floor(math.sqrt(total_area / nodes) * dist_factor)
    # Maximum number of nodes per region
    max_nodes_region = math.floor(region_area / (dist_min ** 2))
    # A too high value would make the process too unbalanced
    if max_nodes_region > 2 * nodes / actual_regions:
        max_nodes_region = math.floor(2 * nodes / actual_regions)

    # Define how many nodes will be placed in each region
    nodes_regions = generate_node_distribution_paven(nodes, regions, max_nodes_region)

    # Create an empty graph and the number of nodes received as parameters
    # The initial name of each node is its number in the sequence.
    graph = nx.Graph()
    graph.add_nodes_from([idx for idx in range(nodes)])

    # The positions (coordinates) of the nodes per region and the generated names will
    # be stored in dictionaries
    pos = {}
    names = {}
    # Idx will be used to generate the names and will be incremented
    idx = 0
    # Go region by region
    for i in range(regions):
        num = nodes_regions[i]
        # Create a list with all the possible coordinates in the region
        coordinates = [(x, y) for x in range(1, region_rows + 1) for y in range(1, region_columns + 1)]
        # Depending on the number of nodes the process to allocate them differs
        match num:
            case 1:
                # For a single node, just select the position randomly
                x = random.randint(1, region_rows)
                y = random.randint(1, region_columns)
                # Assign the coordinates
                region_coord = [(x, y)]
            case 2:
                # For a region with two nodes, the position of the first node is
                # selected randomly in the left part of the region.
                x1 = random.randint(1, region_rows)
                y1 = random.randint(1, math.floor(region_columns / 2))
                # Remove all the coordinates at less than the minimum distances from the first node
                coordinates = remove_coordinates_at_distance(coordinates, dist_min, x1, y1)
                # Select randomly the coordinates for the second node among the ones remaining
                (x2, y2) = random.choice(coordinates)
                # Assign the coordinates
                region_coord = [(x1, y1), (x2, y2)]
                # Create a link between both nodes
                graph.add_edge(idx, idx + 1)
            case 3:
                # For a region with two nodes, the position of the first node is
                # selected randomly in the left part of the region.
                x1 = random.randint(1, region_rows)
                y1 = random.randint(1, math.floor(region_columns / 2))
                # Repeat the process of searching for the other two nodes until it is successful
                while True:
                    # Remove all the coordinates at less than the minimum distances from the first node
                    new_coord = remove_coordinates_at_distance(coordinates, dist_min, x1, y1)
                    # Select randomly the coordinates for the second node among the ones remaining
                    (x2, y2) = random.choice(new_coord)
                    # Remove also all the coordinates at less than the minimum distances from the second node
                    new_coord = remove_coordinates_at_distance(new_coord, dist_min, x2, y2)
                    # If there is at least one coordinate available, we select one randomly for the
                    # third element
                    if len(new_coord) != 0:
                        (x3, y3) = random.choice(new_coord)
                        break
                # Assign the coordinates
                region_coord = [(x1, y1), (x2, y2), (x3, y3)]
                # Create a cycle of links among the three nodes.
                edges = [(idx, idx + 1), (idx, idx + 2), (idx + 1, idx + 2)]
                # Add the edges to the topology
                graph.add_edges_from(edges)
            case _:
                # For more than 3 elements, use a function to generate the coordinates
                region_coord = create_positions_paven(coordinates, num, dist_min)
                # Create a cycle that interconnects all the nodes and store the edges
                existing_edges = region_cycle(region_coord, range(idx, idx + num), graph)
                # Generate additional nodes using the Waxman probability
                # For L (max_distance) we choose a pre-defined factor
                waxman_edges(region_coord, range(idx, idx + num), alpha, beta, graph,
                             max_distance=rows * region_rows * 1.4, existing_edges=existing_edges)
        # We have processed "num" additional nodes.
        idx += num
        # Store the coordinates and the names
        pos[i] = region_coord
        # if rename:
        names[i] = ["Reg" + str(i) + "_" + str(j) for j in range(num)]
        # else:
        #     names[i] = [j for j in range(num)]

    # Once the whole process finishes, rename the nodes
    graph = nx.relabel_nodes(graph, mapping=dict(zip(graph, [i for element in names.values() for i in element])))
    final_coordinates = []
    # Translate the local coordinates within a region to the global coordinate in the area
    for i in range(regions):
        pos[i] = move_coordinates_to_global_area(pos, i, columns, region_rows, region_columns)
        final_coordinates.extend(pos[i])
    # Connect adjacent regions based on Waxman and some additional restrictions
    for i in range(regions):
        connect_region(i, pos, columns, graph, names, alpha, beta)

    # Set additional connections to ensure survivability
    for i in range(regions):
        ensure_survivability(i, pos, graph, names)

    problematic = [(a, b) for a, b in combinations(final_coordinates, 2) if
                   abs(a[0] - b[0]) < dist_min and abs(a[1] - b[1]) < dist_min]
    final_coordinates, res = relocate_problematic(problematic, final_coordinates, dist_min)

    # Normalization if needed
    # norm_row_factor = rows * region_rows / 2
    # norm_column_factor = columns * region_columns / 2
    # normalized_coordinates = [[-1 + i / norm_row_factor, -1 + j / norm_column_factor] for i, j in final_coordinates]

    # Final survivability check and edge addition
    survival_edges = list(nx.k_edge_augmentation(graph, 2))
    graph.add_edges_from(survival_edges)

    # Set colors based in regions
    # colors = [i * 2 for i, count in enumerate(nodes_regions, 1) for _ in range(count)]
    # nx.draw(graph, pos=indexed_pos, node_color=colors, with_labels=True)
    # plt.show()

    # Reformat the positions to the format expected by networkx
    pos = {name: coordinate for name, coordinate in zip(graph.nodes, final_coordinates)}

    # Check for node survivability
    if len(list(nx.minimum_node_cut(graph))) == 1:
        return graph, pos, False

    # figure.savefig("/Users/macian/Documents/test.png")
    return graph, pos, res


def connect_region(n_region, region_elements, columns, graph, names_region, alpha, beta):
    coords_region = region_elements[n_region]

    region_row, region_column = region_to_row_column(n_region, columns)
    region_bottom = None if region_row == 0 else n_region - columns
    region_right = None if region_column == columns - 1 else n_region + 1

    if region_right is not None and region_right in region_elements:
        coords_region_right = region_elements[region_right]

        waxman_edges(coords_region, names_region[n_region], alpha, beta, graph,
                     existing_edges=list(graph.edges),
                     other_coordinates=coords_region_right,
                     second_names=names_region[region_right])
    if region_bottom is not None and region_bottom in region_elements:
        coords_region_bottom = region_elements[region_bottom]

        waxman_edges(coords_region, names_region[n_region], alpha, beta, graph,
                     existing_edges=list(graph.edges),
                     other_coordinates=coords_region_bottom,
                     second_names=names_region[region_bottom])


def ensure_survivability(n_region, region_elements, graph, names_region):
    # If the region is  one of the automatically incorporated to build the rectangle area
    # it will not hold any node
    if n_region not in names_region:
        return
    # Get the names of the nodes in the region
    names_nodes = names_region[n_region]
    # Get how many nodes are in the region
    length_region = len(names_nodes)

    # Get all the connections between nodes in the region and external nodes
    connections = [(i, j) for i, j in list(graph.edges) if i in names_nodes and j not in names_nodes]
    connections.extend([(j, i) for i, j in list(graph.edges) if i not in names_nodes and j in names_nodes])

    # Get the sources, i.e. the nodes in the region connected to others
    sources = set([i for i, j in connections])
    # Get the destinations, i.e. the nodes in other regions connected to this region
    destinations = set([j for i, j in connections])

    # If there are at least 2 nodes in the region connected to 2 external nodes, it is survivable
    if len(sources) > 1 and len(destinations) > 1:
        return

    # Get all the nodes from connected regions
    nodes_connected_regions = [names_region[region] for region, values in names_region.items()
                               if not destinations.isdisjoint(values)]

    # Get the coordinates of the region
    coord_region = region_elements[n_region]
    # Get the coordinates of the nodes from other regions
    coordinates_other = [(i, j) for region, values in region_elements.items() for i, j in values if region != n_region]
    # Get the names of the nodes from other regions
    names_other = [name for region, names in names_region.items() for name in names if region != n_region]

    # Create a KDTree to query for nearest neighbors
    tree = spatial.KDTree(coordinates_other)
    pairs = []
    # We will traverse the region checking its nodes
    for pos in range(length_region):
        # Get the next node in the region
        name = names_nodes[pos]

        # If the region has more than 1 element and this is already connected, go for a different one
        if len(connections) == 1 and connections[0][0] == name and length_region > 1:
            continue

        # Retrieve more than one element to avoid retrieving connection to an existing region or
        # if it is a single node, to the destination
        k = 10
        # Get 10 results of neighbors (distances and index in coordinates_other/names_other) from the current node
        distance, indexes = tree.query(coord_region[pos], k)

        # Traverse the retrieve destinations
        for i in indexes:
            # Create a connection between the node in the region and the node found
            edge = (name, names_other[i])

            # In case that there was already a connection in a region with 1 node
            # could be to this nearest neighbor
            if length_region == 1 and edge in connections:
                continue

            # Do not send the connection to the same region to the existing connection
            if len(connections) == 1 and names_other[i] in nodes_connected_regions:
                continue

            # Add this edge as a possible connection
            pairs.append(edge)

    # We will select a subset of the pairs found randomly
    selected_edges = []
    same_region = True

    # As we might need to select 2 elements, we will try to choose them from different regions
    while same_region:
        # Take as much as 2 connections, 1 if there is already an existing one
        # 1 if there is only 1 pair found
        selected_edges = random.sample(pairs, k=min(2 - min(1, len(connections)), len(pairs)))

        # Select the source and destination of the selected edges
        destinations = set([j for i, j in selected_edges])
        sources = set([i for i, j in selected_edges])

        connected_regions = [region for region, values in names_region.items()
                             if not destinations.isdisjoint(values)]
        # print(connected_regions)

        if (len(selected_edges) == len(connected_regions)
                and (len(sources) == len(selected_edges) or length_region == 1)):
            same_region = False

    graph.add_edges_from(selected_edges)


def region_cycle(coordinates, node_names, graph):
    # find centre point (centre of gravity)
    x0 = sum(x for x, _ in coordinates) / len(coordinates)
    y0 = sum(y for _, y in coordinates) / len(coordinates)

    # calculate angles and create list of tuples(index, angle)
    radial = [(i, math.atan2(y - y0, x - x0)) for i, (x, y) in enumerate(coordinates)]

    # sort by angle
    radial.sort(key=lambda x: x[1])

    # extract indices
    ring = [a[0] for a in radial]

    edges = [(node_names[ring[i]], node_names[ring[i + 1]]) for i in range(0, len(ring) - 1)]
    edges.append((node_names[ring[0]], node_names[ring[len(ring) - 1]]))
    graph.add_edges_from(edges)
    return edges


def waxman_edges(coordinates, node_names, alpha, beta, graph,
                 max_distance=None, existing_edges=None, other_coordinates=None,
                 second_names=None):
    if max_distance is None:
        if other_coordinates is None:
            max_distance = max(math.dist(x, y) for x, y in combinations(coordinates, 2))
        else:
            max_distance = max(math.dist(x, y) for x, y in product(coordinates, other_coordinates))
    pairs = None
    if other_coordinates is None:
        pairs = list(filter(lambda pair: random.random() < beta * math.exp(-math.dist(*pair) / (alpha * max_distance)),
                            combinations(coordinates, 2)))
        second_coordinates = coordinates
        second_names = node_names
    else:
        pairs = list(filter(lambda pair: random.random() < beta * math.exp(-math.dist(*pair) / (alpha * max_distance)),
                            list(product(coordinates, other_coordinates))))
        second_coordinates = other_coordinates
    positions = [(coordinates.index(pair[0]), second_coordinates.index(pair[1])) for pair in pairs]
    edges = [(node_names[i], second_names[j]) for (i, j) in positions]
    filtered_edges = [i for i in edges if i not in existing_edges]
    # print(positions, " --> ", edges)
    graph.add_edges_from(filtered_edges)
    return filtered_edges


def generate_node_distribution_paven(nodes, regions, max_nodes_region):
    nodes_regions = [random.randint(1, max_nodes_region) for _ in range(regions - 1)]
    nodes_last_region = nodes - sum(nodes_regions)

    if nodes_last_region < 1:
        while nodes_last_region < 1:
            indexes = [idx for idx, elem in enumerate(nodes_regions) if elem != min(nodes_regions)]
            if len(indexes) == 0:
                indexes = range(0, len(nodes_regions) - 1)
            for i in indexes:
                nodes_regions[i] -= 1
                nodes_last_region += 1
                if nodes_last_region == 1:
                    break
    elif nodes_last_region > max_nodes_region:
        while nodes_last_region > max_nodes_region:
            indexes = [idx for idx, elem in enumerate(nodes_regions) if elem != max_nodes_region]
            for i in indexes:
                nodes_regions[i] += 1
                nodes_last_region -= 1
                if nodes_last_region == max_nodes_region:
                    break
    nodes_regions.append(nodes_last_region)

    return nodes_regions


def create_positions_paven(coordinates, elements, min_dist):
    completed = False
    coord = []
    while not completed:
        coord.append((min_dist, min_dist))
        new_coord = remove_coordinates_at_distance(coordinates, min_dist, min_dist, min_dist)
        completed = True
        for i in range(elements - 1):
            # If we cannot fit more elements in the region, repeat the process with a different random selection
            if len(new_coord) == 0:
                completed = False
                coord.clear()
                print("failed when allocating element: ", i)
                break
            (x2, y2) = random.choice(new_coord)
            coord.append((x2, y2))
            new_coord = remove_coordinates_at_distance(new_coord, min_dist, x2, y2)

    return coord


def remove_coordinates_at_distance(coordinates, distance, selected_x, selected_y):
    new_coord = [(i, j) for i, j in coordinates if i <= selected_x - distance or i >= selected_x + distance or
                 j <= selected_y - distance or j >= selected_y + distance]
    return new_coord


def find_max_prime(num):
    max_prime = -1

    # Print the number of 2s that divide n
    while num % 2 == 0:
        max_prime = 2
        num >>= 1  # equivalent to n /= 2

    for i in range(3, int(math.sqrt(num)) + 1, 2):
        while num % i == 0:
            max_prime = i
            num = num / i

    if num > 2:
        max_prime = num

    return int(max_prime)


def move_coordinates_to_global_area(coordinates, region, columns, region_rows, region_columns):
    # Position in the Y axis in terms of regions (e.g. second row of regions)
    pos_region_row, pos_region_column = region_to_row_column(region, columns)

    # The actual row will come from the previous regions in the axis and the number of rows per region
    actual_row_factor = pos_region_row * region_rows
    actual_column_factor = pos_region_column * region_columns
    region_coordinates = coordinates[region]
    global_ref_coordinates = [(i + actual_row_factor, j + actual_column_factor) for i, j in region_coordinates]
    return global_ref_coordinates


def region_to_row_column(region, columns):
    # Position in the Y axis in terms of regions (e.g. second row of regions)
    region_row = region // columns
    # Position in the X axis in terms of regions (e.g. second column of regions)
    region_column = region % columns
    return region_row, region_column


def check_metrics(elements, weight, real_distribution, perc=False):
    to_compare = []
    generated_weight = []
    if perc:
        to_compare = real_distribution
        generated_weight = real_distribution
    else:
        real_occurrence = {v: real_distribution.count(v) for v in set(real_distribution)}
        total_occurrence = sum(real_occurrence.values())
        generated_weight = {v: real_occurrence[v] / total_occurrence for v in real_occurrence}
        to_compare = [generated_weight.get(v, 0) for v in elements]

    mae = mean_absolute_error(weight, to_compare)
    mape = mean_absolute_percentage_error(weight, to_compare)
    rsme = root_mean_squared_error(weight, to_compare)

    return mae, mape, rsme, generated_weight


def optimize_distance_ranges(upper_distances, weights, distances):
    upper_limit = max(upper_distances)
    new_distances = distances.copy()
    step = upper_limit / 100
    ref_mape = 1000
    while upper_limit > 0:
        factor = upper_limit / max(distances)
        alt_distances = [distance * factor for distance in distances]
        actual_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_distances)]
        mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_distances, weights,
                                                                       actual_distance_weight, perc=True)
        if mape_distance < ref_mape:
            new_distances = [i if i > 0 else 0.01 for i in alt_distances]
            ref_mape = mape_distance
        upper_limit = upper_limit - step

    return new_distances


# Read the network topology from an XLS file
# file - the filename
# type - backbone or metro
def read_network_xls(file, ntype=nc.BACKBONE):
    # Dataframes to read the info from file
    nodes_df = None
    links_df = None
    # Result
    read = False
    # Read the Nodes and Links excel sheets if the file exists
    if os.path.isfile(file):
        with pd.ExcelFile(file) as xls:
            nodes_df = pd.read_excel(xls, sheet_name=nc.NODES_EXCEL_NAME)
            links_df = pd.read_excel(xls, sheet_name=nc.LINKS_EXCEL_NAME)
            read = True
    # create the topology and the arrays for distances, types, clusters and the auxiliary coordinates
    topo = nx.Graph()
    distances = []
    assigned_types = []
    clusters = []
    coord = []

    # Depending on the type of topology read different coordinates
    if ntype == nc.BACKBONE:
        nodes_df = nodes_df[(nodes_df[nc.XLS_X_BACK] != 0) & (nodes_df[nc.XLS_Y_BACK] != 0)]
    elif ntype == nc.METRO_CORE:
        nodes_df = nodes_df[(nodes_df[nc.XLS_X_MCORE] != 0) & (nodes_df[nc.XLS_X_MCORE] != 0)]
    links_df = links_df[links_df[nc.XLS_SOURCE_ID].isin(list(nodes_df[nc.XLS_NODE_NAME]))]
    links_df = links_df[links_df[nc.XLS_DEST_ID].isin(list(nodes_df[nc.XLS_NODE_NAME]))]
    # Get the ids of the nodes
    names = nodes_df[nc.XLS_NODE_NAME]
    # Add all the nodes to the topology
    topo.add_nodes_from(names)
    # Add all the links to the topology
    topo.add_edges_from(list(zip(links_df[nc.XLS_SOURCE_ID], links_df[nc.XLS_DEST_ID])))
    # Include the distances
    distances.extend(links_df[nc.XLS_DISTANCE])
    # Retrieve the types
    assigned_types.extend(nodes_df[nc.XLS_CO_TYPE])
    # Get the clusters
    clusters.extend(nodes_df[nc.XLS_CLUSTER])

    # Depending on the type of topology read different coordinates
    if ntype == nc.BACKBONE:
        coord = list(zip(nodes_df[nc.XLS_X_BACK], nodes_df[nc.XLS_Y_BACK]))
    elif ntype == nc.METRO_CORE:
        coord = list(zip(nodes_df[nc.XLS_X_MCORE], nodes_df[nc.XLS_Y_MCORE]))
    # Generate the position data structure from the coordinates and name
    pos = {name: np.array([x, y]) for name, (x, y) in zip(names, coord)}
    # Return the results
    return read, topo, distances, assigned_types, pos, clusters


last_vals = None


# Optimization function to approximate distances to expected ranges
def opt_function(pos, topo, upper_distances, weights):
    # rearrange the position in the proper format
    pos = pos.reshape((len(topo.nodes), 2))
    # Get the node names
    nodes = list(topo.nodes)
    # Calculate the distance values in the original axis
    distances = [distance.euclidean(pos[nodes.index(u)], pos[nodes.index(v)]) for (u, v) in list(topo.edges)]
    # Re-dimension by applying a factor based on the maximum distance
    factor = max(upper_distances) / max(distances)
    actual_distances = [(dist * factor) for dist in distances]
    # Calculate the distribution into ranges
    actual_distance_weight = [dist for dist in count_distance_ranges(actual_distances, upper_distances)]
    # Get the absolute difference between expected and current ranges
    res = sum([abs(u - v) for u, v in zip(actual_distance_weight, weights)])
    return res
