from abc import ABC, abstractmethod

from scipy import spatial
import networkconstants as nc
import random
import networkx as nx
from network import gen_topology, calculate_edge_distances, rename_nodes, color_nodes
import pandas as pd


class MetroCoreGenerator(ABC):
    # Generate a region of a Metro Core Network
    # degrees and weights: node connection degrees and percentages per connectivity
    # upper_limits: upper limits per distance range
    # types: type value per node (national, regional...)
    # dict_colors: color assigned per type
    # algo: algorithm to create the distribution (spectral, kamada...)
    # national_nodes: list of national nodes if defined
    # add_prefix: additional prefix to incorporate to each non-national node
    @abstractmethod
    def generate_mesh(self, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                      national_nodes=[], add_prefix="", extra_node_info=None):
        pass

    # Function to define the parameters for the ring Metro-regional networks
    # n_rings number of rings in the structure (1, 2, 3, 4 or 6)
    # end_1 and end_2 name of both ends of the ring.
    # prefix to prepend to the regional nodes
    # initial index for the regional nodes
    # Variability the actual length of the links will be between avg_length [+- var * avg_length]
    @abstractmethod
    def generate_ring(self, n_rings, end_1, end_2, prefix='R',
                      initial_index=1, variability=0.1, dict_colors={}):
        pass

    @staticmethod
    def metro_assign_types(types):
        # print("metro_assign_types: ", types)
        sequence = [code for code, val in types[['code', 'number']].values for _ in range(val)]
        # Generate k random values with the weight frequencies of degrees
        random.shuffle(sequence)
        return sequence


class DefaultMetroCoreGenerator(MetroCoreGenerator):

    def generate_mesh(self, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                      national_nodes=[], add_prefix="", extra_node_info=None):

        # Split the generation into subnets so the connections are balanced for each region
        # With a NCO
        num_subnets = len(national_nodes)
        if num_subnets == 0:
            num_subnets = types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'].values[0]
            national_nodes = [nc.NATIONAL_CO_CODE + str(i + 1) for i in range(num_subnets)]

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
            sub_topo, assigned_types_sub = self.sub_metro(degrees, weights,
                                                          sum(subnet_numbers), types, subnet_numbers)

            # plt.rcParams["figure.figsize"] = [val for val in plt.rcParamsDefault["figure.figsize"]]

            # If it is the first sub topology, create the topology assigning it
            if topo is None:
                topo = sub_topo
            # Otherwise connect the subtopology by selecting 2 random elements of the
            # topology and 2 of the subtopology and connect them.
            else:
                topo = self.connect_sub_metro_regions(topo, sub_topo, degrees[0])
            # print("Total topo: ", len(topo.nodes))
            # Extend the types of the nodes with the new types obtained
            assigned_types.extend(assigned_types_sub)

        # Generate a sequence of colors for each node depending on the type
        colors = color_nodes(assigned_types, dict_colors)

        # Modify the node labels to name them as a combination of the type and an index
        name_nodes = rename_nodes(assigned_types, types, nc.NATIONAL_CO_CODE, national_nodes, add_prefix)
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

        # Define the upper bound for the distances. Could be the maximum limit
        # or something below that (e.g. 1/2 of the last range)
        corrected_max_upper = upper_limits[-1]
        # Scale the distances from the topology to the defined values
        distances = calculate_edge_distances(topo, pos, corrected_max_upper)
        # Check the generate percentages for each range
        # print(count_distance_ranges(distances, upper_limits))

        return topo, distances, assigned_types, pos, colors

    # Generate the subtopologies for the metro topology
    def sub_metro(self, degrees, weights, nodes, types, subnet_number):
        while True:
            # Call the function to generate the topology
            topo = gen_topology(degrees, weights, nodes)
            if not nx.is_connected(topo) or len(nx.minimum_node_cut(topo)) < 2:
                continue
            # Should be edge survivable, but just in case
            survival_edges = list(nx.k_edge_augmentation(topo, 2))
            topo.add_edges_from(survival_edges)

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
        assigned_types = MetroCoreGenerator.metro_assign_types(pd.DataFrame({'code': types.code,
                                                                             'number': subnet_number}))

        return topo, assigned_types

    # Connect nodes to other existing ones of a different topology. Increase degrees.
    def connect_sub_metro_regions(self, topology, sub_topology, min_degree):
        indexes = []
        low_degree = min_degree
        while True:
            temp = [index for index, element in topology.degree if element <= low_degree]
            indexes.extend(temp)
            if len(indexes) > 1:
                break
            low_degree += 1
        topo_cons = random.sample(indexes, k=2)
        relabelled = [val + len(topology.nodes) for val in sub_topology.nodes]

        indexes = []
        low_degree = min_degree
        while True:
            temp = [index + len(topology.nodes) for index, element in sub_topology.degree if element == low_degree]
            indexes.extend(temp)
            if len(indexes) > 1:
                break
            low_degree += 1
        subtopo_cons = random.sample(indexes, k=2)

        sub_topology = nx.relabel_nodes(sub_topology, dict(zip(sub_topology.nodes, relabelled)))
        topology = nx.union(topology, sub_topology)
        topology.add_edge(topo_cons[0], subtopo_cons[0])
        topology.add_edge(topo_cons[1], subtopo_cons[1])
        return topology

    def generate_ring(self, n_rings, end_1, end_2, prefix='R', initial_index=1, variability=0.1,
                      dict_colors={}):
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
        return self.create_ring_network_tel(n_rings, ring_config, end_1, end_2,
                                            prefix, initial_index, variability, dict_colors)

        # Function to create ring Metro-regional networks
        # n_rings number of rings in the structure (1, 2, 3, 4 or 6)
        # ring_config DataFrame structure formed by number of rings

    def create_ring_network_tel(self, n_rings, ring_config, end_1, end_2,
                                prefix, initial_index, variability, dict_colors):
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
            link_lengths = [random.uniform(range_ring[0], range_ring[1]) for _ in range(offices)]
            # Calculate the length of the final link
            pending_length = length - sum(link_lengths)
            # pending_length might be negative due to the way the lengths are built
            # correct it by reducing proportionally from the other lengths
            link_lengths = self.correct_negative_pending(link_lengths, pending_length, offices)
            # Shuffle the lengths so their distribution is random
            random.shuffle(link_lengths)
            # Calculate if amplifiers are needed and place them in the proper positions
            # splitting the links
            link_lengths, pos_amplifiers = self.place_amplifiers(link_lengths, dist_amplifiers)
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
            sub_ring = self.create_graph_rings(office_names, link_lengths)
            # merge the ring and the sub_ring
            ring = nx.compose(ring, sub_ring)

        # Retrieve the distances from the weight parameter of the edges
        distances = [val for key, val in nx.get_edge_attributes(ring, 'weight').items()]

        # Generate positions of the nodes based on the spectral/spring distribution
        # pos = nx.spectral_layout(topo)
        pos_loc = nx.spring_layout(ring)
        # pos_loc = nx.kamada_kawai_layout(ring)
        # Draw and save the figure
        # color_map = ['y' if node == end_1 or node == end_2 else 'c' for node in ring.nodes]

        # Reinitialize the figure for future use
        # plt.show()
        # plt.figure().clear()
        # Types will be NCO for the ends and RCO for the rest of the nodes
        # TODO pass it as an argument
        types = [nc.NATIONAL_CO_CODE if node == end_1 or node == end_2 else nc.REGIONAL_CO_CODE for node in ring.nodes]

        color_map = color_nodes(types, dict_colors)
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

        return ring, distances, types, pos_loc, color_map, reference_node

    # Correct negative length of the final link by reducing it proportionally
    # from the rest of the links
    def correct_negative_pending(self, link_lengths, pending_length, offices):
        if pending_length <= 0:
            # print("Negative value detected for pending length.")
            # print(link_lengths, pending_length)
            updated_link_lengths = [val - abs(2 * pending_length) / offices for val in link_lengths]
            link_lengths = updated_link_lengths
            link_lengths.append(abs(2 * pending_length))
            # print(link_lengths, pending_length)
        else:
            link_lengths.append(pending_length)
        return link_lengths

    # Method to place amplifiers in between nodes when the length is over the max allowed length
    def place_amplifiers(self, lengths, max_length_without):
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

    def create_graph_rings(self, office_names, link_lengths):
        g = nx.Graph()
        g.add_node(office_names[0])
        for i in range(1, len(office_names)):
            g.add_node(office_names[i])
            g.add_edge(office_names[i - 1], office_names[i])

        # Add the distances as an edge attribute for the nodes
        nx.set_edge_attributes(g, dict(zip(g.edges, link_lengths)), 'weight')

        return g


class CoordinatesMetroCoreGenerator(DefaultMetroCoreGenerator):

    def __init__(self, scale_factor=0.65):
        # In case of retrieving BB nodes, fit them in the inner X% of the new image
        self.scale_factor = scale_factor

    def generate_mesh(self, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                      national_nodes=[], add_prefix="", extra_node_info=None):

        # No coordinates available, go to the default generator
        if extra_node_info is None or nc.XLS_X_BACK not in extra_node_info or nc.XLS_Y_BACK not in extra_node_info:
            return super().generate_mesh(degrees, weights, upper_limits, types, dict_colors,
                                         algo, national_nodes, add_prefix, extra_node_info)

        # Increment in %distance in case that a BB node is to be inserted in an already used coordinate.
        increment = 0.01

        # If it does not receive the name of the national nodes, generate them as a sequence
        # based on the number of nodes defined for National COs.
        if len(national_nodes) == 0:
            num_nodes = types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'].values[0]
            national_nodes = [nc.NATIONAL_CO_CODE + str(i + 1) for i in range(num_nodes)]

        # If we do not have enough coordinates we call the basic algorithm
        # When we have just 1 national node, we can just call the basic algorithm.
        if len(national_nodes) != len(extra_node_info):
            print(national_nodes, extra_node_info, len(national_nodes))
            return super().generate_mesh(degrees, weights, upper_limits, types, dict_colors,
                                         algo, national_nodes, add_prefix, extra_node_info)

        # Nodes excluding NCOs to generate the topology without the BB nodes
        nodes_no_nco = sum([num for num, code in zip(types.number, types.code) if code != nc.NATIONAL_CO_CODE])

        # Additional prefix to name the generated nodes
        prefix = "_" + str((list(extra_node_info[nc.XLS_CLUSTER]))[0]) + "_"

        # Repeat the creation of the topology with all nodes except the NCOs until it is survivable
        while True:
            # Call the function to generate the topology
            topo = gen_topology(degrees, weights, nodes_no_nco)
            # Check for node survivability
            if not nx.is_connected(topo) or len(nx.minimum_node_cut(topo)) < 2:
                continue

            # Should be edge survivable, but just in case
            survival_edges = list(nx.k_edge_augmentation(topo, 2))
            topo.add_edges_from(survival_edges)

            # Calculate the actual degrees for each node
            dist_degrees = [val for (node, val) in topo.degree()]
            # Topology generation removes parallel links and self-loops, so
            # it is possible that the number of degrees of some nodes is
            # smaller than the smallest allowed degree.
            if min(dist_degrees) >= degrees[0]:
                # Finish only if the degrees of all nodes are >= minimum accepted degree
                break
            # Otherwise, repeat the node generation

        # Types and related proportions (escluding NCOs)
        types_no_nco = [code for code in types.code if code != nc.NATIONAL_CO_CODE]
        props_no_nco = [prop for code, prop in zip(types.code, types.proportion) if code != nc.NATIONAL_CO_CODE]
        # Assign types to nodes
        assigned_types = random.choices(types_no_nco, weights=props_no_nco, k=len(topo.nodes))

        # Modify the node labels to name them as a combination of the type, an index and the additional prefix
        name_nodes = rename_nodes(assigned_types, types, add_prefix=prefix)
        topo = nx.relabel_nodes(topo, dict(zip(topo.nodes, name_nodes)))

        # Generate positions of the nodes based on the defined algorithm
        pos = None
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

        # Retrieving minimum X, Y from the topology and X width and Y height
        topo_coords = list(pos.values())
        x_coord_topo = [x for x, y in topo_coords]
        y_coord_topo = [y for x, y in topo_coords]
        min_x_topo = min(x_coord_topo)
        min_y_topo = min(y_coord_topo)
        x_dist_topo = max(x_coord_topo) - min(x_coord_topo)
        y_dist_topo = max(y_coord_topo) - min(y_coord_topo)

        # Coordinates of the National nodes when they were generated in the backbone
        # Retrieving minimum X, Y from the topology and X width and Y height
        x_coord_bb = extra_node_info[nc.XLS_X_BACK]
        y_coord_bb = extra_node_info[nc.XLS_Y_BACK]

        min_x_bb = min(x_coord_bb)
        min_y_bb = min(y_coord_bb)
        x_dist_bb = max(x_coord_bb) - min_x_bb
        y_dist_bb = max(y_coord_bb) - min_y_bb

        # Center coordinates for the BB nodes
        # The BB nodes will be allocated starting from a position that is based on the (0,0) value of
        # the topology adding a % of the total x length of the topology. The center point of the BB node
        # coordinates will be moved to that (0, 0) position. As the BB nodes are to be placed
        # in the X% inner part (scale factor, e.g. 0.9--> 90%), there will be 5% below the top limit and
        # 5% above the lower limit where the BBs will not be placed.
        center_x_bb = min_x_bb if len(national_nodes) == 1 else x_dist_bb / 2 + min_x_bb
        center_y_bb = min_y_bb if len(national_nodes) == 1 else y_dist_bb / 2 + min_y_bb

        # Rescaling factor of relative positions from the backbone positions to the metro generated ones.
        factor_x = self.scale_factor * x_dist_topo / x_dist_bb if len(national_nodes) > 1 else 1
        factor_y = self.scale_factor * y_dist_topo / y_dist_bb if len(national_nodes) > 1 else 1

        # Prepare a tree with all the coordinates of the generated topology
        tree = spatial.KDTree(topo_coords)
        # Get the names of the nodes that are in the same order as topo_coords
        key_coord_nodes = list(pos.keys())
        # Connect the national nodes
        for node in national_nodes:
            # Move the axis of the bb coordinates to the center X and Y of the BB coordinates
            x_coord_node = extra_node_info.loc[extra_node_info[nc.XLS_NODE_NAME] == node, nc.XLS_X_BACK].iloc[0]
            y_coord_node = extra_node_info.loc[extra_node_info[nc.XLS_NODE_NAME] == node, nc.XLS_Y_BACK].iloc[0]
            # Distance to the central point of the BB coordinates
            x_node = x_coord_node - center_x_bb
            y_node = y_coord_node - center_y_bb

            # Relocate to the topo x and y axis (center_x_bb, center_y_bb) --> (0, 0) and rescale the distance
            x_node = x_node * factor_x
            y_node = y_node * factor_y

            # Check that the position is available
            # This should not be a problem at this point as we are not defining minimum distances
            xy_coord_topo = [(x, y) for x, y in topo_coords]
            # If it is taken, move slightly the position of the new node until we found an empty space.
            while (x_node, y_node) in xy_coord_topo:
                # Node already existed at this position
                print("Node existed at this position. Relocating BB node")
                x_node = x_node + x_dist_topo * increment
                y_node = y_node + y_dist_topo * increment

            # Create the nodes in the topology at those positions and assign the type
            topo.add_node(node)
            pos[node] = [x_node, y_node]
            assigned_types.append(nc.NATIONAL_CO_CODE)

            # Connect to other nodes
            # Select the connectivity degree
            connectivity = random.choices(degrees, weights=weights, k=1)[0]
            # Find as many nodes as required that are the nearest to this node
            distance, indexes = tree.query((x_node, y_node), k=connectivity)
            # Retrieve the name of each of those nodes and add an edge between them
            for con in range(connectivity):
                name_connected = key_coord_nodes[indexes[con]]
                topo.add_edge(node, name_connected)

        # Define the upper bound for the distances. Could be the maximum limit
        # or something below that (e.g. 1/2 of the last range)
        corrected_max_upper = upper_limits[-1]
        # Scale the distances from the topology to the defined values
        distances = calculate_edge_distances(topo, pos, corrected_max_upper)

        # Generate a sequence of colors for each node depending on the type
        colors = color_nodes(assigned_types, dict_colors)

        return topo, distances, assigned_types, pos, colors


class CoordinatesMetroCoreGenerator2(DefaultMetroCoreGenerator):

    def __init__(self, scale_factor=0.65):
        # In case of retrieving BB nodes, fit them in the inner X% of the new image
        self.scale_factor = scale_factor

    def generate_mesh(self, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                      national_nodes=[], add_prefix="", extra_node_info=None):

        # No coordinates available, go to the default generator
        if extra_node_info is None or nc.XLS_X_BACK not in extra_node_info or nc.XLS_Y_BACK not in extra_node_info:
            return super().generate_mesh(degrees, weights, upper_limits, types, dict_colors,
                                         algo, national_nodes, add_prefix, extra_node_info)
