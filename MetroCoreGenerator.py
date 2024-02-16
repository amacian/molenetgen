from abc import ABC, abstractmethod
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
    # add_prefix: additional prefix to incorporate to each non national node
    @abstractmethod
    def generate_mesh(self, degrees, weights, upper_limits, types, dict_colors, algo="spectral",
                      national_nodes=[], add_prefix=""):
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
                      national_nodes=[], add_prefix=""):

        node_sheet = "Nodes"
        link_sheet = "Links"

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
            link_lengths = [random.uniform(range_ring[0], range_ring[1]) for i in range(offices)]
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

    # Correct negative length of the final link by reducing it propotionally
    # from the rest of the links
    def correct_negative_pending(self, link_lengths, pending_length, offices):
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