from abc import ABC, abstractmethod
import networkconstants as nc
import random
import networkx as nx
from network import gen_topology, calculate_edge_distances, rename_nodes, color_nodes


class BackboneGenerator(ABC):

    # Filename for the Backbone network
    # degrees: A list with the possible node degrees
    # weights: The frequency % of the node degrees in the total number of COs
    # nodes: Total number of COs
    # upper_limits: upper length limits for each range
    # types: available types of nodes and % per type
    @abstractmethod
    def generate(self, degrees, weights, nodes, upper_limits, types, algo="spectral", dict_colors={}):
        pass


class DefaultBackboneGenerator(BackboneGenerator):
    def generate(self, degrees, weights, nodes, upper_limits, types, algo="spectral", dict_colors={}):
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

        # Assign types to nodes
        assigned_types = random.choices(types.code, weights=types.proportion, k=len(topo.nodes))

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


class DualBackboneGenerator(BackboneGenerator):
    def generate(self, degrees, weights, nodes, upper_limits, types, algo="spectral", dict_colors={}):
        # Sheet names in the Excel file for nodes and links
        node_sheet = nc.NODES_EXCEL_NAME
        link_sheet = nc.LINKS_EXCEL_NAME

        # Repeat
        while True:
            # Call the function to generate the topology using half of the nodes with double degree number
            topo = gen_topology([(i * 2)-2 for i in degrees], weights, int(nodes / 2))
            # Calculate the actual degrees for each node
            dist_degrees = [val for (node, val) in topo.degree()]
            # Topology generation removes parallel links and self-loops, so
            # it is possible that the number of degrees of some nodes is
            # smaller than the smallest allowed degree.
            if min(dist_degrees) >= degrees[0]:
                # Finish only if the degrees of all nodes are >= minimum accepted degree
                break
            # Otherwise, repeat the node generation

        # Assign types to nodes
        assigned_types = random.choices(types.code, weights=types.proportion, k=len(topo.nodes))

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

        # Calculating the size of the graph
        positions = list(pos.values())
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions]
        x_size = max(x_pos) - min(x_pos)
        y_size = max(y_pos) - min(y_pos)

        for idx in range(len(topo.nodes)):
            node = list(topo.nodes)[idx]
            self.generate_duplicated_node(node,
                                          pos,
                                          assigned_types,
                                          x_size,
                                          y_size,
                                          topo,
                                          idx)

            # break  # TODO Remove after test

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

    def generate_duplicated_node(self, node, positions, types, x_size, y_size, topo, idx):
        # Get the position and type of the node
        pos = positions[node]
        type_sel = types[idx]
        # do not duplicate transit nodes
        if type_sel == nc.TRANSIT_CODE:
            return

        # Place the new node near the existing one
        x_dis = random.uniform(x_size / 50, x_size / 20)
        y_dis = random.uniform(x_size / 50, y_size / 20)
        signs = random.choices([-1, 1], k=2)
        new_pos = [0, 0]
        new_pos[0] = pos[0] + (signs[0] * x_dis)
        new_pos[1] = pos[1] + (signs[1] * y_dis)

        # Name the new TWin node as the old one followed bu _TW
        new_node = node + "_TW"
        # Add the node to the graph
        topo.add_node(new_node)
        # Copy the type of the existing node
        types.append(type_sel)
        # Set its position
        positions[new_node] = new_pos
        # Split the edges between both nodes
        edges = list(topo.edges(node))
        indexes = random.sample(range(len(edges)), k=int(len(edges)/2))
        for i in indexes:
            edge = edges[i]
            topo.add_edge(new_node, edge[1])
            topo.remove_edge(node, edge[1])

        # Add link between twin nodes
        topo.add_edge(node, new_node)

        # topo.add_edge(new_node, XXXXX)
        return
