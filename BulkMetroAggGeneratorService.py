from abc import ABC, abstractmethod
import random
import pandas as pd

import networkx as nx
from matplotlib import pyplot as plt
from scipy import spatial

from MetroAggGenerator import MetroAggGenerator
import networkconstants as nc
import Texts_EN as texts
from generator import write_network


class BulkMetroAggGeneratorService(ABC):
    @abstractmethod
    def bulk_metro_aggregation(self, nodes_list, links_list, lengths, l_perc, hops, h_perc,
                               linked_only, n_horseshoes, colors, file_path,
                               prefix):
        pass


class DefaultBulkMetroAggGenService(BulkMetroAggGeneratorService):
    def __init__(self, metro_agg_gen):
        assert isinstance(metro_agg_gen, MetroAggGenerator)
        self.generator = metro_agg_gen

    def bulk_metro_aggregation(self, nodes, links_list, lengths, l_perc, hops, h_perc,
                               linked_only, n_horseshoes, d_colors, file_path,
                               prefix=nc.LOCAL_CO_CODE):

        # Select the number of hops for all the horseshoes to be generated
        horseshoe_hops = random.choices(hops, weights=h_perc, k=n_horseshoes)
        # Select, randomly, the initial end index of the horseshoes
        names = list(nodes[nc.XLS_NODE_NAME])
        initial_end_index = random.choices(range(len(names)), k=n_horseshoes)
        # Get tuples for the coordinates of the nodes
        x = list(nodes[nc.XLS_X_MCORE])
        y = list(nodes[nc.XLS_Y_MCORE])
        all_coords = [(i, j) for i, j in zip(x, y)]
        # Prepare an array to store the end pairs for the horseshoes
        ends = []
        for i in initial_end_index:
            # Get the coordinate of the initial end node
            coord_single = all_coords[i]
            # Get the coordinates of the rest of the nodes
            rest_coords = [c for pos, c in enumerate(all_coords) if pos != i]
            # If we have to consider only the list of connected nodes
            if linked_only:
                # Get the list of edges read from the file
                links = links_list
                # Select only those where the selected node is involved and pick the other end
                filtered1 = links[links[nc.XLS_SOURCE_ID] == names[i]]
                filtered2 = links[links[nc.XLS_DEST_ID] == names[i]]
                filtered = pd.concat([filtered1, filtered2], ignore_index=True)
                # Should be at least one link
                if len(filtered) == 0:
                    return False, texts.EMPTY_LIST
                # Sort by link size
                filtered.sort_values(by=[nc.XLS_DISTANCE], inplace=True)
                # Get the name of the other end of the link with shortest distance
                end2 = filtered[nc.XLS_SOURCE_ID].iloc[0] \
                    if filtered[nc.XLS_SOURCE_ID].iloc[0] != names[i] \
                    else filtered[nc.XLS_DEST_ID].iloc[0]
            # If we have to consider all nodes independently of possible links
            else:
                # Use a KDTree to find the nearest one
                tree = spatial.KDTree(rest_coords)
                res = tree.query([coord_single])
                # Get the index of that node
                idx = int(res[1]) if int(res[1]) < i else int(res[1]) + 1
                # And retrieve the name
                end2 = names[idx]
            # Add the pair of horseshoe ends to the list
            ends.append((names[i], end2))

        # Let's create the horseshoes
        for i in range(n_horseshoes):
            # Get the next pair of ends to process
            (end1, end2) = ends[i]
            # Generate a name for the local nodes based on the ends
            h_prefix = prefix + str(i) + "_" + end1 + "_" + end2 + "_"
            # Run the horseshoe creation
            (topo, distances, assigned_types, pos, colors,
             national_ref_nodes) = self.generator.metro_aggregation_horseshoe(end1, 1, end2,
                                                                              horseshoe_hops[i], lengths, l_perc,
                                                                              h_prefix, d_colors)
            # Prepare the figure. At this moment we are only keeping the last one
            # But we may change the code to keep all of them in the future by modifying the file name
            figure = plt.Figure(tight_layout=True, dpi=50)
            ax = figure.add_subplot(111)
            nx.draw(topo, pos=pos, with_labels=True, font_weight='bold',
                    node_color=colors, ax=ax)

            if not file_path:
                return False, texts.FILE_NOT_FOUND

            # Write the excel file
            (result, message) = write_network(file_path, topo, distances,
                                            assigned_types, figure, pos=pos,
                                            reference_nodes=national_ref_nodes,
                                            type_nw=nc.METRO_AGGREGATION)
            if not result:
                return False, message

        return True, ""