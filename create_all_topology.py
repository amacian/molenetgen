import random

import pandas as pd
import networkconstants as nc
from BackboneGenerator import DualBackboneGenerator
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
import time

from ClusterGenerator import DistanceBasedClusterGenerator
from MetroCoreGenerator import DefaultMetroCoreGenerator
from generator import write_network

if __name__ == '__main__':
    # ** BACKBONE  ** #
    # The possible node degrees
    degrees = [2, 3, 4, 5]
    # The frequency % of the node degrees in the total nodes
    weights = [23, 41, 27, 9]
    # Total number of nodes
    nodes = 48
    # Length limits
    upper_limits = [50, 100, 200, 400, 600]

    # Types and percentages for the nodes
    types = pd.DataFrame({'code': [nc.NATIONAL_CO_CODE, nc.REGIONAL_CO_CODE, nc.TRANSIT_CODE],
                          'proportion': [0.826, 0.130, 0.044]})

    # Create and store the backbone network
    dict_colors = {
        nc.DATA_CENTER_CODE: 'm',
        nc.NATIONAL_CO_CODE: 'g',
        nc.REGIONAL_CO_CODE: 'r',
        nc.REGIONAL_NONHUB_CO_CODE: 'y',
        nc.LOCAL_CO_CODE: 'o'
    }

    # The possible ring structures (number of rings)
    ring_nums = [1, 2, 3, 4, 6]
    # The frequency % of the node structures in the total topology
    ring_freqs = [8, 53, 25, 10, 4]

    backbone_generator = DualBackboneGenerator()
    topo, distances, assigned_types, \
        node_sheet, link_sheet, \
        pos, colors = \
        backbone_generator.generate(degrees, weights, nodes, upper_limits, types, "kamada", dict_colors)
    print(topo.nodes)
    cluster_generator = DistanceBasedClusterGenerator()
    cluster_dict, cluster_labels = cluster_generator.find_groups(topo, assigned_types, pos,
                                                                 eps=0.05, avoid_single=True)
    file_prefixes = "/Users/macian/Documents/test/" + str(time.time()) + "_"
    # size_ratio = x_size / self.fig_width
    # Change the figure width based on this and prepare the canvas and widgets
    plt.rcParams["figure.figsize"] = [3 * val for val in plt.rcParamsDefault["figure.figsize"]]
    figure = plt.Figure(tight_layout=True, dpi=50)
    ax = figure.add_subplot(111)
    color_map = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=1, vmax=max(cluster_labels))
    # Draw the result
    nx.draw(topo, pos=pos, with_labels=True, font_weight='bold',
            node_color=color_map(norm(cluster_labels)), ax=ax)
    # Save the figure
    filename = file_prefixes + "clustered_backbone.png"

    filename_xls = file_prefixes + "topology.xlsx"
    # Get x and y coordinates for all the elements
    x_pos = [pos[0] for pos in list(pos.values())]
    y_pos = [pos[1] for pos in list(pos.values())]
    write_network(filename_xls, topo, distances, assigned_types, figure, nc.NODES_EXCEL_NAME,
                  nc.LINKS_EXCEL_NAME, clusters=cluster_labels, pos=pos, alternative_figure_name=filename)

    metro_gen = DefaultMetroCoreGenerator()
    cluster_keys = cluster_dict.keys()
    ring_sizes = random.choices(ring_nums, weights=ring_freqs, k=len(cluster_keys))
    i = 0
    for key in cluster_keys:
        filename = file_prefixes + "cluster_" + str(key) + ".png"
        cluster = cluster_dict.get(key)
        if len(cluster) == 0:
            continue
        ring_size = ring_sizes[i]
        (topo_mc, distances, assigned_types, pos, colors,
         national_ref_nodes) = \
            metro_gen.generate_ring(ring_size,
                                    cluster[0],
                                    cluster[1],
                                    prefix="R" + str(key) + "-",
                                    dict_colors=dict_colors)

        # Draw the figure
        figure = plt.Figure(tight_layout=True, dpi=50)
        ax = figure.add_subplot(111)
        nx.draw(topo_mc, pos=pos, with_labels=True, font_weight='bold',
                node_color=colors, ax=ax)
        write_network(filename_xls, topo_mc, distances, assigned_types, figure, node_sheet=nc.NODES_EXCEL_NAME,
                      link_sheet=nc.LINKS_EXCEL_NAME, clusters=None, pos=pos, reference_nodes=national_ref_nodes,
                      type_nw=nc.METRO_CORE, alternative_figure_name=filename)
        i += 1
