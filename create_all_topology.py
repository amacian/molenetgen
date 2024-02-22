import random

import pandas as pd
import networkconstants as nc
from BackboneGenerator import DualBackboneGenerator
import matplotlib as mpl
from matplotlib import pyplot as plt
import networkx as nx
import time

from ClusterGenerator import DistanceBasedClusterGenerator
from MetroAggGenerator import DefaultMetroAggGenerator
from MetroCoreGenerator import DefaultMetroCoreGenerator
from generator import write_network


def process_single_ring_aggregation(ring_topo, horseshoe_sizes, generator, lengths, l_perc, color_codes,
                                    file_path, file_prefix):
    # Get the nodes
    sources = [u for u, v in ring_topo.edges]
    destinations = [v for u, v in ring_topo.edges]
    for i in range(len(horseshoe_sizes)):
        source = sources[i]
        destination = destinations[i]
        if "AMP" in source or "AMP" in destination:
            print("Amplifier found in link:", source, "-", destination)
            continue
        prefix = nc.LOCAL_CO_CODE + "_" + source + "_" + destination + "_"
        (topo_hs, distances, assigned_types, pos, colors_local,
         national_ref_nodes) = \
            generator.metro_aggregation_horseshoe(source, 1, destination, horseshoe_sizes[i],
                                                  lengths,
                                                  l_perc,
                                                  prefix,
                                                  color_codes)
        figure = plt.Figure(dpi=50)
        ax = figure.add_subplot(111)
        # create a filename for the horseshoe
        filename_hs = file_prefix + "_" + source + "_" + destination + ".png"
        # Draw the figure

        nx.draw(topo_hs, pos=pos, with_labels=True, font_weight='bold',
                node_color=colors_local, ax=ax)
        write_network(file_path, topo_hs, distances, assigned_types, figure,
                      pos=pos, reference_nodes=national_ref_nodes,
                      type_nw=nc.METRO_AGGREGATION,
                      alternative_figure_name=filename_hs)


if __name__ == '__main__':

    directory = "/tmp/"
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
        nc.LOCAL_CO_CODE: 'b'
    }

    # Metro Core
    # The possible ring structures (number of rings)
    ring_nums = [1, 2, 3, 4, 6]
    # The frequency % of the node structures in the total topology
    ring_freqs = [8, 53, 25, 10, 4]

    # Metro aggregation
    # Statistics for horseshoe rings (ALLEGRO - TIM)
    aggr_ring_hops = [2, 3, 4, 5, 6, 7, 8]
    aggr_ring_freqs = [10, 19, 21, 27, 14, 5, 4]
    length_ranges = [0, 50, 100, 200, 302]
    length_percentages = [25, 40, 29, 6]

    # Create a backbone generator and generate the backbone network structure
    backbone_generator = DualBackboneGenerator()
    topo, distances, assigned_types, \
        node_sheet, link_sheet, \
        pos, colors = \
        backbone_generator.generate(degrees, weights, nodes, upper_limits, types, "kamada", dict_colors)

    # Cluster the nodes based on distance generating the dual connections for metro core
    cluster_generator = DistanceBasedClusterGenerator()
    cluster_dict, cluster_labels = cluster_generator.find_groups(topo, assigned_types, pos,
                                                                 eps=0.05, avoid_single=True)
    file_prefixes = directory + str(time.time()) + "_"

    # Prepare the image to be stored
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

    # Prepare the filename
    filename_xls = file_prefixes + "topology.xlsx"
    # Get x and y coordinates for all the elements
    # x_pos = [pos[0] for pos in list(pos.values())]
    # y_pos = [pos[1] for pos in list(pos.values())]
    # Write the Excel file for the backbone
    write_network(filename_xls, topo, distances, assigned_types, figure, nc.NODES_EXCEL_NAME,
                  nc.LINKS_EXCEL_NAME, clusters=cluster_labels, pos=pos, alternative_figure_name=filename)
    # Prepare the metro generator and the metro aggregation generator
    metro_gen = DefaultMetroCoreGenerator()
    metro_agg_gen = DefaultMetroAggGenerator()
    # Prepare to process each cluster
    cluster_keys = cluster_dict.keys()
    # Based on statistics on ring structures, define randomly which one will be applied to each cluster
    ring_sizes = random.choices(ring_nums, weights=ring_freqs, k=len(cluster_keys))
    i = 0
    for key in cluster_keys:
        # create a filename for the cluster
        filename = file_prefixes + "cluster_" + str(key) + ".png"
        # Get the nodes from the cluster
        cluster = cluster_dict.get(key)
        if len(cluster) == 0:
            continue
        # Get the ring structure size and generate the ring
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
        # Write the Excel file
        write_network(filename_xls, topo_mc, distances, assigned_types, figure, node_sheet=nc.NODES_EXCEL_NAME,
                      link_sheet=nc.LINKS_EXCEL_NAME, clusters=None, pos=pos, reference_nodes=national_ref_nodes,
                      type_nw=nc.METRO_CORE, alternative_figure_name=filename)
        horseshoe_sizes = random.choices(aggr_ring_hops, weights=aggr_ring_freqs, k=len(topo_mc.edges))
        process_single_ring_aggregation(topo_mc, horseshoe_sizes, metro_agg_gen, length_ranges,
                                        length_percentages, dict_colors, filename_xls, file_prefixes)
        i += 1
