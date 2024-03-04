
import random

import pandas
import networkconstants as nc

from BackboneGenerator import WaxmanPavenGenerator, DefaultBackboneGenerator, DualBackboneGenerator
from network import check_metrics, count_distance_ranges

import numpy as np


def average_distance_ranges(upper_distances, weights, distances):
    avg_dist = np.average(distances)
    proportions = [w/sum(weights) for w in weights]
    ranges = upper_distances.copy()
    ranges.insert(0, 0)
    avg_per_range = [ranges[i]+(ranges[i+1]-ranges[i])/2 for i in range(len(ranges)-1)]
    total_avg = sum([w * avg_range for w, avg_range in zip(weights, avg_per_range)])
    factor = total_avg / avg_dist
    new_distances = [distance * factor for distance in distances]
    return new_distances

def optimize_distance_ranges(upper_distances, weights, distances):
    second_range = upper_distances[-2]
    upper_limit = second_range + 1
    factor = upper_limit / max(distances)
    new_distances = [distance * factor for distance in distances]
    actual_distance_weight = [dist / 100 for dist in count_distance_ranges(new_distances, upper_limits)]
    mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                   actual_distance_weight, perc=True)
    ref_mape = mape_distance
    sorted_distances = list(set(distances.copy()))
    sorted_distances.sort()
    sorted_distances.reverse()
    # Try changing the points assigned to the lower threshold
    for i in sorted_distances:
        factor = upper_limit / i
        alt_distances = [distance * factor for distance in distances]
        new_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_limits)]
        mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                       new_distance_weight, perc=True)
        if mape_distance > ref_mape:
            break
        new_distances = alt_distances
        ref_mape = mape_distance
    # Calculate the metrics for the distances.

    return new_distances


def optimize_distance_ranges2(upper_distances, weights, distances):
    upper_limit = max(upper_distances)
    new_distances = distances.copy()
    step = upper_limit / 100
    ref_mape = 1000
    while upper_limit > 0:
        factor = upper_limit / max(distances)
        alt_distances = [distance * factor for distance in distances]
        actual_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_limits)]
        mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                       actual_distance_weight, perc=True)
        if mape_distance < ref_mape:
            new_distances = alt_distances
            ref_mape = mape_distance
        upper_limit = upper_limit - step


    # Calculate the metrics for the distances.

    alt_distances = average_distance_ranges(upper_distances, weights, distances)
    new_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_limits)]
    mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                   new_distance_weight, perc=True)
    if mape_distance < ref_mape:
        new_distances = alt_distances

    alt_distances = optimize_distance_ranges(upper_distances, weights, distances)
    new_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_limits)]
    mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                   new_distance_weight, perc=True)
    if mape_distance < ref_mape:
        new_distances = alt_distances

    alt_distances = optimize_distance_ranges(upper_distances, weights, distances)
    new_distance_weight = [dist / 100 for dist in count_distance_ranges(alt_distances, upper_limits)]
    mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                   new_distance_weight, perc=True)
    if mape_distance < ref_mape:
        new_distances = alt_distances
    return new_distances

# Assign the maximum distance a value equals to the smallest value in the highest range.
def reduce_distance_ranges(upper_distances, distances):
    second_range = upper_distances[-2]
    upper_limit = second_range + 1
    factor = upper_limit / max(distances)
    new_distances = [distance * factor for distance in distances]
    return new_distances


if __name__ == '__main__':
    random.seed(12345)
    # The possible node degrees
    degrees = [2, 3, 4, 5]
    # The frequency % of the node degrees in the total nodes
    weights = [23, 41, 27, 9]
    '''graph, pos, res = gen_waxman_paven_topology(60, 20, dist_factor=0.6,
                                                beta=0.4, alpha=0.3)'''
    # Total number of nodes
    nodes = 48
    # Length limits
    upper_limits = [50, 100, 200, 400, 600]
    distance_weights = [0.155, 0.169, 0.338, 0.254, 0.085]

    # Types and percentages for the nodes
    types = pandas.DataFrame({'code': [nc.NATIONAL_CO_CODE, nc.REGIONAL_CO_CODE, nc.TRANSIT_CODE],
                              'proportion': [0.826, 0.130, 0.044]})

    # Create and store the backbone network
    dict_colors = {
        nc.DATA_CENTER_CODE: 'm',
        nc.NATIONAL_CO_CODE: 'g',
        nc.REGIONAL_CO_CODE: 'r',
        nc.REGIONAL_NONHUB_CO_CODE: 'y',
        nc.LOCAL_CO_CODE: 'o'
    }

    generators = [DefaultBackboneGenerator(), DualBackboneGenerator(), WaxmanPavenGenerator(regions=12)]
    name_generators = ["default", "dual", "pavan"]
    algorithms = ["spectral", "spring", "kamada"]

    mape_results_degree = {}
    mape_results_type = {}
    mape_results_distance = {}

    degree_results = {}
    type_results = {}
    distance_results = {}

    best_map_degree_distribution = {}
    best_type_distribution = {}
    best_distance_distribution = {}
    best_mape_distance_distribution = {}

    iterations_per_experiment = 1000

    # Calculate the average degree to see if the final result is close to this value
    degree_pd = pandas.DataFrame({'degrees': degrees, 'weights': weights})
    avg_degree = np.average(degree_pd.degrees, weights=degree_pd.weights)
    normalized_weights = [w / sum(weights) for w in weights]
    dict_weights = {degree: weight for degree, weight in zip(degrees, normalized_weights)}

    for i in range(len(generators)):
        # Select one of the generators
        gen = generators[i]
        for j in range(len(algorithms)):
            algorithm = algorithms[j]
            name_experiment = name_generators[i] + "_" + algorithm
            print("Experiment: ", name_experiment)
            mape_results_degree[name_experiment] = []
            degree_results[name_experiment] = []

            mape_results_type[name_experiment] = []
            type_results[name_experiment] = []

            distance_results[name_experiment] = []
            mape_results_distance[name_experiment] = []

            best_map_degree_distribution[name_experiment] = None
            best_type_distribution[name_experiment] = None
            best_distance_distribution[name_experiment] = None

            best_mape_degree = 1
            best_mape_type = 1
            best_rsme_distance = 100000
            best_mape_distance = 100000

            for k in range(iterations_per_experiment):
                mape_degree = 0
                mape_type = 0
                mape_distance = 0
                rsme_degree = 0
                rsme_type = 0
                rsme_distance = 0

                topo, distances, assigned_types, node_sheet, link_sheet, pos, colors = (
                    gen.generate(degrees, weights, nodes, upper_limits, types, algo=algorithm, dict_colors=dict_colors))

                # Calculate the actual degrees for each node
                degree_sequence = [val for (node, val) in topo.degree()]
                # Calculate the metrics for the degrees
                mae, mape_degree, rsme_degree, actual_dist = check_metrics(degrees, normalized_weights,
                                                                           [val for (node, val) in topo.degree()])
                # Accumulate the results from the experiments
                mape_results_degree[name_experiment].append(mape_degree)
                degree_results[name_experiment].append(actual_dist)

                if mape_degree < best_mape_degree:
                    best_map_degree_distribution[name_experiment] = actual_dist
                    best_mape_degree = mape_degree

                # Calculate the metrics for the types.
                mae, mape_type, rsme_type, actual_dist = check_metrics(types.code, types.proportion,
                                                                       assigned_types)
                mape_results_type[name_experiment].append(mape_type)
                type_results[name_experiment].append(actual_dist)

                if mape_type < best_mape_type:
                    best_type_distribution[name_experiment] = actual_dist
                    best_mape_type = mape_type

                actual_distance_weight = [dist / 100 for dist in count_distance_ranges(distances, upper_limits)]
                # Calculate the metrics for the distances.
                mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                               actual_distance_weight, perc=True)

                # new_distances = reduce_distance_ranges(upper_limits, distances)
                new_distances = optimize_distance_ranges2(upper_limits, distance_weights, distances)
                mae, mape_distance, rsme_distance, actual_dist = check_metrics(upper_limits, distance_weights,
                                                                               actual_distance_weight, perc=True)

                if mape_distance < best_mape_distance:
                    best_mape_distance = mape_distance
                    best_mape_distance_distribution[name_experiment] = actual_dist

                mape_results_distance[name_experiment].append(mape_distance)
                distance_results[name_experiment].append(actual_distance_weight)

    for i in range(len(generators)):
        for j in range(len(algorithms)):
            algorithm = algorithms[j]
            name_experiment = name_generators[i] + "_" + algorithm
            print("Experiment: ", name_experiment)
            print("Degree")
            print(min(mape_results_degree[name_experiment]),
                  np.average(mape_results_degree[name_experiment]),
                  max(mape_results_degree[name_experiment]))
            print("Original: ", dict_weights)
            print("Best: ", best_map_degree_distribution[name_experiment])

            '''print("Mape type")
            print(min(mape_results_type[name_experiment]),
                  np.average(mape_results_type[name_experiment]),
                  max(mape_results_type[name_experiment]))
            print("Original type distribution: ", types)
            # print("Best mape result: ", best_type_distribution[name_experiment])'''

            print("Distance")
            print(min(mape_results_distance[name_experiment]),
                  np.average(mape_results_distance[name_experiment]),
                  max(mape_results_distance[name_experiment]))
            print("Original: ", distance_weights)
            print("Best: ", best_mape_distance_distribution[name_experiment])
            print("---------------------------------------------")