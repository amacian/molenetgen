import pandas as pd
import networkconstants as nc
from MetroGenApp import MetroGenApp

if __name__ == '__main__':

    # The possible node degrees
    degrees = [2, 3, 4, 5]
    # The frequency % of the node degrees in the total nodes
    weights = [23, 41, 27, 9]

    # Create and store the backbone network
    dict_colors = {
        nc.DATA_CENTER_CODE: 'm',
        nc.NATIONAL_CO_CODE: 'g',
        nc.REGIONAL_CO_CODE: 'r',
        nc.REGIONAL_NONHUB_CO_CODE: 'y',
        nc.LOCAL_CO_CODE: 'o'
    }

    # Length limits
    nodes = 95
    upper_limits = [10, 40, 80, 120]
    distance_weights = [39, 37, 21, 3]
    initial_refs = ['NCO1', 'NCO2', 'NCO3', 'NCO4', 'NCO5']
    types = pd.DataFrame({'code': [nc.DATA_CENTER_CODE, nc.NATIONAL_CO_CODE, nc.REGIONAL_CO_CODE,
                                   nc.REGIONAL_NONHUB_CO_CODE],
                          'proportion': [1, 5, 66, 23]})

    iterations = 20

    app = MetroGenApp(degrees, weights, nodes, upper_limits, distance_weights, types,
                      dict_colors, initial_refs, iterations_distance=iterations)
