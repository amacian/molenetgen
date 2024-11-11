from BackboneGenApp import BackboneGenApp
import pandas as pd
import networkconstants as nc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ** BACKBONE  ** #
    # The possible node degrees
    degrees = [2, 3, 4, 5]
    # The frequency % of the node degrees in the total nodes
    weights = [10, 18, 12, 4]
    # Total number of nodes
    nodes = 52
    # Length limits
    upper_limits = [50, 100, 200, 400, 600]
    # the frequency of each range of limits
    propor_limits = [15.5, 16.9, 33.8, 25.4, 8.5]

    # Types and percentages for the nodes
    types = pd.DataFrame({'code': [nc.NATIONAL_CO_CODE, nc.REGIONAL_CO_CODE, nc.TRANSIT_CODE],
                          'proportion': [0.827, 0, 0.173]})

    # Create and store the backbone network
    dict_colors = {
        nc.DATA_CENTER_CODE: 'm',
        nc.NATIONAL_CO_CODE: 'g',
        nc.REGIONAL_CO_CODE: 'r',
        nc.REGIONAL_NONHUB_CO_CODE: 'y',
        nc.LOCAL_CO_CODE: 'o'
    }

    # Topologies to create in order to get the best one and approximate to the requested distances
    iters = 1
    bounds = 0.05

    app = BackboneGenApp(degrees, weights, nodes, upper_limits, propor_limits, types, dict_colors,
                         iterations_distance=iters, bounds=bounds)
