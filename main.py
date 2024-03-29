from BackboneGenApp import BackboneGenApp
import pandas as pd
import networkconstants as nc


# Press the green button in the gutter to run the script.
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

    app = BackboneGenApp(degrees, weights, nodes, upper_limits, types, dict_colors)
