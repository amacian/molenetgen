from BackboneGenApp import BackboneGenApp
import pandas as pd
import networkconstants as nc
from MetroGenApp import MetroGenApp
from generator import metro_core_split, metro_aggregation_horseshoe, ring_structure_tel


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ** BACKBONE  ** #
    # Filename for the Backbone network
    filename = "BackboneSynthetic.xlsx"
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
    # app = BackboneGenApp(degrees, weights, nodes, upper_limits, types, dict_colors)
    #backbone(filename, degrees, weights, nodes, upper_limits, types)


    # Length limits
    nodes = 95
    upper_limits = [10, 40, 80, 120]
    initial_refs = ['NCO1', 'NCO2', 'NCO3', 'NCO4', 'NCO5']
    types = pd.DataFrame({'code': [nc.DATA_CENTER_CODE, nc.NATIONAL_CO_CODE, nc.REGIONAL_CO_CODE,
                                   nc.REGIONAL_NONHUB_CO_CODE],
                          'proportion': [1, 5, 66, 23]})

    app = MetroGenApp(degrees, weights, nodes, upper_limits, types, dict_colors, initial_refs)

    ''' #  ** METRO CORE  ** #
    # Filename for the Metro network
    filename = "MetroSynthetic.xlsx"
    # The possible node degrees
    degrees = [2, 3, 4, 5, 6]
    # The frequency % of the node degrees in the total nodes
    weights = [23, 41, 27, 8, 1]
    # Length limits
    upper_limits = [10, 40, 80, 120]
    # Sheet names in the Excel file for nodes and links
    # Types and percentages
    types = pd.DataFrame({'code': ['DC', 'NCO', 'RCO', 'RCOnh'],
                          'number': [1, 5, 66, 23]})
    # Assign color to each type of node
    dict_colors = {
        'DC': 'm',
        #'NCO': 'g',
        'RCO': 'r',
        'RCOnh': 'y'
    }

    # Create and store the metro network
    metro_core_split(filename, degrees, weights, upper_limits, types, dict_colors)

    # ** METRO AGGREGATION ** #
    # Filename for the Metro Aggregation Horseshoe network
    filename = "HorseShoeSynthetic.xlsx"
    # first and second end of the Horseshoe
    end_1 = "RCO6"
    end_2 = "RCO2"
    # Initial index for the first LCO name
    initial_lco = 7
    # Number of hops in the horseshoe
    hops = 5
    # Distances and percentage between each pair of elements
    length_ranges = [0, 50, 100, 200, 302]
    length_percentages = [25, 40, 29, 6]
    # Prefix for the Local COs
    prefix_lco = "LCO"
    # Generate the horseshoe with the configured parameters

    metro_aggregation_horseshoe(filename, end_1, initial_lco, end_2, hops,
                                length_ranges,
                                length_percentages,
                                prefix_lco)

    ring_structure_tel(6, "NCO1", "NCO2", "1R")'''
