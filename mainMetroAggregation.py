import networkconstants as nc
from MetroAggGenApp import MetroAggGenApp

if __name__ == '__main__':

    # Create and store the backbone network
    dict_colors = {
        nc.DATA_CENTER_CODE: 'm',
        nc.NATIONAL_CO_CODE: 'g',
        nc.REGIONAL_CO_CODE: 'r',
        nc.REGIONAL_NONHUB_CO_CODE: 'y',
        nc.LOCAL_CO_CODE: 'b'
    }
    length_ranges = [0, 50, 100, 200, 302]
    length_percentages = [25, 40, 29, 6]

    app = MetroAggGenApp(length_ranges,  length_percentages, dict_colors)
