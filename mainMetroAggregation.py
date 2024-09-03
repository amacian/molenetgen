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

    hop_ranges = [2, 3, 4, 5, 6, 7, 8]
    hop_percentages = [10, 19, 21, 27, 14, 5, 4]

    app = MetroAggGenApp(length_ranges, length_percentages, hop_number=hop_ranges, hop_percents=hop_percentages,
                         dict_colors=dict_colors)
