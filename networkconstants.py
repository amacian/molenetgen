NATIONAL_CO_CODE = 'NCO'
REGIONAL_CO_CODE = 'RCO'
LOCAL_CO_CODE = 'LCO'
DATA_CENTER_CODE = 'DC'
REGIONAL_NONHUB_CO_CODE = 'RCOnh'
TRANSIT_CODE = 'Transit'
TYPES_EXCLUDED = [TRANSIT_CODE]
AGG_TYPES_EXCLUDED = [REGIONAL_NONHUB_CO_CODE, DATA_CENTER_CODE, LOCAL_CO_CODE, TRANSIT_CODE]
BETWEENNESS_PRIORITY = ['NCO', 'RCO', 'RCOnh', 'Transit', 'LCO', 'DC']
DEGREE_PRIORITY = ['NCO', 'RCO', 'RCOnh', 'Transit', 'LCO', 'DC']

ASSIGN_BY_RANDOM = "random"
ASSIGN_BY_BETWEEN = "betweenness"
ASSIGN_BY_DEGREE = "degree"

KAMADA_ALGO = "kamada"
SPRING_ALGO = "spring"
SPIRAL_ALGO = "spiral"
SHELL_ALGO = "shell"
SPECTRAL_ALGO = "spectral"
RANDOM_ALGO = "random"
MAIN_ALGORITHMS = [SPECTRAL_ALGO, KAMADA_ALGO, SPRING_ALGO]

DEFAULT_GEN = "Default"
DUAL_GEN = "Dual"
REGION_GEN = "Region"
ALL_GEN = "Check ALL"

NODES_EXCEL_NAME = "Nodes"
LINKS_EXCEL_NAME = "Links"

BACKBONE = "backbone"
METRO_CORE = "m_core"
METRO_AGGREGATION = "m_aggreg"
ACCESS = "access"

XLS_NODE_NAME ='node_name'
XLS_CO_TYPE='Central_office_type'
XLS_REF_RCO='Reference_Regional_CO'
XLS_REF_NCO='Reference_National_CO'
XLS_HOUSE_H='Households'
XLS_MACRO_C='Macro_cells_sites'
XLS_SMALL_C='Small_cells_sites'
XLS_TWIN_RCO='Twin_Regional_CO'
XLS_TWIN_NCO='Twin_National_CO'
XLS_CLUSTER='Macro_region'
XLS_X_BACK='x_coord_backbone'
XLS_Y_BACK='y_coord_backbone'
XLS_X_MCORE='x_coord_m_core'
XLS_Y_MCORE='y_coord_m_core'

XLS_SOURCE_ID='sourceID'
XLS_DEST_ID='destinationID'
XLS_DISTANCE='distanceKm'
XLS_CAPACITY_GBPS='capacityGbps'

ITERATIONS_FOR_DISTANCE = 10
