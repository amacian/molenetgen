from network import write_network_xls
import networkconstants as nc


def write_network(filename, topo, distances, assigned_types, figure, node_sheet=nc.NODES_EXCEL_NAME,
                  link_sheet=nc.LINKS_EXCEL_NAME, clusters=None, pos=None, reference_nodes=None,
                  type_nw=nc.BACKBONE, alternative_figure_name=None):
    if alternative_figure_name is None:
        alternative_figure_name = filename + ".png"
    figure.savefig(alternative_figure_name)
    x_coord = [0] * len(topo.nodes)
    y_coord = [0] * len(topo.nodes)
    if pos is not None:
        x_coord, y_coord = zip(*pos.values())
    return write_network_xls(filename, topo, distances, assigned_types, node_sheet, link_sheet, clusters, x_coord,
                             y_coord, reference_nodes, coord_type=type_nw)


def format_node_list(nodes):
    result = ""
    process = 0
    for i in nodes:
        if process != 0:
            result = result + ", "
        process = process + 1
        result = result + i
        if process == 9:
            result = result + "\n"
            process = 0
    return result


def print_ring_to_screen(link_lengths, office_names):
    for i in range(len(link_lengths)):
        print("[" + office_names[i] + "]--- ", end="")
        print(str(link_lengths[i]) + " ---", end="")
    print("[" + office_names[-1] + "]")
    return
