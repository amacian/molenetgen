import numpy as np
from scipy.io import savemat
import pandas as pd
import networkconstants as nc
import os


def create_adjacency_matrix(file_path, file_name, new_file="Topology.mat", topo_name="Topology"):
    try:
        file = os.path.join(file_path, file_name)
        df = pd.read_excel(file, sheet_name=nc.LINKS_EXCEL_NAME)
        nodes = pd.read_excel(file, sheet_name=nc.NODES_EXCEL_NAME)

        N = len(nodes[nc.XLS_NODE_NAME])

        # Initialize adjacency matrix (N x N)
        adj_matrix = np.zeros((N, N))

        # Populate the adjacency matrix
        for _, row in df.iterrows():
            src = np.where(nodes[nc.XLS_NODE_NAME] == row[nc.XLS_SOURCE_ID])[0][0]
            dst = np.where(nodes[nc.XLS_NODE_NAME] == row[nc.XLS_DEST_ID])[0][0]
            adj_matrix[src, dst] = row[nc.XLS_DISTANCE]
            adj_matrix[dst, src] = row[nc.XLS_DISTANCE]  # Assuming an undirected graph

        # Save the matrix as a .mat file
        print(adj_matrix)
        file = os.path.join(file_path, new_file)
        savemat(file, {topo_name: adj_matrix})

    except Exception as e:
        print(f"Error processing {file}: {e}")


if __name__ == '__main__':
    directory_path = "path_to_file/"
    file_name = "topology_file.xlsx"
    create_adjacency_matrix(directory_path, file_name)
