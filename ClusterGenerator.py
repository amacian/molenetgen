from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import DBSCAN
import networkconstants as nc
from scipy import spatial


class ClusterGenerator(ABC):

    # Group nodes creating clusters
    # topo - the networkx topology
    # types - list of assigned types for each node
    # pos - the array of positions created by the corresponding algorithm
    # eps - distance used as reference for grouping
    # avoid_single - decide if avoid clusters with just 1 node by reassigning them ignoring eps
    @abstractmethod
    def find_groups(self, topo, types, pos, eps=0.1, avoid_single=True):
        pass

    # Merge individual clusters by joining them into the closest clusters
    # nodes - the array of existing nodes
    # pos - the array of positions created by the corresponding algorithm
    # cluster_dict - the dictionary with key=cluster index and values the set of nodes of the cluster
    # cluster_labels - a list with cluster values for each node
    # types - list of types for each node
    # excluded_indices - elements from the nodes list that are to be excluded
    @abstractmethod
    def merge_individual_clusters(self, nodes, pos, cluster_dict, cluster_labels, types, excluded_indices):
        pass


class DistanceBasedClusterGenerator(ClusterGenerator):

    # Group nodes creating clusters
    # topo - the networkx topology
    # types - list of assigned types for each node
    # pos - the array of positions created by the corresponding algorithm
    # eps - distance used as reference for grouping
    # avoid_single - decide if avoid clusters with just 1 node by reassigning them ignoring eps
    def find_groups(self, topo, types, pos, eps=0.1, avoid_single=True):
        # Create a list with all the nodes
        nodes = [u for u in topo.nodes]
        # Create a sublist with just the nodes of types included
        nodes_pending = [element for element, type_e in zip(nodes, types) if type_e not in nc.TYPES_EXCLUDED]
        # Define the coordinates as lists of x, y for each node
        coord = [[pos[0], pos[1]] for pos, type_e in zip(list(pos.values()), types) if type_e not in nc.TYPES_EXCLUDED]
        # Apply the DBSCAN clustering algorithm with Euclidean metric
        db = DBSCAN(eps, min_samples=1, algorithm='auto', metric='euclidean').fit(coord)

        # Retrieve the assigned labels
        cluster_labels = db.labels_
        # Get the number of clusters
        num_clusters = len(set(cluster_labels))
        # Convert nodes_pending into an np array for later handling
        nodes_pending = np.array(nodes_pending)

        # Create a dictionary with key=cluster index and values the set of nodes of the cluster
        cluster_dict = {i: nodes_pending[(cluster_labels == i)] for i in range(num_clusters)}

        # Retrieve the indexes of the node list that are of an excluded type
        excluded_indices = [index for index, value in list(enumerate(types)) if value in nc.TYPES_EXCLUDED]
        # convert the cluster labels into a list to insert additional clusters
        cluster_labels = list(cluster_labels)
        # Group all the excluded nodes into the same new cluster label (not really a cluster)
        for i in excluded_indices:
            cluster_labels.insert(i, num_clusters)
        # If we want to avoid clusters with just 1 node
        if avoid_single:
            cluster_labels = self.merge_individual_clusters(nodes, pos, cluster_dict, cluster_labels, types,
                                                            excluded_indices)

        return cluster_dict, cluster_labels

    # Merge individual clusters by joining them into the closest clusters
    # nodes - the array of existing nodes
    # pos - the array of positions created by the corresponding algorithm
    # cluster_dict - the dictionary with key=cluster index and values the set of nodes of the cluster
    # cluster_labels - a list with cluster values for each node
    # types - list of types for each node
    # excluded_indices - elements from the nodes list that are to be excluded
    def merge_individual_clusters(self, nodes, pos, cluster_dict, cluster_labels, types, excluded_indices):
        # Take the coordinates as vectors (TODO remove type if it is no finally needed)
        coord = [(pos[0], pos[1]) for pos, type_e in zip(list(pos.values()), types)]
        # set coordinates far away for the excluded indices, so they are not found as the nearest nodes
        for i in excluded_indices:
            coord[i] = (10e10, 10e10)

        # Turn the nodes into a list for later handling
        list_pending = list(nodes)
        # Retrieve the positions of those clusters with just 1 element
        cluster_one_positions = [idx for idx, val in list(enumerate(cluster_dict.values())) if len(val) == 1]
        # For each of these cluster positions
        for old_cluster in range(len(cluster_one_positions)):
            # Get the nodes from the cluster
            nodes_cluster = cluster_dict.get(cluster_one_positions[old_cluster])
            # Initially they should be one, but as previous iterations may have happened
            # some other nodes may have been already added to this cluster
            if len(nodes_cluster) != 1:
                # Omitting cluster as a previous node was reassigned here
                continue
            # Get the position that correspond to the node in the cluster to use it in other lists
            pos_node = list_pending.index(nodes_cluster[0])
            # Get the type of the node and check if it must be excluded
            if types[pos_node] in nc.TYPES_EXCLUDED:
                # Omitting node as it belongs to one of the excluded types
                continue
            # Get the coordinates of the node
            coord_single = coord[pos_node]
            # And create a list with the coordinates of the rest of the nodes
            rest_coord = coord[:pos_node] + coord[pos_node + 1:]
            # Create a KDTree with the rest of coordinates
            tree = spatial.KDTree(rest_coord)
            # And find the one closest to the single node
            res = tree.query([coord_single])
            # res[1] holds the index in rest_coord. If the index is >=pos_node
            # we must increment 1 to account for the single node when looking into
            # the rest of lists
            pos_other = int(res[1] if res[1] < pos_node else res[1] + 1)
            # Retrieve the cluster label of the nearest node and reassign the single node to that cluster
            cluster_labels[pos_node] = cluster_labels[pos_other]
            # Update that position of the cluster dictionary adding the single node
            cluster_dict[cluster_labels[pos_node]] = np.append(cluster_dict[cluster_labels[pos_node]], nodes[pos_node])
            # And remove it from its old position
            cluster_dict[cluster_one_positions[old_cluster]] = np.array([])
        # Return the updated list of labels
        return cluster_labels
