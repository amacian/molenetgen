import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from KeyValueList import KeyValueList
from MetroCoreGenerator import MetroCoreGenerator, DefaultMetroCoreGenerator, CoordinatesMetroCoreGenerator
from ValueList import ValueList
from generator import write_network
import pandas as pd
from network import format_distance_limits, count_distance_ranges, check_metrics, optimize_distance_ranges, \
    calculate_edge_distances, opt_function, read_network_xls, color_nodes
import networkconstants as nc
import Texts_EN as texts
from DistanceSetterWindow import DistanceSetterWindow
from scipy.optimize import minimize


# Class for the Tkinter Topology Generator Application
class MetroGenApp:
    metro_gen: MetroCoreGenerator

    # degrees - degrees of connectivity
    # weights - distribution of % of elements per each degree
    # nodes - number of nodes
    # upper_limits - distance upper limits per ranges.
    # distance_range_props - proportion for each distance range
    # types - dataframe with office codes (e.g. RCO) and proportions
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, degrees, weights, nodes, upper_limits, distance_range_props, types,
                 dict_colors={}, initial_refs=None, iterations_distance=nc.ITERATIONS_FOR_DISTANCE,
                 bounds=0.05):
        if initial_refs is None:
            initial_refs = []

        self.metro_gen = DefaultMetroCoreGenerator() # CoordinatesMetroCoreGenerator() #

        # Reference national nodes initialized to None
        self.national_ref_nodes = None
        # Cluster list is empty but will be filled if read from file
        self.cluster_list = ["-"]
        # Clusters of nodes will be filled if read from file
        self.nodes_clusters = {}
        # All the node info read from the files and stored into a Dataframe
        self.nodes_from_files = None
        # Variable to hold the default figure
        self.figure = None
        # Root component
        self.root = tk.Tk()
        self.root.geometry("800x800")
        self.root.title("MoleNetwork Metro Core Topology Generator")
        # Value that holds if the single clusters should be removed
        self.remove_single_clusters = tk.BooleanVar(self.root)
        # Color codes depending on the type
        self.color_codes = dict_colors
        # Actual colors per node
        self.colors = []

        # Reference nodes for the metro core
        self.node_refs = initial_refs

        # Default figure height and width
        self.fig_height = 10
        self.fig_width = 10

        # variable for the upper limits of the distances.
        self.upper_limits = upper_limits
        # Requested proportions for distances
        self.req_distance_props = distance_range_props
        # Number of topologies to generate in order to get the one with a distance distribution
        # nearest to the requested one
        self.iterations_distance = iterations_distance
        # Bounds regarding movement of the nodes for optimization of link distances to
        # avoid transforming the graph too much.
        self.bounds = bounds

        # Variable to hold the value of the checkbox to select ring instead of mesh
        self.ring_var = tk.BooleanVar(self.root)

        # Generate the backbone network using the predefined parameters.
        # Retrieve the generated topology, a list with distance per edge,
        # the list of type per node, the name of the node and link sheets at the Excel file
        # the position of the nodes and the assigned colors
        '''self.topo, self.distances, self.assigned_types, \
            self.node_sheet, self.link_sheet, self.pos, self.colors = backbone(degrees, weights, nodes,
                                                                               self.upper_limits, types,
                                                                               dict_colors=dict_colors)'''
        total_proportion = sum(types.proportion)
        total = np.rint(nodes * types.proportion / total_proportion)

        # Proportions might result in a bigger number of nodes, specially if "nodes" is low.
        if sum(total) > nodes:
            decimal_part = (nodes * types.proportion / total_proportion) % 1
            nco_index = types.index[types['code'] == nc.NATIONAL_CO_CODE][0]
            reduce_index = decimal_part[(decimal_part > 0.5) & (decimal_part.index != nco_index)].idxmin()
            total[reduce_index] -= 1
        types["number"] = total.astype(int)

        '''self.topo, self.distances, self.assigned_types, self.pos, self.colors = \
            self.metro_gen.generate_mesh(degrees, weights, self.upper_limits, types, dict_colors,
                                         algo=nc.KAMADA_ALGO)'''
        self.best_fit_topology_n(degrees, weights, types, node_ref_number=[], add_prefix="", extra_node_info=None,
                                 algorithms=[nc.KAMADA_ALGO])
        # Variable to hold the assigned clusters
        self.clusters = None
        self.radio_val = tk.StringVar(value="0")

        # Create a notebook for the tabs
        notebook = ttk.Notebook(self.root, name="notebook_gen")

        # Create tabs and add them to the notebook
        # First tab corresponds to the list of parameters
        # Returns a reference to the tab, another to the list of degree restrictions and
        # another to the type restrictions
        tab1, degree_list, type_list = self.create_tab_list(notebook, "Degrees", "Weights", "Number of nodes",
                                                            "params", "Types", "Weights", degrees, weights, types,
                                                            nodes)
        notebook.add(tab1, text="Metro mesh constraints")

        # The second tab permits selection between set of nodes or reading from file.
        tab2 = self.create_tab_source(notebook)
        notebook.add(tab2, text="Source")
        # The third tab creates the graph and returns a pointer to the canvas
        tab3, canvas = self.create_tab_image(notebook)
        notebook.add(tab3, text="Graph")

        # The fourth tab is the one used to save all the information into an Excel file
        tab4 = self.create_tab_save(notebook)
        notebook.add(tab4, text="Save")

        # Pack the notebook into the main window
        notebook.pack(expand=True, fill=tk.BOTH)

        # Retrieve the pointer to the fields that store the number of nodes
        # and the generation algorithm from the parameters tab (first one)
        node_number = self.root.nametowidget("notebook_gen.params.entry_node")
        algorithm = self.root.nametowidget("notebook_gen.params.algo")
        # Create a Run button that calls back the rerun backbone method
        run_button = tk.Button(self.root, text="Run",
                               command=lambda: self.rerun_metro(tab3, degree_list,
                                                                node_number, type_list,
                                                                algorithm.get()))
        run_button.pack(side=tk.BOTTOM)

        # Variable to hold the window to modify distances
        self.setter = None

        # Start the Tkinter event loop
        self.root.mainloop()

    # Save the information to file
    def save_to_file(self):
        result = False
        message = texts.FILE_NOT_FOUND
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            result, message = write_network(file_path, self.topo, self.distances, self.assigned_types, self.figure,
                                            clusters=self.clusters, pos=self.pos,
                                            reference_nodes=self.national_ref_nodes,
                                            type_nw=nc.METRO_CORE)
        if not result:
            tk.messagebox.showerror('', message)
        else:
            tk.messagebox.showinfo('', texts.COMPLETED)

    def create_tab_save(self, parent):
        frame = ttk.Frame(parent)
        save_button = tk.Button(frame, text="Save to File", command=self.save_to_file)
        save_button.pack(pady=10)
        load_button = tk.Button(frame, text=texts.LOAD_FROM_FILE, command=self.load_topology)
        load_button.pack(pady=10)
        return frame

    def create_tab_source(self, parent):
        frame = ttk.Frame(parent, name="source_frame")
        rb1 = tk.Radiobutton(frame, text="Set of Nodes", variable=self.radio_val, value="0",
                             command=self.enable_nodes)
        rb2 = tk.Radiobutton(frame, text="File", variable=self.radio_val, value="1",
                             command=self.from_file)
        rb1.grid(row=0, column=0)
        rb2.grid(row=0, column=1)

        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=1, columnspan=5, ipadx=300, pady=10)

        subframe = ttk.Frame(frame, name="from_nodes")
        list_nodes = ValueList(subframe, value_name="List of names for main nodes", initial_row=0,
                               initial_value_list=self.node_refs)
        subframe.grid(row=2, column=0, columnspan=2)

        subframe_f = ttk.Frame(frame, name="from_file")
        label = tk.Label(subframe_f, name="select", text="Select file by clicking the button.")
        label.grid(row=0, column=0)
        load_button = tk.Button(subframe_f, text="Load from File", command=self.load_from_file)
        load_button.grid(row=1, column=0)

        label_clusters = tk.Label(subframe_f, name="cluster_list", text="Loaded clusters: ...")
        label_clusters.grid(row=2, column=0)

        separator2 = ttk.Separator(frame, orient="horizontal")
        separator2.grid(row=3, columnspan=5, ipadx=300, pady=10)

        # Clusters
        label_cluster_select = tk.Label(subframe_f, text="Select Cluster")
        label_cluster_select.grid(row=4, column=0)
        combo = ttk.Combobox(subframe_f, name="cluster", state="readonly",
                             values=["-"])
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", self.cluster_selected)
        combo.grid(row=5, columnspan=5)

        ring_check = tk.Checkbutton(subframe_f, text='Generate ring structure (omit mesh constraints)',
                                    variable=self.ring_var, onvalue=1, offvalue=0, state="disabled",
                                    name="ring_check")
        ring_check.grid(row=6, column=0)
        ring_size_label = tk.Label(subframe_f, text="number of rings")
        ring_size_label.grid(row=7, columnspan=5)
        ring_combo = ttk.Combobox(subframe_f, name="ring_size", state="disabled",
                                  values=["1", "2", "3", "4", "6"])
        ring_combo.grid(row=8, columnspan=5)
        # run_button = tk.Button(frame, text="Load from File", command=self.load_from_file)

        return frame

    # Action to be run whenever a cluster is selected to generate a topology
    def cluster_selected(self, event):
        selected_cluster_from_file = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster")
        selected_cluster = selected_cluster_from_file.get()
        nodes_in_cluster = self.nodes_clusters[int(selected_cluster)]
        ring_select_check = self.root.nametowidget("notebook_gen.source_frame.from_file.ring_check")
        ring_combo = self.root.nametowidget("notebook_gen.source_frame.from_file.ring_size")
        if len(nodes_in_cluster) == 2:
            ring_select_check["state"] = "normal"
            ring_combo["state"] = "normal"
            if ring_combo.get() == "":
                ring_combo.set(1)
        else:
            ring_select_check["state"] = "disabled"
            ring_combo["state"] = "disabled"
            self.ring_var.set(False)

    def create_tab_image(self, parent):
        frame = ttk.Frame(parent, name="image_frame")

        x_pos = [pos[0] for pos in list(self.pos.values())]
        y_pos = [pos[1] for pos in list(self.pos.values())]
        x_size = max(x_pos) - min(x_pos)
        y_size = max(y_pos) - min(y_pos)

        # print("x,y:", x_size, ":", y_size)
        x_size = x_size * 10 / y_size
        y_size = 10
        # print("x,y:", x_size, ":", y_size)

        self.figure = plt.Figure(figsize=(x_size, y_size), tight_layout=True, dpi=50)
        self.fig_width = x_size
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # Add the Tkinter canvas to the window
        # canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        canvas_widget.grid(row=0, rowspan=3, column=0, sticky=tk.W + tk.N)
        # print(self.colors)
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)

        btn_set_distances = tk.Button(frame, text="Change \ndistances", command=self.open_dist_window)
        btn_set_distances.grid(row=0, column=1, sticky=tk.N)
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        label_printable = tk.Label(frame,
                                   text=format_distance_limits(self.distances, self.upper_limits,
                                                               req_weights),
                                   name="print_distances", anchor="w")
        # label_printable.pack(side=tk.BOTTOM)
        label_printable.grid(row=1, column=1, sticky=tk.N)
        # frame.rowconfigure(0, weight=1)
        # frame.columnconfigure(0, weight=1)
        return frame, canvas

    def create_tab_list(self, parent, key_names, value_names, node_names, frame_name, type_names,
                        typeval_name, degrees, weights, types, number_nodes_def):
        frame = ttk.Frame(parent, name=frame_name)
        initial_row = 0

        # Degrees
        degree_list = KeyValueList(frame, key_names, value_names, initial_row,
                                   [(str(key), str(value)) for key, value in zip(degrees, weights)])

        # Separator
        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=(7 + initial_row), columnspan=5, ipadx=300, pady=10)

        # Types
        type_list = KeyValueList(frame, type_names, typeval_name, initial_row + 8,
                                 [(str(key), str(value)) for key, value in zip(types.code, types.proportion)])

        # Separator
        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=(16 + initial_row), columnspan=5, ipadx=300, pady=10)

        # Number of nodes
        label_nodes = tk.Label(frame, text=node_names)
        entry_node = tk.Entry(frame, name="entry_node")
        entry_node.insert(0, number_nodes_def)

        label_nodes.grid(row=(17 + initial_row), column=0, pady=5)
        entry_node.grid(row=(17 + initial_row), column=1, pady=5)

        # Algorithm
        label_algo = tk.Label(frame, text="Algorithm")
        combo = ttk.Combobox(frame, name="algo", state="readonly",
                             values=[nc.ALL_GEN, nc.SPECTRAL_ALGO, nc.KAMADA_ALGO, nc.SPRING_ALGO, nc.SPIRAL_ALGO,
                                     nc.SHELL_ALGO])
        combo.current(1)
        label_algo.grid(row=(18 + initial_row), column=0, pady=5)
        combo.grid(row=(18 + initial_row), column=1, pady=5)

        return frame, degree_list, type_list

    # Create a new graph with the same parameters
    # Frame - a reference to the frame where the graph is drawn
    # degree_list - A key value list with the link degree proportion
    # node_number - the number of nodes in the topology
    # type_list - A key value list with the proportion of types
    # algorithm - The algorithm to be used in the graph generation
    def rerun_metro(self, frame, degree_list: KeyValueList, node_number,
                    type_list: KeyValueList, algorithm):
        # old_canvas = None
        # Find the old canvas for the image and destroy it
        # for i in frame.winfo_children():
        #     if isinstance(i, tk.Canvas):
        #        old_canvas = i
        #        break
        # old_canvas.destroy()
        # Get the information of the nodes introduced by the user in the list of reference national nodes
        # when writing them in the GUI. They have priority over the number of national nodes.
        node_ref_number = self.node_refs
        # Get which cluster is selected in the Combo, either "-" or an actual cluster from file
        selected_cluster_from_file = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster")
        selected_cluster = selected_cluster_from_file.get()

        extra_node_info = None
        # If a cluster is selected and the Load from file option is marked
        if selected_cluster != "-" and self.radio_val.get() == "1":
            # Replace the nodes from the GuI by the nodes for the selected cluster that were read from file
            node_ref_number = self.nodes_clusters[int(selected_cluster)]
            # Get all the info for those nodes
            extra_node_info = self.nodes_from_files.loc[self.nodes_from_files['node_name'].isin(node_ref_number)]
            # If the user selected to create a ring structure instead of a mesh structure
            if self.ring_var.get():
                # Get the expected number of rings in the structure and generate it
                ring_size = self.root.nametowidget("notebook_gen.source_frame.from_file.ring_size")
                (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
                 self.national_ref_nodes) = \
                    self.metro_gen.generate_ring(int(ring_size.get()),
                                                 node_ref_number[0],
                                                 node_ref_number[1],
                                                 prefix="R" + selected_cluster + "-",
                                                 dict_colors=self.color_codes)

        # Only if the user selected the mesh option (either introducing the info at the GUI
        # or mapping a cluster read from file to a Mesh and not to a ring)
        if self.radio_val.get() == "0" or not self.ring_var.get():
            # Types and percentages for the nodes
            type_key_vals = type_list.get_entries()
            # Build the expected data structure
            types = pd.DataFrame({'code': [key for key, value in type_key_vals],
                                  'proportion': [float(value) for key, value in type_key_vals]})
            # Calculate the sum of values for proportions as it might not be 100%
            total_proportion = sum(types.proportion)

            # Try to recover the number of nodes or use 50 as default
            nodes = 50
            try:
                nodes = int(node_number.get())
            except ValueError:
                print("Error in number of nodes, using default: ", nodes)

            total = np.rint(nodes * types.proportion / total_proportion)

            # National nodes, if defined, should be considered as a fixed number of nodes.
            if len(node_ref_number) > 0:
                nodes_other_types = nodes - len(node_ref_number)
                types_no_nco = types[types['code'] != nc.NATIONAL_CO_CODE]
                total_no_nco_proportion = sum(types_no_nco.proportion)
                total = np.rint(nodes_other_types * types_no_nco.proportion / total_no_nco_proportion)
                if nc.NATIONAL_CO_CODE in list(types['code']):
                    nco_index = types.index[types['code'] == nc.NATIONAL_CO_CODE][0]
                    total[nco_index] = len(node_ref_number)

            # Proportions might result in a bigger number of nodes, specially if "nodes" is low.
            if sum(total) > nodes:
                decimal_part = (nodes * types.proportion / total_proportion) % 1
                nco_index = types.index[types['code'] == nc.NATIONAL_CO_CODE][0]
                reduce_index = decimal_part[(decimal_part > 0.5) & (decimal_part.index != nco_index)].idxmin()
                total[reduce_index] -= 1
            types["number"] = total.astype(int)

            # Types defined by the user might not include national nodes
            # But they might have included reference nodes either in the GUI list or read from file
            if nc.NATIONAL_CO_CODE in list(types['code']):
                # Only if the list is not empty
                if len(node_ref_number) > 0:
                    types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'] = len(node_ref_number)
            else:
                # If the user defined the NCO type, the proportion and values need to be re-writen to match
                # the reference nodes. TODO: consider if BB clusters read from file include other types of nodes
                row = {'code': nc.NATIONAL_CO_CODE, 'proportion': len(node_ref_number) / int(node_number.get()),
                       'number': len(node_ref_number)}
                types = pd.concat([types, pd.DataFrame([row])], ignore_index=True)

            # Prepare the data of degrees and weights as expected by the metro_core function
            key_values = degree_list.get_entries()
            degrees = [int(key) for key, value in key_values]
            weights = [int(value) for key, value in key_values]

            add_prefix = ""
            if selected_cluster != "-":
                add_prefix = "_" + selected_cluster + "_"
            # Call the metro function with the expected parameters
            '''self.topo, self.distances, self.assigned_types, self.pos, self.colors = \
                self.metro_gen.generate_mesh(degrees, weights, self.upper_limits, types, self.color_codes,
                                             algorithm, node_ref_number, add_prefix, extra_node_info)'''
            algorithms = nc.MAIN_ALGORITHMS if algorithm == nc.ALL_GEN else [algorithm]
            self.best_fit_topology_n(degrees, weights, types, node_ref_number, add_prefix, extra_node_info,
                                     algorithms)
            self.national_ref_nodes = ["" for i in self.topo.nodes]

        self.update_image_frame()
        # Get x and y coordinates for all the elements
        # x_pos = [pos[0] for pos in list(self.pos.values())]
        # y_pos = [pos[1] for pos in list(self.pos.values())]
        # Calculate the horizontal and vertical size of the image
        # x_size = max(x_pos) - min(x_pos)
        # y_size = max(y_pos) - min(y_pos)

        # if x_size > y_size:
        #    y_size = y_size * 12 / x_size
        #    x_size = 12
        #else:
        #    x_size = x_size * 12 / y_size
        #    y_size = 12

        # size_ratio = x_size / self.fig_width
        # Change the figure width based on this and prepare the canvas and widgets
        # self.fig_width = x_size
        # self.figure = plt.Figure(figsize=(x_size, y_size),
        #                         tight_layout=True, dpi=50)
        # ax = self.figure.add_subplot(111)
        # canvas = FigureCanvasTkAgg(self.figure, master=frame)
        # canvas_widget = canvas.get_tk_widget()

        # canvas_widget.config(width=x_size, height=y_size)
        # Add the Tkinter canvas to the window
        # canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        # canvas_widget.grid(row=0, column=0, rowspan=3, sticky=tk.W + tk.N)

        # Draw the figure
        #nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
        #        node_color=self.colors, ax=ax)
        # Retrieve the reference to the label where distance ranges and proportions are drawn
        # output_label = frame.nametowidget("print_distances")
        # req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        # output_label['text'] = format_distance_limits(self.distances, self.upper_limits, req_weights)

    def enable_nodes(self):
        # frame = self.root.nametowidget("notebook_gen.source_frame")
        subframe_list = self.root.nametowidget("notebook_gen.source_frame.from_nodes")
        subframe_file = self.root.nametowidget("notebook_gen.source_frame.from_file")
        subframe_file.grid_forget()
        subframe_list.grid(row=2, column=0, columnspan=3)
        return

    def from_file(self):
        # frame = self.root.nametowidget("notebook_gen.source_frame")
        subframe_list = self.root.nametowidget("notebook_gen.source_frame.from_nodes")
        subframe_file = self.root.nametowidget("notebook_gen.source_frame.from_file")
        subframe_list.grid_forget()
        subframe_file.grid(row=2, column=0, columnspan=3)
        return

    def load_from_file(self):
        file_path = filedialog.askopenfilename(title="Open Topology", defaultextension=".xlsx",
                                               filetypes=[("Excel files", ".xlsx .xls")])
        nodes_df = None
        clusters = []
        if file_path:
            nodes_df = pd.read_excel(file_path, sheet_name=nc.NODES_EXCEL_NAME)
            # links_df = pd.read_excel(file_path, sheet_name="links")
            try:
                clusters = nodes_df[nc.XLS_CLUSTER]
                self.nodes_from_files = nodes_df
            except KeyError:
                print("No clusters found")
            # print("Number of clusters: ", len(set(clusters)))
            label = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster_list")
            label['text'] = self.clusters_as_list_of_nodes(nodes_df['node_name'], clusters)
        return

    def clusters_as_list_of_nodes(self, names, clusters):
        names_clusters = set(clusters)
        if -1 in names_clusters:
            names_clusters.remove(-1)
        self.cluster_list = list(names_clusters)
        self.cluster_list.sort()
        self.nodes_clusters = {}
        text = "Defined clusters:\n"
        for name in names_clusters:
            idx_for_cluster = [pos for pos, value in enumerate(clusters) if value == name]
            text += "Cluster " + str(name) + "["
            newline = 10
            for idx in idx_for_cluster:
                if newline == 1:
                    text += "\n"
                    newline = 10
                text += names[idx] + " "
                newline -= 1
            text += "]\n"
            node_names = [names[idx] for idx in idx_for_cluster]
            self.nodes_clusters[name] = node_names
            combo_nodes = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster")
            combo_nodes["values"] = self.cluster_list
        return text

    def best_fit_topology_n(self, degrees, weights, types, node_ref_number, add_prefix, extra_node_info,
                            algorithms):
        topo, distances, assigned_types, pos, colors = None, None, None, None, None

        ref_mape = 1000
        for algorithm in algorithms:
            for i in range(self.iterations_distance):
                # Generate the network using the predefined parameters.
                topo, distances, assigned_types, pos, colors = \
                    self.metro_gen.generate_mesh(degrees, weights, self.upper_limits, types, self.color_codes,
                                                 algorithm, node_ref_number, add_prefix, extra_node_info)

                # Calculate weights from requested proportions and regenerate distances optimizing the
                # mean error
                req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
                # Optimization by minimizing the optional function
                np_pos = np.array(list(pos.values()))
                bound = [(a - self.bounds, a + self.bounds) for a in np_pos.flatten()]
                opt_min = minimize(opt_function, np_pos.flatten(),
                                   args=(topo, self.upper_limits, self.req_distance_props),
                                   method='L-BFGS-B', options={'eps': 1}, bounds=bound)
                print("Iteration ", i, " for algorithm ", algorithm)
                new_pos = {node: position for node, position in
                           zip(list(topo.nodes), opt_min.x.reshape((len(topo.nodes), 2)))}
                opt_distances = calculate_edge_distances(topo, new_pos,
                                                         max(self.upper_limits))
                pos = new_pos
                # distances = optimize_distance_ranges(self.upper_limits, req_weights, distances)
                distances = optimize_distance_ranges(self.upper_limits, req_weights, opt_distances)
                # Calculate error metrics to try to select the best topology
                new_distance_weight = [dist / 100 for dist in count_distance_ranges(distances, self.upper_limits)]
                mae, mape_distance, rsme_distance, actual_dist = check_metrics(self.upper_limits, req_weights,
                                                                               new_distance_weight, perc=True)

                if mape_distance < ref_mape:
                    self.topo, self.distances, self.assigned_types, \
                        self.pos, self.colors = topo, distances, assigned_types, \
                        pos, colors
                    ref_mape = mape_distance

    # Open the window that will modify the distance ranges.
    def open_dist_window(self):
        if self.setter is None:
            # self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.max_upper)
            self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.req_distance_props,
                                               max(self.distances), self.iterations_distance, self.bounds)
        else:
            self.setter.show(self.upper_limits, self.req_distance_props, max(self.distances), self.iterations_distance,
                             self.bounds)

    def set_upper_limits(self, upper_limits, req_proportions, max_distance, iterations=None, bounds=None):
        if iterations is not None:
            self.iterations_distance = iterations
        if bounds is not None:
            self.bounds = bounds
        # variable for the upper limits of the distances.
        self.upper_limits = upper_limits
        # Requested proportions for distances
        self.req_distance_props = req_proportions
        # Calculate distances based on this parameter
        self.set_distance_parameters()
        # Update the description of % per link
        output_label = self.root.nametowidget("notebook_gen.image_frame.print_distances")
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        output_label['text'] = format_distance_limits(self.distances, self.upper_limits, req_weights)

    def set_distance_parameters(self):
        self.distances = calculate_edge_distances(self.topo, self.pos, max(self.distances))
        # Calculate weights from requested proportions and regenerate distances optimizing the
        # mean error
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        self.distances = optimize_distance_ranges(self.upper_limits, req_weights, self.distances)

    # Load a topology from file
    def load_topology(self):
        # Ask for the filepath to the file to read
        file_path = filedialog.askopenfilename(title="Open Topology", defaultextension=".xlsx",
                                               filetypes=[("Excel files", ".xlsx .xls")])

        # If it does not exist then show an error
        if not file_path:
            tk.messagebox.showerror('', texts.ERROR_READING)
            return

        # Read the BB information from the file
        res, topo, distances, assigned_types, pos, clusters = read_network_xls(file_path, ntype=nc.METRO_CORE)

        # If the result if False, then show an error
        if not res:
            tk.messagebox.showerror('', texts.ERROR_READING)
        else:
            # Otherwise assign the parameters
            self.topo, self.distances, self.assigned_types, self.clusters, self.pos = \
                topo, distances, assigned_types, clusters, pos
            # Generate the colors
            self.colors = color_nodes(assigned_types, self.color_codes)
            # And update the images without reclustering
            self.update_image_frame()
        return

    def update_image_frame(self):
        old_canvas = None

        frame = self.root.nametowidget("notebook_gen.image_frame")
        # Find the old canvas for the image and destroy it
        for i in frame.winfo_children():
            if isinstance(i, tk.Canvas):
                old_canvas = i
                break
        old_canvas.destroy()

        # Get x and y coordinates for all the elements
        x_pos = [pos[0] for pos in list(self.pos.values())]
        y_pos = [pos[1] for pos in list(self.pos.values())]
        # Calculate the horizontal and vertical size of the image
        x_size = max(x_pos) - min(x_pos)
        y_size = max(y_pos) - min(y_pos)

        # y_size will be kept always as 10 while x_size is resized to keep proportions
        if x_size > y_size:
            y_size = y_size * 12 / x_size
            x_size = 12
        else:
            x_size = x_size * 12 / y_size
            y_size = 12

        # size_ratio = x_size / self.fig_width
        # Change the figure width based on this and prepare the canvas and widgets
        self.fig_width = x_size
        self.figure = plt.Figure(figsize=(x_size, y_size),
                                 tight_layout=True, dpi=50)
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # canvas_widget.config(width=x_size, height=y_size)
        # Add the Tkinter canvas to the window
        canvas_widget.grid(row=0, column=0, rowspan=7, sticky=tk.W + tk.N)
        # Draw the figure
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)
        # Retrieve the reference to the label where distance ranges and proportions are drawn
        output_label = frame.nametowidget("print_distances")
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        output_label['text'] = format_distance_limits(self.distances, self.upper_limits, req_weights)