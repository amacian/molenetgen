import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Texts_EN
import networkconstants as nc
from ClusterGenerator import ClusterGenerator, DistanceBasedClusterGenerator, DistanceConnectedBasedClusterGenerator
from BackboneGenerator import BackboneGenerator, DefaultBackboneGenerator, DualBackboneGenerator, WaxmanPavenGenerator
from DistanceSetterWindow import DistanceSetterWindow
from KeyValueList import KeyValueList
from generator import write_network
import pandas as pd
from network import format_distance_limits, calculate_edge_distances, optimize_distance_ranges, count_distance_ranges, \
    check_metrics
import Texts_EN as texts


# Class for the Tkinter Topology Generator Application
class BackboneGenApp:
    back_gen: BackboneGenerator
    cluster_gen: ClusterGenerator

    # degrees - degrees of connectivity
    # weights - distribution of % of elements per each degree
    # nodes - number of nodes
    # upper_limits - distance upper limits per ranges.
    # distance_range_props - proportion for each distance range
    # types - dataframe with office codes (e.g. RCO) and proportions
    # dict_colors - dictionary that maps types to colors for the basic graph
    # iterations_distance - number of topologies to generate to get the best fit for the distances
    def __init__(self, degrees, weights, nodes, upper_limits, distance_range_props, types,
                 dict_colors={}, iterations_distance=nc.ITERATIONS_FOR_DISTANCE):
        # Store the distance ranges, the requested proportions and the max distance value
        self.upper_limits = upper_limits
        self.req_distance_props = distance_range_props
        self.max_upper = max(self.upper_limits)
        # Number of topologies to generate in order to get the one with a distance distribution
        # nearest to the requested one
        self.iterations_distance = iterations_distance

        self.distances = None
        # Backbone generator object
        self.back_gen = DefaultBackboneGenerator()
        # Cluster generator object
        self.cluster_gen = DistanceBasedClusterGenerator()
        # self.cluster_gen = DistanceConnectedBasedClusterGenerator()
        # Variable to hold the default figure
        self.figure = None
        # Root component
        self.root = tk.Tk()
        self.root.title(Texts_EN.BACKBONE_APP_TITLE)
        self.root.geometry('800x800')
        # Value that holds if the single clusters should be removed
        self.remove_single_clusters = tk.BooleanVar(self.root)
        # Color codes depending on the type
        self.color_codes = dict_colors
        # Actual colors per node
        self.colors = []

        # Default figure height and width
        self.fig_height = 10
        self.fig_width = 10

        self.best_fit_topology_n(degrees, weights, nodes, types, dict_colors)

        # Variable to hold the assigned clusters
        self.clusters = None

        # Create a notebook for the tabs
        notebook = ttk.Notebook(self.root, name="notebook_gen")

        # Create tabs and add them to the notebook
        # First tab corresponds to the list of parameters
        # Returns a reference to the tab, another to the list of degree restrictions and
        # another to the type restrictions
        tab1, degree_list, type_list = self.create_tab_list(notebook, "Degrees", "Weights", "Number of nodes",
                                                            "params", "Types", "Weights", degrees, weights, types,
                                                            nodes)
        notebook.add(tab1, text=Texts_EN.TAB_CONTRAINTS)

        # The second tab creates the graph and returns a pointer to the canvas
        tab2, canvas = self.create_tab_image(notebook)
        notebook.add(tab2, text="Graph")

        # The third tab corresponds to the creation of the clusters
        # and the image representing them
        tab3 = self.create_tab_grouping(notebook, "group_tab")
        notebook.add(tab3, text="Grouping")

        # The fourth tab is the one used to save all the information into an Excel file
        tab4 = self.create_tab_save(notebook)
        notebook.add(tab4, text="Save")

        # Pack the notebook into the main window
        notebook.pack(expand=True, fill=tk.BOTH)

        # Retrieve the pointer to the fields that store the number of nodes
        # and the generation algorithm from the parameters tab (first one)
        node_number = self.root.nametowidget("notebook_gen.params.entry_node")
        algorithm = self.root.nametowidget("notebook_gen.params.algo")
        generator = self.root.nametowidget("notebook_gen.params.gen")
        # Create a Run button that calls back the rerun backbone method
        run_button = tk.Button(self.root, text="Run",
                               command=lambda: self.rerun_backbone(tab2, degree_list,
                                                                   node_number, type_list,
                                                                   algorithm, generator.get()))
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
                                            clusters=self.clusters, pos=self.pos)
        if not result:
            tk.messagebox.showerror('', message)
        else:
            tk.messagebox.showinfo('', texts.COMPLETED)

    def create_tab_save(self, parent):
        frame = ttk.Frame(parent)
        save_button = tk.Button(frame, text=texts.SAVE_TO_FILE, command=self.save_to_file)
        save_button.pack(pady=10)
        return frame

    def create_tab_image(self, parent):
        frame = ttk.Frame(parent, name="image_frame")

        x_pos = [pos[0] for pos in list(self.pos.values())]
        y_pos = [pos[1] for pos in list(self.pos.values())]
        x_size = max(x_pos) - min(x_pos)
        y_size = max(y_pos) - min(y_pos)

        # Resize to have always the same Y size
        x_size = x_size * 10 / y_size
        y_size = 10

        self.figure = plt.Figure(figsize=(x_size, y_size), tight_layout=True, dpi=50)
        self.fig_width = x_size
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        canvas_widget.grid(row=0, rowspan=7, column=0, sticky=tk.W + tk.N)
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)

        btn_set_distances = tk.Button(frame, text="Change \ndistances", command=self.open_dist_window)
        btn_set_distances.grid(row=0, column=1, sticky=tk.N)
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        label_printable = tk.Label(frame,
                                   text=format_distance_limits(self.distances, self.upper_limits, req_weights),
                                   name="print_distances", anchor="w")
        label_printable.grid(row=1, column=1, sticky=tk.N)

        # List of links
        combo = ttk.Combobox(frame, name="link_list", state="readonly",
                             values=["-"])
        combo.current(0)
        combo.grid(row=2, column=1)
        self.fill_link_list_combo()
        # Button to delete the selected link from the list
        btn_del_link = tk.Button(frame, text="Del link", command=self.del_link)
        btn_del_link.grid(row=3, column=1, sticky=tk.N)
        combo_node = ttk.Combobox(frame, name="source_list", state="readonly",
                                  values=["-"])
        combo_node.current(0)
        combo_node.grid(row=4, column=1)
        self.fill_node_source_combo()
        combo_node.bind("<<ComboboxSelected>>", self.fill_node_dest_combo)
        combo_dest = ttk.Combobox(frame, name="dest_list", state="readonly",
                                  values=["-"])
        combo_dest.current(0)
        combo_dest.grid(row=5, column=1)
        # Button to add a link
        btn_add_link = tk.Button(frame, text="Add link", command=self.add_link)
        btn_add_link.grid(row=6, column=1, sticky=tk.N)
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
                             values=[nc.SPECTRAL_ALGO, nc.KAMADA_ALGO, nc.SPRING_ALGO, nc.SPIRAL_ALGO, nc.SHELL_ALGO])
        combo.current(0)
        label_algo.grid(row=(18 + initial_row), column=0, pady=5)
        combo.grid(row=(18 + initial_row), column=1, pady=5)

        # Generator
        label_gen = tk.Label(frame, text="Generator")
        combo_gen = ttk.Combobox(frame, name="gen", state="readonly",
                                 values=["Default", "Dual", "Region"])
        combo_gen.current(0)
        label_gen.grid(row=(19 + initial_row), column=0, pady=5)
        combo_gen.grid(row=(19 + initial_row), column=1, pady=5)
        return frame, degree_list, type_list

    def create_tab_grouping(self, parent, frame_name):
        frame = ttk.Frame(parent, name=frame_name)
        label_EPS = tk.Label(frame, text="Select max epsilon to group")
        label_EPS.grid(row=0, column=0)
        slider = tk.Scale(frame, name="max_group", from_=0.001, to=0.5,
                         orient=tk.HORIZONTAL, resolution=0.001)
        slider.set(0.03)
        slider.grid(row=0, column=1)
        label_single = tk.Label(frame, text="Avoid single nodes")
        label_single.grid(row=1, column=0)
        check = ttk.Checkbutton(frame, name="avoid_single", variable=self.remove_single_clusters)
        check.grid(row=1, column=1, pady=5)
        label_algo = tk.Label(frame, text="Select clustering strategy")
        label_algo.grid(row=2, column=0)
        combo = ttk.Combobox(frame, name="algo_cluster",
                             values=["By distance only", "By distance and Connected"])
        combo.current(0)
        combo.grid(row=2, column=1, pady=5)
        group_button = tk.Button(frame, text="Search for groups", command=self.group_graph)
        group_button.grid(row=2, column=2, pady=5)
        label_groups = tk.Label(frame,
                                text="",
                                name="print_groups", anchor="w")
        # label_printable.pack(side=tk.BOTTOM)
        label_groups.grid(row=3, column=0, columnspan=2, sticky=tk.S)

        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.group_graph()

        return frame

    # Create a new graph with the same parameters
    # Frame - a reference to the frame where the graph is drawn
    # degree_list - A key value list with the link degree proportion
    # node_number - the number of nodes in the topology
    # type_list - A key value list with the proportion of types
    # algorithm - The algorithm to be used in the graph generation
    def rerun_backbone(self, frame, degree_list: KeyValueList, node_number,
                       type_list: KeyValueList, algorithm, generator):
        if generator == "Default":
            self.back_gen = DefaultBackboneGenerator()
        elif generator == "Region":
            self.back_gen = WaxmanPavenGenerator()
        else:
            self.back_gen = DualBackboneGenerator()

        # Types and percentages for the nodes
        type_key_vals = type_list.get_entries()
        # Build the expected data structure
        types = pd.DataFrame({'code': [key for key, value in type_key_vals],
                              'proportion': [float(value) for key, value in type_key_vals]})

        # Prepare the data of degrees and weights as expected by the backbone function
        key_values = degree_list.get_entries()
        degrees = [int(key) for key, value in key_values]
        weights = [int(value) for key, value in key_values]

        # Try to recover the number of nodes or use 50 as default
        nodes = 50
        try:
            nodes = int(node_number.get())
        except ValueError:
            print("Error in number of nodes, using default: ", nodes)

        # Call the backbone function with the expected parameters
        self.best_fit_topology_n(degrees, weights, nodes, types, self.color_codes, algorithm.get())

        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        self.distances = optimize_distance_ranges(self.upper_limits, req_weights, self.distances)

        self.update_image_frame()

    # Regenerate the group graph
    def group_graph(self):
        # Find the combo that defines the eps
        slider = self.root.nametowidget("notebook_gen.group_tab.max_group")
        # Find the combo that defines the Clustering method
        cluster_alg = self.root.nametowidget("notebook_gen.group_tab.algo_cluster")
        match cluster_alg.get():
            case "By distance only":
                self.cluster_gen = DistanceBasedClusterGenerator()
            case _:
                self.cluster_gen = DistanceConnectedBasedClusterGenerator()
        # Call the function that generates the tab
        groups, self.clusters = self.cluster_gen.find_groups(self.topo, self.assigned_types, self.pos,
                                                             eps=float(slider.get()),
                                                             avoid_single=self.remove_single_clusters.get())
        # Retrieve a reference to the frame and to the label included in that frame
        frame = self.root.nametowidget("notebook_gen.group_tab")
        label = self.root.nametowidget("notebook_gen.group_tab.print_groups")
        # Remove the old figure with clusters from the canvas
        self.remove_old_group_figure_canvas()
        # And create a new one
        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height),
                                 tight_layout=True, dpi=50)
        # Prepare an exist to plot it and the related canvas
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # canvas_widget.config(width=x_size, height=y_size)
        # Add the Tkinter canvas to the window
        canvas_widget.grid(row=3, column=0, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
        # canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # Define the color map that will be used to represent the clusters
        color_map = plt.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=1, vmax=max(self.clusters))
        # Draw the result
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=color_map(norm(self.clusters)), ax=ax)

        # We might want to include some descriotion in the future
        label['text'] = ""  # format_groups(groups)

    # find and remove the group_tab figure canvas
    def remove_old_group_figure_canvas(self):
        # Get the frame of the group tab
        frame = self.root.nametowidget("notebook_gen.group_tab")
        old_canvas = None
        # Traverse the children until we find one of type tk.Canvas
        for i in frame.winfo_children():
            if isinstance(i, tk.Canvas):
                old_canvas = i
                break
        # Detroy the canvas
        old_canvas.destroy()

    # Open the window that will modify the distance ranges.
    def open_dist_window(self):
        if self.setter is None:
            # self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.max_upper)
            self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.req_distance_props,
                                                   self.max_upper, self.iterations_distance)
        else:
            self.setter.show(self.upper_limits, self.req_distance_props, self.max_upper, self.iterations_distance)

    # Method to repaint the image frame with the new image
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
        # Call to the creation of the grouping/cluster graph with existing values
        self.group_graph()
        self.fill_link_list_combo()
        self.fill_node_source_combo()
        self.fill_node_dest_combo(None)

    # Method to fill the link list combo
    def fill_link_list_combo(self):
        combo = self.root.nametowidget("notebook_gen.image_frame.link_list")
        edges = list(self.topo.edges)
        edges.insert(0, "-")
        combo["values"] = edges
        combo.set("-")

    # Fill the node source combo with all the nodes from the topology
    def fill_node_source_combo(self):
        combo = self.root.nametowidget("notebook_gen.image_frame.source_list")
        nodes = list(self.topo.nodes)
        nodes.sort()
        nodes.insert(0, "-")
        combo["values"] = nodes
        combo.set("-")

    # Fill with all the nodes not already connected to this one
    def fill_node_dest_combo(self, event):
        combo_src = self.root.nametowidget("notebook_gen.image_frame.source_list")
        combo = self.root.nametowidget("notebook_gen.image_frame.dest_list")
        selected_node = combo_src.get()
        values = []
        if selected_node != "_":
            existing_linked_nodes = [v for u, v in self.topo.edges(selected_node)]
            values = [i for i in self.topo.nodes if i not in existing_linked_nodes and i != selected_node]
            values.sort()
        values.insert(0, "-")
        combo["values"] = values
        combo.set("-")

    # Delete the selected link from the graph
    def del_link(self):
        combo = self.root.nametowidget("notebook_gen.image_frame.link_list")
        selected = combo.get()
        if selected == "-":
            tk.messagebox.showerror("", "No link selected")
            return

        nodes = selected.split(" ")
        self.topo.remove_edge(nodes[0], nodes[1])
        self.update_image_frame()
        return

    def add_link(self):
        combo_src = self.root.nametowidget("notebook_gen.image_frame.source_list")
        combo = self.root.nametowidget("notebook_gen.image_frame.dest_list")
        source_node = combo_src.get()
        dest_node = combo.get()
        if source_node == "-" or dest_node == "-":
            tk.messagebox.showerror("", "No link selected")
            return
        self.topo.add_edge(source_node, dest_node)
        self.update_image_frame()

    def set_distance_parameters(self):
        self.distances = calculate_edge_distances(self.topo, self.pos, self.max_upper)
        # Calculate weights from requested proportions and regenerate distances optimizing the
        # mean error
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        self.distances = optimize_distance_ranges(self.upper_limits, req_weights, self.distances)

    def set_upper_limits(self, upper_limits, req_proportions, max_distance, iterations=None):
        if iterations is not None:
            self.iterations_distance = iterations
        # variable for the upper limits of the distances.
        self.upper_limits = upper_limits
        # Requested proportions for distances
        self.req_distance_props = req_proportions
        # Set the highest value to the point in the middle of the highest range
        self.max_upper = max_distance
        # Calculate distances based on this parameter
        self.set_distance_parameters()
        # Update the description of % per link
        output_label = self.root.nametowidget("notebook_gen.image_frame.print_distances")
        req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
        output_label['text'] = format_distance_limits(self.distances, self.upper_limits, req_weights)

    def best_fit_topology_n(self, degrees, weights, nodes, types, dict_colors, algorithm="spectral"):
        aux_topo, aux_distances, aux_assigned_types, \
            aux_node_sheet, aux_link_sheet, \
            aux_pos, aux_colors = None, None, None, None, None, None, None
        ref_mape = 1000
        for i in range(self.iterations_distance):
            # Generate the backbone network using the predefined parameters.
            # Retrieve the generated topology, a list with distance per edge,
            # the list of type per node, the name of the node and link sheets at the excel file
            # the position of the nodes and the assigned colors
            aux_topo, aux_distances, aux_assigned_types, \
                aux_node_sheet, aux_link_sheet, \
                aux_pos, aux_colors = self.back_gen.generate(degrees, weights, nodes,
                                                             self.upper_limits, types,
                                                             algo=algorithm,
                                                             dict_colors=dict_colors,
                                                             max_distance=self.max_upper)

            # Calculate weights from requested proportions and regenerate distances optimizing the
            # mean error
            req_weights = [i / sum(self.req_distance_props) for i in self.req_distance_props]
            aux_distances = optimize_distance_ranges(self.upper_limits, req_weights, aux_distances)
            # Calculate error metrics to try to select the best topology
            new_distance_weight = [dist / 100 for dist in count_distance_ranges(aux_distances, self.upper_limits)]
            mae, mape_distance, rsme_distance, actual_dist = check_metrics(self.upper_limits, req_weights,
                                                                           new_distance_weight, perc=True)

            if mape_distance < ref_mape:
                self.topo, self.distances, self.assigned_types, \
                    self.node_sheet, self.link_sheet, \
                    self.pos, self.colors = aux_topo, aux_distances, aux_assigned_types, \
                    aux_node_sheet, aux_link_sheet, \
                    aux_pos, aux_colors
                ref_mape = mape_distance
