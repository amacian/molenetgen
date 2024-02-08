import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from KeyValueList import KeyValueList
from ValueList import ValueList
from generator import write_backbone, find_groups, metro_core_split, ring_structure_tel
import pandas as pd
from network import format_distance_limits
import networkconstants as nc


# Class for the Tkinter Topology Generator Application
class MetroGenApp:

    # degrees - degrees of connectivity
    # weights - distribution of % of elements per each degree
    # nodes - number of nodes
    # upper_limits - distance upper limits per ranges. TODO - make it configurable
    # types - dataframe with office codes (e.g. RCO) and proportions
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, degrees, weights, nodes, upper_limits, types, dict_colors={}, initial_refs=None):
        if initial_refs is None:
            initial_refs = []

        self.national_ref_nodes = None
        self.cluster_list = ["-"]
        self.nodes_clusters = {}
        # Variable to hold the default figure
        self.figure = None
        # Root component
        self.root = tk.Tk()
        self.root.geometry("800x800")
        self.root.title("Metro Core Topology Generator")
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
        types["number"] = total.astype(int)

        self.topo, self.distances, self.assigned_types, self.pos, self.colors = \
            metro_core_split(None, degrees, weights, self.upper_limits, types, dict_colors, algo="kamada")
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

        # Start the Tkinter event loop
        self.root.mainloop()

    # Save the information to file
    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            write_backbone(file_path, self.topo, self.distances, self.assigned_types, self.figure,
                           clusters=self.clusters, pos=self.pos, reference_nodes=self.national_ref_nodes)

    def create_tab_save(self, parent):
        frame = ttk.Frame(parent)
        save_button = tk.Button(frame, text="Save to File", command=self.save_to_file)
        save_button.pack(pady=10)
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
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)
        # print(self.colors)
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)

        label_printable = tk.Label(frame,
                                   text=format_distance_limits(self.distances, self.upper_limits),
                                   name="print_distances", anchor="w")
        # label_printable.pack(side=tk.BOTTOM)
        label_printable.grid(row=2, column=0, sticky=tk.S)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
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
                             values=["spectral", "kamada", "spring", "spiral", "shell"])
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
        old_canvas = None
        # Find the old canvas for the image and destroy it
        for i in frame.winfo_children():
            if isinstance(i, tk.Canvas):
                old_canvas = i
                break
        old_canvas.destroy()
        # Retrieve information about source of nodes (input or file)
        node_ref_number = self.node_refs
        selected_cluster_from_file = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster")
        selected_cluster = selected_cluster_from_file.get()

        if selected_cluster != "-" and self.radio_val.get() == "1":
            node_ref_number = self.nodes_clusters[int(selected_cluster)]
            if self.ring_var.get():
                ring_size = self.root.nametowidget("notebook_gen.source_frame.from_file.ring_size")
                (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
                 self.national_ref_nodes) = \
                    ring_structure_tel(int(ring_size.get()), node_ref_number[0], node_ref_number[1],
                                       prefix="R" + selected_cluster + "-")
        # print(node_ref_number)

        if self.radio_val.get() == "0" or not self.ring_var.get():
            # Types and percentages for the nodes
            type_key_vals = type_list.get_entries()
            # Build the expected data structure
            types = pd.DataFrame({'code': [key for key, value in type_key_vals],
                                  'proportion': [float(value) for key, value in type_key_vals]})

            total_proportion = sum(types.proportion)
            total = np.rint(int(node_number.get()) * types.proportion / total_proportion)
            types["number"] = total.astype(int)

            # if self.radio_val.get() == "1"
            #    node_ref_number

            if nc.NATIONAL_CO_CODE in list(types['code']):
                # types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'] = len(self.node_refs)
                types.loc[types['code'] == nc.NATIONAL_CO_CODE, 'number'] = len(node_ref_number)
            else:
                '''row = {'code': nc.NATIONAL_CO_CODE, 'proportion': len(self.node_refs) / int(node_number.get()),
                       'number': len(self.node_refs)}'''
                row = {'code': nc.NATIONAL_CO_CODE, 'proportion': len(node_ref_number) / int(node_number.get()),
                       'number': len(node_ref_number)}
                types = pd.concat([types, pd.DataFrame([row])], ignore_index=True)

            # Prepare the data of degrees and weights as expected by the metro_core function
            key_values = degree_list.get_entries()
            degrees = [int(key) for key, value in key_values]
            weights = [int(value) for key, value in key_values]

            # Try to recover the number of nodes or use 50 as default
            nodes = 50
            try:
                nodes = int(node_number.get())
            except ValueError:
                print("Error in number of nodes, using default: ", nodes)

            # Call the metro function with the expected parameters
            self.topo, self.distances, self.assigned_types, self.pos, self.colors = \
                metro_core_split(None, degrees, weights, self.upper_limits, types, self.color_codes,
                                 algorithm, node_ref_number)
            self.national_ref_nodes = ["" for i in self.topo.nodes]
        # Get x and y coordinates for all the elements
        x_pos = [pos[0] for pos in list(self.pos.values())]
        y_pos = [pos[1] for pos in list(self.pos.values())]
        # Calculate the horizontal and vertical size of the image
        x_size = max(x_pos) - min(x_pos)
        y_size = max(y_pos) - min(y_pos)

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
        # canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)

        # Draw the figure
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)
        # Retrieve the reference to the label where distance ranges and proportions are drawn
        output_label = frame.nametowidget("print_distances")
        output_label['text'] = format_distance_limits(self.distances, self.upper_limits)

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
        # Destroy the canvas
        old_canvas.destroy()

    def enable_nodes(self):
        # frame = self.root.nametowidget("notebook_gen.source_frame")
        subframe_list = self.root.nametowidget("notebook_gen.source_frame.from_nodes")
        subframe_file = self.root.nametowidget("notebook_gen.source_frame.from_file")
        subframe_file.grid_forget()
        subframe_list.grid(row=2, column=0, columnspan=3)
        return

    def from_file(self):
        frame = self.root.nametowidget("notebook_gen.source_frame")
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
            nodes_df = pd.read_excel(file_path, sheet_name="nodes")
            # links_df = pd.read_excel(file_path, sheet_name="links")
            try:
                clusters = nodes_df['Macro region']
            except KeyError:
                print("No clusters found")
            # print("Number of clusters: ", len(set(clusters)))
            label = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster_list")
            label['text'] = self.clusters_as_list_of_nodes(nodes_df['node_name'], clusters)
        return nodes_df

    def clusters_as_list_of_nodes(self, names, clusters):
        names_clusters = set(clusters)
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
                    newline == 10
                text += names[idx] + " "
                newline -=1
            text += "]\n"
            node_names = [names[idx] for idx in idx_for_cluster]
            self.nodes_clusters[name] = node_names
            combo_nodes = self.root.nametowidget("notebook_gen.source_frame.from_file.cluster")
            combo_nodes["values"] = self.cluster_list
        return text
