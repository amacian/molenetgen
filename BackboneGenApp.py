import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkconstants as nc
from ClusterGenerator import ClusterGenerator, DistanceBasedClusterGenerator
from BackboneGenerator import BackboneGenerator, DefaultBackboneGenerator, DualBackboneGenerator
from KeyValueList import KeyValueList
from generator import write_backbone
import pandas as pd
from network import format_distance_limits


# Class for the Tkinter Topology Generator Application
class BackboneGenApp:
    back_gen: BackboneGenerator
    cluster_gen: ClusterGenerator

    # degrees - degrees of connectivity
    # weights - distribution of % of elements per each degree
    # nodes - number of nodes
    # upper_limits - distance upper limits per ranges. TODO - make it configurable
    # types - dataframe with office codes (e.g. RCO) and proportions
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, degrees, weights, nodes, upper_limits, types, dict_colors={}):
        # Backbone generator object
        self.back_gen = DefaultBackboneGenerator()
        # Cluster generator object
        self.cluster_gen = DistanceBasedClusterGenerator()
        # Variable to hold the default figure
        self.figure = None
        # Root component
        self.root = tk.Tk()
        self.root.title("Backbone Topology Generator")
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

        # variable for the upper limits of the distances.
        self.upper_limits = upper_limits

        # Generate the backbone network using the predefined parameters.
        # Retrieve the generated topology, a list with distance per edge,
        # the list of type per node, the name of the node and link sheets at the excel file
        # the position of the nodes and the assigned colors
        self.topo, self.distances, self.assigned_types, \
            self.node_sheet, self.link_sheet, \
            self.pos, self.colors = self.back_gen.generate(degrees, weights, nodes,
                                                           self.upper_limits, types,
                                                           dict_colors=dict_colors)
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
        notebook.add(tab1, text="Degree constraints")

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

        # Start the Tkinter event loop
        self.root.mainloop()

    # Save the information to file
    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            write_backbone(file_path, self.topo, self.distances, self.assigned_types, self.figure,
                           clusters=self.clusters, pos=self.pos)

    def create_tab_save(self, parent):
        frame = ttk.Frame(parent)
        save_button = tk.Button(frame, text="Save to File", command=self.save_to_file)
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

        # Add the Tkinter canvas to the window
        # canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
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
                             values=[nc.SPECTRAL_ALGO, nc.KAMADA_ALGO, nc.SPRING_ALGO, nc.SPIRAL_ALGO, nc.SHELL_ALGO])
        combo.current(0)
        label_algo.grid(row=(18 + initial_row), column=0, pady=5)
        combo.grid(row=(18 + initial_row), column=1, pady=5)

        # Generator
        label_gen= tk.Label(frame, text="Generator")
        combo_gen = ttk.Combobox(frame, name="gen", state="readonly",
                             values=["Default", "Dual"])
        combo_gen.current(0)
        label_gen.grid(row=(19 + initial_row), column=0, pady=5)
        combo_gen.grid(row=(19 + initial_row), column=1, pady=5)
        return frame, degree_list, type_list

    def create_tab_grouping(self, parent, frame_name):
        frame = ttk.Frame(parent, name=frame_name)
        combo = ttk.Combobox(frame, name="max_group", state="readonly",
                             values=["0.03", "0.05", "0.055", "0.06", "0.065", "0.07", "0.075",
                                     "0.08", "0.085", "0.09", "0.095", "0.1", "0.11",
                                     "0.12", "0.125", "0.13", "0.14", "0.15"])
        combo.current(0)
        combo.grid(row=0, column=0, pady=5)
        check = ttk.Checkbutton(frame, name="avoid_single", variable=self.remove_single_clusters)
        check.grid(row=0, column=1, pady=5)
        group_button = tk.Button(frame, text="Search for groups", command=self.group_graph)
        group_button.grid(row=0, column=2, pady=5)
        label_groups = tk.Label(frame,
                                text="",
                                name="print_groups", anchor="w")
        # label_printable.pack(side=tk.BOTTOM)
        label_groups.grid(row=1, column=0, columnspan=2, sticky=tk.S)

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
        else:
            self.back_gen = DualBackboneGenerator()
        old_canvas = None
        # Types and percentages for the nodes
        type_key_vals = type_list.get_entries()
        # Build the expected data structure
        types = pd.DataFrame({'code': [key for key, value in type_key_vals],
                              'proportion': [float(value) for key, value in type_key_vals]})
        # Find the old canvas for the image and destroy it
        for i in frame.winfo_children():
            if isinstance(i, tk.Canvas):
                old_canvas = i
                break
        old_canvas.destroy()

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
        self.topo, self.distances, self.assigned_types, \
            self.node_sheet, self.link_sheet, self.pos, self.colors = \
            self.back_gen.generate(degrees, weights, nodes, self.upper_limits, types, algorithm.get(),
                                   self.color_codes)

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
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)
        # Draw the figure
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)
        # Retrieve the reference to the label where distance ranges and proportions are drawn
        output_label = frame.nametowidget("print_distances")
        output_label['text'] = format_distance_limits(self.distances, self.upper_limits)
        # Call to the creation of the grouping/cluster graph with existing values
        self.group_graph()
        # Resize the whole window as the graph width changed
        # root_y = self.root.winfo_height()  # round(y_size*60)+output_label.winfo_height()
        # root_x = self.root.winfo_width() * size_ratio
        # print(root_x, ";", root_y)
        # self.root.geometry(f'{round(root_x)}x{round(root_y)}')

    # Regenerate the group graph
    def group_graph(self):
        # Find the combo that defines the eps
        combo = self.root.nametowidget("notebook_gen.group_tab.max_group")
        # Call the function that generates the tab
        groups, self.clusters = self.cluster_gen.find_groups(self.topo, self.assigned_types, self.pos,
                                            eps=float(combo.get()), avoid_single=self.remove_single_clusters.get())
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
        canvas_widget.grid(row=2, column=0, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)
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