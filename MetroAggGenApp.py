import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkconstants
from generator import write_backbone, format_node_list, \
    metro_aggregation_horseshoe
import pandas as pd


# Class for the Tkinter Topology Generator Application
class MetroAggGenApp:

    # length_ranges - ranges of lengths for each hop in the aggregation ring
    # length_percentages - % of hops in the horseshoe within the related range
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, length_ranges, length_percentages, dict_colors={}):

        self.lengths = length_ranges
        self.l_perc = length_percentages
        # Color codes depending on the type
        self.color_codes = dict_colors
        self.national_ref_nodes = None
        self.nodes_list = ["-"]
        self.links_list = None
        # Variable to hold the default figure
        self.figure = None
        # Root component
        self.root = tk.Tk()
        self.root.geometry("800x800")
        self.root.title("Metro Aggregation Topology Generator")
        # Value that holds if the single clusters should be removed
        self.remove_single_clusters = tk.BooleanVar(self.root)
        # Color codes depending on the type
        self.color_codes = dict_colors
        # Actual colors per node
        self.colors = []

        # Default figure height and width
        self.fig_height = 7
        self.fig_width = 14

        # Create a notebook for the tabs
        notebook = ttk.Notebook(self.root, name="notebook_gen")

        # Create tabs and add them to the notebook
        # The second tab permits selection between set of nodes or reading from file.
        tab1 = self.create_tab_source(notebook)
        notebook.add(tab1, text="Source file")
        # First tab corresponds to the list of parameters
        # Returns a reference to the tab, another to the list of degree restrictions and
        # another to the type restrictions
        # tab1, degree_list, type_list = self.create_tab_list(notebook, "Degrees", "Weights", "Number of nodes",
        #                                                    "params", "Types", "Weights", degrees, weights, types,
        #                                                    nodes)
        # notebook.add(tab1, text="Metro mesh constraints")

        # The third tab creates the graph and returns a pointer to the canvas
        # tab3, canvas = self.create_tab_image(notebook)
        tab3 = self.create_tab_image(notebook)
        notebook.add(tab3, text="Graph")

        # The third tab corresponds to the creation of the clusters
        # and the image representing them
        # tab3 = self.create_tab_grouping(notebook, "group_tab")
        # notebook.add(tab3, text="Grouping")

        # The fourth tab is the one used to save all the information into an Excel file
        tab4 = self.create_tab_save(notebook)
        notebook.add(tab4, text="Save")

        # Pack the notebook into the main window
        notebook.pack(expand=True, fill=tk.BOTH)

        run_button = tk.Button(self.root, text="Run",
                               command=lambda: self.metro_aggregation(tab3))
        run_button.pack(side=tk.BOTTOM)

        # Start the Tkinter event loop
        self.root.mainloop()

    # Save the information to file
    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if file_path:
            write_backbone(file_path, self.topo, self.distances, self.assigned_types, self.figure,
                           pos=self.pos, reference_nodes=self.national_ref_nodes)

    def create_tab_save(self, parent):
        frame = ttk.Frame(parent)
        save_button = tk.Button(frame, text="Save to File", command=self.save_to_file)
        save_button.pack(pady=10)
        return frame

    def create_tab_source(self, parent):
        frame = ttk.Frame(parent, name="source_frame")

        subframe_f = ttk.Frame(frame, name="from_file")
        label = tk.Label(subframe_f, name="select", text="Select file by clicking the button.")
        label.grid(row=0, column=0)
        load_button = tk.Button(subframe_f, text="Load from File", command=self.load_from_file)
        load_button.grid(row=1, column=0)

        label_nodes = tk.Label(subframe_f, name="node_list", text="Loaded nodes: None")
        label_nodes.grid(row=2, column=0)

        separator2 = ttk.Separator(frame, orient="horizontal")
        separator2.grid(row=3, columnspan=5, ipadx=300, pady=10)

        # Clusters
        label_node_select = tk.Label(subframe_f, text="Select Source and Destination Nodes")
        label_node_select.grid(row=4, columnspan=5)
        combo = ttk.Combobox(subframe_f, name="node", state="readonly",
                             values=["-"])
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", self.node_selected)
        combo.grid(row=5, column=0)

        linked_node_combo = ttk.Combobox(subframe_f, name="linked_nodes", state="disabled",
                                         values=["-"])
        linked_node_combo.grid(row=8, column=0)

        label_hops = tk.Label(subframe_f, text="Select Number of Hops in the Horseshoe")
        label_hops.grid(row=9, columnspan=5)
        hops = ttk.Combobox(subframe_f, name="hops", state="enabled",
                            values=["2", "3", "4", "5", "6", "7", "8"],
                            )
        hops.current(0)
        hops.grid(row=10, column=0)

        # run_button = tk.Button(frame, text="Load from File", command=self.load_from_file)
        subframe_f.grid(row=1, column=0, columnspan=3)
        return frame

    # Action to be run whenever a cluster is selected to generate a topology
    def node_selected(self, event):
        selected_node_from_file = self.root.nametowidget("notebook_gen.source_frame.from_file.node")
        selected_node = selected_node_from_file.get()
        print(selected_node)
        links = self.links_list
        filtered = links[links["sourceID"] == selected_node]
        nodes = filtered["destinationID"]
        filtered = links[links["destinationID"] == selected_node]
        nodes = pd.concat([nodes, filtered["sourceID"]], ignore_index=True)
        related_nodes = self.root.nametowidget("notebook_gen.source_frame.from_file.linked_nodes")
        related_nodes['state'] = "normal"
        related_nodes['values'] = list(nodes)

    def create_tab_image(self, parent):
        frame = ttk.Frame(parent, name="image_frame")

        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height), tight_layout=True, dpi=50)
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # Add the Tkinter canvas to the window
        # canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)
        # print(self.colors)
        '''nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)

        label_printable = tk.Label(frame,
                                   text=format_distance_limits(self.distances, self.upper_limits),
                                   name="print_distances", anchor="w")
        # label_printable.pack(side=tk.BOTTOM)
        label_printable.grid(row=2, column=0, sticky=tk.S)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)'''
        # return frame, canvas
        return frame

    def metro_aggregation(self, frame):
        old_canvas = None
        # Find the old canvas for the image and destroy it
        for i in frame.winfo_children():
            if isinstance(i, tk.Canvas):
                old_canvas = i
                break
        if old_canvas is not None:
            old_canvas.destroy()

        source_combo = self.root.nametowidget("notebook_gen.source_frame.from_file.node")
        source = source_combo.get()

        destination_combo = self.root.nametowidget("notebook_gen.source_frame.from_file.linked_nodes")
        destination = destination_combo.get()

        if source == "-" or source == "" or destination == "-" or destination == "":
            print ("source or destination not selected")
            return

        hops = int(self.root.nametowidget("notebook_gen.source_frame.from_file.hops").get())
        prefix = networkconstants.LOCAL_CO_CODE + "_" + source + "_" + destination + "_"

        (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
         self.national_ref_nodes) = \
            metro_aggregation_horseshoe(source, 1, destination, hops,
                                        self.lengths,
                                        self.l_perc,
                                        prefix, self.color_codes)

        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height), dpi=50)
        print(self.fig_width, self.fig_height)
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # Add the Tkinter canvas to the window
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)

        # Draw the figure
        nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
                node_color=self.colors, ax=ax)

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

    def load_from_file(self):
        file_path = filedialog.askopenfilename(title="Open Topology", defaultextension=".xlsx",
                                               filetypes=[("Excel files", ".xlsx .xls")])
        if file_path:
            nodes_df = pd.read_excel(file_path, sheet_name="nodes")
            links_df = pd.read_excel(file_path, sheet_name="links")
            self.nodes_list = nodes_df
            self.links_list = links_df
            label = self.root.nametowidget("notebook_gen.source_frame.from_file.node_list")
            node_names = list(nodes_df['node_name'])
            label['text'] = format_node_list(node_names)
            combo_node = self.root.nametowidget("notebook_gen.source_frame.from_file.node")
            node_names.sort()
            combo_node["values"] = node_names
            print(node_names)