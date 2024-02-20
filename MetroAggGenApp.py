import networkx as nx
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Texts_EN as texts
import networkconstants as nc
from MetroAggGenerator import MetroAggGenerator, DefaultMetroAggGenerator
from generator import write_network, format_node_list
import pandas as pd


# Class for the Tkinter Topology Generator Application
class MetroAggGenApp:
    metro_agg_gen: MetroAggGenerator

    # length_ranges - ranges of lengths for each hop in the aggregation ring
    # length_percentages - % of hops in the horseshoe within the related range
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, length_ranges, length_percentages, dict_colors={}):
        self.topo = None
        self.metro_agg_gen = DefaultMetroAggGenerator()
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
        self.root.title("MoleNetwork Metro Aggregation Topology Generator")
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
        result = False
        message = texts.FILE_NOT_FOUND
        if file_path:
            result, message = write_network(file_path, self.topo, self.distances, self.assigned_types, self.figure,
                                            pos=self.pos, reference_nodes=self.national_ref_nodes,
                                            type_nw=nc.METRO_AGGREGATION)
        if not result:
            tk.messagebox.showerror('', message)
        else:
            tk.messagebox.showinfo('', texts.COMPLETED)

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
        # Get the selected node from the combo
        selected_node_from_file = self.root.nametowidget("notebook_gen.source_frame.from_file.node")
        selected_node = selected_node_from_file.get()
        # Get the list of edges read from the file
        links = self.links_list
        # Select only those where the selected node is involved and pick the other end
        filtered = links[links["sourceID"] == selected_node]
        nodes = filtered["destinationID"]
        filtered = links[links["destinationID"] == selected_node]
        nodes = pd.concat([nodes, filtered["sourceID"]], ignore_index=True)
        # Fill the other combo with this values
        related_nodes = self.root.nametowidget("notebook_gen.source_frame.from_file.linked_nodes")
        related_nodes['state'] = "normal"
        vals = list(nodes)
        vals.insert(0, "-")
        related_nodes['values'] = vals
        related_nodes.set("-")

    def create_tab_image(self, parent):
        frame = ttk.Frame(parent, name="image_frame")

        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height), tight_layout=True, dpi=50)
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # Add the Tkinter canvas to the window
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)
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
            print("source or destination not selected")
            return

        hops = int(self.root.nametowidget("notebook_gen.source_frame.from_file.hops").get())
        prefix = nc.LOCAL_CO_CODE + "_" + source + "_" + destination + "_"

        (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
         self.national_ref_nodes) = \
            self.metro_agg_gen.metro_aggregation_horseshoe(source, 1, destination, hops,
                                                           self.lengths,
                                                           self.l_perc,
                                                           prefix, self.color_codes)

        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height), dpi=50)
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
