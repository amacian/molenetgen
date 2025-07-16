import copy

import networkx as nx
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Texts_EN as texts
import networkconstants as nc
from BulkMetroAggGeneratorService import DefaultBulkMetroAggGenService
from MetroAggGenerator import MetroAggGenerator, DefaultMetroAggGenerator
from PercentageConfigWindow import PercentageConfigWindow
from generator import write_network, format_node_list
import pandas as pd


# Class for the Tkinter Topology Generator Application
class MetroAggGenApp:
    metro_agg_gen: MetroAggGenerator

    # length_ranges - ranges of lengths for each hop in the aggregation ring
    # length_percentages - % of hops in the horseshoe within the related range
    # dict_colors - dictionary that maps types to colors for the basic graph
    def __init__(self, length_ranges, length_percentages, hop_number, hop_percents, dict_colors={},
                 limits_link_length=[0, 1000]):
        self.topo = None
        self.metro_agg_gen = DefaultMetroAggGenerator()
        self.lengths = length_ranges
        self.l_perc = length_percentages
        self.hops = hop_number
        self.h_perc = hop_percents
        # Color codes depending on the type
        self.color_codes = dict_colors
        self.national_ref_nodes = None
        self.nodes_list = ["-"]
        self.links_list = None
        self.limits_link_length = limits_link_length
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
        # Configuration window for percentages and values
        self.setter = None
        # Boolean variable to indicate if the horseshoes generated through Bulk generation
        # should only consider linked nodes as ends
        self.linked_only = tk.BooleanVar(self.root)

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
        subframe_bulk = ttk.Frame(frame, name="create_bulk")

        label = tk.Label(subframe_f, name="select", text="Select file by clicking the button", fg="orange")
        label.grid(row=0, column=0, columnspan=4)
        load_button = tk.Button(subframe_f, text="Load from File", command=self.load_from_file)
        load_button.grid(row=1, column=0, columnspan=4)

        label_nodes = tk.Label(subframe_f, name="node_list", text="Loaded nodes: None")
        label_nodes.grid(row=2, column=0, columnspan=4)

        # Separator for Subframe. Creating individual horseshoes.
        separator3 = ttk.Separator(subframe_f, orient="horizontal")
        separator3.grid(row=3, columnspan=4, ipadx=300, pady=10)
        # Clusters
        label_node_intro = tk.Label(subframe_f,
                                    text="Individual Horseshoe creation",
                                    fg="orange")
        label_node_intro.grid(row=4, columnspan=2)
        label_node_select = tk.Label(subframe_f,
                                     text="Select Source and Destination Nodes")
        label_node_select.grid(row=5, column=0)

        combo = ttk.Combobox(subframe_f, name="node", state="readonly",
                             values=["-"])
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", self.node_selected)
        combo.grid(row=5, column=1)

        linked_node_combo = ttk.Combobox(subframe_f, name="linked_nodes", state="disabled",
                                         values=["-"])
        linked_node_combo.grid(row=6, column=1)

        label_hops_c = tk.Label(subframe_f, text="Select Number of Hops in the Horseshoe")
        label_hops_c.grid(row=7, column=0)
        hops = ttk.Combobox(subframe_f, name="hops", state="enabled",
                            values=self.hops)
        hops.current(0)
        hops.grid(row=7, column=1)

        label_length = tk.Label(subframe_f, text="Select the length range of the Horseshoe")
        label_length.grid(row=11, column=0)
        lengths = ttk.Combobox(subframe_f, name="lengths", state="enabled",
                               values=self.compose_range()
                               )
        lengths.current(0)
        lengths.grid(row=11, column=1)

        label_bulk = tk.Label(subframe_bulk,
                              text="Bulk Horseshoe creation", fg="orange")
        label_bulk.grid(row=1, column=0, columnspan=4)

        label_lengths = tk.Label(subframe_bulk, name="label_length",
                                 text=self.compose_lengths())
        label_lengths.grid(row=2, column=0)
        btn_set_lengths = tk.Button(subframe_bulk, text="Change \nlengths", command=self.open_config_window_length)
        btn_set_lengths.grid(row=2, column=2, sticky=tk.N)

        label_hops = tk.Label(subframe_bulk, name="label_hops",
                              text=self.compose_hops())
        label_hops.grid(row=3, column=0)
        btn_set_hops = tk.Button(subframe_bulk, text="Change \nhops", command=self.open_config_window_hops)
        btn_set_hops.grid(row=3, column=2, sticky=tk.N)
        # run_button = tk.Button(frame, text="Load from File", command=self.load_from_file)
        label_horseshoes = tk.Label(subframe_bulk,
                                    text="Number of horseshoes to create:")
        label_horseshoes.grid(row=4, column=0)
        entry_num = tk.Entry(subframe_bulk, name="num_horseshoes")
        entry_num.grid(row=4, column=2)
        label_only_linked = tk.Label(subframe_bulk,
                                     text="Use only linked nodes as ends:")
        label_only_linked.grid(row=5, column=0)
        check = ttk.Checkbutton(subframe_bulk, name="linked_only", variable=self.linked_only)
        check.grid(row=5, column=1)
        btn_bulk_gen = tk.Button(subframe_bulk, text="Generate", command=self.bulkme)
        btn_bulk_gen.grid(row=5, column=2, sticky=tk.N)

        subframe_f.grid(row=3, column=0, columnspan=4)
        separator2 = ttk.Separator(frame, orient="horizontal")
        separator2.grid(row=4, columnspan=4, ipadx=300, pady=10)
        subframe_bulk.grid(row=5, column=0, columnspan=4)
        return frame

    def compose_range(self):
        ranges = []
        for i in range(len(self.lengths) - 1):
            ranges.append("[" + str(self.lengths[i]) + "-" + str(self.lengths[i + 1]) + "]")
        return ranges

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

        # Select position in combo length to extract ranges from the self.length array
        pos_length = self.root.nametowidget("notebook_gen.source_frame.from_file.lengths").current()
        # Extract max and use it as the only entry so it is always selected
        minval = self.lengths[pos_length]
        maxval = self.lengths[pos_length + 1]

        # (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
        #  self.national_ref_nodes) = \
        #     self.metro_agg_gen.metro_aggregation_horseshoe(source, 1, destination, hops,
        #                                                    self.lengths,
        #                                                    self.l_perc,
        #                                                    prefix, self.color_codes)

        (self.topo, self.distances, self.assigned_types, self.pos, self.colors,
         self.national_ref_nodes) = \
            self.metro_agg_gen.metro_aggregation_horseshoe(source, 1, destination, hops,
                                                           [minval, maxval],
                                                           [1],
                                                           prefix, self.color_codes)

        self.figure = plt.Figure(figsize=(self.fig_width, self.fig_height), dpi=50)
        ax = self.figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.figure, master=frame)
        canvas_widget = canvas.get_tk_widget()

        # Add the Tkinter canvas to the window
        canvas_widget.grid(row=0, column=0, sticky=tk.W + tk.N)

        # Draw the figure
        #nx.draw(self.topo, pos=self.pos, with_labels=True, font_weight='bold',
        #        node_color=self.colors, ax=ax)
        nx.draw_networkx_nodes(self.topo, pos=self.pos, node_color=self.colors,
                               node_size=300, edgecolors='black', linewidths=0.5, ax=ax)
        nx.draw_networkx_edges(self.topo, pos=self.pos, ax=ax, width=1.0, alpha=0.6)

        label_pos = copy.deepcopy(self.pos)
        y_offset = 0.01  # Tune this factor if needed
        for k in label_pos:
            label_pos[k] = (label_pos[k][0], label_pos[k][1] + y_offset)

        # Draw labels above nodes
        nx.draw_networkx_labels(self.topo, pos=label_pos, ax=ax,
                                font_size=14, font_weight='bold')
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
            nodes_df = pd.read_excel(file_path, sheet_name=nc.NODES_EXCEL_NAME)
            links_df = pd.read_excel(file_path, sheet_name=nc.LINKS_EXCEL_NAME)
            self.nodes_list = nodes_df
            self.links_list = links_df
            label = self.root.nametowidget("notebook_gen.source_frame.from_file.node_list")
            node_names = list(nodes_df['node_name'])
            # label['text'] = format_node_list(node_names)
            label['text'] = ("Loaded " + str(len(node_names)) + " nodes: \n" + node_names[0]
                             + "... (see the combos below)")
            combo_node = self.root.nametowidget("notebook_gen.source_frame.from_file.node")
            node_names.sort()
            combo_node["values"] = node_names

    def compose_lengths(self):
        result = "Length ranges : proportion over " + str(sum(self.l_perc)) + "\n"
        ranges = self.compose_range()
        for i in range(len(ranges)):
            result += ranges[i] + " : " + str(self.l_perc[i]) + "\n"
        return result

    def compose_hops(self):
        result = "Number of hops in horseshoe : proportion over " + str(sum(self.h_perc)) + "\n"
        for i in range(len(self.hops)):
            result += str(self.hops[i]) + " : " + str(self.h_perc[i]) + "\n"
        return result

    # Open the window that will modify the length ranges.
    def open_config_window_length(self):
        if self.setter is None:
            # self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.max_upper)
            self.setter = PercentageConfigWindow(self, self.root, self.lengths[1:], self.l_perc, "Lengths",
                                                 "Proportions")
        else:
            self.setter.show(self.lengths[1:], self.l_perc, "Lengths", "Proportions")

    # Open the window that will modify the length ranges.
    def open_config_window_hops(self):
        if self.setter is None:
            # self.setter = DistanceSetterWindow(self, self.root, self.upper_limits, self.max_upper)
            self.setter = PercentageConfigWindow(self, self.root, self.hops, self.h_perc, "Hops", "Proportions")
        else:
            self.setter.show(self.hops, self.h_perc, "Hops", "Proportions")

    # Callback function for the Setter window
    def set_upper_limits(self, new_vals, new_percs, id_text):
        if id_text == "Lengths":
            new_vals.insert(0, 0)
            self.lengths = new_vals
            self.l_perc = new_percs
            lengths_combo = self.root.nametowidget("notebook_gen.source_frame.from_file.lengths")
            lengths_combo['values'] = self.compose_range()
            lengths_label = self.root.nametowidget("notebook_gen.source_frame.create_bulk.label_length")
            lengths_label['text'] = self.compose_lengths()
        elif id_text == "Hops":
            self.hops = new_vals
            self.h_perc = new_percs
            hops_combo = self.root.nametowidget("notebook_gen.source_frame.from_file.hops")
            hops_combo['values'] = self.hops
            hops_label = self.root.nametowidget("notebook_gen.source_frame.create_bulk.label_hops")
            hops_label['text'] = self.compose_hops()

    def bulkme(self):
        # There must be at list 2 nodes
        if len(self.nodes_list) < 2:
            return
        # Get the number of horseshoes to create
        try:
            n_horse = int(self.root.nametowidget("notebook_gen.source_frame.create_bulk.num_horseshoes").get())
        except ValueError:
            n_horse = 1

        # Create a bulk generator
        bulk_gen = DefaultBulkMetroAggGenService(DefaultMetroAggGenerator())

        # Remove all elements of the not allowed types.
        query_out = nc.XLS_CO_TYPE + " not in @nc.AGG_TYPES_EXCLUDED"
        reduced_list = self.nodes_list.query(query_out)
        # Get the nodes with not allowed types to remove the correspondent links
        query_in = nc.XLS_CO_TYPE + " in @nc.AGG_TYPES_EXCLUDED"
        not_allowed = self.nodes_list.query(query_in)
        # Remove those links with nodes of the not allowed types
        reduced_links = self.links_list[~self.links_list[nc.XLS_SOURCE_ID].isin(not_allowed[nc.XLS_NODE_NAME])]
        reduced_links = reduced_links[~reduced_links[nc.XLS_DEST_ID].isin(not_allowed[nc.XLS_NODE_NAME])]

        # Ask for the file path to store the generated horseshoes
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"),
                                                            ("All files", "*.*")])
        if not file_path:
            tk.messagebox.showerror('', texts.FILE_NOT_FOUND)
            return

        # Generate the horseshoes
        (result, message) = bulk_gen.bulk_metro_aggregation(reduced_list, reduced_links, self.lengths,
                                                            self.l_perc, self.hops, self.h_perc,
                                                            self.linked_only.get(), n_horse, self.color_codes,
                                                            file_path, limit_lengths_link=self.limits_link_length,
                                                            prefix=nc.LOCAL_CO_CODE)

        if not result:
            tk.messagebox.showerror('', message)
