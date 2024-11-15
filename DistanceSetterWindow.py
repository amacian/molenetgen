import tkinter as tk
import Texts_EN as tx
from KeyValueList import KeyValueList


class DistanceSetterWindow:
    def __init__(self, generator, root, upper_limits, proportions, max_distance, iterations_distance, bound):
        self.generator = generator
        self.window = None
        self.root = root
        self.distance_list = None
        self.iteration_var = None
        self.bound_var = None
        self.create_window(upper_limits, proportions, max_distance, iterations_distance, bound)

    def create_window(self, upper_limits, proportions, max_distance, iterations_distance, bound):
        self.window = tk.Toplevel(self.root, name="top_distance")
        self.window.title("Set Upper Limits for Distance Ranges")
        self.window.geometry("600x300")

        self.distance_list = KeyValueList(self.window, "Ranges (upper limit)", "Proportions", 1,
                                          [(str(key), str(value)) for key, value in zip(upper_limits, proportions)],
                                          sortme=True)
        label_iterations = tk.Label(self.window, text="Iterations to run to fit a topology")
        label_iterations.grid(row=5, columnspan=2)
        self.iteration_var = tk.StringVar()
        self.iteration_var.set(iterations_distance)
        entry_iter = tk.Entry(self.window, name="iterations_set", textvariable=self.iteration_var)
        entry_iter.grid(row=6, columnspan=2)
        self.bound_var = tk.StringVar()
        self.bound_var.set(bound)
        label_bound = tk.Label(self.window, text="Limit for moving nodes in optimization")
        label_bound.grid(row=7, columnspan=2)
        entry_bound = tk.Entry(self.window, name="bound", textvariable=self.bound_var)
        entry_bound.grid(row=8, columnspan=2)
        btn_save = tk.Button(self.window, text="Save", command=self.save_and_close)
        btn_save.grid(row=9, column=1)

    def save_and_close(self):
        items = self.distance_list.get_entries()
        iterations = int(self.iteration_var.get())
        bound = float(self.bound_var.get())
        if len(items) == 0:
            tk.messagebox.showerror('', tx.EMPTY_LIST)
            return
        distances = [int(key) for key, value in items]
        proportions = [float(value) for key, value in items]
        self.generator.set_upper_limits(distances, proportions, max(distances), iterations, bound)
        self.window.destroy()

    def show(self, upper_limits, proportions, max_distance, iterations, bound):
        self.create_window(upper_limits, proportions, max_distance, iterations, bound)
