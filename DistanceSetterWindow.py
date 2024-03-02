import tkinter as tk
import Texts_EN as tx


class DistanceSetterWindow():
    def __init__(self, generator, root, upper_limits, max_distance):
        self.generator = generator
        self.window = None
        self.root = root
        self.list_distances = None
        self.entry_distance = None
        self.entry_max = None
        self.var_max = None
        self.create_window(upper_limits, max_distance)

        print(self.var_max.get())

    def create_window(self, upper_limits, max_distance):
        self.window = tk.Toplevel(self.root, name="top_distance")
        self.window.title("Set Upper Limits for Distance Ranges")
        self.window.geometry("400x300")
        entry_label = tk.Label(self.window, text="Insert limit between ranges")
        entry_label.grid(row=0, column=0)
        label_existing = tk.Label(self.window, text="Current limits between ranges")
        label_existing.grid(row=0, column=1)
        self.entry_distance = tk.Entry(self.window, name="entry_dist")
        self.entry_distance.grid(row=1, column=0, sticky=tk.N)
        self.list_distances = tk.Listbox(self.window, selectmode=tk.SINGLE)
        for limit in upper_limits:
            self.list_distances.insert(tk.END, limit)
        self.list_distances.grid(row=1, column=1)
        btn_add = tk.Button(self.window, text="Add", command=self.add_element)
        btn_add.grid(row=2, column=0)
        btn_remove = tk.Button(self.window, text="Remove", command=self.del_element)
        btn_remove.grid(row=2, column=1)
        self.var_max = tk.StringVar()
        self.var_max.set(max_distance)
        self.entry_max = tk.Entry(self.window, name="entry_max", textvariable=self.var_max)
        self.entry_max.grid(row=3, column=0, sticky=tk.N)
        btn_save = tk.Button(self.window, text="Save", command=self.save_and_close)
        btn_save.grid(row=4, column=1)

    def save_and_close(self):
        items = self.list_distances.get(0, tk.END)
        max_distance = int(self.var_max.get())
        if len(items) == 0:
            tk.messagebox.showerror('', tx.EMPTY_LIST)
            return

        self.generator.set_upper_limits(items, max_distance)

    def add_element(self):
        items = self.list_distances.get(0, tk.END)
        new_value = self.entry_distance.get()
        if not str.isdigit(new_value):
            return
        new_value = int(new_value)
        if new_value in items:
            return
        for idx in range(len(items)):
            if new_value < items[idx]:
                self.list_distances.insert(idx, new_value)
                return
        self.list_distances.insert(tk.END, new_value)
        return

    def del_element(self):
        selected_index = self.list_distances.curselection()

        if selected_index:
            index = selected_index[0]
            value = self.list_distances.delete(index)
        return

    def show(self, upper_limits):
        self.create_window(upper_limits, max_distance)
