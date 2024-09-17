import tkinter as tk
import Texts_EN as tx
from KeyValueList import KeyValueList


class PercentageConfigWindow:
    def __init__(self, generator, root, upper_limits, proportions, text_limit, text_prop):
        self.generator = generator
        self.window = None
        self.root = root
        self.distance_list = None
        self.id_text = text_limit
        self.create_window(upper_limits, proportions, text_limit, text_prop)

    def create_window(self, upper_limits, proportions, text_limit, text_prop):
        self.window = tk.Toplevel(self.root, name="top_distance")
        self.window.title("Set Values and Proportions")
        self.window.geometry("600x300")

        self.distance_list = KeyValueList(self.window, text_limit, text_prop, 1,
                                          [(str(key), str(value)) for key, value in zip(upper_limits, proportions)],
                                          sortme=True)
        btn_save = tk.Button(self.window, text="Save", command=self.save_and_close)
        btn_save.grid(row=7, column=1)

    def save_and_close(self):
        items = self.distance_list.get_entries()
        if len(items) == 0:
            tk.messagebox.showerror('', tx.EMPTY_LIST)
            return
        distances = [int(key) for key, value in items]
        proportions = [float(value) for key, value in items]
        self.generator.set_upper_limits(distances, proportions, self.id_text)
        self.window.destroy()

    def show(self, upper_limits, proportions, text_limit, text_prop):
        self.id_text = text_limit
        self.create_window(upper_limits, proportions, text_limit, text_prop)
