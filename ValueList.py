import tkinter as tk
from tkinter import messagebox


class ValueList:
    def __init__(self, root, value_name, initial_row, initial_value_list):
        self.root = root

        # Data structure to store values
        self.value_list = initial_value_list

        # GUI components
        self.entry_value = tk.Entry(root)

        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        for value in self.value_list:
            self.listbox.insert(tk.END, value)
        self.btn_add = tk.Button(root, text="Add", command=self.add_entry)
        self.btn_remove = tk.Button(root, text="Remove", command=self.remove_entry)
        self.label_values = tk.Label(root, text=value_name)

        # Layout
        self.label_values.grid(row=initial_row, column=0, pady=5)
        self.entry_value.grid(row=initial_row, column=1, pady=5)

        self.listbox.grid(row=initial_row, column=2, rowspan=7, columnspan=1, padx=10, pady=5, sticky="nsew")
        self.btn_add.grid(row=(1+initial_row), column=0, pady=5)
        self.btn_remove.grid(row=(1 + initial_row), column=1, pady=5)


        # Bind double click event to listbox
        self.listbox.bind("<Double-Button-1>", self.load_selected_entry)

        # Configure grid weights to make the listbox expandable
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    def add_entry(self):
        value = self.entry_value.get()

        if value:
            self.value_list.append(value)
            self.update_listbox()
            self.clear_entries()
        else:
            messagebox.showwarning("Input Error", "Please enter both key and value.")

    def remove_entry(self):
        selected_index = self.listbox.curselection()

        if selected_index:
            index = selected_index[0]
            del self.value_list[index]
            self.update_listbox()
            self.clear_entries()
        else:
            messagebox.showwarning("Selection Error", "Please select an entry to remove.")

    def load_selected_entry(self):
        selected_index = self.listbox.curselection()

        if selected_index:
            index = selected_index[0]
            key, value = self.value_list[index]
            self.entry_value.delete(0, tk.END)
            self.entry_value.insert(0, value)

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for value in self.value_list:
            self.listbox.insert(tk.END, value)

    def clear_entries(self):
        self.entry_value.delete(0, tk.END)

    def get_entries(self):
        return self.value_list
