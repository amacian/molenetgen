import tkinter as tk
from tkinter import messagebox


class KeyValueList:
    def __init__(self, root, key_name, value_name, initial_row, initial_value_list, sortme=False):
        self.root = root
        self.sort = sortme
        # Data structure to store key-value pairs
        self.key_value_list = initial_value_list

        # GUI components
        self.entry_key = tk.Entry(root)
        self.entry_value = tk.Entry(root)
        # self.entry_node = tk.Entry(root, name="entry_node")
        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        for key, value in self.key_value_list:
            self.listbox.insert(tk.END, f"{key}: {value}")
        self.btn_add = tk.Button(root, text="Add", command=self.add_entry)
        self.btn_remove = tk.Button(root, text="Remove", command=self.remove_entry)
        self.btn_update = tk.Button(root, text="Update", command=self.update_entry)
        self.label_key = tk.Label(root, text=key_name)
        self.label_values = tk.Label(root, text=value_name)
        # self.label_nodes = tk.Label(root, text=node_name)

        # Layout
        self.label_key.grid(row=initial_row, column=0, padx=10, pady=10, sticky="n")
        self.entry_key.grid(row=initial_row, column=1, padx=10, pady=10, sticky="n")
        self.label_values.grid(row=(initial_row+1), column=0, pady=5, sticky="n")
        self.entry_value.grid(row=(initial_row+1), column=1, pady=5, sticky="n")

        self.listbox.grid(row=(0+initial_row), column=2, rowspan=7, columnspan=1, padx=10, pady=5, sticky="nsew")
        self.btn_add.grid(row=(2+initial_row), column=0, pady=5)
        self.btn_update.grid(row=(2 + initial_row), column=1, pady=5)
        self.btn_remove.grid(row=(3+initial_row), column=1, pady=5)


        # Bind double click event to listbox
        self.listbox.bind("<Double-Button-1>", self.load_selected_entry)

        # Configure grid weights to make the listbox expandable
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    def add_entry(self):
        key = self.entry_key.get()
        value = self.entry_value.get()

        if key and value:
            self.key_value_list.append((key, value))
            self.update_listbox()
            self.clear_entries()
        else:
            messagebox.showwarning("Input Error", "Please enter both key and value.")

    def remove_entry(self):
        selected_index = self.listbox.curselection()

        if selected_index:
            index = selected_index[0]
            del self.key_value_list[index]
            self.update_listbox()
            self.clear_entries()
        else:
            messagebox.showwarning("Selection Error", "Please select an entry to remove.")

    def update_entry(self):
        selected_index = self.listbox.curselection()

        if selected_index:
            index = selected_index[0]
            key = self.entry_key.get()
            value = self.entry_value.get()

            if key and value:
                self.key_value_list[index] = (key, value)
                self.update_listbox()
                self.clear_entries()
            else:
                messagebox.showwarning("Input Error", "Please enter both key and value.")
        else:
            messagebox.showwarning("Selection Error", "Please select an entry to update.")

    def load_selected_entry(self, event):
        selected_index = self.listbox.curselection()

        if selected_index:
            index = selected_index[0]
            key, value = self.key_value_list[index]
            self.entry_key.delete(0, tk.END)
            self.entry_key.insert(0, key)
            self.entry_value.delete(0, tk.END)
            self.entry_value.insert(0, value)

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        if self.sort:
            self.sort_entries()
        for key, value in self.key_value_list:
            self.listbox.insert(tk.END, f"{key}: {value}")

    def clear_entries(self):
        self.entry_key.delete(0, tk.END)
        self.entry_value.delete(0, tk.END)

    def get_entries(self):
        return self.key_value_list

    def sort_entries(self):
        print(self.key_value_list)
        print(self.key_value_list.sort(key=lambda x: int(x[0])))