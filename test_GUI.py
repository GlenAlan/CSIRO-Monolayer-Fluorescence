import tkinter as tk
from tkinter import ttk
import random

# Initialize the main window
root = tk.Tk()
root.title("Rolling a die")

# Create a Notebook (tab container)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Create the first tab
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Tab 1")

# Add content to the first tab
label1 = tk.Label(tab1, text="This is Tab 1", font=("Arial", 16))
label1.pack(pady=20)

# Create the second tab
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Tab 2")

# Add content to the second tab
label2 = tk.Label(tab2, text="This is Tab 2", font=("Arial", 16))
label2.pack(pady=20)

def randNum():
    lbl_roll['text'] = random.randint(1,6)

root.columnconfigure(0, minsize=150)
root.rowconfigure([0, 1], minsize=50)

btn_roll = tk.Button(text = "Roll!",command=randNum)
btn_roll.grid(row=0, column=0, sticky="nsew")
lbl_roll = tk.Label()
lbl_roll.grid(row=1, column=0)

# Start the Tkinter event loop
root.mainloop()