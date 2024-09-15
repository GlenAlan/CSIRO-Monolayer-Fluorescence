import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import random
import string

class CustomObject:
    def __init__(self, name, description, image_path):
        self.name = name
        self.description = description
        self.image_path = image_path
        self.goto = self.description[random.randint(0, len(self.description) - 1)]

# Sample data: list of objects

objects = []
for i in range(0, 100):
    r = random.randint(1, 3)
    objects.append(CustomObject(f"Object {i}", ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=20)), f"Images/test_image{r}.jpg"))

def on_item_selected(event):
    selected_item = tree.selection()[0]
    obj_index = tree.item(selected_item, 'values')[2]  # Retrieve the object index
    obj = objects[int(obj_index)]  # Use the index to get the object
    print(f"Selected: {obj.name} - {obj.goto}")

root = tk.Tk()
root.title("Table with Images and Data")
root.geometry("800x400")  # Increase window size for better visibility

# Set a style for the Treeview to adjust row height
style = ttk.Style()
style.configure("Treeview", rowheight=150)  # Set row height to fit images

# Frame to hold the treeview and scrollbar
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Create Treeview with columns
columns = ('#1', '#2')  # Define columns for text data
tree = ttk.Treeview(frame, columns=columns, show='tree headings')  # Show the tree column and headings

# Define headings
tree.heading('#0', text='Image')  # First column (tree column) for images
tree.heading('#1', text='Name')
tree.heading('#2', text='Description')

# Adjust the width of the columns
tree.column('#0', width=160, anchor='center')  # Adjust the first column to fit larger images
tree.column('#1', width=200, anchor='center')  # Adjust name column width
tree.column('#2', width=350, anchor='center')  # Adjust description column width

# Create a vertical scrollbar
vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=vsb.set)

# Use a list to store image references to prevent garbage collection
image_references = []

# Insert data into the treeview
for index, obj in enumerate(objects):
    try:
        # Load image
        if os.path.exists(obj.image_path):
            img = Image.open(obj.image_path)
            img = img.resize((140, 140))  # Resize image to fit the larger cell
            photo = ImageTk.PhotoImage(img)

            # Store reference to image to prevent garbage collection
            image_references.append(photo)

            # Insert item into Treeview (add image as an icon in the first column)
            tree.insert("", "end", image=photo, values=(obj.name, obj.description, index))  # Store the index
        else:
            print(f"Image not found: {obj.image_path}")
    except Exception as e:
        print(f"Error loading image: {obj.image_path}, {e}")

# Add treeview and scrollbar to the frame
tree.grid(row=0, column=0, sticky='nsew')
vsb.grid(row=0, column=1, sticky='ns')

# Configure frame to expand treeview
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Bind the item selection event to a callback function
tree.bind('<<TreeviewSelect>>', on_item_selected)

root.mainloop()
