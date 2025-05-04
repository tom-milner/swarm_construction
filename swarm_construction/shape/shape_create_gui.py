import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

root = tk.Tk()
root.title("Bitmap shape creation")

# canvas size
SHAPE_ROWS = 20
SHAPE_COLS = 30
shape_state = np.zeros((SHAPE_ROWS, SHAPE_COLS), dtype=int)


def toggle_pixel(event, row, col):
    # Get the clicked pixel
    pixel = event.widget
    # Toggle its state and background color
    if shape_state[row][col] == 1:
        shape_state[row][col] = 0
        pixel.config(bg="black")
    else:
        shape_state[row][col] = 1
        pixel.config(bg="white")


def save_bitmap(shape_state):
    # Crops any rows and columns that only include black pixels
    shape_state = shape_state[~np.all(shape_state == 0, axis=1)]
    shape_state = shape_state[:, ~np.all(shape_state == 0, axis=0)]
    crop_rows, crop_cols = shape_state.shape

    # Create a new image (1-bit mode)
    shape = Image.new("", (crop_cols, crop_rows))
    # Populate the image with pixel data from grid_state
    for row in range(crop_rows):
        for col in range(crop_cols):
            shape.putpixel((col, row), int(shape_state[row][col]))
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".bmp", filetypes=[("Bitmap files", "*.bmp")]
    )
    if file_path:
        shape.save(file_path)
        print(f"Bitmap saved to {file_path}")


# Create the grid of pixels
for row in range(SHAPE_ROWS):
    for col in range(SHAPE_COLS):
        # Create a pixel as a label
        pixel = tk.Label(root, bg="black", width=4, height=2)
        pixel.grid(row=row, column=col, padx=0, pady=0)
        # Bind click event with its row and column indices
        pixel.bind("<Button-1>", lambda event, r=row, c=col: toggle_pixel(event, r, c))

# Add a "Save" button
save_button = tk.Button(
    root, text="Save Bitmap", command=lambda: save_bitmap(shape_state)
)
save_button.grid(row=SHAPE_ROWS, column=0, columnspan=SHAPE_COLS, pady=10)

# Start the main loop
root.mainloop()
