import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

root = tk.Tk()
root.title("Greyscale bitmap shape creation")

# canvas size
SHAPE_ROWS = 20
SHAPE_COLS = 30
shape_state = np.zeros((SHAPE_ROWS, SHAPE_COLS), dtype=int)


def toggle_pixel(event, row, col):
    """Changes the colour of a pixel on the canvas when clicked

    Args:
        event: tk event
        row (int): pixel row
        col (int): pixel col
    """
    # Get the clicked pixel
    pixel = event.widget
    # Toggle its state and background color
    # clicking rotates through colours
    if shape_state[row][col] == 127:
        shape_state[row][col] = 0
        pixel.config(bg="black")
    elif shape_state[row][col] == 0:
        shape_state[row][col] = 255
        pixel.config(bg="white")
    else:
        shape_state[row][col] = 127
        pixel.config(bg="grey")


def save_bitmap(shape_state):
    """saves the pixel canvas as a bitmap image

    Args:
        shape_state (2D array): array of pixel colours
    """
    # Find rows, cols where white or grey pixels exist
    rows = np.any(shape_state != 0, axis=1)
    cols = np.any(shape_state != 0, axis=0)

    # Get min and max index for rows, cols in 'bounding box'
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]

    # remove rows, cols outside 'bounding box' for slightly nicer bmps
    shape_state = shape_state[row_start : row_end + 1, col_start : col_end + 1]
    crop_rows, crop_cols = shape_state.shape

    # Create a new image (8-bit greyscale mode)
    shape = Image.new("L", (crop_cols, crop_rows))
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
