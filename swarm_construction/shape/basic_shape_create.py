from PIL import Image

width, height = 5, 5
shape = Image.new('1', (width, height))

# creates a basic shape
# origin for pillow is in the top left :(
# gets flipped in main.py

for x in range(width):
    for y in range(height):
        if (x == 1 or y == 1):
            shape.putpixel((x,y), 1)
        else:
            shape.putpixel((x,y), 0)

shape.save('test_shape.bmp')
