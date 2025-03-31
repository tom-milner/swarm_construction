from PIL import Image

width, height = 10, 10
shape = Image.new("1", (width, height))

for x in range(width):
    for y in range(height):
        if x == 0 or y == 0:
            shape.putpixel((x, y), 1)
        else:
            shape.putpixel((x, y), 0)

shape.save("test_shape.bmp")
