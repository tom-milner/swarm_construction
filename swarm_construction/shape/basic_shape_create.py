from PIL import Image

width, height = 5, 5
shape = Image.new('1', (width, height))

# creates an L shaped image
# origin for pillow is in the top left :(
# so its an upside down L
# will probably need to be flipped in the sim

for x in range(width):
    for y in range(height):
        if (x == 0 or y == 0):
            shape.putpixel((x,y), 1)
        else:
            shape.putpixel((x,y), 0)

shape.save('test_shape.bmp')
