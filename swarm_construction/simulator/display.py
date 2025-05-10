import pygame as pg
import numpy as np

# The simulation coordinate system.
class Display:
    def __init__(self, title, window_size, game_size):
        
        self.title = title
        self.window_size = window_size
        self.game_size = game_size

        pg.init()
        pg.display.set_caption(self.title)
        self.surface = pg.display.set_mode((self.window_size, self.window_size))


    def _px(self, pos):
        return np.astype(np.multiply(self.window_size / self.game_size, pos), int)

    def draw_line(self, colour, start, end, width=1):
        pg.draw.line(self.surface, colour, self._px(start), self._px(end), self._px(width))

    def draw_circle(self, colour, pos, radius):
        pg.draw.circle(self.surface, colour, self._px(pos), self._px(radius))

    def blit(self, surface, rect):
        # Might have an issue with scaling here
        self.surface.blit(surface, self._px(rect))
    
    def image_from_string(self, img, size, format):
        return pg.image.fromstring(img, size, format);

    def update(self):
        pg.display.update()
    
    def clear(self,colour):
        self.surface.fill(colour)