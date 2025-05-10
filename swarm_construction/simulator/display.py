import pygame as pg
import numpy as np


# The simulation coordinate system.
class Display:
    def __init__(self, title, window_size, game_size, font_size):

        self.title = title
        self.window_size = window_size
        self._sf = window_size / game_size

        pg.init()
        pg.display.set_caption(self.title)
        self._surface = pg.display.set_mode((self.window_size, self.window_size))

        # The font to use for writing the labels.
        self._font = pg.font.SysFont("Arial", (font_size))  # Font size = agent radius

    def _px(self, pos):
        return np.astype(np.round(np.multiply(self._sf, pos)),int)

    def draw_line(self, colour, start, end, width=1):
        pg.draw.line(
            self._surface, colour, self._px(start), self._px(end), width)


    def draw_circle(self, colour, pos, radius):
        pg.draw.circle(self._surface, colour, self._px(pos), self._px(radius))

    def draw_text(self, text, pos):
        text_surface = self._font.render(
                str(text), True, (0, 0, 0)
            )  # Black text
        text_rect = text_surface.get_rect(center=self._px(pos))
        self._surface.blit(text_surface, text_rect)

    def blit(self, surface, rect):
        self._surface.blit(surface, self._px(rect))

    def image_from_string(self, img, size, format):
        unscaled = pg.image.fromstring(img, size, format)
        new_size = np.multiply(size, self._sf)
        return pg.transform.scale(unscaled, new_size)

    def update(self):
        pg.display.update()

    def clear(self, colour):
        self._surface.fill(colour)
