import pygame as pg
import numpy as np
from swarm_construction.simulator.engine import SimulationEngine
from PIL import Image
import pygame as pg


class TargetShape:

    def __init__(self, origin_pos, shape, sim_engine: SimulationEngine):

        self._sim_engine = sim_engine
        self.origin_pos = origin_pos
        self.shape = shape

        self._sim_engine.add_draw(self.draw)

    def draw(self):

        # Convert the PIL image into a format that pygame will recognise, and store it as raw pixels.
        # pygame formats: https://www.pygame.org/docs/ref/image.html#pygame.image.frombuffer
        image_format = "RGB"
        raw_image = self.shape.convert(image_format).tobytes()

        # Turn the image into a pygame image.
        pygame_image = pg.image.fromstring(raw_image, self.shape.size, image_format)

        # TODO: We don't need to convert the image every frame, only once.
        # Would be a good idea to move the conversion code to the __init__ function, store the converted image,
        # and reference it in the below .blip(...) function call.

        # Draw the image.
        self._sim_engine.surface.blit(pygame_image, self.origin_pos)
