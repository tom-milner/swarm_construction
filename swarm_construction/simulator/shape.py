import pygame as pg
from swarm_construction.simulator.engine import SimulationEngine
from PIL import Image


class TargetShape:

    def __init__(self, origin_pos, shape, sim_engine: SimulationEngine):

        self._sim_engine = sim_engine
        self.origin_pos = origin_pos
        self.shape = shape

        # converts PIL image to a format pygame can use
        conv_shape = self.shape.convert("RGB").tobytes()
        # stores the pygame image so for use in draw()
        self.pgshape = pg.image.fromstring(conv_shape, self.shape.size, "RGB")

        # adds draw() method to the sim engine
        self._sim_engine.add_draw(self.draw)

    def draw(self):
        # draws the shape
        self._sim_engine.surface.blit(self.pgshape, self.origin_pos)
