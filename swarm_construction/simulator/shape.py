import pygame as pg
import numpy as np
from swarm_construction.simulator.engine import SimulationEngine
from PIL import Image

class TargetShape:

    def __init__(
            self, 
            origin_pos,
            shape,
            sim_engine: SimulationEngine
            ):
        
        self._sim_engine = sim_engine
        self.origin_pos = origin_pos
        self.shape = shape
        
        self._sim_engine.add_draw(self.draw)

    def draw(self):
        # this doesnt work
        # think im just not using it right
        # this might be useful?
        # https://stackoverflow.com/questions/25202092/pil-and-pygame-image

        self._sim_engine.surface.blit(self.shape, self.origin_pos)

        
    
