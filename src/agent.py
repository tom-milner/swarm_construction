import pygame as pg
import numpy as np
import math


class Agent:

    def __init__(self, surface, pos, radius=30, color=(255, 255, 255), angle=0):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color
        self.angle = angle

    def update(self):
        pass

    def draw(self):
        # Circles are drawn around the center.
        pg.draw.circle(self.surface, self.color, self.pos, self.radius)

        # Draw direction line.
        line_end = np.array([math.sin(self.angle), math.cos(self.angle)]) * self.radius
        line_end = np.add(line_end, self.pos)
        pg.draw.line(self.surface, (199, 0, 57), self.pos, line_end)
