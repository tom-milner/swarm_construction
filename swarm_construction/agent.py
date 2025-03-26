import pygame as pg
import numpy as np
import math


class Agent:

    def __init__(self, surface, pos, radius=30, color=(255, 255, 255), direction=0):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color
        self.direction = direction
        # Speed is measured in pixels per second!
        self.speed = 0

    def update(self, fps):
        if fps == 0:
            return
        scaled_speed = self.speed / fps
        v = scaled_speed * np.array(
            [math.sin(self.direction), math.cos(self.direction)]
        )
        self.pos = np.add(v, self.pos)

    def draw(self):
        # Circles are drawn around the center.
        pg.draw.circle(self.surface, self.color, self.pos, self.radius)

        # Draw direction line.
        line_end = (
            np.array([math.sin(self.direction), math.cos(self.direction)]) * self.radius
        )
        line_end = np.add(line_end, self.pos)
        direction_line_color = (199, 0, 57)
        pg.draw.line(self.surface, direction_line_color, self.pos, line_end)
