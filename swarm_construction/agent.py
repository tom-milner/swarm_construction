import pygame as pg
import numpy as np
import math


class Agent:

    def _do_nothing(self):
        pass

    def __init__(
        self, surface, pos, radius=30, color=(255, 255, 255), direction=math.pi, speed=0
    ):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color
        self.direction = direction
        self.speed = speed  # Speed is measured in pixels per second!
        self.on_collision = self._do_nothing

    def update(self, fps):
        if fps == 0:
            return

        # Speed is measured in pixels/second. We convert it to pixels/frame using
        # the frame rate.
        scaled_speed = self.speed / fps

        # Turn speed and direction into a velocity vector.
        v = scaled_speed * np.array(
            [math.sin(self.direction), math.cos(self.direction)]
        )
        # Apply vector to current position.
        self.pos = np.add(v, self.pos)

    def draw(self):
        # Circles are drawn around the position coordinate.
        pg.draw.circle(self.surface, self.color, self.pos, self.radius)

        # Draw a line to indicate the circles current direction.
        line_end = (
            np.array([math.sin(self.direction), math.cos(self.direction)]) * self.radius
        )
        line_end = np.add(line_end, self.pos)
        direction_line_color = (199, 0, 57)
        pg.draw.line(self.surface, direction_line_color, self.pos, line_end)

    def check_collision(self, other_agent):
        """Check to see if this agent has collided with another"""
        # Get the vector between the agents.
        diff = np.subtract(self.pos, other_agent.pos)

        threshold = self.radius + other_agent.radius
        distance = np.linalg.norm(diff)
        if distance < threshold:
            # Collision!
            # Work out how deep the collision is.
            norm = diff / distance
            penetration = np.subtract(norm * threshold, diff)
            return (True, penetration)
        return (False, 0)
