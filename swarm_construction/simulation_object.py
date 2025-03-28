import pygame as pg
import numpy as np
import math
from .colors import Color


class SimulationObject:

    def _do_nothing(self):
        pass

    def __init__(
        self,
        surface,
        pos=[0, 0],
        radius=30,
        color=(255, 255, 255),
        direction=math.pi,
        speed=0,
    ):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color
        self.direction = direction
        self.speed = speed  # Speed is measured in pixels per second!
        self.on_collision = self._do_nothing

        self._orbit_object = None

    def _move_orbit(self, scaled_speed):
        # I don't why I did this, but this bit is implemented backwards!
        # I didn't want to store an "orbit_angle" variable in the class,
        # so we spin the object directly, then position it so that it's
        # direction is perpendicular to the orbit_object.
        # It works, but there's most definetely a better way!

        # Apply the object velocity as an angular velocity.
        dist = self.radius + self._orbit_object.radius
        ang_vel = scaled_speed / dist
        self.direction = (self.direction - ang_vel) % (2 * math.pi)

        # Calculate the angle perpendicular to our current direction.
        perp = self.direction - (math.pi / 2)

        # Position ourselves in the orbit at that angle.
        orbit_vec = dist * np.array([math.sin(perp), math.cos(perp)])
        self.pos = np.subtract(self._orbit_object.pos, orbit_vec)

    def _move_vector(self, scaled_speed):
        # Turn speed and direction into a velocity vector.
        v = scaled_speed * np.array(
            [math.sin(self.direction), math.cos(self.direction)]
        )
        # Apply vector to current position.
        self.pos = np.add(v, self.pos)

    def update(self, fps):
        if fps == 0:
            return

        # Speed is measured in pixels/second. We convert it to pixels/frame using
        # the frame rate.
        scaled_speed = self.speed / fps

        if self._orbit_object:
            self._move_orbit(scaled_speed)
        else:
            self._move_vector(scaled_speed)

    def draw(self):

        # Circles are drawn around the position coordinate.
        pg.draw.circle(self.surface, self.color, self.pos, self.radius)

        # Draw a line to indicate the circles current direction.
        line_end = (
            np.array([math.sin(self.direction), math.cos(self.direction)]) * self.radius
        )
        line_end = np.add(line_end, self.pos)
        direction_line_color = Color.red
        pg.draw.line(self.surface, direction_line_color, self.pos, line_end)

    def check_collision(self, other_object):
        """Check to see if this object has collided with another"""

        # Get the vector between the objects.
        diff = np.subtract(self.pos, other_object.pos)

        threshold = self.radius + other_object.radius
        distance = np.linalg.norm(diff)
        if distance <= threshold:
            # Collision!
            # Work out how deep the collision is.
            if distance == 0:
                norm = 0
            else:
                norm = diff / distance
            penetration = np.subtract(norm * threshold, diff)
            return (True, penetration)
        return (False, np.array([0, 0]))

    def set_orbit_object(self, orbit_object):
        self._orbit_object = orbit_object

        # If we're already touching the edge of the object, change our
        # direction and start orbiting from there.
        col = self.check_collision(orbit_object)
        if col[0]:
            # Change our direction to be perpendicular to the object.
            diff = np.subtract(self.pos, orbit_object.pos)
            perp = math.atan2(-diff[1], diff[0])
            self.direction = perp

        # If we're not touching the object, position ourselves so that we're
        # perpendicular to it at our current direction.
        self._move_orbit(0)
