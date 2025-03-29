import pygame as pg
import numpy as np
import math
from .colors import Color
from enum import Enum


class OrbitDirection(Enum):
    CLOCKWISE = 0
    ANTI_CLOCKWISE = 1


class SimulationObject:

    def _do_nothing(self):
        pass

    def __init__(
        self,
        sim_engine,
        pos=[0, 0],
        radius=30,
        color=(255, 255, 255),
        direction=math.pi,
        speed=0,
    ):
        # Private vars - agents can't use these!
        self._sim_engine = sim_engine
        self._pos = pos
        self._radius = radius
        self._direction = direction
        self._orbit_object = None
        self._orbit_direction = OrbitDirection.CLOCKWISE

        # Public - agent can use these
        self.on_collision = self._do_nothing
        self.color = color
        self.speed = speed  # Speed is measured in pixels per second!

        # Add ourselves to the sim_engine.
        self._sim_engine._objects.append(self)

    def _move_orbit(self, scaled_speed):
        # I don't why I did this, but this bit is implemented backwards!
        # I didn't want to store an "orbit_angle" variable in the class,
        # so we spin the object directly, then position it so that it's
        # direction is perpendicular to the orbit_object.
        # It works, but there's most definetely a better way!

        # Apply the object velocity as an angular velocity.
        dist = self._radius + self._orbit_object._radius
        ang_vel = scaled_speed / dist
        orbit_direction = -1 if self._orbit_direction == OrbitDirection.CLOCKWISE else 1
        self._direction = (self._direction + ang_vel * orbit_direction) % (2 * math.pi)

        # Calculate the angle perpendicular to our current direction.
        perp = self._direction + orbit_direction * (math.pi / 2)

        # Position ourselves in the orbit at that angle.
        orbit_vec = dist * np.array([math.sin(perp), math.cos(perp)])
        self._pos = np.subtract(self._orbit_object._pos, orbit_vec)

    def _move_vector(self, scaled_speed):
        # Turn speed and direction into a velocity vector.
        v = scaled_speed * np.array(
            [math.sin(self._direction), math.cos(self._direction)]
        )
        # Apply vector to current position.
        self._pos = np.add(v, self._pos)

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

        # At the moment, all agents look the same.
        # This can be changed by moving the draw function into the agent.
        # Circles are drawn around the position coordinate.
        pg.draw.circle(self._sim_engine.surface, self.color, self._pos, self._radius)

        # Draw a line to indicate the circles current direction.
        line_end = (
            np.array([math.sin(self._direction), math.cos(self._direction)])
            * self._radius
        )
        line_end = np.add(line_end, self._pos)
        direction_line_color = Color.red
        pg.draw.line(
            self._sim_engine.surface, direction_line_color, self._pos, line_end
        )

    def check_collision(self, other_object):
        """Check to see if this object has collided with another"""

        # Get the vector between the objects.
        diff = np.subtract(self._pos, other_object._pos)

        threshold = self._radius + other_object._radius
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

    def set_orbit_object(self, orbit_object, orbit_direction=OrbitDirection.CLOCKWISE):
        self._orbit_direction = orbit_direction

        # If we're already orbiting this object, return
        if orbit_object == self._orbit_object:
            return

        self._orbit_object = orbit_object

        # If we're already touching the edge of the object, change our
        # direction and start orbiting from there.
        col = self.check_collision(orbit_object)
        if col[0]:
            # Change our direction to be perpendicular to the object.
            diff = np.subtract(self._pos, orbit_object._pos)
            perp = math.atan2(-diff[1], diff[0])
            self._direction = perp

        # If we're not touching the object, position ourselves so that we're
        # perpendicular to it at our current direction.
        self._move_orbit(0)

    def get_nearest_neighbours(self, n):
        # Naive implementation - replace with Spacial Hashing.
        neighbours = []
        for obj in self._sim_engine._objects:
            # Skip ourselves.
            if np.array_equal(obj._pos, self._pos):
                continue

            diff = np.subtract(obj._pos, self._pos)
            dist = np.linalg.norm(diff)
            entry = [obj, dist]

            neighbours.append(entry)

        neighbours = np.array(neighbours)
        return neighbours[neighbours[:, 1].argsort()][0:n]

    def is_orbiting(self):
        return bool(self._orbit_object)
