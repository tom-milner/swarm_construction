import pygame as pg
import numpy as np
import math
from .colors import Color
from enum import Enum
from swarm_construction.simulator.engine import SimulationEngine


class OrbitDirection(Enum):
    """Simple enum to encode orbiting directions - either CLOCKWISE or ANTI_CLOCKWISE."""

    CLOCKWISE = 0
    ANTI_CLOCKWISE = 1


class SimulationObject:
    """The lowest level class to use in a SimulationEngine simulation. SimulationObjects support simple physics, and can render themselves to the simulation.
    Simulation "agents" should inherit and extend upon this class.

    IMPORTANT: any class members (functions/variables) beginning with underscores (_) SHOULD NOT be used by any external or child classes. Whilst this may seem limiting, it makes the simulation more realistic.

    SimulationObjets can either move along vectors specified by a direction and speed, or along an orbit.
    """

    def __init__(
        self,
        sim_engine: SimulationEngine,
        pos: list = [0, 0],
        radius=30,
        color: Color = Color.white,
        direction: float = math.pi,
        speed: int = 0,
    ):
        """Initialise a SimulationObject to use in a SimulationEngine simulation.

        Args:
            sim_engine (SimulationEngine): The SimulationEngine to use this SimulationObject in. This simulation object will automatically add itself to the SimulationEngine.
            pos (list, optional): Object's position, as an [x,y] vector. Defaults to [0, 0].
            radius (int, optional): Radius of the object. Defaults to 30.
            color (tuple, optional): Color of the object, as an RGB vector (0-255,0-255,0-255). Defaults to white (255, 255, 255).
            direction (_type_, optional): Direction of the object, IN RADIANS. Defaults to math.pi.
            speed (int, optional): Speed of the object, in PIXLES PER SECOND. Defaults to 0.
        """

        # Initialise private vars - agents can't use these!
        self._sim_engine = sim_engine
        self._pos = pos
        self._radius = radius
        self._direction = direction
        self._orbit_object = None
        self._orbit_direction = OrbitDirection.CLOCKWISE

        # Initialise public vars - agents can access these.
        self.color = color
        self.speed = speed  # Speed is measured in pixels per second!

        # Add ourselves to the Simulation.
        self._sim_engine._objects.append(self)

        # Add our update and draw functions to the Simulation.
        self._sim_engine.add_update(self.update)
        self._sim_engine.add_draw(self.draw)

    def _move_orbit(self, pixels_per_frame: float):
        """Move the object in a circular orbit path around self.orbit_object.
        The direction of the object points along (tangential to) the orbit path.
        Currently, the object orbits with it's edge touching the orbit object.

        Args:
            pixels_per_frame (float): Speed of the object in pixels per frame.
        """

        if self._orbit_object is None:
            return

        # This is implemented weirdly!
        # I didn't want to store a "current_orbit_angle" variable in the class,
        # so we spin the object's direction, then position it so that it's
        # direction is perpendicular to the orbit_object.
        # It works, but there's probbably a better way!

        # Distance from center of this object to center of orbit object.
        dist = self._radius + self._orbit_object._radius

        # The angular velocity of this orbit.
        ang_vel = pixels_per_frame / dist

        # Set the rotation direction (clockwise, anticlockswise) of the orbit using self._orbit_direction.
        orbit_direction = 1 if self._orbit_direction == OrbitDirection.CLOCKWISE else -1

        # Update the objects direction to point along the orbit path (tangent to the orbit path), and clamp between 0-2PI.
        self._direction = (self._direction - ang_vel * orbit_direction) % (2 * math.pi)

        # Now we can position the object on the orbit path.

        # Calculate the angle perpendicular to our current direction.
        perp = self._direction + orbit_direction * (math.pi / 2)

        # Position ourselves in the orbit at that angle.
        orbit_vec = dist * np.array([math.sin(perp), math.cos(perp)])
        self._pos = np.add(self._orbit_object._pos, orbit_vec)

    def _move_vector(self, pixels_per_frame: float):
        """Move the object along the vector specified by self.direction and pixels_per_frame.

        Args:
            pixels_per_frame (_type_): Speed of the object in pixels per frame.s
        """

        # Turn speed and direction into a velocity vector.
        v = pixels_per_frame * np.array(
            [math.sin(self._direction), math.cos(self._direction)]
        )
        # Apply vector to current position.
        self._pos = np.add(v, self._pos)

    def update(self, fps: float):
        """Update the object in current frame. This is called by the simulation engine (Simulation).

        Args:
            fps (float): The frames per second since the last call.
        """

        # If there have been 0 frames since the last call we are on the first frame. Nothing to update, so move straight on!
        if fps == 0:
            return

        # Calculate the number of pixels we must move per frame, in order to produce the desired speed.
        pixels_per_frame = self.speed / fps

        # If we are orbiting an object, continue that orbit.
        if self._orbit_object:
            self._move_orbit(pixels_per_frame)
        else:
            # Else, just continue along our movement vector.
            self._move_vector(pixels_per_frame)

    def draw(self):
        """Draw the SimulationObject to the simulation.
        At the moment, all SimulationObjects look the same! This can be changed by overriding this draw() function.
        """

        # Draw a circle around the position coordinate.
        pg.draw.circle(self._sim_engine.surface, self.color, self._pos, self._radius)

        # Draw a line to indicate the circles current direction.
        line_end = (
            np.array([math.sin(self._direction), math.cos(self._direction)])
            * self._radius
        )
        line_end = np.add(line_end, self._pos)
        pg.draw.line(self._sim_engine.surface, Color.red, self._pos, line_end)

    def check_collision(self, other_object):
        """Check if we have collided with another SimulationObject.
        This currently only works with circles!

        Todo:
            Add bounding box checks first - may increase efficiency?
            Use Spatial Hash approach to increaese efficiency.

        Args:
            other_object (SimulationObject): The object to check for collision.

        Returns:
            tuple(bool, [int,int]): A tuple containing a boolean indicating whether collision has occured, and a vector describing how much we overlap the collided object.
        """

        # Get the vector between the objects.
        diff = np.subtract(self._pos, other_object._pos)

        # We have collided with the object if the distance between the objects is less than the sum of their radii.
        threshold = self._radius + other_object._radius
        distance = np.linalg.norm(diff)

        if distance > threshold:
            # No collision!
            return (False, np.array([0, 0]))

        # Collision!

        # Normalise the vector between the objects.
        norm = diff / distance if distance != 0 else 0

        # Calculate the vector where the object edges are touching but there is no overlap.
        edges_touching = norm * threshold

        # The overlap is the difference between this vector and our current distance vector.
        overlap = np.subtract(edges_touching, diff)

        return (True, overlap)

    def set_orbit_object(
        self, orbit_object, orbit_direction: OrbitDirection = OrbitDirection.CLOCKWISE
    ):
        """Make this object orbit around the provided object, in the provided direction.
        If we are already touching the object, we will orbit from there.
        If we aren't touching the object, we will reposition ourselves to be tangential to the object in our current position, then orbit from there.

        Args:
            orbit_object (SimulationObject): SimulationObject to orbit.
            orbit_direction (OrbitDirection, optional): Direction to orbit the object (clockwise or anticlockwise). Defaults to OrbitDirection.CLOCKWISE.
        """

        self._orbit_direction = orbit_direction

        # If we're already orbiting this object, return
        if orbit_object == self._orbit_object:
            return

        self._orbit_object = orbit_object

        # If we're already touching the edge of the object, change our
        # direction to start orbiting from where we are.
        col = self.check_collision(orbit_object)
        if col[0]:
            # Vector from ourselves to the orbit object.
            diff = np.subtract(self._pos, orbit_object._pos)

            # The angle that is perpencidular to the diff vector.
            perp = math.atan2(-diff[1], diff[0])
            self._direction = perp

            # NOTE: technically, we need to reposition ourselves if the overlap is too high (as we cannot exist inside another object), but the _move_orbit function handles this already.

        # If we're not touching the object, just position ourselves so that we're
        # perpendicular to it at our current direction.

        # The _move_orbit function does everything else!
        self._move_orbit(0)

    def get_nearest_neighbours(self, n: int):
        """Get a sorted list of the SimulationObjects nearest to this object in the Simulation, along with their distances.

        Todo:
            This is inefficient - switch to use Spatial Hashing if things start running slowly!

        Args:
            n (int): Number of neighbours to return.

        Returns:
            list: A list length<=n with the nearest SimulationObjects in the simulation, along with their distances.
        """

        # Naive implementation - replace with Spatial Hashing.
        neighbours = []

        # Iterate through all SimulationObjects in the Simulation.
        for obj in self._sim_engine._objects:

            # Skip ourselves.
            # TODO: Add IDs to each object, to make comparisons like this more efficient.
            if np.array_equal(obj._pos, self._pos):
                continue

            # Work out the distance from the current object.
            diff = np.subtract(self._pos, obj._pos)
            dist = np.linalg.norm(diff)

            # Store object and distance in neighbours array.
            entry = [obj, dist]
            neighbours.append(entry)

        # Turn neighbours into numpy array, and sort based on distance.
        neighbours = np.array(neighbours)
        return neighbours[neighbours[:, 1].argsort()][0:n]

    def is_orbiting(self):
        """Return whether we're currently orbiting an object or not.

        Returns:
            bool: Orbit status.
        """
        return bool(self._orbit_object)
