import pygame as pg
import math
import numpy as np
from swarm_construction.simulator.colors import Colour

# A NOTE ON THE PYGAME COORDINATE SYSTEM
# pygame sets (0,0) as the top left corner. As such, a direction of
# 0 points straight down. How irritating is that.


def do_nothing():
    pass


class SimulationEngine:
    """The main simulation engine. This handles the pygame instance that draws everything to the screen"""

    def __init__(
        self,
        title,
        window_size,
        draw_rate=30,
        update_rate=100,
        analytics_func=do_nothing,
    ):
        """Initialise the Simulation Engine.

        Args:
            title (string): The title of the simulation.
            window_size (int): The length of each side of the square window.
        """
        self.title = title
        self.window_size = window_size

        # Initialise pygame.
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.set_caption(self.title)

        # Initialise simulation.
        self.surface = pg.display.set_mode((self.window_size, self.window_size))
        self.running = False
        self.pause = False
        self.draw_neighbourhoods = False

        # These lists contain the functions to run on each game loop.
        self.update_handlers = []
        self.draw_handlers = []

        # The SimulationObjects in the current simulation - see the SimulationObject class.
        # Simulation objects automatically add themselves to this when they are created.
        self._objects = []

        # How many frames to display per second.
        self._draw_rate = draw_rate

        # How many times to update per second. Don't let this be less than the draw rate
        self._update_rate = max(draw_rate, update_rate)

        self.analytics_func = analytics_func

        # Name of the shape file for use in analytics files
        self.shape_name = None

    def run(self):
        """Run the main game loop. This runs until the "running" flag is set to False.

        The main simulation loop has two main parts:
        1. Update
        This updates the *state* of the simulation. Any logic, physics, or computation, happens in this part of the game loop. No drawing happens here!

        2. Draw
        Draw whatever has been computed by the Update() function to the screen.

        These are split up to keep logic separate from graphics, just to make things a bit cleaner.
        Update and Draw are called once per frame.
        """

        # We use spatial hashing to divide the total game area into "neighbourhoods". This allows
        # each simulation object to immediately query it's neighbours.

        # Calculate the size of dimensions of each neighbourhood based on the agent radius.
        neigh_width = self._objects[0]._radius * 2 * 4
        self._neighbourhood_dim = [neigh_width, neigh_width]

        # Calculate the number of neighbourhoods along each axis (x and y).
        self._neighbourhood_idx = np.astype(
            np.ceil(np.divide(self.window_size, self._neighbourhood_dim)), int
        )

        # Generate neighbourhoods. Each neighbourhood is a np array.
        self._neighbourhoods = np.ndarray(self._neighbourhood_idx, dtype=np.ndarray)
        for x in range(self._neighbourhood_idx[0]):
            for y in range(self._neighbourhood_idx[1]):
                self._neighbourhoods[x][y] = np.array([])

        # The number of updates we should run before drawing a frame.
        draw_frame_count = round(self._update_rate / self._draw_rate)
        count = 0

        self.running = True
        while self.running:

            self.clock.tick(self._update_rate)
            self.update()

            count += 1
            if count == draw_frame_count:
                self.draw()
                count = 0

        pg.quit()

    def update(self):
        """Update the simulation for the current frame. Update is LOGIC ONLY - no drawing!"""

        # First, handle any pygame events.
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                self.running = False

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_n:
                    self.draw_neighbourhoods = not self.draw_neighbourhoods
                if event.key == pg.K_p:
                    self.pause = not self.pause
                if event.key == pg.K_a:
                    if not self.pause:
                        return
                    self.analytics_func()

        if self.pause:
            return

        # Supply the handlers with the current fps for accurate physics.
        fps = self.clock.get_fps()

        # Call any extra update handlers.
        for handler in self.update_handlers:
            handler(fps)

        # Update the objects
        for obj in self._objects:
            obj.update(fps)

    def draw(self):
        """Draw the current frame of the simulation. No simulation logic should happen here!"""

        # Draw background.
        self.surface.fill((0, 0, 0))

        # Call draw handlers.
        for handler in self.draw_handlers:
            handler()

        # Draw the objects.
        for obj in self._objects:
            obj.draw()

        # Draw Neighbourhoods.
        colour = Colour.orange
        width = 2
        if self.draw_neighbourhoods:
            # Vertical Lines
            for i in range(self._neighbourhood_idx[0]):
                x = i * self._neighbourhood_dim[0]
                start = (x, 0)
                end = (x, self.window_size)
                pg.draw.line(self.surface, colour, start, end, width=width)

            # Horizontal Lines
            for i in range(self._neighbourhood_idx[1]):
                y = i * self._neighbourhood_dim[1]
                start = (0, y)
                end = (self.window_size, y)
                pg.draw.line(self.surface, colour, start, end, width=width)

        # Draw the simulation to the pygame window.
        pg.display.update()

    def add_update(self, handler):
        """Add an update handler to the simulation. This will be called once every game loop. Update handlers are logic only - no drawing should happen in the handler.

        Args:
            handler (Callable): The update handler function to add.
        """
        self.update_handlers.append(handler)

    def add_draw(self, handler):
        """Add a draw handler function to the simulation. This will be called once every game loop. Draw handlers are graphics only - no simulation logic should happen in the handler.

        Args:
            handler (Callabel): The draw handler function to add.
        """
        self.draw_handlers.append(handler)

    def assign_neighbourhood(self, sim_obj):
        """Assign a simulation object to a neighbourhood, and update the engine's internal neighbourhood
        to store the object.

        Args:
            sim_obj (SimulationObject): The simulation object to assign to a neighbourhood.
        """
        # Calculate what neighbourhood this object is in.
        new_neighbourhood_idx = np.astype(
            np.divide(sim_obj._pos, self._neighbourhood_dim), int
        )

        # If we haven't changed neighbourhood, do nothing.
        if np.array_equal(sim_obj._neighbourhood, new_neighbourhood_idx):
            return new_neighbourhood_idx

        # Remove the object from the old neighbourhood.
        old_x, old_y = sim_obj._neighbourhood
        if old_x != None and old_y != None:
            for idx, obj in enumerate(self._neighbourhoods[old_x][old_y]):
                if obj.object_id != sim_obj.object_id:
                    continue
                self._neighbourhoods[old_x][old_y] = np.delete(
                    self._neighbourhoods[old_x][old_y], idx
                )

        # Add object to new neighbourhood.
        new_x, new_y = new_neighbourhood_idx
        assert (
            new_x < self._neighbourhood_idx[0]
        ), "Agent out of bounds - neighbourhood cannot be calculated."
        assert (
            new_y < self._neighbourhood_idx[1]
        ), "Agent out of bounds - neighbourhood cannot be calculated."
        new_neighbourhood = self._neighbourhoods[new_x][new_y]
        self._neighbourhoods[new_x][new_y] = np.append(new_neighbourhood, sim_obj)

        # Set neighbourhood of object.
        sim_obj._neighbourhood = new_neighbourhood_idx
        return new_neighbourhood_idx

    def get_nearby_objects(self, neighbourhood_coords):
        """Get all the simulation objects in a 3x3 square around the provided neighbourhood.

        Args:
            neighbourhood_coords (tuple): (x,y) indices of neighbourhood.
        """
        if not np.any(neighbourhood_coords):
            return np.array([])
        x, y = neighbourhood_coords
        max_x, max_y = self._neighbourhood_idx
        nearby = np.array([])

        # Loop through the neighbourhood to the left, ourselves, and the neighbourhood to the right.
        for dx in range(-1, 2):

            # Calculate the x coordinate of the current neighbourhood.
            target_x = x + dx

            # Make sure the neighbourhood exists.
            if (target_x < 0) or (target_x >= max_x):
                continue

            # Loop through the neighbourhood above, ourselves, and the neighbourhood below.
            for dy in range(-1, 2):
                target_y = y + dy
                if (target_y < 0) or (target_y >= max_y):
                    continue

                # Store the objects in the neighbourhood
                nearby = np.concatenate(
                    (nearby, self._neighbourhoods[target_x][target_y])
                )
        return nearby
