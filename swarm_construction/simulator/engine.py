import pygame as pg
import math
import numpy as np

# A NOTE ON THE PYGAME COORDINATE SYSTEM
# pygame sets (0,0) as the top left corner. As such, a direction of
# 0 points straight down. How irritating is that.


class SimulationEngine:
    """The main simulation engine. This handles the pygame instance that draws everything to the screen"""

    def __init__(self, title, window_size, fps=60):
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

        # These lists contain the functions to run on each game loop.
        self.update_handlers = []
        self.draw_handlers = []

        # The SimulationObjects in the current simulation - see SimulationObject.
        # Simulation objects automatically add themselves to this when they are created.
        self._objects = []

        # How many frames to display per second.
        self.fps = fps

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

        self.running = True
        while self.running:
            # Limit the framerate to "self.fps".
            self.clock.tick(self.fps)
            self.update()
            self.draw()

        pg.quit()

    def update(self):
        """Update the simulation for the current frame. Update is LOGIC ONLY - no drawing!"""

        # First, handle any pygame events.
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                self.running = False

        # Call the update handlers.
        # Supply the handlers with the current fps for accurate physics.
        fps = self.clock.get_fps()
        for handler in self.update_handlers:
            handler(fps)

    def draw(self):
        """Draw the current frame of the simulation. No simulation logic should happen here!"""

        # Draw background.
        self.surface.fill((0, 0, 0))

        # Call draw handlers.
        for handler in self.draw_handlers:
            handler()

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
