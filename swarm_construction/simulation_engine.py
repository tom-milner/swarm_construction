import pygame as pg
import math
import numpy as np


class Simulation:

    def __init__(self):
        self.title = "Swarm Construction"
        self.window_size = 800

        # Initialise pygame.
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.set_caption(self.title)

        # Initialise simulation.
        self.surface = pg.display.set_mode((self.window_size, self.window_size))
        self.running = False

        self.update_callbacks = []
        self.draw_callbacks = []

    def run(self):
        # Main loop.
        self.running = True
        while self.running:
            self.clock.tick()
            self.update()
            self.draw()

        pg.quit()

    def update(self):
        # Handle events.
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False

        # Call update callbacks.
        fps = self.clock.get_fps()
        for c in self.update_callbacks:
            c(fps)

    def draw(self):

        # Draw background.
        self.surface.fill((0, 0, 0))

        # Call draw callbacks.
        for c in self.draw_callbacks:
            c()

        pg.display.update()

    def add_update(self, func):
        self.update_callbacks.append(func)

    def add_draw(self, func):
        self.draw_callbacks.append(func)
