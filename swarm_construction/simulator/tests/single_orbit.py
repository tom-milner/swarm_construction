from swarm_construction.simulator.object import SimulationObject, OrbitDirection
from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.simulator.colors import Colour
import math
import numpy as np


class Test:

    def __init__(self):
        self.agents = []

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("Single Orbit Test", 800)
        # Add agents
        middle = self.sim.window_size / 2
        radius = 30

        sun = SimulationObject(
            self.sim, [middle, middle], radius=radius * 2, color=Colour.orange
        )

        planet = SimulationObject(
            self.sim, radius=radius, speed=100, color=Colour.yellow
        )
        planet.set_orbit_object(sun, orbit_direction=OrbitDirection.ANTI_CLOCKWISE)

        moon = SimulationObject(
            self.sim, radius=radius * 0.5, speed=150, color=Colour.light_green
        )
        moon.set_orbit_object(planet)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    mt = Test()
    mt.run()
