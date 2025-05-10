from swarm_construction.simulator.object import SimulationObject
from swarm_construction.simulator.engine import SimulationEngine
import math
import numpy as np


class Test:

    def update(self, fps):
        if fps == 0:
            return

        # Move the agents
        for a in self.agents:
            a.update(fps)

        # Check to see if there have been any collisions.
        col = self.a_left.check_collision(self.a_right)
        if col[0]:  # Collision!
            # Stop agents.
            self.a_right.speed = 0
            self.a_left.speed = 0

            # If the agents are going quick enough, the collision will only be
            # registered once they are some way through each other, so we need
            # to reposition the agents so that they're only just touching.
            self.a_right._pos = np.subtract(self.a_right._pos, col[1])

    def draw(self):
        for a in self.agents:
            a.draw()

    def __init__(self):
        self.agents = []

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("Collision Test", 800)
        # Add agents
        middle = self.sim.game_size / 2
        offset = middle / 2
        speed = 100

        self.a_left = SimulationObject(
            self.sim,
            [middle - offset, middle],
            direction=math.pi / 2,
            speed=speed,
        )

        self.a_right = SimulationObject(
            self.sim,
            [middle + offset, middle],
            direction=math.pi * 3 / 2,
            speed=speed,
        )
        self.agents.append(self.a_left)
        self.agents.append(self.a_right)

        self.sim.add_update(self.update)
        self.sim.add_draw(self.draw)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    mt = Test()
    mt.run()
