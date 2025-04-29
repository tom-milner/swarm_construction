from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.agent import Agent
import math
import numpy as np


class Test:

    def _generate_cluster(self, middle, radius):
        cluster = []
        num_rows = 10
        num_cols = 10
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * radius))
        start = [middle - num_cols * radius, middle - num_rows * radius]

        # generate rectangle
        for i in range(num_rows):
            x_offset = 0 if i % 2 == 0 else radius
            for j in range(num_cols):
                pos = [x_offset + radius * 2 * j, line_spacing * i]
                a = Agent(
                    self.sim,
                    np.add(start, pos),
                )
                cluster.append(a)
        i += 1

        # generate tail
        while i < (num_rows + 6):
            x_offset = 0 if i % 2 == 0 else radius
            pos = [x_offset + radius * 2 * round(j / 2), line_spacing * i]
            a = Agent(
                self.sim,
                np.add(start, pos),
            )
            cluster.append(a)
            i += 1

        return cluster

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("Agent Edge Following Test", 800)
        middle = self.sim.window_size / 2

        # Add stationary agents.
        self._generate_cluster(middle, Agent.radius)

        # Add one moving agent.
        new_agent = Agent(self.sim, [middle, self.sim.window_size-1])
        new_agent.speed = 100

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    Test().run()
