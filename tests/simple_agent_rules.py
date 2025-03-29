from swarm_construction.simulation_engine import Simulation
from swarm_construction.agent import Agent
import math
import numpy as np


class Test:

    def _generate_cluster(self, middle, radius):
        cluster = []
        num_rows = 32
        num_cols = 32
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
        self.sim = Simulation("Simple Agent Rules Test", 800)
        middle = self.sim.window_size / 2

        # Add stationary agents.
        agents = self._generate_cluster(middle, Agent.radius)
        [self.sim.add_update(ag.update) for ag in agents]
        [self.sim.add_draw(ag.draw) for ag in agents]

        # Add one moving agent.
        new_agent = Agent(self.sim, [middle, self.sim.window_size])
        new_agent.speed = 100
        self.sim.add_update(new_agent.update)
        self.sim.add_draw(new_agent.draw)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    Test().run()
