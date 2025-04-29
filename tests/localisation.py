from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.agent import Agent
from swarm_construction.simulator.colors import Colour
import math
import numpy as np


class Test:

    def generate_seeds(self, line_spacing: int, column_spacing: int, origin: list):
        """Generate and place the seed agents.

        Args:
            line_spacing (int): Number of pixels between lines (y-axis) to ensure a snug fit.
            column_spacing (int): Number of pixels between columns (x-axis) to ensure a snug fit.
            origin (list): [x,y] position to build the seed agents around.

        Returns:
            list: Positions of the seed agents.
        """

        # This seed configuration shape is taken from the paper.
        dx = column_spacing
        dy = line_spacing
        seed_deltas = np.array(
            [
                [-dx, 0],
                [dx, 0],
                [0, -dy],
                [0, dy],
            ]
        )

        # Place the seeds relative to the origin.
        seed_pos = np.add(origin, seed_deltas)

        # Localise the seeds by giving them an initial local_pos.
        # This uses a normal x-y coordinate system (0,0 = bottom left), so
        # so we flip our pygame coordinates.
        local_pos = [delta * [1, -1] for delta in seed_deltas]

        # Add the seeds to the simulation.
        Agent(self.sim, seed_pos[0], local_pos=local_pos[0]),
        Agent(self.sim, seed_pos[1], local_pos=local_pos[1]),
        Agent(self.sim, seed_pos[2], local_pos=local_pos[2]),
        Agent(self.sim, seed_pos[3], local_pos=local_pos[3]),

        # Return the positions of the seeds.
        return seed_pos

    def update(self, fps):

        # generate an agent if there isn't one active.
        if self.agent is None:
            self.agent = Agent(self.sim, self.start_pos)
            # Slow speed, just for the test.
            # If the agent is quick, it's local_pos may jump over the y-axis within a single frame,
            # and the zero crossing will miss it.
            self.agent.speed = 100

        # wait for the agent to start localising.
        if self.agent.local_pos is None:
            return

        # stop agent at y-axis crossing.
        threshold = 1
        if -threshold < self.agent.local_pos[1] < threshold:
            self.agent.speed = 0
            print(
                f"stopped @ {self.agent.local_pos}, actual_pos = {np.subtract(self.agent._pos,[400,400]) * [1,-1]}"
            )

            self.agent = None
            self.num_moving_agents -= 1
            if self.num_moving_agents == 0:
                # stop sim
                self.sim.running = False
                return

        else:
            print(
                f"moving @ {self.agent.local_pos}, actual_pos = {np.subtract(self.agent._pos,[400,400])* [1,-1]}"
            )

    def run(self):

        # Setup the simulation
        self.sim = SimulationEngine("Localisation Test", 800)
        middle = self.sim.window_size / 2

        # Setup seed agents.
        Agent.radius = 30
        seed_origin = [middle, middle]
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * Agent.radius))
        column_spacing = Agent.radius

        # Generate the seed agents around the origin.
        seed_pos = self.generate_seeds(line_spacing, column_spacing, seed_origin)

        # Setup moving agents.
        agent_pos = [0, 2 * line_spacing]
        self.start_pos = np.add(seed_pos[3], agent_pos)
        self.num_moving_agents = 2
        self.agent = None

        self.sim.add_update(self.update)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    Test().run()
