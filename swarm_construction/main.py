from .simulator.engine import SimulationEngine
from .simulator.colors import Color
from .agent import Agent
from .simulator.shape import TargetShape

import math
import numpy as np
from PIL import Image


class SwarmConstructionSimulation:
    """The entry point for the shape-constructing-swarm simulation."""

    def calculate_column_spacing(self, window_size, num_agents):
        """Calculate how big we can make each agent given our current window size.

        Args:
            window_size (int): Length of each side of the (square) window.
            num_agents (int): Number of agents to fit in the window.

        Returns:
            int: Calculated agent radius.
        """

        # Agents will take up 0.1 of the area of the screen.
        agent_area_proportion = 0.1
        window_area = window_size * window_size
        total_agent_area = window_area * agent_area_proportion
        agent_area = total_agent_area / num_agents

        # Calculate the radius of each agent.
        return math.floor(math.sqrt(agent_area / math.pi))

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

        # Add the seeds to the simulation.
        self.agents.extend(
            [
                Agent(self.sim, seed_pos[0], swarm_pos=[-1, 0]),
                Agent(self.sim, seed_pos[1], swarm_pos=[1, 0]),
                Agent(self.sim, seed_pos[2], swarm_pos=[0, 1]),
                Agent(self.sim, seed_pos[3], swarm_pos=[0, -1]),
            ]
        )

        # Return the positions of the seeds.
        return seed_pos

    def generate_connecting_agents(self, line_spacing, column_spacing, origin_agent):
        """Setup and place connecting agents that connect the cluster to the seeds.

        Two agents underneath the provided origin agent.
        This is not strictly necessary, I just wanted to copy how it looks in the paper!

        Args:
            line_spacing (int): Number of pixels between lines (along y-axis) to ensure a snug fit.
            column_spacing (int): Number of pixels between columns (along x-axis) to ensure a snug fit.
            origin_agent (list): [x,y] position of the agent to build the connecting agents around.

        Returns:
            list: The positions of the connecting agents.
        """

        # Place the connector agents underneath the origin agent.
        dx = column_spacing
        dy = line_spacing
        conn_deltas = np.array([[-dx, dy], [dx, dy]])
        conn_pos = np.add(conn_deltas, origin_agent)

        # Generate the connecting agents and add them to the simulation.
        self.agents.extend([Agent(self.sim, pos) for pos in conn_pos])

        # Return the positions of the connecting agents.
        return conn_pos

    def generate_cluster_agents(
        self, num_agents, line_spacing, column_spacing, origin_agent
    ):
        """Generate and place the cluster agents, based off the provided origin agent position.
        The cluster is currently square shaped.

        Args:
            num_agents (int): Number of agents to generate in the cluster.
            line_spacing (int): Number of pixels between lines (y-axis) to ensure a snug fit.
            column_spacing (int): Number of pixels between columns (x-axis) to ensure a snug fit.
            origin (list): [x,y] position of the agent to build the connecting agents around.
        """

        # Calculate the side length of the square.
        side_len = math.ceil(math.sqrt(num_agents))

        # Place the cluster underneath, and nudged left, of the origin agent.
        dx = column_spacing
        dy = line_spacing
        cluster_start = np.add(origin_agent, [-dx, dy])

        # Generate the agents in the cluster.
        for i in range(side_len):
            # Every other row is nudged forwards, so the circles fit snuggly.
            x_offset = 0 if i % 2 == 0 else dx

            for j in range(side_len):

                # If all the agents have been generated, we're done.
                if num_agents <= 0:
                    return

                # Place this agent in the cluster square.
                pos = [x_offset + dx * 2 * j, dy * i]
                pos = np.add(pos, cluster_start)

                # Generate the agent.
                self.agents.append(Agent(self.sim, pos))
                num_agents -= 1

    def place_agents(self, num_agents):
        """Generate and place agents in the simulation.

        Args:
            num_agents (int): Number of agents to generate and place.
        """
        # Calculate how big we can make the agents given our current window size.
        Agent.radius = self.calculate_column_spacing(self.sim.window_size, num_agents)

        # The position to build the seed agents around in the window.
        seed_origin = [0.2 * self.sim.window_size, 0.5 * self.sim.window_size]

        # Because our agents are circular, we can make them fit snuggly together if we
        # slot each row of agents into the gaps between agents in the previous row.
        # line_spacing is the y spacing between rows to make them all snuggly, and
        # column_spacing is the x spacing between agents across a pair of snuggly rows.
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * Agent.radius))
        column_spacing = Agent.radius

        # Generate the seed agents around the origin.
        seed_pos = self.generate_seeds(line_spacing, column_spacing, seed_origin)
        num_agents -= len(seed_pos)

        #  Generate the connector agents at the bottom of the seed.
        bottom_of_seeds = seed_pos[np.argmax(seed_pos, axis=0)[1]]
        connectors = self.generate_connecting_agents(
            line_spacing, column_spacing, bottom_of_seeds
        )
        num_agents -= len(connectors)

        # Generate the cluster of agents underneath the left connector agent.
        left_conn = connectors[np.argmin(connectors, axis=0)[1]]
        self.generate_cluster_agents(
            num_agents, line_spacing, column_spacing, left_conn
        )

    def place_shape(self, shape_file):
        # shape area is set to the same as the area of robots currently
        shape_area_proportion = 0.1
        window_area = self.sim.window_size**2
        goal_shape_area = window_area * shape_area_proportion

        # this whole thing scales the inputted shape file to match the robot area
        shape = Image.open(shape_file)
        init_shape_area = sum(pixel == 255 for pixel in shape.getdata())

        scale_factor = math.sqrt(goal_shape_area / init_shape_area)
        scaled_shape = shape.resize(
            (int(shape.width * scale_factor), int(shape.height * scale_factor)),
            Image.NEAREST,
        )
        # This flips the shape, may need to be removed if we change how we generate shapes
        scaled_shape = scaled_shape.transpose(Image.FLIP_TOP_BOTTOM)
        scaled_shape.save("scaled_shape_test.bmp")

        # gets origin for correct placement
        # This is the seed origin - the height of the shape (y axis) due to silly coordinate systems
        # shape_origin is therefore the pygame coordinates of the top left corner of the image
        # This assumes that the bottom left corner of the image is included in the shape
        # nothing to stop the image being placed off the screen currently
        shape_origin = [
            0.2 * self.sim.window_size,
            0.5 * self.sim.window_size - scaled_shape.height,
        ]

        # Instantiates TargetShape class with shape data
        self.target_shape = TargetShape(shape_origin, scaled_shape, self.sim)

    def main(self):
        self.sim = SimulationEngine("Swarm Construction", 800)
        self.agents = []

        self.place_shape("test_shape.bmp")
        # Place the agents (robots!) in the simulation.
        self.place_agents(100)

        # TESTING: make the last one move.
        self.agents[-1].speed = 100

        self.sim.run()


if __name__ == "__main__":
    swarm_sim = SwarmConstructionSimulation()
    swarm_sim.main()
