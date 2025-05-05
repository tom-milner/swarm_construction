from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.simulator.colors import Colour
from swarm_construction.agent import Agent
from swarm_construction.simulator.shape import SimulationShape
from swarm_construction.simulator.analytics import Analytics

import math
import numpy as np
from PIL import Image


class Test:
    """The entry point for the shape-constructing-swarm simulation."""

    def calculate_agent_radius(self, window_size, num_agents):
        """Calculate how big we can make each agent given our current window size.

        Args:
            window_size (int): Length of each side of the (square) window.
            num_agents (int): Number of agents to fit in the window.

        Returns:
            int: Calculated agent radius.
        """

        # Make area of agents same as the shape area.
        window_area = window_size**2
        total_agent_area = window_area * self.shape_area_proportion
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

        # Localise the seeds by giving them an initial local_pos.
        # This uses a normal x-y coordinate system (0,0 = bottom left), so
        # so we flip our pygame coordinates.
        local_pos = [delta * [1, -1] for delta in seed_deltas]

        # Add the seeds to the simulation.
        for i in range(len(seed_deltas)):
            Agent(
                self.sim,
                seed_pos[i],
                local_pos=local_pos[i],
                shape=self.target_shape,
                gradient=0 if i == 0 else 1,
            )

        # Return the positions of the seeds.
        return seed_pos

    def place_agents(self):
        """Generate and place agents in the simulation."""

        # Calculate how big we can make the agents given our current window size.
        Agent.radius = self.calculate_agent_radius(self.sim.window_size, 5)
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * Agent.radius))
        column_spacing = Agent.radius

        # Generate the seed agents around the origin.
        seed_pos = self.generate_seeds(line_spacing, column_spacing, self.seed_origin)

        # Generate dummy agents to make things difficult.
        dummy_pos = np.add(seed_pos[3], [column_spacing, line_spacing])
        Agent(self.sim, start_pos=dummy_pos)
        dummy_pos = np.add(dummy_pos, [-column_spacing * 2, 0])
        Agent(self.sim, start_pos=dummy_pos)

        agent_pos = [-column_spacing, 0]
        start_pos = np.add(dummy_pos, agent_pos)
        mov = Agent(
            self.sim,
            start_pos=start_pos,
            shape=self.target_shape,
        )
        mov.speed = 100

    def place_shape(self, shape_file):

        window_area = self.sim.window_size**2
        goal_shape_area = window_area * self.shape_area_proportion

        # this whole thing scales the inputted shape file to match the robot area
        shape = Image.open(shape_file)
        init_shape_area = sum(pixel == 255 for pixel in shape.getdata())

        scale_factor = math.sqrt(goal_shape_area / init_shape_area)
        scaled_shape = shape.resize(
            (int(shape.width * scale_factor), int(shape.height * scale_factor)),
            Image.NEAREST,
        )
        # This flips the shape, may need to be removed if we change how we generate shapes
        # NOTE not required when using shape_create_gui to make bmp file!
        # scaled_shape = scaled_shape.transpose(Image.FLIP_TOP_BOTTOM)
        # converts image to numpy array
        shape_array = np.array(scaled_shape)

        # finds the bottom left most white pixel in the scaled shape
        # in the array this is actually max y (row) with min x (col)
        # this finds the indices of all white pixels
        white_px_index = np.argwhere(shape_array == True)

        # confusingly when dealing with the image through pillow pixels are identified as [x,y]
        # but when dealing with it as an np array the pixels are targeted by [y,x] ([rows, columns])
        max_y_px = white_px_index[white_px_index[:, 0] == np.max(white_px_index[:, 0])]

        # origin_px is the bottom left white pixel in the shape (when looking at it on screen)
        # in format [y,x] (silly), with origin (0,0) top left.
        origin_px = max_y_px[np.argmin(max_y_px[:, 1])]

        # draw_origin is the location for the top left pixel in the image - of any colour
        # self.seed_origin is the centre of the seeds in format [x,y] (with origin0,0 in top left corner)
        draw_origin = [
            self.seed_origin[0] - origin_px[1],
            self.seed_origin[1] - origin_px[0],
        ]

        # Instantiates SimulationShape class with shape data
        SimulationShape(draw_origin, scaled_shape, self.sim)

        # Create coordinates of bottom left pixel using origin [x,y]=[0,0]=bottom left
        bottom_left = [origin_px[1], shape_array.shape[0] - origin_px[0]]

        # Create an Agent.Shape identical to SimulationShape. This is the same shape, but only allows the
        # agent access to the scaled_shape and the coordinates of the bottom left pixel.
        self.target_shape = Agent.Shape(scaled_shape, bottom_left)

    def run_analytics(self):
        ana_suite = Analytics(self.sim, self.seed_origin)
        ana_suite.run_analytics()

    def run(self):

        # Setup the simulation
        self.sim = SimulationEngine(
            "Localisation Test", 800, analytics_func=self.run_analytics
        )
        # origin of the seed agents
        self.seed_origin = [0.3 * self.sim.window_size, 0.6 * self.sim.window_size]

        # The size of the shape as a proportion of the total area of the screen.
        self.shape_area_proportion = 0.05

        self.place_shape("sheep.bmp")
        self.place_agents()

        self.sim.run()


if __name__ == "__main__":
    swarm_sim = Test()
    swarm_sim.run()
