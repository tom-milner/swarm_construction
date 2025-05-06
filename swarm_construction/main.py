from .simulator.engine import SimulationEngine
from .simulator.colors import Colour
from .agent import Agent
from .simulator.shape import SimulationShape
from .simulator.analytics import Analytics

import math
import numpy as np
from PIL import Image
import cv2
import random


class SwarmConstructionSimulation:
    """The entry point for the shape-constructing-swarm simulation."""

    def calculate_agent_radius(self, window_size, num_agents):
        """Calculate how big we can make each agent given our current window size.

        Args:
            window_size (int): Length of each side of the (square) window.
            num_agents (int): Number of agents to fit in the window.

        Returns:
            int: Calculated agent radius.
        """

        window_area = window_size**2
        total_agent_area = window_area * (self.shape_area_proportion + 0.005)
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
                # [-dx * 2, dy],
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
            self.agents.append(
                Agent(
                    self.sim,
                    seed_pos[i],
                    local_pos=local_pos[i],
                    shape=self.target_shape,
                    mode=self.mode,
                    gradient=0 if i == 0 else 1,
                )
            )
        self.agents[-1].is_bottom_seed = True

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
        conn_deltas = np.array([[-dx, dy], [dx, dy], [0, 2 * dy], [-2 * dx, 2 * dy]])
        conn_pos = np.add(conn_deltas, origin_agent)

        # Generate the connecting agents and add them to the simulation.
        self.agents.extend(
            [
                Agent(
                    self.sim,
                    pos,
                    color=Colour.white,
                    shape=self.target_shape,
                    mode=self.mode,
                )
                for pos in conn_pos
            ]
        )

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
        aspect_ratio = 4 / 3
        for i in range(round(side_len / aspect_ratio)):
            # Every other row is nudged forwards, so the circles fit snuggly.
            x_offset = 0 if i % 2 == 0 else dx

            for j in range(round(side_len * aspect_ratio)):

                # If all the agents have been generated, we're done.
                if num_agents <= 0:
                    return

                # Place this agent in the cluster square.
                pos = [x_offset + dx * 2 * j, dy * i]
                pos = np.add(pos, cluster_start)

                # randomly make roughly the right proportion of agents each colour
                if random.random() < self.p_white_agents:
                    color = Colour.white
                else:
                    color = Colour.grey

                # Generate the agent.
                self.agents.append(
                    Agent(self.sim, pos, color, shape=self.target_shape, mode=self.mode)
                )
                num_agents -= 1

    def place_agents(self, num_agents):
        """Generate and place agents in the simulation.

        Args:
            num_agents (int): Number of agents to generate and place.
        """
        # Calculate how big we can make the agents given our current window size.
        Agent.radius = self.calculate_agent_radius(self.sim.window_size, num_agents)

        # Because our agents are circular, we can make them fit snuggly together if we
        # slot each row of agents into the gaps between agents in the previous row.
        # line_spacing is the y spacing between rows to make them all snuggly, and
        # column_spacing is the x spacing between agents across a pair of snuggly rows.
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * Agent.radius))
        column_spacing = Agent.radius

        # Generate the seed agents around the origin.
        seed_pos = self.generate_seeds(line_spacing, column_spacing, self.seed_origin)
        num_agents -= len(seed_pos)

        #  Generate the connector agents at the bottom of the seed.
        bottom_of_seeds = seed_pos[np.argmax(seed_pos, axis=0)[1]]
        connectors = self.generate_connecting_agents(
            line_spacing, column_spacing, bottom_of_seeds
        )
        num_agents -= len(connectors)

        # Generate the cluster of agents underneath the bottom left connector agent.
        left_conn = connectors[-1]
        self.generate_cluster_agents(
            num_agents, line_spacing, column_spacing, left_conn
        )

    def place_shape(self, shape_file):

        window_area = self.sim.window_size**2
        goal_shape_area = window_area * self.shape_area_proportion

        # this whole thing scales the inputted shape file to match the robot area
        self.sim.shape_name = shape_file
        shape = Image.open(shape_file)
        init_shape_area = sum(pixel != 0 for pixel in shape.getdata())

        scale_factor = math.sqrt(goal_shape_area / init_shape_area)
        scaled_shape = shape.resize(
            (int(shape.width * scale_factor), int(shape.height * scale_factor)),
            Image.NEAREST,
        )
        # This flips the shape, may need to be removed if we change how we generate shapes
        # NOTE not required when using shape_create_gui to make bmp file!
        # scaled_shape = scaled_shape.transpose(Image.FLIP_TOP_BOTTOM)
        # scaled_shape.save("scaled_shape_test.bmp")

        # converts image to numpy array
        shape_array = np.array(scaled_shape)

        # counts how many of each colour pixel there are
        # then works out the proportion of white pixels compared to grey
        # can be expanded if we want more colours
        # this proportion is used to make sure we have the right amount of agents
        # of each colour
        colour_count = np.array(np.unique(shape_array, return_counts=True)).T
        colour_count = colour_count[~np.any(colour_count == 0, axis=1)]
        white_loc_row, white_loc_col = np.where(colour_count == 255)
        self.p_white_agents = colour_count[white_loc_row, 1] / np.sum(
            colour_count[:, 1]
        )

        if self.p_white_agents == 1:
            self.mode = "monochrome"
        else:
            self.mode = "greyscale"

        # finds the bottom left most not black pixel in the scaled shape
        # in the array this is actually max y (row) with min x (col)
        # this finds the indices of all not black pixels
        white_px_index = np.argwhere(shape_array != 0)

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

        # find the islands in the shape
        centroids = self.get_islands(scaled_shape)
        local_frame_centroids = [
            [COM[0] - bottom_left[0], shape_array.shape[0] - bottom_left[1] - COM[1]]
            for COM in centroids
        ]

        print(centroids)
        print(bottom_left)
        print(local_frame_centroids)

        # Create an Agent.Shape identical to SimulationShape. This is the same shape, but only allows the
        # agent access to the scaled_shape and the coordinates of the bottom left pixel.
        self.target_shape = Agent.Shape(
            scaled_shape, bottom_left, local_frame_centroids
        )

    def run_analytics(self):
        ana_suite = Analytics(self.sim, self.seed_origin)
        # Pass in True for figures to save as .eps
        # False for no saving
        ana_suite.run_analytics(False)

    def get_islands(self, bmp_shape):
        """
        Finds white pixel clusters and returns their centroids.
        """
        # Convert PIL image to grayscale NumPy array
        img = np.array(bmp_shape.convert("L"))

        # Threshold to binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels_im = cv2.connectedComponents(binary)

        centroids = []
        for label in range(1, num_labels):  # skip label 0 (background)
            mask = labels_im == label
            ys, xs = np.where(mask)
            x_center = np.mean(xs)
            y_center = np.mean(ys)
            centroids.append((x_center, y_center))

        # ======== debug =================
        # Convert grayscale to BGR so we can draw colored circles
        # Convert PIL image to grayscale NumPy array
        # img = np.array(bmp_shape.convert("L"))
        # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Draw red circles at centroids
        # for (x, y) in centroids:
        #    cv2.circle(img_color, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        # cv2.imshow("Centroids on Scaled Shape", img_color)
        # print("Press any key in the image window to continue...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return centroids

    def main(self):

        # For each pixel an agent travels, we update the sim twice.
        update_rate = Agent.start_speed * 4

        self.sim = SimulationEngine(
            "Swarm Construction",
            800,
            draw_rate=30,
            update_rate=update_rate,
            analytics_func=self.run_analytics,
        )
        self.agents = []

        # origin of the seed agents
        self.seed_origin = [0.25 * self.sim.window_size, 0.55 * self.sim.window_size]

        # The size of the shape as a proportion of the total area of the screen.
        self.shape_area_proportion = 0.13

        self.place_shape("swarm_construction\shape\wrench.bmp")
        self.place_agents(300)

        self.sim.run()


if __name__ == "__main__":
    swarm_sim = SwarmConstructionSimulation()
    swarm_sim.main()
