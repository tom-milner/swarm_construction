import pygame as pg
import math
import numpy as np
from agent import Agent


class Game:

    def generate_seeds(self, line_spacing, agent_radius, origin):
        dx = agent_radius
        dy = line_spacing

        # The seed configuration shape is taken from the paper.
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
        seed_color = (218, 247, 166)

        # Add the seeds to the game.
        self.agents.extend(
            [
                Agent(self.surface, pos, agent_radius, color=seed_color)
                for pos in seed_pos
            ]
        )
        return seed_pos

    def generate_connecting_agents(self, line_spacing, agent_radius, origin_agent):
        # Two agents underneath the provided origin agent.

        # Place the connector agents underneath the origin agent.
        dx = agent_radius
        dy = line_spacing
        conn_deltas = np.array([[-dx, dy], [dx, dy]])
        conn_pos = np.add(conn_deltas, origin_agent)

        # Generate the connecting agents and add them to the game.
        self.agents.extend([Agent(self.surface, pos, agent_radius) for pos in conn_pos])

        return conn_pos

    def generate_cluster_agents(self, line_spacing, agent_radius, origin_agent):
        # Attach rest of the agents in an square - shape shouldn't matter!

        # Calculate the size of the square.
        num_remaining_agents = self.num_agents - len(self.agents)
        side_len = math.ceil(math.sqrt(num_remaining_agents))

        # Place the cluster underneath, and nudged left, of the origin agent.
        cluster_start = np.add(origin_agent, [-agent_radius, line_spacing])

        # Generate the agents in the cluster.
        for i in range(side_len):
            # Every other row is nudged forwards, so the circles fit snuggly.
            x_offset = 0 if i % 2 == 0 else agent_radius

            for j in range(side_len):
                # If all the agents have been generated, we're done.
                if num_remaining_agents <= 0:
                    break

                # Place this agent.
                pos = [x_offset + agent_radius * 2 * j, line_spacing * i]
                pos = np.add(pos, cluster_start)
                self.agents.append(Agent(self.surface, pos, agent_radius))

                num_remaining_agents -= 1

    def generate_agents(self):

        # Agents will take up 0.1 of the area of the screen.
        window_area = self.window_size * self.window_size
        total_agent_area = window_area * 0.1
        agent_area = total_agent_area / self.num_agents

        # Calculate the radius of each agent.
        agent_radius = math.floor(math.sqrt(agent_area / math.pi))

        # Where to start putting the agents in the window.
        seed_origin = [0.2 * self.window_size, 0.5 * self.window_size]

        # Because our agents are circular, we can make them fit snuggly together if we
        # slot each row of agents into the gaps between agents in the previous row.
        # line_spacing is the y spacing between rows to make them all snuggly.
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * agent_radius))

        # Generate the seed agents.
        seeds = self.generate_seeds(line_spacing, agent_radius, seed_origin)

        #  Generate the connector agents at the bottom of the seed.
        bottom_of_seeds = seeds[np.argmax(seeds, axis=0)[1]]
        connectors = self.generate_connecting_agents(
            line_spacing, agent_radius, bottom_of_seeds
        )

        # Generate the cluster of agents at the left connector.
        left_conn = connectors[np.argmin(connectors, axis=0)[1]]
        self.generate_cluster_agents(line_spacing, agent_radius, left_conn)

    def __init__(self):
        self.title = "Swarm Construction"
        self.window_size = 800
        self.num_agents = 100

        # Initialise pygame.
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.set_caption(self.title)

        # Initialise Game.
        self.surface = pg.display.set_mode((self.window_size, self.window_size))
        self.running = True
        self.agents = []

        # Generate the agents (robots!).
        self.generate_agents()

    def main(self):
        # Main game loop.
        while self.running:
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

        # Update agents.
        for player in self.agents:
            player.update()

    def draw(self):

        # Draw background.
        self.surface.fill((0, 0, 0))

        # Draw agents.
        for player in self.agents:
            player.draw()

        pg.display.update()


def main():
    swarm_game = Game()
    swarm_game.main()


if __name__ == "__main__":
    main()
