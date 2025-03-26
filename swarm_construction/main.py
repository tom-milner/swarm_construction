from .simulation_engine import Simulation
from .agent import Agent
import math

import numpy as np


class SwarmConstructionSimulation:
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

        # Add the seeds to the simulation.
        self.agents.extend(
            [
                Agent(self.sim.surface, pos, agent_radius, color=seed_color)
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

        # Generate the connecting agents and add them to the simulation.
        self.agents.extend(
            [Agent(self.sim.surface, pos, agent_radius) for pos in conn_pos]
        )

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
                self.agents.append(Agent(self.sim.surface, pos, agent_radius))

                num_remaining_agents -= 1

    def generate_agents(self):

        # Agents will take up 0.1 of the area of the screen.
        window_area = self.sim.window_size * self.sim.window_size
        total_agent_area = window_area * 0.1
        agent_area = total_agent_area / self.num_agents

        # Calculate the radius of each agent.
        agent_radius = math.floor(math.sqrt(agent_area / math.pi))

        # Where to start putting the agents in the window.
        seed_origin = [0.2 * self.sim.window_size, 0.5 * self.sim.window_size]

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

    def update(self, fps):
        for ag in self.agents:
            ag.update(fps)

    def draw(self):
        for ag in self.agents:
            ag.draw()

    def main(self):
        self.num_agents = 100
        self.agents = []

        self.sim = Simulation("Swarm Construction", 800)

        # Generate the agents (robots!).
        self.generate_agents()

        self.sim.add_draw(self.draw)
        self.sim.add_update(self.update)

        self.sim.run()


if __name__ == "__main__":
    swarm_sim = SwarmConstructionSimulation()
    swarm_sim.main()
