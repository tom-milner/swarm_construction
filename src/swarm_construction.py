import pygame as pg
import math
import numpy as np

TITLE = "Swarm Construction"
WINDOW_SIZE = 800
NUM_AGENTS = 100


class Agent:

    def __init__(self, surface, pos, radius=30, color=(255, 255, 255)):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color

    def update(self):
        pass

    def draw(self):
        # Circles are drawn around the center.
        pg.draw.circle(self.surface, self.color, self.pos, self.radius)


class Game:
    def __init__(self):
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.set_caption(TITLE)
        self.surface = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.loop = True
        self.players = []

        # Setup agents
        # Agents will take up half the area of the screen.
        window_area = WINDOW_SIZE * WINDOW_SIZE
        total_agent_area = window_area * 0.1
        agent_area = total_agent_area / NUM_AGENTS
        # area=pi*r^2 -> r = sqrt(area/pi)
        agent_radius = math.floor(math.sqrt(agent_area / math.pi))

        # Generate Seeds.
        origin = np.array([0.2 * WINDOW_SIZE, 0.5 * WINDOW_SIZE])
        dy = math.fabs(round(math.tan(2 * math.pi / 3) * agent_radius))
        dx = agent_radius
        seed_deltas = np.array(
            [
                [-dx, 0],
                [dx, 0],
                [0, -dy],
                [0, dy],
                # [-dx, 0],
                # [dx, 0],
                # [2 * agent_radius + dx, 0],
                # [4 * agent_radius + dx, 0],
                # [6 * agent_radius + dx, 0],
                # [8 * agent_radius, dy],
            ]
        )
        seed_pos = np.add(origin, seed_deltas)

        # Generate the seed agents and simultaneously add them to the game.
        seed_color = (218, 247, 166)
        self.players.extend(
            [Agent(self.surface, pos, agent_radius, seed_color) for pos in seed_pos]
        )

        # Two agents connect to the bottom of the seed - this isnt' necessary, I just wanted it to look like it does in the paper.

        # Get the seed agent with the largest y-value (the bottom agent).
        bottom_of_seed = seed_pos[np.argmax(seed_pos, axis=0)[1]]

        # The vectors of the connecting agents to the bottom seed.
        conn_deltas = np.array([[-dx, dy], [dx, dy]])
        conn_pos = np.add(conn_deltas, bottom_of_seed)

        # Generate the connecting agents and add them to the game.
        self.players.extend(
            [Agent(self.surface, pos, agent_radius) for pos in conn_pos]
        )

        # Attach rest of the agents in an square - shape shouldn't matter!
        num_remaining_agents = NUM_AGENTS - len(seed_pos) - len(conn_pos)
        side_len = math.ceil(math.sqrt(num_remaining_agents))
        left_conn = conn_pos[np.argmin(conn_pos, axis=0)[1]]
        cluster_start = np.add(left_conn, [-agent_radius, dy])
        for i in range(side_len):
            x_offset = 0 if i % 2 == 0 else dx
            for j in range(side_len):
                if num_remaining_agents <= 0:
                    break
                pos = np.array([x_offset + agent_radius * 2 * j, dy * i])
                pos = np.add(pos, cluster_start)
                self.players.append(Agent(self.surface, pos, agent_radius))
                num_remaining_agents -= 1

    def main(self):
        while self.loop:
            self.update()
            self.draw()

        pg.quit()

    def update(self):
        # Handle events.
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.loop = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.loop = False

        # Update players.
        for player in self.players:
            player.update()

    def draw(self):

        # Draw background.
        self.surface.fill((0, 0, 0))

        # Draw players.
        for player in self.players:
            player.draw()

        pg.display.update()


def main():
    swarm_game = Game()
    swarm_game.main()


if __name__ == "__main__":
    main()
