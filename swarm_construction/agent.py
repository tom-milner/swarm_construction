from .simulator.object import SimulationObject
from .simulator.colors import Color


class Agent(SimulationObject):
    # Define the properties of this type of agent.
    radius = 10
    color = (255, 255, 255)
    speed = 0

    def __init__(self, sim_engine, start_pos):
        # Purposefully keeping the input params minimal, so that all instances of this
        # agent class are exactly the same - trying to keep it faithful to the paper!
        super().__init__(sim_engine, start_pos, speed=self.speed, radius=self.radius)

    def follow_edges(self, neighbours):
        closest = neighbours[0]
        collision = self.check_collision(closest[0])
        if not collision[0]:
            # If we're not touching anything, do nothing.
            return
        self.set_orbit_object(closest[0])

    def update(self, fps):
        super().update(fps)

        # If we're stationary, do nothing.
        if self.speed == 0:
            self.color = Color.light_green
            return

        # This is where we implement the agent rules!
        neighbours = self.get_nearest_neighbours(3)

        # Edge Following.
        self.follow_edges(neighbours)
