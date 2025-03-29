from .simulator.object import SimulationObject
from .simulator.colors import Color
from .simulator.engine import SimulationEngine


class Agent(SimulationObject):
    """A simulation of a swarm agent (robot).

    Physics is provided by an underlying SimulationObject class.
    """

    radius: int = 10
    color: Color = Color.white
    speed: int = 0

    def __init__(self, sim_engine: SimulationEngine, start_pos, swarm_pos=None):
        """Initialise the agent.
        Purposefully keeping the input params minimal, so that all instances of this agent class are similar - trying to keep it faithful to the paper!

        Args:
            sim_engine (SimulationEngine): _description_
            start_pos ([int,int]): The starting position of the agent in the simulation.
            swarm_pos ([int,int], optional): The position of the agent in the swarm. This is used to provide seed agents with their initial positions. Defaults to None.
        """

        # If we've provided this agent with a position in the swarm, it is a seed robot.
        if swarm_pos is not None:
            # Seed robots are stationary and green.
            self.speed = 0
            self.color = Color.light_green

        # Initialise the underlying simulation object.
        super().__init__(
            sim_engine,
            start_pos,
            speed=self.speed,
            radius=self.radius,
            color=self.color,
        )

        # Initialise agent-specific variables.
        self.swarm_pos = swarm_pos

    def follow_edges(self, neighbours):
        """Move round the edges of the provided neighbours.
        Works by orbiting an agent until it collides with another, upon which it orbits the collision agent.

        Args:
            neighbours (list): The nearby agents to follow the edges of, sorted by proximity. (This can be provided by SimulationObject.get_nearest_neighbours)
        """

        # Check if we've collided with our nearest neighbour.
        closest = neighbours[0]
        collision = self.check_collision(closest[0])
        if not collision[0]:
            # If we're not touching anything, do nothing.
            return

        # If we're touching an agent, orbit around it.
        self.set_orbit_object(closest[0])

    def update(self, fps):
        """Update the agents state each frame. This is where the rules are implemented.

        Args:
            fps (float): FPS of the last frame (provided by pygame).
        """
        # Update the underlying SimulationObject.
        super().update(fps)

        # If we're stationary, do nothing.
        if self.speed == 0:
            return

        # ====== AGENT RULES ======
        neighbours = self.get_nearest_neighbours(3)

        # Rule 1: Edge Following.
        self.follow_edges(neighbours)
