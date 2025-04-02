from .simulator.object import SimulationObject
from .simulator.colors import Color
from .simulator.engine import SimulationEngine
import numpy as np


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
        self.seed_robot = False
        if swarm_pos is not None:
            # Seed robots are stationary and green.
            self.speed = 0
            self.color = Color.light_green
            self.seed_robot = True

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
            neighbours (list(tuple)): The nearby agents to follow the edges of, and their distances, sorted by proximity. (This can be provided by SimulationObject.get_nearest_neighbours)
        """

        if self.seed_robot:
            return

        if self.speed == 0:
            return

        # Check if we've collided with our nearest neighbour.
        closest = neighbours[0]
        collision = self.check_collision(closest[0])
        if not collision[0]:
            # If we're not touching anything, do nothing.
            return

        # Orbit around the new neighbour.
        self.set_orbit_object(closest[0])

    def localise(self, neighbours):
        """Localise ourselves using the surrounding agents. Sets swarm_pos to be our calculated location.

        Args:
            neighbours (list(tuple)): List of tuples (neighbouring agent, distance).
        """

        # Don't allow seed robots to localise.
        if self.seed_robot:
            return

        # No need to re-localise if we're stationary.
        # In reality, robots will need to periodically re-localise, as they may be bumped around.
        if self.speed == 0:
            return

        # If our neighbours aren't localised, we can't localise ourselves.
        neighbours_are_localised = np.all(
            [n[0].swarm_pos is not None for n in neighbours]
        )
        if not neighbours_are_localised:
            self.swarm_pos = None
            return

        # Set a starting swarm_pos if necessary.
        pos = [0, 0] if self.swarm_pos is None else self.swarm_pos

        # NOTE: This localisation algorithm is a bit rubbish. It's based off the paper, and was designed to manage
        # the constraints they were facing with small, low-power, low-compute robots, and asynchronous comms.

        # Their robots are constantly localising in the background. We only have one frame.
        # Therefore, we just run the algorithm loads of times ('num_minimisations') to compute a good enough minimisation.

        # Perform the "distributed trilateration" algorithm.
        num_minimisations = 1000
        for i in range(num_minimisations):
            for n in neighbours:

                agent = n[0]
                measured_dist = n[1]

                # vector from neighbour swarm_pos to ourselves.
                neighbour_vec = np.subtract(pos, agent.swarm_pos)

                # the distance between our swarm_pos and the neighbours swarm_pos.
                calculated_dist = np.linalg.norm(neighbour_vec)

                # unit vector pointing from neighbour to ourselves.
                v = neighbour_vec / calculated_dist

                # scale unit vector to size of actual distance measured.
                actual_vec = v * measured_dist

                # compute new position of where we should be, based on the measured distance.
                new_pos = np.add(agent.swarm_pos, actual_vec)

                # In the paper, they only move 1/4 of the way to the new position.
                # For our simulation, this just slows down the minimisation, so we're not doing it.

                # move towards new position
                pos_diff = np.subtract(pos, new_pos)
                # pos_diff = np.subtract(pos, new_pos) / 4
                pos = np.subtract(pos, pos_diff)

        # save calculated position
        self.swarm_pos = pos

    def update(self, fps):
        """Update the agents state each frame. This is where the rules are implemented.

        Args:
            fps (float): FPS of the last frame (provided by pygame).
        """

        # If we're stationary, do nothing.
        # NOTE: If this isn't here, every agent finds it nearest neighbours, which atm takes a longggg time.
        if self.speed == 0:
            return

        # Update the underlying SimulationObject.
        super().update(fps)

        # Get 3 closest neighbours.
        neighbours = self.get_nearest_neighbours(3)

        # ====== AGENT RULES ======
        self.follow_edges(neighbours)
        self.localise(neighbours)
