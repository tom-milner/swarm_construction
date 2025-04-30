from .simulator.object import SimulationObject
from .simulator.colors import Color
from .simulator.engine import SimulationEngine
import numpy as np
import random


class Agent(SimulationObject):
    """A simulation of a swarm agent (robot).

    Physics is provided by an underlying SimulationObject class.
    """

    radius: int = 10
    color: Color = Color.white
    speed: int = 0

    class Shape:
        """This is the agents internal shape representation, used in shape assembly"""

        def __init__(self, shape, bottom_left, centre_of_masses):
            self.shape_data = shape
            self.bottom_left = bottom_left
            self.centre_of_masses = centre_of_masses

    def __init__(
        self,
        sim_engine: SimulationEngine,
        start_pos,
        local_pos=None,
        shape: Shape = None,
    ):
        """Initialise the agent.
        Purposefully keeping the input params minimal, so that all instances of this agent class are similar - trying to keep it faithful to the paper!

        Args:
            sim_engine (SimulationEngine): _description_
            start_pos ([int,int]): The starting position of the agent in the simulation.
            local_pos ([int,int], optional): The position of the agent in the swarm. This is used to provide seed agents with their initial positions. Defaults to None.
        """

        # If we've provided this agent with a position in the swarm, it is a seed robot.
        self.seed_robot = False
        if local_pos is not None:
            # Seed robots are stationary and green and have gradient of 0.
            self.speed = 0
            self.color = Color.light_green
            self.gradient = 0
            self.seed_robot = True
        else:
            # we dont know what the gradient is yet.
            self.gradient = None

        # Initialise the underlying simulation object.
        super().__init__(
            sim_engine,
            start_pos,
            speed=self.speed,
            radius=self.radius,
            color=self.color,
            label=self.gradient,
        )

        # Initialise agent-specific variables.
        self.local_pos = local_pos
        self.shape = shape
        self.prev_inside_shape = False
        # bridging stuff
        self.looped = 0
        self.bridge = False
        self.looped_updated = False

    def start_edge_following(self, fps):
        """Start edge following by checking for a bunch of conditions

        If the conditions are passed, the agent is set to orbit the neighbour it is
        touching with the highest gradient.
        Args:
            fps (int): the sims current frame rate"""
        if self.seed_robot:
            # Don't allow seed robots to edge follow.
            return
        if self.local_pos is not None:
            # If we're localised, we can't edge follow.
            return
        average_start_time = 1
        p = 1 / (average_start_time * (fps + 1))
        if random.random() > p:
            # if we are unlucky, we dont edge follow
            return
        # get the nearest neighbours
        neighbours = self.get_nearest_neighbours()
        if len(neighbours) == 0:
            return
        # get gradients of neighbours
        gradients = [neighbour[0].gradient for neighbour in neighbours]
        speeds = [neighbour[0].speed for neighbour in neighbours]
        if any(
            gradient is not None and gradient > self.gradient for gradient in gradients
        ):
            # there are neighbours with a higher gradient than us
            return
        if any(speeds > 0 for speeds in speeds):
            # there are neighbours moving
            return

        equal_grad_neighbours = 0
        max_grad_neighbour = neighbours[0]
        for neighbour in neighbours:
            # counts neighbours within touching distance with the same gradient as the agent
            if (neighbour[1] <= Agent.radius * 2) and (
                neighbour[0].gradient == self.gradient
            ):
                equal_grad_neighbours += 1

            # find the neighbour with the maximum gradient
            if (neighbour[1] <= Agent.radius * 2) and (
                neighbour[0].gradient > max_grad_neighbour[0].gradient
            ):
                max_grad_neighbour = neighbour

        # if we're touching 2 neighbours with equal gradient to ourselves
        # it is likely that we will get trapped between them in a cycle
        # of collisions and changing orbit objects if we tried to start edge following
        if equal_grad_neighbours > 1:
            return

        # This stops the top right agent flying away - it never collided with other agents
        # so never had an agent to orbit
        # We manually assign the initial orbit object as the neighbour with the highest
        # gradient that is stationary
        if max_grad_neighbour[0].speed == 0:
            self.set_orbit_object(max_grad_neighbour[0])
        self.speed = 100

    def follow_edges(self, neighbours):
        """Move round the edges of the provided neighbours.
        Works by orbiting an agent until it collides with another, upon which it orbits the collision agent.
        Requires the agent to be moving.
        Args:
            neighbours (list(tuple)): List of tuples of neighbours and their distances (neighbour, distance) sorted by distance.
        """

        if self.seed_robot:
            return

        if self.speed == 0:
            return

        if len(neighbours) == 0:
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
        """Localise ourselves using the surrounding agents. Sets local_pos to be our calculated location.

        Args:
            neighbours (list(tuple)): List of tuples of nearest neighbours (neighbouring agent, distance).
        """

        # Don't allow seed robots to relocalise.
        if self.seed_robot:
            return

        # No need to re-localise if we're stationary.
        # In reality, robots will need to periodically re-localise, as they may be bumped around.
        if self.speed == 0:
            return

        # Get the neighbours that are already localised.
        localised_neighbours = [n for n in neighbours if n[0].local_pos is not None]

        # If we have < 3 localised neighbours, we can't localise ourselves
        if len(localised_neighbours) < 3:
            self.local_pos = None
            return

        # Set a starting local_pos if necessary.
        pos = [-self.radius, -self.radius] if self.local_pos is None else self.local_pos

        # NOTE: This localisation algorithm is a bit rubbish. It's based off the paper, and was designed to manage
        # the constraints they were facing with small, low-power, low-compute robots, and asynchronous comms.

        # Their robots are constantly localising in the background. We only have one frame.
        # Therefore, we just run the algorithm loads of times ('num_minimisations') to compute a good enough minimisation.

        # Perform the "distributed trilateration" algorithm.
        num_minimisations = 10
        for i in range(num_minimisations):
            for n in localised_neighbours:
                agent = n[0]
                measured_dist = n[1]

                # vector from neighbour local_pos to ourselves.
                neighbour_vec = np.subtract(pos, agent.local_pos)

                # the distance between our local_pos and the neighbours local_pos.
                calculated_dist = np.linalg.norm(neighbour_vec)

                # unit vector pointing from neighbour to ourselves.
                v = neighbour_vec / calculated_dist

                # scale unit vector to size of actual distance measured.
                actual_vec = v * measured_dist

                # compute new position of where we should be, based on the measured distance.
                new_pos = np.add(agent.local_pos, actual_vec)

                # In the paper, they only move 1/4 of the way to the new position.
                # For our simulation, this just slows down the minimisation, so we're not doing it.

                # move towards new position
                # pos_diff = np.subtract(pos, new_pos) / 4
                pos_diff = np.subtract(pos, new_pos)

                pos = np.subtract(pos, pos_diff)

        # save calculated position
        self.local_pos = pos
        # print(self.local_pos)
        pass

    def is_inside_shape(self) -> bool:
        """Check if our localised position (local_pos) is inside the shape (target_shape)

        Returns:
            bool: Whether we're in the shape (True) or not (False).
        """
        if self.shape is None:
            return

        # Default color if we're not inside the shape
        self.color = Color.white

        # Can only see if we're inside the shape if we're localised.
        if self.local_pos is None:
            return False

        # Map the localised swarm position to the shape, using the bottom left pixel in the shape.
        # They both use origin (x,y) = (0,0) = bottom left (for now...)
        size = self.shape.shape_data.size
        mapped = np.add(self.local_pos, self.shape.bottom_left).round()
        # print(mapped) # Saved my life during debugging so leaving in as a reminder of what went down here.

        # If we're not in the shapes bounding box, return early.
        if not (0 < mapped[0] < size[0] - 1) or not (0 < mapped[1] < size[1] - 1):
            return False

        # Yellow means we're in the bounding box!
        self.color = Color.yellow

        # Get the raw pixels from the shape, and get the color of the pixel at our mapped position.
        pixels = self.shape.shape_data.getdata()

        # Calculate 1D index from 2D mapped position.
        # Pixel origin (0,0) is top left damnit! Flip our mapped position y-axis to deal with this.
        rows = int(size[0] * (size[1] - mapped[1]))
        image_idx = int(rows + mapped[0])

        if pixels[image_idx] != 255:
            # We are not in the shape :(
            return False

        # Woohoo! We're in the shape! Turn orange to celebrate.
        self.color = Color.orange
        return True

    def update_gradient(self, neighbours):
        """updates the gradient of the agent
        Works by finding lowest gradient of neighbours and adds 1
        Args:
            neighbours (list(tuple)): List of tuples (neighbouring agent, distance)
        """

        # we only use the closest neighbours (ones we are touching or almost touching)
        neighbours = [n for n in neighbours if n[1] <= Agent.radius * 2.1]

        # now lets get the gradients
        gradients = [neighbour[0].gradient for neighbour in neighbours]

        valid_gradients = []
        # itterate thro and get all valid gradients
        for i in gradients:
            # make sure they have a set gradient
            if i is None:
                continue
            valid_gradients.append(i)

        # check we have not got an empty list
        if valid_gradients:
            self.gradient = 1000
            lowest_gradient = int(np.array(valid_gradients).min())
            if self.gradient is None:
                # set to lowest + 1
                self.gradient = lowest_gradient + 1
            elif self.gradient > lowest_gradient + 1:
                self.gradient = lowest_gradient + 1

            # update the label on object
            self.label = self.gradient

    def assemble_shape(self):

        # Can't assemble the shape if we're not localised.
        if self.local_pos is None:
            return

        # Stopping Conditions.

        # Condition 1: We are about to edge follow around a stationary robot that has a larger gradient value than our own.
        orb_agent = self.get_orbit_object()
        if orb_agent is None:
            return

        inside_shape = self.is_inside_shape()

        if (
            self.prev_inside_shape
            and inside_shape
            and orb_agent.speed == 0
            and self.gradient <= orb_agent.gradient
        ):
            self.speed = 0

        # Condition 2: We are about to exit the shape.
        if not inside_shape and self.prev_inside_shape:
            self.speed = 0

        self.prev_inside_shape = inside_shape
    
    def check_bridging(self, tolerance = 5):
        """
        Checks if self.local_pos lies within a given perpendicular distance (tolerance)
        from any line segment connecting two COMs, and that the closest point lies
        between the two COMs (not extended beyond them).

        Args:
            tolerance (float): Maximum allowed perpendicular distance to consider a bridge.

        Returns:
            bool: True if bridging condition is met, else False.
        """
        # arrays of current position and COMs
        COMs = np.array(self.shape.centre_of_masses, dtype=np.float64)
        pos = np.array(self.local_pos, dtype=np.float64)
        # if there isnt another one, we cant bridge.
        if len(COMs) < 2:
            return None
        # for now lets just consider the closest COM and find all the vectors to other ones
        deltas = COMs - pos
        distances = np.linalg.norm(deltas, axis=1)
        closest_index = np.argmin(distances)

        # go thro and check the other COMs
        for i in range(len(COMs)):
            A = COMs[i]
            B = COMs[closest_index]
            AB = B - A

            # Skip if the same COMs
            AB_len_sq = np.dot(AB, AB)
            if AB_len_sq == 0:
                continue  
            
            # project the point onto the segment
            AP = pos - A
            t = np.dot(AP, AB) / AB_len_sq

            # Check if projection lies within the segment
            if 0 <= t <= 1:  
                closest_point = A + t * AB
                # get perpendicular distance, return if in tolerance
                dist = np.linalg.norm(pos - closest_point)
                if dist <= tolerance:
                    return COMs[i]

        return None
        
        """    def check_bridging(self):
                tolerance = 0.5
                COMs = np.array(self.shape.centre_of_masses, dtype=np.float64)
                # get the closest COM
                distances_sq = np.sum((COMs - self.local_pos) ** 2, axis=1)
                min_idx = np.argmin(distances_sq)
                # Get the closest COM position
                curent_COM = COMs[min_idx]

                # get possible vectors for bridges to be built current COM -> other COM
                v = COMs - curent_COM
                u = COMs - self.local_pos

                # vector maths
                cross = np.cross(v, u)
                colinear = np.abs(cross) < tolerance
                dot = np.sum(v * u, axis=1)
                dist_v = np.linalg.norm(v, axis=1)
                dist_u = np.linalg.norm(u, axis=1)

                # local pos must be between the two COMs and on the line between them
                between  = (dot >= 0) & (dist_u <= dist_v)
                mask = colinear & between
                # get the indices of the matching COMs
                matching_indices = np.where(mask)[0]
                if len(matching_indices) > 0:
                    return True
                return False"""

    def update_bridge(self):
        """Bridge to an island if this one is already full!
        Assumptions: closest COM is the shape we currently belong to."""
        
        # ain't localised, dont bridge
        if self.local_pos is None:
            return
        # cheating, maybe? Don't bridge if we are in a new shape
        if self.color is Color.orange:
            return
        # check if we have gone round -  there is no more space
        if self.looped < 2:
            self.check_if_looped()
            # no need to bridge yet
            return
        # We can bridge, lets check if possible
        desired_COM = self.check_bridging(10)
        if desired_COM is not None:
            # below this distance, we encourage bridging. Above, we discourage bridging
            ideal_COM_distance = 30
            # what is the average probability we should bridge
            nominal_bridging_probability = 0.2
            # calculate
            p_bridging = np.clip((ideal_COM_distance / np.linalg.norm(desired_COM)) * nominal_bridging_probability * (self.looped-1))
            # print(f"probability bridging at {self.local_pos}, probility: {p_bridging}")
            if random.random() < p_bridging:
                # print(f"we have bridged at local position {self.local_pos}, gobally at:{self._pos}")
                # we are bridging, stop
                self.color = Color.red
                self.speed = 0
        return

    def check_if_looped(self):
        """Check if the agent has looped around the shape.
        Returns:
            int: The number of times the agent has passed the negitive x axis.
        """
        # if we are directly left to seeds
        if self.local_pos[1]**2 < 0.5 and self.local_pos[0] < 0:
            # only trigger once
            if not self.looped_updated:
                self.looped += 1
                self.looped_updated = True
        else:
            self.looped_updated = False
        return
    
    def update(self, fps):
        """Update the agents state each frame. This is where the rules are implemented.

        Args:
            fps (float): FPS of the last frame (provided by pygame).
        """

        # NOTE: If this isn't here, every agent finds it nearest neighbours, which atm takes a longggg time.
        if self.speed == 0:
            # if we are briding, we dont need to do anything
            if self.bridge:
                return
            if self.gradient is None:
                # we are at start of sim, need to initalise gradients
                neighbours = self.get_nearest_neighbours()
                self.update_gradient(neighbours)
                return
            # We might need to start edge following. Better check.
            self.start_edge_following(fps)
            return

        # Update the underlying SimulationObject.
        super().update(fps)

        # Get closest neighbours.
        neighbours = self.get_nearest_neighbours()

        # ====== AGENT RULES ======
        self.follow_edges(neighbours)
        self.localise(neighbours)
        self.update_gradient(neighbours)
        self.assemble_shape()
        self.update_bridge()
