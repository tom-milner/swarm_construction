from .simulator.object import SimulationObject
from .simulator.colors import Colour
from .simulator.engine import SimulationEngine
import numpy as np
import random
from enum import Enum


class AgentState(Enum):
    IDLE = (1,)
    MOVING_AROUND_CLUSTER = (2,)
    MOVING_OUTSIDE_SHAPE = (3,)
    MOVING_INSIDE_SHAPE = (4,)
    LOCALISED = 5,


class Agent(SimulationObject):
    """A simulation of a swarm agent (robot).

    Physics is provided by an underlying SimulationObject class.
    """

    # Defaults
    radius: int = 10
    speed: int = 0
    start_speed = 100
    average_start_time = 1

    class Shape:
        """This is the agents internal shape representation, used in shape assembly"""

        def __init__(self, shape, bottom_left):
            self.shape_data = shape
            self.bottom_left = bottom_left

    def __init__(
        self,
        sim_engine: SimulationEngine,
        start_pos,
        color = Colour.white,
        local_pos=None,
        shape: Shape = None,
        mode='monochrome',
        gradient=None
    ):
        """Initialise the agent.
        Purposefully keeping the input params minimal, so that all instances of this agent class are similar - trying to keep it faithful to the paper!

        Args:
            sim_engine (SimulationEngine): _description_
            start_pos ([int,int]): The starting position of the agent in the simulation.
            local_pos ([int,int], optional): The position of the agent in the swarm. This is used to provide seed agents with their initial positions. Defaults to None.
        """

        self.color = Colour.white
        self.gradient = gradient
        self.speed = 0
        self.local_pos = local_pos
        self.shape = shape
        self.is_seed = False
        self.state = AgentState.IDLE
        self.mode = mode

        if local_pos is not None:
            # Seed robots are stationary and green and have gradient of 0.
            self.state = AgentState.LOCALISED
            self.color = Colour.light_green
            self.is_seed = True
        else:
            self.color = color
            # this colour flag is set to the same value as the pixel in the bitmap shape
            # allows for easy checking if we're in our part of the shape or not
            # agents effectively only 'see' their section of the shape
            if color == Colour.grey:
                self.color_flag = 127
            elif color == Colour.white:
                self.color_flag = 255

        # Initialise the underlying simulation object.
        super().__init__(
            sim_engine,
            start_pos,
            radius=self.radius,
            color=self.color,
            label=self.gradient,
        )

    def start_edge_following(self, fps, neighbours):
        """Start edge following by checking for a bunch of conditions

        If the conditions are passed, the agent is set to orbit the neighbour it is
        touching with the highest gradient.
        Args:
            fps (int): the sims current frame rate"""

        # get the nearest non-localised neighbours
        neighbours = [n for n in neighbours if n[0].local_pos is None]

        # If there are no neighbours, start.
        if len(neighbours) == 0:
            return True

        # get gradients of neighbours
        gradients = [neighbour[0].gradient for neighbour in neighbours]
        
        # there are neighbours with a higher gradient than us
        if any(
            gradient is not None and gradient > self.gradient for gradient in gradients
        ):
            return False
        
        # there are neighbours moving
        if any(n[0].speed > 0 and n[1] < Agent.radius * 2 * 4 for n in neighbours):
            
            return False

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
            return False

        if max_grad_neighbour[0].speed == 0:
            self.set_orbit_object(max_grad_neighbour[0])

        return True

    def follow_edges(self, neighbours):
        """Move round the edges of the provided neighbours.
        Works by orbiting an agent until it collides with another, upon which it orbits the collision agent.
        Requires the agent to be moving.
        Args:
            neighbours (list(tuple)): List of tuples of neighbours and their distances (neighbour, distance) sorted by distance.
        """

        if len(neighbours) == 0:
            return

        # Check if we've collided with our nearest neighbour.
        closest = neighbours[0]
        collision = self.check_collision(closest[0])
        if not collision[0]:
            # If we're not touching anything, do nothing.
            return
        
        # If we crash into an agent, stop moving for a bit.
        if closest[0].speed != 0:    
            self.fix_collision(collision)
            return
        

        # Orbit around the new neighbour.
        self.set_orbit_object(closest[0])

    def localise(self, neighbours):
        """Localise ourselves using the surrounding agents. Sets local_pos to be our calculated location.

        Args:
            neighbours (list(tuple)): List of tuples of nearest neighbours (neighbouring agent, distance).
        """

        # Don't allow seed robots to relocalise.
        if self.is_seed:
            return

        # No need to re-localise if we're stationary.
        # In reality, robots will need to periodically re-localise, as they may be bumped around.
        if self.speed == 0:
            return

        # Get the neighbours that are already localised.
        localised_neighbours = [
            n for n in neighbours if n[0].state == AgentState.LOCALISED and n[0].local_pos is not None
        ]

        # If we have < 3 localised neighbours, we can't localise ourselves
        if len(localised_neighbours) < 3:
            self.local_pos = None
            return

        # DEBUG - return the agents actual position as the localised position.
        # Useful for debugging other parts of the agent that rely on localisation to be working properly.
        # seed = self._sim_engine._objects[0]
        # self.local_pos = [self._pos[0] - (seed._pos[0] - seed.local_pos[0]),
        #                              seed._pos[1] - self._pos[1]]
        # return

        # Set a starting local_pos if necessary.
        pos = [-self.radius, +self.radius] if self.local_pos is None else self.local_pos

        # NOTE: This localisation algorithm is a bit rubbish. It's based off the paper, and was designed to manage
        # the constraints they were facing with small, low-power, low-compute robots, and asynchronous comms.

        # Their robots are constantly localising in the background. We only have one frame.

        # Loop the localisation algorithm until the calculated position stops changing per neighbour.
        # Once this happens 10 times and the position still isn't changing, we are localised (ish)
        run_minimise = 0
        last_pos = pos
        while run_minimise < 10:
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
                pos_diff = np.subtract(pos, new_pos) / 4
                # pos_diff = np.subtract(pos, new_pos)

                pos = np.subtract(pos, pos_diff)

            # If our new position is different from our last position, we haven't minimised,
            # so run the localisation again.
            diff_from_last = np.subtract(pos, last_pos)
            # This would usually be linalg.norm, but we omit the sqrt to improve performance.
            dist = diff_from_last[0] ** 2 + diff_from_last[1] ** 2
            last_pos = pos
            threshold = 0.5
            if dist < threshold**2:
                run_minimise += 1
            else:
                run_minimise = 0


        # Add localisation noise to simulate real life.
        noise_threshold = 0.00
        pos += random.uniform(-noise_threshold, noise_threshold)

        # save our localised position
        self.local_pos = pos

    def is_inside_shape(self) -> bool:
        """Check if our localised position (local_pos) is inside the shape (target_shape)

        Returns:
            bool: Whether we're in the shape (True) or not (False).
        """
        if self.shape is None:
            return False

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

        # We're in the bounding box!

        # Get the raw pixels from the shape, and get the color of the pixel at our mapped position.
        pixels = self.shape.shape_data.getdata()

        # Calculate 1D index from 2D mapped position.
        # Pixel origin (0,0) is top left damnit! Flip our mapped position y-axis to deal with this.
        rows = int(size[0] * (size[1] - mapped[1]))
        image_idx = int(rows + mapped[0])

        if pixels[image_idx] != self.color_flag:
            # We are not in our part of the shape :(
            return False

        # Woohoo! We're in the shape!
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
        if not valid_gradients:
            return

        # Calculate new gradient.
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
            return False

        # There will never be a shape in the bottom left quadrant.
        if self.local_pos[0] < 0 and self.local_pos[1] < 0:
            return

        # Stopping Conditions.

        # Condition 1: We are about to edge follow around a stationary robot that has a larger gradient value than our own.
        orb_agent = self.get_orbit_object()
        if orb_agent is None:
            return False

        inside_shape = self.is_inside_shape()

        if (
            orb_agent is not None
            and orb_agent.state == AgentState.LOCALISED
            and inside_shape
            and self.gradient <= orb_agent.gradient
            and self.color_flag == orb_agent.color_flag
            and self.gradient > 1
        ):
            return True

        # Condition 2: We are about to exit the shape.
        if not inside_shape and self.gradient > 1: 
            return True
        
        return False

    def state_idle(self, fps):
        self.speed = 0
        if self.mode == 'monochrome':
            self.color = Colour.white

        # Initialise gradients if we need to.
        if self.gradient == None:
            neighbours = self.get_nearest_neighbours()
            self.update_gradient(neighbours)
            return

        # If we're unlucky, we don't get to start.
        p = 1 / (self.average_start_time * (fps + 1))
        if random.random() > p:
            return False

        # We might need to start edge following. Better check.
        neighbours = self.get_nearest_neighbours()
        start = self.start_edge_following(fps, neighbours)
        if not start:
            return

        # Start!
        self.state = AgentState.MOVING_AROUND_CLUSTER
        self.state_moving_around_cluster(fps)

    def state_moving_around_cluster(self, fps):
        self.speed = self.start_speed
        if self.mode == 'monochrome':
            self.color = Colour.white

        # Get closest neighbours.
        neighbours = self.get_nearest_neighbours(3)
        self.follow_edges(neighbours)
        self.update_gradient(neighbours)

        # If we touch a localised agent, move to the next state.
        if neighbours[0][0].state == AgentState.LOCALISED and self.gradient > 1:
            self.state = AgentState.MOVING_OUTSIDE_SHAPE

    def state_moving_outside_shape(self, fps):
        if self.mode == 'monochrome':
            self.color = Colour.light_blue
        neighbours = self.get_nearest_neighbours(3)

        self.follow_edges(neighbours)
        self.localise(neighbours)
        self.update_gradient(neighbours)

        # If we touch an unlocalised agent, we're outside the seeds.
        if len(neighbours) and neighbours[0][0].state == AgentState.IDLE:
            self.state = AgentState.MOVING_AROUND_CLUSTER
            return

        
        if self.is_inside_shape() and self.gradient > 1:
            self.state = AgentState.MOVING_INSIDE_SHAPE
            return

    def state_moving_inside_shape(self, fps):
        if self.mode == 'monochrome':
            self.color = Colour.orange
        neighbours = self.get_nearest_neighbours()
        self.follow_edges(neighbours)
        self.localise(neighbours)
        self.update_gradient(neighbours)

        # Check if we need to stop and assemble the shape.
        if self.assemble_shape() and self.gradient > 1:
            self.state = AgentState.LOCALISED
            self.speed = 0
            return
        
        # If we touch an unlocalised agent, we're outside the seeds.
        if len(neighbours) and neighbours[0][0].state == AgentState.IDLE:
            self.state = AgentState.MOVING_AROUND_CLUSTER
            return

    def state_localised(self, fps):
        self.speed = 0
        if not self.is_seed:
            if self.mode == 'monochrome':
                self.color = Colour.orange

    def update(self, fps):
        """Update the agents state each frame. This is where the rules are implemented.

        Args:
            fps (float): FPS of the last frame (provided by pygame).
        """

        # Update the underlying SimulationObject.
        super().update(fps)

        match self.state:
            case AgentState.IDLE:
                self.state_idle(fps)
            case AgentState.MOVING_AROUND_CLUSTER:
                self.state_moving_around_cluster(fps)
            case AgentState.MOVING_OUTSIDE_SHAPE:
                self.state_moving_outside_shape(fps)
            case AgentState.MOVING_INSIDE_SHAPE:
                self.state_moving_inside_shape(fps)
            case AgentState.LOCALISED:
                self.state_localised(fps)

