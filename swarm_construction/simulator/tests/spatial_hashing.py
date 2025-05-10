from swarm_construction.simulator.object import SimulationObject
from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.simulator.colors import Colour
import math
import numpy as np
import pygame as pg


class Test:

    def update(self, fps):
        if fps == 0:
            return
        curr_agent = self.objects[45]
        curr_agent.color = Colour.light_green
        nearby_agents = self.sim.get_nearby_objects(curr_agent._neighbourhood)
        for a in nearby_agents:
            if a.object_id == curr_agent.object_id:
                continue
            a.color = Colour.light_blue

    def draw(self):
        pass

    def __init__(self):
        self.objects = []
        self.sim = None

    def _generate_cluster(self, middle, radius, side_length):
        cluster = []
        num_rows = side_length
        num_cols = side_length
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * radius))
        start = [middle - num_cols * radius, middle - num_rows * radius]

        # generate rectangle
        for i in range(num_rows):
            x_offset = 0 if i % 2 == 0 else radius
            for j in range(num_cols):
                pos = [x_offset + radius * 2 * j, line_spacing * i]
                a = SimulationObject(
                    self.sim, pos=np.add(start, pos), radius=radius, speed=0
                )
                cluster.append(a)

        return cluster

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("Spatial Hashing", 800)
        middle = self.sim.game_size / 2
        radius = 30
        side_length = 10
        self.cluster = self._generate_cluster(middle, radius, side_length)
        print(f"Number of cluster objects: {len(self.cluster)}")

        self.objects.extend(self.cluster)

        self.sim.add_update(self.update)
        self.sim.add_draw(self.draw)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    mt = Test()
    mt.run()
