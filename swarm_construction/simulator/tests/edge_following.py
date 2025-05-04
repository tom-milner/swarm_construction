from swarm_construction.simulator.object import SimulationObject
from swarm_construction.simulator.engine import SimulationEngine
from swarm_construction.simulator.colors import Colour
import math
import numpy as np


class Test:

    def update(self, fps):
        if fps == 0:
            return

        # Move the objects
        for a in self.objects:
            a.update(fps)

        for cluster_object in self.cluster:

            # Check if we're touching the object.
            col = self.follower.check_collision(cluster_object)
            if not col[0]:
                continue

            # Check if we're already orbiting this object.
            if np.array_equal(cluster_object._pos, self.follower._orbit_object._pos):
                continue

            # Start orbiting the new object!
            self.follower.set_orbit_object(cluster_object)

            # gradually speed up the follower because it's fun
            if self.follower.speed < 200:
                self.follower.speed += 10

            break

    def draw(self):
        for a in self.objects:
            a.draw()

    def __init__(self):
        self.objects = []

    def _generate_cluster(self, middle, radius):
        cluster = []
        num_rows = 32
        num_cols = 32
        line_spacing = math.fabs(round(math.tan(2 * math.pi / 3) * radius))
        start = [middle - num_cols * radius, middle - num_rows * radius]

        # generate rectangle
        for i in range(num_rows):
            x_offset = 0 if i % 2 == 0 else radius
            for j in range(num_cols):
                pos = [x_offset + radius * 2 * j, line_spacing * i]
                a = SimulationObject(
                    self.sim,
                    pos=np.add(start, pos),
                    radius=radius,
                    color=Colour.light_green,
                )
                cluster.append(a)
        i += 1

        # generate tail
        while i < (num_rows + 6):
            x_offset = 0 if i % 2 == 0 else radius
            pos = [x_offset + radius * 2 * round(j / 2), line_spacing * i]
            a = SimulationObject(
                self.sim,
                pos=np.add(start, pos),
                radius=radius,
                color=Colour.light_green,
            )
            cluster.append(a)
            i += 1

        return cluster

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("SimulationObject Edge Following", 800)

        middle = self.sim.window_size / 2
        radius = 10
        self.cluster = self._generate_cluster(middle, radius)
        print(f"Number of cluster objects: {len(self.cluster)}")

        self.follower = SimulationObject(self.sim, radius=radius, speed=50)
        self.follower.set_orbit_object(self.cluster[0])

        self.objects.extend(self.cluster)
        self.objects.append(self.follower)

        self.sim.add_update(self.update)
        self.sim.add_draw(self.draw)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    mt = Test()
    mt.run()
