from swarm_construction.simulation.engine import SimulationEngine
from swarm_construction.simulation.object import SimulationObject
import math


class Test:
    def update(self, fps):
        if fps == 0:
            return
        self.new_agent._direction += self.delta / fps

        # If we exceed 2PI, wrap around to 0.
        self.new_agent._direction %= 2 * math.pi

        self.new_agent.update(fps)

    def draw(self):
        self.new_agent.draw()

    def run(self):
        # Setup the simulation
        self.sim = SimulationEngine("Movement Test", 800)

        # Add one agent.
        middle = self.sim.window_size / 2
        self.new_agent = SimulationObject(self.sim, [middle, middle])

        self.delta = 4
        self.new_agent.speed = 200
        self.new_agent._direction = 0

        self.sim.add_update(self.update)
        self.sim.add_draw(self.draw)

        # Run the simulation.
        self.sim.run()


if __name__ == "__main__":
    Test().run()
