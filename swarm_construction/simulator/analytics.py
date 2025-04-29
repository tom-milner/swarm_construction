from swarm_construction.simulator.engine import SimulationEngine

import numpy as np
import matplotlib.pyplot as plt


class Analytics:
    def __init__(self, sim_engine:SimulationEngine, seed_origin):
        self._sim_engine = sim_engine
        self.agents = self._sim_engine._objects
        self.seed_origin = seed_origin
        self.active = True

        self.axis_x_min = -1 * seed_origin[0]
        self.axis_x_max = self._sim_engine.window_size - seed_origin[0]
        self.axis_y_min = -1 * (self._sim_engine.window_size - seed_origin[1]) * 0.5
        self.axis_y_max = seed_origin[1]
    
    def run_analytics(self):
        self.get_positions(self.agents, self.seed_origin)
        #self.plot_local_pos()
        #self.plot_actual_pos()
        self.plot_sidebyside()
    
    def get_positions(self, agents, seed_origin):
        self.agent_positions = []
        for agent in agents:
            if np.all(agent.local_pos != None):
                pos_in_seedcoords = [agent._pos[0] - seed_origin[0],
                                     seed_origin[1] - agent._pos[1]]
                self.agent_positions.append([agent.local_pos, pos_in_seedcoords])

        self.agent_positions = np.array(self.agent_positions)
        #print(np.asarray(self.agent_positions))
        #print(self.agent_positions[:,0])


    def plot_local_pos(self):
        plt.axis([self.axis_x_min, self.axis_x_max, self.axis_y_min, self.axis_y_max])
        plt.axis("equal")
        for (idx, pos) in enumerate(self.agent_positions[:,0]):
            if self.agents[idx].seed_robot:
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            point = plt.Circle(pos, radius = self.agents[0]._radius, ec = outline, fill = fill, color = color)
            plt.gca().add_artist(point)
        plt.show()
        
        
    def plot_actual_pos(self):
        plt.axis([self.axis_x_min, self.axis_x_max, self.axis_y_min, self.axis_y_max])
        plt.axis("equal")
        for (idx, pos) in enumerate(self.agent_positions[:,1]):
            if self.agents[idx].seed_robot:
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            point = plt.Circle(pos, radius = self.agents[0]._radius, ec = outline, fill = fill, color = color)
            plt.gca().add_artist(point)
        plt.show()


    def plot_sidebyside(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        for ax in fig.get_axes():
            ax.set_aspect(1)
            ax.axis([self.axis_x_min, self.axis_x_max, self.axis_y_min, self.axis_y_max])
        
        ax1.set_title("Agent Local Position")
        ax2.set_title("Agent Actual Position")

        for (idx, pos) in enumerate(self.agent_positions):
            if self.agents[idx].seed_robot:
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            point1 = plt.Circle(pos[0], radius = self.agents[0]._radius, ec = outline, fill = fill, color = color)
            ax1.add_artist(point1)
            ax1.annotate(
                idx,
                xy=pos[0],
                fontsize=5,
                verticalalignment="center",
                horizontalalignment="center"
            )

            point2 = plt.Circle(pos[1], radius = self.agents[0]._radius, ec = outline, fill = fill, color = color, label = idx)
            ax2.add_artist(point2)
            ax2.annotate(
                idx,
                xy=pos[1],
                fontsize=5,
                verticalalignment="center",
                horizontalalignment="center"
            )

        plt.show()
