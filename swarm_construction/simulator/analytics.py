from swarm_construction.simulator.engine import SimulationEngine

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib import colors
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


class Analytics:
    def __init__(self, sim_engine: SimulationEngine, seed_origin):
        self._sim_engine = sim_engine
        self.agents = self._sim_engine._objects
        self.seed_origin = seed_origin
        self.active = True

        # Sets the axis limits for the plots
        self.axis_x_min = -1 * seed_origin[0]
        self.axis_x_max = self._sim_engine.window_size - seed_origin[0]
        self.axis_y_min = -1 * (self._sim_engine.window_size - seed_origin[1]) * 0.5
        self.axis_y_max = seed_origin[1]

        self.filename_base = (
            self._sim_engine.shape_name[:-4]
            + "_"
            + str(len(self.agents))
            + "agents_"
            + str(self.agents[0].start_speed)
            + "speed_"
        )

    def run_analytics(self, save=False):
        """Run analytics suite

        self.get_agent is required to get the local and actual agent positions
        Then uncomment any graphs wanted
        plt.show() required to show the graphs
        """
        self.get_agent_data(self.agents, self.seed_origin)
        # self.plot_local_pos(save)
        # self.plot_actual_pos(save)
        # self.plot_sidebyside(save)
        self.plot_comparison(True, save)
        self.plot_error_heatmap(save)
        plt.show()

    def get_agent_data(self, agents, seed_origin):
        """Get data from the agents

        Local and actual positions of each agent are placed in self.agent_positions
        self.agent_positions is a 3D array of seed based coords
        Indexing is a little confusing

        Gradient values are placed in self.agent_gradients

        Args:
            agents (list of objects): list of agent objects
            seed_origin ([x, y]): location of the seed origin in global coords
        """
        self.agent_positions = []
        self.agent_gradients = []
        for agent in agents:
            # ignores unlocalised or moving agents
            if np.all(agent.local_pos != None) and agent.speed == 0:
                # converts actual position to coordinates relative to seed_pos
                # means plot origin is at the seed
                pos_in_seedcoords = [
                    agent._pos[0] - seed_origin[0],
                    seed_origin[1] - agent._pos[1],
                ]
                self.agent_positions.append([agent.local_pos, pos_in_seedcoords])
                self.agent_gradients.append(agent.gradient)

        self.agent_positions = np.array(self.agent_positions)

    ######### PLOTS ##########

    def plot_local_pos(self, save):
        """Plots a map of local positions

        This is basically where the agents think they are
        """
        fig, ax = plt.subplots()
        ax.set_xlim([self.axis_x_min, self.axis_x_max])
        ax.set_ylim([self.axis_y_min, self.axis_y_max])
        ax.set_aspect(1)
        ax.set_title("Agent Local Position")

        for idx, pos in enumerate(self.agent_positions[:, 0]):
            if self.agents[idx].seed_robot:
                # distinguish seeds
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            # each agent is a plt circle patch
            point = plt.Circle(
                pos, radius=self.agents[0]._radius, ec=outline, fill=fill, color=color
            )
            ax.add_artist(point)
        
        if save:
                filename = self.filename_base + "localpos.eps"
                fig.savefig(filename, format="eps")

    def plot_actual_pos(self, save):
        """Plots a map of actual positions

        This is where the agents actually are
        no matter what they think - they're silly guys
        """
        fig, ax = plt.subplots()
        ax.set_xlim([self.axis_x_min, self.axis_x_max])
        ax.set_ylim([self.axis_y_min, self.axis_y_max])
        ax.set_aspect(1)
        ax.set_title("Agent Actual Position")

        for idx, pos in enumerate(self.agent_positions[:, 1]):
            if self.agents[idx].seed_robot:
                # distinguish seeds
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            point = plt.Circle(
                pos, radius=self.agents[0]._radius, ec=outline, fill=fill, color=color
            )
            ax.add_artist(point)

        if save:
            filename = self.filename_base + "actualpos.eps"
            fig.savefig(filename, format="eps")

    def plot_sidebyside(self, save):
        """Plots local pos and actual pos side by side in subplots"""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for ax in fig.get_axes():
            ax.set_aspect(1)
            ax.axis(
                [self.axis_x_min, self.axis_x_max, self.axis_y_min, self.axis_y_max]
            )

        ax1.set_title("Agent Local Position")
        ax2.set_title("Agent Actual Position")

        for idx, pos in enumerate(self.agent_positions):
            if self.agents[idx].seed_robot:
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False

            point1 = plt.Circle(
                # pos [0] is Local position [x,y]
                pos[0],
                radius=self.agents[0]._radius,
                ec=outline,
                fill=fill,
                color=color,
            )
            ax1.add_artist(point1)
            # adds a label of the agents number for comparison
            ax1.annotate(
                idx,
                xy=pos[0],
                fontsize=5,
                verticalalignment="center",
                horizontalalignment="center",
            )

            point2 = plt.Circle(
                # pos [1] is actual position [x,y]
                pos[1],
                radius=self.agents[0]._radius,
                ec=outline,
                fill=fill,
                color=color,
                label=idx,
            )
            ax2.add_artist(point2)
            ax2.annotate(
                idx,
                xy=pos[1],
                fontsize=5,
                verticalalignment="center",
                horizontalalignment="center",
            )

        if save:
            filename = self.filename_base + "sbs.eps"
            fig.savefig(filename, format="eps")

    def plot_comparison(self, stats=False, save=False):
        """Plots a comparison of actual and local position

        Actual position is black circles, with local position overlayed with red dots
        Red lines connect the agents positions

        Args:
            stats (bool, optional): Statistics display toggle. Defaults to False.
                                    Current stats are:
                                        Overall average localisation error
                                        Average localisation error per gradient
        """
        fig, ax1 = plt.subplots()
        for idx, pos in enumerate(self.agent_positions):
            if self.agents[idx].seed_robot:
                outline = "green"
                fill = True
                color = "green"
            else:
                outline = "black"
                fill = False
            # plots agents actual position as black circles
            point = plt.Circle(
                pos[1],
                radius=self.agents[0]._radius,
                ec=outline,
                fill=fill,
                color=color,
            )
            ax1.add_artist(point)
            # plots agents local pos as red dots
            point2 = plt.Circle(pos[0], radius=3, color="red")
            ax1.add_artist(point2)
            # plots a line between the actual and local pos
            ax1.plot([pos[0, 0], pos[1, 0]], [pos[0, 1], pos[1, 1]], color="r")

        ax1.set_xlim([self.axis_x_min, self.axis_x_max])
        ax1.set_ylim([self.axis_y_min, self.axis_y_max])
        ax1.set_aspect(1)
        # makes the legend pretty
        legend_elements = [
            Line2D(
                [],
                [],
                color="white",
                marker="o",
                markeredgecolor="black",
                label="Actual Position",
                markersize=10,
            ),
            Line2D(
                [],
                [],
                color="white",
                marker="o",
                markerfacecolor="red",
                label="Local Position",
                markersize=5,
            ),
        ]
        ax1.legend(handles=legend_elements)
        ax1.set_title("Agents actual position vs local position")
        if save:
            filename = self.filename_base + "comp.eps"
            fig.savefig(filename, format="eps")

        # stats module
        if stats:
            # gets average localisation error and writes in bottom right corner
            error = self.localisation_av_err()
            err_label = "Average localisation error = " + str(f"{error:.3}") + "px"
            ax1.text(
                0.98,
                0.02,
                err_label,
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax1.transAxes,
            )

            num_agents_label = "Number of agents = " + str(len(self.agent_positions))
            ax1.text(
                0.98,
                0.08,
                num_agents_label,
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax1.transAxes,
            )

            fig, ax2 = plt.subplots()
            grad_errors = self.localisation_err_gradient()
            gradients = np.arange(len(grad_errors))
            ax2.plot(gradients, grad_errors)
            ax2.set_ylabel("Average localisation error (px)")
            ax2.set_xlabel("Gradient")
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.set_title("Average localisation error per gradient")
            if save:
                filename = self.filename_base + "error.eps"
                fig.savefig(filename, format="eps")

    def plot_error_heatmap(self, save):
        """Plots an error heatmap on the agents actual position

        Heatmap shows magnitude of the localisation error for each agent
        """
        errors = []
        fig, ax = plt.subplots()
        ax.set_xlim([self.axis_x_min, self.axis_x_max])
        ax.set_ylim([self.axis_y_min, self.axis_y_max])
        ax.set_aspect(1)
        ax.set_title("Error Heatmap")
        cmap = colormaps["plasma"]
        # makes a list of all localisation errors
        for idx, pos in enumerate(self.agent_positions):
            agent_error = np.linalg.norm(pos[1] - pos[0])
            errors.append(agent_error)

        # finds error min and max, uses it to get a normalisation function.
        # the norm function relates to the colour map, allowing scaling to
        # the actual error values
        error_min = np.min(errors)
        error_max = np.max(errors)
        norm_fun = colors.Normalize(vmin=error_min, vmax=error_max)
        errors = norm_fun(errors)

        # plots actual positions, applies colourmap for face colour
        for idx, pos in enumerate(self.agent_positions):
            point = plt.Circle(
                pos[1], radius=self.agents[0]._radius, color=cmap(errors[idx])
            )
            ax.add_artist(point)

        # plots colour legend
        fig.colorbar(
            cm.ScalarMappable(norm=norm_fun, cmap=cmap),
            ax=ax,
            pad=0.1,
            shrink=0.7,
            label="Localisation Error (px)",
        )

        if save:
                filename = self.filename_base + "heatmap.eps"
                fig.savefig(filename, format="eps")

    ######### CALCULATIONS ##########

    def localisation_av_err(self):
        """Calculates average localisation error

        Returns:
            np.float64: Average localisation error for all agents
        """
        total_error = 0
        # finds total localisation error across all agents
        for idx, pos in enumerate(self.agent_positions):
            agent_error = np.linalg.norm(pos[1] - pos[0])
            total_error += agent_error

        # calcs average error
        return total_error / len(self.agent_positions)

    def localisation_err_gradient(self):
        """Calculates the average location error per gradient value

        Returns:
            array: list of av loc errors, gradient is the array index
        """
        total_error = np.zeros(np.max(self.agent_gradients) + 1)
        num_agents = np.zeros(np.max(self.agent_gradients) + 1)
        # finds total error and the number of agents for each gradient value
        for idx, pos in enumerate(self.agent_positions):
            agent_error = np.linalg.norm(pos[1] - pos[0])
            total_error[self.agent_gradients[idx]] += agent_error
            num_agents[self.agent_gradients[idx]] += 1

        # calculates errors per gradient
        # removes any nan values
        grad_errs = np.divide(total_error, num_agents)
        grad_errs = grad_errs[~np.isnan(grad_errs)]

        return grad_errs
