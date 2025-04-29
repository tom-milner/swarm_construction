from swarm_construction.simulator.engine import SimulationEngine
import numpy as np


class Analytic:
    def __init__(self, sim_engine:SimulationEngine):
        self._sim_engine = sim_engine

    
    def get_positions(agents):
        agent_positions = []
        for agent in agents:
            #print(agent.local_pos)
            #print(agent._pos)
            if np.all(agent.local_pos != None):
                #agent_positions.append([[agent.local_pos[0],agent.local_pos[1]],[agent._pos[0],agent._pos[1]]])
                agent_positions.append([agent.local_pos,agent._pos])

        print(np.asarray(agent_positions))