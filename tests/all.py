import swarm_construction.simulator.tests.all
from . import simple_agent_rules, localisation


tests = [
    # Run the simulator tests.
    swarm_construction.simulator.tests.all,
    # Run the agent tests.
    simple_agent_rules,
    localisation,
]

[t.Test().run() for t in tests]
