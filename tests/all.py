import swarm_construction.simulation.tests.all
from . import simple_agent_rules


tests = [
    # Run the simulator tests.
    swarm_construction.simulation.tests.all,
    # Run the simple Agent test.
    simple_agent_rules,
]

[t.Test().run() for t in tests]
