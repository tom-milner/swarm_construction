import swarm_construction.simulator.tests.all
from . import simple_agent_rules


tests = [
    # Run the simulator tests.
    swarm_construction.simulator.tests.all,
    # Run the simple Agent test.
    simple_agent_rules,
]

[t.Test().run() for t in tests]
