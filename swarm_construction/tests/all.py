import swarm_construction.simulator.tests.all
from . import simple_agent_rules, localisation, shape_awareness


tests = [
    # Run the simulator tests.
    swarm_construction.simulator.tests.all,
    # Run the agent tests.
    simple_agent_rules,
    localisation,
    shape_awareness,
]

for t in tests:
    print('===== TEST ====')
    print(t.__file__)
    t.Test().run()
    print('=====')
    print()
