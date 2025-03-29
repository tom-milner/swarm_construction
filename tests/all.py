from . import (
    movement_test,
    collision_detection,
    single_orbit,
    edge_following,
    simple_agent_rules,
)

tests = [
    movement_test,
    collision_detection,
    single_orbit,
    edge_following,
    simple_agent_rules,
]

[t.Test().run() for t in tests]
