from . import movement_test, collision_detection, single_orbit, edge_following

tests = [movement_test, collision_detection, single_orbit, edge_following]

[t.Test().run() for t in tests]
