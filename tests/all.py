from . import movement_test, collision_detection, single_orbit

tests = [movement_test, collision_detection, single_orbit]

[t.Test().run() for t in tests]
