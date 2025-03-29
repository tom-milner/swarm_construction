from . import (
    movement_test,
    collision_detection,
    single_orbit,
    edge_following,
)


class Test:

    def run(self):
        tests = [movement_test, collision_detection, single_orbit, edge_following]
        [t.Test().run() for t in tests]


if __name__ == "__main__":
    Test().run()
