# Bio-Inspired Computing Coursework 2: Swarm Shape Construction

- [Bio-Inspired Computing Coursework 2: Swarm Shape Construction](#bio-inspired-computing-coursework-2-swarm-shape-construction)
  - [Statement of Contribution](#statement-of-contribution)
  - [Installing and Running the Simulation](#installing-and-running-the-simulation)
    - [Install Guide - Pip](#install-guide---pip)
    - [Install Guide - Poetry](#install-guide---poetry)
  - [Using the Simulation](#using-the-simulation)
    - [Shape Files](#shape-files)
    - [Keyboard Controls](#keyboard-controls)
    - [Changing Parameters](#changing-parameters)
  - [Documentation](#documentation)
  - [Developing](#developing)


## Statement of Contribution
All team members were equally involved in the design, development, testing, profiling, and bug fixing of the project. The rough contributions of each team member were as follows:

Tom Reynolds
- Shape Creation GUI.
- Shape Assembly.
- Analytics (graphs).
- Multicoloured shape formation.

Tyler Green
- Bridging algorithm.
- Start shape assembly algorithm.
- Stopping criteria.
- Gradient calculations.

Tom Milner
- Simulation engine & simulation objects.
- Collision detection.
- Edge-following.
- Neighbourhood detection.
- Localisation.
- Agent state machine.

## Installing and Running the Simulation

The project was developed, and runs on, Python 3.13.2.

The simulation can be set up in two different ways.
1. [Poetry](#install-guide---poetry) 
2. [Pip](#install-guide---pip)

If you have Poetry installed, follow the [Poetry install guide](#install-guide---poetry). Else, follow the [pip install guide](#install-guide---pip).


### Install Guide - Pip

1. [Install pip](https://pip.pypa.io/en/stable/installation/), if it is not already installed.
2. Install the project requirements using `pip install -r requirements.txt`.
3. Run the project using `python3 -m swarm_construction.main`

### Install Guide - Poetry


`Poetry` was used to create the project, with the idea that it should make dependency management much easier.

To download and run the simulation:
1. Install `Poetry`.
   ```
   pip install poetry
   ```
2. Git clone the project
   ```
   git clone git@github.com:tom-milner/swarm_construction.git
   ```
3. Install the dependencies.
   - Navigate into `swarm_construction/`.
   - Run `poetry install`

4. Run the simulation
   ```
   poetry run python -m swarm_construction.main
   ```

The first run will take a while to start, but will be much quicker after that!


## Using the Simulation
### Shape Files
To change the shape file used in the simulation, change the `shape_file` parameter in `main.py`.
Options for shape files can be found in `swarm_construction/shape`.

### Keyboard Controls
The simulation supports the following keyboard commands:
- `n`: Show the neighbourhoods.
- `p`: Pause the simulation.
- `p` -> `a`: Show analytics graphs of the simulation. Simulation must be paused to run - these will block the simulation.
- `esc` --> Quit the simulation.

### Changing Parameters
To tweak the situation, you can change the following parameters:
- Number of agents

## Documentation
For documentation, see the `/docs` directory [here](./docs/readme.md).

## Developing
> [!IMPORTANT]
> When developing the project, make sure you always run using step 4.

To add dependencies, use Poetry instead of pip, e.g.
```
poetry add my-new-dependency
```

For Poetry help, see the docs [here](https://python-poetry.org).

To run tests, run
```
poetry run python -m tests.my_test
```

e.g. 

```
poetry run python -m tests.all
```