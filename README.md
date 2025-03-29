# Bio-Inspired Computing Coursework 2: Swarm Shape Construction

## Running the Simulation
`Poetry` was used to create the project, with the idea that it should make dependency management much easier.
However, this is my first time using it, so there may be some teething issues!

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