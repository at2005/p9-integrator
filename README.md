# A CUDA implementation of the Mercury N-body Integrator for use on High-Performance GPU Supercomputing Clusters

Right now not trying to make this efficient, just trying to make it not blow up the solar system.

Currently in-progress: Bulirsch Stoer integrator during close encounters

## Compiling

To compile, run `make` in the root directory. This will create a binary in the `bin` directory.

## Running

Run the binary using `./bin/mint` with a set of optional command line arguments:

- `-i` prints simulation information
- `-p` prints positions at each timestep for each body in the simulation
- `-t` sets the number of timesteps to run the simulation for (default is 1)


## TODO

- Read sim config from JSON instead of manually setting it
- 
