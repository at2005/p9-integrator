# A CUDA implementation of the Mercury N-body Integrator for use on GPU Supercomputing Clusters

Right now not trying to make this efficient, just trying to make it not blow up the solar system.

Currently in-progress: Merging Bulirsch-Stoer into main integration kernel

## Compiling

To compile, run `make` in the root directory. This will create a binary in the `bin` directory. Currently only tested on H100s, but should work on other GPUs – modify the Makefile to specify an arch flag. Ensure that the latest version of CUDA is installed.

## Running

Run the binary on a GPU node using `./bin/mint` with a set of command line arguments:

- `-c` specifies the path to the JSON config file. For an example of what this should look like, check out examples/test.json

### Optional:

- `-i` prints simulation information
- `-p` prints positions at each timestep for each body in the simulation
- `-t` sets the number of timesteps to run the simulation for (default is 1)