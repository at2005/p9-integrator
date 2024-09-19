# A CUDA implementation of the Mercury N-body Integrator for use on GPU Supercomputing Clusters

TODO: optimize the hell out of this.

Note: Units are such that G=1, ie mass is in solar masses, distance is in AU, time is in years, etc.

## Compiling

To compile, run `make` in the root directory. This will create a binary in the `bin` directory. Currently only tested on H100s, but should work on other GPUs – modify the Makefile to specify an arch flag. Ensure that the latest version of CUDA is installed.

## Running

Run the binary on a GPU node using `./bin/mint` with a set of command line arguments:

- `-c` specifies the path to the JSON config file. For an example of what this should look like, check out the examples directory

### Optional:

- `-i` prints simulation information
- `-p` prints positions at each timestep for each body in the simulation
- `-t` sets the number of timesteps to run the simulation for (default is 1)