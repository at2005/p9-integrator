#include "sim.cuh"
#include "simutils.cuh"

__host__ int main(int argc, char **argv) {
  // cli args
  bool print_sim_info = false;
  bool print_positions = false;
  std::string config_file;
  // default is one orbital period (of Earth)
  int NUM_TIMESTEPS = 1;
  args_parse(argc, argv, &print_sim_info, &print_positions, &NUM_TIMESTEPS, &config_file);

  Sim sim;
  sim_from_config_file(&sim, config_file, NUM_TIMESTEPS);

  // set integration timestep to the one BB21 use
  // double dt = 0.8219;
  double dt = 0.1;
  // double dt = 5 / 365;

  // this is bc we need to allocate memory on the device (on HBM – global
  // memory, copy to SRAM later)
  double *vec_longitude_of_ascending_node_device, *vec_inclination_device, *vec_argument_of_perihelion_device, *vec_mean_anomaly_device, *vec_eccentricity_device, *vec_semi_major_axis_device, *masses_device;
  double3 *output_positions_device;
  double3 *output_positions = (double3 *)malloc(sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS);

  cudaMalloc((void **)&vec_longitude_of_ascending_node_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_inclination_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_argument_of_perihelion_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_mean_anomaly_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_eccentricity_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_semi_major_axis_device, sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&masses_device, (sim.num_bodies + 1) * sizeof(double));
  cudaMalloc((void **)&output_positions_device, sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS);

  cudaMemcpy(vec_longitude_of_ascending_node_device, sim.vec_longitude_of_ascending_node, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_inclination_device, sim.vec_inclination, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_argument_of_perihelion_device, sim.vec_argument_of_perihelion, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_mean_anomaly_device, sim.vec_mean_anomaly, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_eccentricity_device, sim.vec_eccentricity, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_semi_major_axis_device, sim.vec_semi_major_axis, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(masses_device, sim.masses, (sim.num_bodies + 1) * sizeof(double), cudaMemcpyHostToDevice);

  // print sim information
  if (print_sim_info) {
    dump_sim(&sim);
    std::cout << "Launching kernel on " << sim.num_bodies << " threads" << std::endl;
  }

  // positions and velocity 3-vectors, 6 orbital elements for each body, mass for each body (so 7 doubles) + 1 for sun
  size_t sram_size = sim.num_bodies * sizeof(double3) * 2 + sim.num_bodies * sizeof(double) * 7 + sizeof(double);
  mercurius_solver<<<1, sim.num_bodies, sram_size>>>(vec_argument_of_perihelion_device, vec_mean_anomaly_device, vec_eccentricity_device, vec_semi_major_axis_device, vec_inclination_device, vec_longitude_of_ascending_node_device, masses_device, output_positions_device, dt, NUM_TIMESTEPS);

  if (print_sim_info)
    std::cout << "Simulation Finished. Synchronizing...\n";
  cudaDeviceSynchronize();
  cudaMemcpy(output_positions, output_positions_device, sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS, cudaMemcpyDeviceToHost);

  if (print_positions)
    pretty_print_positions(&sim, output_positions);

  cudaFree(vec_longitude_of_ascending_node_device);
  cudaFree(vec_inclination_device);
  cudaFree(vec_argument_of_perihelion_device);
  cudaFree(vec_mean_anomaly_device);
  cudaFree(vec_eccentricity_device);
  cudaFree(vec_semi_major_axis_device);
  cudaFree(masses_device);
  cudaFree(output_positions_device);
  cudaDeviceReset();
}
