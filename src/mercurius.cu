#include "sim.cuh"
#include "simutils.cuh"

#define CUDA_ALLOCATE_SWEEP(element_name) (cudaMalloc((void **)&(sweep_host.element_name), SWEEPS_PER_GPU * sizeof(double)))

__host__ int get_num_massive_bodies(Sim *sim)
{
  int num_massive_bodies = 0;
  for (int i = 0; i < sim->num_bodies; i++)
  {
    if (sim->masses[i] > 1e-12)
    {
      num_massive_bodies += 1;
    }
  }
  return num_massive_bodies;
}

__host__ int main(int argc, char **argv)
{
  // cli args
  bool print_sim_info = false;
  bool print_positions = false;
  std::string output_file = "";
  std::string config_file;
  long NUM_TIMESTEPS;
  int device = 0;
  args_parse(argc,
             argv,
             &print_sim_info,
             &print_positions,
             &NUM_TIMESTEPS,
             &config_file,
             &output_file,
             &device);

  // we'll be launching multiple processes using a script, so set device via cli arg
  cudaSetDevice(device);
  Sim sim;
  sim_from_config_file(&sim, config_file, NUM_TIMESTEPS, device);

  // set integration timestep to the one BB21 use
  double dt = 0.8219;
  // double dt = 0.1;
  cudaLaunchConfig_t config = {0};
  int cluster_size = 16;
  cudaFuncSetAttribute((void *)mercurius_solver, cudaFuncAttributeNonPortableClusterSizeAllowed, cluster_size);
  assert(sim.num_bodies % cluster_size == 0);
  config.blockDim = dim3(sim.num_bodies / cluster_size, 1, 1);
  config.gridDim = dim3(cluster_size * SWEEPS_PER_GPU, 1, 1);
  config.dynamicSmemBytes = config.blockDim.x * (sizeof(double3) * 2 + sizeof(double));
  if (print_sim_info) std::cout << "Allocating " << config.dynamicSmemBytes << " bytes of SRAM per block" << std::endl;
  cudaError_t error = cudaFuncSetAttribute((void *)mercurius_solver, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
  if (error != cudaSuccess)
  {
    std::cout << "Error setting max shared memory size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.numAttrs = 1;
  config.attrs = attribute;

  // check occupancy
  int potential_cluster_size;
  cudaOccupancyMaxActiveClusters(&potential_cluster_size, mercurius_solver, &config);
  if (print_sim_info) std::cout << "Cluster size: " << potential_cluster_size << std::endl;
  // this is bc we need to allocate memory on the device (on HBM – global
  // memory, copy to SRAM later)
  double *vec_longitude_of_ascending_node_device, *vec_inclination_device,
      *vec_argument_of_perihelion_device, *vec_mean_anomaly_device,
      *vec_eccentricity_device, *vec_semi_major_axis_device, *masses_device;
  double3 *output_positions_device;
  double3 *output_positions =
      (double3 *)malloc(sim.num_bodies * SWEEPS_PER_GPU * sizeof(double3));

  cudaMalloc((void **)&vec_longitude_of_ascending_node_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_inclination_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_argument_of_perihelion_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_mean_anomaly_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_eccentricity_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&vec_semi_major_axis_device,
             sim.num_bodies * sizeof(double));
  cudaMalloc((void **)&masses_device, sim.num_bodies * sizeof(double));

  cudaMalloc((void **)&output_positions_device,
             SWEEPS_PER_GPU * sim.num_bodies * sizeof(double3));

  cudaMemcpy(vec_longitude_of_ascending_node_device,
             sim.vec_longitude_of_ascending_node,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vec_inclination_device,
             sim.vec_inclination,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vec_argument_of_perihelion_device,
             sim.vec_argument_of_perihelion,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vec_mean_anomaly_device,
             sim.vec_mean_anomaly,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vec_eccentricity_device,
             sim.vec_eccentricity,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vec_semi_major_axis_device,
             sim.vec_semi_major_axis,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(masses_device,
             sim.masses,
             sim.num_bodies * sizeof(double),
             cudaMemcpyHostToDevice);

  // allocate sweep memory. rn we'll only be sweeping over planet nine's orbital elements
  Sweep *sweep_device;
  Sweep sweep_host;
  cudaMalloc((void **)&sweep_device, sizeof(Sweep));

  CUDA_ALLOCATE_SWEEP(longitude_of_ascending_nodes);
  CUDA_ALLOCATE_SWEEP(inclinations);
  CUDA_ALLOCATE_SWEEP(argument_of_perihelion);
  CUDA_ALLOCATE_SWEEP(eccentricities);
  CUDA_ALLOCATE_SWEEP(semi_major_axes);
  CUDA_ALLOCATE_SWEEP(masses);

  cudaMemcpy(sweep_device, &sweep_host, sizeof(Sweep), cudaMemcpyHostToDevice);

  // memcpy from sim->sweeps
  cudaMemcpy(sweep_host.longitude_of_ascending_nodes,
             sim.sweeps->longitude_of_ascending_nodes,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sweep_host.inclinations,
             sim.sweeps->inclinations,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sweep_host.argument_of_perihelion,
             sim.sweeps->argument_of_perihelion,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sweep_host.eccentricities,
             sim.sweeps->eccentricities,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sweep_host.semi_major_axes,
             sim.sweeps->semi_major_axes,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sweep_host.masses,
             sim.sweeps->masses,
             SWEEPS_PER_GPU * sizeof(double),
             cudaMemcpyHostToDevice);

  // print sim information
  if (print_sim_info)
  {
    dump_sim(&sim);
    std::cout << "Launching kernel on " << sim.num_bodies << " threads"
              << std::endl;
  }

  // this is important bc we can then only iterate over the massive bodies when computing gravitational kicks
  // TNOs are not massive bodies, ie their mass is zero
  int num_massive_bodies = get_num_massive_bodies(&sim);
  // ie after BATCH_SIZE timesteps, we want to print the output
  // and run kernel with updated orbital elements this is to save memory
  long NUM_ITERS = NUM_TIMESTEPS > BATCH_SIZE ? NUM_TIMESTEPS / BATCH_SIZE : NUM_TIMESTEPS;
  if (NUM_TIMESTEPS > BATCH_SIZE) assert(NUM_TIMESTEPS % BATCH_SIZE == 0);

  for (int batch = 0; batch < NUM_ITERS; batch++)
  {
    cudaLaunchKernelEx(&config,
                       mercurius_solver,
                       vec_argument_of_perihelion_device,
                       vec_mean_anomaly_device,
                       vec_eccentricity_device,
                       vec_semi_major_axis_device,
                       vec_inclination_device,
                       vec_longitude_of_ascending_node_device,
                       masses_device,
                       sweep_device,
                       output_positions_device,
                       num_massive_bodies,
                       batch,
                       dt);

    cudaDeviceSynchronize();
    if (print_sim_info) std::cout << "Batch " << (batch + 1) << " Simulation Complete.\n";
    cudaMemcpy(output_positions,
               output_positions_device,
               sim.num_bodies * sizeof(double3) * SWEEPS_PER_GPU,
               cudaMemcpyDeviceToHost);

    if (print_positions) write_positions(&sim, output_positions, output_file, batch, device);
  }

  cudaFree(vec_longitude_of_ascending_node_device);
  cudaFree(vec_inclination_device);
  cudaFree(vec_argument_of_perihelion_device);
  cudaFree(vec_mean_anomaly_device);
  cudaFree(vec_eccentricity_device);
  cudaFree(vec_semi_major_axis_device);
  cudaFree(masses_device);
  cudaFree(output_positions_device);
  cudaFree(sweep_host.longitude_of_ascending_nodes);
  cudaFree(sweep_host.inclinations);
  cudaFree(sweep_host.argument_of_perihelion);
  cudaFree(sweep_host.eccentricities);
  cudaFree(sweep_host.semi_major_axes);
  cudaFree(sweep_host.masses);
  cudaFree(sweep_device);
  cudaDeviceReset();
}