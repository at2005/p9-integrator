#include "simutils.cuh"
#include "sim.cuh"

__host__
int main(int argc, char** argv) {
    Sim sim;
    initialize_std_sim(&sim, NUM_BODIES);
    double dt = 0.5;

    // testing 3-body system
    Body Earth;
    Earth.inclination = 0.00005 * M_PI / 180.0;
    Earth.longitude_of_ascending_node = -11.26064 * M_PI / 180.0;
    Earth.argument_of_perihelion = 102.94719 * M_PI / 180.0;
    Earth.mean_anomaly = 100.46435 * M_PI / 180.0;
    Earth.eccentricity = 0.01671022;
    Earth.semi_major_axis = 1.00000011;
    Earth.mass = 5.97237e24 / 1.98855e30;
    Earth.name = "Earth";

    // add earth to simulation
    add_body_to_sim(&sim, Earth, 0);

    Body Mars;
    Mars.inclination = 1.848 * M_PI / 180.0;
    Mars.longitude_of_ascending_node = 49.57854 * M_PI / 180.0;
    Mars.argument_of_perihelion = 336.04084 * M_PI / 180.0;
    Mars.mean_anomaly = 0;
    Mars.eccentricity = 0.0934;
    Mars.semi_major_axis = 1.5;
    Mars.mass = 0.000954588;
    Mars.name = "Mars";

    // yay now we add mars to the simulation
    add_body_to_sim(&sim, Mars, 1);

    // this is bc we need to allocate memory on the device
    double *vec_longitude_of_ascending_node_device, *vec_inclination_device, *vec_argument_of_perihelion_device, 
        *vec_mean_anomaly_device, *vec_eccentricity_device, *vec_semi_major_axis_device, *masses_device;
    double3 *output_positions_device;
    double3* output_positions = (double3*)malloc(sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS);

    cudaMalloc((void**)&vec_longitude_of_ascending_node_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&vec_inclination_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&vec_argument_of_perihelion_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&vec_mean_anomaly_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&vec_eccentricity_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&vec_semi_major_axis_device, sim.num_bodies * sizeof(double));
    cudaMalloc((void**)&masses_device, (sim.num_bodies + 1) * sizeof(double));
    cudaMalloc((void**)&output_positions_device, sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS);

    cudaMemcpy(vec_longitude_of_ascending_node_device, sim.vec_longitude_of_ascending_node, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_inclination_device, sim.vec_inclination, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_argument_of_perihelion_device, sim.vec_argument_of_perihelion, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_mean_anomaly_device, sim.vec_mean_anomaly, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_eccentricity_device, sim.vec_eccentricity, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_semi_major_axis_device, sim.vec_semi_major_axis, sim.num_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(masses_device, sim.masses, (sim.num_bodies+1) * sizeof(double), cudaMemcpyHostToDevice);

    bool print_sim_info = false;
    bool print_positions = false;
    args_parse(argc, argv, &print_sim_info, &print_positions);
 
    // print sim information 
    if(print_sim_info) {
        dump_sim(&sim);
        std::cout << "Launching kernel on " << sim.num_bodies << " threads" << std::endl;
    }

    mercurius_keplerian_solver<<<1, sim.num_bodies>>>(
        vec_argument_of_perihelion_device,
        vec_mean_anomaly_device,
        vec_eccentricity_device,
        vec_semi_major_axis_device,
        vec_inclination_device,
        vec_longitude_of_ascending_node_device,
        masses_device,
        dt,
        output_positions_device
    );

    if(print_sim_info) std::cout << "Simulation Finished. Synchronizing...\n";
    cudaDeviceSynchronize();
    cudaMemcpy(output_positions, output_positions_device, sim.num_bodies * sizeof(double3) * NUM_TIMESTEPS, cudaMemcpyDeviceToHost);
    
   if(print_positions) pretty_print_positions(&sim, output_positions);

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
 