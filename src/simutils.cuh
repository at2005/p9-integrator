// A set of host utils for creating and interacting with the simulation

#ifndef __SIMUTILS_CUH__
#define __SIMUTILS_CUH__

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "constants.cuh"

struct Body {
    double inclination;
    double longitude_of_ascending_node;
    double argument_of_perihelion;
    double mean_anomaly;
    double eccentricity;
    double semi_major_axis;
    double mass;
    std::string name;
};


struct Sim {
    int num_bodies;
    double3* positions;
    double3* velocities;
    double* masses;
    double* vec_inclination;
    double* vec_longitude_of_ascending_node;
    double* vec_argument_of_perihelion;
    double* vec_mean_anomaly;
    double* vec_eccentricity;
    double* vec_semi_major_axis;
    std::string* body_names;
};

__host__
void initialize_std_sim(Sim* sim, int num_bodies) {
    sim->vec_inclination = (double*)malloc(num_bodies * sizeof(double));
    sim->vec_longitude_of_ascending_node = (double*)malloc(num_bodies * sizeof(double));
    sim->vec_argument_of_perihelion = (double*)malloc(num_bodies * sizeof(double));
    sim->vec_mean_anomaly = (double*)malloc(num_bodies * sizeof(double));
    sim->vec_eccentricity = (double*)malloc(num_bodies * sizeof(double));
    sim->vec_semi_major_axis = (double*)malloc(num_bodies * sizeof(double));
    sim->masses = (double*)malloc((num_bodies+1) * sizeof(double));
    sim->body_names = (std::string*)malloc(num_bodies * sizeof(std::string));
    
    // assume convention that main body mass is 1
    sim->masses[0] = 1.0;
    sim->num_bodies = num_bodies;
}

__host__
void add_body_to_sim(Sim* sim, Body body, int idx) {
    sim->vec_inclination[idx] = body.inclination;
    sim->vec_longitude_of_ascending_node[idx] = body.longitude_of_ascending_node;
    sim->vec_argument_of_perihelion[idx] = body.argument_of_perihelion;
    sim->vec_mean_anomaly[idx] = body.mean_anomaly;
    sim->vec_eccentricity[idx] = body.eccentricity;
    sim->vec_semi_major_axis[idx] = body.semi_major_axis;
    sim->masses[idx+1] = body.mass;
    sim->body_names[idx] = body.name;
}

__host__
void dump_sim(Sim* sim) {
    std::cout << "Simulation with " << sim->num_bodies << " bodies" << std::endl;
    std::cout << "Main body mass: " << sim->masses[0] << std::endl;
    for(int i = 0; i < sim->num_bodies; i++) {
        std::cout << "Body: " << i << std::endl;
        std::cout << "inclination: " << sim->vec_inclination[i] << std::endl;
        std::cout << "longitude of ascending node: " << sim->vec_longitude_of_ascending_node[i] << std::endl;
        std::cout << "argument of perihelion: " << sim->vec_argument_of_perihelion[i] << std::endl;
        std::cout << "mean anomaly: " << sim->vec_mean_anomaly[i] << std::endl;
        std::cout << "eccentricity: " << sim->vec_eccentricity[i] << std::endl;
        std::cout << "semi major axis: " << sim->vec_semi_major_axis[i] << std::endl;
        std::cout << "mass: " << sim->masses[i+1] << std::endl << std::endl;
    }
}

__host__
void args_parse(int argc, char** argv, bool* print_sim_info, bool* print_positions) {
    for(int i = 0; i < argc; i++) {
        // print sim info
        if(!strcmp(argv[i], "-i")) {
           *print_sim_info = true;
           continue;
        }

        if(!strcmp(argv[i], "-p")) {
            *print_positions = true;
        }
    }
}

__host__
void pretty_print_positions(Sim* sim, double3* output_positions) {
    for(int i = 0; i < NUM_TIMESTEPS; i++) {
        std::cout << "Timestep " << i << std::endl;
        for(int j = 0; j < sim->num_bodies; j++) {
            std::cout << sim->body_names[j] << ": " << output_positions[i*sim->num_bodies + j].x << " " << output_positions[i*sim->num_bodies + j].y << " " << output_positions[i*sim->num_bodies + j].z << std::endl;
        }
        std::cout << std::endl;
    }
}

#endif