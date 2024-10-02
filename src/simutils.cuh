// A set of host utils for creating and interacting with the simulation

#ifndef __SIMUTILS_CUH__
#define __SIMUTILS_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include <fstream>
#include <iostream>

#include "constants.cuh"
#include "json.hpp"

// macro to allocate host memory for the sweep
#define ALLOCATE_SWEEP_HOST_MEMORY(sweep_name) (sim->sweeps->sweep_name = (double *)malloc(num_bodies * sizeof(double)))

struct Body
{
  double inclination;
  double longitude_of_ascending_node;
  double argument_of_perihelion;
  double mean_anomaly;
  double eccentricity;
  double semi_major_axis;
  double mass;
  std::string name;
};

struct Sweep
{
  double *masses;
  double *inclinations;
  double *longitude_of_ascending_nodes;
  double *argument_of_perihelion;
  double *eccentricities;
  double *semi_major_axes;
};

struct Sim
{
  int num_bodies;
  int num_timesteps;
  double3 *positions;
  double3 *velocities;
  double *masses;
  double *vec_inclination;
  double *vec_longitude_of_ascending_node;
  double *vec_argument_of_perihelion;
  double *vec_mean_anomaly;
  double *vec_eccentricity;
  double *vec_semi_major_axis;
  std::string *body_names;
  Sweep *sweeps;
};

struct PosVel
{
  double3 pos;
  double3 vel;
};

struct KR_Crit
{
  double r_crit;
  double K;
};

__host__ void initialize_std_sim(Sim *sim, int num_bodies, int num_timesteps)
{
  sim->vec_inclination = (double *)malloc(num_bodies * sizeof(double));
  sim->vec_longitude_of_ascending_node =
      (double *)malloc(num_bodies * sizeof(double));
  sim->vec_argument_of_perihelion =
      (double *)malloc(num_bodies * sizeof(double));
  sim->vec_mean_anomaly = (double *)malloc(num_bodies * sizeof(double));
  sim->vec_eccentricity = (double *)malloc(num_bodies * sizeof(double));
  sim->vec_semi_major_axis = (double *)malloc(num_bodies * sizeof(double));
  sim->masses = (double *)malloc(num_bodies * sizeof(double));
  sim->body_names = new std::string[num_bodies];

  sim->num_bodies = num_bodies;
  sim->num_timesteps = num_timesteps;
}

__host__ void add_body_to_sim(Sim *sim, Body body, int idx)
{
  sim->vec_inclination[idx] = body.inclination;
  sim->vec_longitude_of_ascending_node[idx] = body.longitude_of_ascending_node;
  sim->vec_argument_of_perihelion[idx] = body.argument_of_perihelion;
  sim->vec_mean_anomaly[idx] = body.mean_anomaly;
  sim->vec_eccentricity[idx] = body.eccentricity;
  sim->vec_semi_major_axis[idx] = body.semi_major_axis;
  sim->masses[idx] = body.mass;
  sim->body_names[idx] = body.name;
}

__host__ void dump_sim(Sim *sim)
{
  std::cout << "Simulation with " << sim->num_bodies << " bodies" << std::endl;
  for (int i = 0; i < sim->num_bodies; i++)
  {
    std::cout << "Body: " << i << std::endl;
    std::cout << "inclination: " << sim->vec_inclination[i] << std::endl;
    std::cout << "longitude of ascending node: "
              << sim->vec_longitude_of_ascending_node[i] << std::endl;
    std::cout << "argument of perihelion: "
              << sim->vec_argument_of_perihelion[i] << std::endl;
    std::cout << "mean anomaly: " << sim->vec_mean_anomaly[i] << std::endl;
    std::cout << "eccentricity: " << sim->vec_eccentricity[i] << std::endl;
    std::cout << "semi major axis: " << sim->vec_semi_major_axis[i]
              << std::endl;
    std::cout << "mass: " << sim->masses[i] << std::endl
              << std::endl;
  }
}

__host__ void args_parse(int argc,
                         char **argv,
                         bool *print_sim_info,
                         bool *print_positions,
                         int *num_timesteps,
                         std::string *config_file,
                         std::string *output_file,
                         int *device)

{
  for (int i = 0; i < argc; i++)
  {
    *print_sim_info = *print_sim_info || !strcmp(argv[i], "-i");
    *print_positions = *print_positions || !strcmp(argv[i], "-p");

    if (!strcmp(argv[i], "-t"))
    {
      *num_timesteps = atoi(argv[i + 1]);
      continue;
    }

    if (!strcmp(argv[i], "-c"))
    {
      *config_file = std::string(argv[i + 1]);
      continue;
    }

    if (!strcmp(argv[i], "-o"))
    {
      *output_file = std::string(argv[i + 1]);
    }

    if (!strcmp(argv[i], "-d"))
    {
      *device = atoi(argv[i + 1]);
    }
  }
}

__host__ void pretty_print_positions(Sim *sim, double3 *output_positions, int batch_index)
{
  int offset = BATCH_SIZE * batch_index;

  std::cout << "# Timestep " << offset + 1 << std::endl;
  for (int i = 0; i < SWEEPS_PER_GPU; i++)
  {
    std::cout << "Experiment " << (i + 1) << std::endl;
    for (int j = 0; j < sim->num_bodies; j++)
    {
      std::cout << sim->body_names[j] << ": "
                << output_positions[i * sim->num_bodies + j].x << " "
                << output_positions[i * sim->num_bodies + j].y << " "
                << output_positions[i * sim->num_bodies + j].z << std::endl;
    }
    std::cout << std::endl;
  }
}

__host__ void sim_from_config_file(Sim *sim,
                                   std::string config_file,
                                   int num_timesteps,
                                   int device)
{
  /*
  The structure of the JSON config file is as follows:

  {
      "bodies": [
          {
              "name" : "Earth",
              "mass" : 3.00338e-06,
              "semi_major_axis" : 1.00000011,
              "eccentricity" : 0.01671022,
              "inclination" : 8.72665e-07,
              "longitude_of_ascending_node" : -11.26064,
              "argument_of_perihelion" : 102.94719,
              "mean_anomaly" : 100.46435
          },

          etc.
  }
  */
  std::ifstream config_file_stream(config_file);
  nlohmann::json config_file_json = nlohmann::json::parse(config_file_stream);
  auto bodies = config_file_json["bodies"];
  int num_bodies = bodies.size();
  initialize_std_sim(sim, num_bodies, num_timesteps);
  for (int i = 0; i < num_bodies; i++)
  {
    auto body = bodies[i];
    Body sim_body;
    sim_body.name = body["name"];
    sim_body.mass = body["mass"];
    sim_body.semi_major_axis = body["semi_major_axis"];
    sim_body.eccentricity = body["eccentricity"];
    sim_body.inclination = body["inclination"];
    sim_body.longitude_of_ascending_node = body["longitude_of_ascending_node"];
    sim_body.argument_of_perihelion = body["argument_of_perihelion"];
    sim_body.mean_anomaly = body["mean_anomaly"];
    add_body_to_sim(sim, sim_body, i);
  }

  auto sweeps = config_file_json["sweeps"];
  int num_sweeps = sweeps.size();
  if (num_sweeps == 0) return;
  sim->sweeps = (Sweep *)malloc(sizeof(Sweep));
  ALLOCATE_SWEEP_HOST_MEMORY(masses);
  ALLOCATE_SWEEP_HOST_MEMORY(inclinations);
  ALLOCATE_SWEEP_HOST_MEMORY(longitude_of_ascending_nodes);
  ALLOCATE_SWEEP_HOST_MEMORY(argument_of_perihelion);
  ALLOCATE_SWEEP_HOST_MEMORY(eccentricities);
  ALLOCATE_SWEEP_HOST_MEMORY(semi_major_axes);

  for (int i = device; i < num_sweeps; i++)
  {
    // each sweep is a body
    auto sweep = sweeps[i];
    sim->sweeps->masses[i] = sweep["mass"];
    sim->sweeps->inclinations[i] = sweep["inclination"];
    sim->sweeps->longitude_of_ascending_nodes[i] = sweep["longitude_of_ascending_node"];
    sim->sweeps->argument_of_perihelion[i] = sweep["argument_of_perihelion"];
    sim->sweeps->eccentricities[i] = sweep["eccentricity"];
    sim->sweeps->semi_major_axes[i] = sweep["semi_major_axis"];
  }
}

#endif