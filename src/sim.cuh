// core numerical integration kernel
#ifndef __SIM_CUH__
#define __SIM_CUH__
#include <cooperative_groups.h>

#include "constants.cuh"
#include "simutils.cuh"
/*
This file contains the core numerical integration kernel and associated helper
functions for my implementation of the Mercury N-body Integrator.
*/
namespace cg = cooperative_groups;
// map "massive body" indices to block 0 rank
// if in block zero we do not need to map to zero
__device__ void get_mapped_block(double3 **mapped_positions, double3 **mapped_velocities, double **mapped_masses, cg::cluster_group cluster, char *sram)
{
  size_t velocities_offset = sizeof(double3) * blockDim.x;
  size_t masses_offset = velocities_offset + sizeof(double3) * blockDim.x;
  double3 *positions = (double3 *)sram;
  double3 *velocities = (double3 *)(sram + velocities_offset);
  double *masses = (double *)(sram + masses_offset);

  *mapped_positions = (double3 *)cluster.map_shared_rank(positions, (cluster.block_rank() == 0) * cluster.block_rank());
  *mapped_velocities = (double3 *)cluster.map_shared_rank(velocities, (cluster.block_rank() == 0) * cluster.block_rank());
  *mapped_masses = (double *)cluster.map_shared_rank(masses, (unsigned int)(cluster.block_rank() == 0) * cluster.block_rank());
}

__device__ double3 cross(const double3 &a, const double3 &b)
{
  return make_double3(fma(a.y, b.z, -a.z * b.y),
                      fma(a.z, b.x, -a.x * b.z),
                      fma(a.x, b.y, -a.y * b.x));
}

__device__ double stable_sqrt(double x)
{
  double gtz = (double)(x >= 0.00);
  // eval to zero if x less than zero
  return sqrt(x * gtz);
}

__device__ double magnitude_squared(const double3 &a)
{
  double temp = a.z * a.z;
  double temp2 = fma(a.y, a.y, temp);
  double temp3 = fma(a.x, a.x, temp2);
  return temp3;
}

// compute mag_sq + norm in one
__device__ void efficient_magnitude(double *mag, double *mag_sq, const double3 &a)
{
  *mag_sq = magnitude_squared(a);
  *mag = stable_sqrt(*mag_sq);
}

__device__ double stable_log(double x)
{
  double gtz = (double)(x > 0.00);
  double lez = (double)(x <= 0.00);
  return log(x * gtz + lez);
}

__device__ double stable_acos(double x)
{
  double alto = (double)(fabs(x) <= 1.00);
  // so, basically this computes acos(x) if x within bounds
  // otherwise it computes acos(+/- 1.00)
  // x*alto evals to x when x inside bounds
  // copysign((1.00 - alto), x) evals to +/- 1.00 when x outside bounds
  return acos(fma(x, alto, copysign((1.00 - alto), x)));
}

__device__ double stable_asin(double x)
{
  double alto = (double)(fabs(x) <= 1.00);
  return asin(fma(x, alto, copysign((1.00 - alto), x)));
}

// returns zero if y is zero
__device__ double stable_division(double x, double y)
{
  double etz = (double)(y == 0.00);
  double netz = (double)(y != 0.00);
  double res = (netz * x) / (y + etz);
  return res;
}

// solves kepler's equation for the eccentric anomaly E
__device__ double danby_burkardt(double mean_anomaly, double eccentricity)
{
  // init eccentric anomaly to mean anomaly
  double E = mean_anomaly;
  for (int i = 0; i < MAX_ITERATIONS_ROOT_FINDING; i++)
  {
    double sin_E, cos_E;
    sincos(E, &sin_E, &cos_E);
    double e_ngto = (double)(eccentricity < 1.00);
    double e_gto = (double)(eccentricity > 1.00);
    sin_E = e_ngto * sin_E + e_gto * sinh(E);
    cos_E = e_ngto * cos_E + e_gto * cosh(E);

    double e_sin = eccentricity * sin_E;
    // f = M - esinh + H
    double sign = copysign(1.00, eccentricity - 1);
    double f = (sign * mean_anomaly) - e_sin + E;
    // double f = E - e_sin - mean_anomaly;
    double e_cos = eccentricity * cos_E;
    double f_prime = 1 - e_cos;
    double dE = -f / f_prime;
    // higher order convergence only near the end
    if (i > MAX_ITERATIONS_ROOT_FINDING - 2)
    {
      // cubic convergence
      dE = -f / (f_prime + dE * e_sin / 2.00);
      // quartic convergence
      dE = -f / ((f_prime + dE * e_sin / 2.00) + (dE * dE * e_cos / 6.00));
      // quintic convergence
      dE = -f / ((f_prime + dE * e_sin / 2.00) + (dE * dE * e_cos / 6.00) -
                 (dE * dE * dE * e_sin / 24.00));
    }
    E += dE;
  }

  return E;
}

__device__ double dist(double3 a, double3 b)
{
  return norm3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double fetch_r_crit(
    MappedBlock *mapped_block,
    int idx_large_body)
{
  double3 r = mapped_block->positions[idx_large_body];
  double3 v = mapped_block->velocities[idx_large_body];

  double3 h;
  h.x = fma(r.y, v.z, -r.z * v.y);
  h.y = fma(r.z, v.x, -r.x * v.z);
  h.z = fma(r.x, v.y, -r.y * v.x);

  double h_mag2 = magnitude_squared(h);

  double r_mag = magnitude(r);
  double v_mag2 = magnitude_squared(v);
  double r_dot_v = fma(r.x, v.x, fma(r.y, v.y, r.z * v.z));

  double3 e;
  double factor = v_mag2 - 1.0 / r_mag;
  e.x = fma(factor, r.x, -r_dot_v * v.x);
  e.y = fma(factor, r.y, -r_dot_v * v.y);
  e.z = fma(factor, r.z, -r_dot_v * v.z);

  double e_mag2 = magnitude_squared(e);

  // Calculate semi-major axis: a = h*h/mu(1-e*e)
  double semi_major_axis = h_mag2 / (1.0 - e_mag2);

  // Calculate Hill radius
  return semi_major_axis * cbrt(mapped_block->masses[idx_large_body] / 3.0);
}

__device__ KR_Crit changeover(
    PosVel current_coords,
    int idx_other,
    double dt,
    MappedBlock *mapped_block)
{
  int idx = threadIdx.x;
  double r_crit = fetch_r_crit(current_coords, masses, idx_other, dt);
  // we can assume that massive bodies are in cluster 0
  double r_ij = (double)(idx != idx_other) * dist(mapped_block->positions[idx_other], current_coords.pos);
  double y = (r_ij - 0.1 * r_crit) / (0.9 * r_crit);
  double K = y * y / (2 * y * y - 2 * y + 1);
  // trying to avoid branching
  double gtz = (double)(y > 0);
  double gto = (double)(y > 1);
  double valid = (double)(y <= 1 && y >= 0);
  KR_Crit res;
  res.K = K * gtz * valid + gto;
  res.r_crit = r_crit;
  return res;
}

__device__ PosVel cartesian_from_elements(double inclination,
                                          double longitude_of_ascending_node,
                                          double argument_of_perihelion,
                                          double mean_anomaly,
                                          double eccentricity,
                                          double semi_major_axis)

{
  double cos_i, sin_i, cos_o, sin_o, cos_a, sin_a;
  sincos(inclination, &sin_i, &cos_i);
  sincos(longitude_of_ascending_node, &sin_o, &cos_o);
  sincos(argument_of_perihelion, &sin_a, &cos_a);

  double z1 = cos_a * cos_o;
  double z2 = cos_a * sin_o;
  double z3 = sin_a * cos_o;
  double z4 = sin_a * sin_o;
  double d11 = fma(-z4, cos_i, z1);
  double d12 = fma(z3, cos_i, z2);
  double d13 = sin_a * sin_i;
  double d21 = fma(-z2, cos_i, -z3);
  double d22 = fma(z1, cos_i, -z4);
  double d23 = cos_a * sin_i;

  double romes = stable_sqrt(copysign(1.00, eccentricity - 1) * fma(eccentricity, eccentricity, -1.00));
  double e_gto = (double)(eccentricity > 1.00);
  double e_ngto = (double)(eccentricity < 1.00);

  double eccentric_anomaly = danby_burkardt(mean_anomaly, eccentricity);
  double sin_E, cos_E;
  sincos(eccentric_anomaly, &sin_E, &cos_E);
  sin_E = fma(e_ngto, sin_E, e_gto * sinh(eccentric_anomaly));
  cos_E = fma(e_ngto, cos_E, e_gto * cosh(eccentric_anomaly));
  z1 = semi_major_axis * (cos_E - eccentricity);
  z2 = copysign(1.00, 1 - eccentricity) * semi_major_axis * romes * sin_E;

  eccentric_anomaly =
      stable_division(stable_sqrt(stable_division(1.00, fabs(semi_major_axis))), copysign(1.00, 1 - eccentricity) * (fma(-eccentricity, cos_E, 1.00)));

  z3 = -sin_E * eccentric_anomaly;
  z4 = romes * cos_E * eccentric_anomaly;

  PosVel res;
  res.pos = make_double3(fma(d11, z1, d21 * z2),
                         fma(d12, z1, d22 * z2),
                         fma(d13, z1, d23 * z2));
  res.vel = make_double3(fma(d11, z3, d21 * z4),
                         fma(d12, z3, d22 * z4),
                         fma(d13, z3, d23 * z4));

  return res;
}

__device__ void elements_from_cartesian(
    const double3 *current_positions,
    const double3 *current_velocities,
    double *current_inclination,
    double *current_longitude_of_ascending_node,
    double *current_argument_of_perihelion,
    double *current_mean_anomaly,
    double *current_eccentricity,
    double *current_semi_major_axis)
{
  int idx = threadIdx.x;
  double3 current_p = current_positions[idx];
  double3 current_v = current_velocities[idx];
  double3 angular_momentum = cross(current_p, current_v);
  double epsilon = 1e-8;
  double h_sq = magnitude_squared(angular_momentum) + epsilon;
  double inclination = stable_acos(angular_momentum.z * rsqrt(h_sq));
  double longitude_of_ascending_node = atan2(angular_momentum.x, -angular_momentum.y);

  double r_v = fma(current_p.x, current_v.x, fma(current_p.y, current_v.y, (current_p.z * current_v.z)));
  double v_sq = magnitude_squared(current_v);
  double r = norm3d(current_p.x, current_p.y, current_p.z);
  double s = h_sq;
  double eccentricity = stable_sqrt(fma(s, v_sq - (2.00 / r), 1.00));
  double perihelion_distance = s / (1.00 + eccentricity);

  // true longitude
  double cos_i = cos(inclination);
  double true_longitude;

  // evals to zero if angular momentum y component is zero
  double to = stable_division(-angular_momentum.x, angular_momentum.y);
  double temp = (1.00 - cos_i) * to;
  double etz = (double)(angular_momentum.y == 0.00);
  double netz = (double)(angular_momentum.y != 0.00);
  // temp2 set to one if angular momentum y component is zero
  double temp2 = fma(netz, to * to, etz);
  true_longitude =
      atan2((fma(current_p.y, fma(temp2, cos_i, (netz * 1.00)), -current_p.x * temp)),
            fma(current_p.x, fma(netz, cos_i, temp2), -current_p.y * temp));

  // mean anomaly and longitude of perihelion
  etz = (double)(eccentricity == 0.00);
  netz = (double)(eccentricity != 0.00);
  // set to zero if e = 0
  double cos_E_anomaly = stable_division(fma(v_sq, r, -1.00), eccentricity);
  // set to pi/2 if e = 0
  double E_anomaly = stable_acos(cos_E_anomaly);
  // set to true longitude if e = 0
  double r_v_ltz = (double)(r_v < 0.00);
  E_anomaly = fma(r_v_ltz, TWOPI, copysign(E_anomaly, r_v));

  double M_anomaly = netz * fma(-eccentricity, sin(E_anomaly), E_anomaly) + etz * true_longitude;

  double e_gto = (double)(eccentricity > 1.00);
  double e_ngto = (double)(eccentricity < 1.00);
  E_anomaly = fma(e_gto, stable_log(cos_E_anomaly + stable_sqrt(fma(cos_E_anomaly, cos_E_anomaly, -1.00))), e_ngto * E_anomaly);
  E_anomaly = fma(e_gto, copysign(E_anomaly, r_v), e_ngto * E_anomaly);
  M_anomaly = fma(e_gto, fma(eccentricity, sinh(E_anomaly), -E_anomaly), e_ngto * M_anomaly);

  // set to zero if e = 0
  double cos_f = stable_division((s - r), (eccentricity * r));
  // set to pi/2 if e = 0
  double f = stable_acos(cos_f);
  f = fma(r_v_ltz, TWOPI, copysign(f, r_v));
  double p = true_longitude - f;
  // p set to zero if e = 0
  p = netz * fmod(p + TWOPI + TWOPI, TWOPI);

  double M_ltz = (double)(M_anomaly < 0.00);
  M_anomaly = fma(M_ltz, TWOPI, M_anomaly);
  M_anomaly = fmod(M_anomaly, TWOPI);

  // argument of perihelion
  double argument_of_perihelion = p - longitude_of_ascending_node;
  double semi_major_axis = perihelion_distance / (1.00 - eccentricity);

  *current_inclination = inclination;
  *current_longitude_of_ascending_node = longitude_of_ascending_node;
  *current_argument_of_perihelion = argument_of_perihelion;
  *current_mean_anomaly = M_anomaly;
  *current_eccentricity = eccentricity;
  *current_semi_major_axis = semi_major_axis;
}

// this function returns an updated velocity
__device__ double3 body_interaction_kick(
    PosVel current_coords,
    // position and velocity arrays are read-only
    int num_massive_bodies,
    double dt,
    MappedBlock *mapped_block,
    bool possible_close_encounter = false)
{
  int idx = threadIdx.x;
  const double3 my_position = current_coords.pos;
  double3 acc = make_double3(0.0, 0.0, 0.0);
  double3 dist;
  for (int i = 0; i < num_massive_bodies; i++)
  {
    // 3-vec displacement, let r = x, y, z, this is the direction of the
    // acceleration it is directed towards the other body since gravity is
    // attractive
    dist.x = mapped_block->positions[i].x - my_position.x;
    dist.y = mapped_block->positions[i].y - my_position.y;
    dist.z = mapped_block->positions[i].z - my_position.z;

    double r_sq;
    double r;
    efficient_magnitude(&r, &r_sq, dist);

    // compute both K and r_crit
    KR_Crit changeover_vals = changeover(current_coords, i, dt, mapped_block);

    double changeover_weight = changeover_vals.K;
    double r_crit = changeover_vals.r_crit;
    if (possible_close_encounter)
    {
      // 1 - K weighting if close encounter
      changeover_weight = (r <= r_crit) * (1 - changeover_weight) + (r > r_crit) * changeover_weight;
    }

    // add smoothing constant
    r_sq += (double)(idx != i) * SMOOTHING_CONSTANT_SQUARED;
    // the (r^2 + s^2) comes from inv sq law w/ smoothing, and the other r bc
    // direction is normalized
    double force_denom = r_sq * r;

    double weighted_acceleration =
        changeover_weight * stable_division(mapped_block->masses[i], force_denom);
    // // accumulate total acceleration due to all bodies, except self
    acc.x = fma(weighted_acceleration, dist.x, acc.x);
    acc.y = fma(weighted_acceleration, dist.y, acc.y);
    acc.z = fma(weighted_acceleration, dist.z, acc.z);
  }

  // update momenta (velocity here) with total acceleration
  current_coords.vel.x = fma(acc.x, dt, current_coords.vel.x);
  current_coords.vel.y = fma(acc.y, dt, current_coords.vel.y);
  current_coords.vel.z = fma(acc.z, dt, current_coords.vel.z);
  return current_coords.vel;
}

__device__ double3 main_body_kinetic(
    const double3 *positions,
    int num_massive_bodies,
    double dt,
    MappedBlock *mapped_block)
{
  int idx = threadIdx.x;
  double3 my_position = positions[idx];
  double3 p = make_double3(0.0, 0.0, 0.0);
  // calculate total momentum of all bodies
  for (int i = 0; i < num_massive_bodies; i++)
  {
    p.x = fma(mapped_block->masses[i], mapped_block->velocities[i].x, p.x);
    p.y = fma(mapped_block->masses[i], mapped_block->velocities[i].y, p.y);
    p.z = fma(mapped_block->masses[i], mapped_block->velocities[i].z, p.z);
  }

  // assume central mass is 1
  my_position.x = fma(p.x, dt, my_position.x);
  my_position.y = fma(p.y, dt, my_position.y);
  my_position.z = fma(p.z, dt, my_position.z);
  return my_position;
}

// this numerically advances the velocity, taking into account the masses of all
// bodies (incl sun)
__device__ double3 update_my_velocity_total(
    PosVel current_coords,
    int num_massive_bodies,
    double dt,
    MappedBlock *mapped_block)

{
  /*
  updates the velocities of all bodies based on the mass distribution of all
  bodies (incl the sun)
  */

  const double3 my_position = current_coords.pos;
  // the body interaction kick updates velocities based on the masses of all
  // minor bodies
  double3 my_velocity = body_interaction_kick(current_coords, num_massive_bodies, dt, mapped_block, true);
  // now for the main body:
  // update acceleration due to main body
  // a = - G * M / r^3 * r, which in this case simplifies to 1/(r^3) * r_vec
  double r;
  double r_sq;
  efficient_magnitude(&r, &r_sq, my_position);
  // add smoothing constant
  r_sq += SMOOTHING_CONSTANT_SQUARED;
  double force_denom = r_sq * r;
  double3 r_vec = make_double3(my_position.x / force_denom,
                               my_position.y / force_denom,
                               my_position.z / force_denom);
  // negative bc directed inwards
  my_velocity.x = fma(-r_vec.x, dt, my_velocity.x);
  my_velocity.y = fma(-r_vec.y, dt, my_velocity.y);
  my_velocity.z = fma(-r_vec.z, dt, my_velocity.z);
  return my_velocity;
}

__device__ PosVel modified_midpoint(
    PosVel current_coords,
    int num_massive_bodies,
    double dt,
    int N,
    MappedBlock *mapped_block)
{
  /*
  returns an updated position based on the modified midpoint method.
  */

  double subdelta = dt / (double)N;
  // euler step
  // this sets up z_1 and z_0
  // so we can use z_0 to calc z_2 and use z_1 for z_3
  double3 z_1 = current_coords.pos;
  z_1.x = fma(current_coords.vel.x, subdelta, z_1.x);
  z_1.y = fma(current_coords.vel.y, subdelta, z_1.y);
  z_1.z = fma(current_coords.vel.z, subdelta, z_1.z);
  double double_step = 2 * subdelta;
  double3 z_0 = current_coords.pos;
  current_coords.pos = z_1;
  current_coords.vel =
      update_my_velocity_total(current_coords, num_massive_bodies, subdelta, mapped_block);

  // so now, we need to calculate z_(m+1) using z_(m-1)
  // we can store z_(m-1) in z_0 and z_(m+1) in z_1
  // we already have z_0 and z_1 so we can use them for computing z_2/3

  for (int i = 1; i < N; i++)
  {
    double3 temp;
    // here we are computing the new (m+1) position
    temp.x = fma(double_step, current_coords.vel.x, z_0.x);
    temp.y = fma(double_step, current_coords.vel.y, z_0.y);
    temp.z = fma(double_step, current_coords.vel.z, z_0.z);
    // so we set this position to be the new z_0
    z_0 = z_1;
    z_1 = temp;
    current_coords.pos = z_1;
    // update velocities
    current_coords.vel =
        update_my_velocity_total(current_coords, num_massive_bodies, subdelta, mapped_block);
  }

  // final little bit
  double3 out;
  out.x = 0.5 * (z_1.x + fma(current_coords.vel.x, subdelta, z_0.x));
  out.y = 0.5 * (z_1.y + fma(current_coords.vel.y, subdelta, z_0.y));
  out.z = 0.5 * (z_1.z + fma(current_coords.vel.z, subdelta, z_0.z));
  PosVel res;
  // store final positions and velocities
  res.pos = out;
  res.vel = current_coords.vel;
  return res;
}

__device__ PosVel richardson_extrapolation(
    PosVel current_coords,
    int num_massive_bodies,
    double dt,
    MappedBlock *mapped_block)
{
  uint32_t N = 1;
  PosVel out;
  PosVel buffer[MAX_ROWS_RICHARDSON][MAX_ROWS_RICHARDSON];
  // for our first approx, we use the OG dt
  buffer[0][0] = modified_midpoint(current_coords, num_massive_bodies, dt, N, mapped_block);
  for (int i = 1; i < MAX_ROWS_RICHARDSON; i++)
  {
    N <<= 1;
    // N = pow(2, i);
    buffer[i][0] = modified_midpoint(current_coords, num_massive_bodies, dt, N, mapped_block);

    for (int j = 1; j <= i; j++)
    {
      // NOTE: do not use implement this with uint32_t bitshifts bc it does not work for some really strange reason
      double dpow2 = pow(4.00, j);
      double denom = dpow2 - 1.00;

      // for positions
      buffer[i][j].pos.x =
          fma(dpow2, buffer[i][j - 1].pos.x, -buffer[i - 1][j - 1].pos.x);
      buffer[i][j].pos.y =
          fma(dpow2, buffer[i][j - 1].pos.y, -buffer[i - 1][j - 1].pos.y);
      buffer[i][j].pos.z =
          fma(dpow2, buffer[i][j - 1].pos.z, -buffer[i - 1][j - 1].pos.z);

      // for velocities
      buffer[i][j].vel.x = fma(dpow2, buffer[i][j - 1].vel.x, -buffer[i - 1][j - 1].vel.x);
      buffer[i][j].vel.y = fma(dpow2, buffer[i][j - 1].vel.y, -buffer[i - 1][j - 1].vel.y);
      buffer[i][j].vel.z = fma(dpow2, buffer[i][j - 1].vel.z, -buffer[i - 1][j - 1].vel.z);

      buffer[i][j].pos.x /= denom;
      buffer[i][j].pos.y /= denom;
      buffer[i][j].pos.z /= denom;

      buffer[i][j].vel.x /= denom;
      buffer[i][j].vel.y /= denom;
      buffer[i][j].vel.z /= denom;
    }
  }

  out.pos = buffer[MAX_ROWS_RICHARDSON - 1][MAX_ROWS_RICHARDSON - 1].pos;
  out.vel = buffer[MAX_ROWS_RICHARDSON - 1][MAX_ROWS_RICHARDSON - 1].vel;
  return out;
}

// this ensures that the sun is in a reference frame in which it is stationary
// and at the origin
__device__ double3 democratic_heliocentric_conversion(
    PosVel current_coords,
    int num_massive_bodies,
    MappedBlock *mapped_block,
    bool reverse = false)
{
  double total_mass = 0.0;
  double not_reverse_d = (double)(!reverse);
  double add_or_sub = pow(-1, not_reverse_d);

  double3 mass_weighted_v = make_double3(0.0, 0.0, 0.0);
  for (int i = 0; i < num_massive_bodies; i++)
  {
    total_mass += mapped_block->masses[i];
    mass_weighted_v.x = fma(mapped_block->masses[i], mapped_block->velocities[i].x, mass_weighted_v.x);
    mass_weighted_v.y = fma(mapped_block->masses[i], mapped_block->velocities[i].y, mass_weighted_v.y);
    mass_weighted_v.z = fma(mapped_block->masses[i], mapped_block->velocities[i].z, mass_weighted_v.z);
  }

  // if we are performing the reverse conversion, we just need to divide by the main mass = 1
  double scaling_factor = 1.00 / fma(not_reverse_d, total_mass, 1.00);

  mass_weighted_v.x *= scaling_factor;
  mass_weighted_v.y *= scaling_factor;
  mass_weighted_v.z *= scaling_factor;

  // if we are performing the reverse conversion, we need to add not subtract
  double3 res = make_double3(0.0, 0.0, 0.0);
  res.x = fma(add_or_sub, mass_weighted_v.x, current_coords.vel.x);
  res.y = fma(add_or_sub, mass_weighted_v.y, current_coords.vel.y);
  res.z = fma(add_or_sub, mass_weighted_v.z, current_coords.vel.z);
  return res;
}

__device__ bool close_encounter_p(
    double3 *positions,
    double3 *velocities,
    double *masses,
    int num_massive_bodies,
    double dt,
    MappedBlock *mapped_block)
{
  // get indices of bodies i am undergoing close encounters with
  int idx = threadIdx.x;
  bool has_close_encounter = false;
  for (int i = 0; i < num_massive_bodies; i++)
  {
    double3 r_ij;
    r_ij.x = mapped_block->positions[i].x - positions[idx].x;
    r_ij.y = mapped_block->positions[i].y - positions[idx].y;
    r_ij.z = mapped_block->positions[i].z - positions[idx].z;
    double r_ij_mag = norm3d(r_ij.x, r_ij.y, r_ij.z);
    double r_crit = fetch_r_crit(mapped_block, i);
    bool r_crit_reached = (r_ij_mag < r_crit) && (idx != i);
    has_close_encounter = has_close_encounter || r_crit_reached;
  }

  return has_close_encounter;
}

__global__ void mercurius_solver(double *vec_argument_of_perihelion_hbm,
                                 double *vec_mean_anomaly_hbm,
                                 double *vec_eccentricity_hbm,
                                 double *vec_semi_major_axis_hbm,
                                 double *vec_inclination_hbm,
                                 double *vec_longitude_of_ascending_node_hbm,
                                 double *vec_masses_hbm,
                                 double3 *output_positions_hbm,
                                 int num_massive_bodies,
                                 int batch_idx,
                                 double dt)

{
  // declare SRAM buffer
  extern __shared__ char sram[];
  // namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int num_threads = cluster.num_threads();
  int body_idx = cluster.thread_rank();
  int idx = threadIdx.x;
  cg::grid_group g = cg::this_grid();
  int cluster_idx = g.cluster_rank();

  cluster.sync();

  size_t velocities_offset = sizeof(double3) * blockDim.x;
  size_t masses_offset = velocities_offset + sizeof(double3) * blockDim.x;
  double3 *positions = (double3 *)&sram;
  double3 *velocities = (double3 *)(sram + velocities_offset);
  double *masses = (double *)(sram + masses_offset);

  MappedBlock mapped_block;
  get_mapped_block(&mapped_block.positions, &mapped_block.velocities, &mapped_block.masses, cluster, sram);

  cluster.sync();
  // so each cluster has num_threads, and so we are reading the corresponding experiment block from hbm
  int hbm_idx = num_threads * cluster_idx + body_idx;
  // there are only as many masses as there are bodies
  masses[idx] = vec_masses_hbm[body_idx];

  double argument_of_perihelion = vec_argument_of_perihelion_hbm[hbm_idx];
  double mean_anomaly = vec_mean_anomaly_hbm[hbm_idx];
  double eccentricity = vec_eccentricity_hbm[hbm_idx];
  double semi_major_axis = vec_semi_major_axis_hbm[hbm_idx];
  double inclination = vec_inclination_hbm[hbm_idx];
  double longitude_of_ascending_node = vec_longitude_of_ascending_node_hbm[hbm_idx];

  double half_dt = 0.5 * dt;
  // initially populate positions and velocities
  PosVel pos_vel = cartesian_from_elements(inclination,
                                           longitude_of_ascending_node,
                                           argument_of_perihelion,
                                           mean_anomaly,
                                           eccentricity,
                                           semi_major_axis);

  cluster.sync();
  positions[idx] = pos_vel.pos;
  velocities[idx] = pos_vel.vel;
  cluster.sync();
  // convert to democratic heliocentric coordinates
  velocities[idx] = democratic_heliocentric_conversion((PosVel){.pos = positions[idx], .vel = velocities[idx]}, num_massive_bodies, &mapped_block);

  for (int i = 0; i < BATCH_SIZE; i++)
  {
    // first "kicks"
    cluster.sync();
    double3 velocity_after_body_interaction =
        body_interaction_kick((PosVel){.pos = positions[idx], .vel = velocities[idx]}, num_massive_bodies, half_dt, &mapped_block);
    velocities[idx] = velocity_after_body_interaction;
    cluster.sync();
    double3 position_after_main_body_kick =
        main_body_kinetic(positions, num_massive_bodies, half_dt, &mapped_block);
    positions[idx] = position_after_main_body_kick;
    cluster.sync();

    double n = rsqrt(semi_major_axis * semi_major_axis * semi_major_axis);

    // SOLUTION TO MAIN (largest) HAMILTONIAN
    // if i am undergoing a close encounter with anyone
    // use bulirsch-stoer aka richardson extrapolation w/ mod midpoint
    PosVel numerical_soln_to_close_encounter = PosVel{.pos = make_double3(0.0, 0.0, 0.0), .vel = make_double3(0.0, 0.0, 0.0)};
    PosVel analyical_soln_to_kepler = PosVel{.pos = make_double3(0.0, 0.0, 0.0), .vel = make_double3(0.0, 0.0, 0.0)};
    bool is_close_encounter = close_encounter_p(positions, velocities, masses, num_massive_bodies, dt, &mapped_block);
    // directly updates positions and velocities by dt
    numerical_soln_to_close_encounter =
        richardson_extrapolation((PosVel){.pos = positions[idx], .vel = velocities[idx]}, num_massive_bodies, dt, &mapped_block);

    elements_from_cartesian(positions, velocities, &inclination, &longitude_of_ascending_node, &argument_of_perihelion, &mean_anomaly, &eccentricity, &semi_major_axis);
    // advance mean anomaly, this is essentially advancing to the next
    // timestep
    mean_anomaly = (double)(!is_close_encounter) * fmod(fma(n, dt, mean_anomaly), TWOPI) + (double)(is_close_encounter)*mean_anomaly;
    analyical_soln_to_kepler = cartesian_from_elements(inclination,
                                                       longitude_of_ascending_node,
                                                       argument_of_perihelion,
                                                       mean_anomaly,
                                                       eccentricity,
                                                       semi_major_axis);

    // separating out calculation with update ensures that no race conditions
    // occur
    cluster.sync();
    double not_close_encounter = (1.00 - (double)is_close_encounter);
    positions[idx].x =
        fma(not_close_encounter, analyical_soln_to_kepler.pos.x, (double)is_close_encounter * numerical_soln_to_close_encounter.pos.x);
    positions[idx].y =
        fma(not_close_encounter, analyical_soln_to_kepler.pos.y, (double)is_close_encounter * numerical_soln_to_close_encounter.pos.y);
    positions[idx].z =
        fma(not_close_encounter, analyical_soln_to_kepler.pos.z, (double)is_close_encounter * numerical_soln_to_close_encounter.pos.z);
    velocities[idx].x =
        fma(not_close_encounter, analyical_soln_to_kepler.vel.x, (double)is_close_encounter * numerical_soln_to_close_encounter.vel.x);
    velocities[idx].y =
        fma(not_close_encounter, analyical_soln_to_kepler.vel.y, (double)is_close_encounter * numerical_soln_to_close_encounter.vel.y);
    velocities[idx].z =
        fma(not_close_encounter, analyical_soln_to_kepler.vel.z, (double)is_close_encounter * numerical_soln_to_close_encounter.vel.z);
    cluster.sync();

    // final "kicks"
    // we only read in our position and velocity, so we don't need to sync
    position_after_main_body_kick =
        main_body_kinetic(positions, num_massive_bodies, half_dt, &mapped_block);
    positions[idx] = position_after_main_body_kick;
    cluster.sync();
    velocity_after_body_interaction =
        body_interaction_kick((PosVel){.pos = positions[idx], .vel = velocities[idx]}, num_massive_bodies, half_dt, &mapped_block);
    velocities[idx] = velocity_after_body_interaction;
  }

  output_positions_hbm[hbm_idx] = positions[idx];

  cluster.sync();
  // convert back to heliocentric coordinates
  velocities[idx] = democratic_heliocentric_conversion((PosVel){.pos = positions[idx], .vel = velocities[idx]}, num_massive_bodies, &mapped_block, true);
  //  convert back to elements
  elements_from_cartesian(
      positions,
      velocities,
      &inclination,
      &longitude_of_ascending_node,
      &argument_of_perihelion,
      &mean_anomaly,
      &eccentricity,
      &semi_major_axis);
  // copy elements to hbm, this is so that the next batch iteration uses these values to pick up where we left off
  vec_semi_major_axis_hbm[hbm_idx] = semi_major_axis;
  vec_eccentricity_hbm[hbm_idx] = eccentricity;
  vec_mean_anomaly_hbm[hbm_idx] = mean_anomaly;
  vec_argument_of_perihelion_hbm[hbm_idx] = argument_of_perihelion;
  vec_inclination_hbm[hbm_idx] = inclination;
  vec_longitude_of_ascending_node_hbm[hbm_idx] = longitude_of_ascending_node;
}

#endif
