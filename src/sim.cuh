// core numerical integration kernel
#ifndef __SIM_CUH__
#define __SIM_CUH__
#include "constants.cuh"
#include "simutils.cuh"

/*
This file contains the core numerical integration kernel and associated helper
functions for my implementation of the Mercury N-body Integrator.
*/

// cache powers of two for richardson extrapolation
__device__ double *get_pow_two_table()
{
  static double pow_two_table[MAX_ROWS_RICHARDSON + 1];
  static bool initialized = false;
  if (!initialized)
  {
    for (int i = 1; i < MAX_ROWS_RICHARDSON; i++)
    {
      pow_two_table[i] = exp2(i);
    }
    initialized = true;
  }
  return pow_two_table;
}

__device__ double3 cross(const double3 &a, const double3 &b)
{
  return make_double3(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

__device__ double stable_sqrt(double x)
{
  double gtz = (double)(x >= 0.00);
  // eval to zero if x less than zero
  return sqrt(x * gtz);
}

__device__ void efficient_magnitude(double *mag, double *mag_sq, const double3 &a)
{
  *mag_sq = a.x * a.x + a.y * a.y + a.z * a.z;
  *mag = stable_sqrt(*mag_sq);
}

__device__ double magnitude(const double3 &a)
{
  return stable_sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ double magnitude_squared(const double3 &a) { return a.x * a.x + a.y * a.y + a.z * a.z; }

__device__ double stable_acos(double x)
{
  double alto = (double)(fabs(x) <= 1.00);
  // so, basically this computes acos(x) if x within bounds
  // otherwise it computes acos(+/- 1.00)
  // x*alto evals to x when x inside bounds
  // copysign((1.00 - alto), x) evals to +/- 1.00 when x outside bounds
  return acos(x * alto + copysign((1.00 - alto), x));
}

__device__ double stable_asin(double x)
{
  double alto = (double)(fabs(x) <= 1.00);
  return asin(x * alto + copysign((1.00 - alto), x));
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
    double e_sin = eccentricity * sin_E;
    double f = E - e_sin - mean_anomaly;
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
  return magnitude(make_double3(a.x - b.x, a.y - b.y, a.z - b.z));
}

__device__ double fetch_r_crit(
    PosVel current_coords,
    // read only
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    int idx_other,
    double dt)
{
  // // anywhere between 3-10
  // double n1 = 3.50;
  // // anywhere between 0.3-2.0
  // double n2 = 2;
  // double r1 = magnitude(current_coords.pos);
  // double r2 = magnitude(positions[idx_other]);
  // double v1 = magnitude(current_coords.vel);
  // double v2 = magnitude(velocities[idx_other]);
  // double mutual_hill_radius =
  //     cbrt(masses[threadIdx.x + blockIdx.x * blockDim.x] + masses[idx_other] / 3.00) * (r1 + r2) / 2.00;
  // double vmax = max(v1, v2);
  // return max(n1 * mutual_hill_radius, n2 * vmax * dt);
  return 0.06;
}

__device__ KR_Crit changeover(
    PosVel current_coords,
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    int idx_other,
    double dt)
{
  double r_crit = fetch_r_crit(current_coords, positions, velocities, masses, idx_other, dt);
  KR_Crit res;
  res.r_crit = r_crit;

  double r_ij = dist(positions[idx_other], current_coords.pos);
  double y = (r_ij - 0.1 * r_crit) / (0.9 * r_crit);
  double K = y * y / (2 * y * y - 2 * y + 1);
  // trying to avoid branching
  double gtz = (double)(y > 0);
  double gto = (double)(y > 1);
  double valid = (double)(y <= 1 && y >= 0);
  res.K = K * gtz * valid + gto;
  return res;
}

__device__ void cartesian_from_elements(double *vec_inclination,
                                        double *vec_longitude_of_ascending_node,
                                        double *vec_argument_of_perihelion,
                                        double *vec_mean_anomaly,
                                        double *vec_eccentricity,
                                        double *vec_semi_major_axis,
                                        double3 *current_positions,
                                        double3 *current_velocities)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double inclination = vec_inclination[idx];
  double longitude_of_ascending_node = vec_longitude_of_ascending_node[idx];
  double argument_of_perihelion = vec_argument_of_perihelion[idx];
  double mean_anomaly = vec_mean_anomaly[idx];
  double eccentricity = vec_eccentricity[idx];
  double semi_major_axis = vec_semi_major_axis[idx];

  double cos_i, sin_i, cos_o, sin_o, cos_a, sin_a;
  sincos(inclination, &sin_i, &cos_i);
  sincos(longitude_of_ascending_node, &sin_o, &cos_o);
  sincos(argument_of_perihelion, &sin_a, &cos_a);

  double z1 = cos_a * cos_o;
  double z2 = cos_a * sin_o;
  double z3 = sin_a * cos_o;
  double z4 = sin_a * sin_o;
  double d11 = z1 - z4 * cos_i;
  double d12 = z2 + z3 * cos_i;
  double d13 = sin_a * sin_i;
  double d21 = -z3 - z2 * cos_i;
  double d22 = -z4 + z1 * cos_i;
  double d23 = cos_a * sin_i;

  double romes = stable_sqrt(1 - eccentricity * eccentricity);
  double eccentric_anomaly = danby_burkardt(mean_anomaly, eccentricity);
  double sin_e, cos_e;
  sincos(eccentric_anomaly, &sin_e, &cos_e);
  z1 = semi_major_axis * (cos_e - eccentricity);
  z2 = semi_major_axis * romes * sin_e;
  eccentric_anomaly =
      stable_sqrt(1.00 / semi_major_axis) / (1.0 - eccentricity * cos_e);
  z3 = -sin_e * eccentric_anomaly;
  z4 = romes * cos_e * eccentric_anomaly;

  current_positions[idx] = make_double3(d11 * z1 + d21 * z2,
                                        d12 * z1 + d22 * z2,
                                        d13 * z1 + d23 * z2);
  current_velocities[idx] = make_double3(d11 * z3 + d21 * z4,
                                         d12 * z3 + d22 * z4,
                                         d13 * z3 + d23 * z4);
}

__device__ void elements_from_cartesian(double3 *current_positions,
                                        double3 *current_velocities,
                                        double *vec_inclination,
                                        double *vec_longitude_of_ascending_node,
                                        double *vec_argument_of_perihelion,
                                        double *vec_mean_anomaly,
                                        double *vec_eccentricity,
                                        double *vec_semi_major_axis)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double3 current_p = current_positions[idx];
  double3 current_v = current_velocities[idx];
  double3 angular_momentum = cross(current_p, current_v);
  double epsilon = 1e-8;
  double h_sq = magnitude_squared(angular_momentum) + epsilon;
  double inclination = stable_acos(angular_momentum.z / stable_sqrt(h_sq));
  double longitude_of_ascending_node = atan2(angular_momentum.x, -angular_momentum.y);

  double v_sq = magnitude_squared(current_v);
  double r = magnitude(current_p);
  double s = h_sq;
  double eccentricity = stable_sqrt(1 + s * (v_sq - (2.00 / r)));
  double perihelion_distance = s / (1.00 + eccentricity);

  // true longitude
  double cos_i = cos(inclination);
  double true_longitude;
  if (angular_momentum.y != 0)
  {
    double to = -angular_momentum.x / angular_momentum.y;
    double temp = (1.00 - cos_i) * to;
    double temp2 = to * to;
    true_longitude =
        atan2((current_p.y * (1.00 + temp2 * cos_i) - current_p.x * temp),
              (current_p.x * (temp2 + cos_i) - current_p.y * temp));
  }
  else
  {
    true_longitude = atan2(current_p.y * cos_i, current_p.x);
  }

  // mean anomaly and longitude of perihelion
  double p;
  double M_anomaly;
  if (eccentricity < epsilon)
  {
    p = 0;
    M_anomaly = true_longitude;
  }
  else
  {
    double cos_E_anomaly = (v_sq * r - 1) / eccentricity;
    double E_anomaly = stable_acos(cos_E_anomaly);
    M_anomaly = E_anomaly - eccentricity * sin(E_anomaly);
    double cos_f = (s - r) / (eccentricity * r);
    double f = stable_acos(cos_f);
    p = true_longitude - f;
    p = fmod(p + TWOPI + TWOPI, TWOPI);
  }

  // argument of perihelion
  double argument_of_perihelion = p - longitude_of_ascending_node;
  double semi_major_axis = perihelion_distance / (1.00 - eccentricity);

  vec_inclination[idx] = inclination;
  vec_longitude_of_ascending_node[idx] = longitude_of_ascending_node;
  vec_argument_of_perihelion[idx] = argument_of_perihelion;
  vec_mean_anomaly[idx] = M_anomaly;
  vec_eccentricity[idx] = eccentricity;
  vec_semi_major_axis[idx] = semi_major_axis;
}

// this function returns an updated velocity
__device__ double3 body_interaction_kick(
    PosVel current_coords,
    // position and velocity arrays are read-only
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    double dt,
    bool possible_close_encounter = false)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const double3 my_position = current_coords.pos;
  double3 acc = make_double3(0.0, 0.0, 0.0);
  double3 dist;
  for (int i = 0; i < blockDim.x; i++)
  {
    if (i == idx) continue;
    // 3-vec displacement, let r = x, y, z, this is the direction of the
    // acceleration it is directed towards the other body since gravity is
    // attractive
    dist.x = positions[i].x - my_position.x;
    dist.y = positions[i].y - my_position.y;
    dist.z = positions[i].z - my_position.z;

    double r_sq;
    double r;
    efficient_magnitude(&r, &r_sq, dist);

    // compute both K and r_crit
    KR_Crit changeover_vals = changeover(current_coords, positions, velocities, masses, i, dt);

    double changeover_weight = changeover_vals.K;
    double r_crit = changeover_vals.r_crit;
    if (possible_close_encounter)
    {
      if (r < r_crit)
      {
        // 1 - K weighting if close encounter
        changeover_weight = 1 - changeover_weight;
      }
    }

    // add smoothing constant

    r_sq += SMOOTHING_CONSTANT_SQUARED;
    // the (r^2 + s^2) comes from inv sq law w/ smoothing, and the other r bc
    // direction is normalized
    double force_denom = r_sq * r;

    double weighted_acceleration =
        changeover_weight * masses[i] / force_denom;
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

__device__ double3 main_body_kinetic(const double3 *positions,
                                     const double3 *velocities,
                                     const double *masses,
                                     double dt)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double3 my_position = positions[idx];
  double3 p = make_double3(0.0, 0.0, 0.0);
  // calculate total momentum of all bodies
  for (int i = 0; i < blockDim.x + 1; i++)
  {
    p.x = fma(masses[i], velocities[i].x, p.x);
    p.y = fma(masses[i], velocities[i].y, p.y);
    p.z = fma(masses[i], velocities[i].z, p.z);
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
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    double dt)

{
  /*
  updates the velocities of all bodies based on the mass distribution of all
  bodies (incl the sun)
  */

  const double3 my_position = current_coords.pos;
  // the body interaction kick updates velocities based on the masses of all
  // minor bodies
  double3 my_velocity = body_interaction_kick(current_coords, positions, velocities, masses, dt, true);
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
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    double dt,
    int N)
{
  /*
  returns an updated position based on the modified midpoint method.
  */

  double subdelta = dt / N;
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
      update_my_velocity_total(current_coords, positions, velocities, masses, subdelta);

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
        update_my_velocity_total(current_coords, positions, velocities, masses, subdelta);
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

__device__ bool is_converged(PosVel a_1, PosVel a_0)
{
  PosVel diff;
  diff.pos.x = a_1.pos.x - a_0.pos.x;
  diff.pos.y = a_1.pos.y - a_0.pos.y;
  diff.pos.z = a_1.pos.z - a_0.pos.z;
  diff.vel.x = a_1.vel.x - a_0.vel.x;
  diff.vel.y = a_1.vel.y - a_0.vel.y;
  diff.vel.z = a_1.vel.z - a_0.vel.z;
  double mag_vel = magnitude_squared(diff.vel);
  double mag_pos = magnitude_squared(diff.pos);
  return (mag_vel + mag_pos) < BULIRSCH_STOER_TOLERANCE;
}

__device__ PosVel richardson_extrapolation(
    PosVel current_coords,
    const double3 *positions,
    const double3 *velocities,
    const double *masses,
    double dt)
{
  double *pow_two_table = get_pow_two_table();
  int N = 1;
  PosVel out;
  PosVel buffer[MAX_ROWS_RICHARDSON][MAX_ROWS_RICHARDSON];
  // for our first approx, we use the OG dt
  buffer[0][0] = modified_midpoint(current_coords, positions, velocities, masses, dt, N);
  for (int i = 1; i < MAX_ROWS_RICHARDSON; i++)
  {
    N = pow_two_table[i];
    buffer[i][0] = modified_midpoint(current_coords, positions, velocities, masses, dt, N);
    for (int j = 1; j <= i; j++)
    {
      double pow2 = pow_two_table[j];

      // for positions
      buffer[i][j].pos.x =
          fma(pow2, buffer[i][j - 1].pos.x, -buffer[i - 1][j - 1].pos.x);
      buffer[i][j].pos.y =
          fma(pow2, buffer[i][j - 1].pos.y, -buffer[i - 1][j - 1].pos.y);
      buffer[i][j].pos.z =
          fma(pow2, buffer[i][j - 1].pos.z, -buffer[i - 1][j - 1].pos.z);

      // for velocities
      buffer[i][j].vel.x =
          fma(pow2, buffer[i][j - 1].vel.x, -buffer[i - 1][j - 1].vel.x);
      buffer[i][j].vel.y =
          fma(pow2, buffer[i][j - 1].vel.y, -buffer[i - 1][j - 1].vel.y);
      buffer[i][j].vel.z =
          fma(pow2, buffer[i][j - 1].vel.z, -buffer[i - 1][j - 1].vel.z);

      pow2 -= 1;

      buffer[i][j].pos.x /= pow2;
      buffer[i][j].pos.y /= pow2;
      buffer[i][j].pos.z /= pow2;

      buffer[i][j].vel.x /= pow2;
      buffer[i][j].vel.y /= pow2;
      buffer[i][j].vel.z /= pow2;
    }

    if (is_converged(buffer[i][i], buffer[i - 1][i - 1]))
    {
      out.pos = buffer[i][i].pos;
      out.vel = buffer[i][i].vel;
      return out;
    }
  }

  out.pos = buffer[MAX_ROWS_RICHARDSON - 1][MAX_ROWS_RICHARDSON - 1].pos;
  out.vel = buffer[MAX_ROWS_RICHARDSON - 1][MAX_ROWS_RICHARDSON - 1].vel;
  return out;
}

// this ensures that the sun is in a reference frame in which it is stationary
// and at the origin
__device__ void democratic_heliocentric_conversion(
    double3 *positions,
    double3 *velocities,
    double *masses,
    bool reverse = false)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double total_mass = 0.0;
  double not_reverse_d = (double)(!reverse);
  double add_or_sub = pow(-1, not_reverse_d);

  double3 mass_weighted_v = make_double3(0.0, 0.0, 0.0);
  for (int i = 0; i < blockDim.x; i++)
  {
    total_mass += masses[i];
    mass_weighted_v.x = fma(masses[i], velocities[i].x, mass_weighted_v.x);
    mass_weighted_v.y = fma(masses[i], velocities[i].y, mass_weighted_v.y);
    mass_weighted_v.z = fma(masses[i], velocities[i].z, mass_weighted_v.z);
  }

  // prevent race condition, ensure all threads have finished reading old
  // velocities
  __syncthreads();

  // if we are performing the reverse conversion, we just need to divide by the main mass = 1
  double scaling_factor = 1.00 / ((not_reverse_d * total_mass) + 1.00);

  mass_weighted_v.x *= scaling_factor;
  mass_weighted_v.y *= scaling_factor;
  mass_weighted_v.z *= scaling_factor;

  // if we are performing the reverse conversion, we need to add not subtract
  velocities[idx].x = fma(add_or_sub, mass_weighted_v.x, velocities[idx].x);
  velocities[idx].y = fma(add_or_sub, mass_weighted_v.y, velocities[idx].y);
  velocities[idx].z = fma(add_or_sub, mass_weighted_v.z, velocities[idx].z);
}

__device__ bool close_encounter_p(
    double3 *positions,
    double3 *velocities,
    double *masses,
    double dt)
{
  // get indices of bodies i am undergoing close encounters with
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < blockDim.x; i++)
  {
    if (i == idx) continue;
    double3 r_ij;
    r_ij.x = positions[i].x - positions[idx].x;
    r_ij.y = positions[i].y - positions[idx].y;
    r_ij.z = positions[i].z - positions[idx].z;
    double r_ij_mag = magnitude(r_ij);
    double r_crit = fetch_r_crit((PosVel){.pos = positions[idx], .vel = velocities[idx]}, positions, velocities, masses, i, dt);
    if (r_ij_mag < r_crit)
    {
      return true;
    }
  }
  return false;
}

__global__ void mercurius_solver(double *vec_argument_of_perihelion_hbm,
                                 double *vec_mean_anomaly_hbm,
                                 double *vec_eccentricity_hbm,
                                 double *vec_semi_major_axis_hbm,
                                 double *vec_inclination_hbm,
                                 double *vec_longitude_of_ascending_node_hbm,
                                 double *vec_masses_hbm,
                                 double3 *output_positions,
                                 double dt)

{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // declare SRAM buffer
  extern __shared__ char total_memory[];
  double3 *positions = (double3 *)total_memory;
  double3 *velocities = (double3 *)(positions + blockDim.x);
  double *masses = (double *)(velocities + blockDim.x);
  double *vec_inclination = (double *)(masses + blockDim.x);
  double *vec_longitude_of_ascending_node =
      (double *)(vec_inclination + blockDim.x);
  double *vec_argument_of_perihelion =
      (double *)(vec_longitude_of_ascending_node + blockDim.x);
  double *vec_mean_anomaly =
      (double *)(vec_argument_of_perihelion + blockDim.x);
  double *vec_eccentricity = (double *)(vec_mean_anomaly + blockDim.x);
  double *vec_semi_major_axis = (double *)(vec_eccentricity + blockDim.x);

  masses[idx] = vec_masses_hbm[idx];
  vec_argument_of_perihelion[idx] = vec_argument_of_perihelion_hbm[idx];
  vec_mean_anomaly[idx] = vec_mean_anomaly_hbm[idx];
  vec_eccentricity[idx] = vec_eccentricity_hbm[idx];
  vec_semi_major_axis[idx] = vec_semi_major_axis_hbm[idx];
  vec_inclination[idx] = vec_inclination_hbm[idx];
  vec_longitude_of_ascending_node[idx] =
      vec_longitude_of_ascending_node_hbm[idx];

  double half_dt = 0.5 * dt;
  // initially populate positions and velocities
  cartesian_from_elements(vec_inclination,
                          vec_longitude_of_ascending_node,
                          vec_argument_of_perihelion,
                          vec_mean_anomaly,
                          vec_eccentricity,
                          vec_semi_major_axis,
                          positions,
                          velocities);
  __syncthreads();
  // convert to democratic heliocentric coordinates
  democratic_heliocentric_conversion(positions, velocities, masses);

  for (int i = 0; i < BATCH_SIZE; i++)
  {
    // first "kicks"
    __syncthreads();
    double3 velocity_after_body_interaction =
        body_interaction_kick((PosVel){.pos = positions[idx], .vel = velocities[idx]}, positions, velocities, masses, half_dt);
    __syncthreads();
    velocities[idx] = velocity_after_body_interaction;
    __syncthreads();
    double3 position_after_main_body_kick =
        main_body_kinetic(positions, velocities, masses, half_dt);
    __syncthreads();
    positions[idx] = position_after_main_body_kick;
    __syncthreads();

    double semi_major_axis = vec_semi_major_axis[idx];
    double n = rsqrt(semi_major_axis * semi_major_axis * semi_major_axis);

    // SOLUTION TO MAIN (largest) HAMILTONIAN
    // if i am undergoing a close encounter with anyone
    // use bulirsch-stoer aka richardson extrapolation w/ mod midpoint
    PosVel numerical_soln_to_close_encounter;
    bool is_close_encounter =
        close_encounter_p(positions, velocities, masses, dt);
    if (is_close_encounter)
    {
      // directly updates positions and velocities by dt
      numerical_soln_to_close_encounter =
          // modified_midpoint((PosVel){.pos = positions[idx], .vel = velocities[idx]}, positions, velocities, masses, dt, 1);
          richardson_extrapolation((PosVel){.pos = positions[idx], .vel = velocities[idx]}, positions, velocities, masses, dt);
    }
    else
    {
      // side-effect here: updates eccentric anomaly directly –– analytical soln
      // to Kepler's equation, computed using a numerical root-finding algo
      elements_from_cartesian(positions,
                              velocities,
                              vec_inclination,
                              vec_longitude_of_ascending_node,
                              vec_argument_of_perihelion,
                              vec_mean_anomaly,
                              vec_eccentricity,
                              vec_semi_major_axis);
      // advance mean anomaly, this is essentially advancing to the next
      // timestep
      vec_mean_anomaly[idx] = fmod(fma(n, dt, vec_mean_anomaly[idx]), TWOPI);
      cartesian_from_elements(vec_inclination,
                              vec_longitude_of_ascending_node,
                              vec_argument_of_perihelion,
                              vec_mean_anomaly,
                              vec_eccentricity,
                              vec_semi_major_axis,
                              positions,
                              velocities);
    }

    // separating out calculation with update ensures that no race conditions
    // occur
    __syncthreads();
    double not_close_encounter = (1.00 - (double)is_close_encounter);
    positions[idx].x =
        not_close_encounter * positions[idx].x +
        (double)is_close_encounter * numerical_soln_to_close_encounter.pos.x;
    positions[idx].y =
        not_close_encounter * positions[idx].y +
        (double)is_close_encounter * numerical_soln_to_close_encounter.pos.y;
    positions[idx].z =
        not_close_encounter * positions[idx].z +
        (double)is_close_encounter * numerical_soln_to_close_encounter.pos.z;
    velocities[idx].x =
        not_close_encounter * velocities[idx].x +
        (double)is_close_encounter * numerical_soln_to_close_encounter.vel.x;
    velocities[idx].y =
        not_close_encounter * velocities[idx].y +
        (double)is_close_encounter * numerical_soln_to_close_encounter.vel.y;
    velocities[idx].z =
        not_close_encounter * velocities[idx].z +
        (double)is_close_encounter * numerical_soln_to_close_encounter.vel.z;
    __syncthreads();

    // final "kicks"
    __syncthreads();
    position_after_main_body_kick =
        main_body_kinetic(positions, velocities, masses, half_dt);
    __syncthreads();
    positions[idx] = position_after_main_body_kick;
    __syncthreads();
    velocity_after_body_interaction =
        body_interaction_kick((PosVel){.pos = positions[idx], .vel = velocities[idx]}, positions, velocities, masses, half_dt);
    __syncthreads();
    velocities[idx] = velocity_after_body_interaction;

    // basically the layout here is:
    // [[body0, body1, body2, ...], [body0, body1, body2, ...], ...]
    // where each subarray is a timestep
    // so we need to index into the timestep and then add idx to index a
    // particular body
    output_positions[i * blockDim.x + idx] = positions[idx];
  }

  __syncthreads();
  // convert back to heliocentric coordinates
  democratic_heliocentric_conversion(positions, velocities, masses, true);
  // convert back to elements
  elements_from_cartesian(positions,
                          velocities,
                          vec_inclination,
                          vec_longitude_of_ascending_node,
                          vec_argument_of_perihelion,
                          vec_mean_anomaly,
                          vec_eccentricity,
                          vec_semi_major_axis);

  // copy elements to hbm, this is so that the next batch iteration uses these values to pick up where we left off
  vec_semi_major_axis_hbm[idx] = vec_semi_major_axis[idx];
  vec_eccentricity_hbm[idx] = vec_eccentricity[idx];
  vec_mean_anomaly_hbm[idx] = vec_mean_anomaly[idx];
  vec_argument_of_perihelion_hbm[idx] = vec_argument_of_perihelion[idx];
  vec_inclination_hbm[idx] = vec_inclination[idx];
  vec_longitude_of_ascending_node_hbm[idx] = vec_longitude_of_ascending_node[idx];
}

#endif