// core numerical integration kernel
#ifndef __SIM_CUH__
#define __SIM_CUH__
#include "constants.cuh"

// solves kepler's equation for the eccentric anomaly E
__device__
double danby_burkardt(double mean_anomaly, double eccentricity) {
    // init eccentric anomaly to mean anomaly
    double E = mean_anomaly;
    for(int i = 0; i < MAX_ITERATIONS_ROOT_FINDING; i++) {
        double sin_E, cos_E;
        sincos(E, &sin_E, &cos_E);
        double e_sin = eccentricity * sin_E;
        double f = E - e_sin - mean_anomaly;
        if(fabs(f) < CUTOFF) break;
        double e_cos = eccentricity * cos_E;
        double f_prime = 1 - e_cos; 
        double dE =  - f / f_prime;
        dE = - f / (f_prime + dE*e_sin / 2.00);
        // quartic convergence
        dE = - f / ((f_prime + dE*e_sin / 2.00) + (dE*dE*e_cos / 6.00));
        // quintic convergence
        dE = - f / ((f_prime + dE*e_sin / 2.00) + (dE*dE*e_cos / 6.00) - (dE*dE*dE*e_sin / 24.00));
        E += dE;    
    }

    return E;
}

__device__
double changeover(double r_ij) {
    double r_crit = 0.001;
    double y = (r_ij - 0.1*r_crit) / (0.9*r_crit);
    double K = y*y / (2*y*y - 2*y + 1);
    // trying to avoid branching
    double gtz = (double)(y > 0);
    double gto = (double)(y > 1);
    double valid = (double)(y <= 1 && y >= 0);
    return K * gtz * valid + gto;
}

__device__
void cartesian_from_elements(
    double* vec_inclination, 
    double* vec_longitude_of_ascending_node, 
    double* vec_argument_of_perihelion, 
    double* vec_mean_anomaly,
    double* vec_eccentricity,
    double* vec_semi_major_axis,
    double3* current_positions,
    double3* current_velocities
) {
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
    double d11 =  z1 - z4*cos_i;
    double d12 =  z2 + z3*cos_i;
    double d13 = sin_a * sin_i;
    double d21 = -z3 - z2*cos_i;
    double d22 = -z4 + z1*cos_i;
    double d23 = cos_a * sin_i;
   
    double romes = sqrt(1 - eccentricity*eccentricity);
    double eccentric_anomaly = danby_burkardt(mean_anomaly, eccentricity);
    double sin_e, cos_e;
    sincos(eccentric_anomaly, &sin_e, &cos_e);
    z1 = semi_major_axis * (cos_e - eccentricity);
    z2 = semi_major_axis * romes * sin_e;
    eccentric_anomaly = sqrt(G/semi_major_axis) / (1.0 - eccentricity*cos_e);
    z3 = -sin_e * eccentric_anomaly;
    z4 = romes * cos_e * eccentric_anomaly;
    
    current_positions[idx] = make_double3(d11 * z1 + d21 * z2, d12 * z1 + d22 * z2, d13 * z1 + d23 * z2);
    current_velocities[idx] = make_double3(d11 * z3 + d21 * z4, d12 * z3 + d22 * z4, d13 * z3 + d23 * z4);
}

__device__ double3 cross(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ double magnitude(const double3& a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ double3 magnitude_squared(const double3& a) {
    return make_double3(a.x * a.x, a.y * a.y, a.z * a.z);
}

__device__ double stable_acos(double x) {
    double alto = (double)(fabs(x) <= 1.00);
    // so, basically this computes acos(x) if x within bounds
    // otherwise it computes acos(+/- 1.00)
    // x*alto evals to x when x inside bounds
    // copysign((1.00 - alto), x) evals to +/- 1.00 when x outside bounds
    return acos(x*alto + copysign((1.00 - alto), x));
} 

__device__ double stable_asin(double x) {
    double alto = (double)(fabs(x) <= 1.00);
    return asin(x*alto + copysign((1.00 - alto), x));
}

__device__ double stable_sqrt(double x) {
    double gtz = (double)(x >= 0.00);
    // eval to zero if x less than zero
    return sqrt(x*gtz);
}

__device__
void elements_from_cartesian(
    double3* current_positions,
    double3* current_velocities,
    double* vec_inclination, 
    double* vec_longitude_of_ascending_node, 
    double* vec_argument_of_perihelion, 
    double* vec_mean_anomaly,
    double* vec_eccentricity,
    double* vec_semi_major_axis
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double3 current_p = current_positions[idx];
    double3 current_v = current_velocities[idx];
    double3 angular_momentum = cross(current_p, current_v);
    double epsilon = 1e-8;
    double h_sq = magnitude_squared(angular_momentum).x + magnitude_squared(angular_momentum).y + magnitude_squared(angular_momentum).z + epsilon;
    double inclination = stable_acos(angular_momentum.z / sqrt(h_sq));
    // TODO: find way to do this without branching
    double longitude_of_ascending_node = atan2(angular_momentum.x, -angular_momentum.y == 0.0 ? 0.0 : -angular_momentum.y);
    double v_sq = magnitude_squared(current_v).x + magnitude_squared(current_v).y + magnitude_squared(current_v).z;
    double r = magnitude(current_p);
    double s = h_sq / G;
    double eccentricity = stable_sqrt(1 + s * ((v_sq / G) - (2.00 / r)));
    double perihelion_distance = s / (1.00 + eccentricity);
    
    double cos_E_anomaly_denom = (eccentricity*G);
    double cos_E_anomaly = (v_sq*r - G) / cos_E_anomaly_denom;
    double E_anomaly = stable_acos(cos_E_anomaly);
    double M_anomaly = E_anomaly - eccentricity * sin(E_anomaly);
    double cos_f = (s - r ) / (eccentricity * r);
    double f = stable_acos(cos_f);

    double cos_i = cos(inclination);
    double to = -angular_momentum.x / angular_momentum.y;
    double temp = (1.00 - cos_i) * to;
    double temp2 = to * to;
    double true_longitude = atan2((current_p.y * (1.00 + temp2 * cos_i) - current_p.x * temp), (current_p.x * (temp2 + cos_i) - current_p.y * temp));

    double p = true_longitude - f;
    p = fmod(p + TWOPI + TWOPI, TWOPI);
    double argument_of_perihelion = p - longitude_of_ascending_node;
    double semi_major_axis = perihelion_distance / (1.00 - eccentricity);

    vec_inclination[idx] = inclination;
    vec_longitude_of_ascending_node[idx] = longitude_of_ascending_node;
    vec_argument_of_perihelion[idx] = argument_of_perihelion;
    vec_mean_anomaly[idx] = M_anomaly;
    vec_eccentricity[idx] = eccentricity;
    vec_semi_major_axis[idx] = semi_major_axis;
}

__device__
void body_interaction_kick(double3* positions, double3* velocities, double* masses, double dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double3 acc = make_double3(0.0, 0.0, 0.0);
    double dist_x, dist_y, dist_z = 0.0;
    for(int i = 0; i < NUM_BODIES; i++) {
        if(i == idx) continue;
       // 3-vec displacement, let r = x, y, z, this is the direction of the acceleration
        // it is directed towards the other body since gravity is attractive
        dist_x = positions[i].x - positions[idx].x;
        dist_y = positions[i].y - positions[idx].y;
        dist_z = positions[i].z - positions[idx].z;
        double epsilon = 1e-8;
        double r = sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
        // // magnitude of acceleration = mass_of_other_body * G / |r|^3
        double weighted_acceleration = changeover(r) * masses[i+1] * G / pow(r + epsilon, 3);
        // // accumulate total acceleration due to all bodies, except self
        acc.x += weighted_acceleration * dist_x;
        acc.y += weighted_acceleration * dist_y;
        acc.z += weighted_acceleration * dist_z;
    }

    // update momenta (velocity here) with total acceleration
    velocities[idx].x += acc.x * dt;
    velocities[idx].y += acc.y * dt;
    velocities[idx].z += acc.z * dt;     
}

__device__
void main_body_kinetic(double3* positions, double3* velocities, double* masses, double dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double3 p = make_double3(0.0, 0.0, 0.0);
    // calculate total momentum of all bodies
    for(int i = 1; i < blockDim.x + 1; i++) {
        p.x += velocities[i-1].x * masses[i];
        p.y += velocities[i-1].y * masses[i];
        p.z += velocities[i-1].z * masses[i];
    }

    double scaling_factor = dt/(masses[0]);
    positions[idx].x += p.x * scaling_factor;
    positions[idx].y += p.y * scaling_factor;
    positions[idx].z += p.z * scaling_factor;
}

// this ensures that the sun is in a reference frame in which it is stationary and at the origin
__device__
void convert_to_democratic_heliocentric_coordinates(double3* positions, double3* velocities, double* masses) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double total_mass = 0.0;
    double3 mass_weighted_v = make_double3(0.0, 0.0, 0.0);
    for(int i = 0; i < blockDim.x; i++) {
        total_mass += masses[i + 1];
        mass_weighted_v.x += masses[i + 1] * velocities[i].x;
        mass_weighted_v.y += masses[i + 1] * velocities[i].y;
        mass_weighted_v.z += masses[i + 1] * velocities[i].z;
    }

    double scaling_factor = 1.00 / (total_mass + masses[0]);
    mass_weighted_v.x *= scaling_factor;
    mass_weighted_v.y *= scaling_factor;
    mass_weighted_v.z *= scaling_factor;

    velocities[idx].x -= mass_weighted_v.x;
    velocities[idx].y -= mass_weighted_v.y;
    velocities[idx].z -= mass_weighted_v.z;
    
}

__global__ 
void mercurius_keplerian_solver(
    double* vec_argument_of_perihelion_hbm,
    double* vec_mean_anomaly_hbm,
    double* vec_eccentricity_hbm,
    double* vec_semi_major_axis_hbm,
    double* vec_inclination_hbm,
    double* vec_longitude_of_ascending_node_hbm,
    double* vec_masses_hbm,
    double3* output_positions,
    double dt,
    int NUM_TIMESTEPS 
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // declare buffers for positions in SRAM
    __shared__ double3 positions[NUM_BODIES];  
    __shared__ double3 velocities[NUM_BODIES];
    __shared__ double masses[NUM_BODIES + 1];
    __shared__ double vec_inclination[NUM_BODIES];
    __shared__ double vec_longitude_of_ascending_node[NUM_BODIES];
    __shared__ double vec_argument_of_perihelion[NUM_BODIES];
    __shared__ double vec_mean_anomaly[NUM_BODIES];
    __shared__ double vec_eccentricity[NUM_BODIES];
    __shared__ double vec_semi_major_axis[NUM_BODIES];

    // copy data to shared memory
    // special case to avoid race condition
    if (idx == 0) masses[0] = vec_masses_hbm[0];

    masses[idx+1] = vec_masses_hbm[idx+1]; 
    vec_argument_of_perihelion[idx] = vec_argument_of_perihelion_hbm[idx];
    vec_mean_anomaly[idx] = vec_mean_anomaly_hbm[idx];
    vec_eccentricity[idx] = vec_eccentricity_hbm[idx];
    vec_semi_major_axis[idx] = vec_semi_major_axis_hbm[idx];
    vec_inclination[idx] = vec_inclination_hbm[idx];
    vec_longitude_of_ascending_node[idx] = vec_longitude_of_ascending_node_hbm[idx];
    __syncthreads(); 
    // initially populate positions and velocities
    cartesian_from_elements(
        vec_inclination,
        vec_longitude_of_ascending_node,
        vec_argument_of_perihelion,
        vec_mean_anomaly,
        vec_eccentricity,
        vec_semi_major_axis,
        positions,
        velocities
    );

    __syncthreads();
    // convert to democratic heliocentric coordinates
    convert_to_democratic_heliocentric_coordinates(positions, velocities, masses);

    for(int i = 0; i < NUM_TIMESTEPS; i++) {
        __syncthreads();
        body_interaction_kick(positions, velocities, masses, dt/2.00);
        __syncthreads();
        main_body_kinetic(positions, velocities, masses, dt/2.00);
        double semi_major_axis = vec_semi_major_axis[idx];
        double n = 1.00 / (semi_major_axis * semi_major_axis * semi_major_axis); 
        __syncthreads(); 
        elements_from_cartesian(
            positions,
            velocities,
            vec_inclination,
            vec_longitude_of_ascending_node,
            vec_argument_of_perihelion,
            vec_mean_anomaly,
            vec_eccentricity,
            vec_semi_major_axis
        );
        __syncthreads();
        // // advance mean anomaly, this is essentially advancing to the next timestep
        vec_mean_anomaly[idx] = fmod(n * dt + vec_mean_anomaly[idx], TWOPI);
        __syncthreads();
        cartesian_from_elements(
            vec_inclination,
            vec_longitude_of_ascending_node,
            vec_argument_of_perihelion,
            vec_mean_anomaly,
            vec_eccentricity,
            vec_semi_major_axis,
            positions,
            velocities
        );

        __syncthreads(); 
        main_body_kinetic(positions, velocities, masses, dt/2.00);
        __syncthreads();
        body_interaction_kick(positions, velocities, masses, dt/2.00);

        // basically the layout here is:
        // [[body0, body1, body2, ...], [body0, body1, body2, ...], ...]
        // where each subarray is a timestep
        // so we need to index into the timestep and then add idx to index a particular body
        output_positions[i* blockDim.x + idx] = positions[idx];
        __syncthreads();
    }
    
}

#endif