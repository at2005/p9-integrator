#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ double CUTOFF = 1e-13;
__device__ int MAX_ITERATIONS_ROOT_FINDING = 20;
__device__ int NUM_TIMESTEPS_KEPLER = 100000;
__device__ double G = 6.6743e-11;
__device__ double TWOPI = 6.283185307179586476925286766559005768394338798750211641949;

// solves kepler's equation for the eccentric anomaly E
__device__
double danby_burkardt(double mean_anomaly, double eccentricity) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // init eccentric anomaly to mean anomaly
    double E = mean_anomaly;
    for(int i = 0; i < MAX_ITERATIONS_ROOT_FINDING; i++) {
        double e_sin = eccentricity * sin(E);
        double f = E - e_sin - mean_anomaly;
        if(fabs(f) < CUTOFF) break;
        double e_cos = eccentricity * cos(E);
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
    return 1.0;
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

    double cos_i = cos(inclination);
    double sin_i = sin(inclination);
    double cos_o = cos(longitude_of_ascending_node);
    double sin_o = sin(longitude_of_ascending_node);
    double cos_a = cos(argument_of_perihelion);
    double sin_a = sin(argument_of_perihelion);
    
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
    double sin_e = sin(eccentric_anomaly);
    double cos_e = cos(eccentric_anomaly);
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
    double h_sq = magnitude_squared(angular_momentum).x + magnitude_squared(angular_momentum).y + magnitude_squared(angular_momentum).z;
    double inclination = acos(angular_momentum.z / sqrt(h_sq));
    double longitude_of_ascending_node = atan2(angular_momentum.x, -angular_momentum.y);
    double v_sq = magnitude_squared(current_v).x + magnitude_squared(current_v).y + magnitude_squared(current_v).z;
    double r = magnitude(current_p);
    double s = h_sq / G;
    double eccentricity = sqrt(1 + s * ((v_sq / G) - (2.00 / r)));
    double perihelion_distance = s / (1.00 + eccentricity);
    double cos_e = (v_sq*r - G) / (eccentricity*G);
    double E_anomaly = acos(cos_e);
    double M_anomaly = E_anomaly - eccentricity * sin(E_anomaly); 
    double cos_f = (s - r ) / (eccentricity * r);
    double f = acos(cos_f);

    // weird calc for true longitude
    double to = -angular_momentum.x / angular_momentum.y;
    double temp = (1.00 - cos(inclination)) * to;
    double temp2 = to * to;
    double true_longitude = atan2((current_p.y * (1.00 + temp2 * cos(inclination)) - current_p.x * temp), (current_p.x * (temp2 + cos(inclination)) - current_p.y * temp));

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
    for(int i = 0; i < blockDim.x; i++) {
        // so here we convert abs position to democratic heliocentric coordinates
        // 3-vec displacement, let r = x, y, z, this is the direction of the acceleration
        double dist_x = positions[i].x - positions[idx].x;
        double dist_y = positions[i].y - positions[idx].y;
        double dist_z = positions[i].z - positions[idx].z;
        double r = sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
        // magnitude of acceleration = mass_of_main_body * -G / |r|^3
        double weighted_acceleration = changeover(r) * masses[0] * G / pow(r, 3);
        // accumulate total acceleration due to all bodies
        double comp = i == idx ? 1.0 : 0.0;
        acc.x -= comp * weighted_acceleration * dist_x;
        acc.y -= comp * weighted_acceleration * dist_y;
        acc.z -= comp * weighted_acceleration * dist_z;
    }

    // update momenta (velocity here) with total acceleration
    velocities[idx].x += acc.x * dt;
    velocities[idx].y += acc.y * dt;
    velocities[idx].z += acc.z * dt;     
}

__device__
void main_body_kinetic(double3* positions, double3* velocities, double* masses, double main_body_mass, double dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double3 p = make_double3(0.0, 0.0, 0.0);
    // calculate total momentum of all bodies
    for(int i = 0; i < blockDim.x; i++) {
        p.x += velocities[i].x * masses[i];
        p.y += velocities[i].y * masses[i];
        p.z += velocities[i].z * masses[i];
    }
    double scaling_factor = dt/(2.00 * main_body_mass);
    positions[idx].x += p.x * scaling_factor;
    positions[idx].y += p.y * scaling_factor;
    positions[idx].z += p.z * scaling_factor;
}

__global__ 
void mercurius_keplerian_solver(
    double* vec_argument_of_perihelion,
    double* vec_mean_anomaly,
    double* vec_eccentricity,
    double* vec_semi_major_axis,
    double* vec_inclination,
    double* vec_longitude_of_ascending_node,
    double* masses,
    double dt,
    int NUM_BODIES
) {
    // declare buffers for positions in SRAM
    extern __shared__ double3 shared_memory[];
    double3* positions = shared_memory;
    double3* velocities = (double3*)&positions[NUM_BODIES];

    // initialize the buffer for each thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // populate positions and velocities
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

    for(int i = 0; i < NUM_TIMESTEPS_KEPLER; i++) {
        body_interaction_kick(positions, velocities, masses, dt/2.00);
        main_body_kinetic(positions, velocities, masses, masses[0], dt/2.00);
       
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
        
        main_body_kinetic(positions, velocities, masses, masses[0], dt/2.00);
        body_interaction_kick(positions, velocities, masses, dt/2.00);
    }
}

__device__
void burlisch_stoer() {
    
}

__global__
void mercurius() {

} 