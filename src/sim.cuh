// core numerical integration kernel
#ifndef __SIM_CUH__
#define __SIM_CUH__
#include "constants.cuh"

/*
This file contains the core numerical integration kernel and associated helper functions for my implementation of the Mercury N-body Integrator.
Note: A key goal here is to minimise branching. However, it's not entirely escapable.
*/

__device__ double3 cross(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ double stable_sqrt(double x) {
    double gtz = (double)(x >= 0.00);
    // eval to zero if x less than zero
    return sqrt(x*gtz);
}

__device__ double magnitude(const double3& a) {
    return stable_sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ double magnitude_squared(const double3& a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
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



// solves kepler's equation for the eccentric anomaly E
__device__ double danby_burkardt(double mean_anomaly, double eccentricity) {
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


__device__ double fetch_r_crit(double3* positions, double3* velocities, double* masses, int idx1, int idx2, double dt) {
    // anywhere between 3-10
    double n1 = 6.50;
    // anywhere between 0.3-2.0
    double n2 = 1.15;
    double r1 = magnitude(positions[idx1]);
    double r2 = magnitude(positions[idx2]);
    double v1 = magnitude(velocities[idx1]);
    double v2 = magnitude(velocities[idx2]);
    double mutual_hill_radius = cbrt(masses[idx1] + masses[idx2] / 3.00) * (r1 + r2) / 2.00;
    double vmax = max(v1, v2);
    return max(n1*mutual_hill_radius, n2*vmax*dt);
} 

__device__ double changeover(double3* positions, double3* velocities, double* masses, double r_ij, int idx1, int idx2, double dt) {
    double r_crit = fetch_r_crit(positions, velocities, masses, idx1, idx2, dt);
    double y = (r_ij - 0.1*r_crit) / (0.9*r_crit);
    double K = y*y / (2*y*y - 2*y + 1);
    // trying to avoid branching
    double gtz = (double)(y > 0);
    double gto = (double)(y > 1);
    double valid = (double)(y <= 1 && y >= 0);
    return K * gtz * valid + gto;
}

__device__ void cartesian_from_elements(
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
   
    double romes = stable_sqrt(1 - eccentricity*eccentricity);
    double eccentric_anomaly = danby_burkardt(mean_anomaly, eccentricity);
    double sin_e, cos_e;
    sincos(eccentric_anomaly, &sin_e, &cos_e);
    z1 = semi_major_axis * (cos_e - eccentricity);
    z2 = semi_major_axis * romes * sin_e;
    eccentric_anomaly = stable_sqrt(1.00/semi_major_axis) / (1.0 - eccentricity*cos_e);
    z3 = -sin_e * eccentric_anomaly;
    z4 = romes * cos_e * eccentric_anomaly;
    
    current_positions[idx] = make_double3(d11 * z1 + d21 * z2, d12 * z1 + d22 * z2, d13 * z1 + d23 * z2);
    current_velocities[idx] = make_double3(d11 * z3 + d21 * z4, d12 * z3 + d22 * z4, d13 * z3 + d23 * z4);
}

__device__ void elements_from_cartesian(
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
    double h_sq = magnitude_squared(angular_momentum) + epsilon;
    double inclination = stable_acos(angular_momentum.z / stable_sqrt(h_sq));
    // TODO: find way to do this without branching
    double longitude_of_ascending_node = atan2(angular_momentum.x, -angular_momentum.y);
    double v_sq = magnitude_squared(current_v); 
    double r = magnitude(current_p);
    double s = h_sq;
    double eccentricity = stable_sqrt(1 + s * (v_sq - (2.00 / r)));
    double perihelion_distance = s / (1.00 + eccentricity);
    
    double cos_E_anomaly_denom = eccentricity;
    double cos_E_anomaly = (v_sq*r - 1) / cos_E_anomaly_denom;
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

__device__ void body_interaction_kick(double3* positions, double3* velocities, double* masses, double dt) {
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
        double r = stable_sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
        // // magnitude of acceleration = mass_of_other_body * G / |r|^3
        double weighted_acceleration = changeover(positions, velocities, masses, r, i, idx, dt) * masses[i+1] / pow(r + epsilon, 3);
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

__device__ void main_body_kinetic(double3* positions, double3* velocities, double* masses, double dt) {
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


__device__ void update_velocities(double3* positions, double3* velocities, double* masses, double dt) {
    /*
    updates the velocities of all bodies based on the mass distribution of all bodies (incl the sun)
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // the body interaction kick updates velocities based on the masses of all minor bodies
    body_interaction_kick(positions, velocities, masses, dt);
    // now for the main body:
    // update acceleration due to main body
    // a = - G * M / r^3 * r, which in this case simplifies to 1/(r^3) * r_vec
    double3 z_1 = positions[idx];
    double r = magnitude(z_1);
    double3 r_vec = make_double3(z_1.x / pow(r, 3), z_1.y / pow(r, 3), z_1.z / pow(r, 3));
    // negative bc directed inwards
    velocities[idx].x -= r_vec.x * dt;
    velocities[idx].y -= r_vec.y * dt;
    velocities[idx].z -= r_vec.z * dt;
}

struct PosVel {
    double3 pos;
    double3 vel;
};

__device__ PosVel modified_midpoint(double3* positions, double3* velocities, double* masses, double dt, int N) {
    /*
    returns an updated position based on the modified midpoint method.
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // initial velocity, we need this for later
    double3 v = velocities[idx];
    double3 x = positions[idx];
    double subdelta = dt / N;
    
    // euler step
    // this sets up z_1 and z_0
    // so we can use z_0 to calc z_2 and use z_1 for z_3
    double3 z_1 = positions[idx];
    z_1.x += v.x * subdelta;
    z_1.y += v.y * subdelta;
    z_1.z += v.z * subdelta;
    double double_step = 2 * subdelta;
    double3 z_0 = positions[idx];
    positions[idx] = z_1;
    update_velocities(positions, velocities, masses, subdelta);

    // so now, we need to calculate z_(m+1) using z_(m-1)
    // we can store z_(m-1) in z_0 and z_(m+1) in z_1
    // we already have z_0 and z_1 so we can use them for computing z_2/3
    
    for(int i = 1; i < N; i++) {
        double3 temp;
        // here we are computing the new (m+1) position
        temp.x = z_0.x + velocities[idx].x * double_step;
        temp.y = z_0.y + velocities[idx].y * double_step;
        temp.z = z_0.z + velocities[idx].z * double_step;
        // so we set this position to be the new z_0
        z_0 = z_1;
        z_1 = temp;
        positions[idx] = z_1;
        // update velocities
        update_velocities(positions, velocities, masses, subdelta);
    }

    // final little bit
    double3 out;
    out.x = 0.5 * (z_1.x + z_0.x + velocities[idx].x * subdelta);
    out.y = 0.5 * (z_1.y + z_0.y + velocities[idx].y * subdelta);
    out.z = 0.5 * (z_1.z + z_0.z + velocities[idx].z * subdelta);
    PosVel res;
    // store final positions and velocities
    res.pos = out;
    res.vel = velocities[idx];
    // restore velocities and positions
    velocities[idx] = v;
    positions[idx] = x;
    return res;
}

__device__ bool is_converged(PosVel a_1, PosVel a_0) {
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

__device__ void richardson_extrapolation(double3* positions, double3* velocities, double* masses, double dt) {
    const int MAX_ROWS = 4;
    int N = 1;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    PosVel buffer[MAX_ROWS][MAX_ROWS];
    // for our first approx, we use the OG dt
    buffer[0][0] = modified_midpoint(positions, velocities, masses, dt, N);
    for(int i = 1;  i < MAX_ROWS; i++) {
        N = pow(2, i);
        buffer[i][0] = modified_midpoint(positions, velocities, masses, dt, N);
        for(int j = 1; j <= i; j++) {
            double pow2 = pow(2, j);
            
            // for positions
            buffer[i][j].pos.x = pow2 * buffer[i][j-1].pos.x - buffer[i-1][j-1].pos.x;
            buffer[i][j].pos.y = pow2 * buffer[i][j-1].pos.y - buffer[i-1][j-1].pos.y;
            buffer[i][j].pos.z = pow2 * buffer[i][j-1].pos.z - buffer[i-1][j-1].pos.z;

            // for velocities
            buffer[i][j].vel.x = pow2 * buffer[i][j-1].vel.x - buffer[i-1][j-1].vel.x;
            buffer[i][j].vel.y = pow2 * buffer[i][j-1].vel.y - buffer[i-1][j-1].vel.y;
            buffer[i][j].vel.z = pow2 * buffer[i][j-1].vel.z - buffer[i-1][j-1].vel.z;

            pow2 -= 1;

            buffer[i][j].pos.x /= pow2;
            buffer[i][j].pos.y /= pow2;
            buffer[i][j].pos.z /= pow2;

            buffer[i][j].vel.x /= pow2;
            buffer[i][j].vel.y /= pow2;
            buffer[i][j].vel.z /= pow2;
        }

        if(is_converged(buffer[i][i], buffer[i-1][i-1])) {            
            positions[idx] = buffer[i][i].pos;
            velocities[idx] = buffer[i][i].vel;
            return;
        }
    }
}

// this ensures that the sun is in a reference frame in which it is stationary and at the origin
__device__ void convert_to_democratic_heliocentric_coordinates(double3* positions, double3* velocities, double* masses) {
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

__device__ bool close_encounter_p(double3* positions, double3* velocities, double* masses, double dt) {
    // get indices of bodies i am undergoing close encounters with
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < blockDim.x; i++) {
        if(i == idx) continue;
        double3 r_ij;
        r_ij.x = positions[i].x - positions[idx].x;
        r_ij.y = positions[i].y - positions[idx].y;
        r_ij.z = positions[i].z - positions[idx].z;
        double r_ij_mag = magnitude(r_ij);
        double r_crit = fetch_r_crit(positions, velocities, masses, idx, i, dt);
        if (r_ij_mag < r_crit) {
            return true;
        }
    }
    return false;
}


__global__ void mercurius_solver(
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

        // if i am undergoing a close encounter with anyone
        // use bulirsch-stoer aka richardson extrapolation w/ mod midpoint 
        if(close_encounter_p(positions, velocities, masses, dt)) {}
        else {
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
        }
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