#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#define MAX_ITERATIONS_ROOT_FINDING 8
#define CUTOFF 1e-13
#define BULIRSCH_STOER_TOLERANCE 1e-12
#define SMOOTHING_CONSTANT 3e-8
#define SMOOTHING_CONSTANT_SQUARED 8.999999999999998e-16
#define TWOPI 6.283185307179586476925286766559005768394338798750211641949
#define MAX_ROWS_RICHARDSON 3
#define BATCH_SIZE 100
#define SWEEPS_PER_GPU 4
// we have a 8 GPU node for now
#define NUM_GPUS 8
#endif