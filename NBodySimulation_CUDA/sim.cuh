#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void compute_bodies(float* bodies, int n, float min);
__global__ void update_bodies(float* bodies, int n, float dt);

void call_compute(float* bodies, int n, dim3 dim_grid, dim3 dim_block, float min);
void call_update(float* bodies, int n, dim3 dim_grid, dim3 dim_block, float dt);