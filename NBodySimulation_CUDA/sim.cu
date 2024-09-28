#include "sim.cuh"

__global__ void compute_bodies(float* bodies, int n, float min) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		float p1[2] = { bodies[i * 7], bodies[i * 7 + 1] };
		float m1 = bodies[i * 7 + 6];

		for (int j = 0; j < n; j++) {
			if (j != i) {
				float p2[2] = { bodies[j * 7], bodies[j * 7 + 1] };
				float m2 = bodies[j * 7 + 6];

				// -> calculate x, y distance between bodies
				float r[2] = { p2[0] - p1[0], p2[1] - p1[1] };

				// -> calculate dist^2
				float mag_sq = (r[0] * r[0]) + (r[1] * r[1]);

				// -> calculate dist
				float mag = sqrtf(mag_sq);

				float temp[2] = { r[0] / (fmaxf(mag_sq, min) * mag), r[1] / (fmaxf(mag_sq, min) * mag) };

				bodies[i * 7 + 4] += m2 * temp[0]; bodies[i * 7 + 5] += m2 * temp[1];
			}
		}
	}
}
__global__ void update_bodies(float* bodies, int n, float dt) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		bodies[i * 7] += bodies[i * 7 + 2] * dt; bodies[i * 7 + 1] += bodies[i * 7 + 3] * dt;
		bodies[i * 7 + 2] += bodies[i * 7 + 4] * dt; bodies[i * 7 + 3] += bodies[i * 7 + 5] * dt;
		bodies[i * 7 + 4] = 0.0f; bodies[i * 7 + 5] = 0.0f;
	}
}

void call_compute(float* bodies, int n, dim3 dim_grid, dim3 dim_block, float min) {
	compute_bodies << < dim_grid, dim_block >> > (bodies, n, min);
}

void call_update(float* bodies, int n, dim3 dim_grid, dim3 dim_block, float dt) {
	update_bodies << < dim_grid, dim_block >> > (bodies, n, dt);
}