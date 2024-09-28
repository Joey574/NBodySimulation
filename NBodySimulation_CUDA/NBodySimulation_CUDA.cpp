#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <immintrin.h>
#include <GLFW/glfw3.h>

#include "sim.cuh"
#include "../Application/Application.h"
#include "../Simulation_Initializations/Simulation_Initializations.h"

const float DT = 0.0001f;
const float MIN_DISTANCE = 0.0001f;

const int width = 720;
const int height = 720;

const int number_bodies = 10000;

struct simulation {

    simulation(size_t size, int seed, initialize::init_types type) : n(size) {
        bodies = (float*)calloc(n * 7, sizeof(float));
        cudaMalloc(&d_data, n * 7 * sizeof(float));
        
        initialize::initialize_galaxy(type, bodies, n, seed);

        // copy data into d_data
        cudaMemcpy(d_data, bodies, n * 7 * sizeof(float), cudaMemcpyHostToDevice);
    }

    void update() {

        dim3 dim_block(8, 1, 1);
        dim3 dim_grid(ceil(n / 8), 1, 1);

        call_compute(d_data, n, dim_grid, dim_block, MIN_DISTANCE);
        call_update(d_data, n, dim_grid, dim_block, DT);

        // copy data back to display
        cudaMemcpy(bodies, d_data, n * 7 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    ~simulation() {
        free(bodies);
        cudaFree(d_data);
    }

    float* d_data;
    float* bodies;
    size_t n;
};

int main()
{
    simulation sim(number_bodies, 0, initialize::init_types::spiral);

    GLFWwindow* window = create_window(width, height);

    double sum = 0.0;
    size_t count = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        // update simulation
        auto start = std::chrono::high_resolution_clock::now();
        sim.update();
        auto time = std::chrono::high_resolution_clock::now() - start;
        sum += time.count() / 1000000.00; count++;

        std::cout << "\u001b[HAverage: " + std::to_string(sum / count).append("ms  \nLast: ").append(std::to_string(time.count() / 1000000.00)).append("ms  \nCount: ").append(std::to_string(count));

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        // render simulation data
        render_data(sim.bodies, sim.n);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}