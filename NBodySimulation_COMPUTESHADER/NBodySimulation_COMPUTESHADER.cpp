#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <immintrin.h>
#include <omp.h>

#define NOMINMAX
#include <Windows.h>

#include <GLFW/glfw3.h>
#include "../Application/Application.h"
#include "../Simulation_Initializations/Simulation_Initializations.h"


const float TAU = 6.2831853f;
const float DT = 0.0001f;
const float MIN_DISTANCE = 0.000001f;

const int width = 720;
const int height = 720;

const int number_bodies = 10000;

struct simulation {

    simulation(size_t size, int seed, initialize::init_types type) : n(size) {
        bodies = (float*)calloc(n * 7, sizeof(float));

        initialize::initialize_galaxy(type, bodies, n, seed);

        GLuint compute_shader;
    }

    void update() {

    }

    ~simulation() {
        free(bodies);
    }

    alignas(64) float* bodies;
    size_t n;

};


int main()
{
    SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);

    simulation sim(number_bodies, 0, initialize::init_types::spiral);
    GLFWwindow* window = create_window(width, height);

    double sim_sum = 0.0;
    double ren_sum = 0.0;
    double tot_sum = 0.0;

    size_t count = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window)) {
        count++;

        auto total_start = std::chrono::high_resolution_clock::now();
        std::string out = "";

        // update simulation
        auto sim_start = std::chrono::high_resolution_clock::now();
        sim.update();
        auto sim_time = std::chrono::high_resolution_clock::now() - sim_start;
        sim_sum += sim_time.count() / 1000000.00;

        out.append("\u001b[HFrame: ").append(std::to_string(count)).append("\n\nSim Update:\nAverage: ").append(std::to_string(sim_sum / count)).append(" (ms)   \nLast: ").
            append(std::to_string(sim_time.count() / 1000000.00)).append(" (ms)   \n");

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        // render simulation data
        auto ren_start = std::chrono::high_resolution_clock::now();
        render_data(sim.bodies, sim.n);
        auto ren_time = std::chrono::high_resolution_clock::now() - ren_start;
        ren_sum += ren_time.count() / 1000000.00;

        out.append("\nRendering:\nAverage: ").append(std::to_string(ren_sum / count)).append(" (ms)   \n");

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        auto total_time = std::chrono::high_resolution_clock::now() - total_start;
        tot_sum += total_time.count() / 1000000.00;
        out.append("\nTotal:\nAverage: ").append(std::to_string(tot_sum / count)).append(" (ms)   \n").append("Average FPS: ").append(std::to_string(1000.00 / (tot_sum / count))).append("   \n");

        std::cout << out;
    }

    glfwTerminate();
    return 0;
}