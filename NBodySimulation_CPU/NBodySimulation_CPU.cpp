#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <immintrin.h>
#include <omp.h>

#include <GLFW/glfw3.h>
#include "../Application/Application.h"
#include "../Simulation_Initializations/Simulation_Initializations.h"


const float TAU = 6.2831853f;
const float DT = 0.00001f;
const float MIN_DISTANCE = 0.000001f;

const int width = 720;
const int height = 720;

const int number_bodies = 2000;

struct simulation {

    simulation(size_t size, int seed, initialize::init_types type) : n(size) {
        bodies = (float*)calloc(n * 7, sizeof(float));
        omp_init_lock(&omp_lock);

        initialize::initialize_galaxy(type, bodies, n, seed);
    }

    void update() {

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float p1[2] = { bodies[i * 7], bodies[i * 7 + 1] };
         
            for (size_t j = 0; j < n; j++) {
                if (i != j) {
                    float p2[2] = { bodies[j * 7], bodies[j * 7 + 1] };
                    float m2 = bodies[j * 7 + 6];

                    // -> calculate x, y distance between bodies
                    float r[2] = { p2[0] - p1[0], p2[1] - p1[1] };

                    // -> calculate dist^2
                    float mag_sq = (r[0] * r[0]) + (r[1] * r[1]);

                    // -> calculate dist
                    float mag = std::sqrt(mag_sq);

                    float temp[2] = { r[0] / (std::max(mag_sq, MIN_DISTANCE) * mag), r[1] / (std::max(mag_sq, MIN_DISTANCE) * mag) };

                    bodies[i * 7 + 4] += m2 * temp[0]; bodies[i * 7 + 5] += m2 * temp[1];
                }
            }
        }

        // update bodies
        for (size_t i = 0; i < n; i++) {
            bodies[i * 7] += bodies[i * 7 + 2] * DT; bodies[i * 7 + 1] += bodies[i * 7 + 3] * DT;
            bodies[i * 7 + 2] += bodies[i * 7 + 4] * DT; bodies[i * 7 + 3] += bodies[i * 7 + 5] * DT;
            bodies[i * 7 + 4] = 0.0f; bodies[i * 7 + 5] = 0.0f;
        }
    }

    ~simulation() {
        free(bodies);
    }

    alignas(64) float* bodies;
    size_t n;

    omp_lock_t omp_lock;
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