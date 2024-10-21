#include <iostream>
#include <chrono>
#include <random>
#include <string>

#include <GLFW/glfw3.h>
#include "../Application/Application.h"
#include "../Simulation_Initializations/Simulation_Initializations.h"

const int width = 720;
const int height = 720;

const int number_bodies = 10000;

struct simulation {
    simulation(initialize::init_types type, size_t size, int seed) : n(size) {

        bodies = (float*)calloc(n * 7, sizeof(float));

        initialize::initialize_galaxy(type, bodies, n, seed);
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
    simulation sim(initialize::init_types::cluster, number_bodies, 0);

    GLFWwindow* window = create_window(width, height);

    double sum = 0.0;
    size_t count = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        // update simulation
        //auto start = std::chrono::high_resolution_clock::now();
        //sim.update();
        //auto time = std::chrono::high_resolution_clock::now() - start;
        //sum += time.count() / 1000000.00;
        //count++;
        //std::cout << "\u001b[HAverage: " + std::to_string(sum / count).append("ms  \nLast: ").append(std::to_string(time.count() / 1000000.00)).append("ms  \nCount: ").append(std::to_string(count));

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        render_data(sim.bodies, sim.n);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}