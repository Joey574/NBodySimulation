#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <GLFW/glfw3.h>

#include "cuda_update.cuh"
//#include "cuda_update.cu"

const float PI = 3.1415926f;
const float TAU = 6.2831853f;

const float DT = 0.00001f;
const float MIN_DISTANCE = 0.000001f;

const int width = 720;
const int height = 720;

const int number_bodies = 2000;

struct simulation {

    simulation() : bodies(nullptr), n(0) {}

    simulation(size_t size, int seed) : n(size) {

        srand(seed);
        std::default_random_engine gen(seed);

        std::uniform_real_distribution<float>vel(0, 0.5f);
        std::normal_distribution<float>pos(0.0f, 0.2f);
        std::normal_distribution<float>mass(0.005f, 0.05f);

        bodies = (float*)calloc(n * 7, sizeof(float));

        for (size_t i = 0; i < n; i++) {

            float x = pos(gen);
            float y = pos(gen);

            float temp = vel(gen) * TAU;
            float d_x = std::cos(temp);
            float d_y = std::sin(temp) * pos(gen);

            // set pos
            bodies[i * 7] = x; bodies[i * 7 + 1] = y;

            // set vel
            bodies[i * 7 + 2] = d_x; bodies[i * 7 + 3] = d_y;

            // set mass
            bodies[i * 7 + 6] = 0.001f + std::abs(mass(gen) / 2.0f);
        }
    }

    void update() {

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float p1[2] = { bodies[i * 7], bodies[i * 7 + 1] };
            float m1 = bodies[i * 7 + 6];

            for (size_t j = i + 1; j < n; j++) {
                float p2[2] = { bodies[j * 7], bodies[j * 7 + 1] };
                float m2 = bodies[j * 7 + 6];

                float r[2] = { p2[0] - p1[0], p2[1] - p1[1] };
                float mag_sq = (r[0] * r[0]) + (r[1] * r[1]);
                float temp = std::max(mag_sq, MIN_DISTANCE) * std::sqrt(mag_sq);

                float d_acc[2] = { r[0] / temp, r[1] / temp };

                bodies[i * 7 + 4] += m2 * d_acc[0]; bodies[i * 7 + 5] += m2 * d_acc[1];
                bodies[j * 7 + 4] -= m1 * d_acc[0]; bodies[j * 7 + 5] -= m1 * d_acc[1];
            }
        }

        //float* d_data;

        //dim3 dim_block(8, 1, 1);
        //dim3 dim_grid(ceil(n / 8), 1, 1);

        //cudaMalloc(&d_data, n * 7 * sizeof(float));
        //cudaMemcpy(d_data, bodies, n * 7 * sizeof(float), cudaMemcpyHostToDevice);

        //call_compute(d_data, n, dim_grid, dim_block, MIN_DISTANCE);
        //call_update(d_data, n, dim_grid, dim_block, DT);

        //// copy data back to display
        //cudaMemcpy(bodies, d_data, n * 7 * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaFree(d_data);

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

    float* bodies;
    size_t n;
};

void render_data(float* bodies, size_t n);
void draw_circle(GLfloat x, GLfloat y, GLfloat r);

int main()
{
    simulation sim(number_bodies, 0);

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "N-body simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    double sum = 0.0;
    size_t count = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        // update simulation
        auto start = std::chrono::high_resolution_clock::now();
        sim.update();
        auto time = std::chrono::high_resolution_clock::now() - start;
        sum += time.count() / 1000000.00;
        count++;
        std::cout << "\u001b[HAverage: " + std::to_string(sum / count).append("ms  \nLast: ").append(std::to_string(time.count() / 1000000.00)).append("ms  \nCount: ").append(std::to_string(count));

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

void render_data(float* bodies, size_t n) {

    // set color to white
    glColor3f(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        draw_circle(bodies[i * 7], bodies[i * 7 + 1], std::log(bodies[i * 7 + 6] + 1.0f) / 20.0f);
    }
}

void draw_circle(GLfloat x, GLfloat y, GLfloat r) {
    int triangles = 20;

    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);

    for (int i = 0; i <= triangles; i++) {
        glVertex2f(
            x + (r * std::cos(i * 2.0f * PI / triangles)),
            y + (r * std::sin(i * 2.0f * PI / triangles))
            );
    }

    glEnd();
}