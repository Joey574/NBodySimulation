#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <immintrin.h>
#include <GLFW/glfw3.h>

#include "sim.cuh"

const float PI = 3.1415926f;
const float TAU = 6.2831853f;
const float DT = 0.00001f;
const float MIN_DISTANCE = 0.001f;

const int width = 720;
const int height = 720;

const int number_bodies = 10000;

float zoom = 1.0f;
float x_offset = 0.0f;
float y_offset = 0.0f;

struct simulation {

    simulation() : bodies(nullptr), d_data(nullptr), n(0) {}

    simulation(size_t size, int seed) : n(size) {
        std::default_random_engine gen(seed);

        std::uniform_real_distribution<float>vel(0, 0.5f);
        std::normal_distribution<float>pos(0.0f, 0.5f);
        std::normal_distribution<float>mass(0.001f, 0.025f);

        bodies = (float*)calloc(n * 7, sizeof(float));

        for (size_t i = 0; i < n; i++) {

            float x = pos(gen);
            float y = pos(gen);

            float temp = vel(gen) * TAU;
            float d_x = std::cos(temp);
            float d_y = std::sin(temp) * (pos(gen) * 2.0f);

            // set pos
            bodies[i * 7] = x; bodies[i * 7 + 1] = y;

            // set vel
            bodies[i * 7 + 2] = d_x; bodies[i * 7 + 3] = d_y;

            // set mass
            bodies[i * 7 + 6] = 0.0001f + std::abs(mass(gen) / 2.0f);
        }
        
        bodies[0] = 0.0f; bodies[1] = 0.0f;
        bodies[6] = 0.75f;

        cudaMalloc(&d_data, n * 7 * sizeof(float));
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

void render_data(float* bodies, size_t n);
void draw_circle(GLfloat x, GLfloat y, GLfloat r);

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    const float zoomSpeed = 0.1f;

    if (yoffset > 0) {
        zoom += zoomSpeed;
    }
    else if (yoffset < 0) {
        zoom -= zoomSpeed;
    }

    zoom = std::max(zoom, 0.001f);
}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    const float offset_speed = 0.05f;

    if (key == GLFW_KEY_W) {
        y_offset -= offset_speed;
    }
    if (key == GLFW_KEY_A) {
        x_offset += offset_speed;
    }
    if (key == GLFW_KEY_S) {
        y_offset += offset_speed;
    }
    if (key == GLFW_KEY_D) {
        x_offset -= offset_speed;
    }        
}

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
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

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

        float final_x = (bodies[i * 7] * zoom) + x_offset;
        float final_y = (bodies[i * 7 + 1] * zoom) + y_offset;

        float final_r = (std::log(bodies[i * 7 + 6] + 1.0f) / 20.0f) * zoom;
        
        draw_circle(final_x, final_y, final_r);

        //if (final_x >= -1.0f && final_x <= 1.0f && final_y >= -1.0f && final_y <= 1.0f) {
        //
        //}
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