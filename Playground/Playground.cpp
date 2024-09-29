#include <iostream>

#include <GLFW/glfw3.h>
#include "../Application/Application.h"

const int width = 1000;
const int height = 1000;

const float PI = 3.1415926f;

float zoom = 1.0f;

void local_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    const float zoomSpeed = 0.05f;

    if (yoffset > 0) {
        zoom += zoomSpeed * zoom;
    }
    else if (yoffset < 0) {
        zoom -= zoomSpeed * zoom;
    }

    zoom = std::max(zoom, 0.0001f);
}

int main()
{
    GLFWwindow* window = create_window(width, height);

    glfwSetScrollCallback(window, local_scroll_callback);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        float ry = 0.025f;
        float rx = 0.025;
        float rotation = 0.0f;
        float inc = 0.02f;

        int ellipses = 50;;
        int segments = 100;

        int b_idx = 0;

        for (int e = 0; e < ellipses; e++) {

            draw_ellipse(0.0f, 0.0f, rx * zoom, ry * zoom, rotation);

            rotation += 0.25f;
            ry += inc;
            rx += inc * 1.2f;

            inc += 0.001f;
        }

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}