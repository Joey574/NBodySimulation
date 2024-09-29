#include "Application.h"

static float zoom = 1.0f;
static float x_offset = 0.0f;
static float y_offset = 0.0f;

static const float PI = 3.1415926f;
const int TRIANGLES = 20;


// user input
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    const float zoomSpeed = 0.05f;

    if (yoffset > 0) {
        zoom += zoomSpeed * zoom;
    } else if (yoffset < 0) {
        zoom -= zoomSpeed * zoom;
    }

    zoom = std::max(zoom, 0.0001f);
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

// rendering
void render_data(float* bodies, size_t n) {

    float* ren_data = (float*)malloc(n * 3 * sizeof(float));

    //#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        ren_data[i * 3] = (bodies[i * 7] * zoom) + x_offset;
        ren_data[i * 3 + 1] = (bodies[i * 7 + 1] * zoom) + y_offset;

        ren_data[i * 3 + 2] = (std::log(bodies[i * 7 + 6] + 1.0f) / 20.0f) * zoom;
    }
    for (int i = 0; i < n; i++) {
        if (
            ren_data[i * 3] + ren_data[i * 3 + 2] >= -1.0f &&
            ren_data[i * 3] - ren_data[i * 3 + 2] <= 1.0f &&
            ren_data[i * 3 + 1] + ren_data[i * 3 + 2] >= -1.0f &&
            ren_data[i * 3 + 1] - ren_data[i * 3 + 2] <= 1.0f) {
        
            draw_circle(ren_data[i * 3], ren_data[i * 3 + 1], ren_data[i * 3 + 2]);
        }
    }

    free(ren_data);
}
void draw_circle(GLfloat x, GLfloat y, GLfloat r) {
    
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);

    float theta = 2.0f * PI / TRIANGLES;
    for (int i = 0; i <= TRIANGLES; i++) {
        glVertex2f(
            x + (r * std::cos(i * theta)),
            y + (r * std::sin(i * theta))
        );
    }

    glEnd();
}
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r, GLfloat rotation) {
    const int segments = 100;

    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; i++) {

        float angle = (2.0f * PI * i / (float)segments);
        float x = x_c + x_r * cos(angle);
        float y = y_c + y_r * sin(angle);

        float rot_x = x * cosf(rotation) - y * sinf(rotation);
        float rot_y = x * sinf(rotation) + y * cosf(rotation);

        glVertex2f(rot_x, rot_y);
        
    }
    glEnd();
}
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r) {
    const int segments = 100;

    float theta = 2 * PI / (float)segments;
    float c = cosf(theta);
    float s = sinf(theta);
    float t;

    float x = 1.0f;
    float y = 0.0f;

    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; i++) {

        glVertex2f(x * x_r + x_c, y * y_r + y_c);

        t = x;
        x = c * x - s * y;
        y = s * t + c * y;

    }
    glEnd();
}

// window creation
GLFWwindow* create_window(size_t width, size_t height) {
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit()) {
        return nullptr;
    }

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "N-body simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return nullptr;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // Set up user input functions
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);


    return window;
}