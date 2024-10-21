#include "Application.h"

static float zoom = 0.04f;
static float x_offset = 0.0f;
static float y_offset = 0.0f;

static const float PI = 3.1415926f;

// user input
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    const float zoomSpeed = 0.05f;

    if (yoffset > 0) {
        zoom += zoomSpeed * zoom;
    } else if (yoffset < 0) {
        zoom -= zoomSpeed * zoom;
    }

    zoom = zoom > 0.0001f ? zoom : 0.0001f;
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

/// <summary>
/// Expects full data set to be passed, ie each sample contains pos, vel, acc, and mass
/// </summary>
/// <param name="bodies"></param>
/// <param name="n"></param>
void render_data(float* bodies, size_t n, GLfloat aspect_ratio) {

    float* ren_data = (float*)malloc(n * 3 * sizeof(float));

    //#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        ren_data[i * 3] = (bodies[i * 7] * zoom) + x_offset;
        ren_data[i * 3 + 1] = (bodies[i * 7 + 1] * zoom) + y_offset;

        ren_data[i * 3 + 2] = (std::log(bodies[i * 7 + 6] + 1.0f) / 10.0f) * zoom;
    }

    for (int i = 0; i < n; i++) {

        //draw_circle(ren_data[i * 3], ren_data[i * 3 + 1], ren_data[i * 3 + 2]);

        if (
            ren_data[i * 3] + ren_data[i * 3 + 2] >= -1.0f &&
            ren_data[i * 3] - ren_data[i * 3 + 2] <= 1.0f &&
            ren_data[i * 3 + 1] + ren_data[i * 3 + 2] >= -1.0f &&
            ren_data[i * 3 + 1] - ren_data[i * 3 + 2] <= 1.0f) {
        
            draw_circle(ren_data[i * 3], ren_data[i * 3 + 1], ren_data[i * 3 + 2], aspect_ratio);
        }
    }

    free(ren_data);
}
void draw_circle(GLfloat x, GLfloat y, GLfloat r, GLfloat aspect_ratio) {
    const __m256 _t = _mm256_set1_ps(PI / 11.5f);

    const __m256 _r = _mm256_set1_ps(r);
    const __m256 _x = _mm256_set1_ps(x);
    const __m256 _y = _mm256_set1_ps(y);
    const __m256 _aspect_ratio = _mm256_set1_ps(aspect_ratio);

    // const iteration values for triangle fan
    const __m256 _a = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    const __m256 _b = { 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f };
    const __m256 _c = { 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f };

    // x values for vertexes
    const __m256 _x1 = _mm256_fmadd_ps(_r, _mm256_mul_ps(_mm256_cos_ps(_mm256_mul_ps(_t, _a)), _aspect_ratio), _x);
    const __m256 _x2 = _mm256_fmadd_ps(_r, _mm256_mul_ps(_mm256_cos_ps(_mm256_mul_ps(_t, _b)), _aspect_ratio), _x);
    const __m256 _x3 = _mm256_fmadd_ps(_r, _mm256_mul_ps(_mm256_cos_ps(_mm256_mul_ps(_t, _c)), _aspect_ratio), _x);

    // y values for vertexes
    const __m256 _y1 = _mm256_fmadd_ps(_r, _mm256_sin_ps(_mm256_mul_ps(_t, _a)), _y);
    const __m256 _y2 = _mm256_fmadd_ps(_r, _mm256_sin_ps(_mm256_mul_ps(_t, _b)), _y);
    const __m256 _y3 = _mm256_fmadd_ps(_r, _mm256_sin_ps(_mm256_mul_ps(_t, _c)), _y);

    float comp_x[24];
    float comp_y[24];

    _mm256_store_ps(&comp_x[0], _x1);
    _mm256_store_ps(&comp_x[8], _x2);
    _mm256_store_ps(&comp_x[16], _x3);

    _mm256_store_ps(&comp_y[0], _y1);
    _mm256_store_ps(&comp_y[8], _y2);
    _mm256_store_ps(&comp_y[16], _y3);

    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);

    for (int i = 0; i < 24; i++) {
        glVertex2f(comp_x[i], comp_y[i]);
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

void take_screenshot(std::string filepath, size_t width, size_t height) {
    GLubyte* pixels = new GLubyte[3 * width * height];
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, nullptr, 0);
    std::wstring wideStr(wideStrLength, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, &wideStr[0], wideStrLength);

    CImage image;
    image.Create(width, height, 24);

    /*size_t idx = 0;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            image.SetPixel(x, y, RGB(pixels[idx * 3] * 255.0f, pixels[idx * 3 + 1] * 255.0f, pixels[idx * 3 + 2] * 255.0f));
            idx++;
        }
    }*/

    for (size_t y = 0; y < height; y++) {
        std::memcpy(image.GetPixelAddress(0, y), &pixels[y * width * 3], width * 3 * sizeof(GLubyte));
    }

    image.Save(wideStr.c_str(), Gdiplus::ImageFormatBMP);
    image.Destroy();

    delete[] pixels;
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