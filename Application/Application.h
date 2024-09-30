#pragma once
#include <iostream>
#include <immintrin.h>
#include <GLFW/glfw3.h>

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

// rendering
void render_data(float* bodies, size_t n);
void render_data_cuda(float* pos, float* mass, size_t n);
void draw_circle(GLfloat x, GLfloat y, GLfloat r);
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r);
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r, GLfloat rotation);

// window creation
GLFWwindow* create_window(size_t width, size_t height);