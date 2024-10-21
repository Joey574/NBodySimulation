#pragma once
#include <iostream>
#include <immintrin.h>
#include <GLFW/glfw3.h>
#include <atlimage.h>

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

// rendering
void render_data(float* bodies, size_t n, GLfloat aspect_ratio);
void draw_circle(GLfloat x, GLfloat y, GLfloat r, GLfloat aspect_ratio);
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r);
void draw_ellipse(GLfloat x_c, GLfloat y_c, GLfloat x_r, GLfloat y_r, GLfloat rotation);

void take_screenshot(std::string filepath, size_t width, size_t height);

// window creation
GLFWwindow* create_window(size_t width, size_t height);