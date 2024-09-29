#include "Simulation_Initializations.h"

static const float TAU = 6.2831853f;
static const float PI = 3.1415926f;

void initialize::initialize_galaxy(init_types type, float* bodies, size_t n, int seed) {

    switch (type) {
    case init_types::cluster:
        cluster_init(bodies, n, seed);
        break;
    case init_types::spiral:
        spiral_init(bodies, n, seed);
        break;
    case init_types::video:
        video_init(bodies, n, seed);
        break;
    }
}

void initialize::cluster_init(float* bodies, size_t n, int seed) {
    std::default_random_engine gen(seed);

    std::uniform_real_distribution<float>vel(0, 0.5f);
    std::normal_distribution<float>pos(0.0f, 0.5f);
    std::normal_distribution<float>mass(0.005f, 0.05f);

    for (size_t i = 1; i < n; i++) {

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
void initialize::spiral_init(float* bodies, size_t n, int seed) {
    std::default_random_engine gen(seed);

    std::normal_distribution<float>pos(0.0f, 0.01f);
    std::normal_distribution<float>mass(0.0f, 0.005f);

    float ry = 0.05f;
    float angle = 0.0f;
    float inc = 0.01f;

    int ellipses = 20;;
    int segments = 100;

    int per_point = n / (segments * ellipses);
    int b_idx = 0;

    for (int e = 0; e < ellipses; e++) {
        for (int i = 0; i < segments; i++) {

            float theta = (2.0f * PI * i / (float)segments);
            float x = 0.0f + (ry * 1.4f) * cos(theta);
            float y = 0.0f + ry * sin(theta);

            float rot_x = x * cosf(angle) - y * sinf(angle);
            float rot_y = x * sinf(angle) + y * cosf(angle);

            // generate some # bodies in normal dist. centered at rot_x, rot_y
            for (int j = 0; j < per_point; j++) {

                bodies[b_idx * 7] = rot_x + pos(gen);
                bodies[b_idx * 7 + 1] = rot_y + pos(gen);

                //bodies[b_idx * 7 + 6] = 0.01f + std::abs(mass(gen));
                bodies[b_idx * 7 + 6] = 0.05f;

                float dx = sinf(theta);
                float dy = cosf(theta);
                
                bodies[b_idx * 7 + 2] = dx;
                bodies[b_idx * 7 + 3] = -dy;

                b_idx++;
            }
        }

        angle += 0.2f;
        ry += inc;
        
        inc += 0.001f;
    }
    
    // hardcoded centeral body
    bodies[0] = 0.0f;
    bodies[1] = 0.0f;

    bodies[2] = 0.0f;
    bodies[3] = 0.0f;

    bodies[6] = 0.5f;
}
void initialize::video_init(float* bodies, size_t n, int seed) {
    std::default_random_engine gen(seed);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        float a = dist(gen) * TAU;

        float sin = std::sin(a); float cos = std::cos(a);

        float r =
            dist(gen) +
            dist(gen) +
            dist(gen) +
            dist(gen) +
            dist(gen) +
            dist(gen);

        r = std::abs(r / 3.0f - 1.0f);

        float pos[2] = {
            cos * std::sqrt((float)n) * 2.0f * r,
            sin * std::sqrt((float)n) * 2.0f * r
        };

        float vel[2] = { sin, -cos };

        // set pos
        bodies[i * 7] = pos[0];
        bodies[i * 7 + 1] = pos[1];

        // set vel
        bodies[i * 7 + 2] = vel[0];
        bodies[i * 7 + 3] = vel[1];

        // set mass
        bodies[i * 7 + 6] = 1.5f;
    }

    // sort bodies
    for (size_t i = 1; i < n; i++) {
        float val = bodies[i * 7] * bodies[i * 7] + bodies[i * 7 + 1] * bodies[i * 7 + 1];

        float tmp[7] = {
            bodies[i * 7],
            bodies[i * 7 + 1],
            bodies[i * 7 + 2],
            bodies[i * 7 + 3],
            bodies[i * 7 + 4],
            bodies[i * 7 + 5],
            bodies[i * 7 + 6],
        };

        int j = i - 1;
        while (j >= 0 && bodies[j * 7] * bodies[j * 7] + bodies[j * 7 + 1] * bodies[j * 7 + 1] > val) {

            //unsorted[j + 1] = unsorted[j];
            bodies[(j + 1) * 7] = bodies[j * 7];
            bodies[(j + 1) * 7 + 1] = bodies[j * 7 + 1];
            bodies[(j + 1) * 7 + 2] = bodies[j * 7 + 2];
            bodies[(j + 1) * 7 + 3] = bodies[j * 7 + 3];
            bodies[(j + 1) * 7 + 4] = bodies[j * 7 + 4];
            bodies[(j + 1) * 7 + 5] = bodies[j * 7 + 5];
            bodies[(j + 1) * 7 + 6] = bodies[j * 7 + 6];

            j--;
        }

        // unsorted[j + 1] = val;
        bodies[(j + 1) * 7] = tmp[0];
        bodies[(j + 1) * 7 + 1] = tmp[1];
        bodies[(j + 1) * 7 + 2] = tmp[2];
        bodies[(j + 1) * 7 + 3] = tmp[3];
        bodies[(j + 1) * 7 + 4] = tmp[4];
        bodies[(j + 1) * 7 + 5] = tmp[5];
        bodies[(j + 1) * 7 + 6] = tmp[6];
    }

    // scale velocities
    for (size_t i = 0; i < n; i++) {
        float v = std::sqrt((float)i / std::sqrt(bodies[i * 7] * bodies[i * 7] + bodies[i * 7 + 1] * bodies[i * 7 + 1]));
        bodies[i * 7 + 2] *= v;
        bodies[i * 7 + 3] *= v;
    }

    bodies[0] = 0.0f;
    bodies[1] = 0.0f;
    bodies[2] = 0.0f;
    bodies[3] = 0.0f;

    bodies[6] = 100.0f;
}