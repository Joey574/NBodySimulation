#include <iostream>
#include <random>

struct initialize {
public:
	enum class init_types {
		cluster, spiral, video
	};

	static void initialize_galaxy(init_types type, float* bodies, size_t n, int seed);

private:

	static void cluster_init(float* bodies, size_t n, int seed);
	static void spiral_init(float* bodies, size_t n, int seed);
	static void video_init(float* bodies, size_t n, int seed);

	static void sort_bodies(float* bodies, size_t n);
};