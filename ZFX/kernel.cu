__global__ void zfx_particles_wrangle(
    float **arrays, size_t narrays, float *params) {
    globals;
    zfx_wrangle_func(globals, params);
}
