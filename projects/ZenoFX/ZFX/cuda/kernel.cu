__device__ void zfx_wrangle_func(float *globals, float const *params);

extern "C" __global__ void zfx_array_wrangle
    ( float *array
    , size_t nsize
    , float const *params
    ) {
    for
        ( int i = blockDim.x * blockIdx.x + threadIdx.x
        ; i < nsize
        ; i += blockDim.x * gridDim.x
        ) {
        float globals[1];
        globals[0] = array[i];
        zfx_wrangle_func(globals, params);
        array[i] = globals[0];
    }
}
