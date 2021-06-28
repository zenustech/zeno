

PER_CUDA_FUNCTION(memcpyAsync, cudaMemcpyAsync, void *, void *, size_t, void *);
// PER_CUDA_FUNCTION(vmalloc, cudaMallocManaged, void **, size_t);
// PER_CUDA_FUNCTION(malloc, cudaMalloc, void **, size_t);
PER_CUDA_FUNCTION(memsetAsync, cudaMemsetAsync, void *, uint8_t, size_t, void *);
// PER_CUDA_FUNCTION(free, cudaFree, void *);
// PER_CUDA_FUNCTION(mmemAdvise, cudaMemAdvise, void *, size_t, uint32_t, int);

// Stream management
// PER_CUDA_FUNCTION(createStream, cudaStreamCreate, void **);
// PER_CUDA_FUNCTION(destroyStream, cudaStreamDestroy, void *)
// PER_CUDA_FUNCTION(syncStream, cudaStreamSynchronize, void *);
// PER_CUDA_FUNCTION(streamWaitEvent, cudaStreamWaitEvent, void *, void *, uint32_t);

// PER_CUDA_FUNCTION(createEvent, cudaEventCreateWithFlags, void **, uint32_t)
// PER_CUDA_FUNCTION(destroyEvent, cudaEventDestroy, void *)
// PER_CUDA_FUNCTION(recordEvent, cudaEventRecord, void *, void *)
// PER_CUDA_FUNCTION(eventElapsedTime, cudaEventElapsedTime, float *, void *, void *);
// PER_CUDA_FUNCTION(syncEvent, cudaEventSynchronize, void *);

// kernels
PER_CUDA_FUNCTION(launchCallback, cudaLaunchHostFunc, void *, void *, void *);
PER_CUDA_FUNCTION(launch, cudaLaunchKernel, const void *, dim3, dim3, void **, size_t, void *);
PER_CUDA_FUNCTION(cooperativeLaunch, cudaLaunchCooperativeKernel, const void *, dim3, dim3, void **, size_t, void *);
PER_CUDA_FUNCTION(launchMD, cudaLaunchCooperativeKernelMultiDevice, void *, uint32_t, uint32_t);
