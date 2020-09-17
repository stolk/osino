#ifdef __cplusplus
extern "C"
{
#endif

//! Before using compute kernels. Returns 0 on failure.
extern int  osino_cuda_client_init(void);

//! When done with compute kernels.
extern void osino_cuda_client_exit(void);

//! Synchronize the cuda stream.
extern void osino_cuda_client_sync(int slot);

//! Called when done with stream.
extern void osino_cuda_client_release(int slot);

//! Check to see how many slots in use.
extern int  osino_cuda_client_usage(void);

//! Returns a request id.
extern int  osino_cuda_client_computefield (int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Returns a request id.
extern int  osino_cuda_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Classify the Marching Cubes in the field.
extern void osino_cuda_client_classifyfield(int slot, value_t isoval);

//! Kick off a memory transfer from GPU to CPU.
extern void osino_cuda_client_stagefield(int slot);

//! Kick off a memory transfer from GPU to CPU.
extern void osino_cuda_client_stagecases(int slot);

//! Collect the results.
extern void osino_cuda_client_collectfield(int slot, value_t* output);

//! Collect the results.
extern void osino_cuda_client_collectcases(int slot, uint8_t* output);

#ifdef __cplusplus
}
#endif
