#ifdef __cplusplus
extern "C"
{
#endif

#if defined(STOREFP16)
typedef __fp16 value_t;
#elif defined(STORECHARS)
typedef unsigned char value_t;
#elif defined(STORESHORTS)
typedef signed short value_t;
#else
typedef float value_t;
#endif

#define NUMSTREAMS 16

extern void osino_cuda_client_init(void);

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

extern void osino_cuda_client_classifyfield(int slot, value_t isoval);

extern void osino_cuda_client_stagefield(int slot);
extern void osino_cuda_client_stagecases(int slot);

extern void osino_cuda_client_collectfield(int slot, value_t* output);
extern void osino_cuda_client_collectcases(int slot, uint8_t* output);

#ifdef __cplusplus
}
#endif
