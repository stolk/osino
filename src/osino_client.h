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

#define BLKRES	(1<<BLKMAG)
#define BLKSIZ	(BLKRES*BLKRES*BLKRES)


enum OsinoPlatformFlags
{
	OSINO_CUDA=1,
	OSINO_OPENCL=2,
};

//! Before using compute kernels.
extern uint64_t osino_client_init(uint64_t flags);

//! When done with compute kernels.
extern void osino_client_exit(void);

//! Returns "opencl" or "cuda"
extern const char* osino_client_platformname(void);

//! Synchronize the cuda stream.
extern void osino_client_sync(int slot);

//! Called when done with stream.
extern void osino_client_release(int slot);

//! Check to see how many slots in use.
extern int  osino_client_usage(void);

//! Check how many opencl platforms.
extern int  osino_client_num_opencl_platforms(void);

//! Returns a request id.
extern int  osino_client_computefield (int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Returns a request id.
extern int  osino_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Classify the Marching Cubes in the field.
extern void osino_client_classifyfield(int slot, value_t isoval);

//! Kick off a memory transfer from GPU to CPU.
extern void osino_client_stagefield(int slot);

//! Kick off a memory transfer from GPU to CPU.
extern void osino_client_stagecases(int slot);

//! Collect the results.
extern void osino_client_collectfield(int slot, value_t* output);

//! Collect the results.
extern void osino_client_collectcases(int slot, uint8_t* output);

//! Test implementation by executing a kernel and retrieving the results.
extern void osino_client_test(void);

#ifdef __cplusplus
}
#endif
