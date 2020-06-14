
#include "osino_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "cudaresultname.h"

#include "threadtracer.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

#define BLKRES 	(1<<BLKMAG)
#define BLKSIZ	(BLKRES*BLKRES*BLKRES)

static const char* module_fname = "osino.ptx";
static const char* kernel_name  = "osino_computefield";

static CUdevice device;
static CUcontext context;
static CUmodule module;
static CUfunction function;


static CUstream streams[NUMSTREAMS];
static CUdeviceptr deviceptrs[NUMSTREAMS];
static float* stagingareas[NUMSTREAMS];


#define CHECK_CUDA \
	{ \
		const cudaError_t err = cudaGetLastError(); \
		fprintf(stderr,"%s\n", cudaGetErrorString(err)); \
	}


static void pick_device(void)
{
	int nDevices=-1;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		int maxthr=-1;
		cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerBlock, i);
		int wrpsiz=-1;
		cudaDeviceGetAttribute(&wrpsiz, cudaDevAttrWarpSize, i);
		fprintf(stderr, "Device Number: %d\n", i);
		fprintf(stderr, "  Device name: %s\n", prop.name); 
		fprintf(stderr, "  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		fprintf(stderr, "  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		fprintf(stderr, "  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		fprintf(stderr, "  Max threads per block: %d\n", maxthr);
		fprintf(stderr, "  Warp size: %d\n", wrpsiz);
	}
	CUresult deviceGetResult = cuDeviceGet(&device, 0);
	if (deviceGetResult != CUDA_SUCCESS)
		fprintf(stderr,"cuDeviceGet error: 0x%x (%s)\n", deviceGetResult, cudaResultName(deviceGetResult));
}


void osino_client_init(void)
{
	const CUresult initResult = cuInit(0);
	if (initResult != CUDA_SUCCESS)
		fprintf(stderr,"cuInit error: 0x%x (%s)\n", initResult, cudaResultName(initResult));
	assert(initResult == CUDA_SUCCESS);

	pick_device();

	const CUresult createContextResult = cuCtxCreate(&context, 0, device);
	if (createContextResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", createContextResult, cudaResultName(createContextResult));
	assert(createContextResult == CUDA_SUCCESS);

	const CUresult moduleLoadResult = cuModuleLoad(&module, module_fname);
	if (moduleLoadResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", moduleLoadResult, cudaResultName(moduleLoadResult));
	assert(moduleLoadResult == CUDA_SUCCESS);

	const CUresult getFunctionResult = cuModuleGetFunction(&function, module, kernel_name);
	if (getFunctionResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModulkeGetFunction error: 0x%x (%s)\n", getFunctionResult, cudaResultName(getFunctionResult));
	assert(getFunctionResult == CUDA_SUCCESS);

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaStreamCreate( streams+s );
		assert(err == cudaSuccess);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const CUresult allocResult = cuMemAlloc(deviceptrs+s, BLKSIZ*sizeof(float));
		if (allocResult != CUDA_SUCCESS)
			fprintf(stderr,"cuMemAlloc error: 0x%x (%s)\n", allocResult, cudaResultName(allocResult));
		assert(allocResult == CUDA_SUCCESS);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaMallocHost((void**)(stagingareas+s), BLKSIZ*sizeof(float));
		assert(err == cudaSuccess);
	}
}


int osino_client_computefield(int gridoff[3], int fullgridsz, float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	static int callcount=0;
	const int slot = callcount++ % NUMSTREAMS;

	void* kernelParms[] =
	{
		deviceptrs+slot,
		gridoff+0,
		gridoff+1,
		gridoff+2,
		&fullgridsz,
		offsets+0,
		offsets+1,
		offsets+2,
		&domainwarp,
		&freq,
		&lacunarity,
		&persistence,
		0
	};
	const CUresult launchResult = cuLaunchKernel
	(
		function,
		BLKRES*BLKRES,1,1,	// grid dim
		BLKRES,1,1,		// block dim
		0,			// shared mem bytes
		streams[slot],		// hStream
		kernelParms,
		0			// extra
	);
	if (launchResult != CUDA_SUCCESS)
		fprintf(stderr,"cuLaunchKernel error: 0x%x (%s)\n", launchResult, cudaResultName(launchResult));
	assert(launchResult == CUDA_SUCCESS);
	return slot;
}


void osino_client_stagefield(int slot)
{
	const char* tags[2][NUMSTREAMS] =
	{
		{ "streamsync0", "streamsync1", "streamsync2" },
		{ "asynccopy0", "asynccopy1", "asynccopy2" },
	};
	TT_BEGIN(tags[0][slot]);
	cudaStreamSynchronize(streams[slot]);
	TT_END  (tags[0][slot]);

	TT_BEGIN(tags[1][slot]);
	const cudaError_t copyErr = cudaMemcpyAsync(stagingareas[slot], (void*)deviceptrs[slot], BLKSIZ*sizeof(float), cudaMemcpyDeviceToHost, streams[slot]);
	assert( copyErr == cudaSuccess );
	TT_END  (tags[1][slot]);
}


void osino_client_collectfield(int slot, float* output)
{
	const char* tags[2][NUMSTREAMS] =
	{
		{ "streamsync0", "streamsync1", "streamsync2" },
		{ "memcpy0", "memcpy1", "memcpy2" },
	};

	TT_BEGIN(tags[0][slot]);
	cudaStreamSynchronize(streams[slot]);
	TT_END  (tags[0][slot]);
	
	TT_BEGIN(tags[1][slot]);
	memcpy(output, stagingareas[slot], BLKSIZ*sizeof(float));
	TT_END  (tags[1][slot]);
}


