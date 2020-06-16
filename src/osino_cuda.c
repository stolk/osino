
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



static CUdevice device;
static CUcontext context;

static CUmodule module_compute;
static CUmodule module_classify;

static CUfunction function_compute;
static CUfunction function_classify;

static CUstream streams[NUMSTREAMS];
static CUdeviceptr deviceptrs[NUMSTREAMS];
static value_t* stagingareas[NUMSTREAMS];


#define CHECK_CUDA \
	{ \
		const cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) \
			fprintf(stderr,"%s\n", cudaGetErrorString(err)); \
		assert(err == cudaSuccess); \
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
	fprintf(stderr,"osino_cuda: value_t has size %lu\n", sizeof(value_t));

	const CUresult initResult = cuInit(0);
	if (initResult != CUDA_SUCCESS)
		fprintf(stderr,"cuInit error: 0x%x (%s)\n", initResult, cudaResultName(initResult));
	assert(initResult == CUDA_SUCCESS);

	pick_device();

	const CUresult createContextResult = cuCtxCreate(&context, 0, device);
	if (createContextResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", createContextResult, cudaResultName(createContextResult));
	assert(createContextResult == CUDA_SUCCESS);

	CUresult moduleLoadResult;
	CUresult getFunctionResult;

	// compute

#if defined(STOREFP16)
	const char* ptxname = "computefieldfp16.ptx";
	const char* funname = "osino_computefield_fp16";
#elif defined(STORESHORTS)
	const char* ptxname = "computefieldh.ptx";
	const char* funname = "osino_computefield_h";
#else
	const char* ptxname = "computefield.ptx";
	const char* funname = "osino_computefield";
#endif

	moduleLoadResult = cuModuleLoad(&module_compute, ptxname);
	if (moduleLoadResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", moduleLoadResult, cudaResultName(moduleLoadResult));
	assert(moduleLoadResult == CUDA_SUCCESS);
	CHECK_CUDA

	getFunctionResult = cuModuleGetFunction(&function_compute, module_compute, funname);
	if (getFunctionResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModulkeGetFunction error: 0x%x (%s)\n", getFunctionResult, cudaResultName(getFunctionResult));
	assert(getFunctionResult == CUDA_SUCCESS);
	CHECK_CUDA

	// classify

	moduleLoadResult = cuModuleLoad(&module_classify, "classifyfield.ptx");
	if (moduleLoadResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", moduleLoadResult, cudaResultName(moduleLoadResult));
	assert(moduleLoadResult == CUDA_SUCCESS);
	CHECK_CUDA

	getFunctionResult = cuModuleGetFunction(&function_classify, module_classify, "osino_classifyfield");
	if (getFunctionResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModulkeGetFunction error: 0x%x (%s)\n", getFunctionResult, cudaResultName(getFunctionResult));
	assert(getFunctionResult == CUDA_SUCCESS);
	CHECK_CUDA

	// set up streams

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaStreamCreate( streams+s );
		assert(err == cudaSuccess);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const CUresult allocResult = cuMemAlloc(deviceptrs+s, BLKSIZ*sizeof(value_t));
		if (allocResult != CUDA_SUCCESS)
			fprintf(stderr,"cuMemAlloc error: 0x%x (%s)\n", allocResult, cudaResultName(allocResult));
		assert(allocResult == CUDA_SUCCESS);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaMallocHost((void**)(stagingareas+s), BLKSIZ*sizeof(value_t));
		assert(err == cudaSuccess);
	}
	CHECK_CUDA
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
		function_compute,
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

	CHECK_CUDA
	assert(slot>=0 && slot<NUMSTREAMS);
	assert(deviceptrs[slot]);

	TT_BEGIN(tags[0][slot]);
	cudaStreamSynchronize(streams[slot]);
	TT_END  (tags[0][slot]);
	CHECK_CUDA

	TT_BEGIN(tags[1][slot]);
	const cudaError_t copyErr = cudaMemcpyAsync(stagingareas[slot], (void*)deviceptrs[slot], BLKSIZ*sizeof(value_t), cudaMemcpyDeviceToHost, streams[slot]);
	if (copyErr != cudaSuccess)
		fprintf(stderr,"cudaMemcpyAsync error: %s\n", cudaGetErrorString(copyErr));
	assert( copyErr == cudaSuccess );
	TT_END  (tags[1][slot]);
}


void osino_client_collectfield(int slot, value_t* output)
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
	memcpy(output, stagingareas[slot], BLKSIZ*sizeof(value_t));
	TT_END  (tags[1][slot]);
}


