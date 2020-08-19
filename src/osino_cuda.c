#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "osino_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "cudaresultname.h"

#include "threadtracer.h"


#define BLKRES 	(1<<BLKMAG)
#define BLKSIZ	(BLKRES*BLKRES*BLKRES)



static CUdevice device;
static CUcontext context;

static CUmodule module_computefield;
//static CUmodule module_computematter;
static CUmodule module_classify;

static CUfunction function_computefield;
static CUfunction function_computematter;
static CUfunction function_classify;

static CUstream streams[NUMSTREAMS];
static CUdeviceptr fieldptrs[NUMSTREAMS];
static CUdeviceptr casesptrs[NUMSTREAMS];
static value_t* stagingareas_f[NUMSTREAMS];
static uint8_t* stagingareas_c[NUMSTREAMS];


#define CHECK_CUDA \
	{ \
		const cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) \
			fprintf(stderr,"%s:%d err %d - %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
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
	size_t totalmem=0;
	cuDeviceTotalMem(&totalmem, device);
	fprintf(stderr,"Picked device with %lu MiB of memory.\n", totalmem / (1024*1024));
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

	const char* ptxname  = "computefield.ptx";
	const char* ptxname1 = "computefield.ptx";

	const char* funname0 = "osino_computefield";
	const char* funname1 = "osino_computematter";

	moduleLoadResult = cuModuleLoad(&module_computefield, ptxname);
	if (moduleLoadResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModuleLoad error: 0x%x (%s)\n", moduleLoadResult, cudaResultName(moduleLoadResult));
	assert(moduleLoadResult == CUDA_SUCCESS);
	CHECK_CUDA

	getFunctionResult = cuModuleGetFunction(&function_computefield, module_computefield, funname0);
	if (getFunctionResult != CUDA_SUCCESS)
		fprintf(stderr,"cuModulkeGetFunction error: 0x%x (%s)\n", getFunctionResult, cudaResultName(getFunctionResult));
	assert(getFunctionResult == CUDA_SUCCESS);
	CHECK_CUDA

	getFunctionResult = cuModuleGetFunction(&function_computematter, module_computefield, funname1);
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
		const CUresult allocResult = cuMemAlloc(fieldptrs+s, BLKSIZ*sizeof(value_t));
		if (allocResult != CUDA_SUCCESS)
			fprintf(stderr,"cuMemAlloc error: 0x%x (%s)\n", allocResult, cudaResultName(allocResult));
		assert(allocResult == CUDA_SUCCESS);
		assert(fieldptrs[s]);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const CUresult allocResult = cuMemAlloc(casesptrs+s, BLKSIZ*sizeof(uint8_t));
		if (allocResult != CUDA_SUCCESS)
			fprintf(stderr,"cuMemAlloc error: 0x%x (%s)\n", allocResult, cudaResultName(allocResult));
		assert(allocResult == CUDA_SUCCESS);
		assert(casesptrs[s]);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaMallocHost((void**)(stagingareas_f+s), BLKSIZ*sizeof(value_t));
		assert(err == cudaSuccess);
	}

	for (int s=0; s<NUMSTREAMS; ++s)
	{
		const cudaError_t err = cudaMallocHost((void**)(stagingareas_c+s), BLKSIZ*sizeof(uint8_t));
		assert(err == cudaSuccess);
	}

	CHECK_CUDA
}


void osino_client_exit(void)
{
	for (int s=0; s<NUMSTREAMS; ++s)
	{
		cudaError_t err;
		CUresult freeResult;

		err=cudaFreeHost(stagingareas_c[s]);
		assert(err!=cudaErrorInvalidValue);
		assert(err==cudaSuccess);
		stagingareas_c[s] = 0;

		assert(stagingareas_f[s]);
		err=cudaFreeHost(stagingareas_f[s]);
		assert(err!=cudaErrorInvalidValue);
		assert(err==cudaSuccess);
		stagingareas_f[s] = 0;

		freeResult=cuMemFree(casesptrs[s]);
		assert(freeResult == CUDA_SUCCESS);
		casesptrs[s] = 0;

		freeResult=cuMemFree(fieldptrs[s]);
		assert(freeResult == CUDA_SUCCESS);
		fieldptrs[s] = 0;

		err = cudaStreamDestroy( streams[s] );
		assert(err==cudaSuccess);
	}

	CUresult unloadResult;
	unloadResult = cuModuleUnload( module_computefield );
	assert(unloadResult == CUDA_SUCCESS);
	unloadResult = cuModuleUnload( module_classify );
	assert(unloadResult == CUDA_SUCCESS);

	CUresult destroyResult;
	destroyResult = cuCtxDestroy(context);
	assert(destroyResult == CUDA_SUCCESS);
	
}


static int callcount=0;
static int usedcount=0;

int osino_client_computefield(int stride, int gridoff[3], int fullgridsz, float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	const int slot = callcount++ % NUMSTREAMS;
	usedcount++;
	assert(usedcount<NUMSTREAMS);

	void* kernelParms[] =
	{
		fieldptrs+slot,
		&stride,
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
		function_computefield,
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
	//fprintf(stderr,"Executed kernel computefield in slot %d 1st\n", slot);
	return slot;
}


int osino_client_computematter(int stride, int gridoff[3], int fullgridsz, float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	const int slot = callcount++ % NUMSTREAMS;
	usedcount++;
	assert(usedcount<NUMSTREAMS);

	void* kernelParms[] =
	{
		fieldptrs+slot,
		&stride,
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
		function_computematter,
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
	//fprintf(stderr,"Executed kernel computematter in slot %d 2nd\n", slot);
	return slot;
}


void osino_client_classifyfield
(
	int slot, 
	value_t isoval
)
{
	void* kernelParms[] =
	{
		&isoval,
		fieldptrs+slot,
		casesptrs+slot,
		0
	};
	CHECK_CUDA
	const CUresult launchResult = cuLaunchKernel
	(
		function_classify,
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
	CHECK_CUDA
}


void osino_client_sync(int slot)
{
	const char* tags[NUMSTREAMS] =
	{
		"streamsync0", "streamsync1", "streamsync2", "streamsync3", "streamsync4", "streamsync5", "streamsync6", "streamsync7",
		"streamsync8", "streamsync9", "streamsync10", "streamsync11", "streamsync12", "streamsync13", "streamsync14", "streamsync15",
	};
	assert(slot>=0 && slot<NUMSTREAMS);
	TT_BEGIN(tags[slot]);
	cudaStreamSynchronize(streams[slot]);
	TT_END  (tags[slot]);
	CHECK_CUDA
}


void osino_client_release(int slot)
{
	usedcount--;
	assert(usedcount>=0);
}


void osino_client_stagecases(int slot)
{
	const char* tags[NUMSTREAMS] =
	{
		"asynccopy0_c", "asynccopy1_c", "asynccopy2_c", "asynccopy3_c", "asynccopy4_c", "asynccopy5_c", "asynccopy6_c", "asynccopy7_c",
		"asynccopy8_c", "asynccopy9_c", "asynccopy10_c", "asynccopy11_c", "asynccopy12_c", "asynccopy13_c", "asynccopy14_c", "asynccopy15_c",
	};

	assert(slot>=0 && slot<NUMSTREAMS);
	assert(casesptrs[slot]);

	TT_BEGIN(tags[slot]);
	const cudaError_t copyErr = cudaMemcpyAsync(stagingareas_c[slot], (void*)casesptrs[slot], BLKSIZ*sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[slot]);
	if (copyErr != cudaSuccess)
		fprintf(stderr,"cudaMemcpyAsync error: %s\n", cudaGetErrorString(copyErr));
	assert( copyErr == cudaSuccess );
	TT_END  (tags[slot]);
	CHECK_CUDA
}


void osino_client_stagefield(int slot)
{
	const char* tags[NUMSTREAMS] =
	{
		"asynccopy0_f", "asynccopy1_f", "asynccopy2_f", "asynccopy3_f", "asyncopy4_f", "asynccopy5_f", "asynccopy6_f", "asynccopy7_f",
		"asynccopy8_f", "asynccopy9_f", "asynccopy10_f", "asynccopy11_f", "asyncopy12_f", "asynccopy13_f", "asynccopy14_f", "asynccopy15_f",
	};

	assert(slot>=0 && slot<NUMSTREAMS);
	assert(fieldptrs[slot]);

	TT_BEGIN(tags[slot]);
	const cudaError_t copyErr = cudaMemcpyAsync(stagingareas_f[slot], (void*)fieldptrs[slot], BLKSIZ*sizeof(value_t), cudaMemcpyDeviceToHost, streams[slot]);
	if (copyErr != cudaSuccess)
		fprintf(stderr,"cudaMemcpyAsync error: %s\n", cudaGetErrorString(copyErr));
	assert( copyErr == cudaSuccess );
	TT_END  (tags[slot]);
	CHECK_CUDA
}


void osino_client_collectfield(int slot, value_t* output)
{
	const char* tags[NUMSTREAMS] =
	{
		"memcpy0_f", "memcpy1_f", "memcpy2_f", "memcpy3_f", "memcpy4_f", "memcpy5_f", "memcpy6_f", "memcpy7_f",
		"memcpy8_f", "memcpy9_f", "memcpy10_f", "memcpy11_f", "memcpy12_f", "memcpy13_f", "memcpy14_f", "memcpy15_f",
	};

	TT_BEGIN(tags[slot]);
	memcpy(output, stagingareas_f[slot], BLKSIZ*sizeof(value_t));
	TT_END  (tags[slot]);
}


void osino_client_collectcases(int slot, uint8_t* output)
{
	const char* tags[NUMSTREAMS] =
	{
		"memcpy0_c", "memcpy1_c", "memcpy2_c", "memcpy3_c", "memcpy4_c", "memcpy5_c", "memcpy6_c", "memcpy7_c",
		"memcpy8_c", "memcpy9_c", "memcpy10_c", "memcpy11_c", "memcpy12_c", "memcpy13_c", "memcpy14_c", "memcpy15_c",
	};

	TT_BEGIN(tags[slot]);
	memcpy(output, stagingareas_c[slot], BLKSIZ*sizeof(uint8_t));
	TT_END  (tags[slot]);
}


