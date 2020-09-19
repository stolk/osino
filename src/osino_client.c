#include <stdint.h>

#include "osino_client.h"

#include "osino_cuda_client.h"
#include "osino_opencl_client.h"

#include "logx.h"

static uint64_t platform=0;

//! Before using compute kernels.
uint64_t osino_client_init(uint64_t flags)
{
	if (flags & OSINO_CUDA)
	{
		const int ok = osino_cuda_client_init();
		if ( ok )
		{
			platform = OSINO_CUDA;
			return platform;
		}
	}
	if (flags & OSINO_OPENCL)
	{
		const int ok = osino_opcl_client_init();
		if ( ok )
		{
			platform = OSINO_OPENCL;
			return platform;
		}
	}
	platform = 0;
	return platform;
}


//! When done with compute kernels.
void osino_client_exit(void)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_exit();
	if (platform == OSINO_OPENCL)
		osino_opcl_client_exit();
}


const char* osino_client_platformname(void)
{
	if (platform == OSINO_CUDA)
		return "cuda";
	if (platform == OSINO_OPENCL)
		return "opencl";
	return "none";
}

//! Synchronize the cuda stream.
void osino_client_sync(int slot)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_sync(slot);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_sync(slot);
}

//! Called when done with stream.
void osino_client_release(int slot)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_release(slot);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_release(slot);
}

//! Check to see how many slots in use.
int  osino_client_usage(void)
{
	if (platform == OSINO_CUDA)
		return osino_cuda_client_usage();
	if (platform == OSINO_OPENCL)
		return osino_opcl_client_usage();
	return -1;
}

//! Returns a request id.
int osino_client_computefield(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	if (platform == OSINO_CUDA)
		return osino_cuda_client_computefield  (stride, gridOff, fullres, offsets, domainwarp, freq, lacunarity, persistence);
	if (platform == OSINO_OPENCL)
		return osino_opcl_client_computefield(stride, gridOff, fullres, offsets, domainwarp, freq, lacunarity, persistence);
	return -1;
}

//! Returns a request id.
int  osino_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	if (platform == OSINO_CUDA)
		return osino_cuda_client_computematter(stride, gridOff, fullres, offsets, domainwarp, freq, lacunarity, persistence);
	if (platform == OSINO_OPENCL)
		return osino_opcl_client_computematter(stride, gridOff, fullres, offsets, domainwarp, freq, lacunarity, persistence);
	return -1;
}

//! Classify the Marching Cubes in the field.
void osino_client_classifyfield(int slot, value_t isoval)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_classifyfield(slot, isoval);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_classifyfield(slot, isoval);
}

//! Kick off a memory transfer from GPU to CPU.
void osino_client_stagefield(int slot)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_stagefield(slot);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_stagefield(slot);
}

//! Kick off a memory transfer from GPU to CPU.
void osino_client_stagecases(int slot)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_stagecases(slot);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_stagecases(slot);
}

//! Collect the results.
void osino_client_collectfield(int slot, value_t* output)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_collectfield(slot, output);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_collectfield(slot, output);
}

//! Collect the results.
void osino_client_collectcases(int slot, uint8_t* output)
{
	if (platform == OSINO_CUDA)
		osino_cuda_client_collectcases(slot, output);
	if (platform == OSINO_OPENCL)
		osino_opcl_client_collectcases(slot, output);
}


//! Test the Osino client implementation by running kernel, and retrieving the results.
void osino_client_test(void)
{
	const int stride=1;
	int gridOff[3] = { 0,0,0 };
	const int fullres = BLKRES;
	float offsets[3] = { 12.34f, 34.56f, 56.67f };
	const float domainwarp = 0.4f;
	const float freq = 1.0f;
	const float lacunarity = 0.4f;
	const float persistence = 0.9f;
	const int rq = osino_client_computefield
	(
		stride,
		gridOff,
		fullres,
		offsets,
		domainwarp,
		freq,
		lacunarity,
		persistence
	);
	LOGI("Fired off compute rq %d.", rq);

	osino_client_classifyfield(rq, -28000);
	LOGI("Fired off classification.");

	osino_client_stagefield(rq);
	LOGI("Fired off staging of field.");

	osino_client_stagecases(rq);
	LOGI("Fired off staging of cases.");

	osino_client_sync(rq);
	LOGI("Synchronized queue.");

	value_t fimg[BLKSIZ];
	osino_client_collectfield(rq, fimg);
	LOGI("Collected field.");

	uint8_t cimg[BLKSIZ];
	osino_client_collectcases(rq, cimg);
	LOGI("Collected cases.");

	FILE* f;

	value_t* freader = fimg + BLKRES * BLKRES * BLKRES / 2;
	f = fopen("field.pgm","wb");
	ASSERT(f);
	fprintf( f, "P5\n%d %d\n%d\n", BLKRES, BLKRES, 65535 );
	fwrite( freader, BLKRES*BLKRES*sizeof(value_t), 1, f );
	fclose(f);

	uint8_t* creader = cimg + BLKRES * BLKRES * BLKRES / 2;
	f = fopen("cases.pgm","wb");
	ASSERT(f);
	fprintf( f, "P5\n%d %d\n%d\n", BLKRES, BLKRES, 255 );
	fwrite( creader, BLKRES*BLKRES*sizeof(uint8_t), 1, f );
	fclose(f);

	osino_client_release(rq);
	LOGI("Released client rq %d.", rq);
	LOGI("Wrote osino test results to field.pgm and cases.pgm");
}

