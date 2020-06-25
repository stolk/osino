#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h>

#define BLKRES	(1<<BLKMAG)

#if defined(STOREFP16)
typedef __half value_t;
#elif defined(STORECHARS)
typedef unsigned char value_t;
#else
typedef float value_t;
#endif


extern "C"
{
__global__
void osino_classifyfield
(
	float isoval,
	const value_t* field,
	uint8_t* cases
)
{
	const int zc = threadIdx.x;
	const int yc = blockIdx.x & 0xff;
	const int xc = (blockIdx.x >> 8);

	const int i0 = zc + yc*BLKRES + xc*BLKRES*BLKRES;
	//assert(i0 < BLKRES*BLKRES*BLKRES);

	const int stridex = xc < BLKRES-1 ? BLKRES*BLKRES : 0;
	const int stridey = yc < BLKRES-1 ? BLKRES : 0;
	const int stridez = zc < BLKRES-1 ? 1 : 0;

	const int i1 = i0 + stridex;
	const int i2 = i1 + stridey;
	const int i3 = i0 + stridey;

	const int i4 = i0+stridez;
	const int i5 = i1+stridez;
	const int i6 = i2+stridez;
	const int i7 = i3+stridez;

#if defined(STORECHARS)
	const float scl = 1.0f / 128.0f
	const float v0 = -1.0f + field[i0]*scl;
	const float v1 = -1.0f + field[i1]*scl;
	const float v2 = -1.0f + field[i2]*scl;
	const float v3 = -1.0f + field[i3]*scl;
	const float v4 = -1.0f + field[i4]*scl;
	const float v5 = -1.0f + field[i5]*scl;
	const float v6 = -1.0f + field[i6]*scl;
	const float v7 = -1.0f + field[i7]*scl;
#elif defined(STOREFP16)
	const float v0 = __half2float(field[i0]);
	const float v1 = __half2float(field[i1]);
	const float v2 = __half2float(field[i2]);
	const float v3 = __half2float(field[i3]);
	const float v4 = __half2float(field[i4]);
	const float v5 = __half2float(field[i5]);
	const float v6 = __half2float(field[i6]);
	const float v7 = __half2float(field[i7]);
#else
	const float v0 = field[i0];
	const float v1 = field[i1];
	const float v2 = field[i2];
	const float v3 = field[i3];
	const float v4 = field[i4];
	const float v5 = field[i5];
	const float v6 = field[i6];
	const float v7 = field[i7];
#endif
	const int bit0 = v0 <= isoval ? 0x01 : 0;
	const int bit1 = v1 <= isoval ? 0x02 : 0;
	const int bit2 = v2 <= isoval ? 0x04 : 0;
	const int bit3 = v3 <= isoval ? 0x08 : 0;
	const int bit4 = v4 <= isoval ? 0x10 : 0;
	const int bit5 = v5 <= isoval ? 0x20 : 0;
	const int bit6 = v6 <= isoval ? 0x40 : 0;
	const int bit7 = v7 <= isoval ? 0x80 : 0;

	const uint8_t c = bit0|bit1|bit2|bit3|bit4|bit5|bit6|bit7;
	cases[i0] = c;
}

__host__
void query(void)
{
	int nDevices=-1;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
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
}

#define CHECK_CUDA \
	{ \
		const cudaError_t err = cudaGetLastError(); \
		fprintf(stderr,"%s\n", cudaGetErrorString(err)); \
	}
}


__host__
int main(int argc, char* argv[])
{
	query();

	const int N = 256*256*256;

	value_t* field = 0;
	cudaMallocManaged(&field, N*sizeof(value_t));
	assert(field);
	CHECK_CUDA

	uint8_t* cases = 0;
	cudaMallocManaged(&cases, N*sizeof(uint8_t));
	assert(cases);
	CHECK_CUDA

	//osino_computefield<<<BLKRES*BLKRES,BLKRES>>>(field, 0,0,0, BLKRES, 0,0,0, 1.0f, 1.0f, 0.5f, 0.5f );

	osino_classifyfield<<<BLKRES*BLKRES,BLKRES>>>( 0.0f, field, cases );
	CHECK_CUDA

	cudaDeviceSynchronize();
	CHECK_CUDA

	FILE* f = fopen("out_classify.pgm","wb");
	fprintf(f, "P5\n%d %d\n255\n", BLKRES, BLKRES);
	for (int i=0; i<256*256; ++i)
		fputc(cases[i],f);
	fclose(f);

	cudaFree(cases);
	cudaFree(field);

	return 0;
}
