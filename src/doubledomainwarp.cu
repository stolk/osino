// This is the Open Simplex Noise algorithm, ported to Enoki/CUDA.
//
// Original Open Simplex Algorithm: Stefan Gustavson (public domain.)
// http://webstaff.itn.liu.se/~stegu/simplexnoise
//
// C port by Bram Stolk
// https://github.com/stolk/sino
//
// Port to Enoki by Bram Stolk
// https://github.com/stolk/osino
//
// License: 3-clause BSD to match Enoki License.
//
// osino.cpp

#include <inttypes.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_fp16.h>

#if defined(STORECHARS)
#include "bluenoise.h"
#include "honeycomb.h"
#endif

#define IMRES	(1<<IMMAG)
#define IMMSK	(IMRES-1)


// Skewing / Unskewing factors for 2, 3, and 4 dimensions.
#define F2      0.3660254037844386f     // 0.5*(Math.sqrt(3.0)-1.0);
#define G2      0.21132486540518713f    // (3.0-Math.sqrt(3.0))/6.0;
#define F3      0.3333333333333333f     // 1.0/3.0;
#define G3      0.16666666666666666f    // 1.0/6.0;
#define F4      0.30901699437494745f    // (Math.sqrt(5.0)-1.0)/4.0;
#define G4      0.1381966011250105f     // (5.0-Math.sqrt(5.0))/20.0;


typedef short value_t;

unsigned char pal[4][3] =
{
	0xff,0xff,0xff,
	0x7F,0xC2,0xC2,
	0xE1,0x77,0x4B,
	0xF4,0xDB,0x60,
};


// Hash function that we use to generate random directions.
__device__ __forceinline__
int murmur(int key, uint32_t seed)
{
	int k = ( key ^ seed ) * 0x5bd1e995;
	k = k ^ (k>>24);
	return k;
}


// Dot product.
__device__ __forceinline__
float dot_2d( float ax, float ay, float bx, float by )
{
	return ax * bx + ay * by;
}



// Generates a random 2D direction for specified grid location.
#define RANDOMDIR_2D(x0,y0,PRF) \
	const int PRF ## hx = murmur( x0*8887+y0*7213, 0x17295179 ); \
	const int PRF ## hy = murmur( x0*8887+y0*7213, 0x18732214 ); \
	const int PRF ## ax = (PRF ## hx)>>16; \
	const int PRF ## ay = (PRF ## hy)>>16; \
	const int PRF ## bx = PRF ## hx & 0x0000ffff; \
	const int PRF ## by = PRF ## hy & 0x0000ffff; \
	const float PRF ## cand_a_x = PRF ## ax * (2/65536.0f) - 1; \
	const float PRF ## cand_a_y = PRF ## ay * (2/65536.0f) - 1; \
	const float PRF ## cand_b_x = PRF ## bx * (2/65536.0f) - 1; \
	const float PRF ## cand_b_y = PRF ## by * (2/65536.0f) - 1; \
        const float PRF ## lensq_a = dot_2d(PRF ## cand_a_x, PRF ## cand_a_y, PRF ## cand_a_x, PRF ## cand_a_y); \
        const float PRF ## lensq_b = dot_2d(PRF ## cand_b_x, PRF ## cand_b_y, PRF ## cand_b_x, PRF ## cand_b_y); \
        const float PRF ## ilen_a = rsqrtf( PRF ## lensq_a ); \
        const float PRF ## ilen_b = rsqrtf( PRF ## lensq_b ); \
        const auto PRF ## a_is_shorter = ( PRF ## lensq_a < PRF ## lensq_b ); \
        const float PRF ## norm_a_x = ( PRF ## cand_a_x * PRF ## ilen_a ); \
        const float PRF ## norm_a_y = ( PRF ## cand_a_y * PRF ## ilen_a ); \
        const float PRF ## norm_b_x = ( PRF ## cand_b_x * PRF ## ilen_b ); \
        const float PRF ## norm_b_y = ( PRF ## cand_b_y * PRF ## ilen_b ); \
	const float PRF ## _x = PRF ## a_is_shorter ? PRF ## norm_a_x :  PRF ## norm_b_x; \
	const float PRF ## _y = PRF ## a_is_shorter ? PRF ## norm_a_y :  PRF ## norm_b_y; \


// Open Simplex Noise 2D
__device__
__noinline__
float osino_2d(float x, float y)
{
	// Skew
	const float s = ( x + y ) * F2;
	const float flx = floorf(x+s);
	const float fly = floorf(y+s);
	const float t = (flx+fly) * G2;
	const int i = (int)flx;
	const int j = (int)fly;
	// Unskew
	const float X0 = flx - t;
	const float Y0 = fly - t;
	const float x0 = x - X0;
	const float y0 = y - Y0;
	// Determine which simplex.
	const int i1 = x0>y0 ? 1 : 0;
	const int j1 = x0>y0 ? 0 : 1;
	const float x1 = x0 - i1 + G2;
	const float y1 = y0 - j1 + G2;
	const float x2 = x0 - 1.0f + 2.0f * G2;
	const float y2 = y0 - 1.0f + 2.0f * G2;
	// Generate a random direction for each corner.
	RANDOMDIR_2D((i   ), (j   ), grad0);
	RANDOMDIR_2D((i+i1), (j+j1), grad1);
	RANDOMDIR_2D((i+ 1), (j+ 1), grad2);
	const float t0 = 0.5f - x0*x0 - y0*y0;
	const float t1 = 0.5f - x1*x1 - y1*y1;
	const float t2 = 0.5f - x2*x2 - y2*y2;
	const float p0 = t0*t0*t0*t0 * dot_2d(grad0_x, grad0_y, x0, y0);
	const float p1 = t1*t1*t1*t1 * dot_2d(grad1_x, grad1_y, x1, y1);
	const float p2 = t2*t2*t2*t2 * dot_2d(grad2_x, grad2_y, x2, y2);
	const float n0 = t0<0 ? 0 : p0;
	const float n1 = t1<0 ? 0 : p1;
	const float n2 = t2<0 ? 0 : p2;
	// Add contributions from each corner and scale to [-1,1] interval.
	return 70.0f * ( n0 + n1 + n2 );
}


// Do 4 octaves of open simplex noise in 2D.
__device__
float osino_2d_4o( float x, float y )
{
	const float n0 = osino_2d(  x,   y);
	const float n1 = osino_2d(2*x, 2*y);
	const float n2 = osino_2d(4*x, 4*y);
	const float n3 = osino_2d(8*x, 8*y);
	return (1/1.875f) * ( n0 + 0.5f * n1 + 0.25f * n2 + 0.125f * n3 );
}



extern "C"
{

__global__
void doubledomainwarp
(
	value_t* field,
	float offset_x,
	float offset_y,
	float domainwarp0,
	float domainwarp1,
	float freq
)
{
	const int xc = threadIdx.x;
	const int yc = blockIdx.x & IMMSK;

	const float ifull = 1.0f / IMRES;
	const float s0 = 2.017f * ifull;
	const float s1 = 2.053f * ifull;
	float x = xc * s0;
	float y = yc * s1;

	const float w0x = osino_2d(offset_x+411+y, offset_y+423-x) * domainwarp1;
	const float w0y = osino_2d(offset_x+419-y, offset_y+413+x) * domainwarp1;

	const float w1x = osino_2d(offset_x+711-w0x, offset_y+723-w0y) * domainwarp0;
	const float w1y = osino_2d(offset_x-719+w0x, offset_y+713+w0y) * domainwarp0;

	x += w1x;
	y += w1y;

	const int idx = (yc * (IMRES)) + xc;
	float result = osino_2d_4o(offset_x+freq*x, offset_y+freq*y);
	result = result < -1 ? -1 : result;
	result = result >  1 ?  1 : result;
	field[ idx ] = (value_t) ( result * 32767.0f );
}


}// extern C


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



__host__
int main(int argc, char* argv[])
{
	float opt_freq = 1.092f;
	float opt_warp0 = 0.1f;
	float opt_warp1 = 0.1f;
	const char* opt_out="out_doubledomainwarp.ppm";
	for ( int i=1; i<argc; ++i)
	{
		if (!strncmp(argv[i],"freq=",5))  opt_freq  = atof(argv[i]+5);
		if (!strncmp(argv[i],"warp0=",6)) opt_warp0 = atof(argv[i]+6);
		if (!strncmp(argv[i],"warp1=",6)) opt_warp1 = atof(argv[i]+6);
		if (!strncmp(argv[i],"out=",4))   opt_out   = argv[i]+4;
	}
	query();

	const int N = IMRES*IMRES;

	value_t* field = 0;
	cudaMallocManaged(&field, N*sizeof(value_t));
	assert(field);

	CHECK_CUDA

	doubledomainwarp<<<IMRES,IMRES>>>(field, 14567.89f, 21123.46f, opt_warp0, opt_warp1, opt_freq );
	fprintf( stderr,"warp0 %f warp1 %f", opt_warp0, opt_warp1 );

	cudaDeviceSynchronize();
	CHECK_CUDA

	unsigned char im[IMRES*IMRES][3];

	for ( int i=0; i<IMRES*IMRES; ++i )
	{
		int idx = 0;
		const value_t v = field[i];
		if ( v>  2000 && v< 4000 ) idx=1;
		if ( v>  5000 && v< 9000 ) idx=2;
		if ( v> 10000 && v<12000 ) idx=1;
		if ( v<-16000 ) idx=3;
		im[i][0] = pal[idx][0];
		im[i][1] = pal[idx][1];
		im[i][2] = pal[idx][2];
	}

	FILE* f = fopen(opt_out,"wb");
	fprintf(f, "P6\n%d %d\n255\n", IMRES, IMRES);
	fwrite( im, sizeof(im), 1, f );
	fclose(f);
	cudaFree(field);
	return 0;
}

