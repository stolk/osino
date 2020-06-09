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

#include <iostream>
#include <enoki/array.h>
#include <enoki/cuda.h>

#include "threadtracer.h"


typedef enoki::CUDAArray<int32_t>  IV;	// Int vector
typedef enoki::CUDAArray<float>    FV;	// Flt vector



// Skewing / Unskewing factors for 2, 3, and 4 dimensions.
#define F2      0.3660254037844386f     // 0.5*(Math.sqrt(3.0)-1.0);
#define G2      0.21132486540518713f    // (3.0-Math.sqrt(3.0))/6.0;
#define F3      0.3333333333333333f     // 1.0/3.0;
#define G3      0.16666666666666666f    // 1.0/6.0;
#define F4      0.30901699437494745f    // (Math.sqrt(5.0)-1.0)/4.0;
#define G4      0.1381966011250105f     // (5.0-Math.sqrt(5.0))/20.0;



// Hash function that we use to generate random directions.
static inline IV murmur(IV key, uint32_t seed)
{
	IV k = ( key ^ seed ) * 0x5bd1e995;
	k = k ^ enoki::sr<24>(k);
	return k;
}


// Dot product.
static inline FV dot_2d( FV ax, FV ay, FV bx, FV by )
{
	return ax * bx + ay * by;
}


// Dot product.
static inline FV dot_3d( FV ax, FV ay, FV az, FV bx, FV by, FV bz )
{
	return ax * bx + ay * by + az * bz;
}


// Generates a random 2D direction for specified grid location.
#define RANDOMDIR_2D(x0,y0,PRF) \
	const IV PRF ## hx = murmur( x0*8887+y0*7213, 0x17295179 ); \
	const IV PRF ## hy = murmur( x0*8887+y0*7213, 0x18732214 ); \
	const IV PRF ## ax = enoki::sr<16>(PRF ## hx); \
	const IV PRF ## ay = enoki::sr<16>(PRF ## hy); \
	const IV PRF ## bx = PRF ## hx & 0x0000ffff; \
	const IV PRF ## by = PRF ## hy & 0x0000ffff; \
	const FV PRF ## cand_a_x = FV(PRF ## ax) * (2/65536.0f) - 1; \
	const FV PRF ## cand_a_y = FV(PRF ## ay) * (2/65536.0f) - 1; \
	const FV PRF ## cand_b_x = FV(PRF ## bx) * (2/65536.0f) - 1; \
	const FV PRF ## cand_b_y = FV(PRF ## by) * (2/65536.0f) - 1; \
        const FV PRF ## lensq_a = dot_2d(PRF ## cand_a_x, PRF ## cand_a_y, PRF ## cand_a_x, PRF ## cand_a_y); \
        const FV PRF ## lensq_b = dot_2d(PRF ## cand_b_x, PRF ## cand_b_y, PRF ## cand_b_x, PRF ## cand_b_y); \
        const FV PRF ## ilen_a = enoki::rsqrt( PRF ## lensq_a ); \
        const FV PRF ## ilen_b = enoki::rsqrt( PRF ## lensq_b ); \
        const auto PRF ## a_is_shorter = ( PRF ## lensq_a < PRF ## lensq_b ); \
        const FV PRF ## norm_a_x = ( PRF ## cand_a_x * PRF ## ilen_a ); \
        const FV PRF ## norm_a_y = ( PRF ## cand_a_y * PRF ## ilen_a ); \
        const FV PRF ## norm_b_x = ( PRF ## cand_b_x * PRF ## ilen_b ); \
        const FV PRF ## norm_b_y = ( PRF ## cand_b_y * PRF ## ilen_b ); \
	const FV PRF ## _x = enoki::select( PRF ## a_is_shorter, PRF ## norm_a_x, PRF ## norm_b_x ); \
	const FV PRF ## _y = enoki::select( PRF ## a_is_shorter, PRF ## norm_a_y, PRF ## norm_b_y ); \


// Generates a random 3D direction for specified grid location.
#define RANDOMDIR_3D(x,y,z, PRF) \
	const IV key ## PRF = x * 8887 + y * 7213 + z * 6637; \
	const IV hx ## PRF = murmur( key ## PRF, 0x17295179 ); \
	const IV hy ## PRF = murmur( key ## PRF, 0x18732214 ); \
	const IV hz ## PRF = murmur( key ## PRF, 0x1531f133 ); \
	const IV ax ## PRF = enoki::sr<16>(hx ## PRF); \
	const IV ay ## PRF = enoki::sr<16>(hy ## PRF); \
	const IV az ## PRF = enoki::sr<16>(hz ## PRF); \
	const IV bx ## PRF = hx ## PRF & 0x0000ffff; \
	const IV by ## PRF = hy ## PRF & 0x0000ffff; \
	const IV bz ## PRF = hz ## PRF & 0x0000ffff; \
	const FV cand_a_x ## PRF = FV(ax ## PRF) * (2/65536.0f) - 1; \
	const FV cand_a_y ## PRF = FV(ay ## PRF) * (2/65536.0f) - 1; \
	const FV cand_a_z ## PRF = FV(az ## PRF) * (2/65536.0f) - 1; \
	const FV cand_b_x ## PRF = FV(bx ## PRF) * (2/65536.0f) - 1; \
	const FV cand_b_y ## PRF = FV(by ## PRF) * (2/65536.0f) - 1; \
	const FV cand_b_z ## PRF = FV(bz ## PRF) * (2/65536.0f) - 1; \
        const FV lensq_a ## PRF = dot_3d(cand_a_x ## PRF, cand_a_y ## PRF, cand_a_z ## PRF, cand_a_x ## PRF, cand_a_y ## PRF, cand_a_z ## PRF); \
        const FV lensq_b ## PRF = dot_3d(cand_b_x ## PRF, cand_b_y ## PRF, cand_b_z ## PRF, cand_b_x ## PRF, cand_b_y ## PRF, cand_b_z ## PRF); \
        const FV ilen_a ## PRF = enoki::rsqrt( lensq_a ## PRF ); \
        const FV ilen_b ## PRF = enoki::rsqrt( lensq_b ## PRF ); \
        const auto a_is_shorter ## PRF = ( lensq_a ## PRF < lensq_b ## PRF ); \
        const FV norm_a_x ## PRF = ( cand_a_x ## PRF * ilen_a ## PRF ); \
        const FV norm_a_y ## PRF = ( cand_a_y ## PRF * ilen_a ## PRF ); \
        const FV norm_a_z ## PRF = ( cand_a_z ## PRF * ilen_a ## PRF ); \
        const FV norm_b_x ## PRF = ( cand_b_x ## PRF * ilen_b ## PRF ); \
        const FV norm_b_y ## PRF = ( cand_b_y ## PRF * ilen_b ## PRF ); \
        const FV norm_b_z ## PRF = ( cand_b_z ## PRF * ilen_b ## PRF ); \
	const FV PRF ## _x = enoki::select( a_is_shorter ## PRF, norm_a_x ## PRF, norm_b_x ## PRF ); \
	const FV PRF ## _y = enoki::select( a_is_shorter ## PRF, norm_a_y ## PRF, norm_b_y ## PRF ); \
	const FV PRF ## _z = enoki::select( a_is_shorter ## PRF, norm_a_z ## PRF, norm_b_z ## PRF ); \


// Open Simplex Noise 2D
FV osino_2d(FV x, FV y)
{
	// Skew
	const FV s = ( x + y ) * F2;
	const FV flx = enoki::floor(x+s);
	const FV fly = enoki::floor(y+s);
	const FV t = (flx+fly) * G2;
	const IV i = IV(flx);
	const IV j = IV(fly);
	// Unskew
	const FV X0 = flx - t;
	const FV Y0 = fly - t;
	const FV x0 = x - X0;
	const FV y0 = y - Y0;
	// Determine which simplex.
	const IV i1 = enoki::select(x0>y0, IV(1), IV(0));
	const IV j1 = enoki::select(x0>y0, IV(0), IV(1));
	const FV x1 = x0 - i1 + G2;
	const FV y1 = y0 - j1 + G2;
	const FV x2 = x0 - 1.0f + 2.0f * G2;
	const FV y2 = y0 - 1.0f + 2.0f * G2;
	// Generate a random direction for each corner.
	RANDOMDIR_2D((i   ), (j   ), grad0);
	RANDOMDIR_2D((i+i1), (j+j1), grad1);
	RANDOMDIR_2D((i+ 1), (j+ 1), grad2);
	const FV t0 = 0.5f - x0*x0 - y0*y0;
	const FV t1 = 0.5f - x1*x1 - y1*y1;
	const FV t2 = 0.5f - x2*x2 - y2*y2;
	const FV n0 = enoki::select(t0<0, 0, t0*t0*t0*t0 * dot_2d(grad0_x, grad0_y, x0, y0));
	const FV n1 = enoki::select(t1<0, 0, t1*t1*t1*t1 * dot_2d(grad1_x, grad1_y, x1, y1));
	const FV n2 = enoki::select(t2<0, 0, t2*t2*t2*t2 * dot_2d(grad2_x, grad2_y, x2, y2));
	// Add contributions from each corner and scale to [-1,1] interval.
	return 70.0f * ( n0 + n1 + n2 );
}


// Open Simplex Noise 3D
FV osino_3d(FV x, FV y, FV z)
{
	// Skew
	const FV s = ( x+y+z ) * F3;
	const FV flx = enoki::floor(x+s);
	const FV fly = enoki::floor(y+s);
	const FV flz = enoki::floor(z+s);

	const IV i( flx );
	const IV j( fly );
	const IV k( flz );

	const FV t = (flx+fly+flz) * G3;

	// Unskew
	const FV X0 = flx - t;
	const FV Y0 = fly - t;
	const FV Z0 = flz - t;

	const FV x0 = x - X0;
	const FV y0 = y - Y0;
	const FV z0 = z - Z0;

	// Which simplex?
	const auto is_order_xyz = ( x0 >= y0 && y0 >= z0);
	const auto is_order_xzy = ( x0 >= z0 && z0 >= y0);
	const auto is_order_zxy = ( z0 >= x0 && x0 >= y0);
	const auto is_order_zyx = ( z0 >= y0 && y0 >= x0);
	const auto is_order_yzx = ( y0 >= z0 && z0 >= x0);
	const auto is_order_yxz = ( y0 >= x0 && x0 >= z0);

	const IV i1 = enoki::select(is_order_xyz || is_order_xzy, IV(1), IV(0));
	const IV j1 = enoki::select(is_order_yzx || is_order_yxz, IV(1), IV(0));
	const IV k1 = enoki::select(is_order_zxy || is_order_zyx, IV(1), IV(0));

	const IV i2 = enoki::select(is_order_zyx || is_order_yzx, IV(0), IV(1));
	const IV j2 = enoki::select(is_order_xzy || is_order_zxy, IV(0), IV(1));
	const IV k2 = enoki::select(is_order_xyz || is_order_yxz, IV(0), IV(1));

	const FV x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	const FV y1 = y0 - j1 + G3;
	const FV z1 = z0 - k1 + G3;
	const FV x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
	const FV y2 = y0 - j2 + 2.0f*G3;
	const FV z2 = z0 - k2 + 2.0f*G3;
	const FV x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
	const FV y3 = y0 - 1.0f + 3.0f*G3;
	const FV z3 = z0 - 1.0f + 3.0f*G3;

	// Get 4 random directions, for each corner of simplex.
	RANDOMDIR_3D((i   ), (j   ), (k   ), grad0);
	RANDOMDIR_3D((i+i1), (j+j1), (k+k1), grad1);
	RANDOMDIR_3D((i+i2), (j+j2), (k+k2), grad2);
	RANDOMDIR_3D((i+1 ), (j+1 ), (k+1 ), grad3);

	const FV t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
	const FV n0 = enoki::select( t0<0, 0, t0*t0*t0*t0 * dot_3d(grad0_x, grad0_y, grad0_z, x0, y0, z0) );

	const FV t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
	const FV n1 = enoki::select( t1<0, 0, t1*t1*t1*t1 * dot_3d(grad1_x, grad1_y, grad1_z, x1, y1, z1) );

	const FV t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
	const FV n2 = enoki::select( t2<0, 0, t2*t2*t2*t2 * dot_3d(grad2_x, grad2_y, grad2_z, x2, y2, z2) );

	const FV t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
	const FV n3 = enoki::select( t3<0, 0, t3*t3*t3*t3 * dot_3d(grad3_x, grad3_y, grad3_z, x3, y3, z3) );

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0f * ( n0 + n1 + n2 + n3 );
}


// Do 4 octaves of open simplex noise in 2D.
FV osino_2d_4o( FV x, FV y )
{
	const FV n0 = osino_2d(  x,   y);
	const FV n1 = osino_2d(2*x, 2*y);
	const FV n2 = osino_2d(4*x, 4*y);
	const FV n3 = osino_2d(8*x, 8*y);
	return (1/1.875f) * ( n0 + 0.5f * n1 + 0.25f * n2 + 0.125f * n3 );
}


// Do 4 octaves of open simplex noise in 3D.
FV osino_3d_4o( FV x, FV y, FV z )
{
	const FV n0 = osino_3d(  x,   y,   z);
	const FV n1 = osino_3d(2*x, 2*y, 2*z);
	const FV n2 = osino_3d(4*x, 4*y, 4*z);
	const FV n3 = osino_3d(8*x, 8*y, 8*z);
	return (1/1.875f) * ( n0 + 0.5f * n1 + 0.25f * n2 + 0.125f * n3 );
}


extern "C" {


static FV* field=0;

void osino_computefield(const int gridoff[3], int fullgridsz)
{
	if (!field) field = new FV;
	const int mag = BLKMAG;
	const int sz = (1<<mag);
	const int msk = sz-1;
	const int cnt = sz * sz * sz;
	const IV ix = enoki::arange<IV>( cnt );
	const IV zc = ix & msk;
	const IV yc = enoki::sr<mag>(ix) & msk;
	const IV xc = enoki::sr<mag+mag>(ix) & msk;
	const float s0 = 2.000f / (fullgridsz);
	const float s1 = 2.003f / (fullgridsz);
	const float s2 = 2.005f / (fullgridsz);
	FV x = ( FV(xc+gridoff[0]) - fullgridsz/2 ) * s0;
	FV y = ( FV(yc+gridoff[1]) - fullgridsz/2 ) * s1;
	FV z = ( FV(zc+gridoff[2]) - fullgridsz/2 ) * s2;

#if 0
	const FV lsq_unwarped = x*x + y*y + z*z; // 0 .. 0.25
	const FV depth = 0.25f - lsq_unwarped;
	const FV warpstrength = 0.29f + enoki::max(0, depth) * 6.6f;
	const FV wx = osino_3d(11+y, 23-z, 17+x) * warpstrength;
	const FV wy = osino_3d(19-z, 13+x, 11-y) * warpstrength;
	const FV wz = osino_3d(31+x, 41-z, 61+y) * warpstrength;
	x += wx;
	y += wy;
	z += wz;
#endif

	const FV lsq = x*x + y*y + z*z;	// 0 .. 0.25
	const FV len = enoki::sqrt(lsq);
	const FV d = 2.0f - 4.0f * len;

	const FV v = osino_3d_4o(1.2f*x,1.2f*y,1.2f*z);
	//const FV v = osino_3d(x,y,z);

	*field = enoki::clamp(v + d, -1, 1);
	//*field = v;

	TT_BEGIN("cuda_eval");
	enoki::cuda_eval(); // may return before the GPU finished executing the kernel.
	TT_END  ("cuda_eval");

	
	TT_BEGIN("cuda_sync");
	enoki::cuda_sync();
	TT_END  ("cuda_sync");
}


void osino_collectfield(float* __restrict volume)
{
	//enoki::cuda_sync();
	const float* data = field->managed().data();

#if 0
	const char* whos = enoki::cuda_whos();
	fprintf(stderr, "%s", whos);
#endif

	const int mag = BLKMAG;
	const int sz = (1<<mag);
	const int cnt = sz * sz * sz;
	memcpy(volume, data, cnt*sizeof(float));
}


void osino_mkfield(float* __restrict volume)
{
	const int off[3] = {0,0,0};
	const int fullgridsz = (1<<BLKMAG);
	osino_computefield(off, fullgridsz);
	osino_collectfield(volume);
}

#if 0
static void bench_osino2d(bool write)
{
	TT_BEGIN("osino_2d");
	const int mag = 13;
	const int sz  = (1<<mag);
	const int cnt = sz * sz;

	const IV ix = enoki::arange< IV >( cnt );
	const IV xc = ix & 0x001fff;
	const IV yc = enoki::sr<mag>(ix) & 0x001fff;

	const FV x = FV(xc) * 0.00101f;
	const FV y = FV(yc) * 0.00102f;

	const FV wx = osino_2d(102-0.2f*y,  13+0.2f*x);
	const FV wy = osino_2d(  6+0.2f*x, 107-0.2f*y);

	//const FV values = osino_2d(x,y);
	const FV values = osino_2d_4o(x+wx,y+wy);
	TT_END("osino_2d");

	TT_BEGIN("eval");
	enoki::cuda_eval();
	TT_END  ("eval");

	TT_BEGIN("sync");
	enoki::cuda_sync();
	TT_END  ("sync");

	TT_BEGIN("managed");
	const float* data = values.managed().data();
	TT_END("managed");

	const char* s = enoki::cuda_whos();
	fprintf(stderr, "%s", s);

	if (write)
	{
		unsigned char* im = (unsigned char*)malloc(cnt);
		for (int i=0; i<cnt; ++i)
			im[i] = (unsigned char) ( 127.5f + 127.5f * data[i] );
		fprintf(stdout, "P5\n%d %d\n255\n", sz, sz );
		fwrite(im, sz, sz, stdout);
		free(im);
	}
}

static void bench_osino3d(bool write)
{
	TT_BEGIN("osino_3d");
	const int mag = 13;
	const int sz  = (1<<mag);
	const int cnt = sz * sz;

	const IV ix = enoki::arange< IV >( cnt );
	const IV xc = ix & 0x001fff;
	const IV yc = enoki::sr<mag>(ix) & 0x001fff;
	const IV zc(700);

	const FV x = FV(xc) * 0.00101f;
	const FV y = FV(yc) * 0.00102f;
	const FV z = FV(zc) * 0.00103f;

	const FV values = osino_3d_4o(x,y,z);
	//const FV values = osino_3d(x,y,z);
	TT_END("osino_3d");

	TT_BEGIN("eval");
	enoki::cuda_eval();
	TT_END  ("eval");

	TT_BEGIN("sync");
	enoki::cuda_sync();
	TT_END  ("sync");

	TT_BEGIN("managed");
	const float* data = values.managed().data();
	TT_END("managed");

	const char* s = enoki::cuda_whos();
	fprintf(stderr, "%s", s);

	if (write)
	{
		unsigned char* im = (unsigned char*)malloc(cnt);
		for (int i=0; i<cnt; ++i)
			im[i] = (unsigned char) ( 127.5f + 127.5f * data[i] );
		fprintf(stdout, "P5\n%d %d\n255\n", sz, sz );
		fwrite(im, sz, sz, stdout);
		free(im);
	}
}
#endif


#if defined(XSTANDALONE)
int main(int argc, char* argv[])
{
	tt_signin(-1, "main");

#if 1
	bench_osino3d(0);
	bench_osino3d(1);
#endif

#if 0
	bench_osino2d(0);
	bench_osino2d(1);
#endif

	tt_report("osino.json");

	return 0;
}
#endif

} // extern C
