#include <metal_math>
#include <metal_assert>

#define BLKMAG  8
#define	BLKRES	(1<<BLKMAG)


// Skewing / Unskewing factors for 2, 3, and 4 dimensions.
#define F2      0.3660254037844386f     // 0.5*(Math.sqrt(3.0)-1.0);
#define G2      0.21132486540518713f    // (3.0-Math.sqrt(3.0))/6.0;
#define F3      0.3333333333333333f     // 1.0/3.0;
#define G3      0.16666666666666666f    // 1.0/6.0;
#define F4      0.30901699437494745f    // (Math.sqrt(5.0)-1.0)/4.0;
#define G4      0.1381966011250105f     // (5.0-Math.sqrt(5.0))/20.0;


typedef float value_t;


// Hash function that we use to generate random directions.
inline
int murmur(int key, uint32_t seed)
{
	int k = ( key ^ seed ) * 0x5bd1e995;
	k = k ^ (k>>24);
	return k;
}


// Dot product.
inline
float dot_2d( float ax, float ay, float bx, float by )
{
	return ax * bx + ay * by;
}


// Dot product.
inline
float dot_3d( float ax, float ay, float az, float bx, float by, float bz )
{
	return ax * bx + ay * by + az * bz;
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
        const float PRF ## ilen_a = metal::rsqrt( PRF ## lensq_a ); \
        const float PRF ## ilen_b = metal::rsqrt( PRF ## lensq_b ); \
        const auto PRF ## a_is_shorter = ( PRF ## lensq_a < PRF ## lensq_b ); \
        const float PRF ## norm_a_x = ( PRF ## cand_a_x * PRF ## ilen_a ); \
        const float PRF ## norm_a_y = ( PRF ## cand_a_y * PRF ## ilen_a ); \
        const float PRF ## norm_b_x = ( PRF ## cand_b_x * PRF ## ilen_b ); \
        const float PRF ## norm_b_y = ( PRF ## cand_b_y * PRF ## ilen_b ); \
	const float PRF ## _x = PRF ## a_is_shorter ? PRF ## norm_a_x :  PRF ## norm_b_x; \
	const float PRF ## _y = PRF ## a_is_shorter ? PRF ## norm_a_y :  PRF ## norm_b_y; \


// Generates a random 3D direction for specified grid location.
#define RANDOMDIR_3D(x,y,z, PRF) \
	const int key ## PRF = x * 8887 + y * 7213 + z * 6637; \
	const int hx ## PRF = murmur( key ## PRF, 0x17295179 ); \
	const int hy ## PRF = murmur( key ## PRF, 0x18732214 ); \
	const int hz ## PRF = murmur( key ## PRF, 0x1531f133 ); \
	const int ax ## PRF = (hx ## PRF)>>16; \
	const int ay ## PRF = (hy ## PRF)>>16; \
	const int az ## PRF = (hz ## PRF)>>16; \
	const int bx ## PRF = hx ## PRF & 0x0000ffff; \
	const int by ## PRF = hy ## PRF & 0x0000ffff; \
	const int bz ## PRF = hz ## PRF & 0x0000ffff; \
	const float cand_a_x ## PRF = ax ## PRF * (2/65536.0f) - 1; \
	const float cand_a_y ## PRF = ay ## PRF * (2/65536.0f) - 1; \
	const float cand_a_z ## PRF = az ## PRF * (2/65536.0f) - 1; \
	const float cand_b_x ## PRF = bx ## PRF * (2/65536.0f) - 1; \
	const float cand_b_y ## PRF = by ## PRF * (2/65536.0f) - 1; \
	const float cand_b_z ## PRF = bz ## PRF * (2/65536.0f) - 1; \
        const float lensq_a ## PRF = dot_3d(cand_a_x ## PRF, cand_a_y ## PRF, cand_a_z ## PRF, cand_a_x ## PRF, cand_a_y ## PRF, cand_a_z ## PRF); \
        const float lensq_b ## PRF = dot_3d(cand_b_x ## PRF, cand_b_y ## PRF, cand_b_z ## PRF, cand_b_x ## PRF, cand_b_y ## PRF, cand_b_z ## PRF); \
        const float ilen_a ## PRF = metal::rsqrt( lensq_a ## PRF ); \
        const float ilen_b ## PRF = metal::rsqrt( lensq_b ## PRF ); \
        const auto a_is_shorter ## PRF = ( lensq_a ## PRF < lensq_b ## PRF ); \
        const float norm_a_x ## PRF = ( cand_a_x ## PRF * ilen_a ## PRF ); \
        const float norm_a_y ## PRF = ( cand_a_y ## PRF * ilen_a ## PRF ); \
        const float norm_a_z ## PRF = ( cand_a_z ## PRF * ilen_a ## PRF ); \
        const float norm_b_x ## PRF = ( cand_b_x ## PRF * ilen_b ## PRF ); \
        const float norm_b_y ## PRF = ( cand_b_y ## PRF * ilen_b ## PRF ); \
        const float norm_b_z ## PRF = ( cand_b_z ## PRF * ilen_b ## PRF ); \
	const float PRF ## _x = a_is_shorter ## PRF ? norm_a_x ## PRF : norm_b_x ## PRF; \
	const float PRF ## _y = a_is_shorter ## PRF ? norm_a_y ## PRF : norm_b_y ## PRF; \
	const float PRF ## _z = a_is_shorter ## PRF ? norm_a_z ## PRF : norm_b_z ## PRF; \


// Open Simplex Noise 2D
float osino_2d(float x, float y)
{
	// Skew
	const float s = ( x + y ) * F2;
    const float flx = metal::floor(x+s);
    const float fly = metal::floor(y+s);
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


// Open Simplex Noise 3D
float osino_3d(float x, float y, float z)
{
	// Skew
	const float s = ( x+y+z ) * F3;
    const float flx = metal::floor(x+s);
    const float fly = metal::floor(y+s);
    const float flz = metal::floor(z+s);

	const int i = (int) flx;
	const int j = (int) fly;
	const int k = (int) flz;

	const float t = (flx+fly+flz) * G3;

	// Unskew
	const float X0 = flx - t;
	const float Y0 = fly - t;
	const float Z0 = flz - t;

	const float x0 = x - X0;
	const float y0 = y - Y0;
	const float z0 = z - Z0;

	// Which simplex?
	const auto is_order_xyz = ( x0 >= y0 && y0 >= z0);
	const auto is_order_xzy = ( x0 >= z0 && z0 >= y0);
	const auto is_order_zxy = ( z0 >= x0 && x0 >= y0);
	const auto is_order_zyx = ( z0 >= y0 && y0 >= x0);
	const auto is_order_yzx = ( y0 >= z0 && z0 >= x0);
	const auto is_order_yxz = ( y0 >= x0 && x0 >= z0);

	const int i1 = is_order_xyz || is_order_xzy ? 1:0;
	const int j1 = is_order_yzx || is_order_yxz ? 1:0;
	const int k1 = is_order_zxy || is_order_zyx ? 1:0;

	const int i2 = is_order_zyx || is_order_yzx ? 0:1;
	const int j2 = is_order_xzy || is_order_zxy ? 0:1;
	const int k2 = is_order_xyz || is_order_yxz ? 0:1;

	const float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	const float y1 = y0 - j1 + G3;
	const float z1 = z0 - k1 + G3;
	const float x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
	const float y2 = y0 - j2 + 2.0f*G3;
	const float z2 = z0 - k2 + 2.0f*G3;
	const float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
	const float y3 = y0 - 1.0f + 3.0f*G3;
	const float z3 = z0 - 1.0f + 3.0f*G3;

	// Get 4 random directions, for each corner of simplex.
	RANDOMDIR_3D((i   ), (j   ), (k   ), grad0);
	RANDOMDIR_3D((i+i1), (j+j1), (k+k1), grad1);
	RANDOMDIR_3D((i+i2), (j+j2), (k+k2), grad2);
	RANDOMDIR_3D((i+1 ), (j+1 ), (k+1 ), grad3);

	const float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
	const float p0 = t0*t0*t0*t0 * dot_3d(grad0_x, grad0_y, grad0_z, x0, y0, z0);
	const float n0 = t0<0 ? 0 : p0;

	const float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
	const float p1 = t1*t1*t1*t1 * dot_3d(grad1_x, grad1_y, grad1_z, x1, y1, z1);
	const float n1 = t1<0 ? 0 : p1;

	const float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
	const float p2 = t2*t2*t2*t2 * dot_3d(grad2_x, grad2_y, grad2_z, x2, y2, z2);
	const float n2 = t2<0 ? 0 : p2;

	const float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
	const float p3 = t3*t3*t3*t3 * dot_3d(grad3_x, grad3_y, grad3_z, x3, y3, z3);
	const float n3 = t3<0 ? 0 : p3;

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0f * ( n0 + n1 + n2 + n3 );
}


// Do 4 octaves of open simplex noise in 2D.
float osino_2d_4o( float x, float y )
{
	const float n0 = osino_2d(  x,   y);
	const float n1 = osino_2d(2*x, 2*y);
	const float n2 = osino_2d(4*x, 4*y);
	const float n3 = osino_2d(8*x, 8*y);
	return (1/1.875f) * ( n0 + 0.5f * n1 + 0.25f * n2 + 0.125f * n3 );
}


// Do 4 octaves of open simplex noise in 3D.
float osino_3d_4o( float x, float y, float z, float lacunarity, float persistence )
{
	const float a1 = persistence;
	const float a2 = persistence*a1;
	const float a3 = persistence*a2;
	const float scl = 1.0f / (1.0f+a1+a2+a3);

	const float ilac = 1.0f / lacunarity;
	const float f1 = ilac;
	const float f2 = f1*f1;
	const float f3 = f2*f2;

	const float n0 = osino_3d(   x,    y,    z);
	const float n1 = osino_3d(f1*x, f1*y, f1*z);
	const float n2 = osino_3d(f2*x, f2*y, f2*z);
	const float n3 = osino_3d(f3*x, f3*y, f3*z);
	return scl * ( n0 + a1 * n1 + a2 * n2 + a3 * n3 );
}

struct GridParams
{
    int stride;
    int gridoff_x;
    int gridoff_y;
    int gridoff_z;
    int fullgridsz;
};

struct NoiseParams
{
    float offset_x;
    float offset_y;
    float offset_z;
    float domainwarp;
    float freq;
    float lacunarity;
    float persistence;
};

kernel
void osino_computefield2
(
 device float* field [[ buffer(0) ]],
 device struct GridParams& gridparams [[ buffer(1) ]],
 device struct NoiseParams& noiseparams [[ buffer(2) ]],
 uint index [[thread_position_in_grid]]
)
{
    const int zc = ((index>>0) & 0xff);
    const int yc = ((index>>8) & 0xff);
    const int xc = ((index>>8) & 0xff);
    field[index] = yc / 255.0f;
}

kernel
void osino_computefield
(
    device float* field [[ buffer(0) ]],
    device struct GridParams& gridparams [[ buffer(1) ]],
    device struct NoiseParams& noiseparams [[ buffer(2) ]],
    uint3 index [[thread_position_in_grid]]
)
{
	const int zc = index[2];
	const int yc = index[1];
	const int xc = index[0];
	const float ifull = 1.0f / gridparams.fullgridsz;
	const float s0 = 2.017f * ifull;
	const float s1 = 2.053f * ifull;
	const float s2 = 2.099f * ifull;
	float x = ( (xc*gridparams.stride+gridparams.gridoff_x) - 0.5f*gridparams.fullgridsz ) * s0;
	float y = ( (yc*gridparams.stride+gridparams.gridoff_y) - 0.5f*gridparams.fullgridsz ) * s1;
	float z = ( (zc*gridparams.stride+gridparams.gridoff_z) - 0.5f*gridparams.fullgridsz ) * s2;

#if 1
	const float lsq_unwarped = x*x + y*y + z*z; // 0 .. 0.25
	const float depth = 0.25f - lsq_unwarped;
	const float clippeddepth = metal::fmax(depth, 0.0f);
	const float warpstrength = noiseparams.domainwarp * (0.5 + clippeddepth * 10.0f);
	const float wx = osino_3d(noiseparams.offset_x+11+y, noiseparams.offset_y+23-z, noiseparams.offset_z+17+x) * warpstrength;
	const float wy = osino_3d(noiseparams.offset_x+19-z, noiseparams.offset_y+13+x, noiseparams.offset_z+11-y) * warpstrength;
	const float wz = osino_3d(noiseparams.offset_x+31+x, noiseparams.offset_y+41-z, noiseparams.offset_z+61+y) * warpstrength;
	x += wx;
	y += wy;
	z += wz;
#endif
	const float lsq = x*x + y*y + z*z;	// 0 .. 0.25
	const float len = metal::sqrt(lsq);
	const float d = 2.0f - 4.0f * len;

	const float v = osino_3d_4o
	(
		noiseparams.offset_x+noiseparams.freq*x,
		noiseparams.offset_y+noiseparams.freq*y,
		noiseparams.offset_z+noiseparams.freq*z,
		noiseparams.lacunarity,
		noiseparams.persistence
	);

	const int idx = xc*BLKRES*BLKRES + yc*BLKRES + zc;
	float result = v;  //v+d;
	result = result < -1 ? -1 : result;
	result = result >  1 ?  1 : result;

	field[ idx ] = result;
}

