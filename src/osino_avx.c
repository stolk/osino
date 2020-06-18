
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <fenv.h>

#include "prtintrin.h"

#if defined(MAIN)
#	include "threadtracer.h"
#endif

#if defined( MSWIN )
#	define ALIGNEDPRE __declspec(align(32))
#	define ALIGNEDPST
#else
#	define ALIGNEDPRE
#	define ALIGNEDPST __attribute__ ((aligned (32)))
#endif

#define _mm256_print_ps( V ) \
	{ \
		static ALIGNEDPRE float mem[ 8 ] ALIGNEDPST; \
		_mm256_store_ps( mem, (V) ); \
		printf( "%-5s %f %f %f %f %f %f %f %f\n", #V, mem[7],mem[6],mem[5],mem[4],mem[3],mem[2],mem[1],mem[0] ); \
	}


static inline __m256 dot_2d( __m256 ax, __m256 ay, __m256 bx, __m256 by )
{
	return _mm256_add_ps
	(
		_mm256_mul_ps( ax, bx ),
		_mm256_mul_ps( ay, by )
	);
}


static inline __m256 dot_3d( __m256 ax, __m256 ay, __m256 az,  __m256 bx, __m256 by, __m256 bz )
{
	return _mm256_add_ps
	(
		_mm256_mul_ps( ax, bx ),
		_mm256_add_ps
		(
			_mm256_mul_ps( ay, by ),
			_mm256_mul_ps( az, bz )
		)
	);
}


// Dot product of two vectors with 8 elements each.
#define DOT_8D( A0, A1, A2, A3, A4, A5, A6, A7, B0, B1, B2, B3, B4, B5, B6, B7 ) \
	_mm256_add_ps \
	( \
		_mm256_add_ps \
		( \
			_mm256_add_ps( _mm256_mul_ps(A0, B0), _mm256_mul_ps(A1, B1) ), \
			_mm256_add_ps( _mm256_mul_ps(A2, B2), _mm256_mul_ps(A3, B3) )  \
		), \
		_mm256_add_ps \
		( \
			_mm256_add_ps( _mm256_mul_ps(A4, B4), _mm256_mul_ps(A5, B5) ), \
			_mm256_add_ps( _mm256_mul_ps(A6, B6), _mm256_mul_ps(A7, B7) )  \
		) \
	)



static inline __m256 quintic8( __m256 t )
{
	// (t*6 - 15)
	const __m256 term0 = _mm256_sub_ps
	(
		_mm256_mul_ps( t, _mm256_set1_ps(6) ),
		_mm256_set1_ps(15)
	);
	// (t * (t*6 - 15) + 10)
	const __m256 fact0 = _mm256_add_ps
	(
		_mm256_mul_ps
		(
			t,
			term0
		),
		_mm256_set1_ps(10)
	);
	// t * t * t * ( t * ( t * 6 - 15 ) + 10 )
	return _mm256_mul_ps
	(
		_mm256_mul_ps( t, t ),
		_mm256_mul_ps( t, fact0 )
	);
}


static inline __m256 quinticderiv8( __m256 t )
{
	// (t * (t - 2) + 1)
	const __m256 fact0 = _mm256_add_ps
	(
		_mm256_mul_ps
		(
			t,
			_mm256_sub_ps(t, _mm256_set1_ps(2))
		),
		_mm256_set1_ps(1)
	);
	// 30 * t * t * (t * (t - 2) + 1)
	return _mm256_mul_ps
	(
		_mm256_mul_ps( _mm256_set1_ps(30), t ),
		_mm256_mul_ps( t, fact0 )
	);
}


static inline __m256 mix8(__m256 a, __m256 b, __m256 t)
{
	return _mm256_add_ps
	(
		_mm256_mul_ps( a, _mm256_sub_ps( _mm256_set1_ps(1), t ) ),
		_mm256_mul_ps( b, t )
	);
}


// Murmur2 inspired hashing.
static inline __m256i murmur8( __m256i key, __m256i seed )
{
	const __m256i m = _mm256_set1_epi32(0x5bd1e995);
	__m256i k = _mm256_mullo_epi32
	(
		_mm256_xor_si256( key, seed ),
		m
	);
	k = _mm256_xor_si256
	(
		k,
		_mm256_srli_epi32(k, 24)
	);
	return k;
}


// Creates 8 random directions from the hashing of the 8 (x,y,z) keys.
// Output are 8 unit length vectors.
static inline void randomdir_2d
(
	__m256i x,
	__m256i y,
	__m256* restrict dirx,
	__m256* restrict diry
)
{
	const __m256i seedx = _mm256_set1_epi32(0x17295179);
	const __m256i seedy = _mm256_set1_epi32(0x18732214);

	const __m256i key = _mm256_add_epi32
	(
		_mm256_mullo_epi32( x, _mm256_set1_epi32(8887)),
		_mm256_mullo_epi32( y, _mm256_set1_epi32(7213) )
	);

	const __m256i hx = murmur8( key, seedx );
	const __m256i hy = murmur8( key, seedy );

	// get random bits 0..ffff for 2 candidates.
	const __m256i msk= _mm256_set1_epi32(0x0000ffff);
	const __m256i ax = _mm256_srli_epi32(hx,16);
	const __m256i ay = _mm256_srli_epi32(hy,16);
	const __m256i bx = _mm256_and_si256( hx, msk );
	const __m256i by = _mm256_and_si256( hy, msk );

	// normalize to -1 .. 1
	const __m256 scl = _mm256_set1_ps( 2 / 32768.0f );
	const __m256 one = _mm256_set1_ps( 1 );
	const __m256 cand_a_x = _mm256_sub_ps(_mm256_mul_ps( _mm256_cvtepi32_ps( ax ), scl ), one);
	const __m256 cand_a_y = _mm256_sub_ps(_mm256_mul_ps( _mm256_cvtepi32_ps( ay ), scl ), one);
	const __m256 cand_b_x = _mm256_sub_ps(_mm256_mul_ps( _mm256_cvtepi32_ps( bx ), scl ), one);
	const __m256 cand_b_y = _mm256_sub_ps(_mm256_mul_ps( _mm256_cvtepi32_ps( by ), scl ), one);

	const __m256 lensq_a = dot_2d(cand_a_x, cand_a_y, cand_a_x, cand_a_y );
	const __m256 lensq_b = dot_2d(cand_b_x, cand_b_y, cand_b_x, cand_b_y );

	const __m256 ilen_a = _mm256_rsqrt_ps( lensq_a );
	const __m256 ilen_b = _mm256_rsqrt_ps( lensq_b );

	const __m256i a_is_shorter = _mm256_cmp_ps( lensq_a, lensq_b, _CMP_LE_OS);

	const __m256 norm_a_x = _mm256_mul_ps( cand_a_x, ilen_a );
	const __m256 norm_a_y = _mm256_mul_ps( cand_a_y, ilen_a );
	const __m256 norm_b_x = _mm256_mul_ps( cand_b_x, ilen_b );
	const __m256 norm_b_y = _mm256_mul_ps( cand_b_y, ilen_b );

	*dirx = _mm256_blendv_ps( norm_b_x, norm_a_x, a_is_shorter );
	*diry = _mm256_blendv_ps( norm_b_y, norm_a_y, a_is_shorter );
}

// Creates 8 random directions from the hashing of the 8 (x,y,z) keys.
// Output are 8 unit length vectors.
static inline void randomdir_3d
(
	__m256i x,
	__m256i y,
	__m256i z,
	__m256* restrict dirx,
	__m256* restrict diry,
	__m256* restrict dirz
)
{
	const __m256i seedx = _mm256_set1_epi32(0x17295179);
	const __m256i seedy = _mm256_set1_epi32(0x18732214);
	const __m256i seedz = _mm256_set1_epi32(0x1531f133);

	const __m256i key = _mm256_add_epi32
	(
		_mm256_mullo_epi32( x, _mm256_set1_epi32(8887)),
		_mm256_add_epi32
		(
			_mm256_mullo_epi32( y, _mm256_set1_epi32(7213) ),
			_mm256_mullo_epi32( z, _mm256_set1_epi32(6637) )
		)
	);

	const __m256i hx = murmur8( key, seedx );
	const __m256i hy = murmur8( key, seedy );
	const __m256i hz = murmur8( key, seedz );

	// get random bits 0..ffff for 2 candidates.
	const __m256i msk= _mm256_set1_epi32(0x0000ffff);
	const __m256i ax = _mm256_srli_epi32(hx,16);
	const __m256i ay = _mm256_srli_epi32(hy,16);
	const __m256i az = _mm256_srli_epi32(hz,16);
	const __m256i bx = _mm256_and_si256( hx, msk );
	const __m256i by = _mm256_and_si256( hy, msk );
	const __m256i bz = _mm256_and_si256( hz, msk );

	// normalize to -1 .. 1
	const __m256 scl = _mm256_set1_ps( 2 / 65536.0 );
	const __m256 one = _mm256_set1_ps( 1 );
	const __m256 cand_a_x = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( ax ), scl), one );
	const __m256 cand_a_y = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( ay ), scl), one );
	const __m256 cand_a_z = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( az ), scl), one );
	const __m256 cand_b_x = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( bx ), scl), one );
	const __m256 cand_b_y = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( by ), scl), one );
	const __m256 cand_b_z = _mm256_sub_ps( _mm256_mul_ps( _mm256_cvtepi32_ps( bz ), scl), one );

	const __m256 lensq_a = dot_3d(cand_a_x, cand_a_y, cand_a_z, cand_a_x, cand_a_y, cand_a_z);
	const __m256 lensq_b = dot_3d(cand_b_x, cand_b_y, cand_b_z, cand_b_x, cand_b_y, cand_b_z);

	const __m256 ilen_a = _mm256_rsqrt_ps( lensq_a );
	const __m256 ilen_b = _mm256_rsqrt_ps( lensq_b );

	const __m256i a_is_shorter = _mm256_cmp_ps( lensq_a, lensq_b, _CMP_LE_OS);

	const __m256 norm_a_x = _mm256_mul_ps( cand_a_x, ilen_a );
	const __m256 norm_a_y = _mm256_mul_ps( cand_a_y, ilen_a );
	const __m256 norm_a_z = _mm256_mul_ps( cand_a_z, ilen_a );
	const __m256 norm_b_x = _mm256_mul_ps( cand_b_x, ilen_b );
	const __m256 norm_b_y = _mm256_mul_ps( cand_b_y, ilen_b );
	const __m256 norm_b_z = _mm256_mul_ps( cand_b_z, ilen_b );

	*dirx = _mm256_blendv_ps( norm_b_x, norm_a_x, a_is_shorter );
	*diry = _mm256_blendv_ps( norm_b_y, norm_a_y, a_is_shorter );
	*dirz = _mm256_blendv_ps( norm_b_z, norm_a_z, a_is_shorter );
}


__m256 osino_avx_2d
(
	__m256 x,
	__m256 y
)
{
	__m256 flx = _mm256_floor_ps(x);
	__m256 fly = _mm256_floor_ps(y);

	__m256 tx = _mm256_sub_ps(x, flx);
	__m256 ty = _mm256_sub_ps(y, fly);

	__m256i xi0 = _mm256_cvtps_epi32( flx );
	__m256i yi0 = _mm256_cvtps_epi32( fly );
	__m256i xi1 = _mm256_add_epi32( xi0, _mm256_set1_epi32(1) );
	__m256i yi1 = _mm256_add_epi32( yi0, _mm256_set1_epi32(1) );

	__m256 g00_x, g00_y;
	__m256 g01_x, g01_y;
	__m256 g10_x, g10_y;
	__m256 g11_x, g11_y;

	randomdir_2d( xi0,yi0, &g00_x, &g00_y );
	randomdir_2d( xi0,yi1, &g01_x, &g01_y );
	randomdir_2d( xi1,yi0, &g10_x, &g10_y );
	randomdir_2d( xi1,yi1, &g11_x, &g11_y );

	__m256 ux = _mm256_sub_ps( tx, _mm256_set1_ps(1) );
	__m256 uy = _mm256_sub_ps( ty, _mm256_set1_ps(1) );
	__m256 n00 = dot_2d( g00_x, g00_y, tx, ty );
	__m256 n10 = dot_2d( g10_x, g10_y, ux, ty );
	__m256 n01 = dot_2d( g01_x, g01_y, tx, uy );
	__m256 n11 = dot_2d( g11_x, g11_y, ux, uy );

	__m256 u = quintic8(tx);
	__m256 v = quintic8(ty);

	__m256 m0 = mix8( n00, n10, u );
	__m256 m1 = mix8( n01, n11, u );
	return mix8(m0, m1, v);
}


__m256 osino_avx_3d
(
	__m256 x,
	__m256 y,
	__m256 z
)
{
	__m256 flx = _mm256_floor_ps(x);
	__m256 fly = _mm256_floor_ps(y);
	__m256 flz = _mm256_floor_ps(z);

	__m256 tx = _mm256_sub_ps(x, flx);
	__m256 ty = _mm256_sub_ps(y, fly);
	__m256 tz = _mm256_sub_ps(z, flz);

	__m256i xi0 = _mm256_cvtps_epi32( flx );
	__m256i yi0 = _mm256_cvtps_epi32( fly );
	__m256i zi0 = _mm256_cvtps_epi32( flz );
	__m256i xi1 = _mm256_add_epi32( xi0, _mm256_set1_epi32(1) );
	__m256i yi1 = _mm256_add_epi32( yi0, _mm256_set1_epi32(1) );
	__m256i zi1 = _mm256_add_epi32( zi0, _mm256_set1_epi32(1) );

	__m256 g000_x, g000_y, g000_z;
	__m256 g001_x, g001_y, g001_z;
	__m256 g010_x, g010_y, g010_z;
	__m256 g011_x, g011_y, g011_z;
	__m256 g100_x, g100_y, g100_z;
	__m256 g101_x, g101_y, g101_z;
	__m256 g110_x, g110_y, g110_z;
	__m256 g111_x, g111_y, g111_z;

	randomdir_3d( xi0,yi0,zi0, &g000_x, &g000_y, &g000_z );
	randomdir_3d( xi1,yi0,zi0, &g100_x, &g100_y, &g100_z );
	randomdir_3d( xi0,yi1,zi0, &g010_x, &g010_y, &g010_z );
	randomdir_3d( xi1,yi1,zi0, &g110_x, &g110_y, &g110_z );

	randomdir_3d( xi0,yi0,zi1, &g001_x, &g001_y, &g001_z );
	randomdir_3d( xi1,yi0,zi1, &g101_x, &g101_y, &g101_z );
	randomdir_3d( xi0,yi1,zi1, &g011_x, &g011_y, &g011_z );
	randomdir_3d( xi1,yi1,zi1, &g111_x, &g111_y, &g111_z );

	__m256 ux = _mm256_sub_ps( tx, _mm256_set1_ps(1) );
	__m256 uy = _mm256_sub_ps( ty, _mm256_set1_ps(1) );
	__m256 uz = _mm256_sub_ps( tz, _mm256_set1_ps(1) );

	__m256 n000 = dot_3d( g000_x, g000_y, g000_z, tx, ty, tz );
	__m256 n100 = dot_3d( g100_x, g100_y, g100_z, ux, ty, tz );
	__m256 n010 = dot_3d( g010_x, g010_y, g010_z, tx, uy, tz );
	__m256 n110 = dot_3d( g110_x, g110_y, g110_z, ux, uy, tz );

	__m256 n001 = dot_3d( g001_x, g001_y, g001_z, tx, ty, uz );
	__m256 n101 = dot_3d( g101_x, g101_y, g101_z, ux, ty, uz );
	__m256 n011 = dot_3d( g011_x, g011_y, g011_z, tx, uy, uz );
	__m256 n111 = dot_3d( g111_x, g111_y, g111_z, ux, uy, uz );

	const __m256 u = quintic8(tx);
	const __m256 v = quintic8(ty);
	const __m256 w = quintic8(tz);

	const __m256 k0 = n000;
	const __m256 k1 = _mm256_sub_ps( n100, n000 );
	const __m256 k2 = _mm256_sub_ps( n010, n000 );
	const __m256 k3 = _mm256_sub_ps( n001, n000 );
	const __m256 k4 = _mm256_sub_ps
	(
		_mm256_add_ps( n000, n110 ),
		_mm256_add_ps( n100, n010 )
	);
	const __m256 k5 = _mm256_sub_ps
	(
		_mm256_add_ps( n000, n101 ),
		_mm256_add_ps( n100, n001 )
	);
	const __m256 k6 = _mm256_sub_ps
	(
		_mm256_add_ps( n000, n011 ),
		_mm256_add_ps( n010, n001 )
	);
	const __m256 k7 = _mm256_sub_ps
	(
		_mm256_add_ps
		(
			_mm256_add_ps( n100, n010 ),
			_mm256_add_ps( n001, n111 )
		),
		_mm256_add_ps
		(
			_mm256_add_ps( n000, n110 ),
			_mm256_add_ps( n101, n011 )
		)
	);

	const __m256 vw = _mm256_mul_ps( v, w );

	const __m256 uw  = _mm256_mul_ps( u, w );
	const __m256 uv  = _mm256_mul_ps( u, v );
	const __m256 uvw = _mm256_mul_ps( u, vw );
	const __m256 one = _mm256_set1_ps( 1 );

	const __m256 fieldvalue = DOT_8D(k0,k1,k2,k3,k4,k5,k6,k7, one,u,v,w,uv,uw,vw,uvw);

	return _mm256_mul_ps(fieldvalue, _mm256_set1_ps(1.50f));	// not sure why we need to do this, but the range was too narrow.

#if 0
	__m256 nx00 = mix8( n000, n100, u);
	__m256 nx01 = mix8( n001, n101, u);
	__m256 nx10 = mix8( n010, n110, u);
	__m256 nx11 = mix8( n011, n111, u);

	__m256 nxy0 = mix8( nx00, nx10, v);
	__m256 nxy1 = mix8( nx01, nx11, v);
	*valu = mix8(nxy0, nxy1, w);
#endif

}


__m256 osino_avx_2d_4o
(
	__m256 x, __m256 y,				// sample position.
	float lacunarity,				// gaps between frequencies.
	float persistence				// amplitude scaling between 2 successive octaves.
)
{
	const float invlacunarity = 1.0f / lacunarity;
	const __m256 fr1 = _mm256_set1_ps( invlacunarity );
	const __m256 fr2 = _mm256_set1_ps( invlacunarity*invlacunarity );
	const __m256 fr3 = _mm256_set1_ps( invlacunarity*invlacunarity*invlacunarity );

	const float a0 = 1.0f;
	const float a1 = persistence;
	const float a2 = persistence*persistence;
	const float a3 = persistence*persistence*persistence;
	const float tot= a0+a1+a2+a3;

	const __m256 am0 = _mm256_set1_ps( a0 / tot );
	const __m256 am1 = _mm256_set1_ps( a1 / tot );
	const __m256 am2 = _mm256_set1_ps( a2 / tot );
	const __m256 am3 = _mm256_set1_ps( a3 / tot );

	__m256 nv0,nv1,nv2,nv3;

	nv0 = osino_avx_2d(x,y);
	const __m256 v0 = _mm256_mul_ps( nv0, am0 );

	const __m256 x1 = _mm256_mul_ps( x, fr1 );
	const __m256 y1 = _mm256_mul_ps( y, fr1 );
	nv1 = osino_avx_2d(x1,y1);
	const __m256 v1 = _mm256_mul_ps( nv1, am1 );

	const __m256 x2 = _mm256_mul_ps( x, fr2 );
	const __m256 y2 = _mm256_mul_ps( y, fr2 );
	nv2 = osino_avx_2d(x2,y2);
	const __m256 v2 = _mm256_mul_ps( nv2, am2 );

	const __m256 x3 = _mm256_mul_ps( x, fr3 );
	const __m256 y3 = _mm256_mul_ps( y, fr3 );
	nv3 = osino_avx_2d(x3,y3);
	const __m256 v3 = _mm256_mul_ps( nv3, am3 );

	const __m256 v = _mm256_add_ps
	(
		_mm256_add_ps( v0, v1 ),
		_mm256_add_ps( v2, v3 )
	);

	return v;
}


__m256 osino_avx_3d_4o
(
	__m256 x, __m256 y, __m256 z,			// sample position.
	float lacunarity,				// gaps between frequencies.
	float persistence				// amplitude scaling between 2 successive octaves.
)
{
	const float invlacunarity = 1.0f / lacunarity;
	const __m256 fr1 = _mm256_set1_ps( invlacunarity );
	const __m256 fr2 = _mm256_set1_ps( invlacunarity*invlacunarity );
	const __m256 fr3 = _mm256_set1_ps( invlacunarity*invlacunarity*invlacunarity );

	const float a0 = 1.0f;
	const float a1 = persistence;
	const float a2 = persistence*persistence;
	const float a3 = persistence*persistence*persistence;
	const float tot= a0+a1+a2+a3;

	const __m256 am0 = _mm256_set1_ps( a0 / tot );
	const __m256 am1 = _mm256_set1_ps( a1 / tot );
	const __m256 am2 = _mm256_set1_ps( a2 / tot );
	const __m256 am3 = _mm256_set1_ps( a3 / tot );

	__m256 nv0,nv1,nv2,nv3;

	nv0 = osino_avx_3d(x,y,z);
	const __m256 v0 = _mm256_mul_ps( nv0, am0 );

	const __m256 x1 = _mm256_mul_ps( x, fr1 );
	const __m256 y1 = _mm256_mul_ps( y, fr1 );
	const __m256 z1 = _mm256_mul_ps( z, fr1 );
	nv1 = osino_avx_3d(x1,y1,z1);
	const __m256 v1 = _mm256_mul_ps( nv1, am1 );

	const __m256 x2 = _mm256_mul_ps( x, fr2 );
	const __m256 y2 = _mm256_mul_ps( y, fr2 );
	const __m256 z2 = _mm256_mul_ps( z, fr2 );
	nv2 = osino_avx_3d(x2,y2,z2);
	const __m256 v2 = _mm256_mul_ps( nv2, am2 );

	const __m256 x3 = _mm256_mul_ps( x, fr3 );
	const __m256 y3 = _mm256_mul_ps( y, fr3 );
	const __m256 z3 = _mm256_mul_ps( z, fr3 );
	nv3 = osino_avx_3d(x3,y3,z3);
	const __m256 v3 = _mm256_mul_ps( nv3, am3 );

	const __m256 v = _mm256_add_ps
	(
		_mm256_add_ps( v0, v1 ),
		_mm256_add_ps( v2, v3 )
	);

	return v;
}

