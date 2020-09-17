#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "osino_cuda_client.h"

typedef short value_t;

#define BLKRES	(1<<BLKMAG)

static void dump_volume_as_pgm( const value_t* volume, const char* fname )
{
	uint8_t* raw = (uint8_t*)malloc( BLKRES*BLKRES*BLKRES );
	for (int i=0; i<BLKRES*BLKRES*BLKRES; ++i)
		raw[i] = volume[i]>>8;

	FILE* f_pgm = fopen( fname, "wb" );
	assert( f_pgm );
	//fprintf( f_pgm, "P5\n%d %d\n%d\n", 128, 128*128, 65535 );
	//fwrite( volume, sizeof(value_t)*BLKRES*BLKRES*BLKRES, 1, f_pgm);
	fprintf( f_pgm, "P5\n%d %d\n%d\n", 128, 128*128, 255 );
	fwrite( raw, sizeof(uint8_t)*BLKRES*BLKRES*BLKRES, 1, f_pgm);
	fclose( f_pgm );

	free(raw);
}


int main( int argc, char* argv[] )
{
	osino_cuda_client_init();

	const int fullgridsz = BLKRES-3;
	const int stride = 1;
	float offsets[3] = {-123.4f, -345.6f, -567.890f};
	//float offsets[3] = {0,0,0};
	int gridoff[3] = {0,0,0};

#if 1
	const int rq = osino_cuda_client_computematter
	(
		stride,
		gridoff,
		fullgridsz,
		offsets,
		0.40f,		// domain warp
		1.00f,		// frequency
		0.60f,		// lacunarity
		0.40f		// persistence
	);
#else
	const int rq = osino_cuda_client_computefield
	(
		stride,
		gridoff,
		fullgridsz,
		offsets,
		0.40f,
		1.00f,
		0.60f,
		0.40f
	);
#endif
	osino_cuda_client_sync(rq);

	osino_cuda_client_stagefield(rq);

	value_t* img = (value_t*) malloc( sizeof(value_t) * BLKRES*BLKRES*BLKRES );
	assert(img);

	osino_cuda_client_collectfield(rq, img);

	dump_volume_as_pgm( img, "computed_matter.pgm" );

	return 0;
}

