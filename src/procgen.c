// procgen.c
// (c) Bram Stolk

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <fenv.h>
#include <math.h>

#include "procgen.h"
#include "osino.h"
#include "surface.h"

#include "threadtracer.h"

static float fielddensity[BLKSIZ];
static uint8_t fieldtype[BLKSIZ];

static int numtria=0;

#define MAXTRIA	600000
#define MAXVERTS (MAXTRIA*3)

static float   surface_v[MAXVERTS*3];
static float   surface_n[MAXVERTS*3];
static uint8_t surface_m[MAXVERTS  ];


void splat(void)
{
	float o = 0.5f * ( BLKRES-1 );
	float s = 1.0f / o;
	float *writer = fielddensity;
	for (int x=0; x<BLKRES; ++x)
		for (int y=0; y<BLKRES; ++y)
			for (int z=0; z<BLKRES; ++z)
			{
				float xc = (x-o) * s;
				float yc = (y-o) * s;
				float zc = (z-o) * s;
				float lensqr = xc*xc + yc*yc + zc*zc;
				float len = lensqr > 0 ? sqrtf(lensqr) : 1;
				*writer++ = -0.5f + len;
			}
}


int procgen_asteroid(float* fdens, uint8_t* ftype, float* v, float* n, uint8_t* m)
{
	// Create the noise field.
	osino_mkfield(fdens);

	// Extract the surface.
	const float isoval = -0.87f;
	numtria = surface_extract
	(
		fdens,
		ftype,
		isoval,
		1,BLKRES-1,
		1,BLKRES-1,
		v,
		n,
		m,
		MAXTRIA,
		0
	);
	return numtria;
}



#ifdef STANDALONE

int main(int argc, char* argv[])
{
	tt_signin(-1,"main");
	fprintf(stderr, "Enabling floating point exceptions...\n");
	feenableexcept( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );
	const int numt = procgen_asteroid(fielddensity, fieldtype, surface_v, surface_n, surface_m);
	fprintf(stderr,"Generated %d triangles.\n", numt);
	const float off = -0.5f * (BLKRES-1);
	const float offset[3] = { off, off, off };
	surface_writeobj("output.obj", numt, surface_v, surface_n, offset);
	tt_report("procgen.json");
}

#endif
