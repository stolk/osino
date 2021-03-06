// Marching cubes implementation
// by Bram Stolk.
// tables by Cory Bloyd.

// The resolution: number of samples per dimension.
#define BLKRES		(1<<BLKMAG)

// Total size of a 3D block.
#define BLKSIZ		(BLKRES*BLKRES*BLKRES)

#if defined(STOREFP16)
typedef __fp16 value_t;	// __half is unknown, CPU-side, so we use __fp16
#elif defined(STORECHARS)
typedef unsigned char value_t;
#elif defined(STORESHORTS)
typedef short value_t;
#else
typedef float value_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Extract cases.
extern int surface_extract_cases
(
	const value_t* __restrict__ fielddensity,	// density field.
	const value_t* __restrict__ fieldtype,		// materials.
	const uint8_t* __restrict__ cases,
	value_t isoval,					// iso value that separates volumes.
	const int* __restrict__ gridoff,
	int xlo,					// x-range
	int xhi,
	int ylo,					// y-range
	int yhi,
	int zlo,
	int zhi,
	float* __restrict__ outputv,			// surface verts
	float* __restrict__ outputn,			// surface normals
	float* __restrict__ outputm,			// surface materials
	float* __restrict__ outputo,			// surface ore types
	int*   __restrict__ numsurfacecells,		// written with the affected cell count.
	int**  __restrict__ surfacecells,		// written with the affected cell addresses
	int maxtria,					// maximum number of triangles.
	int threadnr					// Use scratch pool 0,1,2 or 3.
);


// Reclassify a subsector of the field.
void osino_reclassifyfield
(
	value_t isoval,
	const value_t* field,
	uint8_t* cases,
	int xlo,
	int xhi,
	int ylo,
	int yhi,
	int zlo,
	int zhi
);


// Dump geometry to Wavefront OBJ file.
extern void surface_writeobj
(
	const char* fname,	// output file name, should have .obj suffix.
	int numtria,		// number of triangles to write.
	const float* verts,	// vertices.
	const float* norms,	// vertex normals.
	const float* offs	// x,y,z offset to apply to all vertices.
);


#ifdef __cplusplus
} // extern C
#endif

