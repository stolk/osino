// Marching cubes implementation
// by Bram Stolk.
// tables by Cory Bloyd.

// The resolution: number of samples per dimension.
#define BLKRES		(1<<BLKMAG)

// Total size of a 3D block.
#define BLKSIZ		(BLKRES*BLKRES*BLKRES)

#ifdef __cplusplus
extern "C" {
#endif

// Extract a surface from a 3D field using an iso value.
extern int surface_extract
(
	const float* __restrict__ fielddensity,		// density field.
	const uint8_t* __restrict__ fieldtype,		// materials.
	float isoval,					// iso value that separates volumes.
	float*  __restrict__ outputv,			// surface verts
	float*  __restrict__ outputn,			// surface normals
	uint8_t*  __restrict__ outputm,			// surface materials
	int maxtria					// maximum number of triangles.
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

