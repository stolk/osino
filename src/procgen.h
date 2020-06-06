// Sample use of the OSINO code:
// Procedurally generate an asteroid.
// Uses marching cubes to extract a surface.

#define MATTER_DUST	0	// Dust
#define MATTER_ROCK	1	// Rock
#define MATTER_FWAT	2	// Frozen water
#define MATTER_MRAL	3	// Mineral

#ifdef __cplusplus
extern "C" {
#endif

extern int procgen_asteroid
(
	float* fdens,		// 3d density field
	uint8_t* ftype,		// 3d matter type
	float* v,		// surface verts
	float* n,		// surface normal
	uint8_t* m		// surface materials
);

#ifdef __cplusplus
}
#endif

