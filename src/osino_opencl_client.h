// osino_opencl_client.h
//
// Open SImplex NOise (OpenCL version.)
// (c)2020 Game Studio Abraham Stolk Inc.
//

#ifdef __cplusplus
extern "C"
{
#endif

extern int  osino_opcl_client_init( void );
extern void osino_opcl_client_exit( void );
extern void osino_opcl_client_sync( int rq );
extern void osino_opcl_client_release( int rq );
extern int  osino_opcl_client_usage( void );
extern int  osino_opcl_num_platforms( void );

//! Returns a request id.
extern int  osino_opcl_client_computefield (int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Returns a request id.
extern int  osino_opcl_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

extern void osino_opcl_client_collectfield(int slot, value_t* output);

extern void osino_opcl_client_collectcases(int slot, uint8_t* output);

extern void osino_opcl_client_classifyfield(int slot, value_t isoval);

extern void osino_opcl_client_stagefield(int slot);

extern void osino_opcl_client_stagecases(int slot);

#ifdef __cplusplus
}
#endif
