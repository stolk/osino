// osino_opencl_client.h
//
// Open SImplex NOise (OpenCL version.)
// (c)2020 Game Studio Abraham Stolk Inc.
//

#ifdef __cplusplus
extern "C"
{
#endif

extern int  osino_opencl_client_init( void );
extern void osino_opencl_client_exit( void );
extern void osino_opencl_client_release( int rq );

extern void osino_opencl_client_test( void );

//! Returns a request id.
extern int  osino_opencl_client_computefield (int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

//! Returns a request id.
extern int  osino_opencl_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);

extern void osino_opencl_client_collectfield(int slot, value_t* output);

extern void osino_opencl_client_collectcases(int slot, uint8_t* output);

#ifdef __cplusplus
}
#endif
