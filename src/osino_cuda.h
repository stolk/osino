#ifdef __cplusplus
extern "C"
{
#endif

#define NUMSTREAMS 3

extern void osino_client_init(void);
extern int  osino_client_computefield(int gridOff[3], int fullres, float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);
extern void osino_client_stagefield(int slot);
extern void osino_client_collectfield(int slot, float* output);

#ifdef __cplusplus
}
#endif
