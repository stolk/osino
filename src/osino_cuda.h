#ifdef __cplusplus
extern "C"
{
#endif

#if defined(STOREFP16)
typedef __fp16 value_t;
#elif defined(STORECHARS)
typedef unsigned char value_t;
#elif defined(STORESHORTS)
typedef signed short value_t;
#else
typedef float value_t;
#endif

#define NUMSTREAMS 3

extern void osino_client_init(void);
extern int  osino_client_computefield(int gridOff[3], int fullres, float offsets[3], float domainwarp, float freq, float lacunarity, float persistence);
extern void osino_client_stagefield(int slot);
extern void osino_client_collectfield(int slot, value_t* output);

#ifdef __cplusplus
}
#endif
