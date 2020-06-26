#ifdef __cplusplus
extern "C"
{
#endif

extern float clut[256][3];

extern void clut_init(int palnr, int shft);

extern void clut_override(const uint8_t* v);


#ifdef __cplusplus
}
#endif
