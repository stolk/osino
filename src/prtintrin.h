#define PRTPS( A ) \
{ \
	float values [8] __attribute__ ((aligned(32))); \
	_mm256_store_ps( values, A ); \
	fprintf( stderr, #A " %f %f %f %f  %f %f %f %f\n", values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0] ); \
}

#define PRTI256( A ) \
{ \
	uint32_t values[8] __attribute__ ((aligned(32))); \
	_mm256_store_si256( (__m256i*) values, A ); \
	fprintf( stderr, #A " %x %x  %x %x  %x %x  %x %x\n", values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0] ); \
}
