// perlin noise functions.


extern __m256 osino_avx_2d
(
	__m256 x,
	__m256 y
);


extern __m256 osino_avx_3d
(
	__m256 x,
	__m256 y,
	__m256 z
);


// 4-octave versions.


extern __m256 osino_avx_2d_4o
(
	__m256 x,
	__m256 y,
	float lacunarity,
	float persistence
);


extern __m256 osino_avx_3d_4o
(
	__m256 x, __m256 y, __m256 z,	// sample location.
	float lacunarity,		// gaps between frequencies for octaves.
	float persistence		// amplitude scaling between two successive octaves.
);
