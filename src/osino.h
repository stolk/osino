// This is the Open Simplex Noise algorithm, ported to Enoki/CUDA.
//
// Original Open Simplex Algorithm: Stefan Gustavson (public domain.)
// http://webstaff.itn.liu.se/~stegu/simplexnoise
//
// C port by Bram Stolk
// https://github.com/stolk/sino
//
// Port to Enoki by Bram Stolk
// https://github.com/stolk/osino
//
// License: 3-clause BSD to match Enoki License.

#ifdef __cplusplus
extern "C" {
#endif

extern void osino_mkfield(float* volume);

#ifdef __cplusplus
}
#endif

