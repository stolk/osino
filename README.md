# osino - An Open Simplex Noise implementation for Enoki/CUDA.

This is an 
[Enoki](https://github.com/mitsuba-renderer/enoki)-port
of the [OpenSimplex.java code by Stefan Gustavson](http://webstaff.itn.liu.se/~stegu/simplexnoise/).

It is derived from my C-port [sino](https://github.com/stolk/sino) but improves in several ways:

* Uses Enoki and [CUDA](https://developer.nvidia.com/cuda-downloads) to run on the GPU.
* Avoids memory gather by having purely procedural pseudo random directions from [Murmur2](https://en.wikipedia.org/wiki/MurmurHash) hashing.
* More varied noise due to using pseudo random directions instead of using 12 cardinal directions that are shuffled.

It comes with a Marching Cubes implementation to extract a surface, so that a 3D density field can be visualized.

![Procgen Asteroid](images/asteroid.png "Procgen Asteroid")

## License
3-clause BSD

## Dependencies
* Enoki
* CUDA
* ThreadTracer

## Building

Install cuda development environment. I used /usr/local/cuda for the destination.

Get the source and dependencies, recursively:

```
$ git clone --recursive git@github.com:stolk/osino.git
```

Build enoki:

```
$ cd externals/enoki
$ mkdir build
$ cd build
$ CXX=clang++-8 CC=clang-8 cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Debug -DENOKI_CUDA=ON ..
$ make
$ cd ../../..
```

Before building osino, edit the Makefile to set your compiler.
Then use `make` to build the example.

Test osino:
```
$ make output.obj
```

## API

The sample coordinates and noise values are contained in Enoki dynamic arrays with this signature:
```
typedef enoki::CUDAArray<float>    FV;  // Flt vector
```
To get noise values for a specified set of coordinates, use:
```
FV osino_2d(FV x, FV y);
FV osino_3d(FV x, FV y, FV z);
```
Or if you want multi octave noise (also called fractal noise) then use:
```
FV osino_2d_4o(FV x, FV y);
FV osino_3d_4o(FV x, FV y, FV z);
```
Note that to make it worthwhile to have a round trip to the GPU, you need to compute a lot of values in one go. Which means millions of noise values, not thousands. Otherwise the communication overhead would defeat the purpose of doing this GPU-side. In that case, you are better off using the AVX backend of Enoki.


