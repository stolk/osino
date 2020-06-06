# osino - An Open Simplex Noise implementation for Enoki/CUDA.

This is an 
[Enoki](https://github.com/mitsuba-renderer/enoki)-port
of the [OpenSimplex.java code by Stefan Gustavson](http://webstaff.itn.liu.se/~stegu/simplexnoise/).

It is derived from my C-port [sino](https://github.com/stolk/sino) but improves in several ways:

* Uses Enoki and [CUDA](https://developer.nvidia.com/cuda-downloads) to run on the GPU.
* Avoids memory gather by having purely procedural pseudo random directions from [Murmur2](https://en.wikipedia.org/wiki/MurmurHash) hashing.
* More varied noise due to using pseudo random directions instead of using 12 cardinal directions that are shuffled.

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



