CC=clang-8
CXX=clang++-8


ENOKIPREFIX=externals/enoki

TTPREFIX=externals/ThreadTracer


# Open Simplex Noise object files.
OSINOOBJS=\
	src/osino.o \
	src/surface.o \
	src/procgen.o


# ThreadTracer object files.
TTOBJS=\
	$(TTPREFIX)/threadtracer.o


CFLAGS=-O2 -mavx2 -mfma -g -MMD -I$(ENOKIPREFIX)/include -I$(TTPREFIX) -DSTANDALONE -D_GNU_SOURCE -DBLKMAG=7
CXXFLAGS=$(CFLAGS) -std=c++17

all: osino

osino: $(OSINOOBJS) $(TTOBJS)
	$(CXX) -oosino -L$(ENOKIPREFIX)/build $(OSINOOBJS) $(TTOBJS) -lenoki-cuda

#run: osino
#	./osino > im.pgm
#	-display im.pgm

output.obj: osino
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):externals/enoki/build ./osino

clean:
	rm -f output.obj
	rm -f *.a
	rm -f $(TTOBJS)
	rm -f $(OSINOOBJS)

-include $(OSINOOBJS:.o=.d)

