CC=clang-8
CXX=clang++-8


ENOKIPREFIX=$(HOME)/src/enoki
TTPREFIX=$(HOME)/src/ThreadTracer

OSINOOBJS=\
	src/osino.o \
	src/surface.o \
	src/procgen.o

TTOBJS=\
	$(TTPREFIX)/threadtracer.o


CFLAGS=-O2 -mavx -mfma -g -MMD -I$(ENOKIPREFIX)/include -I$(TTPREFIX) -DSTANDALONE -D_GNU_SOURCE
CXXFLAGS=$(CFLAGS) -std=c++17

all: osino

osino: $(OSINOOBJS) $(TTOBJS)
	$(CXX) -oosino -L$(ENOKIPREFIX)/build $(OSINOOBJS) $(TTOBJS) -lenoki-cuda

#run: osino
#	./osino > im.pgm
#	-display im.pgm

-include $(OSINOOBJS:.o=.d)

