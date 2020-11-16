NVCC=/usr/local/cuda/bin/nvcc
NVPROF=/usr/local/cuda/bin/nvprof

NVCFLAGS=--ptxas-options=-v -keep --generate-line-info --source-in-ptx --use_fast_math -arch sm_50

#STORAGEFLAGS=-USTORECHARS -DSTOREFP16 -USTORESHORTS
STORAGEFLAGS=-USTORECHARS -USTOREFP16 -DSTORESHORTS



#run: computefield
#	$(NVPROF) --device-buffer-size 160 ./computefield 

run: classifyfield computefield #intersectedges
	$(NVPROF) --device-buffer-size 160 ./computefield 
	-display out_compute.pgm
	$(NVPROF) --device-buffer-size 160 ./classifyfield 
	-display out_classify.pgm
	#$(NVPROF) --device-buffer-size 160 ./intersectedges 
	#-display out_intersectedges.ppm

runcm: computematter
	$(NVPROF) --device-buffer-size 160 ./computematter
	-display out_computematter.pgm


computefieldfp16: computefieldfp16.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=6 $(STORAGEFLAGS) computefieldfp16.cu -o computefieldfp16

computefield: computefield.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=6 $(STORAGEFLAGS) computefield.cu -o computefield

computematter: computematter.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=6 $(STORAGEFLAGS) computematter.cu -o computematter

classifyfield: classifyfield.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=6 $(STORAGEFLAGS) classifyfield.cu -o classifyfield

intersectedges: intersectedges.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=6 $(STORAGEFLAGS) intersectedges.cu -o intersectedges

doubledomainwarp: doubledomainwarp.cu Makefile
	$(NVCC) $(NVCFLAGS) -DIMMAG=10  doubledomainwarp.cu -o doubledomainwarp
	$(NVPROF) --device-buffer-size 160 ./doubledomainwarp
	-display out_doubledomainwarp.ppm


testcase: testcase.cu
	$(NVCC) testcase.cu -o testcase

clienttest: clienttest.c osino_cuda_client.c computefield.cu
	clang-10 -g -O0 -o clienttest -DSTORESHORTS -DBLKMAG=6 -D_GNU_SOURCE -I$(HOME)/src/ThreadTracer -I/usr/local/cuda/include -L/usr/local/cuda/lib64 clienttest.c osino_cuda_client.c $(HOME)/src/ThreadTracer/threadtracer.c -lcudart -lcuda
	$(NVPROF) --device-buffer-size 160 ./clienttest
	-gimp computed_matter.pgm


anim.mp4: anim/im0000.ppm Makefile
	ffmpeg -r 30 -i anim/im%04d.ppm -c:v mpeg4 -b:v 5M anim.mp4

