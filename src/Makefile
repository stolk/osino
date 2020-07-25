NVCC=/usr/local/cuda/bin/nvcc
NVPROF=/usr/local/cuda/bin/nvprof

NVCFLAGS=--ptxas-options=-v -keep --generate-line-info --source-in-ptx --use_fast_math -arch sm_50



#run: computefield
#	$(NVPROF) --device-buffer-size 160 ./computefield 

run: classifyfield computefield intersectedges
	$(NVPROF) --device-buffer-size 160 ./computefield 
	-display out_compute.pgm
	$(NVPROF) --device-buffer-size 160 ./classifyfield 
	-display out_classify.pgm
	$(NVPROF) --device-buffer-size 160 ./intersectedges 
	-display out_intersectedges.ppm


computefieldfp16: computefieldfp16.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=7 -USTORECHARS -DSTOREFP16 computefieldfp16.cu -o computefieldfp16

computefield: computefield.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=7 -USTORECHARS -DSTOREFP16 computefield.cu -o computefield

classifyfield: classifyfield.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=7 -USTORECHARS -DSTOREFP16 classifyfield.cu -o classifyfield

intersectedges: intersectedges.cu
	$(NVCC) $(NVCFLAGS) -DBLKMAG=7 -USTORECHARS -DSTOREFP16 intersectedges.cu -o intersectedges

testcase: testcase.cu
	$(NVCC) testcase.cu -o testcase

