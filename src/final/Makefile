CFLAGS=-std=c99 -Wall -Wpedantic -fopenmp
LDLIBS=-lgomp -lrt
VCFLAGS=--gpu-architecture compute_30 

ALL: openmp cuda	

openmp: omp-skyline
	gcc -o omp-skyline ${CFLAGS} ${LDLIBS} omp-skyline.c


cuda: cuda-skyline.cu
	nvcc -o cuda-skyline ${VCFLAGS} cuda-skyline.cu 

.PHONY: clean

clean:
	rm -f omp-skyline cuda-skyline *~ core
