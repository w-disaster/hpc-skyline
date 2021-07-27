EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE:=$(EXE_OMP) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600
LDLIBS+=-lm
NVCC:=nvcc
NVCFLAGS+=-Wno-deprecated-gpu-targets
NVLDLIBS+=-lm

.PHONY: clean

ALL: $(EXE)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@ $(NVLDLIBS)

$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

cuda: $(EXE_CUDA)

clean:
	\rm -f $(EXE) *.o *~
