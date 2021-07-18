#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "hpc.h"
#define WARP_SIZE 32

__global__ void test_function(int *array, int length){
	int x = threadIdx.x;
	printf("x: %d\n", x);
	array[x] = array[x] + 1;
}

int main(){
	size_t size = WARP_SIZE * sizeof(int);
	int* array = (int*) malloc(size);
	int* d_array;
	
	for(int i = 0; i < WARP_SIZE; i++) {
		array[i] = i;
	}

	cudaSafeCall(cudaMalloc((void**)&d_array, size));
	int n_threads = 4;//omp_get_max_threads();
	printf("%d\n", n_threads);
#pragma omp parallel default(none) num_threads(n_threads) shared(array, d_array, n_threads)
	{
		int thread_id = omp_get_thread_num();
		int local_start = WARP_SIZE * thread_id / n_threads;
		int local_end = WARP_SIZE * (thread_id + 1) / n_threads;
		printf("start: %d, end: %d\n", local_start, local_end);
		cudaSafeCall(cudaMemcpy(&d_array[local_start], &array[local_start], local_end - local_start, cudaMemcpyHostToDevice)); 
	}

	test_function<<<1, WARP_SIZE>>>(d_array, WARP_SIZE);

	cudaSafeCall(cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost));
	for(int i = 0; i < WARP_SIZE; i++){
		printf("%d ", array[i]);
	}
	printf("\n");
	return 0;
}
