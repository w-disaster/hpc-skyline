/****************************************************************************
 *
 * cuda-skyline.cu - Skyline set computation with CUDA
 *
 * Author: Fabri Luca 
 * Serial Number: 0000892878
 * Email: luca.fabri@studio.unibo.it
 * 
 * ---------------------------------------------------------------------------
 * 
 * Skyline set computation with CUDA. 
 * Given P a set of N points with dimension D, p1, p2 two points in P,
 * we say that p1 dominates p2 if:
 * - for each dimension k: p1[k] >= p2[k] , 0 <= k < D;
 * - exists at least one dimension j such that: p1[j] > p2[j] , 0 <= j < D.
 *
 * The Skyline set is, for definition, composed of all points in P that aren't
 * dominated by other points in P.
 *
 * Compile with:
 * nvcc -o cuda-skyline --gpu-architecture compute_30 cuda-skyline.cu
 * Or from Makefile:
 * make cuda
 *
 * Run with:
 * ./cuda-skyline < input_file > output_file
 *
 * Please not that the input_file provided as argument must contains:
 * - The dimension of the points, in the first line. Next chars, if present, 
 *   are ignored;
 * - The number of points in the 2nd line;
 * - The input set P whose points are in separed rows and each dimension value 
 *   separated by space or tab.
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define WARP_SIZE 32

/* 
 * This function reads the points from a file descriptor and saves
 * them into an array. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 * 
 * Parameters:
 * - fd: file descriptor
 * - N: pointer to integer where this function stores the number of points read
 * - D: pointer to int where this function stores the dimension of the points.
 * 
 * It returns the pointer to the allocated array containing the points.
 */

double* read_points(FILE* fd, int* N, int* D){
    char line[LINE_LENGHT];
    const size_t BUF_SIZE = sizeof(line);
	    
    /* Read the dimension: fetch the first line until space */
    char* dim;
    dim = fgets(line, BUF_SIZE, fd);
    sscanf(dim, "%d", D);
    
    /* Read the number of points: fetch the second line until newline */
    char* n;
    n = fgets(line, BUF_SIZE, fd);
    sscanf(n, "%d", N);
    
    /* Allocate the matrix (N x D), where each line i contains the values
	   of the points on that dimension i.
	*/
    double *matrix = (double*) malloc((*N) * (*D) * sizeof(double));
	
    char* str;
    const char* s = " ";
    char* token;
    char* ptr;
    for(int i = 0; i < *N; i++){
        /* Read current line */
        str = fgets(line, BUF_SIZE, fd);
        /* Split the string read on s=" " separator and fetch the values */
        token = strtok(str, s);
        for(int k = 0; k < *D && token != NULL; k++){
            /* convert ASCII string to doubleing-point number */
            matrix[k * (*N) + i] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return matrix;
}

/* Returns true if the array s dominates the array d. 
 * Parameters:
 * - s, d: arrays of double
 * - dim: number of elements of s and d
 * - offset: distance between two elements that we must read in array s, d
 */
__device__ bool dominance(double *s, double *d, int dim, int offset){
    bool strictly_major = false;
    /* Iterate over each index: 
     * if s[i] < d[i] then s doesn't dominate d --> return */
    for(int i = 0; i < dim; i++){
        if(s[i * offset] < d[i * offset]){
			 return false;
		}
        if(s[i * offset] > d[i * offset]){
			strictly_major = true;
		}
    }
    /* If there aren't elements strictly minor and exist at least on element
     * strictly major then s dominates d
     */
    return strictly_major;
}

/* Kernel function:
 * each thread has the purpose to determine if the number in charge is in
 * the Skyline set. To do so, this function iterates on all the points and stops
 * if any of them dominates it.
 * The result, in the end, is put in the array S, stored in the global memory. 
 */
__global__ void compute_skyline(double *points, bool *S, int *k, int n, int d){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y < n){
		bool is_skyline_point = true;
		for(int i = 0; i < n && is_skyline_point; i++){
			/* If num is dominates by another number then it is not
			   in the Skyline set
			*/
			if(i != y){
				if(dominance(&points[i], &points[y], d, n)){
					is_skyline_point = false;						 
				}
			}
		}
		/* Copy the results on the device global memory */
		S[y] = is_skyline_point;
        /* Increment by 1 the cardinality of Skyline set if the number 
           is in the Skyline set, by 0 otherwise 
         */
        atomicAdd(k, is_skyline_point); 
	}
}

/*
 * This function prints to the file descriptor fd given as parameter:
 * - The dimension D of the points;
 * - The cardinality K of the Skyline set;
 * - The Skyline set.
 */
__host__ void print_skyline(FILE* fd, bool *S, double *points, int N, int D, int K){
    int i, j;
    /* Print D, K */
    fprintf(fd, "%d\n%d\n", D, K);

    /* Print the Skyline set */
    for(i = 0; i < N; i++){
        if(S[i]){
            for(j = 0; j < D; j++){
                fprintf(fd, "%f ", points[j * N + i]);
            }
            fprintf(fd, "\n");
        }
    }
}

int main(void){
/* Allocate memory to store the number of points, them dimension and the points */
	int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));

	double* points = read_points(stdin, N, D);
   
	/* - Define the matrix dimension, 
	   - Allocate space on the device global memory 
	   - Copy the array points on the allocated space
	 */
	const size_t size = (*N) * (*D) * sizeof(double);
    double* d_points;
	cudaSafeCall(cudaMalloc((void**)&d_points, size));
	cudaSafeCall(cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice));

	/* Allocate space where the kernel function will store the result */
	bool *S, *d_S;
	cudaSafeCall(cudaMalloc((void**)&d_S, (*N) * sizeof(bool)));

	/* Allocate space in order to store the cardinality of the Skyline set */
    int *K, *d_K;
    K = (int*) malloc(sizeof(int));
    *K = 0;
    cudaSafeCall(cudaMalloc((void**)&d_K, sizeof(int)));    
    cudaSafeCall(cudaMemcpy(d_K, K, sizeof(int), cudaMemcpyHostToDevice));

	/* Define the block and grid dimensions */
	dim3 block(1, WARP_SIZE * 2);
	dim3 grid(1, ((*N) + WARP_SIZE * 2 - 1)/(WARP_SIZE * 2));
	
	cudaEvent_t t_kernel_start, t_kernel_stop;
	cudaEventCreate(&t_kernel_start);
	cudaEventCreate(&t_kernel_stop);	

	cudaEventRecord(t_kernel_start);
	
	/* Kernel function call to determine the Skyline set */
	compute_skyline<<<grid, block>>>(d_points, d_S, d_K, *N, *D);
	
	cudaEventRecord(t_kernel_stop);

	/* Wait the Kernel to finish and check errors */
	cudaCheckError();	

    /* While Kernel function is executing on device, allocate memory on heap 
	 * in order to store the result 
     */
	S = (bool*) malloc((*N) * sizeof(bool));
	
	/* - Copy the result from device memory to host's
       - Copy the Skyline cardinality from device to host memory
	   - Print the points in the Skyline set 
	*/
	cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(bool), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(K, d_K, sizeof(int), cudaMemcpyDeviceToHost));
    print_skyline(stdout, S, points, *N, *D, *K);

	/* Free space on device and host heap memory */
	cudaFree(d_points);
    cudaFree(d_K);
	free(points);
	free(S);
	free(D);
	free(N);
    free(K);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, t_kernel_start, t_kernel_stop);	
	fprintf(stdout, "%f\n", milliseconds / 1000);   
	return 0;
}

