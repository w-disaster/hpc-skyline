#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define WARP_SIZE 32
#define MAX_DIM 200

/* This function reads the points from a file descriptor and saves
 * them in the return matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 */
double* build_matrix(FILE* fd, int* N, int* D){
    char line[LINE_LENGHT];
    const size_t BUF_SIZE = sizeof(line);
	    
    /* Read the dimension: fetch the first line until space */
    char* dim;
    dim = fgets(line, BUF_SIZE, fd);
    sscanf(dim, "%d", D);
    printf("%d\n", *D);
    
    /* Read the number of points: fetch the second line until newline */
    char* n;
    n = fgets(line, BUF_SIZE, fd);
    sscanf(n, "%d", N);
    printf("%d\n", *N);

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
            /* convert ASCII string to floating-point number */
            matrix[k * (*N) + i] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return matrix;
}

/* Returns true if the array s dominates the array d. 
 * Parameters:
 * - s, d: arrays of double
 * - length: number of elements of s and d
 * - offset: distance between two elements that we must read in array s
 */
__device__ bool dominance(double *s, double *d, int length, int offset){
    bool strictly_minor = false;
    bool strictly_major = false;
    for(int i = 0; i < length && !strictly_minor; i++){
        if(s[i * offset] < d[i]){
			 strictly_minor = true;
		}
        if(s[i * offset] > d[i]){
			strictly_major = true;
		}
    }
    return !strictly_minor && strictly_major;
}

/* Kernel function:
 * each thread has the purpose to determine if the number in charge is in
 * the Skyline set. To do so, this function iterates on all the points and stops
 * if any of them dominates it.
 * The result, in the end, is put in the array S, stored in the global memory. 
 */
__global__ void skyline(double *points, int *S, int n, int d){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y < n){
		/* Copy the number in charge to the local memory in order
		   to perform coalesced memory accesses
		*/
		double num[MAX_DIM];
		for(int i = 0; i < d; i++){
			num[i] = points[i * n + y];
		}

		int is_skyline_point = 1;
		for(int i = 0; i < n && is_skyline_point; i++){
			/* If num is dominates by another number then it is not
			   in the Skyline set
			*/
			if(i != y){
				if(dominance(&points[i], num, d, n)){
					is_skyline_point = 0;						 
				}
			}
		}
		/* Copy the results on the device global memory */
		S[y] = is_skyline_point;
	}
}

int main(int argc, char* argv[]){
   	/* Allocate memory to store the number of points, them dimension and the points */
	int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));
    double* points = build_matrix(stdin, N, D);

	/* - Define the matrix dimension, 
	   - Allocate space on the device global memory 
	   - Copy the array points on the allocated space
	 */
	const size_t size = (*N) * (*D) * sizeof(double);
    double* d_points;
	cudaSafeCall(cudaMalloc((void**)&d_points, size));
	cudaSafeCall(cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice));

	/* Allocate space where the kernel function will store the result */
	int *S, *d_S;
	cudaSafeCall(cudaMalloc((void**)&d_S, (*N) * sizeof(int)));	

	/* Define the block and grid dimensions */
	dim3 block(1, WARP_SIZE);
	dim3 grid(1, ((*N) + WARP_SIZE - 1)/WARP_SIZE);
		
	/* - Kernel function call to determine the Skyline set
	   - Calculate the time spent 
	   - Check for errors occurred in the GPU
	*/
	cudaEvent_t t_kernel_start, t_kernel_stop;
	cudaEventCreate(&t_kernel_start);
	cudaEventCreate(&t_kernel_stop);	
	
	cudaEventRecord(t_kernel_start);
	skyline<<<grid, block>>>(d_points, d_S, *N, *D);
	cudaEventRecord(t_kernel_stop);
	cudaCheckError();	
	
	/* - Allocate space on heap (host memory) in order to store the result
	   - Copy the result from device memory to host's
	   - Print the points in the Skyline set 
	*/
	S = (int*) malloc((*N) * sizeof(int));
	cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(int), cudaMemcpyDeviceToHost));
	for(int i = 0; i < *N; i++){
		if(S[i]){
			for(int k = 0; k < *D; k++){
				printf("%lf ", points[k * (*N) + i]);
			}
			printf("\n");
		}
	}

	/* Print the time spent  by the kernel to determine the Skyline set */
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, t_kernel_start, t_kernel_stop);	
	printf("%f\n", milliseconds / 1000);
	
	/* Free space on device and host heap memory */
	cudaFree(d_points);
	free(points);
	free(S);
	free(D);
	free(N);
    return 0;
}
