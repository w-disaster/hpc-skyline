#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "lib/hpc.h"
#include <cuda_runtime.h>

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

    /* Allocate the matrix (D x N), where each line i contains the values
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

/* Returns true if s dominates d */
__device__ bool dominance(double *s, double *d, int n, int dim){
    bool weakly_major = true;
    bool stricly_major = false;
    for(int i = 0; i < dim && weakly_major; i++){
        if(s[i * n] < d[i]){
			 weakly_major = false;
		}
        if(s[i * n] > d[i]){
			stricly_major = true;
		}
    }
    return weakly_major && stricly_major;
}

__global__ void copy(double *points, int *S, int n, int d){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y < n){
                /* Copy the number in charge to the local memory in order
                   to perform coalesced memory accesses
                */
                double num[MAX_DIM];
                //memcpy(num, &points[y * d], d * sizeof(double));
                for(int i = 0; i < d; i++){
                        num[i] = points[i * n + y];
                }
		}	
}


/* Kernel function */
__global__ void skyline(double *points, int *S, int n, int d){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y < n){
                /* Copy the number in charge to the local memory in order
                   to perform coalesced memory accesses
                */
                double num[MAX_DIM];
                //memcpy(num, &points[y * d], d * sizeof(double));
                for(int i = 0; i < d; i++){
                         num[i] = points[i * n + y];
                }
			
                int is_skyline_point = 1;
                for(int i = 0; i < n && is_skyline_point; i++){
                        /* If num is dominates by another number then it is not
                           in the Skyline set
                        */
                        if(i != y){
                                if(dominance(&points[i], num, n, d)){
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
	double avg[1024/32] = {0};
	int num_blocks;

	int device;
	int maxActiveBlocks;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	for(int k = 0; k < 10; k++){
		for(int i = 1; i <= 32; i++){
			int y_dim = i * 32;
			dim3 block(1, y_dim);
			//printf("%d\n", ((*N) + y_dim - 1)/y_dim);
			dim3 grid(1, ((*N) + y_dim - 1)/y_dim);
			
			/* - Kernel function call to determine the Skyline set
			- Wait it completion to calculate the time spent 
			- Check for errors occurred in the GPU
			 */
		
			double t_kernel_start = hpc_gettime();	
			skyline<<<grid, block>>>(d_points, d_S, *N, *D);
			cudaDeviceSynchronize();
			double t_kernel_end = hpc_gettime();

			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, skyline, y_dim, 0);
			float occupancy = (maxActiveBlocks * y_dim / props.warpSize) / 
				(float)(props.maxThreadsPerMultiProcessor / 
						props.warpSize);
			//printf("y_dim: %d, occ: %f\n", y_dim, occupancy);
			
			avg[i - 1] += t_kernel_end - t_kernel_start;
			//printf("%d %lf\n", y_dim, t_kernel_end - t_kernel_start);
			cudaCheckError();	
		}
	}

	for(int i = 1; i <= 32; i++){
		avg[i - 1] = avg[i - 1] / 10;
		printf("%d %lf\n", i * 32, avg[i - 1]);
	}
		
	/* - Allocate space on the host memory to store the result
	   - Copy the result from device memory to host's
	   - Print the points in the Skyline set 
	*/
	S = (int*) malloc((*N) * sizeof(int));
	cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(int), cudaMemcpyDeviceToHost));
/*	for(int i = 0; i < *N; i++){
		if(S[i]){
			for(int k = 0; k < *D; k++){
				printf("%lf ", points[k * (*N) + i]);
			}
			printf("\n");
		}
	}
*/
	/* Print the time spent by the kernel to determine the Skyline set */
	//printf("%lf\n", t_kernel_end - t_kernel_start);
	
	/* Free space on device and host heap memory */
	cudaFree(d_points);
	free(points);
	free(S);
	free(D);
	free(N);
    return 0;
}
