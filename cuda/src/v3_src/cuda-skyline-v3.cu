#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define BLKDIM 32
#define MAX_DIM 300

/* This function reads the points from a file descriptor and saves
 * them in the return matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 */
double* build_matrix(FILE* fd, int* N, int* D){
    char line[LINE_LENGHT];
    const size_t BUFSIZE = sizeof(line);
      
    /* Read the dimension: fetch the first line until space */
    char* dim;
    dim = fgets(line, BUFSIZE, fd);
    sscanf(dim, "%d", D);
    printf("%d\n", *D);
    
    /* Read the number of points: fetch the second line until newline */
    char* n;
    n = fgets(line, BUFSIZE, fd);
    sscanf(n, "%d", N);
    printf("%d\n", *N);

    /* Allocate the matrix (D x N), where each line i contains the values
     of the points on that dimension i.
    */
    double *matrix = (double*) malloc((*N) * (*D) * sizeof(double));
    //for(int i = 0; i < (*D); i++) matrix[i] = (double *) malloc((*N) * sizeof(double));

    char* str;
    const char* s = " ";
    char* token;
    char* ptr;
    for(int i = 0; i < *N; i++){
        /* Read current line */
        str = fgets(line, BUFSIZE, fd);
        /* Split the string read on s=" " separator and fetch the values */
        token = strtok(str, s);
        for(int k = 0; k < *D && token != NULL; k++){
            /* convert ASCII string to floating-point number */
            matrix[i * (*D) + k] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return matrix;
}

/* Returns true if s dominates d */
__device__ int dominance(double *s, double *d, int dim){
    bool strictly_minor = false;
    bool strictly_major = false;
    for(int i = 0; i < dim && !strictly_minor; i++){
        if(s[i] < d[i]){
            strictly_minor = true;
        }
        if(s[i] > d[i]){
            strictly_major = true;
        }
    }

	if(strictly_minor){
		return 0;
	}else if(!strictly_minor && strictly_major){
		return 2;
	}else{
		return 1;
	}
}

/* Kernel function */
__global__ void skyline(double *points, int *S, int n, int d){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
   	if(x < d && y < n){
		extern __shared__ int compare[];
		bool is_dominated = false;
		int size = (threadIdx.x == blockDim.x - 1 ? (d / blockDim.x + d % blockDim.x) : (d / blockDim.x)); 		
		double local_array[200];
		int x_offset = (d / blockDim.x) * threadIdx.x;
		memcpy(local_array, &points[y * d + x_offset], size * sizeof(double));
		//for(int i = 0; i < 2; i++){
		//	local_array[i] = points[y * d + x_offset + i];
		//}

		for(int i = 0; i < n; i++){
			if(i != y){
				int ris = dominance(&points[i * d + x_offset], &local_array[0], size);
				compare[threadIdx.y * blockDim.x + threadIdx.x] = ris;
				
				__syncthreads();
				
				if(threadIdx.x == 0){
					int el;
					bool strictly_minor = false;
					bool strictly_major = false;
					for(int k = 0; k < blockDim.x && !strictly_minor; k++){
						el = compare[threadIdx.y * blockDim.x + k]; 
						if(el == 0){
							strictly_minor = true;
						} else if (el == 2){
							strictly_major = true;
						}
					}
					
					if(!strictly_minor && strictly_major){
						is_dominated = true;
						break;
					}
				}
			} 
		}
		if(threadIdx.x == 0) S[y] = !is_dominated;	
	}
}

int main(int argc, char* argv[]){
    double t_start = hpc_gettime();

    int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));
    double* points = build_matrix(stdin, N, D);
    const size_t size = (*N) * (*D) * sizeof(double);

    /* Allocate memory on the device */
    double* d_points;
    cudaSafeCall(cudaMalloc((void**)&d_points, size));
    /* Copy points to d_points on GPU memory */
    cudaSafeCall(cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice));

    //bool* d_array;
    //cudaSafeCall(cudaMalloc((void**)&d_array, (*N) * (*D) * (*N) * sizeof(bool)));      

    /* Result array */
    int *S, *d_S;
    cudaSafeCall(cudaMalloc((void**)&d_S, (*N) * sizeof(int)));

    /* Block and Grid size */
	int block_x_dim = 10;//(int) sqrt(*D);
	dim3 block(block_x_dim, BLKDIM);
    dim3 grid(1, ((*N) + BLKDIM - 1)/BLKDIM);
	//dim3 block(block_x_dim, 1);    
	//dim3 grid(1, *N/2);

/*	double t_kernel_start = hpc_gettime();
    // Kernel function
	size_t x_arr_dim = BLKDIM * block_x_dim * sizeof(int);
	skyline<<<grid, block, x_arr_dim>>>(d_points, d_S, *N, *D);

	cudaDeviceSynchronize();
    double t_kernel_end = hpc_gettime();
    cudaCheckError();
*/
	size_t block_arr_dim;
	double t_i_start;
	for(int i = 1; i <= BLKDIM; i++){
		block_arr_dim = BLKDIM * i * sizeof(int);
		dim3 block(i, BLKDIM);
		t_i_start = hpc_gettime();
		skyline<<<grid, block, block_arr_dim>>>(d_points, d_S, *N, *D);
		cudaDeviceSynchronize();
		printf("i: %d, time: %lf\n", i, hpc_gettime() - t_i_start);
		cudaCheckError();
	}

    S = (int*) malloc((*N) * sizeof(int));
    cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(int), cudaMemcpyDeviceToHost));
    /*for(int i = 0; i < *N; i++){
        if(S[i]){
            for(int k = 0; k < *D; k++){
                printf("%lf ", points[i * (*D) + k]);
            }
            printf("\n");
        }
    }*/
    double t_end = hpc_gettime();
    //printf("%lf\n", t_kernel_end - t_kernel_start);
    //printf("Time: %lf\n", t_end - t_start);
    /* Free memory on GPU device */
    cudaFree(d_points);
    return 0;
}
