#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define BLKDIM 32
#define MAX_DIM 200
#define SHARED_MEM_SIZE 49152
#define MAX_BLOCKS_PER_GRID 65536
#define MAX_THREADS_PER_BLOCK 1024

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
__device__ bool dominance(double *s, double *d, int dim){
    bool weakly_major = true;
    bool stricly_major = false;
    for(int i = 0; i < dim && weakly_major; i++){
        if(s[i] < d[i]){
			 weakly_major = false;
		}
        if(s[i] > d[i]){
			stricly_major = true;
		}
    }
    return weakly_major && stricly_major;
}

/* Kernel function */
__global__ void skyline(double *points, bool *S, int n, int d){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y < n){
		/* Copy the number in charge to the local memory in order
		   to not saturate with accesses the global memory
		 */
		extern __shared__ double num[];
		int off = d/blockDim.x * x;
		int dim = threadIdx.x == blockDim.x - 1 ? (d/blockDim.x + d % blockDim.x) : d/blockDim.x;
		memcpy(&num[threadIdx.y * d + off], &points[y * d + off], dim * sizeof(double)); 
		
		int *is_skyline_point = (int*) &num[blockDim.y * d];
		
		if(threadIdx.x == 0){
			 is_skyline_point[threadIdx.y] = 1;
		}		
 
		int offset = x * n/blockDim.x * d;
		int size = threadIdx.x == blockDim.x - 1 ? (n/blockDim.x + n % blockDim.x) : n/blockDim.x; 
		__syncthreads();
		/* Try to not use a shared variable */
		for(int i = 0; i < size && is_skyline_point[threadIdx.y]; i++){
			/* If num is dominates by another number then it is not
			   in the Skyline set
			*/
			if(i != y){
				if(dominance(&points[i * d + offset], &num[threadIdx.y * d], d)){
					atomicAnd(&is_skyline_point[threadIdx.y], 0);		 
				}
			}
		}
		/* Copy the results on the device global memory by
		   syncronizing threads to perform coalisced memory accesses 
		 */
		__syncthreads();
		if(threadIdx.x == 0) S[y] = is_skyline_point[threadIdx.y] == 1 ? true : false;
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
	bool *S, *d_S;
	cudaSafeCall(cudaMalloc((void**)&d_S, (*N) * sizeof(bool)));	
	S = (bool*) malloc((*N) * sizeof(bool));

	/* Block and Grid size
	 * 	Allocated shared memory = blockDim.y * (*D) * sizeof(double) + blockDim.y * sizeof(int)
	 *	then, 
	 *	blockDim.y * ((*D) * sizeof(double) + sizeof(int)) <= SHARED_MEM_SIZE
	 *	max_y_dim = floow(SHARED_MEM_SIZE / ((*D) * sizeof(double) + sizeof(int)))
	 * 	min_y_dim = ceil((*N) / MAX_BLOCKS_PER_GRID)  
	 */
	int max_y_dim = floor((double)SHARED_MEM_SIZE / ((*D) * sizeof(double) + sizeof(int)));
	int min_y_dim = ceil((double)(*N) / MAX_BLOCKS_PER_GRID);

	int x_dim;

	for(int r = min_y_dim; r <= max_y_dim; r *= 2){
		/* 2, 4, 8, ..., 32 * t <= max_y_dim where t = 1,... */
		int y_dim = r;
		x_dim = 32/y_dim >= 1 ? 32/y_dim : 1;
		
		for(int i = 1; x_dim * y_dim <= MAX_THREADS_PER_BLOCK; i++){	
			dim3 block(x_dim, y_dim);
			dim3 grid(1, (*N)/y_dim);
			double t_kernel_start = hpc_gettime();	
			/* Kernel function */
			size_t block_mem_size = y_dim * (*D) * sizeof(double) + y_dim * sizeof(int); 
			skyline<<<grid, block, block_mem_size>>>(d_points, d_S, *N, *D);
			cudaDeviceSynchronize();
			double t_kernel_end = hpc_gettime();
			printf("x dim: %d, y dim: %d, time: %lf\n", x_dim, y_dim, t_kernel_end - t_kernel_start);
			
			cudaCheckError();
			
			cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(bool), cudaMemcpyDeviceToHost));
			char filename[25] = "test7-v4-";
			char tmp[10] = "";
			sprintf(tmp, "%d", i);
			strcat(filename, tmp);
			FILE *fp = fopen(filename, "w");
			fprintf(fp, "%d\n%d\n", *D, *N);
			if(fp != NULL){
				for(int k = 0; k < *N; k++){
					if(S[k]){
						for(int j = 0; j < *D; j++){
							fprintf(fp, "%lf ", points[k * (*D) + j]);
						}
						fprintf(fp, "\n");
					}
				}
				fclose(fp);
			}
			x_dim = 32/y_dim >= 1 ? 32/y_dim * (i + 1) : (i + 1);
		}
	}
	double t_end = hpc_gettime();
	//printf("%lf\n", t_kernel_end - t_kernel_start);
	//printf("Time: %lf\n", t_end - t_start);
	/* Free memory on GPU device */
	cudaFree(d_points);
    return 0;
}
