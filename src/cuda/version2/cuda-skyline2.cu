#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define BLKDIM 32
#define MAX_DIM 300

/* This function reads the points from a file descriptor and saves
 * them in the return matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 */
double* buildMatrix(FILE* fd, int* N, int* D){
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
	double d;
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
        if(s[i] < d[i]) weakly_major = false;
        if(s[i] > d[i]) stricly_major = true;
    }
    return weakly_major && stricly_major;
}

__global__ void skyline(double *points, bool *S, int n, int d){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//__shared__ double local_points[MAX_DIM * BLKDIM];
	//memcpy(local_points, &points[i * d], MAX_DIM * BLKDIM);
	__shared__ bool ris[BLKDIM];	

	if(y < n){
		//if(y == 0) S[y] = true;
		ris[threadIdx.y] = true;
		
		//__syncthreads();		
		/* point to check if in skyline set */
		double num[MAX_DIM];
		memcpy(num, &points[y * d], d * sizeof(double)); 
		//bool flag = true;
		for(int i = 0; i < n && ris[threadIdx.y] /*flag*/; i++){
			//printf("%d\n", i);
			if(i != y && dominance(&points[i * d], num, d)){
				//flag = false;
				ris[threadIdx.y] = false;  						 
			}
		}
		//S[y] = flag;
		/* Copy the results on the device global memory */
		if(threadIdx.y == 0) memcpy(&S[blockIdx.y * BLKDIM], ris, BLKDIM * sizeof(bool)); 
	}
}

int main(int argc, char* argv[]){
    int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));
    double* points = buildMatrix(stdin, N, D);
	const size_t size = (*N) * (*D) * sizeof(double);
	
    /* Allocate memory on the device */
    double* d_points;
	cudaSafeCall(cudaMalloc((void**)&d_points, size));
	/* Copy points to d_points on GPU memory */
	cudaSafeCall(cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice));
	  
	/* Result array */
	bool *S, *d_S;
	cudaSafeCall(cudaMalloc((void**)&d_S, (*N) * sizeof(bool)));	

	/* Block and Grid size */
	dim3 block(1, BLKDIM);
	dim3 grid(1, ((*N) + BLKDIM - 1)/BLKDIM);
	
	double t_start = hpc_gettime();	
	/* Kernel function */
	skyline<<<grid, block>>>(d_points, d_S, *N, *D);
	cudaDeviceSynchronize();
	double t_end = hpc_gettime();

	cudaCheckError();	
	//double *tmp = (double*) malloc(size);
	S = (bool*) malloc((*N) * sizeof(bool));
	cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(bool), cudaMemcpyDeviceToHost));
	for(int i = 0; i < *N; i++){
		if(S[i]){
			for(int k = 0; k < *D; k++){
				printf("%lf ", points[i * (*D) + k]);
			}
			printf("\n");
		}
	}
	printf("Time: %lf\n", t_end - t_start);
	/* Free memory on GPU device */
	cudaFree(d_points);
    return 0;
}
