#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 1024
#define BLKDIM 32

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
    double *matrix = (double*) malloc((*D) * (*N) * sizeof(double));
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
            matrix[k * (*N) + i] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return matrix;
}

/* Returns true if s dominates d */
bool dominance(double* s, double *d, int dim){
    bool weakly_major = true;
    bool stricly_major = false;
    for(int i = 0; i < dim && weakly_major; i++){
        if(s[i] < d[i]) weakly_major = false;
        if(s[i] > d[i]) stricly_major = true;
    }
    return weakly_major && stricly_major;
}

__global__ void skyline(double *points, bool *S, int n, int d){
	__shared__ double* p;
	memcpy(p, points, n * d * sizeof(double));
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < n && i < d){
		if(i == 0) S[j] = true; 
		__syncthreads();		

		double el = points[i * n + j];
		bool s_j = true;
		for(int k = 0; k < n && s_j; k++){
			if(j != k){
				if(points[i * n + k] > el){
					s_j = false;
					S[j] = s_j;
				}
			}
		}
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
	dim3 block(BLKDIM, BLKDIM);
	dim3 grid(((*N) + BLKDIM - 1)/BLKDIM, ((*D) + BLKDIM - 1)/BLKDIM);
	
	/* Kernel function */
	skyline<<<grid, block>>>(d_points, d_S, *N, *D);
	
	/* [TODO] */
	//cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(bool), cudaMemcpyDeviceToHost));
	S = (bool*) malloc((*N) * sizeof(bool)); 
	/*for(int i = 0; i < *N; i++){
		printf("%d\n", S[i]);
	}*/
	cudaCheckError();
	cudaDeviceSynchronize(); 
	printf("Kernel OK\n");
	//double *tmp = (double*) malloc(size);
	S = (bool*) malloc((*N) * sizeof(bool));
	cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(bool), cudaMemcpyDeviceToHost));
	for(int i = 0; i < *N; i++){
		if(S[i]) printf("%d ", S[i]);
	}
	printf("\n");
	/* Free memory on GPU device */
	cudaFree(d_points);
    return 0;
}
