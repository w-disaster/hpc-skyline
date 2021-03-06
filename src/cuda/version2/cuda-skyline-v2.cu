#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000
#define BLKDIM 32
#define MAX_DIM 300

//#define _CG_ABI_EXPERIMENTAL
//using namespace cooperative_groups::experimental;

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

__device__ int is_subpoint_dominated(int* array, int length, int step){
    bool contains_zero = false;
    int strictly_minor = 0;
    
    for(int k = 0; k < length && !contains_zero; k+=step){
        int next = array[k];
        /* contains_zero = true, quando una coordinata del punto points[y * d]
        ?? maggiore di quella confrontata */
        contains_zero = (next == 0);
        strictly_minor += (next == 2); 
    }

    /* Se non c'?? una coordinata strettamente maggiore di quella confrontata e 
    c'?? n'?? almeno una strettamente minore allora il punto points[i * d] domina points[y * d] */ 
    if(strictly_minor > 0 && !contains_zero){
        return 2;
    } else if(strictly_minor == 0 && !contains_zero){
        return 1;
    } else{
        return 0;
    }
}

/* Kernel function */
__global__ void skyline(double *points, int *S, int n, int d){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    //cooperative_groups::thread_block my_block = cooperative_groups::this_thread_block();
    //cooperative_groups::thread_block_tile<256> my_tile = cooperative_groups::tiled_partition<256>(my_block);

    if(x < d && y < n){
        //auto g = cg::this_grid();
        //g.sync();

        __shared__ int compare[2][200]; 
        double el = points[y * d + x];
        //compare[threadIdx.y][threadIdx.x] = 0;
        int local_compare;
        double local_next; 
        int warp_id = threadIdx.x % 32;
        __shared__ bool is_dominated[2];
        if(threadIdx.x == 0) is_dominated[threadIdx.y] = false;
        
        __syncthreads();

        for(int i = 0; i < n; i++){
            if(i != y){
                local_next = points[i * d + x];
                local_compare = 0;
                local_compare += (el < local_next ? 2 : 0);
                local_compare += (el == local_next ? 1 : 0);
                compare[threadIdx.y][threadIdx.x] = local_compare;
                
                if(warp_id == 0) {
                    int length = (x == d - (d % 32) ? d % 32 : 32); 
                    int ris = is_subpoint_dominated(&compare[threadIdx.y][threadIdx.x], length, 1);
                    compare[threadIdx.y][threadIdx.x] = ris;  
                }            

                __syncthreads();

                if(threadIdx.x == 0){
                    switch(is_subpoint_dominated(compare[threadIdx.y], d, 32)){
                        case 0:
                            is_dominated[threadIdx.y] = false;
                            break;
						case 1:
                        case 2:
                            is_dominated[threadIdx.y] = true;
                            break;
                    }
                    if(is_dominated[threadIdx.y]){
                        S[y] = !is_dominated[threadIdx.y];
						break;
                    }
                }
            }
        }
        if(threadIdx.x == 0) S[y] = !is_dominated[threadIdx.y];
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
    dim3 block(*D, 2);
    dim3 grid(1, *N/2);
    double t_kernel_start = hpc_gettime();
    /* Kernel function */
    skyline<<<grid, block>>>(d_points, d_S, *N, *D);
    cudaDeviceSynchronize();
    double t_kernel_end = hpc_gettime();
    //cudaCheckError();

    S = (int*) malloc((*N) * sizeof(int));
    cudaSafeCall(cudaMemcpy(S, d_S, (*N) * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0; i < *N; i++){
        if(S[i]){
            for(int k = 0; k < *D; k++){
                printf("%lf ", points[i * (*D) + k]);
            }
            printf("\n");
        }
    }
    double t_end = hpc_gettime();
    printf("%lf\n", t_kernel_end - t_kernel_start);
    //printf("Time: %lf\n", t_end - t_start);
    /* Free memory on GPU device */
    cudaFree(d_points);
    return 0;
}
