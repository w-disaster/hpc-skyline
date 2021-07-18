#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000

/* This function reads the points from a file descriptor and saves
 * them into a matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 
 */
double** read_points(FILE* fd, int* N, int* D){
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

    /* Allocate the matrix */
    long double **matrix = (long double**) malloc((*N) * sizeof(long double*));
    for(int i = 0; i < (*N); i++) matrix[i] = (long double *) malloc((*D) * sizeof(long double));

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
            matrix[i][k] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return matrix;
}

bool dominance(double* s, double *d, int dim){
    bool strictly_minor = false;
    bool strictly_major = false;
    for(int i = 0; i < dim && !strictly_minor; i++){
        if(s[i] < d[i]){
			strictly_minor = true;
		}
        if(s[i] > d[i]){
			striclty_major = true;
		}
    }
    return !strictly_minor && strictly_major;
}

bool* compute_skyline(long double** matrix, int rows, int cols){
    bool* S = (bool*) malloc(rows * sizeof(bool)); 
    int n_threads = omp_get_max_threads();
    printf("threads: %d\n", n_threads);
    int i, j;
    
#pragma omp parallel default(none) num_threads(n_threads) private(i, j) shared(S, rows, cols, matrix, n_threads)
    {    
        int thread_id = omp_get_thread_num();
        int local_start = rows * thread_id / n_threads;
        int local_end = rows * (thread_id + 1) / n_threads;
        for(i = local_start; i < local_end; i++) S[i] = true;
#pragma omp barrier    
        
        for(i = local_start; i < local_end; i++){
            if(S[i]){
                for(j = 0; j < rows; j++){
                    if(S[j] && dominance(matrix[i], matrix[j], cols)){
//#pragma omp critical
                        S[j] = false;
                    }
                } 
            }
        }
    }
    return S;
}

int main(int argc, char* argv[]){
    int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));
    double** skyline_matrix = build_matrix(stdin, N, D);
    /*for(int i = 0; i < *N; i++){
        for(int k = 0; k < *D; k++){
            printf("%Lf ", skyline_matrix[i][k]);
        }
        printf("\n");
    }*/

	for(i = 0; i < rows; i++){
        if(S[i]){
            for(j = 0; j < cols; j++){
                printf("%Lf ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    double tstart = omp_get_wtime();
    bool* skyline = compute_skyline(skyline_matrix, *N, *D);
    printf("Time: %lf\n", omp_get_wtime() - tstart);
    return 0;
}
