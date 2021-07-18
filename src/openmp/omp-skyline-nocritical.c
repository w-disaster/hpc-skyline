#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000

/* This function reads the points from a file descriptor and saves
 * them into a matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 * Parameters:
 * - fd: file descriptor
 * - N: pointer to integer where this function stores the number of points read
 * - D: pointer to int where this function stores the dimension of the points.
 * It returns the double pointer to the allocated matrix containing the points.
 */
double **read_points(FILE *fd, int *N, int *D){
    char line[LINE_LENGHT];
    const size_t BUFSIZE = sizeof(line);
    
    /* Read the dimension: fetch the first line until space */
    char *dim;
    dim = fgets(line, BUFSIZE, fd);
    sscanf(dim, "%d", D);
    printf("%d\n", *D);
    
    /* Read the number of points: fetch the second line until newline */
    char *n;
    n = fgets(line, BUFSIZE, fd);
    sscanf(n, "%d", N);
    printf("%d\n", *N);

    /* Allocate the return matrix of dimension N x D where each line contains
     * the coordinates of a point. 
     */
    double **points = (double**) malloc((*N) * sizeof(double*));
    for(int i = 0; i < (*N); i++) points[i] = (double*) malloc((*D) * sizeof(double));

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
            points[i][k] = strtod(token, &ptr);
            token = strtok(NULL, s);
        }
    }
    return points;
}

/* Returns true if the array s dominates the array d. 
 * Parameters:
 * - s, d: arrays of double
 * - dim: number of elements of s and d 
 */
bool dominance(double *s, double *d, int dim){
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
    return !strictly_minor && strictly_major;
}

/* This function computes the Skyline set, Given:
 * - points, a matrix containing the points
 * - rows, the number of points
 * - cols, the dimension of the points
 * Returns an array of rows booleans where array[i] == true, 0 <= i < rows, 
 * if the i-th element is in the Skyline set, array[i] == false otherwise.
 */
bool* compute_skyline(double **points, int rows, int cols){
    bool *S = (bool*) malloc(rows * sizeof(bool)); 
    int n_threads = omp_get_max_threads();
    int i, j;
    
    /* This section creates a pool of threads:
     * each one compares the assigned points (of its subset) with all the others
     * in the set. If a point in the set is dominated by one of the subset then
     * we assign false into the corrisponding element of the return array to
     * indicate that it's not in the Skyline set.
     */
#pragma omp parallel default(none) num_threads(n_threads) private(i, j) shared(S, rows, cols, points, n_threads)
    {   
        /* Compute local start and local end indexes, initialize array S*/ 
        int thread_id = omp_get_thread_num();
        int local_start = rows * thread_id / n_threads;
        int local_end = rows * (thread_id + 1) / n_threads;
        for(i = local_start; i < local_end; i++) S[i] = true;
#pragma omp barrier    
        /* Once S is full initialized, start computing Skyline set */
        for(i = local_start; i < local_end; i++){
                for(j = 0; j < rows && S[i]; j++){
                    if(dominance(points[j], points[i], cols)){
//#pragma omp critical
                        S[i] = false;
                    }
                } 
            
        }
    }
    return S;
}

int main(int argc, char* argv[]){
    int *D = (int*) malloc(sizeof(int));
    int *N = (int*) malloc(sizeof(int));
    /* Read the points from stdin */
    double **points = read_points(stdin, N, D);

    /* Calculate the Skyline set and measure time spent */
    double t_start = omp_get_wtime();
    bool *skyline = compute_skyline(points, *N, *D);
    double t_end = omp_get_wtime();

    /* Print the Skyline set and the time spent */
    int i, j;
    for(i = 0; i < *N; i++){
        if(skyline[i]){
            for(j = 0; j < *D; j++){
                printf("%lf ", points[i][j]);
            }
            printf("\n");
        }
    }
    printf("Time: %lf\n", t_end - t_start);
    
    return EXIT_SUCCESS;
}
