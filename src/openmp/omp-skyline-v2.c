/****************************************************************************
 *
 * omp-skyline.c - Skyline set computation with OpenMP
 *
 * Author: Fabri Luca 
 * Serial Number: 0000892878
 * Email: luca.fabri@studio.unibo.it
 * 
 * ---------------------------------------------------------------------------
 * 
 * Skyline set computation with OpenMP. 
 * Given P a set of N points with dimension D, p1, p2 two points in P,
 * we say that p1 dominates p2 if:
 * - for each dimension k: p1[k] >= p2[k] , 0 <= k < D;
 * - exists at least one dimension j such that: p1[j] > p2[j] , 0 <= j < D.
 *
 * The Skyline set is, for definition, composed of all points in P that aren't
 * dominated by other points in P.
 *
 * Compile with:
 * gcc -Wall -Wpedantic -std=c99 -fopenmp -o omp-skyline omp-skyline.c
 * Or from Makefile:
 * make openmp
 *
 * Run with:
 * ./omp-skyline < input_file > output_file
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
#include <omp.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000

/* 
 * This function reads the points from a file descriptor and saves
 * them into a matrix. Also, it stores the dimension D and
 * the number of points N onto two int memory locations.
 * 
 * Parameters:
 * - fd: file descriptor
 * - N: pointer to integer where this function stores the number of points read
 * - D: pointer to int where this function stores the dimension of the points.
 * 
 * It returns the double pointer to the allocated matrix containing the points.
 */
double **read_points(FILE *fd, int *N, int *D){
    char line[LINE_LENGHT];
    const size_t BUFSIZE = sizeof(line);
    
    /* Read the dimension: fetch the first line until space */
    char *dim;
    dim = fgets(line, BUFSIZE, fd);
    sscanf(dim, "%d", D);
    
    /* Read the number of points: fetch the second line until newline */
    char *n;
    n = fgets(line, BUFSIZE, fd);
    sscanf(n, "%d", N);
    
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

/* 
 * Returns true if the array s dominates the array d. 
 * Parameters:
 * - s, d: arrays of double
 * - dim: number of elements of s and d 
 */
bool dominance(double *s, double *d, int dim){
    bool strictly_minor = false;
    bool strictly_major = false;
    /* Iterate over each dimension: 
     * if s[i] < d[i] then s doesn't dominate d --> exit from loop and return */
    for(int i = 0; i < dim && !strictly_minor; i++){
        if(s[i] < d[i]){
			strictly_minor = true;
		}
        if(s[i] > d[i]){
			strictly_major = true;
		}
    }
    /* If there aren't elements strictly minor and exist at least on element
     * strictly major then s dominates d
     */
    return !strictly_minor && strictly_major;
}

/* 
 * This function computes the Skyline set, Given:
 * - points, a matrix containing the points;
 * - rows, the number of points;
 * - cols, the dimension of the points;
 * - skyline_length, pointer to int to store the cardinality 
 *   of the Skyline set.
 *
 * Returns an array of rows booleans where array[i] == true, 0 <= i < rows, 
 * if the i-th element is in the Skyline set, array[i] == false otherwise.
 */
bool* compute_skyline(double **points, int rows, int cols, int *skyline_card){
    bool *S = (bool*) malloc(rows * sizeof(bool));
    int n_threads = omp_get_max_threads(); 
    int i, j;
    *skyline_card = 0;
    
    /* This section creates a pool of threads:
     * each one compares the assigned points (of its subset) with all the others
     * in the set. If a point in the set is dominated by one of the subset then
     * we assign false into the corrisponding element of the return array to
     * indicate that it's not in the Skyline set.
     */
#pragma omp parallel default(none) num_threads(n_threads) private(i, j) \
    shared(S, rows, cols, points, n_threads)
    {   
        /* Compute local start and local end indexes, initialize array S*/ 
        int thread_id = omp_get_thread_num();
        int local_start = rows * thread_id / n_threads;
        int local_end = rows * (thread_id + 1) / n_threads;
        int local_s_card = local_end - local_start;
  
        /* Start computing Skyline set */
        for(i = local_start; i < local_end; i++){
            S[i] = true;
            for(j = 0; j < rows && S[i]; j++){
                if(dominance(points[j], points[i], cols)){
                    S[i] = false;
                    local_s_card--;
                }
            }  
        }
#pragma omp atomic
        *skyline_card += local_s_card;
    }
    return S;
}

/*
 * This function prints to the file descriptor fd given as parameter:
 * - The dimension D of the points;
 * - The cardinality K of the Skyline set;
 * - The Skyline set.
 */
void print_skyline(FILE* fd, bool *S, double **points, int N, int D, int K){
    int i, j;
    /* Print D, K */
    fprintf(fd, "%d\n%d\n", D, K);

    /* Print the Skyline set */
    for(i = 0; i < N; i++){
        if(S[i]){
            for(j = 0; j < D; j++){
                fprintf(fd, "%lf ", points[i][j]);
            }
            fprintf(fd, "\n");
        }
    }
}

int main(int argc, char* argv[]){
    /* int pointers to store dimension D, cardinality N of the input set
     * and Skyline cardinality K.
     */
    int *D = (int*) malloc(sizeof(int));
    int *N = (int*) malloc(sizeof(int));
    int *K = (int*) malloc(sizeof(int));

    /* Read the points from stdin */
    double **points = read_points(stdin, N, D);

    /* Calculate the Skyline set and measure time spent */
    double t_start = omp_get_wtime();
    bool *skyline = compute_skyline(points, *N, *D, K);
    double t_end = omp_get_wtime();

	/* Print Skyline set */
    print_skyline(stdout, skyline, points, *N, *D, *K);
	/* Print the time spent */
    fprintf(stdout, "Time: %lf\n", t_end - t_start);
    
    return EXIT_SUCCESS;
}
