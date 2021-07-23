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
            /* convert ASCII string to doubleing-point number */
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
 * - length: number of elements of s and d 
 */
bool dominance(double *s, double *d, int length){
    bool strictly_major = false;
    /* Iterate over each index: 
     * if s[i] < d[i] then s doesn't dominate d --> exit from loop and return */
    for(int i = 0; i < length; i++){
        if(s[i] < d[i]){
			return false;
		}
        if(s[i] > d[i]){
			strictly_major = true;
		}
    }
    /* If there aren't elements strictly minor and exist at least on element
     * strictly major then s dominates d
     */
    return strictly_major;
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
bool* compute_skyline(double **points, int rows, int cols, int *skyline_length){
    bool *S = (bool*) malloc(rows * sizeof(bool)); 
    int n_threads = omp_get_max_threads();
    int i, j;
    int S_length[n_threads];

    /* This section creates a pool of threads:
     * each one compares the assigned points (of its subset) with all the others
     * in the set. If a point in the set is dominated by one of the subset then
     * we assign false into the corrisponding element of the return array to
     * indicate that it's not in the Skyline set.
     * The writes on the array S are atomic because it's shared between the
     * threads and we must avoid race conditions.
     * When all threads exit the external for the cardinality of the Skyline set
     * is computed.
     */
#pragma omp parallel default(none) num_threads(n_threads) private(i, j) \
    shared(S, S_length, rows, cols, points, n_threads)
    {   
        /* Compute local start and local end indexes, initialize array S */ 
        int thread_id = omp_get_thread_num();
        int local_start = rows * thread_id / n_threads;
        int local_end = rows * (thread_id + 1) / n_threads;
		S_length[thread_id] = 0;
        for(i = local_start; i < local_end; i++) S[i] = true;
#pragma omp barrier    
        /* Once S is full initialized, start computing Skyline set */
        for(i = local_start; i < local_end; i++){
            if(S[i]){
                for(j = 0; j < rows; j++){
                    if(S[j] && dominance(points[i], points[j], cols)){
#pragma omp atomic write 
                        S[j] = false;            
                    }
                } 
            }
        }
#pragma omp barrier
        /* Each thread calculates a partial cardinality of Skyline set */
        for(i = local_start; i < local_end; i++) {
            if(S[i]){
                S_length[thread_id]++;
            }
        }
    }

    /* Sum all partial cardinalities */ 
	*skyline_length = 0;
	for(int i = 0; i < n_threads; i++){
		*skyline_length += S_length[i];
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

int main(void){
    double t_start = omp_get_wtime();

    /* int pointers to store dimension D, cardinality N of the input set
     * and Skyline cardinality K.
     */
    int *D = (int*) malloc(sizeof(int));
    int *N = (int*) malloc(sizeof(int));
    int *K = (int*) malloc(sizeof(int));

    /* Read the points from stdin */
    double **points = read_points(stdin, N, D);

    /* Calculate the Skyline set and measure time spent */
    bool *skyline = compute_skyline(points, *N, *D, K);
   
	/* Print Skyline set */
    print_skyline(stdout, skyline, points, *N, *D, *K);
	
    double t_end = omp_get_wtime();
    /* Print the time spent */
    fprintf(stdout, "Time: %lf\n", t_end - t_start);
    return EXIT_SUCCESS;
}
