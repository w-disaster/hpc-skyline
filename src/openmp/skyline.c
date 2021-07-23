#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lib/hpc.h"

#define LINE_LENGHT 4000

/* This function reads the points from a file descriptor and saves
 * them in the matrix "points". Also, it stores the dimension D and
 * the number of points N onto two int memory location.
 */
float* buildMatrix(FILE* fd, int* N, int* D){
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
    float *matrix = (float*) malloc((*N) * (*D) * sizeof(float));

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

bool dominance(float* s, float *d, int dim){
    bool weakly_major = true;
    bool stricly_major = false;
    for(int i = 0; i < dim && weakly_major; i++){
        if(s[i] < d[i]) {
			weakly_major = false;
		}
        if(s[i] > d[i]) {
			stricly_major = true;
		}
    }
    return weakly_major && stricly_major;
}

bool* computeSkyline(float* matrix, int rows, int cols){
    
    bool* S = (bool*) malloc(rows * sizeof(bool));
    int i, j;
    for(i = 0; i < rows; i++) S[i] = true;
    for(i = 0; i < rows; i++){
        if(S[i]){
            for(j = 0; j < rows; j++){
               if(S[j] && dominance(&matrix[i * cols], &matrix[j * cols], cols)) {
					S[j] = false;
				}
            }
        }
    }
    return S;
}

int main(int argc, char* argv[]){
    int* D = (int*) malloc(sizeof(int));
    int* N = (int*) malloc(sizeof(int));
    float* skyline_matrix = buildMatrix(stdin, N, D);
    /*for(int i = 0; i < *N; i++){
        for(int k = 0; k < *D; k++){
            printf("%Lf ", skyline_matrix[i][k]);
        }
        printf("\n");
    }*/
    float tstart = hpc_gettime();
    bool* skyline = computeSkyline(skyline_matrix, *N, *D);
	float tend = hpc_gettime();    

	int i, j;
	for(i = 0; i < *N; i++){
        if(skyline[i]){
            for(j = 0; j < *D; j++){
                printf("%f ", skyline_matrix[i * (*D) + j]);
            }
            printf("\n");
        }
    }

    printf("Time: %lf\n", tend - tstart);
    return 0;
}
