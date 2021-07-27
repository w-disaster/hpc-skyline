Per compilare entrambi i sorgenti è possibile utilizzare il Makefile lanciando il comando 'make'.
Per eliminare i file binary lanciare invece 'make clean'.
Se si vuole compilare manualmente i sorgenti lanciare i rispettivi comandi per OpenMP e CUDA:
- gcc -std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600 -fopenmp omp-skyline.c -lm -o omp-skyline
- nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm 
