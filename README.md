# Progetto di High Performance Computing 2020/2021

## L'operatore SKYLINE
Consideriamo un insieme P = {P0, P2, ... PN-1} di N punti in uno spazio a D dimensioni; il punto Pi
ha coordinate (Pi,0; Pi,1; ...; Pi,D-1). Diciamo che Pi domina Pj se si verificano entrambe le
condizioni seguenti:
- Per ogni dimensione k = 0, ... D-1, Pi,k ≥ Pj,k
- Esiste almeno una coordinata per cui la disuguaglianza di cui al punto precedente vale in
senso stretto

Lo skyline Sk(P) dell'insieme P è composto da tutti i punti che non sono dominati da alcun altro
punto (si noti che per definizione un punto non domina se stesso):

Sk(P) = {s ∈ P : non esiste alcun punto t in P tale che t domini s}

Scopo del progetto è la realizzazione di due implementazioni parallele dell'algoritmo per il calcolo
dello skyline visto nella sezione precedente. La prima deve essere realizzata usando OpenMP; la
seconda deve essere realizzata, a scelta, usando MPI oppure CUDA (uno dei due, non entrambi).

## Compilazione

Per compilare entrambi i sorgenti è possibile utilizzare il Makefile lanciando il comando ``` make ```.
Per eliminare i file binari lanciare invece ``` make clean ```.
Per compilare solamente la versione OpenMP: ``` make openmp ```, mentre solo la versione CUDA: ``` make cuda ```.
Se si vuole compilare manualmente i sorgenti lanciare i rispettivi comandi per OpenMP e CUDA:

- ``` gcc -std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600 -fopenmp omp-skyline.c -lm -o omp-skyline ```
- ``` nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm ```
