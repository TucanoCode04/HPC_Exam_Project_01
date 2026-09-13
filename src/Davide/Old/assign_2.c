#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#define M 500
#define N 700
#define S_STEP 0.01
#define T_STEP 0.01

//function to allocate memory for the matrices
double** matrix_alloc(int n) {
    double** matrix = (double**)malloc(n*sizeof(double*));
    for(int i=0; i<n; i++){
        matrix[i] = (double*)malloc(n*sizeof(double));
    }
    return matrix;
}

//function fo free matrices
void matrix_free(double** matrix, int n) {
    for(int i=0; i<n; i++){
        free(matrix[i]);
    }
    free(matrix);
    return;
}

//function for the writing of the matrix file
void write_matrix(double** matrix, int n, int t, int rank) {
    
    double max = 5; //obtained froma  trial simualtion
    double factor = 127.0 / max; //to scale the values to the range in 0-255
    char filename [23]; //name is ./sim/frame_xxxxx.pgm + the \0
    sprintf(filename, "./sim%d/frame_%05d.pgm",rank+1, t);

    FILE* fp = fopen(filename, "w");
    if(fp == NULL) {
        fprintf(stderr, "Error in opening file %s\n", filename);
        return;
    }
    fprintf(fp, "P5\n%d %d\n255\n", n, n);
    int value;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            value = (int)(127 + factor*matrix[i][j]);
            if(value > 255) value = 255;
            if(value < 0) value = 0;
            unsigned char pixel = (unsigned char)value;
            fwrite(&pixel, sizeof(unsigned char), 1, fp);
        }
    }
    fclose(fp);
    return;
}

int main(int argc, char* argv[]) {

    //define constants
    const double gamma = 0.187;
    const double c = 0.33;
    const double init_u = -49;
    const double init_u1 = 96;
    const double init_u2 = -37;
    const int tstart = N/5;
    //allocate matrices
    double** u = matrix_alloc(M);
    double** u_past = matrix_alloc(M);
    double** u_fut = matrix_alloc(M);
    double** temp;

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //initialization of u = u_past
    for (int i=0; i<M; i++){
        for(int j=0; j<M; j++){
            u[i][j] = u_past[i][j] = 0;
        }
    }
    if(rank == 0) {
        u[M/2][M/2] = u_past[M/2][M/2] = init_u;
    }
    if(rank == 1) {
        u[M/4][M/3] = u_past[M/4][M/3] = init_u1;
        u[M/2][M*2/3] = u_past[M/2][M*2/3] = init_u2;
    }
    if(rank == 2) {
        u[M/4][M/3] = u_past[M/4][M/3] = init_u1; //the other wave will be generated after time tstart

    }
    

    //calculation of the future matrix element from the differential equation
    double coeff = 1/(gamma*T_STEP + 1);
    double max = 0; //min and max used for the scale factor in the write matrix
    double min = 0;
    double start = omp_get_wtime();

    //save only some frames
    int frame_interv = 1;
    int frame_count = 0;
    #pragma omp parallel
    {   
        for(int t=0; t<N; t++){
            #pragma omp single 
            {
                if(t == tstart && rank == 2) {
                u[M/2][M*2/3] = u_past[M/2][M*2/3] = init_u2;
                }
            }
            #pragma omp for schedule(static) collapse(2)
            for(int i=1; i<M-1; i++){
                for(int j=1; j<M-1; j++){
                    /*if(i == 0 || i == M-1 || j == 0 || j == M-1) { poi commenta che questo if rende tutto molto meno efficiente
                        u_fut[i][j] = 0;
                        continue;
                    }*/
                    u_fut[i][j] = coeff * (c*c*((u[i+1][j] -2*u[i][j] + u[i-1][j])/(S_STEP*S_STEP) + (u[i][j+1] -2*u[i][j] + u[i][j-1])/(S_STEP*S_STEP) )*T_STEP*T_STEP + gamma*u[i][j]*T_STEP - u_past[i][j] + 2*u[i][j]);
                    /*if(u_fut[i][j] > max) {
                        max = u_fut[i][j];
                    }
                    if(u_fut[i][j] < min) {
                        min = u_fut[i][j];
                    }*/
                }
            }
            //write the calcualted matrix to a file (can be done by 1 thread separated, maybe use omp single)
            #pragma omp single
            {
                if(t % frame_interv == 0) {
                    write_matrix(u_fut, M, frame_count,rank);
                    frame_count++;
                }
                //write_matrix(u_fut, M, t);
                //change the pointers to go to the next time step
                temp = u_past;
                u_past = u;
                u = u_fut;
                u_fut = temp;
            }
        }
    }

    double finish = omp_get_wtime();
    printf("execution time of process %d: %f seconds\n",rank+1, (finish - start));
    //free the allocated memory for the matrices
    matrix_free(u, M);
    matrix_free(u_past, M);
    matrix_free(u_fut, M);
    MPI_Finalize();
}