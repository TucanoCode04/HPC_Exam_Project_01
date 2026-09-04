#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1000
#define N 1000
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

//function for write in a buffer the value of the matrix scaled to 0-255
void write_buffer(double** u, unsigned char* buffer, int m){
    double max = 5;
    double factor = 127.0 / max;
    #pragma omp for schedule(static) collapse(2)
        for(int i=0; i<m; i++){
            for(int j=0; j<m; j++){
                int value = (int)(127 + factor*u[i][j]);
                if(value > 255) value = 255;
                if(value < 0) value = 0;
                buffer[i*m + j] = (unsigned char) value;
            }
        }
}
//function for the writing of the matrix file
void write_matrix(unsigned char* buffer, int n, int t) {
    char filename [22]; //name is ./sim/frame_xxxxx.pgm + the \0
    sprintf(filename, "./sim/frame_%05d.pgm", t);

    FILE* fp = fopen(filename, "w");
    if(fp == NULL) {
        fprintf(stderr, "Error in opening file %s\n", filename);
        return;
    }
    fprintf(fp, "P5\n%d %d\n255\n", n, n);
    fwrite(buffer,sizeof(unsigned char), n*n, fp);
    fclose(fp);
    return;
}



void run(double** u, double** u_past, double** u_fut, double gamma, double c, double coeff_forw,int frame_interv){
    double** temp;
    int frame_count = 0;
    unsigned char* buffer = (unsigned char*)malloc(M*M*sizeof(unsigned char));
    #pragma omp parallel
    {
        for(int t=0; t<N; t++){
            #pragma omp for schedule(static) collapse(2)  //try to eliminate this ? collapse
            for(int i=1; i<M-1; i++){
                for(int j=1; j<M-1; j++){
                    /*if(i == 0 || i == M-1 || j == 0 || j == M-1) { poi commenta che questo if rende tutto molto meno efficiente
                        u_fut[i][j] = 0;
                        continue;
                    }*/
                    u_fut[i][j] = coeff_forw * (c*c*((u[i+1][j] -2*u[i][j] + u[i-1][j])/(S_STEP*S_STEP) + (u[i][j+1] -2*u[i][j] + u[i][j-1])/(S_STEP*S_STEP) )*T_STEP*T_STEP + gamma*u[i][j]*T_STEP - u_past[i][j] + 2*u[i][j]);
                    //uncomment for central difference method
                    //u_fut[i][j] = coeff_cent * (c*c*((u[i+1][j] -2*u[i][j] + u[i-1][j])/(S_STEP*S_STEP) + (u[i][j+1] -2*u[i][j] + u[i][j-1])/(S_STEP*S_STEP) )*T_STEP*T_STEP + gamma*u[i][j]*T_STEP - 2*u_past[i][j] + 4*u[i][j]);
                }
            }
            if(t % frame_interv == 0) {
                write_buffer(u_fut, buffer, M);
                //write the calcualted matrix to a file (must be done by 1 thread separated, maybe use omp single)
                #pragma omp single
                {
                    if(t % frame_interv == 0) {
                        write_matrix(buffer, M, frame_count);
                        frame_count++;
                    }
                }
            }
            #pragma omp single
            {
                //change the pointers to go to the next time step
                temp = u_past;
                u_past = u;
                u = u_fut;
                u_fut = temp;
            }
        }
    }
    free(buffer);
    return;
}


int main(int argc, char* argv[]) {

    //define constants
    const double gamma = 0.187; //damping coefficient
    const double c = 0.33; //wave speed 
    const double init_u = -49; //initial impulse
    //choose method for differential equation
    int forward = 1; //1 for forward, 0 for central difference method
    //allocate matrices
    double** u = matrix_alloc(M);
    double** u_past = matrix_alloc(M);
    double** u_fut = matrix_alloc(M);
    double** temp;

    //initialization of u = u_past
    for (int i=0; i<M; i++){
        for(int j=0; j<M; j++){
            u[i][j] = u_past[i][j] = 0;
        }
    }
    u[M/2][M/2] = u_past[M/2][M/2] = init_u;

    //calculation of the future matrix element from the differential equation
    double coeff_forw = 1/(gamma*T_STEP + 1);
    double coeff_cent = 1/(gamma*T_STEP + 2);
    double max = 0; //min and max used for the scale factor in the write matrix
    double min = 0;
    double start = omp_get_wtime();
    /* trial simulation for the normalization of values
    for(int t=0; t<N; t++){
        for(int i=1; i<M-1; i++){
            for(int j=1; j<M-1; j++){
                u_fut[i][j] = coeff_forw * (c*c*((u[i+1][j] -2*u[i][j] + u[i-1][j])/(S_STEP*S_STEP) + (u[i][j+1] -2*u[i][j] + u[i][j-1])/(S_STEP*S_STEP) )*T_STEP*T_STEP + gamma*u[i][j]*T_STEP - u_past[i][j] + 2*u[i][j]);
                //u_fut[i][j] = coeff_cent * (c*c*((u[i+1][j] -2*u[i][j] + u[i-1][j])/(S_STEP*S_STEP) + (u[i][j+1] -2*u[i][j] + u[i][j-1])/(S_STEP*S_STEP) )*T_STEP*T_STEP + gamma*u[i][j]*T_STEP - 2*u_past[i][j] + 4*u[i][j]);
                if(u_fut[i][j] > max && u_fut[i][j] != 0) {
                    max = u_fut[i][j];
                }
                if(u_fut[i][j] < min) {
                    min = u_fut[i][j];
                }
            
            }
        }
        temp = u_past;
        u_past = u;
        u = u_fut;
        u_fut = temp;
    }  */
    int frame_interv = 1; //save only some frames
    run(u,u_past,u_fut,gamma,c,coeff_forw,frame_interv);
    //printf("max: %f, min: %f\n", max, min);
    double finish = omp_get_wtime();
    printf("execution time: %f seconds\n", (finish - start));
    //free the allocated memory for the matrices
    matrix_free(u, M);
    matrix_free(u_past, M);
    matrix_free(u_fut, M);

}