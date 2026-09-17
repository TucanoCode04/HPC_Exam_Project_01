#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

/**
 * Allocates a contiguous 2D array to optimize spatial locality, 
 * reduce cache misses, and avoid memory fragmentation.
 */
double** matrix_alloc(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    matrix[0] = (double**)malloc(n * n * sizeof(double));
    for (int i = 1; i < n; i++) {
        matrix[i] = matrix[i - 1] + n;
    }
    return matrix;
}

/**
 * Frees the previously allocated contiguous 2D matrix memory block.
 */
void matrix_free(double** matrix, int n) {
    free(matrix[0]);
    free(matrix);
}

/**
 * Executes the 2D wave equation simulation using hybrid MPI + OpenMP parallelism.
 */
void run_simulation(double** u, double** u_past, double** u_fut, int rank,
                    int i2, int j2, double impulso2, int n_start,
                    int M, int N, double dx, double dt) {
    // Physical parameters governing the wave propagation model
    double gamma_val = 0.130;
    double c = 0.21;
    double coeff_forw = 1.0 / (gamma_val * dt + 1.0);
    double c2_dt2_dx2 = (c * c * dt * dt) / (dx * dx);
    double gamma_dt = gamma_val * dt;

    double t_start = omp_get_wtime();

    // Parallel region spanning across all simulation timesteps
    #pragma omp parallel
    {
        for (int t = 0; t < N; t++) {
            
            // Conditional injection of the secondary time-delayed impulse (Simulation 3)
            if (rank == 2 && t == n_start) {
                #pragma omp single
                {
                    u[i2][j2] += impulso2;
                    u_past[i2][j2] += impulso2;
                }
            }

            // Parallel explicit finite difference stencil computation for the wave grid
            #pragma omp for schedule(static)
            for (int i = 1; i < M - 1; i++) {
                for (int j = 1; j < M - 1; j++) {
                    double laplacian = u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j] +
                                       u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1];

                    u_fut[i][j] = coeff_forw * (c2_dt2_dx2 * laplacian + gamma_dt * u[i][j] - u_past[i][j] + 2.0 * u[i][j]);
                }
            }
            
            // Master thread handles temporal pointer rotation safely
            #pragma omp single
            {
                double** temp = u_past;
                u_past = u;
                u = u_fut;
                u_fut = temp;
            }
        }
    }

    double t_end = omp_get_wtime();
    double local_time = t_end - t_start;

    // Each MPI process reports its isolated execution performance
    printf("Rank %d (M=%d): Completata in %f sec.\n", rank, M, local_time);
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    
    // Initialize the MPI execution environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int M = 1000; // Default grid dimension

    // Override grid size via command-line argument if provided (e.g., from sbatch)
    if (argc > 1) {
        M = atoi(argv[1]); 
    }
    
    int N = 1000;         // Total time steps
    double dx = 0.01;     // Spatial step
    double dt = 0.01;     // Temporal step

    // Primary initial impulse configuration (shared central disturbance)
    const double impulso1 = -84.0;
    int i1 = M / 2, j1 = M / 2;

    // Secondary impulse and scenario-specific parameters
    int i2 = 0, j2 = 0, n_start = -1;
    double impulso2 = 0.0;

    // Configure distinct scenario parameters based on the MPI process rank
    if (rank == 1) {
        i2 = (3 * M) / 4; j2 = M / 3;
        impulso2 = 36.0; n_start = 0;
    } else if (rank == 2) {
        i2 = (3 * M) / 4; j2 = (2 * M) / 3;
        impulso2 = 60.0; n_start = N / 7;
    }

    // Allocate simulation field matrices dynamically
    double** u = matrix_alloc(M);
    double** u_past = matrix_alloc(M);
    double** u_fut = matrix_alloc(M);

    // Initialize grid states to zero in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            u[i][j] = u_past[i][j] = u_fut[i][j] = 0.0;
        }
    }

    // Inject initial impulses at time t = 0
    u[i1][j1] = u_past[i1][j1] = impulso1;
    if (rank == 1 && n_start == 0) u[i2][j2] = u_past[i2][j2] = impulso2;

    // Run the core solver simulation
    run_simulation(u, u_past, u_fut, rank, i2, j2, impulso2, n_start, M, N, dx, dt);

    // Clean up allocated memory resources
    matrix_free(u, M);
    matrix_free(u_past, M);
    matrix_free(u_fut, M);

    // Finalize the MPI execution environment
    MPI_Finalize();
    return 0;
}