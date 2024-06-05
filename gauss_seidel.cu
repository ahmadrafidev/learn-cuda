#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_ITER 1000
#define TOLERANCE 1e-6

__global__ void gaussSeidelIterationKernel(float *A, float *b, float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sigma = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != idx) {
                sigma += A[idx * n + j] * x[j];
            }
        }
        if (A[idx * n + idx] != 0) {
            x[idx] = (b[idx] - sigma) / A[idx * n + idx];
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void gaussSeidelIteration(float *A, float *b, float *x, int n) {
    float *d_A, *d_b, *d_x;
    size_t size = n * n * sizeof(float);
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    checkCudaError(err, "Failed to allocate device memory for A");

    err = cudaMalloc(&d_b, n * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for b");

    err = cudaMalloc(&d_x, n * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for x");

    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy matrix A to device");

    err = cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector b to device");

    err = cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector x to device");

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        gaussSeidelIterationKernel<<<gridSize, blockSize>>>(d_A, d_b, d_x, n);
        checkCudaError(cudaGetLastError(), "Kernel execution failed: gaussSeidelIterationKernel");

        err = cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy result x to host");

        // Debugging: Print intermediate values of x
        printf("Iteration %d: ", iter);
        for (int i = 0; i < n; i++) {
            printf("%f ", x[i]);
        }
        printf("\n");

        // Check convergence
        bool converged = true;
        for (int i = 0; i < n; i++) {
            float sigma = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sigma += A[i * n + j] * x[j];
                }
            }
            float new_x = (b[i] - sigma) / A[i * n + i];
            if (fabs(x[i] - new_x) > TOLERANCE) {
                converged = false;
                break;
            }
        }
        if (converged) {
            printf("Converged at iteration %d\n", iter);
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
}

int main(int argc, char *argv[]) {
    int n = 16; // Default size
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    float *A = (float *)malloc(n * n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *x = (float *)malloc(n * sizeof(float));

    // Inisialisasi matriks A dan vektor b dengan variasi elemen
    for (int i = 0; i < n; i++) {
        b[i] = 1.0 + i; // Variasi nilai pada vektor b
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 10.0 + i; // Diagonal dominan dengan variasi
            } else {
                A[i * n + j] = 1.0 / (i + j + 1); // Variasi elemen non-diagonal
            }
        }
    }

    gaussSeidelIteration(A, b, x, n);

    // Cetak beberapa elemen hasil x
    printf("Hasil vektor x:\n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", x[i]);
    }

    free(A);
    free(b);
    free(x);

    return 0;
}
