#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_ITER 1000 // jumlah iterasi maksimum
#define TOLERANCE 1e-6 // toleransi konvergensi

__global__ void jacobiIterationKernel(float *A, float *b, float *x, float *x_new, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sigma = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != idx) {
                sigma += A[idx * n + j] * x[j];
            }
        }
        if (A[idx * n + idx] != 0) {
            x_new[idx] = (b[idx] - sigma) / A[idx * n + idx];
        } else {
            x_new[idx] = x[idx]; // Hindari pembagian dengan nol
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

bool checkConvergence(float *x, float *x_new, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        if (fabs(x[i] - x_new[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void jacobiIteration(float *A, float *b, float *x, int n) {
    float *d_A, *d_b, *d_x, *d_x_new;
    size_t size = n * n * sizeof(float);
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    checkCudaError(err, "Failed to allocate device memory for A");

    err = cudaMalloc(&d_b, n * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for b");

    err = cudaMalloc(&d_x, n * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for x");

    err = cudaMalloc(&d_x_new, n * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for x_new");

    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy matrix A to device");

    err = cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector b to device");

    err = cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector x to device");

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    float *h_x_new = (float *)malloc(n * sizeof(float));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        jacobiIterationKernel<<<gridSize, blockSize>>>(d_A, d_b, d_x, d_x_new, n);
        err = cudaGetLastError();
        checkCudaError(err, "Kernel execution failed");

        err = cudaMemcpy(h_x_new, d_x_new, n * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy result x_new to host");

        // Print values for debugging
        printf("Iteration %d: ", iter);
        for (int i = 0; i < n; i++) {
            printf("%f ", h_x_new[i]);
        }
        printf("\n");

        // Check for NaN or Inf
        for (int i = 0; i < n; i++) {
            if (isnan(h_x_new[i]) || isinf(h_x_new[i])) {
                printf("NaN or Inf detected at index %d\n", i);
                free(h_x_new);
                cudaFree(d_A);
                cudaFree(d_b);
                cudaFree(d_x);
                cudaFree(d_x_new);
                return; // Stop execution
            }
        }

        // Check convergence
        if (checkConvergence(x, h_x_new, n, TOLERANCE)) {
            printf("Converged at iteration %d\n", iter);
            break;
        }

        // Copy new values to x for next iteration
        err = cudaMemcpy(d_x, d_x_new, n * sizeof(float), cudaMemcpyDeviceToDevice);
        checkCudaError(err, "Failed to copy x_new to x on device");

        // Update host x for next iteration
        for (int i = 0; i < n; i++) {
            x[i] = h_x_new[i];
        }
    }

    err = cudaMemcpy(x, d_x_new, n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy final result x to host");

    free(h_x_new);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);
}

int main(int argc, char *argv[]) {
    int n = 1024; // Default size
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    float *A = (float *)malloc(n * n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *x = (float *)malloc(n * sizeof(float));

    // Inisialisasi matriks A dan vektor b
    for (int i = 0; i < n; i++) {
        b[i] = (float)(i + 1); // Nilai yang lebih beragam untuk b
        x[i] = 0.0; // Inisialisasi dengan nilai nol
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = (float)(n); // Diagonal dominan
            } else {
                A[i * n + j] = 0.1; // Nilai lebih kecil untuk elemen non-diagonal
            }
        }
    }

    jacobiIteration(A, b, x, n);

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
