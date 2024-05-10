#include <stdio.h>
#include <cuda_runtime.h>

__global__ void jacobiKernel(float *A, float *b, float *x, float *x_new, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sigma = 0.0f;
        for (int j = 0; j < N; j++) {
            if (j != idx) {
                sigma += A[idx * N + j] * x[j];
            }
        }
        x_new[idx] = (b[idx] - sigma) / A[idx * N + idx];
    }
}

void jacobiSolver(float *A, float *b, float *x, int N, int maxIterations, float tolerance) {
    float *d_A, *d_b, *d_x, *d_x_new;
    size_t size = N * sizeof(float);
    size_t sizeA = N * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_x_new, size);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int iter = 0; iter < maxIterations; iter++) {
        jacobiKernel<<<numBlocks, blockSize>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();

        // Check for convergence (not optimized for performance)
        float maxError = 0.0f;
        cudaMemcpy(x, d_x_new, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float error = fabs(x[i] - d_x[i]);
            if (error > maxError) {
                maxError = error;
            }
        }
        if (maxError < tolerance) {
            break;
        }

        // Swap pointers
        float *temp = d_x;
        d_x = d_x_new;
        d_x_new = temp;
    }

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);
}

int main() {
    int N = 1024; // Adjust as necessary
    int maxIterations = 10000;
    float tolerance = 1e-6;

    float *A = (float*)malloc(N * N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *x = (float*)malloc(N * sizeof(float));

    // Initialize A, b, and x (example initialization)
    for (int i = 0; i < N; i++) {
        b[i] = i;
        x[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0f : 1.0f; // Example values
        }
    }

    jacobiSolver(A, b, x, N, maxIterations, tolerance);

    // Print the solution
    for (int i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    free(A);
    free(b);
    free(x);

    return 0;
}
