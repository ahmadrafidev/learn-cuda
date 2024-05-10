__global__ void gaussSeidelKernel(float *A, float *b, float *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sigma = 0.0f;
        for (int j = 0; j < idx; j++) {
            sigma += A[idx * N + j] * x[j];
        }
        for (int j = idx + 1; j < N; j++) {
            sigma += A[idx * N + j] * x[j];
        }
        x[idx] = (b[idx] - sigma) / A[idx * N + idx];
    }
}

void gaussSeidelSolver(float *A, float *b, float *x, int N, int maxIterations, float tolerance) {
    float *d_A, *d_b, *d_x;
    size_t size = N * sizeof(float);
    size_t sizeA = N * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_x, size);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int iter = 0; iter < maxIterations; iter++) {
        gaussSeidelKernel<<<numBlocks, blockSize>>>(d_A, d_b, d_x, N);
        cudaDeviceSynchronize();

        // Check for convergence (not optimized for performance)
        float maxError = 0.0f;
        cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float error = fabs(x[i] - d_x[i]);
            if (error > maxError) {
                maxError = error;
            }
        }
        if (maxError < tolerance) {
            break;
        }
    }

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
}
