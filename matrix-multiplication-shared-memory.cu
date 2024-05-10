__global__ void matrixMultiplySharedKernel(float *A, float *B, float *C, int N) {
    __shared__ float s_A[32][32];
    __shared__ float s_B[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (N + 32 - 1) / 32; k++) {
        if (k * 32 + threadIdx.x < N && row < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + k * 32 + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (k * 32 + threadIdx.y < N && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(k * 32 + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int n = 0; n < 32; ++n) {
            sum += s_A[threadIdx.y][n] * s_B[n][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void parallelMatrixMultiplyShared(float *A, float *B, float *C, int N, int blockSize) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    matrixMultiplySharedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1024; // Change as needed
    int blockSize = 32; // Change as needed
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    parallelMatrixMultiplyShared(A, B, C, N, blockSize);

    // Print a small part of the matrix to verify correctness
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
