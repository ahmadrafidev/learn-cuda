#include <stdio.h>
#include <cuda_runtime.h>

__global__ void incrementKernel(int *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] += 1;
    }
}

void incrementArray(int *a, int N, int blockSize) {
    int *d_a;
    size_t size = N * sizeof(int);

    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int numBlocks = (N + blockSize - 1) / blockSize;
    incrementKernel<<<numBlocks, blockSize>>>(d_a, N);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}

int main() {
    const int N = 1024;  // Change as needed for your experiment
    int a[N];

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }

    // Choose different block sizes to see the effect
    incrementArray(a, N, 256);  // Change block size here

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}
