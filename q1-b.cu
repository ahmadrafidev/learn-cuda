void incrementArrayLarge(int *a, int N, int blockSize) {
    int *d_a;
    size_t size = N * sizeof(int);

    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Process the array in chunks
    for (int i = 0; i < N; i += blockSize * 1024) {
        int currentSize = min(blockSize * 1024, N - i);
        int numBlocks = (currentSize + blockSize - 1) / blockSize;
        incrementKernel<<<numBlocks, blockSize>>>(d_a + i, currentSize);
    }

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}
