__global__ void vecAdd(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vecSub(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void vecDot(float *a, float *b, float *c, int N) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (idx < N) {
        temp += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

void conjugateGradientSolver(float *A, float *b, float *x, int N, int maxIterations, float tolerance) {
    float *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap, *d_temp;
    size_t size = N * sizeof(float);
    size_t sizeA = N * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_r, size);
    cudaMalloc(&d_p, size);
    cudaMalloc(&d_Ap, size);
    cudaMalloc(&d_temp, size);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // r = b - Ax
    vecSub<<<numBlocks, blockSize>>>(d_b, d_Ap, d_r, N);
    cudaDeviceSynchronize();

    // p = r
    cudaMemcpy(d_p, d_r, size, cudaMemcpyDeviceToDevice);

    float rsold, rsnew;
    float *d_rsold, *d_rsnew;
    cudaMalloc(&d_rsold, sizeof(float));
    cudaMalloc(&d_rsnew, sizeof(float));

    vecDot<<<numBlocks, blockSize>>>(d_r, d_r, d_temp, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&rsold, d_temp, sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < maxIterations; i++) {
        // Ap = A * p
        // Here you need to implement matrix-vector multiplication for A * p

        // alpha = rsold / (p' * Ap)
        vecDot<<<numBlocks, blockSize>>>(d_p, d_Ap, d_temp, N);
        cudaDeviceSynchronize();
        float pAp;
        cudaMemcpy(&pAp, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
        float alpha = rsold / pAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
        // rsnew = r' * r

        vecDot<<<numBlocks, blockSize>>>(d_r, d_r, d_temp, N);
        cudaDeviceSynchronize();
        cudaMemcpy(&rsnew, d_temp, sizeof(float), cudaMemcpyDeviceToHost);

        if (sqrt(rsnew) < tolerance) {
            break;
        }

        // p = r + (rsnew / rsold) * p
        rsold = rsnew;
    }

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_temp);
    cudaFree(d_rsold);
    cudaFree(d_rsnew);
}
