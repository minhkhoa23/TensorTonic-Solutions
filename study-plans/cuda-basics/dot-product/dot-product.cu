#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float product = A[i] * B[i];
        atomicAdd(result, product);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
