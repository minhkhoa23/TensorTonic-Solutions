#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float x = input[i];
        atomicAdd(mean_out, x / N);
        atomicAdd(var_out, (x * x) / N);
    }
}

__global__ void finalize_kernel(
    const float* mean_out,
    float* var_out
) {
    float mean = *mean_out;
    *var_out -= mean * mean;
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    finalize_kernel<<<1, 1>>>(mean_out, var_out);
    cudaDeviceSynchronize();
}
