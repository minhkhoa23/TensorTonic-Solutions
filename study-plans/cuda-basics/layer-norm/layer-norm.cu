#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    // Write code here
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M || N <= 0) return;

    __shared__ float shared[256];
    __shared__ float mean;
    __shared__ float inv_std;

    long long offset = (long long)row * N;

    // Step 1: Calculate sum
    float sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        sum += input[offset + j];
    }

    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean = shared[0] / N;
    }
    __syncthreads();

    // Step 2: Calculate variance
    float var_sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        float diff = input[offset + j] - mean;
        var_sum += diff * diff;
    }

    shared[tid] = var_sum;
    __syncthreads();

    // Parallel reduction for variance
    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        inv_std = rsqrtf(shared[0] / N + eps);
    }
    __syncthreads();

    // Step 3: Normalize, scale and shift
    for (int j = tid; j < N; j += blockDim.x) {
        float x = input[offset + j];
        output[offset + j] =
            (x - mean) * inv_std * gamma[j] + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
