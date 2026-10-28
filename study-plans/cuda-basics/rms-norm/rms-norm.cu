#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // Write code here
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_sum[256];
    __shared__ float inv_rms;

    // Step 1: Compute sum of squares
    float sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        float x = input[row * N + j];
        sum += x * x;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Step 2: Parallel reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }

        __syncthreads();
    }

    // Step 3: Compute inverse RMS
    if (tid == 0) {
        float mean_square = shared_sum[0] / N;
        inv_rms = rsqrtf(mean_square + eps);
    }

    __syncthreads();

    // Step 4: Normalize and apply gamma
    for (int j = tid; j < N; j += blockDim.x) {
        int idx = row * N + j;
        output[idx] = input[idx] * inv_rms * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}