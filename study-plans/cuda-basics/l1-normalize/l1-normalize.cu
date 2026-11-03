
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Calculate L1 sum
__global__ void l1_sum_kernel(
    const float* input, float* total, int N
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (i < N) {
        val = fabsf(input[i]);
    }

    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total, sdata[0]);
    }
}

// Kernel 2: Normalize
__global__ void l1_normalize_kernel(
    const float* input,
    float* output,
    const float* total,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = *total;
        output[i] = (sum > 0.0f)
            ? input[i] / sum
            : 0.0f;
    }
}

extern "C" void solve(
    const float* input, float* output, int N
) {
    if (N <= 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_total;
    cudaMalloc(&d_total, sizeof(float));
    cudaMemset(d_total, 0, sizeof(float));

    l1_sum_kernel<<<blocks, threads>>>(
        input, d_total, N
    );

    l1_normalize_kernel<<<blocks, threads>>>(
        input, output, d_total, N
    );

    cudaDeviceSynchronize();
    cudaFree(d_total);
}
