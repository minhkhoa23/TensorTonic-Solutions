#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (N <= 0) return;

        float max_val = input[0];

        for (int j = 1; j < N; j++) {
            if (input[j] > max_val) {
                max_val = input[j];
            }
        }

        float sum = 0.0f;

        for (int j = 0; j < N; j++) {
            sum += expf(input[j] - max_val);
        }

        for (int j = 0; j < N; j++) {
            output[j] = expf(input[j] - max_val) / sum;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}