#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        output[i] = (input[i] / 2) * (1 + erff(input[i] / sqrtf(2)));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
