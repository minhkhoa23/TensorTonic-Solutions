#include <cuda_runtime.h>
#include <float.h>

__device__ void atomicMaxFloat(float* address, float val) {
    int* addr = reinterpret_cast<int*>(address);
    int old = *addr;
    int assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;

        old = atomicCAS(addr, assumed, __float_as_int(val));
    } while (old != assumed);
}


__global__ void max_kernel(const float* input, float* result, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        atomicMaxFloat(result, input[i]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
