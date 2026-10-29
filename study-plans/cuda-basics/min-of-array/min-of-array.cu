#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__device__ void atomicMinFloat(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int;
    int assumed;

    do {
        assumed = old;

        float current = __int_as_float(assumed);

        if (val >= current) return;

        old = atomicCAS(
            addr_int,
            assumed,
            __float_as_int(val)
        );

    } while (old != assumed);
}
__global__ void min_kernel(const float* input, float* result, int N) {
    // Write code here
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Bước 1: Load vào Shared Memory
    sdata[tid] = (idx < N) ? input[idx] : FLT_MAX;

    __syncthreads();

    // Bước 2: Parallel Reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            sdata[tid] = fminf(
                sdata[tid],
                sdata[tid + stride]
            );
        }

        __syncthreads();
    }

    // Bước 3: Cập nhật Global Minimum
    if (tid == 0) {
        atomicMinFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
