#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    // Write code here
    __shared__ float s_dot[256];
    __shared__ float s_a2[256];
    __shared__ float s_b2[256];

    int tid = threadIdx.x;

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Step 1: Local sums
    float dot = 0.0f;
    float sumA2 = 0.0f;
    float sumB2 = 0.0f;

    for (int i = idx; i < N; i += stride) {
        float a = A[i];
        float b = B[i];

        dot += a * b;
        sumA2 += a * a;
        sumB2 += b * b;
    }

    // Step 2: Store in shared memory
    s_dot[tid] = dot;
    s_a2[tid] = sumA2;
    s_b2[tid] = sumB2;

    __syncthreads();

    // Step 3: Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_a2[tid] += s_a2[tid + s];
            s_b2[tid] += s_b2[tid + s];
        }

        __syncthreads();
    }

    // Step 4: Atomic addition
    if (tid == 0) {
        atomicAdd(&scratch[0], s_dot[0]);
        atomicAdd(&scratch[1], s_a2[0]);
        atomicAdd(&scratch[2], s_b2[0]);
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    // Write code here
    float dot = scratch[0];
    float sumA2 = scratch[1];
    float sumB2 = scratch[2];

    if (sumA2 > 0.0f && sumB2 > 0.0f) {
        result[0] = dot /
            (sqrtf(sumA2) * sqrtf(sumB2));
    } else {
        result[0] = 0.0f;
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}