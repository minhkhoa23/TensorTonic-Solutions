#include <cuda_runtime.h>
#include <float.h>

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Write code here
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Step 1: Load data into shared memory
    if (idx < N) {
        s_vals[tid] = input[idx];
        s_idxs[tid] = idx;
    } else {
        s_vals[tid] = FLT_MAX;
        s_idxs[tid] = -1;
    }

    __syncthreads();

    // Step 2: Parallel reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            int other = tid + stride;

            if (s_idxs[other] != -1 &&
                (s_idxs[tid] == -1 ||
                 s_vals[other] < s_vals[tid] ||
                 (s_vals[other] == s_vals[tid] &&
                  s_idxs[other] < s_idxs[tid]))) {

                s_vals[tid] = s_vals[other];
                s_idxs[tid] = s_idxs[other];
            }
        }

        __syncthreads();
    }

    // Step 3: Save block minimum
    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    // Write code here
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;

    float min_val = FLT_MAX;
    int min_idx = -1;

    // Step 1: Each thread processes multiple blocks
    for (int i = tid; i < num_blocks;
         i += blockDim.x) {

        float val = block_vals[i];
        int idx = block_idxs[i];

        if (idx != -1 &&
            (min_idx == -1 ||
             val < min_val ||
             (val == min_val && idx < min_idx))) {

            min_val = val;
            min_idx = idx;
        }
    }

    s_vals[tid] = min_val;
    s_idxs[tid] = min_idx;

    __syncthreads();

    // Step 2: Final reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            int other = tid + stride;

            if (s_idxs[other] != -1 &&
                (s_idxs[tid] == -1 ||
                 s_vals[other] < s_vals[tid] ||
                 (s_vals[other] == s_vals[tid] &&
                  s_idxs[other] < s_idxs[tid]))) {

                s_vals[tid] = s_vals[other];
                s_idxs[tid] = s_idxs[other];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
