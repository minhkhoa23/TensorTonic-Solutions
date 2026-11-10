#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Write code here
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load du lieu vao shared memory
    if (idx < N) {
        s_vals[tid] = input[idx];
        s_idxs[tid] = idx;
    } else {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = -1;
    }

    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            int other = tid + stride;

            if (s_idxs[other] != -1 &&
                (s_idxs[tid] == -1 ||
                 s_vals[other] > s_vals[tid] ||
                 (s_vals[other] == s_vals[tid] &&
                  s_idxs[other] < s_idxs[tid]))) {

                s_vals[tid] = s_vals[other];
                s_idxs[tid] = s_idxs[other];
            }
        }

        __syncthreads();
    }

    // Thread 0 luu ket qua block
    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    // Write code here
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;

    float best_val = -FLT_MAX;
    int best_idx = -1;

    // Moi thread co the xu ly nhieu block
    for (int i = tid; i < num_blocks;
         i += blockDim.x) {

        float val = block_vals[i];
        int idx = block_idxs[i];

        if (idx != -1 &&
            (best_idx == -1 ||
             val > best_val ||
             (val == best_val && idx < best_idx))) {

            best_val = val;
            best_idx = idx;
        }
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;

    __syncthreads();

    // Reduction lan cuoi
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            int other = tid + stride;

            if (s_idxs[other] != -1 &&
                (s_idxs[tid] == -1 ||
                 s_vals[other] > s_vals[tid] ||
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

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
