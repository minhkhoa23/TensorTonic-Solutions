#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    // Write code here
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Step 1: Mỗi thread tính tổng cục bộ
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[i];
        sum += x * x;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Step 2: Parallel Reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // Step 3: Ghi tổng cuối cùng
    if (tid == 0) {
        sumv[0] = sdata[0];
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float norm = sqrtf(sumv[0]);

        if (norm > 0.0f) {
            output[i] = input[i] / norm;
        } else {
            output[i] = 0.0f;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
