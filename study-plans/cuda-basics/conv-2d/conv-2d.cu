#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    // Write code here
    // Step 1: Output dimensions
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Step 2: Global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Step 3: Boundary check
    if (x >= outW || y >= outH) {
        return;
    }

    // Step 4: Compute convolution
    float sum = 0.0f;

    for (int i = 0; i < kH; i++) {
        for (int j = 0; j < kW; j++) {
            float input_val =
                input[(y + i) * W + (x + j)];

            float kernel_val =
                kernel[i * kW + j];

            sum += input_val * kernel_val;
        }
    }

    // Step 5: Write output
    output[y * outW + x] = sum;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}
