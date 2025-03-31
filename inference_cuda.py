import pycuda.driver as cuda
import pycuda.autoinit
import torch
import numpy as np
 
from pycuda.compiler import SourceModule

class NeuralNetworkCUDA:
    def __init__(self, n_input, n_output, hidden_size):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_size = hidden_size
        self.weights_gpu = []
        self.biases_gpu = []

    def load_model(self, weights, biases):
        self.weights_gpu = [cuda.mem_alloc(w.cpu().numpy().nbytes) for w in weights]
        self.biases_gpu = [cuda.mem_alloc(b.cpu().numpy().nbytes) for b in biases]

        for w, w_gpu in zip(weights, self.weights_gpu):
            cuda.memcpy_htod(w_gpu, w.cpu().numpy())

        for b, b_gpu in zip(biases, self.biases_gpu):
            cuda.memcpy_htod(b_gpu, b.cpu().numpy())

    def apply(self, x):
        input_data = x.numpy()
               
        matrix_mul_mod = SourceModule("""
        __global__ void matmul_bias(float * A, float * B, float * C, float *bias, int B, int M, int K) {
            int Row = blockIdx.y * blockDim.y + threadIdx.y;
            int Col = blockIdx.x * blockDim.x + threadIdx.x;

            if ((Row < B) && (Col < K)) {
                float cValue = 0.0f;
                for (int k = 0; k < M; k++) {
                    cValue += A[Row*M + k] * B[k*K + Col];
                }
                C[Row*numCColumns + Col] = fmaxf(0.0f, cValue + bias[col]);
            }
        }
        """)
        
        matmul_bias_relu = matrix_mul_mod.get_function("matmul_bias")

        B = input_data.shape[0]

        input_gpu = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod(input_gpu, input_data)

        for i, (w_gpu, b_gpu) in enumerate(zip(self.weights_gpu, self.biases_gpu)):
            if i == 0:
                K = self.hidden_size
                M = self.n_input
            elif i == len(self.weights_gpu) - 1:
                K = self.n_output
                M = self.hidden_size
            else:
                K = self.hidden_size
                M = self.hidden_size

            output_gpu = cuda.mem_alloc(input_data.nbytes // self.n_input * K)

            cuda.memcpy_htod(input_gpu, input_data)

            block_size = (16, 16, 1)
            grid_size = ((B + block_size[0] - 1) // block_size[0],
                         (M + block_size[1] - 1) // block_size[1],
                         1)

            matmul_bias_relu(input_gpu, w_gpu, output_gpu, b_gpu, np.int32(B),
                        np.int32(M), np.int32(K), block=block_size, grid=grid_size)
            
            input_gpu = output_gpu

        return torch.from_numpy(output_gpu).cpu()
