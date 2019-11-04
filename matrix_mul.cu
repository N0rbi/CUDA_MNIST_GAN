#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


using namespace std;

#define EPOCHS 10
#define BATCH_SIZE 1
#define BLOCK_SIZE 8

// MNIST SOURCE: http://yann.lecun.com/exdb/mnist/
// READING MNIST DATA: http://eric-yuan.me/cpp-read-mnist/
// GLOROT INIT: https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/
// MATRIX MULTIPLICATION: https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/matrix_mul/matrix_mul/matrix_mul.cu
//__global__
//void matMulKernel(double *batch, double *bias, double *weight, double *result, int N, int K, int M) {
//    int row = blockIdx.x * blockDim.x + threadIdx.x;
//    int col = blockIdx.y * blockDim.y + threadIdx.y;
//
//    double product_val = 0;
//    // Boundary protection
//    if ((row < N) && (col < M)) {
//        for (int k = 0; k < K; k++) {
//            product_val += batch[row*M+k]*weight[k*M+N];
//        }
//        // Assign result
//        result[row * M + col] = product_val; //todo: bias
//    }
//}

__global__ void matrixMultiplicationKernel2(double* A, double* B, double* C, int N, int K, int M) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < M) {
        for (int i = 0; i < K; i++) {
            tmpSum += A[ROW * K + i] * B[i * M + COL];
        }
    }
    C[ROW * M + COL] = tmpSum;
}

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

int main() {
    const int N=2, K=3, M=5;
    double mat1[N*K] = {2,2,3,  3,4,4};
    double mat2[K*M] = {5,5,5,5,5,  5,5,5,5,5,  5,5,5,5,5};
    double* mat3 = new double[N*M];
    cout << "Read complete" << endl;


    double *d_mat1, *d_mat2, *d_mat3;
    cudaMalloc((void**)&d_mat1, N * K * sizeof(double));
    cudaMalloc((void**)&d_mat2, K * M * sizeof(double));
    cudaMalloc((void**)&d_mat3, N * M * sizeof(double));

    cudaMemcpy(d_mat1, mat1, N * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, K * M * sizeof(double), cudaMemcpyHostToDevice);

    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//	// double *batch, double *bias, double *weight, double *result, int N, int M, int K
//    matrixMultiplicationKernel<<<grid,threads>>> (d_mat1, d_mat2, d_mat3, size);
    matrixMultiplicationKernel2<<<dimGrid, dimBlock>>> (d_mat1, d_mat2, d_mat3, N, K, M);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    cudaMemcpy(mat3, d_mat3, N * M * sizeof(double), cudaMemcpyDeviceToHost);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    cudaDeviceSynchronize();

    for (int i=0; i<N*K; i++) {
        if (i % K == 0) {
            cout << "\t";
        }

        cout << mat1[i] << " ";
    }
    cout << "\n";

    for (int i=0; i<K*M; i++) {
        if (i % M == 0) {
            cout << "\t";
        }

        cout << mat2[i] << " ";
    }
    cout << "\n";

    for (int i=0; i<N*M; i++) {
        if (i % M == 0) {
            cout << "\t";
        }

        cout << mat3[i] << " ";
    }
    cout << "\n";


    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
    return 0;
}
