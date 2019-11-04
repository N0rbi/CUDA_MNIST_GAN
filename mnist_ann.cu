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
#define BATCH_SIZE 10
#define BLOCK_SIZE 32


// MNIST SOURCE: http://yann.lecun.com/exdb/mnist/
// READING MNIST DATA: http://eric-yuan.me/cpp-read-mnist/
// GLOROT INIT: https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/
// MATRIX MULTIPLICATION: https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/matrix_mul/matrix_mul/matrix_mul.cu
__global__
void matMulKernel(double *batch, double *bias, double *weight, double *result, int N, int M, int K) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	double product_val = 0;
	// Boundary protection
	if ((row < N) && (col < M)) {
		for (int k = 0; k < M; k++) {
			product_val += batch[row*M+k]*weight[k*M+N];
		}
		// Assign result
		result[row * M + col] = product_val; //todo: bias
	}
}

__global__
void reluKernel(double *source) {
    int idx = blockIdx.x;
    source[idx] = max(0.0, source[idx]);
}

__global__
void sigmoidKernel(double *source) {
    int idx = blockIdx.x;
    source[idx] = 1 / (1 + exp(-source[idx]));
}

__global__
void softmaxKernel(double *source, int size) {
    int idx = blockIdx.x;
    if (idx % size == 0) {
        double sum = 0.0;
        for (int i=0; i < size; i++) {
            sum += exp(source[idx+i]);
        }
        for (int i=0; i < size; i++) {
            source[idx+i] = exp(source[idx+i]) / sum;
        }
    }
}

__global__
void softmaxPrimeKernel(double *source, double *gradient, int size) {
//    int idx = blockIdx.x;
//    if (idx % size == 0) {
//        double sum = 0.0;
//        for (int i=0; i < size; i++) {
//
//            sum += exp(source[idx+i]);
//        }
//    }
}

__global__
void onehotKernel(int *source, double *dest, int numFeatures) {
    int idx = blockIdx.x;
    for (int i=0; i<numFeatures; i++) {
        dest[numFeatures*idx + i] = (source[idx] == i) ? 1 : 0;
    }
}

__global__
void divideKernel(double* source, double divideBy) {
    int idx = blockIdx.x;
    source[idx] = source[idx] / divideBy;
}

__global__
void crossEntropy(double* p, double* q, double* loss, int numFeatures) {
    int idx = blockIdx.x;
    loss[idx] = 0;
    for (int i = 0; i < numFeatures; i++) {
        loss[idx] -= p[numFeatures*idx + i] * log(q[numFeatures*idx + i]);
    }
}

__global__
void matrixMultiplicationKernel2(double* A, double* B, double* C, int N, int K, int M) {

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

__global__
void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {

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

inline double* glorotWeightInit(int fan_in, int fan_out) {
    double variance = 2.0 / (fan_in + fan_out);
    double stddev = sqrt(variance);
    boost::mt19937 *rng = new boost::mt19937();
    rng->seed(time(NULL));

    boost::normal_distribution<> distribution(0.0, stddev);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);
    double* weights = new double[fan_in*fan_out];
    for (int i=0; i < fan_in; i++) {
        for (int j=0; j < fan_out; j++) {
            weights[i * fan_out + j] = dist();
        }
    }
    return weights;
}


int ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(const char* filename, double* vec) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i) {
            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    vec[c + r * n_cols + i * n_cols * n_rows] = (float) temp;
                }
            }
        }
    }
}

void read_Mnist_Label(const char* filename, int* labels) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        // int n_rows = 0;
        // int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[i]= (int)temp;
        }
    }
}

int main() {

    const char* data_filename = "data/train-images-idx3-ubyte";
    const char* label_filename = "data/train-labels-idx1-ubyte";

    const char* test_data_filename = "data/t10k-images-idx3-ubyte";
    const char* test_label_filename = "data/t10k-labels-idx1-ubyte";
    const int number_of_images = 60000;
    const int test_number_of_images = 10000;
    int number_of_pixels = 28*28;

    //read MNIST image into double vector
    double* train_vec = new double[number_of_images*number_of_pixels];
    read_Mnist(data_filename, train_vec);
    int* train_vec_labels= new int[number_of_images];
    read_Mnist_Label(label_filename, train_vec_labels);

    // now we have all the images, we create a kernel to normalize it

    double *d_train_vec, *d_test_vec;
    cudaMalloc((void**)&d_train_vec, number_of_images * number_of_pixels * sizeof(double));

    cudaMemcpy(d_train_vec, train_vec, number_of_images * number_of_pixels * sizeof(double), cudaMemcpyHostToDevice);


    divideKernel<<<number_of_images*number_of_pixels, 1>>> (d_train_vec, 255);
    cudaMemcpy(train_vec, d_train_vec, number_of_images * number_of_pixels * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_train_vec);
    // now that we have the normalized image set, we can create a neural net

    // input -> 256 w + 1 b -> relu -> 256 w + 1 b -> relu -> 256 w + 1 b -> relu -> 10 w + 1 b -> sigmoid
    int weight_params[4][2] = {{number_of_pixels, 32}, {32, 32}, {32, 16}, {16, 10}};
    int num_weight_matrices = sizeof(weight_params) / (2*sizeof(int));
    double** weights = new double*[num_weight_matrices];
    double** biases = new double*[num_weight_matrices];

    for (int i = 0; i < num_weight_matrices; i++) {
        weights[i] = glorotWeightInit(weight_params[i][0], weight_params[i][1]);
    }

    double* h_result = new double[BATCH_SIZE * number_of_pixels];
    memcpy(h_result, train_vec, BATCH_SIZE * number_of_pixels * sizeof(double));

    double* d_result;
    for (int i = 0; i < num_weight_matrices; i++) {
        double* d_batch, *d_bias, *d_weight;

        cudaMalloc((void**)&d_batch, BATCH_SIZE * number_of_pixels * sizeof(double));
        cudaMalloc((void**)&d_weight, weight_params[i][0] * weight_params[i][1] * sizeof(double));
        cudaMemcpy(d_batch, h_result, BATCH_SIZE * number_of_pixels * sizeof(double), cudaMemcpyHostToDevice);

        delete [] h_result;
        cudaMalloc((void**)&d_result, BATCH_SIZE * weight_params[i][1] * sizeof(double));
        h_result = new double[BATCH_SIZE * weight_params[i][1]];
//    cudaMemcpy(d_bias, biases, weight_params[0][1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weights[i], weight_params[i][0] * weight_params[i][1] * sizeof(double), cudaMemcpyHostToDevice);

        unsigned int grid_rows = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (weight_params[i][1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        matrixMultiplicationKernel2<<<dimGrid, dimBlock>>>(d_batch, d_weight, d_result, BATCH_SIZE, weight_params[i][0], weight_params[i][1]);

        cout << cudaGetErrorString(cudaGetLastError()) << endl;

        if (i != num_weight_matrices -1) {
            reluKernel<<<BATCH_SIZE * weight_params[i][1], 1>>>(d_result);
        } else {
            softmaxKernel<<<BATCH_SIZE * weight_params[i][1], 1>>>(d_result, BATCH_SIZE);
        }

        cout << cudaGetErrorString(cudaGetLastError()) << endl;

        cudaMemcpy(h_result, d_result, BATCH_SIZE * weight_params[i][1] * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_batch);
//    cudaFree(d_bias);
        cudaFree(d_weight);

    }

    for (int j=0; j < BATCH_SIZE * weight_params[num_weight_matrices-1][1]; j++) {
        if (j % weight_params[num_weight_matrices-1][1] == 0) {
            cout << endl;
        }
        cout << h_result[j] << " ";
    }
    cout << endl;

//    for (int j=0; j < BATCH_SIZE; j++) {
//        cout << train_vec_labels[j] << " ";
//    }
//    cout << endl;

    double* h_onehotLabels, *d_onehotLabels;
    int* d_labels;

    h_onehotLabels = new double[BATCH_SIZE * 10];
    cudaMalloc((void**) &d_onehotLabels, BATCH_SIZE * 10 * sizeof(double));
    cudaMalloc((void**) &d_labels, BATCH_SIZE * sizeof(double));
    cudaMemcpy(d_labels, train_vec_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    onehotKernel<<<BATCH_SIZE, 1>>>(d_labels, d_onehotLabels, 10);

    cudaMemcpy(h_onehotLabels, d_onehotLabels, BATCH_SIZE * 10 * sizeof(double), cudaMemcpyDeviceToHost);

    

    for (int i=0; i < BATCH_SIZE * 10; i++) {
        if (i % 10 == 0) {
            cout << endl;
        }
        cout << h_onehotLabels[i] << " ";
    }

    cout << endl;

    for (int i=0; i < BATCH_SIZE * 10; i++) {
        if (i % 10 == 0) {
            cout << endl;
        }
        cout << h_result[i] << " ";
    }

    cout << endl;


    double * h_losses, *d_losses;

    h_losses = new double[BATCH_SIZE];
    cudaMalloc((void**) &d_losses, BATCH_SIZE * sizeof(double));
    crossEntropy<<<BATCH_SIZE, 1>>>(d_onehotLabels, d_result, d_losses, 10);

    cudaMemcpy(h_losses, d_losses, 10 * sizeof(double), cudaMemcpyDeviceToHost);

    double loss = 0.0;
    for (int i = 0; i < 10; i++) {
        loss += h_losses[i];
    }

    loss /= BATCH_SIZE; // LOSS OF THE BATCH


    cout << endl;

    cudaFree(d_result);
    cudaFree(d_onehotLabels);
    cudaFree(d_labels);
    delete [] h_onehotLabels;
    delete [] h_result;

//    double* test_vec = new double[test_number_of_images*number_of_pixels];
//    read_Mnist(test_data_filename, test_vec);
//    //read MNIST label into double vector
//    int* test_vec_labels = new int[test_number_of_images];
//    read_Mnist_Label(test_label_filename, test_vec_labels);
//    cudaMalloc((void**)&d_test_vec, test_number_of_images * number_of_pixels * sizeof(double));
//    cudaMemcpy(d_test_vec, test_vec, test_number_of_images * number_of_pixels * sizeof(double), cudaMemcpyHostToDevice);
//    divideKernel<<<test_number_of_images, 1>>> (d_test_vec);
//    cudaMemcpy(test_vec, d_test_vec, test_number_of_images * number_of_pixels * sizeof(double), cudaMemcpyDeviceToHost);
//    cout << "got here";
//    cudaFree(d_test_vec);


//    /*
//    for (int j=0; j < weight_params[0][1]; j++) {
//        cout << biases[0][j] << ";";
//    }
//
//    cout << endl;
//    cout << endl;
//
//    for (int i=0; i< weight_params[0][0]; i++) {
//        for (int j=0; j < weight_params[0][1]; j++) {
//            cout << (weights[0][i * weight_params[0][1] + j]) << ";";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    */
//    for (int j=0; j < weight_params[0][1]; j++) {
//        cout << (h_result[j]) << ";";
//    }
//
//    cout << endl;
//
//    delete[] h_result;
//
//    // TODO: forward pass
//
//    // softmax over the outputs
//
//    // calculate loss
//
//    // calculate gradient
//
//    // apply SGD
//
//    // rinse and repeat
//
//    delete[] train_vec;
//    delete[] test_vec;
//    delete[] train_vec_labels;
//    delete[] test_vec_labels;
//    for (int i=0; i < num_weight_matrices; i++) {
//        delete[] weights[i];
//        delete[] biases[i];
//    }
//    delete[] biases;
//    delete[] weights;
//    return 0;
}
