#include <cuda.h>
#include <stdio.h>

__global__ void add( int a, int b, int* c )
{
    *c = a + b;

    return;
}

int main(int argc, char** argv)
{
    int c;
    int* dev_c;

    cudaMalloc((void**)&dev_c, sizeof(int) );

    add<<<1,1>>>(1, 2, dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("a + b = %d\n", c);

    cudaFree(dev_c);

    return 0;
}