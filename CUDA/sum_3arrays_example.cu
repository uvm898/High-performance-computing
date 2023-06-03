#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <stdio.h>

// for random init
#include <time.h>
#include <stdlib.h>

// for memset
#include <cstring>

// for  cuda run time error handling
#include "cuda_common.cuh"

__global__ void sum_array_gpu(int *a, int *b, int *c,int *d, int size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d[gid] = a[gid] + b[gid] + c[gid];
}

void sum_array_cpu(int *a, int *b, int *c,int *d, int size)
{
    for (int i = 0; i < size; i++)
        d[i] = a[i] + b[i] + c[i];
}

void array_comparison(int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
        if (a[i] != b[i])
        {
            printf("Arrays are not the same!\n");
            return;
        }
    printf("Arrays are the same!\n");
}

int main(int argc, char const *argv[])
{

    int size = 1<<25;
    int NUM_BYTES = size * sizeof(int);
    // host arrays
    int *h_a = (int *)malloc(NUM_BYTES);
    int *h_b = (int *)malloc(NUM_BYTES);
    int *h_c = (int *)malloc(NUM_BYTES);
    int *h_d = (int *)malloc(NUM_BYTES);
    int *gpu_results = (int *)malloc(NUM_BYTES);
    memset(gpu_results, 0, NUM_BYTES); // instead we could use calloc
    srand((unsigned)time(NULL));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() && 0xff);
        h_b[i] = (int)(rand() && 0xff);
        h_c[i] = (int)(rand() && 0xff);
    }

    // cpu summation
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c,h_d, size);
    cpu_end = clock();
    // device arrays
    int *d_a, *d_b, *d_c, *d_d;
    gpuErrchk(cudaMalloc((void **)&d_a, NUM_BYTES));
    gpuErrchk(cudaMalloc((void **)&d_b, NUM_BYTES));
    gpuErrchk(cudaMalloc((void **)&d_c, NUM_BYTES));
    gpuErrchk(cudaMalloc((void **)&d_d, NUM_BYTES));

    int block_size = 128;
    dim3 block(block_size);
    dim3 grid(size / block_size + 1);

    clock_t htod_start, htod_end;
    htod_start = clock();
    gpuErrchk(cudaMemcpy(d_a, h_a, NUM_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, NUM_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c, h_c, NUM_BYTES, cudaMemcpyHostToDevice));
    htod_end = clock();

    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c,d_d, size);
    gpuErrchk(cudaDeviceSynchronize());
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    gpuErrchk(cudaMemcpy(gpu_results, d_d, NUM_BYTES, cudaMemcpyDeviceToHost));
    dtoh_end = clock();

    array_comparison(h_d, gpu_results, size);

    printf("Sum array CPU execution time: %4.6f \n",
           (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
    printf("Sum array GPU execution time: %4.6f \n",
           (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
    printf("host to device memory transfer time: %4.6f \n",
           (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
    printf("device to host memory transfer time: %4.6f \n",
           (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
    printf("Sum array total GPU execution time: %lf \n",
           ((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_d));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(gpu_results);
    gpuErrchk(cudaDeviceReset());
    return 0;
}
