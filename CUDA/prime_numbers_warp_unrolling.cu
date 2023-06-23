
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/device_launch_parameters.h"
#define NUM_OF_GPU_THREADS 1024
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void kernel_func(unsigned int *results, int limit)
{

    __shared__ int sdata[NUM_OF_GPU_THREADS];
    int tid = threadIdx.x;
    sdata[tid] = 0; // WITHOUT THIS NO CORRECT VALUE

    // calucaltion here ...
    int number = 3 + (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (number <= limit)
    {
        int prime = 1;
        for (int j = 3; j < number; j += 2)
        {
            if (number % j == 0)
                prime = 0;
        }
        sdata[tid] = prime;
    }
    // sync for threads inside the block
    __syncthreads();
    // warp unrolling
    if (blockDim.x >= 1024 && tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int *data = sdata;
        data[tid] += data[tid + 32];
        data[tid] += data[tid + 16];
        data[tid] += data[tid + 8];
        data[tid] += data[tid + 4];
        data[tid] += data[tid + 2];
        data[tid] += data[tid + 1];
    }

    if (tid == 0)
        results[blockIdx.x] = sdata[0];
}

double cpu_time(void)
{
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}

int prime_number(int n)
{
    int i;
    int j;
    int prime;
    int total;

    total = 0;

    for (i = 2; i <= n; i++)
    {
        prime = 1;
        for (j = 2; j < i; j++)
        {
            if ((i % j) == 0)
            {
                prime = 0;
                break;
            }
        }
        total = total + prime;
    }
    return total;
}

void timestamp(void)
{
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    size_t len;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    printf("%s\n", time_buffer);

    return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor, unsigned *verification_results);

void test_parallel(int n_lo, int n_hi, int n_factor, unsigned *verification_results);

void verification(int n_lo, int n_hi, int n_factor, unsigned *A, unsigned *B);

int main(int argc, char *argv[])
{
    int n_factor;
    int n_hi;
    int n_lo;

    timestamp();
    printf("\n");
    printf("PRIME TEST - sequential execution\n");

    if (argc != 4)
    {
        n_lo = 1;
        n_hi = 131072;
        n_factor = 2;
    }
    else
    {
        n_lo = atoi(argv[1]);
        n_hi = atoi(argv[2]);
        n_factor = atoi(argv[3]);
    }

    unsigned *cpu_reults = (unsigned *)malloc(sizeof(unsigned) * ((n_hi - n_lo) / n_factor));
    unsigned *gpu_reults = (unsigned *)malloc(sizeof(unsigned) * ((n_hi - n_lo) / n_factor));
    test(n_lo, n_hi, n_factor, cpu_reults);

    printf("\n");
    printf("PRIME TEST - sequential execution\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    printf("PRIME TEST - parallel execution\n");

    test_parallel(n_lo, n_hi, n_factor, gpu_reults);

    printf("\n");
    printf("PRIME TEST - parallel execution\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    verification(n_lo, n_hi, n_factor, cpu_reults, gpu_reults);
    timestamp();

    return 0;
}

void test(int n_lo, int n_hi, int n_factor, unsigned *verification_results)
{
    int i = 0;
    int n;
    int primes;
    double ctime;

    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");

    n = n_lo;

    while (n <= n_hi)
    {
        ctime = cpu_time();

        primes = prime_number(n);

        ctime = cpu_time() - ctime;

        verification_results[i++] = primes;

        printf("  %8d  %8d  %14f\n", n, primes, ctime);
        n = n * n_factor;
    }

    return;
}

void test_parallel(int n_lo, int n_hi, int n_factor, unsigned *verification_results)
{
    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");

    int n = n_lo;
    int number_of_primes;
    int i = 0;

    while (n <= n_hi)
    {
        float elapsed_time = 0;
        cudaEvent_t start, stop;
        // dummy call to create the cuda context
        // cudaDeviceSynchronize();
        number_of_primes = n > 1 ? 1 : 0;
        unsigned number_of_blocks = (n / 2) / NUM_OF_GPU_THREADS + (((n / 2) % NUM_OF_GPU_THREADS) != 0 ? 1 : 0);
        unsigned *gpu_results, *results;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        results = (unsigned *)malloc(sizeof(unsigned) * number_of_blocks);
        cudaMalloc((void **)&gpu_results, sizeof(unsigned) * number_of_blocks);

        // kernel call here
        kernel_func<<<number_of_blocks, NUM_OF_GPU_THREADS>>>(gpu_results, n);

        cudaMemcpy(results, gpu_results, number_of_blocks * sizeof(unsigned), cudaMemcpyDeviceToHost);

        for (int i = 0; i < number_of_blocks; i++)
            number_of_primes += results[i];

        // cleanup
        cudaFree(gpu_results);
        free(results);
        // cudaEventRecordWithFlags(stop,0)
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("  %8d  %8d  %14f\n", n, number_of_primes, elapsed_time*1e-3);
        n *= n_factor;
        verification_results[i++] = number_of_primes;
    }
}

void verification(int n_lo, int n_hi, int n_factor, unsigned *A, unsigned *B)
{
    int i = 0;
    while (n_lo <= n_hi)
    {
        if (A[i] != B[i])
        {
            printf("\nVERIFICATION FAILED!\n");
            return;
        }
        i++;
        n_lo *= n_factor;
    }
    printf("\nVERIFICATION PASSED!\n");
}