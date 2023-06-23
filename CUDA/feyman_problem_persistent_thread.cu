#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/device_launch_parameters.h"
#include "/usr/local/cuda/include/curand_kernel.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define ACCURACY 0.01f
#define NUM_OF_GPU_THREADS 1024

int count_of_SM_chips = -1;

// N_inside = 364 and Sm chips = 28 --> 13 works per thread

__device__ __host__ float potential(float a, float b, float c, float x, float y, float z)
{
  return 2.0 * (powf(x / a / a, 2) + powf(y / b / b, 2) + powf(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

// size =  n_inside * 3
// num_works = n_inside / count_of_SM_chips + ( n_inside % count_of_SM_chips != 0 )
__global__ void feynman(const int N, float *params, float *results, const int a, const int b, const int c, const float h, const float stepsz, const int size, const int num_works, const int count_of_SM_chips)
{

  curandState rand;
  curand_init(123456789, threadIdx.x, 0, &rand);

  for (int i = 0; i < num_works; i++)
  {
    float x = params[(i * count_of_SM_chips + blockIdx.x) * 3 + 0];
    float y = params[(i * count_of_SM_chips + blockIdx.x) * 3 + 1];
    float z = params[(i * count_of_SM_chips + blockIdx.x) * 3 + 2];
    float wt = 0;
    if (threadIdx.x == 0)
    {
      results[blockIdx.x + i * count_of_SM_chips] = 0;
    }
    for (int trial = threadIdx.x; trial < N; trial += blockDim.x)
    {
      float x1 = x;
      float x2 = y;
      float x3 = z;
      float w = 1.0;
      float chk = 0.0;
      while (chk < 1.0)
      {
        float ut = curand_uniform(&rand);
        float us;
        float dx;
        if (ut < 1.0 / 3.0)
        {
          us = curand_uniform(&rand) - 0.5;
          if (us < 0.0)
            dx = -stepsz;
          else
            dx = stepsz;
        }
        else
          dx = 0.0;

        float dy;
        ut = curand_uniform(&rand);
        if (ut < 1.0 / 3.0)
        {
          us = curand_uniform(&rand) - 0.5;
          if (us < 0.0)
            dy = -stepsz;
          else
            dy = stepsz;
        }
        else
          dy = 0.0;

        float dz;
        ut = curand_uniform(&rand);
        if (ut < 1.0 / 3.0)
        {
          us = curand_uniform(&rand) - 0.5;
          if (us < 0.0)
            dz = -stepsz;
          else
            dz = stepsz;
        }
        else
          dz = 0.0;

        float vs = potential(a, b, c, x1, x2, x3);
        x1 += dx;
        x2 += dy;
        x3 += dz;

        float vh = potential(a, b, c, x1, x2, x3);
        float we = (1.0 - h * vs) * w;
        w = w - 0.5 * h * (vh * we + vs * w);
        chk = powf(x1 / a, 2) + powf(x2 / b, 2) + powf(x3 / c, 2);
      }
      wt += w;
    }

    atomicAdd(&(results[blockIdx.x + i * count_of_SM_chips]), wt);

    //potencijalni problem da ostale krenu da rade dalje blokove a da 0-ta nit dodje i zajebe to tako sto ponisti
    //izgleda da to jeste bio problem pustam izvrsavanje svega
    //
    __syncthreads();

  }
}

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

float seq_exe(const int N, float *elapsed_time);

float par_exe(const int N, float *elapsed_time);

// print na stdout upotrebiti u validaciji paralelnog resenja
int main(int arc, char **argv)
{
  int N = atoi(argv[1]);

  timestamp();
  int devCount;
  cudaGetDeviceCount(&devCount);
  printf("\n----------------\n");
  printf("\nCUDA Device Query...\n");
  printf("There are %d CUDA devices.\n", devCount);
  // Iterate through devices
  for (int i = 0; i < devCount; ++i)
  {
    // Get device properties
    printf("\nCUDA Device #%d\n", i);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    printf("Count of SMchips on GPU: %d \n\n", devProp.multiProcessorCount);
    count_of_SM_chips = devProp.multiProcessorCount;
    printf("\n----------------\n");
  }

  float seq_exe_time;
  float seq_ret = seq_exe(N, &seq_exe_time);
  printf("\nSEQUENTIAL EXECUTION:\n");
  printf("RMS absolute error in solution = %e\n", seq_ret);
  printf("Sequential execution elapsed time: %10f\n\n", seq_exe_time);

  float par_exe_time;
  float par_ret = par_exe(N, &par_exe_time);
  printf("PARALLEL EXECUTION:\n");
  printf("RMS absolute error in solution = %e\n", par_ret);
  printf("Sequential execution elapsed time: %10f\n\n", par_exe_time * 1e-3);

  if (seq_ret - par_ret <= ACCURACY && seq_ret - par_ret >= -ACCURACY)
    printf("VERIFICATION PASSED :)\n\n");
  else
    printf("VERIFICATION FAILED!\n\n");

  timestamp();
}

float seq_exe(const int N, float *elapsed_time)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  int dim = 3;
  double dx;
  double dy;
  double dz;
  double err;
  double h = 0.001;
  int i;
  int j;
  int k;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps;
  int steps_ave;
  int trial;
  double us;
  double ut;
  double vh;
  double vs;
  double x;
  double x1;
  double x2;
  double x3;
  double y;
  double w;
  double w_exact;
  double we;
  double wt;
  double z;

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;
  clock_t start = clock();
  for (i = 1; i <= ni; i++)
  {
    x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);

    for (j = 1; j <= nj; j++)
    {
      y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);

      for (k = 1; k <= nk; k++)
      {
        z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

          continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;
        for (trial = 0; trial < N; trial++)
        {
          x1 = x;
          x2 = y;
          x3 = z;
          w = 1.0;
          chk = 0.0;
          while (chk < 1.0)
          {
            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dx = -stepsz;
              else
                dx = stepsz;
            }
            else
              dx = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dy = -stepsz;
              else
                dy = stepsz;
            }
            else
              dy = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dz = -stepsz;
              else
                dz = stepsz;
            }
            else
              dz = 0.0;

            vs = potential(a, b, c, x1, x2, x3);
            x1 = x1 + dx;
            x2 = x2 + dy;
            x3 = x3 + dz;

            steps++;

            vh = potential(a, b, c, x1, x2, x3);

            we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = pow(x1 / a, 2) + pow(x2 / b, 2) + pow(x3 / c, 2);
          }
          wt = wt + w;
        }
        wt = wt / (double)(N);
        steps_ave = steps / (double)(N);

        err = err + pow(w_exact - wt, 2);

        // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
        //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
      }
    }
  }
  err = sqrt(err / (double)(n_inside));

  clock_t end = clock();

  *elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

  return err;
}

float par_exe(const int N, float *elapsed_time)
{
  float a = 3.0;
  float b = 2.0;
  float c = 1.0;
  int dim = 3;
  float h = 0.001;
  int n_inside = 0;
  int ni;
  int nj;
  int nk;
  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  float coordinates_arr[ni * nj * nk * 3];
  float results_arr[ni * nj * nk];
  float w_exact_arr[ni * nj * nk];

  float stepsz = sqrt((float)dim * h);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 1; i <= ni; i++)
  {
    for (int j = 1; j <= nj; j++)
    {
      for (int k = 1; k <= nk; k++)
      {
        float x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);
        float y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);
        float z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        float chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          continue;
        }

        coordinates_arr[n_inside * 3 + 0] = x;
        coordinates_arr[n_inside * 3 + 1] = y;
        coordinates_arr[n_inside * 3 + 2] = z;
        w_exact_arr[n_inside] = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);
        ++n_inside;
      }
    }
  }

  int num_works = n_inside / count_of_SM_chips + (n_inside % count_of_SM_chips != 0);
  printf("\nN_INSIDE: %d\n", n_inside);
  printf("Count of SM chips: %d\n", count_of_SM_chips);
  printf("N_WORKS: %d\n\n", num_works);

  float *gpu_coordinates_arr;
  float *gpu_results_arr;

  cudaMalloc(&gpu_coordinates_arr, n_inside * 3 * sizeof(float));
  cudaMalloc(&gpu_results_arr, n_inside * sizeof(float));
  cudaMemcpy(gpu_coordinates_arr, coordinates_arr, sizeof(float) * 3 * n_inside, cudaMemcpyHostToDevice);

  // CALL KERNEL HERE
  // feynman<<<n_inside, NUM_OF_GPU_THREADS>>>(N, gpu_coordinates_arr, gpu_results_arr, a, b, c, h, stepsz, n_inside * 3, num_works, count_of_SM_chips);
  feynman<<<count_of_SM_chips, NUM_OF_GPU_THREADS>>>(N, gpu_coordinates_arr, gpu_results_arr, a, b, c, h, stepsz, n_inside * 3, num_works, count_of_SM_chips);

  cudaMemcpy(results_arr, gpu_results_arr, sizeof(float) * n_inside, cudaMemcpyDeviceToHost);
  cudaFree(gpu_coordinates_arr);
  cudaFree(gpu_results_arr);

  float err = 0.0f;
  for (int i = 0; i < n_inside; i++)
  {
    err += pow(w_exact_arr[i] - (results_arr[i] / (float)(N)), 2);
  }
  err = sqrt(err / (float)(n_inside));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(elapsed_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return err;
}