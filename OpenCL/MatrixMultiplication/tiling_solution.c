#define CL_TARGET_OPENCL_VERSION 220
#define KERNEL_SOURCE_CODE "matrixMulTiling.cl"
#define UPPER_BOUND 10.0f
#define ACCURACY 0.01
#define PEAK_GFLOPS 1911
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void randomize_matrices(float *, float *, int, int, int);

float generateRandomFloat();

void matrix_mul_straight(float *, float *, float *, int, int, int);

void print_matrix(char *, float *, int, int);

cl_uint fetch_platform_from_vendor(cl_platform_id *, cl_uint, char *);

void print_general_device_info(cl_device_id);

void sequential_execution(float *, float *, float *, int, int, int);

void parallel_execution(int, cl_context, cl_command_queue, cl_kernel, cl_mem, float *, int, int, int);

void verification(float *, float *, int, int);

float diff(float, float);

cl_program fetch_kernel_source_code(const cl_context, const char *);

int main(int argc, char const *argv[])
{

    fflush(stdout);
    int m, n, k, N;
    clock_t start, end;
    if (argc == 5)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        N = atoi(argv[4]);
    }
    else
    {
        printf("\n INVALID NUMBER OF ARGUMENTS \n");
        exit(-1);
    }

    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)calloc(m * n, sizeof(float));
    float *C_straight = (float *)malloc(m * n * sizeof(float));

    randomize_matrices(A, B, m, n, k);

    cl_int err;
    // Query the number of available platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get the number of platforms\n");
        return 1;
    }

    // Get the platform IDs
    cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get platform IDs\n");
        free(platforms);
        return 1;
    }

    cl_uint NVIDIA_platform = fetch_platform_from_vendor(platforms, numPlatforms, "NVIDIA");

    cl_uint numDevices;
    err = clGetDeviceIDs(platforms[NVIDIA_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get the number of devices for NVIDIA platform\n");
    }

    // Get the device IDs
    cl_device_id *devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platforms[NVIDIA_platform], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device IDs for platform NVIDIA\n");
        free(devices);
    }

    // Choose the desired device based on your requirements
    cl_device_id chosenDevice = devices[0];

    // Create a context
    cl_context context = clCreateContext(NULL, 1, &chosenDevice, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create context\n");
        free(devices);
    }

    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, chosenDevice, 0, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create command queue\n");
        clReleaseContext(context);
        free(devices);
    }

    // Create input/output buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, m * k * sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, k * n * sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, m * n * sizeof(float), NULL, NULL);

    // Transfer data to devices buffers
    clEnqueueWriteBuffer(commandQueue, bufA, CL_TRUE, 0, m * k * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, bufB, CL_TRUE, 0, k * n * sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, bufC, CL_TRUE, 0, m * n * sizeof(float), C, 0, NULL, NULL);

    // Create program with source code
    cl_program program = fetch_kernel_source_code(context, KERNEL_SOURCE_CODE);

    // Build program
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create your kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMulNaive", &err);
    if (err != CL_SUCCESS)
    {
        printf("\nKERNEL IS NOT BUILT\n");
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(int), &m);
    clSetKernelArg(kernel, 1, sizeof(int), &n);
    clSetKernelArg(kernel, 2, sizeof(int), &k);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bufC);

    // Device info and randomize matrices
    print_general_device_info(chosenDevice);
    //---sequential execution---
    sequential_execution(A, B, C_straight, m, n, k);

    //---parallel execution---
    double exe_time;
    parallel_execution(N, context, commandQueue, kernel, bufC, C, m, n, k);

    //---verification---
    verification(C, C_straight, m, n);

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    free(devices);
    free(platforms);
    free(A);
    free(B);
    free(C);
    free(C_straight);

    return 0;
}

float diff(float a, float b)
{
    return (a - b) < 0 ? -1 * (a - b) : (a - b);
}

void verification(float *C, float *C_straight, int m, int n)
{
    //print_matrix("C",C,m,n);
    printf("\nVERIFICATION TEST STARTED\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (diff(C[i * n + j], C_straight[i * n + j]) > ACCURACY)
            {
                printf("\nVERIFICATION TEST FAILED\n");
                return;
            }
    printf("\nVERIFICATION TEST PASSED\n");
    printf("\n--------------------------------\n");
}

void parallel_execution(int N, cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem bufC, float *C, int m, int n, int k)
{
    clock_t start, end;
    cl_event event;
    double exe_time[N];
    printf("\nSTART OF PARALLEL EXECUTION");
    const size_t global[2] = {m, n};
    const size_t local[2] = {32, 32};
    for (int i = 0; i < N; i++)
    {
        start = clock();
        clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, 0, &event);
        clWaitForEvents(1, &event);
        end = clock();
        exe_time[i] = (((double)end - start) / CLOCKS_PER_SEC);
    }
    // cl_ulong start,end;
    // clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,0);
    // clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,0);
    // exe_time=(double)(end-start)*1.0e-9;
    double avg_time = 0;
    for (int i = 0; i < N; i++)
        avg_time += exe_time[i];
    avg_time /= N;
    double standard_deviation = 0;
    for (int i = 0; i < N; i++)
        standard_deviation += (exe_time[i] - avg_time) * (exe_time[i] - avg_time);
    standard_deviation /= N;
    standard_deviation = sqrt(standard_deviation);
    printf("\nEND OF PARALLEL EXECUTION\n");
    printf("AVERAGE RUNTIME: %lf seconds\n", avg_time);
    printf("STANDARD DEVITAION: %lf seconds\n", standard_deviation);
    printf("\nACHIVED GFLOPS: %lf\n", (2LL * k * m * n) * 1.0 / (1000 * 1000 * 1000)/avg_time);
    printf("\nPEAK GFLOPS: %d\n", PEAK_GFLOPS);
    printf("\nEFFICIENCY: %lf\n",(2LL * k * m * n) * 1.0 / (1000 * 1000 * 1000)/avg_time/PEAK_GFLOPS*100);
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * m * n, C, 0, 0, 0);
    // print_matrix("C",C,m,n);
}

cl_program fetch_kernel_source_code(const cl_context context, const char *source_code_file_name)
{
    FILE *fp;
    char *source_code_str;
    size_t source_size, program_size;

    fp = fopen(source_code_file_name, "rb");
    if (!fp)
    {
        printf("\nERROR: FAILED TO LOAD KERNEL FILE!\n");
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    source_code_str = (char *)malloc(program_size + 1);
    source_code_str[program_size] = '\0';
    fread(source_code_str, sizeof(char), program_size, fp);
    fclose(fp);
    // printf("\n%s\n",source_code_str);
    size_t program_length;
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code_str, 0, &err);
    if (err != CL_SUCCESS)
        printf("\nPROGRAM NOT BUILT\n");
    return program;
}

void print_general_device_info(cl_device_id chosenDevice)
{
    printf("\n--------------------------------\n");
    char deviceName[1024];
    clGetDeviceInfo(chosenDevice, CL_DEVICE_NAME, 1024, deviceName, NULL);
    printf("DEVICE NAME: %s\n", deviceName);
    cl_uint compute_units;
    clGetDeviceInfo(chosenDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, 0);
    printf("MAXIMUM NUMBER OF COMPUTE UNITS: %d\n", compute_units);
    size_t max_workgroup_size;
    clGetDeviceInfo(chosenDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, 0);
    printf("MAXIMUM WORKGROUP SIZE: %d\n", max_workgroup_size);
    cl_ulong global_mem_size;
    clGetDeviceInfo(chosenDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, 0);
    printf("GLOBAL MEMORY SIZE: %llu BYTES\n", global_mem_size);
    cl_ulong local_mem_size;
    clGetDeviceInfo(chosenDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, 0);
    printf("GLOBAL MEMORY SIZE: %llu BYTES\n", local_mem_size);
}

void sequential_execution(float *A, float *B, float *C, int m, int n, int k)
{
    clock_t start, end;
    printf("\nTEST STARTED WITH DIMENSIONS: %d\n", m);
    printf("\nSTART OF SEQUENTIAL EXECUTION");
    start = clock();
    matrix_mul_straight(A, B, C, m, n, k);
    end = clock();
    // print_matrix("C matrix:",C,m,n);
    printf("\nEND OF SEQUENTIAL EXECUTION\nSEQUENTIAL EXECUTION TIME: %lf seconds\n\n", ((double)(end - start) / CLOCKS_PER_SEC));
}

cl_uint fetch_platform_from_vendor(cl_platform_id *platforms, cl_uint numPlatforms, char *vendor_name)
{

    cl_int err;
    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        cl_platform_id platform = platforms[i];
        // Get the platform vendor information
        char vendor[128];
        err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to get platform vendor information for platform %d\n", i);
            continue;
        }

        if (strstr(vendor, vendor_name) != NULL)
            return i;
    }

    return -1;
}

void print_matrix(char *matrix_name, float *matrix, int rows, int columns)
{

    printf("\n%s\n", matrix_name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
            printf("%f ", matrix[i * columns + j]);
        printf("\n");
    }
}

void matrix_mul_straight(float *A, float *B, float *C, int m, int n, int k)
{

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            float result = 0.0f;
            for (int r = 0; r < k; r++)
                result += A[i * k + r] * B[r * n + j];
            C[i * n + j] = result;
        }
}

void randomize_matrices(float *A, float *B, int m, int n, int k)
{
    srand(time(NULL));

    for (int i = 0; i < m * k; ++i)
        A[i] = generateRandomFloat();
    for (int i = 0; i < k * n; ++i)
        B[i] = generateRandomFloat();
}

// Function to generate a random float value between 0 and 10
float generateRandomFloat()
{
    return ((float)rand() / (float)RAND_MAX) * UPPER_BOUND;
}