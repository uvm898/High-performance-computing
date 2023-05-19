__kernel void matrixMulNaive(const int m, const int n, const int k,
                         __global const float *A, __global const float *B,
                         __global float *C) {
                            const int globalRow = get_global_id(0);
                            const int globalCol = get_global_id(1);

                            float result = 0.0f;
                            for(int r=0;r<k;++r)
                                result+=A[globalRow*k+r]*B[r*n+globalCol];
                            
                            C[globalRow*n+globalCol]=result;
                         }