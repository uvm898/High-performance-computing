__kernel void matrixMulNaive(const int m, const int n, const int k,
                             __global const float *A, __global const float *B,
                             __global float *C) {

  const int num_tiles = k / 32;
  __local float Asub[32][32];
  __local float Bsub[32][32];

  const int row = get_local_id(0);
  const int column = get_local_id(1);

  const int global_row = get_global_id(0);
  const int global_column = get_global_id(1);

  float result = 0.0f;

  for (int i = 0; i < num_tiles; i++) {

    Asub[row][column] = A[global_row * k  + column + i*32];
    Bsub[row][column] = B[(i * 32 + row) * n + global_column];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int r=0;r<32;r++){
        result+=Asub[row][r]*Bsub[r][column];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    

  }

  C[global_row*n+global_column]=result;
}