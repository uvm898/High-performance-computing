#include <Python.h>
#include <immintrin.h>
#define ALIGNMENT 16
void core_multyply(float *coord, __m128* t_matrix_rows)
{
    float vector[4] __attribute__((aligned(16)))={coord[0],coord[1],coord[2],1.0}; //attribute specific to GNU compiler
    //float* vector = (float*)_mm_malloc(4*sizeof(float),ALIGNMENT); THIS ALSO MAKES THE SAME PROBLEM is it smth with CACHES
    // vector[0]=coord[0];
    // vector[1]=coord[1];
    // vector[2]=coord[2];
    // vector[3]=1.0;
    __m128 vec = _mm_load_ps(&vector);
    __m128 res[4];
    for( int i=0;i<4;++i){
        res[i]= _mm_mul_ps(t_matrix_rows[i],vec);
    }
    __m128 temp1 = _mm_add_ps(res[0],res[1]);
    __m128 temp2 = _mm_add_ps(res[2],res[3]);
    __m128 temp3 = _mm_add_ps(temp1,temp2);

    _mm_store_ps(vector,temp3);
    coord[0]=vector[0];
    coord[1]=vector[1];
    coord[2]=vector[2];
    //_mm_free(vector);
}
// _mm_prefetch((const char *)&vector_in[0], _MM_HINT_T0);
// The second argument to _mm_prefetch is a hint that tells the CPU how aggressively to prefetch the data.1
//  In this case, we use the _MM_HINT_T0 hint, which tells the CPU to prefetch the data into the L1 cache with low to moderate priority.
//   This is appropriate when we are processing a large number of vectors and want to avoid cache thrashing.

void transform_vertices_simd(float *coord, size_t num_of_coord, const float *t_matrix)
{
    // loading 128 bits of t_matrix into 4 diff SSE regs - one row
    //load all of them
    // working with SSE -> 128b width registers x 16 register
    float rows[4][4] __attribute__((aligned(16)))={
        {t_matrix[0],t_matrix[4],t_matrix[8],t_matrix[12]},
        {t_matrix[1],t_matrix[5],t_matrix[9],t_matrix[13]},
        {t_matrix[2],t_matrix[6],t_matrix[10],t_matrix[14]},
        {t_matrix[3],t_matrix[7],t_matrix[11],t_matrix[15]}
    };
    __m128 t_matrix_rows[4];
    for (int i = 0; i < 4; ++i)
    {
        t_matrix_rows[i] = _mm_load_ps(&rows[i]);
    }
    // POSSIBLE OPTIMIZATION WITH PREFETCHING IN CACHE
    //float* vector[4]; if we try to declare it here it makes problem why!!??
    for (int i = 0;i<num_of_coord/3;i++){
        core_multyply(coord+i*3,t_matrix_rows);
    }
}

static PyObject *transform_vertices(PyObject *self, PyObject *args)
{
    PyObject *vtxarray_obj;
    PyObject *mat4_obj;
    Py_buffer vtxarray_buf;
    Py_buffer mat4_buf;
    // Try to read two arguments: the vertex array and the transformation matrix
    if (!PyArg_ParseTuple(args, "OO", &vtxarray_obj, &mat4_obj))
        return NULL;
    // Try to read the objects as contiguous buffers
    if (PyObject_GetBuffer(vtxarray_obj, &vtxarray_buf,
                           PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1 ||
        PyObject_GetBuffer(mat4_obj, &mat4_buf,
                           PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    // Check that the vertex buffer has one dimension and contains floats ("f")
    if (vtxarray_buf.ndim != 1 || strcmp(vtxarray_buf.format, "f") != 0)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Vertex buffer must be a flat array of floats");
        PyBuffer_Release(&vtxarray_buf);
        return NULL;
    }
    // Check that the transformation matrix buffer has 16 float elements
    if (mat4_buf.ndim != 1 || mat4_buf.shape[0] != 16 ||
        strcmp(mat4_buf.format, "f") != 0)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Transformation matrix must be an array of 16 floats");
        PyBuffer_Release(&mat4_buf);
        return NULL;
    }
    // Call the actual processing function
    transform_vertices_simd(vtxarray_buf.buf, vtxarray_buf.shape[0], mat4_buf.buf);
    // Clean up resources and return to Python code
    PyBuffer_Release(&vtxarray_buf);
    PyBuffer_Release(&mat4_buf);
    Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit_assignment(void)
{
    static PyMethodDef module_methods[] = {{"transform_vertices",
                                            (PyCFunction)transform_vertices,
                                            METH_VARARGS,
                                            "Transform vertex array"},
                                           {NULL}};
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "assignment",
        "This module contains my assignment code",
        -1,
        module_methods};
    return PyModule_Create(&module_def);
}
