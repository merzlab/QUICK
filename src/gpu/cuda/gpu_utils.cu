#include "gpu_utils.h"

#include <stdio.h>
#include <assert.h>
#if defined(MPIV_GPU)
  #include <mpi.h>
#endif


/* Safe wrapper around cudaGetDeviceCount
 *
 * count: num. of GPUs on system
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuGetDeviceCount(int * count, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaGetDeviceCount(count);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaGetDeviceCount failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }  
}


/* Safe wrapper around cudaSetDevice
 *
 * device: ID to device to set for execution
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuGetDeviceCount(int device, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaSetDevice(device);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        if (ret == cudaErrorInvalidDevice) {
            fprintf(stderr, "[ERROR] invalid GPU device ID set (%d).\n", device);
        } else if (ret == cudaErrorDeviceAlreadyInUse) {
            fprintf(stderr, "[ERROR] GPU device with specified ID already in use (%d).\n", device);
        }

        fprintf(stderr, "[ERROR] GPU error: cudaSetDevice failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }  
}


/* Safe wrapper around cudaMalloc
 *
 * ptr: pointer to allocated device memory
 * size: reqested allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMalloc(void **ptr, size_t size, const char * const filename,
        int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fprintf(stderr, "[INFO] gpuMalloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            size, line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuMalloc: requesting %zu bytes at line %d in file %.*s\n",
            size, line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif

    ret = cudaMalloc(ptr, size);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMalloc failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }  

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    fprintf(stderr, "[INFO] gpuMalloc: granted memory at address %p at line %d in file %.*s on MPI processor %d\n",
            *ptr, line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuMalloc: granted memory at address %p at line %d in file %.*s\n",
            *ptr, line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif
}


/* Safe wrapper around cudaHostAlloc
 *
 * ptr: pointer to allocated device memory
 * size: reqested allocation size in bytes
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuHostAlloc(void **ptr, size_t size, unsigned int flags, const char * const filename,
        int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fprintf(stderr, "[INFO] gpuHostAlloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            size, line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuHostAlloc: requesting %zu bytes at line %d in file %.*s\n",
            size, line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif

    ret = cudaHostAlloc(ptr, size, flags);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaHostAlloc failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }  

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    fprintf(stderr, "[INFO] gpuHostAlloc: granted memory at address %p with flags %u at line %d in file %.*s on MPI processor %d\n",
            *ptr, flags, line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuHostAlloc: granted memory at address %p with flags %u at line %d in file %.*s\n",
            *ptr, flags, line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif
}


/* Safe wrapper around cudaFree
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuFree(void *ptr, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    if (ptr == NULL)
    {
        fprintf(stderr, "[WARNING] trying to free the already NULL pointer\n");
        fprintf(stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        return;
    }  

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fprintf(stderr, "[INFO] gpuFree: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuFree: freeing ptr at line %d in file %.*s\n",
            line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif

    ret = cudaFree(ptr);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[WARNING] GPU error: cudaFree failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);
        fprintf(stderr, "  [INFO] Memory address: %ld\n", 
                (long int) ptr);

        return;
    }  
}


/* Safe wrapper around cudaFreeHost
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuFreeHost(void * ptr, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    if (ptr == NULL)
    {
        fprintf(stderr, "[WARNING] trying to free the already NULL pointer\n");
        fprintf(stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        return;
    }  

#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fprintf(stderr, "[INFO] gpuFreeHost: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank);
  #else
    fprintf(stderr, "[INFO] gpuFreeHost: freeing ptr at line %d in file %.*s\n",
            line, (int) strlen(filename), filename);
  #endif
    fflush(stderr);
#endif

    ret = cudaFreeHost(ptr);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[WARNING] GPU error: cudaFreeHost failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);
        fprintf(stderr, "  [INFO] Memory address: %ld\n", 
                (long int) ptr);

        return;
    }  
}


/* Safe wrapper around cudaMemset
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemset(void *ptr, int data, size_t count,
        const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaMemset(ptr, data, count);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMemset failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaMemsetAsync
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * s: GPU stream to perform memset in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemsetAsync(void *ptr, int data, size_t count,
        cudaStream_t s, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaMemsetAsync(ptr, data, count, s);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMemsetAsync failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to allocated device memory (if required)
 * cur_size: current allocation size in bytes
 * new_size: reqested new allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuCheckMalloc(void **ptr, size_t *cur_size, size_t new_size,
        const char * const filename, int line)
{
    assert(new_size > 0 || *cur_size > 0);

    if (new_size > *cur_size)
    {
#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
        int rank;
    
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
        fprintf(stderr, "[INFO] gpuCheckMalloc: requesting %zu bytes (%zu currently allocated) at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, line, (int) strlen(filename), filename, rank);
  #else
        fprintf(stderr, "[INFO] gpuCheckMalloc: requesting %zu bytes (%zu currently allocated) at line %d in file %.*s\n",
                new_size, *cur_size, line, (int) strlen(filename), filename);
  #endif
        fflush(stderr);
#endif

        if (*cur_size != 0)
        {
            _gpuFree(*ptr, filename, line);
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate by 20% to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = (size_t) ceil(new_size * 1.2);
        _gpuMalloc(ptr, *cur_size, filename, line);
    }
}


/* Safe wrapper around cudaMemcpy
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: GPU enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemcpy(void * const dest, void const * const src, size_t count,
        cudaMemcpyKind dir, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaMemcpy(dest, src, count, dir);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMemcpy failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaMemcpyAsync
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: GPU enum specifying address types for dest and src
 * s: GPU stream to perform the copy in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemcpyAsync(void * const dest, void const * const src, size_t count,
        cudaMemcpyKind dir, cudaStream_t s, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaMemcpyAsync(dest, src, count, dir, s);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMemcpyAsync failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaMemcpyToSymbol
 *
 * symbol: device symbol address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemcpyToSymbol(void const * const symbol, void const * const src, size_t count,
        const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaMemcpyToSymbol(symbol, src, count);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaMemcpyToSymbol failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaHostGetDevicePointer
 *
 * pdev: returned pointer from mapped memory
 * phost: request host pointer mapping
 * flags: flags for extensions (must be 0 for now)
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuHostGetDevicePointer(void ** pdev, void * const phost, unsigned int flags,
        const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaHostGetDevicePointer(pdev, phost, flags);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaHostGetDevicePointer failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around check first and reallocate if needed routine for pinned memory:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void _gpuHostAllocCheck(void **ptr, size_t *cur_size, size_t new_size,
        unsigned int flags, int over_alloc, double over_alloc_factor,
        const char * const filename, int line)
{
    assert(new_size > 0 || *cur_size > 0);

    if (new_size > *cur_size)
    {
#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
        int rank;
    
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
        fprintf(stderr, "[INFO] gpuHostAllocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename, rank);
  #else
        fprintf(stderr, "[INFO] gpuHostAllocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename);
  #endif
        fflush(stderr);
#endif

        if (*cur_size != 0)
        {
            _gpuFreeHost(*ptr, filename, line);
        }

        if (over_alloc == 1)
        {
            *cur_size = (int) ceil(new_size * over_alloc_factor);
        }
        else
        {
            *cur_size = new_size;
        }

        _gpuHostAlloc(ptr, *cur_size, flags, filename, line);
    }
}


/* Safe wrapper around check first and reallocate if needed
 * while preserving current memory contents routine for pinned memory:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void _gpuHostReallocCheck(void **ptr, size_t *cur_size, size_t new_size,
        unsigned int flags, int over_alloc, double over_alloc_factor,
        const char * const filename, int line)
{
    void *old_ptr;
    size_t old_ptr_size;

    assert(new_size > 0 || *cur_size > 0);

    if (new_size > *cur_size)
    {
#if defined(DEBUG_FOCUS)
  #if defined(MPIV_GPU)
        int rank;
    
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
        fprintf(stderr, "[INFO] gpuHostReallocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename, rank);
  #else
        fprintf(stderr, "[INFO] gpuHostReallocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename);
  #endif
        fflush(stderr);
#endif

        old_ptr = *ptr;
        old_ptr_size = *cur_size;
        *ptr = NULL;

        if (over_alloc == 1)
        {
            *cur_size = (int) ceil(new_size * over_alloc_factor);
        }
        else
        {
            *cur_size = new_size;
        }

        _gpuHostAlloc(ptr, *cur_size, flags, filename, line);

        if (old_ptr_size != 0)
        {
            _gpuMemcpy(*ptr, old_ptr, old_ptr_size, cudaMemcpyHostToHost,
                    __FILE__, __LINE__);

            _gpuFreeHost(old_ptr, filename, line);
        }
    }
}


/* Safe wrapper around cudaEventCreate
 *
 * event: created GPU event
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventCreate(cudaEvent_t * event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaEventCreate(event);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaEventCreate failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaEventDestroy
 *
 * event: GPU event to destroy
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventDestroy(cudaEvent_t event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaEventDestroy(event);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaEventDestroy failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaEventElapsedTime
 *
 * time: elapsed time between GPU events (in ms)
 * start, end: GPU events to compute elapsed time for
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventElapsedTime(float * time, cudaEvent_t start, cudaEvent_t end, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaEventElapsedTime(time, start, end);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaEventElapsedTime failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaEventRecord
 *
 * event: GPU event to record
 * stream: GPU stream in which to record event
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventRecord(cudaEvent_t event, cudaStream_t stream, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaEventRecord(event, stream);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaEventRecord failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaEventSynchronize
 *
 * event: GPU event to record
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventSynchronize(cudaEvent_t event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaEventSynchronize(event);

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaEventSynchronize failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}


/* Safe wrapper around cudaDeviceSynchronize
 *
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuDeviceSynchronize(const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    cudaError_t ret;

    ret = cudaDeviceSynchronize();

    if (ret != cudaSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = cudaGetErrorString(ret);

        fprintf(stderr, "[ERROR] GPU error: cudaDeviceSynchronize failure\n");
#if defined(MPIV_GPU)
        fprintf(stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank);
#else
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
#endif
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

#if defined(MPIV_GPU)
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }
}
