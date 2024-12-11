#include "gpu_utils.h"

#include <stdio.h>
#include <assert.h>
#if defined(MPIV_GPU)
  #include <mpi.h>
#endif


/* Safe wrapper around hipGetDeviceCount
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
    hipError_t ret;

    ret = hipGetDeviceCount(count);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipGetDeviceCount failure\n");
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


/* Safe wrapper around hipSetDevice
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
    hipError_t ret;

    ret = hipSetDevice(device);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        if (ret == hipErrorInvalidDevice) {
            fprintf(stderr, "[ERROR] invalid CUDA device ID set (%d).\n", device);
        } else if (ret == hipErrorContextAlreadyInUse) {
            fprintf(stderr, "[ERROR] CUDA device with specified ID already in use (%d).\n", device);
        }

        fprintf(stderr, "[ERROR] CUDA error: hipSetDevice failure\n");
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


/* Safe wrapper around hipMalloc
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
    hipError_t ret;

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

    ret = hipMalloc(ptr, size);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMalloc failure\n");
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


/* Safe wrapper around hipHostAlloc
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
    hipError_t ret;

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

    ret = hipHostMalloc(ptr, size, flags);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipHostAlloc failure\n");
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


/* Safe wrapper around hipFree
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
    hipError_t ret;

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

    ret = hipFree(ptr);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[WARNING] CUDA error: hipFree failure\n");
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


/* Safe wrapper around hipHostFree
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
    hipError_t ret;

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

    ret = hipHostFree(ptr);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[WARNING] CUDA error: hipHostFree failure\n");
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


/* Safe wrapper around hipMemset
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
    hipError_t ret;

    ret = hipMemset(ptr, data, count);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMemset failure\n");
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


/* Safe wrapper around hipMemsetAsync
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * s: GPU stream to perform memset in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemsetAsync(void *ptr, int data, size_t count,
        hipStream_t s, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipMemsetAsync(ptr, data, count, s);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMemsetAsync failure\n");
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


/* Safe wrapper around hipMemcpy
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: GPU enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuMemcpy(void * const dest, void const * const src, size_t count,
        hipMemcpyKind dir, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipMemcpy(dest, src, count, dir);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMemcpy failure\n");
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


/* Safe wrapper around hipMemcpyAsync
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
        hipMemcpyKind dir, hipStream_t s, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipMemcpyAsync(dest, src, count, dir, s);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMemcpyAsync failure\n");
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


/* Safe wrapper around hipMemcpyToSymbol
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
    hipError_t ret;

    ret = hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipMemcpyToSymbol failure\n");
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


/* Safe wrapper around hipHostGetDevicePointer
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
    hipError_t ret;

    ret = hipHostGetDevicePointer(pdev, phost, flags);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipHostGetDevicePointer failure\n");
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
            _gpuMemcpy(*ptr, old_ptr, old_ptr_size, hipMemcpyHostToHost,
                    __FILE__, __LINE__);

            _gpuFreeHost(old_ptr, filename, line);
        }
    }
}


/* Safe wrapper around hipEventCreate
 *
 * event: created CUDA event
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventCreate(hipEvent_t * event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipEventCreate(event);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipEventCreate failure\n");
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


/* Safe wrapper around hipEventDestroy
 *
 * event: CUDA event to destroy
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventDestroy(hipEvent_t event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipEventDestroy(event);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipEventDestroy failure\n");
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


/* Safe wrapper around hipEventElapsedTime
 *
 * time: elapsed time between CUDA events (in ms)
 * start, end: CUDA events to compute elapsed time for
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventElapsedTime(float * time, hipEvent_t start, hipEvent_t end, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipEventElapsedTime(time, start, end);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipEventElapsedTime failure\n");
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


/* Safe wrapper around hipEventRecord
 *
 * event: CUDA event to record
 * stream: CUDA stream in which to record event
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventRecord(hipEvent_t event, hipStream_t stream, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipEventRecord(event, stream);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipEventRecord failure\n");
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


/* Safe wrapper around hipEventSynchronize
 *
 * event: CUDA event to record
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void _gpuEventSynchronize(hipEvent_t event, const char * const filename, int line)
{
#if defined(MPIV_GPU)
    int rank;
#endif
    hipError_t ret;

    ret = hipEventSynchronize(event);

    if (ret != hipSuccess)
    {
#if defined(MPIV_GPU)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        const char *str = hipGetErrorString(ret);

        fprintf(stderr, "[ERROR] CUDA error: hipEventSynchronize failure\n");
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
