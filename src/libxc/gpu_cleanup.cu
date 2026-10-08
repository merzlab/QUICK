#if defined(HIP) || defined(HIP_MPIV)
  #define gpuError_t hipError_t
  #define gpuSuccess hipSuccess
  #define gpuMemcpyKind hipMemcpyKind
  #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define gpuGetErrorString_ hipGetErrorString
#elif defined(CUDA) || defined(CUDA_MPIV)
  #define gpuError_t cudaError_t
  #define gpuSuccess cudaSuccess
  #define gpuMemcpyKind cudaMemcpyKind
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuGetErrorString_ cudaGetErrorString
#endif


/* Safe wrapper around gpuMemcpy2_
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: GPU enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
static void gpuMemcpy2_(void * const dest, void const * const src, size_t count,
        gpuMemcpyKind dir, const char * const filename, int line)
{
    gpuError_t ret;

#if defined(HIP) || defined(HIP_MPIV)
    ret = hipMemcpy(dest, src, count, dir);
#elif defined(CUDA) || defined(CUDA_MPIV)
    ret = cudaMemcpy(dest, src, count, dir);
#endif

    if (ret != gpuSuccess) {
        const char *str = gpuGetErrorString_(ret);

        fprintf(stderr, "[ERROR] GPU error: gpuMemcpy failure\n");
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

        exit(1);
    }
}


/* Safe wrapper around gpuFree_
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void gpuFree_(void *ptr, const char * const filename, int line)
{
    gpuError_t ret;

    if (ptr == NULL) {
        fprintf(stderr, "[WARNING] trying to free the already NULL pointer\n");
        fprintf(stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        return;
    }  

#if defined(HIP) || defined(HIP_MPIV)
    ret = hipFree(ptr);
#elif defined(CUDA) || defined(CUDA_MPIV)
    ret = cudaFree(ptr);
#endif

    if (ret != gpuSuccess) {
        const char *str = gpuGetErrorString_(ret);

        fprintf(stderr, "[WARNING] GPU error: gpuFree failure\n");
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);
        fprintf(stderr, "  [INFO] Memory address: %ld\n", 
                (long int) ptr);

        return;
    }  
}


void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo)
{
    // d_glinfo is a pointer to a struct hosting some device pointers. In order to free
    // device memory pointed by these child pointers, we must access them.
    gpu_libxc_info *h_glinfo;

    h_glinfo = (gpu_libxc_info *) malloc(sizeof(gpu_libxc_info));
    gpuMemcpy2_(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), gpuMemcpyDeviceToHost, __FILE__, __LINE__);

    if (h_glinfo->d_maple2c_params != NULL) {
        gpuFree_(h_glinfo->d_maple2c_params, __FILE__, __LINE__);
    }
    gpuFree_(h_glinfo->d_gdm, __FILE__, __LINE__);
    gpuFree_(h_glinfo->d_ds, __FILE__, __LINE__);
    gpuFree_(h_glinfo->d_rhoLDA, __FILE__, __LINE__);
    gpuFree_(d_glinfo, __FILE__, __LINE__);

    free(h_glinfo);
}
