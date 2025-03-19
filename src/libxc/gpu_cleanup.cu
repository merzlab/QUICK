#if defined(HIP) || defined(HIP_MPIV)
  #include "../gpu/hip/gpu_utils.h"
  #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#elif defined(CUDA) || defined(CUDA_MPIV)
  #include "../gpu/cuda/gpu_utils.h"
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#endif


void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo)
{
    // d_glinfo is a pointer to a struct hosting some device pointers. In order to free
    // device memory pointed by these child pointers, we must access them.
    gpu_libxc_info *h_glinfo;

    h_glinfo = (gpu_libxc_info *) malloc(sizeof(gpu_libxc_info));
    gpuMemcpy(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), gpuMemcpyDeviceToHost);

    gpuFree(h_glinfo->d_maple2c_params);
    gpuFree(h_glinfo->d_gdm);
    gpuFree(h_glinfo->d_ds);
    gpuFree(h_glinfo->d_rhoLDA);
    gpuFree(d_glinfo);

    free(h_glinfo);
}
