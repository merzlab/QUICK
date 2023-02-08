
#if defined HIP || defined HIP_MPIV

#define DEV_FREE hipFree
#define DEV_MEMCPY hipMemcpy
#define DEV_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost

#else

#define DEV_FREE cudaFree
#define DEV_MEMCPY cudaMemcpy
#define DEV_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost

#endif

void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo){

        // d_glinfo is a pointer to a struct hosting some device pointers. In order to free
        // device memory pointed by these child pointers, we must access them.
	gpu_libxc_info* h_glinfo;
	h_glinfo = (gpu_libxc_info*)malloc(sizeof(gpu_libxc_info));
	DEV_MEMCPY(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), DEV_MEMCPY_DEVICE_TO_HOST);

        DEV_FREE(h_glinfo->d_maple2c_params);
        DEV_FREE(h_glinfo->d_gdm);
        DEV_FREE(h_glinfo->d_ds);
        DEV_FREE(h_glinfo->d_rhoLDA);
        DEV_FREE(d_glinfo);

        free(h_glinfo);

}

