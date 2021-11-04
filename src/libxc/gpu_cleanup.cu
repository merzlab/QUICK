
void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo){

        // d_glinfo is a pointer to a struct hosting some device pointers. In order to free
        // device memory pointed by these child pointers, we must access them.
	gpu_libxc_info* h_glinfo;
	h_glinfo = (gpu_libxc_info*)malloc(sizeof(gpu_libxc_info));
	hipMemcpy(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), hipMemcpyDeviceToHost);

        hipFree(h_glinfo->d_maple2c_params);
        hipFree(h_glinfo->d_gdm);
        hipFree(h_glinfo->d_ds);
        hipFree(h_glinfo->d_rhoLDA);
        hipFree(d_glinfo);

        free(h_glinfo);

}
