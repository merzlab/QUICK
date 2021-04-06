
void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo){

        // d_glinfo is a pointer to a struct hosting some device pointers. In order to free
        // device memory pointed by these child pointers, we must access them.
	gpu_libxc_info* h_glinfo;
	h_glinfo = (gpu_libxc_info*)malloc(sizeof(gpu_libxc_info));
	cudaMemcpy(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), cudaMemcpyDeviceToHost);

        cudaFree(h_glinfo->d_maple2c_params);
        cudaFree(h_glinfo->d_gdm);
        cudaFree(h_glinfo->d_ds);
        cudaFree(h_glinfo->d_rhoLDA);
        cudaFree(d_glinfo);

        free(h_glinfo);

}
