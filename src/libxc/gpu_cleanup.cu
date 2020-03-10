
void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo, gpu_libxc_in* d_glin, gpu_libxc_out* d_glout){

	gpu_libxc_info* h_glinfo;
	gpu_libxc_in* h_glin;
	gpu_libxc_out* h_glout;

	h_glinfo = (gpu_libxc_info*)malloc(sizeof(gpu_libxc_info));
	h_glin = (gpu_libxc_in*)malloc(sizeof(gpu_libxc_in));
	h_glout = (gpu_libxc_out*)malloc(sizeof(gpu_libxc_out));

	cudaMemcpy(h_glinfo, d_glinfo, sizeof(gpu_libxc_info), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_glin, d_glin, sizeof(gpu_libxc_in), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_glout, d_glout, sizeof(gpu_libxc_out), cudaMemcpyDeviceToHost);

        cudaFree(d_glout);
        cudaFree(h_glout->d_zk);
        cudaFree(h_glout->d_vrho);
        cudaFree(h_glout->d_vsigma);

        cudaFree(h_glinfo->d_maple2c_params);
        cudaFree(h_glinfo->d_gdm);
        cudaFree(h_glinfo->d_ds);
        cudaFree(h_glinfo->d_rhoLDA);
        cudaFree(h_glinfo->d_std_libxc_work_params);
        cudaFree(d_glinfo);

        cudaFree(h_glin->d_rho);
        cudaFree(h_glin->d_sigma);
        cudaFree(d_glin);

        free(h_glout);
        free(h_glinfo);
        free(h_glin);

}
