#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "util.h"
#include "gpu_work.cuh"
//Definitions for switch statements 
#define XC_GGA_X_B88          106
#define XC_GGA_X_OPTB88_VDW   139
#define XC_GGA_X_MB88         149
#define XC_GGA_X_EB88         271
#define XC_GGA_K_LLP          522
#define XC_GGA_K_FR_B88       514
#define XC_GGA_X_B88M         570

#ifdef CUDA

//maple2c param typedefs
/*typedef struct{
  double beta, gamma;
} gga_x_b88_params;
*/
/*typedef struct{
        double *d_rho; //Input array, holds densities
        double *d_sigma; //Input array, holds the gradient of densities
}gpu_libxc_in;

//hosts device pointers reqired for libxc kernels
typedef struct {
	//common worker params
	void *d_maple2c_params;
	void *d_worker_params;

	//Specific for gga_x worker
	double *d_gdm;
        double *d_ds;
        double *d_rhoLDA;
	void *d_std_libxc_work_params; 

	//Output device array pointers 
        //double *d_zk; //Output array, holds energy per particle
        //double *d_vrho; //Output array
        //double *d_vsigma; //Output array

}gpu_libxc_info;

typedef struct {
        //Output device array pointers 
        double *d_zk; //Output array, holds energy per particle
        double *d_vrho; //Output array
        double *d_vsigma; //Output array
}gpu_libxc_out;
*/
#ifndef QUICK_LIBXC
#include "gpu_work_gga_x.cu"
#endif
#include "gpu_upload.cu"
#include "gpu_cleanup.cu"

//This is the main method for gpu calculations
extern "C" void test_cu(const xc_func_type *p, gpu_ggax_work_params *ggwp, xc_gga_work_x_t h_r, const double *h_rho, const double *h_sigma, int np){

#ifndef QUICK_LIBXC

        if(GPU_DEBUG){
                printf("\n ============ Testing GPU LIBXC ============ \n");
                printf("test_cu.cu:dens_threshold: %f \n", p->dens_threshold);
                printf("test_cu.cu:work parameter ggwp.beta: %f \n", ggwp->beta);
        }

       if(GPU_DEBUG){
               printf("FILE: %s, LINE: %d, FUNCTION: %s \n", __FILE__, __LINE__, __func__);
        }
	
	//----------------- pointer hosting work parameters on device --------------------------
	gpu_libxc_info* d_glinfo;
	d_glinfo = gpu_upload_libxc_info(p, ggwp, h_r, np);

	//rho and sigma pointers on device
	gpu_libxc_in* d_glin;
	d_glin = gpu_upload_libxc_in(h_rho, h_sigma, np);

	//Output arry pointers on device
	gpu_libxc_out* d_glout;
	d_glout = gpu_upload_libxc_out(np);

	//cudaMalloc((void**)&d_glinfo, sizeof(gpu_libxc_info));
	//cudaMemcpy(d_glinfo, &h_glinfo, sizeof(gpu_libxc_info), cudaMemcpyHostToDevice);

       if(GPU_DEBUG){
               printf("FILE: %s, LINE: %d, FUNCTION: %s \n", __FILE__, __LINE__, __func__);
        }

	//------------------ Allocate host memory for results ------------------------	
        gpu_libxc_out* h_glout;
        double *h_zk; //Output array, holds energy per particle
        double *h_vrho; //Output array
        double *h_vsigma; //Output array        

	h_glout = (gpu_libxc_out*)malloc(sizeof(gpu_libxc_out));
        h_zk = (double*)malloc(sizeof(double)*np);
        h_vrho = (double*)malloc(sizeof(double)*np);
        h_vsigma = (double*)malloc(sizeof(double)*np);
	//----------------------------------------------------------------------------

        //-------- GPU params ---------
        int block_size=32;
        dim3 block(block_size);
        dim3 grid((np/block.x) + 1);
	//------------------------------

	//--------------------- Call the GPU worker based on functional family and kind ------------------- 
	//Check the family of functional
	switch(p->info->family){
	case(XC_FAMILY_GGA):

		//Now check the kind. 
		switch(p->info->kind){
			case(XC_EXCHANGE):

				if(GPU_DEBUG){
					printf("FILE: %s, LINE: %d, FUNCTION: %s \n", __FILE__, __LINE__, __func__);
				}
				gpu_work_gga_x <<< grid, block >>> (d_glinfo, d_glin, d_glout, np);
			break;
		}
	break;

	}	
	//-------------------------------------------------------------------------------------------------

	cudaDeviceSynchronize();

	//----------- Copy the results back to host --------------

	cudaMemcpy(h_glout, d_glout, sizeof(gpu_libxc_out), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_zk, (h_glout->d_zk), np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vrho, (h_glout->d_vrho), np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vsigma, (h_glout->d_vsigma), np*sizeof(double), cudaMemcpyDeviceToHost);

	for(int i=0; i<np;i++){
		printf("Copied zk: %f, vrho: %f, vsigma: %f \n", h_zk[i],h_vrho[i],h_vsigma[i]);
	}
	
	//------------------------------------------------------

	//-------------- Clean up the memory -------------------
	/*cudaFree(d_glout);
	cudaFree(h_glout->d_zk);
	cudaFree(h_glout->d_vrho);
	cudaFree(h_glout->d_vsigma);

        gpu_libxc_info* h_glinfo;
	gpu_libxc_in* h_glin;

        h_glinfo = (gpu_libxc_info*)malloc(sizeof(gpu_libxc_info));
	h_glin = (gpu_libxc_in*)malloc(sizeof(gpu_libxc_in));

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

	free(h_zk);
	free(h_vrho);
	free(h_vsigma);
	*/
	gpu_libxc_cleanup(d_glinfo, d_glin, d_glout);

	cudaDeviceReset();
	//------------------------------------------------------

        if(GPU_DEBUG){
		printf("\n =========== Test end GPU LIBXC ============ \n");
	}
#endif
}

#endif
