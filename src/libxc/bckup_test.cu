#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "util.h"
#include "test.cuh"
//Definitions for switch statements 
#define XC_GGA_X_B88          106
#define XC_GGA_X_OPTB88_VDW   139
#define XC_GGA_X_MB88         149
#define XC_GGA_X_EB88         271
#define XC_GGA_K_LLP          522
#define XC_GGA_K_FR_B88       514
#define XC_GGA_X_B88M         570

#ifdef CUDA

#define GPU_DEBUG 1 

//maple2c param typedefs
typedef struct{
  double beta, gamma;
} gga_x_b88_params;

#include "maple2c/gga_x_b88.c"

//This method will determine the type of the maple2c params and upload it to gpu
//It will then send back a void pointer. 
void* gpu_upload_maple_params(int func_id, void * h_maple_params){

	void* d_p=NULL;

	switch(func_id){
	case XC_GGA_X_B88:
	case XC_GGA_X_OPTB88_VDW:
	case XC_GGA_K_LLP:
	case XC_GGA_K_FR_B88:
	case XC_GGA_X_MB88:
	case XC_GGA_X_EB88:
	case XC_GGA_X_B88M:
	if(GPU_DEBUG){
		printf("test_cu.cu: gpu_upload_maple_params(): Calling gpu_upload_gga_x_b88.. \n");
	}
		d_p = gpu_upload_gga_x_b88(h_maple_params);
		break;
	default:
		fprintf(stderr, " Internal error while uploading paramters \n");
		exit(1);
	}
	
	return d_p;

}

//Allocate memory for gga_x_b88_params and return the corresponding pointer
void* gpu_upload_gga_x_b88(void * h_maple_params){
	//Convert void pointer in to gga_x_b88_params
	gga_x_b88_params *h_p;
	h_p = (gga_x_b88_params*)(h_maple_params);

	if(GPU_DEBUG){
		printf("test_cu.cu: gpu_upload_gga_x_b88(): h_p -> beta:, %f \n", h_p -> beta);
	}
	//Define device pointer, allocate device memory and copy host data to device
	gga_x_b88_params *d_p;
	cudaMalloc((void**)&d_p, sizeof(gga_x_b88_params));
	cudaMemcpy(d_p, h_p, sizeof(gga_x_b88_params), cudaMemcpyHostToDevice);

	return (void*)d_p;
}

//This is the main method for gpu calculations
extern "C" void test_cu(const xc_func_type *p, gpu_ggax_work_params *ggwp, xc_gga_work_x_t h_r, const double *h_rho, const double *h_sigma){

	if(GPU_DEBUG){
		printf("\n ============ Testing GPU LIBXC ============ \n");
		printf("test_cu.cu:dens_threshold: %f \n", p->dens_threshold);
		printf("test_cu.cu:work parameter ggwp.beta: %f \n", ggwp->beta);
	}

	//----------------upload parameters required for maple2c codes----------------
	void *d_maple_params = NULL;
	d_maple_params = gpu_upload_maple_params(p->info->number, (void*)(p->params));
	//----------------------------------------------------------------------------

	//----------------Upload input and output arrays into GPU---------------------
        int size = 5;
	int byte_size = size * sizeof(double);
	double *d_rho; //Input array, holds densities
	double *d_sigma; //Input array, holds the gradient of densities
	double *d_zk; //Output array, holds energy per particle
	double *d_vrho; //Output array
	double *d_vsigma; //Output array

	cudaMalloc((void**)&d_rho, byte_size);
	cudaMalloc((void**)&d_sigma, byte_size);
	cudaMalloc((void**)&d_zk, byte_size);
	cudaMalloc((void**)&d_vrho, byte_size);
	cudaMalloc((void**)&d_vsigma, byte_size);

	cudaMemcpy(d_rho, h_rho, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sigma, h_sigma, byte_size, cudaMemcpyHostToDevice);
	//----------------------------------------------------------------------------

        //----------------upload libxc gga worker paramsters onto gpu-----------------

        xc_gga_work_x_t *d_r;
        cudaMalloc((void**)&d_r, size*sizeof(xc_gga_work_x_t)); //At this point, only
	//saved paramter inside h_r is the order. We pass it to gpu kernal seperately.  
	//----------------------------------------------------------------------------

	
	//-------------Allocate device memory for several work variables--------------
	double *d_gdm;
	double *d_ds;
	double *d_rhoLDA;
	cudaMalloc((void**)&d_gdm, byte_size);
	cudaMalloc((void**)&d_ds, byte_size);
	cudaMalloc((void**)&d_rhoLDA, byte_size);
        //----------------------------------------------------------------------------

	//------------------ Allocate host memory for results ------------------------	
        double *h_zk; //Output array, holds energy per particle
        double *h_vrho; //Output array
        double *h_vsigma; //Output array        

        h_zk = (double*)malloc(byte_size);
        h_vrho = (double*)malloc(byte_size);
        h_vsigma = (double*)malloc(byte_size);
	//----------------------------------------------------------------------------

        //-------- GPU params ---------
        int block_size=32;
        dim3 block(block_size);
        dim3 grid((size/block.x) + 1);
	//------------------------------

	//--------------------- Call the GPU worker based on functional family and kind ------------------- 
	//Check the family of functional
	switch(p->info->family){
	case(XC_FAMILY_GGA):

		//Now check the kind. 
		switch(p->info->kind){
			case(XC_EXCHANGE):

				//Upload necessary work parameters to GPU
				gpu_ggax_work_params *d_w;
				cudaMalloc((void**)&d_w, sizeof(gpu_ggax_work_params));
				cudaMemcpy(d_w, ggwp, sizeof(gpu_ggax_work_params), cudaMemcpyHostToDevice);

				test_gpu_param <<< grid, block >>> (d_maple_params, d_w, d_r, h_r.order, d_gdm, d_ds, d_rhoLDA, size, d_rho, d_sigma, d_zk, d_vrho, d_vsigma);
			break;
		}
	break;

	}	
	//-------------------------------------------------------------------------------------------------

	cudaDeviceSynchronize();

	//----------- Copy the results back to host --------------

	cudaMemcpy(h_zk, d_zk, byte_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vrho, d_vrho, byte_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vsigma, d_vsigma, byte_size, cudaMemcpyDeviceToHost);

	//------------------------------------------------------

	//-------------- Clean up the memory -------------------
	cudaFree(d_rho);
	cudaFree(d_sigma);
	cudaFree(d_zk);
	cudaFree(d_vrho);
	cudaFree(d_vsigma);
	cudaFree(d_gdm);
	cudaFree(d_ds);
	cudaFree(d_rhoLDA);

	cudaDeviceReset();
	//------------------------------------------------------

        if(GPU_DEBUG){
		printf("\n =========== Test end GPU LIBXC ============ \n");
	}
}

__global__ void test_gpu_param(void * d_p, gpu_ggax_work_params *d_w, xc_gga_work_x_t *d_r, int order, double *gdm, double *ds, double *rhoLDA, int size, double *d_rho, double *d_sigma, double *d_zk, double *d_vrho, double *d_vsigma){

        gga_x_b88_params *param;
        param = (gga_x_b88_params*) d_p;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid<size){

		xc_gga_work_x_t *d_rg;
		d_rg = (xc_gga_work_x_t*) &d_r[gid];		

		if(GPU_DEBUG){
			printf("test_cu.cu: test_gpu_param(): d_p -> beta: %f \n", param -> beta);
			printf("test_cu.cu: test_gpu_param(): ggwp -> alpha: %f \n", d_w -> alpha);
			printf("test_cu.cu: test_gpu_params(): rho, sigma: %f, %f \n", d_rho[gid], d_sigma[gid]);
		}

		d_rg->order = order;
        	gdm[gid]    = max(sqrt(d_sigma[gid])/d_w->sfact, d_w->dens_threshold);
        	ds[gid]     = d_rho[gid]/d_w->sfact;
        	rhoLDA[gid] = pow(ds[gid], d_w->alpha);
        	d_rg->x    = gdm[gid]/pow(ds[gid], d_w->beta);

		if(GPU_DEBUG){
                	printf("test_cu.cu: test_gpu_params(): d_r->x: %f \n", d_rg->x);		
		}
	        switch(d_w->func_id){
	        case XC_GGA_X_B88:
        	case XC_GGA_X_OPTB88_VDW:
        	case XC_GGA_K_LLP:
        	case XC_GGA_K_FR_B88:
        	case XC_GGA_X_MB88:
        	case XC_GGA_X_EB88:
        	case XC_GGA_X_B88M:
			if(GPU_DEBUG){
				printf("test_cu.cu: test_gpu_params(): d_w->func_id: %d \n", d_w->func_id);			
			}
			xc_gga_x_b88_enhance((void*)d_p, d_rg);
                	break;
        	}

		if(GPU_DEBUG){
			printf("test_cu.cu: test_gpu_params(): f: %f, dfdr: %f \n", d_rg->f, d_rg->dfdx);
		}

		d_zk[gid] = (rhoLDA[gid] * d_w->c_zk0 * d_rg->f)/ d_rho[gid];

        	d_vrho[gid] = (rhoLDA[gid]/ds[gid])* (d_w->c_vrho0 * d_rg->f + d_w->c_vrho1 * d_rg->dfdx*d_rg->x);

        	if(gdm[gid] > d_w->dens_threshold){
                	d_vsigma[gid] = rhoLDA[gid]* (d_w->c_vsigma0 * d_rg->dfdx*d_rg->x/(2*d_sigma[gid]));
        	}

		if(GPU_DEBUG){
	        	printf("zk: %f, vrho: %f, vsigma: %f \n", d_zk[gid], d_vrho[gid], d_vsigma[gid]);
        	}		
		
	}
}

#endif
