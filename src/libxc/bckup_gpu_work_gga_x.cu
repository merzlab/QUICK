//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include "util.h"
//#include "gpu_work.h"

typedef struct{
  double beta, gamma;
} gga_x_b88_params;

#include "maple2c/gga_x_b88.c"

//typedef void (libxc_functional*) (const void *p,  xc_gga_work_x_t *r);

//__global__ void test_gpu_param(void * d_p, void *d_work_params, void *d_r, double *gdm, double *ds, double *rhoLDA, int size, double *d_rho, double *d_sigma, double *d_zk, double *d_vrho, double *d_vsigma, gpu_libxc_info* d_glinfo){
#ifdef QUICK_LIBXC
__device__ void gpu_work_gga_x(gpu_libxc_info* glinfo, gpu_libxc_in* glin, gpu_libxc_out* glout, int size){

	int gid = 0;	
	//int offset =  blockIdx.x * blockDim.x + threadIdx.x;
#else
__global__ void gpu_work_gga_x(gpu_libxc_info* glinfo, gpu_libxc_in* glin, gpu_libxc_out* glout, int size){
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
	if(gid<size){

		gpu_libxc_info* d_glinfo;
		d_glinfo = (gpu_libxc_info*)glinfo;

//		gga_x_b88_params *param;
//		param = (gga_x_b88_params*) (d_glinfo->d_maple2c_params);

		gpu_ggax_work_params *d_w;
		d_w = (gpu_ggax_work_params*)(d_glinfo->d_worker_params);

		xc_gga_work_x_t* d_r;
		d_r = (xc_gga_work_x_t*) (d_glinfo->d_std_libxc_work_params);

		xc_gga_work_x_t* d_rg = &d_r[gid];

		//xc_gga_work_x_t *d_rg;
		//d_rg = (xc_gga_work_x_t*) &d_r[gid];		

		if(GPU_DEBUG){
		/*	//printf("test_cu.cu: test_gpu_param(): d_p -> beta: %f \n", param -> beta);
			printf("test_cu.cu: test_gpu_param(): ggwp -> alpha: %f \n", d_w -> alpha);
			//printf("test_cu.cu: test_gpu_params(): rho, sigma: %f, %f \n", d_rho[gid], d_sigma[gid]);
			printf("test_cu.cu: test_gpu_params(): d_rg[gid].order: %f \n", d_rg->order);
			printf("test_cu.cu: test_gpu_params(): d_rg[gid].x: %f \n", d_rg->x);
			printf("test_cu.cu: test_gpu_params(): d_rg[gid].dfdx: %f \n", d_rg->dfdx);
			printf("test_cu.cu: test_gpu_params(): d_rg[gid].d2fdx2: %f \n", d_rg->d2fdx2);
			printf("test_cu.cu: test_gpu_params(): d_rg[gid].d3fdx3: %f \n", d_rg->d3fdx3);
		*/
		}

		//d_rg[gid] = order;
		double test_gdm = max(sqrt(glin->d_sigma[gid])/d_w->sfact, d_w->dens_threshold);
		d_glinfo->d_gdm[gid] = test_gdm;
        	//d_glinfo->d_gdm[gid]    = max(sqrt(glin->d_sigma[gid])/d_w->sfact, d_w->dens_threshold);
        	//d_glinfo->d_ds[gid]     = glin->d_rho[gid]/d_w->sfact;
		double test_ds = glin->d_rho[gid]/d_w->sfact;
		d_glinfo->d_ds[gid] = test_ds;
        	//d_glinfo->d_rhoLDA[gid] = pow(d_glinfo->d_ds[gid], d_w->alpha);
		double test_rhoLDA = pow(d_glinfo->d_ds[gid], d_w->alpha);
		d_glinfo->d_rhoLDA[gid] = test_rhoLDA;
        	//d_rg->x    = d_glinfo->d_gdm[gid]/pow(d_glinfo->d_ds[gid], d_w->beta);
		double test_rgx  = test_gdm/pow(test_ds, d_w->beta);
		d_rg->x = test_rgx;

		if(GPU_DEBUG){
//			printf("rho: %.10e  sigma: %.10e  sfac: %.10e  alpha: %.10e  beta: %.10e  d_rg->x: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
//			d_w->sfact, d_w->alpha, d_w->beta, d_rg->x);		
//			printf("rho: %.10e  sigma: %.10e  sfac: %.10e  dens_threshold: %.10e  d_rg->x: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
//			d_w->sfact, d_w->dens_threshold, d_rg->x); 
		}

		if(GPU_DEBUG){
                	//printf("test_cu.cu: test_gpu_params(): d_r->x: %f \n", d_rg->x);		
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
				//printf("test_cu.cu: test_gpu_params(): d_w->func_id: %d \n", d_w->func_id);	
			}
			//libxc_functional test_method_ptr = xc_gga_x_b88_enhance;
			
			xc_gga_x_b88_enhance(d_glinfo->d_maple2c_params, d_rg);
			
                	break;
        	}

		if(GPU_DEBUG){
                        printf("rho: %.10e  sigma: %.10e  test_rhoLDA: %.10e  d_w->c_zk0: %.10e  d_rg->f: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
                        test_rhoLDA, d_w->c_zk0, d_rg->f);
			//printf("test_cu.cu: test_gpu_params(): f: %f, dfdr: %f \n", d_rg->f, d_rg->dfdx);
		}

		//glout->d_zk[gid] = (d_glinfo->d_rhoLDA[gid] * d_w->c_zk0 * d_rg->f)/ glin->d_rho[gid];
		double test_zk = (test_rhoLDA * d_w->c_zk0 * d_rg->f)/ glin->d_rho[gid];
		glout->d_zk[gid] = test_zk;
	
        	//glout->d_vrho[gid] = (d_glinfo->d_rhoLDA[gid]/d_glinfo->d_ds[gid])* (d_w->c_vrho0 * d_rg->f + d_w->c_vrho1 * d_rg->dfdx*d_rg->x);
		double test_vrho = (test_rhoLDA/test_ds)* (d_w->c_vrho0 * d_rg->f + d_w->c_vrho1 * d_rg->dfdx*d_rg->x);
		glout->d_vrho[gid] = test_vrho;

        	//if(d_glinfo->d_gdm[gid] > d_w->dens_threshold){
		if(test_gdm > d_w->dens_threshold){
                	//glout->d_vsigma[gid] = d_glinfo->d_rhoLDA[gid]* (d_w->c_vsigma0 * d_rg->dfdx*d_rg->x/(2*glin->d_sigma[gid]));
			double test_vsigma = test_rhoLDA* (d_w->c_vsigma0 * d_rg->dfdx*d_rg->x/(2*glin->d_sigma[gid]));
			glout->d_vsigma[gid] = test_vsigma;
        	}

		if(GPU_DEBUG){
                        //printf("rho: %.10e  sigma: %.10e  glout->d_zk: %.10e  glout->d_vrho: %.10e  glout->d_vsigma: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
                        //test_zk, test_vrho, test_gdm);
	        	//printf("zk: %f, vrho: %f, vsigma: %f \n", glout->d_zk[gid], glout->d_vrho[gid], glout->d_vsigma[gid]);
        	}		
//#ifndef QUICK_LIBXC
	}
//#endif
}

