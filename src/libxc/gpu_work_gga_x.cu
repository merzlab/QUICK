
typedef struct{
  double beta, gamma;
} gga_x_b88_params;

#include "maple2c/gga_x_b88.c"

#ifdef QUICK_LIBXC
__device__ void gpu_work_gga_x(gpu_libxc_info* glinfo, double d_rhoa, double d_rhob, double d_sigma, double *d_zk, double *d_vrho, double *d_vsigma){

	double d_rho=d_rhoa + d_rhob;
#else
__global__ void gpu_work_gga_x(gpu_libxc_info* glinfo, gpu_libxc_in* glin, gpu_libxc_out* glout, int size){
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid<size){
#endif
		gpu_libxc_info* d_glinfo;
		d_glinfo = (gpu_libxc_info*)glinfo;

		gpu_ggax_work_params *d_w;
		d_w = (gpu_ggax_work_params*)(d_glinfo->d_worker_params);

		xc_gga_work_x_t d_rg;
		d_rg.order = 1;

		double test_gdm = max(sqrt(d_sigma)/d_w->sfact, d_w->dens_threshold);
		double test_ds = d_rho/d_w->sfact;
		double test_rhoLDA = pow(test_ds, d_w->alpha);
		double test_rgx  = test_gdm/pow(test_ds, d_w->beta);
		d_rg.x = test_rgx;

		if(GPU_DEBUG){
//			printf("rho: %.10e  sigma: %.10e  sfac: %.10e  alpha: %.10e  beta: %.10e  d_rg->x: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
//			d_w->sfact, d_w->alpha, d_w->beta, d_rg->x);		
//			printf("rho: %.10e  sigma: %.10e  test_gdm: %.10e  test_ds: %.10e  test_rhoLDA: %.10e  d_rg->x: %.10e \n ", d_rho,d_sigma
//			,test_gdm, test_ds, test_rhoLDA, d_rg->x); 
		}

	        switch(d_w->func_id){
	        case XC_GGA_X_B88:
        	case XC_GGA_X_OPTB88_VDW:
        	case XC_GGA_K_LLP:
        	case XC_GGA_K_FR_B88:
        	case XC_GGA_X_MB88:
        	case XC_GGA_X_EB88:
        	case XC_GGA_X_B88M:
			
			xc_gga_x_b88_enhance(d_glinfo->d_maple2c_params, &d_rg);
			
                	break;
        	}

		if(GPU_DEBUG){
//                        printf("rho: %.10e  sigma: %.10e  test_rhoLDA: %.10e  test_ds: %.10e  d_rg->f: %.10e \n ", d_rho, d_sigma,
//                        test_rhoLDA, d_w->c_zk0, d_rg->f);
			//printf("test_cu.cu: test_gpu_params(): f: %f, dfdr: %f \n", d_rg->f, d_rg->dfdx);
		}

		double test_zk = (test_rhoLDA * d_w->c_zk0 * d_rg.f)/ d_rho;
		*d_zk = test_zk;
	
		double test_vrho = (test_rhoLDA/test_ds)* (d_w->c_vrho0 * d_rg.f + d_w->c_vrho1 * d_rg.dfdx * d_rg.x);
		*d_vrho = test_vrho;

		if(test_gdm > d_w->dens_threshold){
			double test_vsigma = test_rhoLDA* (d_w->c_vsigma0 * d_rg.dfdx*d_rg.x/(2*d_sigma));
			*d_vsigma = test_vsigma;
        	}

		if(GPU_DEBUG){
                        printf("rho: %.10e  sigma: %.10e  test_gdm: %.10e  test_ds: %.10e  test_rhoLDA: %.10e  d_rg->x: %.10e \n ", d_rho,d_sigma
                        ,test_gdm, test_ds, test_rhoLDA, d_rg.x);

//			printf("rho: %.10e  sigma: %.10e  d_rg->f: %.10e  d_rg->dfdx: %.10e \n",d_rho, d_sigma, d_rg.f, d_rg.dfdx);

                        //printf("rho: %.10e  sigma: %.10e  glout->d_zk: %.10e  glout->d_vrho: %.10e  glout->d_vsigma: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
                        //test_zk, test_vrho, test_gdm);
	        	//printf("zk: %f, vrho: %f, vsigma: %f \n", glout->d_zk[gid], glout->d_vrho[gid], glout->d_vsigma[gid]);
        	}		
#ifndef QUICK_LIBXC
	}
#endif
}

