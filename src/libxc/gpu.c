#ifndef GPU_H
#include "util.h"
#endif
#include <stdio.h>

#if defined CUDA || defined CUDA_MPIV
int dryrun;

void get_gpu_work_params(xc_func_type* p, void *gpu_work_params){


	dryrun = 1;

	double rho[2] = {0.0, 0.0};
	double sigma[3] = {0.0, 0.0, 0.0};
	double exc[1], vrho[2], vsigma[3];

	switch(p->info->family){
	case(XC_FAMILY_LDA):
		xc_lda_exc_vxc(p, 1, rho, exc, vrho, gpu_work_params);
		break;
        case(XC_FAMILY_GGA):
	case(XC_FAMILY_HYB_GGA):
		xc_gga_exc_vxc(p, 1, rho, sigma, exc, vrho, vsigma, gpu_work_params);
		break;
	}

	dryrun = 0 ;

}

void set_gpu_ggax_work_params(double sfact, double dens_threshold, double alpha, 
	double beta, double c_zk0, double c_vrho0, double c_vrho1, double c_vrho2, double c_vsigma0, double c_vsigma1, int k_index, gpu_ggax_work_params *w){

        w -> sfact = sfact;
        w -> dens_threshold = dens_threshold;
        w -> alpha = alpha;
        w -> beta = beta;
        w -> c_zk0 = c_zk0;
        w -> c_vrho0 = c_vrho0;
        w -> c_vrho1 = c_vrho1;
        w -> c_vrho2 = c_vrho2;
        w -> c_vsigma0 = c_vsigma0;
        w -> c_vsigma1 = c_vsigma1;
	w -> k_index = k_index;

}

void set_gpu_ggac_work_params(double dens_threshold, int k_index, gpu_ggac_work_params *w){

        w -> dens_threshold = dens_threshold;
        w -> k_index = k_index;

}

void set_gpu_lda_work_params(double dens_threshold, double cnst_rs, int xc_dim, int k_index, gpu_lda_work_params *w){

	w -> dens_threshold = dens_threshold;
	w -> cnst_rs = cnst_rs;
	w -> xc_dim = xc_dim;
        w -> k_index = k_index;

}
#endif
