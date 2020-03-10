#ifndef GPU_H
#include "util.h"
#endif
#include <stdio.h>
//#include "gpu_work.h"


#ifdef CUDA
int dryrun;

void get_gpu_work_params(xc_func_type* p, void *gpu_work_params){


	dryrun = 1;
	//if(glds) glds = NULL;

	double rho[1] = {0.0};
	double sigma[1] = {0.0};
	double exc[1], vrho[1], vsigma[1];

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, GET_GPU_WORK_PARAMS AT WORK \n", __FILE__, __LINE__, __func__);
        }

	switch(p->info->family){
	case(XC_FAMILY_LDA):

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, GET_GPU_WORK_PARAMS CALLING XC_LDA_EXC_VXC \n", __FILE__, __LINE__, __func__);
        }
		xc_lda_exc_vxc(p, 1, rho, exc, vrho, gpu_work_params);

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, w->xc_dim: %d, GET_GPU_WORK_PARAMS TESTING XC_LDA_EXC_VXC \n", __FILE__, __LINE__, __func__, ((gpu_lda_work_params*)gpu_work_params)->xc_dim);
        }

		break;
        case(XC_FAMILY_GGA):
	case(XC_FAMILY_HYB_GGA):
	//	switch(p->info->kind){
	//		case(XC_EXCHANGE):
	if(GPU_DEBUG){
        	printf("FILE: %s, LINE: %d, FUNCTION: %s, GET_GPU_WORK_PARAMS CALLING XC_GGA_EXC_VXC \n", __FILE__, __LINE__, __func__);
	}		
		//gpu_ggax_work_params* tmp_ggwp;
		//tmp_ggwp = (gpu_ggax_work_params*) malloc(sizeof(gpu_ggax_work_params));	
				//xc_gga_exc_vxc(p, 1, rho, sigma, exc, vrho, vsigma, (void*)(glinfo));
		xc_gga_exc_vxc(p, 1, rho, sigma, exc, vrho, vsigma, gpu_work_params);
	if(GPU_DEBUG){

                gpu_ggac_work_params* tmp_ggwp;
                tmp_ggwp = (gpu_ggac_work_params*)gpu_work_params;

                //printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp VALUE: %f \n", __FILE__, __LINE__, __func__, tmp_ggwp->func_id);

	}

	if(GPU_DEBUG){
        	printf("FILE: %s, LINE: %d, FUNCTION: %s, INITIALIZING LIBXC \n", __FILE__, __LINE__, __func__);
	}
			//break;
		//}
		break;
	}

	//We update the value of the static pointer variable in a work method (eg. work_gga_x.c).	
	dryrun = 0 ;

//	printf("Just before returning glds");
	//return glds; 

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
