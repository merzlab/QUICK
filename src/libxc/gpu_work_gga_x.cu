
/*typedef struct{
  double beta, gamma;
} gga_x_b88_params;
*/

#define DEVICE

static const double B1 =  1.74959015598863046792081721182; /* 3*muGE/5 + 8 pi/15 */
static const double B2 = -1.62613336586517367779736042170; /* muGE - B1 */

typedef void (*ggaxk_ptr)(const void *p,  xc_gga_work_x_t *r);

//****************** Uncomment to compile all libxc kernels*************************//
#include "gpu_fstructs.h"
//#include "gpu_finclude_ggaxk.h"
#include "gpu_fsign_ggaxk.h"
//****************** Uncomment to compile all libxc kernels*************************//


/*__device__ void test_func_ptr(void maple_func(const void *p,  xc_gga_work_x_t *r)){
	maple_func(p, r);
}*/

__device__ void gpu_work_gga_x(gpu_libxc_info* glinfo, double d_rhoa, double d_rhob, double d_sigma, double *d_zk, double *d_vrho, double *d_vsigma){

	double d_rho=d_rhoa + d_rhob;

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

#ifdef DEBUG 
//			printf("rho: %.10e  sigma: %.10e  sfac: %.10e  alpha: %.10e  beta: %.10e  d_rg->x: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
//			d_w->sfact, d_w->alpha, d_w->beta, d_rg->x);		
//			printf("rho: %.10e  sigma: %.10e  test_gdm: %.10e  test_ds: %.10e  test_rhoLDA: %.10e  d_rg->x: %.10e \n ", d_rho,d_sigma
//			,test_gdm, test_ds, test_rhoLDA, d_rg->x); 
#endif
	
/*	        switch(d_glinfo->func_id){
	        case XC_GGA_X_B88:
        	case XC_GGA_X_OPTB88_VDW:
        	case XC_GGA_K_LLP:
        	case XC_GGA_K_FR_B88:
        	case XC_GGA_X_MB88:
        	case XC_GGA_X_EB88:
        	case XC_GGA_X_B88M:
*/			
//			xc_gga_x_b88_enhance(d_glinfo->d_maple2c_params, &d_rg);
			//point_kernel test_ptr = xc_gga_x_b88_enhance;
			//point_kernel pkernel = (point_kernel) (d_w->pkernel)

		
//****************** Uncomment to compile all libxc kernels*************************//
		(maple2cf_ggaxk[d_w->k_index])(d_glinfo->d_maple2c_params, &d_rg);
//****************** Uncomment to compile all libxc kernels*************************//
			
                	//break;
        	//}

#ifdef DEBUG 
//                        printf("rho: %.10e  sigma: %.10e  test_rhoLDA: %.10e  test_ds: %.10e  d_rg->f: %.10e \n ", d_rho, d_sigma,
//                        test_rhoLDA, d_w->c_zk0, d_rg->f);
			//printf("test_cu.cu: test_gpu_params(): f: %f, dfdr: %f \n", d_rg->f, d_rg->dfdx);
#endif	

		double test_zk = (test_rhoLDA * d_w->c_zk0 * d_rg.f)/ d_rho;
		*d_zk = test_zk;
	
		double test_vrho = (test_rhoLDA/test_ds)* (d_w->c_vrho0 * d_rg.f + d_w->c_vrho1 * d_rg.dfdx * d_rg.x);
		*d_vrho = test_vrho;

		if(test_gdm > d_w->dens_threshold){
			double test_vsigma = test_rhoLDA* (d_w->c_vsigma0 * d_rg.dfdx*d_rg.x/(2*d_sigma));
			*d_vsigma = test_vsigma;
        	}

#ifdef DEBUG 
//                        printf("rho: %.10e  sigma: %.10e  test_gdm: %.10e  test_ds: %.10e  test_rhoLDA: %.10e  d_rg->x: %.10e \n ", d_rho,d_sigma
//                        ,test_gdm, test_ds, test_rhoLDA, d_rg.x);

//			printf("rho: %.10e  sigma: %.10e  d_rg->f: %.10e  d_rg->dfdx: %.10e \n",d_rho, d_sigma, d_rg.f, d_rg.dfdx);

                        //printf("rho: %.10e  sigma: %.10e  glout->d_zk: %.10e  glout->d_vrho: %.10e  glout->d_vsigma: %.10e \n ", glin->d_rho[gid], glin->d_sigma[gid],
                        //test_zk, test_vrho, test_gdm);
	        	//printf("zk: %f, vrho: %f, vsigma: %f \n", glout->d_zk[gid], glout->d_vrho[gid], glout->d_vsigma[gid]);
#endif
        		
}

