
#define DEVICE

static const double B1 =  1.74959015598863046792081721182; /* 3*muGE/5 + 8 pi/15 */
static const double B2 = -1.62613336586517367779736042170; /* muGE - B1 */

typedef void (*ggaxk_ptr)(const void *p,  xc_gga_work_x_t *r);

#include "gpu_fstructs.h"
#include "gpu_fsign_ggaxk.h"

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

		(maple2cf_ggaxk[d_w->k_index])(d_glinfo->d_maple2c_params, &d_rg);

		double test_zk = (test_rhoLDA * d_w->c_zk0 * d_rg.f)/ d_rho;
		*d_zk = test_zk;
	
		double test_vrho = (test_rhoLDA/test_ds)* (d_w->c_vrho0 * d_rg.f + d_w->c_vrho1 * d_rg.dfdx * d_rg.x);
		*d_vrho = test_vrho;

		if(test_gdm > d_w->dens_threshold){
			double test_vsigma = test_rhoLDA* (d_w->c_vsigma0 * d_rg.dfdx*d_rg.x/(2*d_sigma));
			*d_vsigma = test_vsigma;
        	}

}

