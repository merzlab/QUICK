
#define DEVICE

static const double B1 =  1.74959015598863046792081721182; /* 3*muGE/5 + 8 pi/15 */
static const double B2 = -1.62613336586517367779736042170; /* muGE - B1 */

typedef void (*ggaxk_ptr)(const void *p,  xc_gga_work_x_t *r);

#include "gpu_fstructs.h"
#include "gpu_fsign_ggaxk.h"

__device__ void gpu_work_gga_x(gpu_libxc_info* glinfo, double d_rhoa, double d_rhob, double *d_sigma, double *d_zk, double *d_vrho, double *d_vsigma, int nspin){

        double d_rho[2]; 
        if(nspin == XC_UNPOLARIZED){
          d_rho[0] = d_rhoa+d_rhob;
          d_rho[1] = 0.0;
        }else{
          d_rho[0] = d_rhoa;
          d_rho[1] = d_rhob;          
        }

	gpu_libxc_info* d_glinfo;
	d_glinfo = (gpu_libxc_info*)glinfo;

	gpu_ggax_work_params *d_w;
	d_w = (gpu_ggax_work_params*)(d_glinfo->d_worker_params);

	xc_gga_work_x_t d_rg;
	d_rg.order = 1;

        int is, is2;
        double zk = 0.0;

        for(is =0;is<nspin; is++){
          is2 = 2*is;
          
          double gdm = max(sqrt(d_sigma[is2])/d_w->sfact, d_w->dens_threshold);
          double ds = d_rho[is]/d_w->sfact;
          double rhoLDA = pow(ds, d_w->alpha);
          d_rg.x  = gdm/pow(ds, d_w->beta);
          
          (maple2cf_ggaxk[d_w->k_index])(d_glinfo->d_maple2c_params, &d_rg);
          
          zk += (rhoLDA * d_w->c_zk0 * d_rg.f);
          
          d_vrho[is] += (rhoLDA/ds) * (d_w->c_vrho0 * d_rg.f + d_w->c_vrho1 * d_rg.dfdx * d_rg.x);
          
          if(gdm > d_w->dens_threshold){
            d_vsigma[is2] = rhoLDA * (d_w->c_vsigma0 * d_rg.dfdx * d_rg.x/(2.0 * d_sigma[is2]));
          }
        
        }

        *d_zk = zk/(d_rho[0]+d_rho[1]);

/*	double gdm = max(sqrt(d_sigma)/d_w->sfact, d_w->dens_threshold);
	double ds = d_rho/d_w->sfact;
	double rhoLDA = pow(ds, d_w->alpha);
	double rgx  = gdm/pow(ds, d_w->beta);
	d_rg.x = rgx;

	(maple2cf_ggaxk[d_w->k_index])(d_glinfo->d_maple2c_params, &d_rg);

	double zk = (rhoLDA * d_w->c_zk0 * d_rg.f)/ d_rho;
	*d_zk = zk;

	double vrho = (rhoLDA/ds)* (d_w->c_vrho0 * d_rg.f + d_w->c_vrho1 * d_rg.dfdx * d_rg.x);
	*d_vrho = vrho;

	if(gdm > d_w->dens_threshold){
		double vsigma = rhoLDA* (d_w->c_vsigma0 * d_rg.dfdx*d_rg.x/(2*d_sigma));
		*d_vsigma = vsigma;
	}
*/
}

