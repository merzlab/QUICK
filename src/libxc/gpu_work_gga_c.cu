#include "util.h"
#include "gpu_util.cu"

typedef void (*ggac_ptr)(const void *p,  xc_gga_work_c_t *r);

#include "gpu_fsign_ggac.h"

__device__ void gpu_work_gga_c(gpu_libxc_info* glinfo, const double d_rhoa, const double d_rhob, double d_sigma, double *d_zk, double *d_vrho, double *d_vsigma, int nspin){


		gpu_libxc_info* d_glinfo;
		d_glinfo = (gpu_libxc_info*)glinfo;

		//Define gpu_ggac_work_params 
		gpu_ggac_work_params *d_w;
		d_w = (gpu_ggac_work_params*)(d_glinfo->d_worker_params);

		xc_gga_work_c_t r;
		r.order = 1;		

		double min_grad2 = d_w->dens_threshold * d_w->dens_threshold;
		double drs, dxtdn, dxtds, dxsdn[2], dxsds[2];

		gpu_xc_rho2dzeta(nspin, d_rhoa, d_rhob, &(r.dens), &(r.z));

		if(r.dens > d_w->dens_threshold){
		
			r.rs = RS(r.dens);
			if(nspin == XC_UNPOLARIZED){

				r.ds[0]  = r.dens/2.0;
				r.ds[1]  = r.ds[0];
				
				//r.sigmat = max(min_grad2, sigma[0]);
				r.sigmat = max(min_grad2, d_sigma);
				r.xt     = sqrt(r.sigmat)/ pow(r.dens, 4.0/3.0);

				r.sigmas[0] = r.sigmat/4.0;
				r.sigmas[1] = r.sigmas[0];
				r.sigmas[2] = r.sigmas[0];

				r.xs[0]  = CBRT(2.0)*r.xt;
				r.xs[1]  = r.xs[0];
			}
			else{
				/*if(1.0 + r.z < DBL_EPSILON) r.z = -1.0 + DBL_EPSILON;
				if(1.0 - r.z < DBL_EPSILON) r.z =  1.0 - DBL_EPSILON;

				r.ds[0]  = max(d_w->dens_threshold, d_rhoa);
				r.ds[1]  = max(d_w->dens_threshold, d_rhob);

				r.sigmat = max(min_grad2, sigma[0] + 2.0*sigma[1] + sigma[2]);
				r.xt     = sqrt(r.sigmat)/ pow(r.dens, 4.0/3.0);

				r.sigmas[0] = max(min_grad2, sigma[0]);
				r.sigmas[1] = max(min_grad2, sigma[1]);
				r.sigmas[2] = max(min_grad2, sigma[2]);

				r.xs[0] = sqrt(r.sigmas[0])/pow(r.ds[0], 4.0/3.0);
				r.xs[1] = sqrt(r.sigmas[2])/pow(r.ds[1], 4.0/3.0); */
			}

				(maple2cf_ggac[d_w->k_index])(d_glinfo->d_maple2c_params, &r);

			*d_zk = r.f;

			if(r.order > 0){
				/* setup auxiliary variables */
				drs   =     -r.rs/(3.0*r.dens);
				dxtdn = -4.0*r.xt/(3.0*r.dens);
				dxtds = r.xt/(2.0*r.sigmat);

				
				if(nspin == XC_POLARIZED){
					/*ndzdn[1] = -(r.z + 1.0);
					ndzdn[0] = -(r.z - 1.0);

					dxsdn[1] = -4.0/3.0*r.xs[1]/r.ds[1];
					dxsdn[0] = -4.0/3.0*r.xs[0]/r.ds[0];

					dxsds[1] = r.xs[1]/(2.0*r.sigmas[2]);
					dxsds[0] = r.xs[0]/(2.0*r.sigmas[0]);*/
				}else{
					double m_cbrt2 = (double)  M_CBRT2;
					dxsdn[0] = m_cbrt2 * dxtdn;
					dxsds[0] = m_cbrt2 * dxtds;
				}
  
  				//if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
  					//vrho[0]   = r.f + r.dens*(r.dfdrs*drs + r.dfdxt*dxtdn);
					//vsigma[0] = r.dens*r.dfdxt*dxtds;
					*d_vrho = r.f + r.dens*(r.dfdrs*drs + r.dfdxt*dxtdn);
					*d_vsigma = r.dens*r.dfdxt*dxtds;

  					if(nspin == XC_POLARIZED){
  						/*vrho[1] = vrho[0] + r.dfdz*ndzdn[1] + r.dens*r.dfdxs[1]*dxsdn[1];
  						vrho[0] = vrho[0] + r.dfdz*ndzdn[0] + r.dens*r.dfdxs[0]*dxsdn[0];
  
  						vsigma[2] = vsigma[0] + r.dens*r.dfdxs[1]*dxsds[1];
  						vsigma[1] = 2.0*vsigma[0];
  						vsigma[0] = vsigma[0] + r.dens*r.dfdxs[0]*dxsds[0];*/
  
  					}else{
						*d_vrho   += 2.0*r.dens*r.dfdxs[0]*dxsdn[0];
						*d_vsigma += 2.0*r.dens*r.dfdxs[0]*dxsds[0];

  					}
				//}


			}

		}
}
