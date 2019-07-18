#include "util.h"
//#include "gpu_util.cu"

typedef struct{
  double alpha;       /* parameter for Xalpha functional */
} lda_x_params;

#include "maple2c/lda_x.c"

#define XC_LDA_X         1   /* Exchange                            */
#define XC_LDA_C_XALPHA  6   /* Slater Xalpha                       */
#define XC_LDA_X_RAE   549   /* Rae self-energy corrected exchange  */

#ifdef QUICK_LIBXC
__device__ void gpu_work_lda_c(gpu_libxc_info* glinfo, const double d_rhoa, const double d_rhob, double *d_zk, double *d_vrho, int nspin){

#else

#endif
	gpu_libxc_info* d_glinfo;
	d_glinfo = (gpu_libxc_info*)glinfo;

	gpu_lda_work_params *d_w;
	d_w = (gpu_lda_work_params*)(d_glinfo->d_worker_params);

	xc_lda_work_t r;
	r.order = 1;

	double dens, drs;
	//double d2rs, d3rs; //Variables required for order >1. 
	gpu_xc_rho2dzeta(nspin, d_rhoa, d_rhob, &dens, &(r.z));

	if(dens > d_w->dens_threshold){

	r.rs = d_w->cnst_rs*pow(dens, -1.0/d_w->XC_DIMENSIONS);
	
	xc_lda_x_func(d_glinfo->d_maple2c_params, &r, nspin);

	*d_zk = r.f;

	if(r.order > 0){
		drs = -r.rs/(d_w->XC_DIMENSIONS*dens);

		*d_vrho = r.f + dens*r.dfdrs*drs;
		/*if(p->nspin == XC_POLARIZED){
			vrho[1] = vrho[0] - (r.z + 1.0)*r.dfdz;
			vrho[0] = vrho[0] - (r.z - 1.0)*r.dfdz;
		}*/
	}
	
	/*if(r.order > 2){

		d2rs = -drs*(1.0 + XC_DIMENSIONS)/(XC_DIMENSIONS*dens);
		v2rho2[0] = r.dfdrs*(2.0*drs + dens*d2rs) + dens*r.d2fdrs2*drs*drs;

      		if(p->nspin == XC_POLARIZED){
        		double sign[3][2] = {{-1.0, -1.0}, {-1.0, +1.0}, {+1.0, +1.0}};

        		for(is=2; is>=0; is--){
          			v2rho2[is] = v2rho2[0] - r.d2fdrsz*(2.0*r.z + sign[is][0] + sign[is][1])*drs
            			+ (r.z + sign[is][0])*(r.z + sign[is][1])*r.d2fdz2/dens;
        		}
      		}

	}*/

	/*if(r.order > 3){

		d3rs = -d2rs*(1.0 + 2.0*XC_DIMENSIONS)/(XC_DIMENSIONS*dens);
		
		if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)){
			v3rho3[0] = r.dfdrs*(3.0*d2rs + dens*d3rs) +
			3.0*r.d2fdrs2*drs*(drs + dens*d2rs) + r.d3fdrs3*dens*drs*drs*drs;

			if(p->nspin == XC_POLARIZED){
        		double sign[4][3] = {{-1.0, -1.0, -1.0}, {-1.0, -1.0, +1.0}, {-1.0, +1.0, +1.0}, {+1.0, +1.0, +1.0}};

        		for(is=3; is>=0; is--){
          			double ff;

          			v3rho3[is]  = v3rho3[0] - (2.0*r.z  + sign[is][0] + sign[is][1])*(d2rs*r.d2fdrsz + drs*drs*r.d3fdrs2z);
          			v3rho3[is] += (r.z + sign[is][0])*(r.z + sign[is][1])*(-r.d2fdz2/dens + r.d3fdrsz2*drs)/dens;

          			ff  = r.d2fdrsz*(2.0*drs + dens*d2rs) + dens*r.d3fdrs2z*drs*drs;
          			ff += -2.0*r.d2fdrsz*drs - r.d3fdrsz2*(2.0*r.z + sign[is][0] + sign[is][1])*drs;
          			ff += (r.z + sign[is][0])*(r.z + sign[is][1])*r.d3fdz3/dens;
          			ff += (2.0*r.z  + sign[is][0] + sign[is][1])*r.d2fdz2/dens;

          			v3rho3[is] += -ff*(r.z + sign[is][2])/dens;
        		}
      			}
	}

	}*/

	}
}
