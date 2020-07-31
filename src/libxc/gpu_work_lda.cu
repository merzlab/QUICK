
typedef void (*lda_ptr)(const void *p,  xc_lda_work_t *r);

//****************** Uncomment to compile all libxc kernels*************************//
#include "gpu_fsign_lda.h"
//****************** Uncomment to compile all libxc kernels*************************//

//#define XC_LDA_X         1   /* Exchange                            */
//#define XC_LDA_C_XALPHA  6   /* Slater Xalpha                       */
//#define XC_LDA_X_RAE   549   /* Rae self-energy corrected exchange  */
//#define XC_LDA_C_VWN_RPA  8   /* Vosko, Wilk, & Nusair (RPA) */

__device__ void gpu_work_lda_c(gpu_libxc_info* glinfo, const double d_rhoa, const double d_rhob, double *d_zk, double *d_vrho, int nspin){

#ifdef DEBUG 
		 //printf("FILE: %s, LINE: %d, FUNCTION: %s, GPU_WORK_LDA.. \n", __FILE__, __LINE__, __func__);
		//printf("func_id: %d rho: %f \n", glinfo->func_id, (d_rhoa+d_rhob));
#endif	

	gpu_libxc_info* d_glinfo;
	d_glinfo = (gpu_libxc_info*)glinfo;

#ifdef DEBUG 
        	//printf("FILE: %s, LINE: %d, FUNCTION: %s, d_w \n", __FILE__, __LINE__, __func__);
#endif
	gpu_lda_work_params *d_w;
	d_w = (gpu_lda_work_params*)(d_glinfo->d_worker_params);

#ifdef DEBUG 
                //printf("FILE: %s, LINE: %d, FUNCTION: %s, got d_w \n", __FILE__, __LINE__, __func__);
#endif

	xc_lda_work_t r;
	r.order = 1;
	r.nspin = nspin;

	double dens, drs;
	gpu_xc_rho2dzeta(nspin, d_rhoa, d_rhob, &dens, &(r.z));

#ifdef DEBUG 
		//printf("func_id: %d rho: %f dens: %f r.z: %f \n", glinfo->func_id, (d_rhoa+d_rhob), dens, r.z);
                 //printf("FILE: %s, LINE: %d, FUNCTION: %s, CALLING MAPLE2C FUNCTION \n", __FILE__, __LINE__, __func__);
#endif

	if(dens > d_w->dens_threshold){

	r.rs = d_w->cnst_rs*pow(dens, -1.0/d_w->xc_dim);

#ifdef DEBUG 
                //printf("func_id: %d rho: %f dens: %f r.z: %f r.rs: %f d_w->cnst_rs: %f d_w->xc_dim: %d \n", glinfo->func_id, (d_rhoa+d_rhob), dens, r.z, r.rs, d_w->cnst_rs, d_w->xc_dim);
                 //printf("FILE: %s, LINE: %d, FUNCTION: %s, CALLING MAPLE2C FUNCTION \n", __FILE__, __LINE__, __func__);
#endif

	//if(d_glinfo->func_id == XC_LDA_C_VWN_RPA){

#ifdef DEBUG 
                 //printf("FILE: %s, LINE: %d, FUNCTION: %s, CALLING MAPLE2C FUNCTION \n", __FILE__, __LINE__, __func__);
#endif

//****************** Uncomment to compile all libxc kernels*************************//
	(maple2cf_lda[d_w->k_index])(d_glinfo->d_maple2c_params, &r);
//****************** Uncomment to compile all libxc kernels*************************//

	*d_zk = r.f;

	if(r.order > 0){
		drs = -r.rs/(d_w->xc_dim*dens);

		*d_vrho = r.f + dens*r.dfdrs*drs;
		/*if(p->nspin == XC_POLARIZED){
			vrho[1] = vrho[0] - (r.z + 1.0)*r.dfdz;
			vrho[0] = vrho[0] - (r.z - 1.0)*r.dfdz;
		}*/
	}
	

	}
}
