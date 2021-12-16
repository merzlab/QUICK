#if defined HIP || defined HIP_MPIV
#include <hip/hip_runtime.h>
#endif
//#include "device_launch_parameters.h"
#include <stdio.h>
#include "util.h"

__device__ void gpu_xc_rho2dzeta(int nspin, const double rhoa, const double rhob,double *d, double *zeta){
	if(nspin==XC_UNPOLARIZED){
		*d    = max((rhoa+rhob), 0.0);//rhoa and rhob are equal quantities
		*zeta = 0.0;
	}else{
		*d = rhoa + rhob;
		if(*d > 0.0){
			*zeta = (rhoa - rhob)/(*d);
			*zeta = min(*zeta,  1.0);
			*zeta = max(*zeta, -1.0);
		}else{
			*d    = 0.0;
			*zeta = 0.0;
		}
	}	
}
