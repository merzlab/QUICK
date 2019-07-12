#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 
#include "xc.h"
#include "gpu_libxc.h"
#include "util.h"

extern "C" void init_gpu_libxc_(int * num_of_funcs, int * arr_func_id, bool * xc_polarization, double* hyb_x_coeff){
	xc_func_type func;

#ifdef DEBUG
	printf("gpu_libxc.cu: init_gpu_libxc() \n");
#endif	
	for(int i=0; i< *num_of_funcs; i++){
#ifdef DEBUG
		printf("gpu_libxc.cu: init_gpu_libxc(): num_of_funcs: %d, arr_func_id: %d \n", *num_of_funcs, arr_func_id[i]);
#endif
		if(*xc_polarization){
			xc_func_init(&func, arr_func_id[i], XC_POLARIZED);
		}else{
			xc_func_init(&func, arr_func_id[i], XC_UNPOLARIZED);

			gpu_ggax_work_params unkptr;
			
			get_gga_gpu_params(func, (void*)&unkptr);
#ifdef DEBUG
			printf("FILE: %s, LINE: %d, FUNCTION: %s \n", __FILE__, __LINE__, __func__);
#endif
		}
	}
}

