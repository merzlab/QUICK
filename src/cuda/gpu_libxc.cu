#include "util.h"
#include "gpu_work.cuh"

gpu_libxc_info** init_gpu_libxc(int * num_of_funcs, int * arr_func_id, int* xc_polarization){

        //A device array of pointers for gpu_libxc_info data
	gpu_libxc_info** h_glinfo_array;
	cudaHostAlloc((void**)&h_glinfo_array, *num_of_funcs*sizeof(gpu_libxc_info*), cudaHostAllocMapped);

        xc_func_type func;

#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, INITIALIZING LIBXC \n", __FILE__, __LINE__, __func__);
#endif

        for(int i=0; i< *num_of_funcs; i++){
#ifdef DEBUG
                printf("gpu_libxc.cu: init_gpu_libxc(): num_of_funcs: %d, arr_func_id: %d, xc_pol: %d \n", *num_of_funcs, arr_func_id[i], *xc_polarization);
#endif
                if(*xc_polarization > 0){
                        xc_func_init(&func, arr_func_id[i], XC_POLARIZED);
                }else{
                        xc_func_init(&func, arr_func_id[i], XC_UNPOLARIZED);

		//Madu: Put this inside a new libxc param init method
		gpu_libxc_info* unkptr;

        	switch(func.info->family){
        	case(XC_FAMILY_GGA):
			xc_gga_work_x_t* h_std_w_t;
			h_std_w_t = (xc_gga_work_x_t*)malloc(sizeof(xc_gga_work_x_t));
			h_std_w_t -> order =1;

                	switch(func.info->kind){
                        	case(XC_EXCHANGE):
				gpu_ggax_work_params *ggwp;
				ggwp = (gpu_ggax_work_params*) malloc(sizeof(gpu_ggax_work_params));
				
				get_gpu_work_params(&func, ggwp);
				unkptr = gpu_upload_libxc_info(&func, ggwp, *h_std_w_t, 1);

#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp VALUE: %f \n", __FILE__, __LINE__, __func__, ggwp->beta);
#endif
				break;
			}
		break;
		}			
		
#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, MEM ADD unkptr: %p \n", __FILE__, __LINE__, __func__, (void*)unkptr);
#endif
        h_glinfo_array[i] = unkptr;

                }
        }

        return h_glinfo_array;

}

void libxc_cleanup(gpu_libxc_info** d_glinfo, int *n_func){

	for(int i=0; i< *n_func; i++){

#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, LIBXC CLEANUP: %p \n", __FILE__, __LINE__, __func__, (void*)d_glinfo[i]);
#endif
	 gpu_libxc_cleanup(d_glinfo[i], NULL, NULL);

	}

	cudaFreeHost(d_glinfo);

}
