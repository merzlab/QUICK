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
                //printf("gpu_libxc.cu: init_gpu_libxc(): num_of_funcs: %d, arr_func_id: %d, xc_pol: %d \n", *num_of_funcs, arr_func_id[i], *xc_polarization);
#endif
                if(*xc_polarization > 0){
                        xc_func_init(&func, arr_func_id[i], XC_POLARIZED);
                }else{
                        xc_func_init(&func, arr_func_id[i], XC_UNPOLARIZED);

		//Madu: Put this inside a new libxc param init method
		gpu_libxc_info* unkptr;
		//void* ggwp;
        	switch(func.info->family){
		case(XC_FAMILY_LDA):
			gpu_lda_work_params *ldawp;
			ldawp = (gpu_lda_work_params*)malloc(sizeof(gpu_lda_work_params));
#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Obtaining paramters \n", __FILE__, __LINE__, __func__);
#endif
			get_gpu_work_params(&func, (void*)ldawp);
#ifdef DEBUG            
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Uploading paramters \n", __FILE__, __LINE__, __func__);
#endif
			unkptr = gpu_upload_libxc_info(&func, (void*)ldawp, 1);
			break;

        	case(XC_FAMILY_GGA):
			//xc_gga_work_x_t* h_std_w_t;
			//h_std_w_t = (xc_gga_work_x_t*)malloc(sizeof(xc_gga_work_x_t));
			//h_std_w_t -> order =1;
                	switch(func.info->kind){
                        	case(XC_EXCHANGE):
				gpu_ggax_work_params *ggaxwp;
				ggaxwp = (gpu_ggax_work_params*) malloc(sizeof(gpu_ggax_work_params));
				get_gpu_work_params(&func, (void*)ggaxwp);
				unkptr = gpu_upload_libxc_info(&func, (void*)ggaxwp, 1);

#ifdef DEBUG
       // printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp VALUE: %f \n", __FILE__, __LINE__, __func__, ggwp->beta);
#endif
				break;
				case(XC_CORRELATION):
#ifdef DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, CALLING GET_GPU_WORK_PARAMS \n", __FILE__, __LINE__, __func__);
#endif
				gpu_ggac_work_params *ggacwp;
				ggacwp = (gpu_ggac_work_params*) malloc(sizeof(gpu_ggac_work_params));
				get_gpu_work_params(&func, (void*)ggacwp);
				unkptr = gpu_upload_libxc_info(&func, (void*)ggacwp, 1);
#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp VALUE: %f \n", __FILE__, __LINE__, __func__, ggwp->func_id);
#endif

				break;
			}
		break;
		}
		//get_gpu_work_params(&func, ggwp);

		//gpu_ggac_work_params *test_ggwp;
		//test_ggwp = (gpu_ggac_work_params*) ggwp;

#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp func_id: %d \n", __FILE__, __LINE__, __func__, test_ggwp->func_id);
#endif

		//unkptr = gpu_upload_libxc_info(&func, ggwp, 1);			
		
#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, MEM ADD unkptr: %p \n", __FILE__, __LINE__, __func__, (void*)unkptr);
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
