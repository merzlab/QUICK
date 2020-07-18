#include "util.h"
#include "gpu_work.cuh"
#include "gpu_upload.cu"
#include "gpu_cleanup.cu"

gpu_libxc_info** init_gpu_libxc(int * num_of_funcs, int * arr_func_id, int* xc_polarization){

        xc_func_type hyb_func;

#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, INITIALIZING LIBXC \n", __FILE__, __LINE__, __func__);
#endif

	int num_of_funcs_;
	int * arr_func_id_;
	double * arr_mix_coeffs_;

	if(*num_of_funcs == 1){
		if(*xc_polarization > 0){
			//xc_func_init(&func, arr_func_id[0], XC_POLARIZED);
		}else{
			xc_func_init(&hyb_func, arr_func_id[0], XC_UNPOLARIZED);
		}

		if( hyb_func.info->family == XC_FAMILY_HYB_GGA){
			num_of_funcs_ = hyb_func.n_func_aux;
			//Head-gordon's dispersion functionals has been categorized as HYB but they have no aux_funcs. 
			//This should be fixed with another conditional statement. 
			if(num_of_funcs_ > 0){

				*num_of_funcs = num_of_funcs_;


#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, num_of_funcs: %d \n", __FILE__, __LINE__, __func__, *num_of_funcs);
#endif
				arr_func_id_ = (int*) malloc(sizeof(int)*num_of_funcs_);
				arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);			

				for(int i=0;i<num_of_funcs_; i++){
					arr_func_id_[i] = (hyb_func.func_aux[i])->info->number;				
#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, func_id: %d \n", __FILE__, __LINE__, __func__, arr_func_id_[i]);
#endif
					arr_mix_coeffs_[i] = hyb_func.mix_coef[i];
				
#ifdef DEBUG                    
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, mix_coeff: %f  \n", __FILE__, __LINE__, __func__, arr_mix_coeffs_[i]);
#endif
				}
			}else{
				num_of_funcs_ = *num_of_funcs;
				arr_func_id_ = arr_func_id;
				arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);
	                        for(int i=0; i<num_of_funcs_; i++){
        	                        arr_mix_coeffs_[i]=1.0;
                	        }

			}						
		}else{
			num_of_funcs_ = *num_of_funcs;
			arr_func_id_ = arr_func_id;
			arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);
			
			for(int i=0; i<num_of_funcs_; i++){
				arr_mix_coeffs_[i]=1.0;
			}	

		}
		
	//	xc_func_end(&hyb_func);
	
	}else{
		num_of_funcs_ = *num_of_funcs;
		arr_func_id_ = arr_func_id;
		arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);
		for(int i=0; i<num_of_funcs_; i++){
			arr_mix_coeffs_[i]=1.0;
#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, *num_of_funcs: %d, func_id: %d  \n", __FILE__, __LINE__, __func__, *num_of_funcs, arr_func_id_[i]);
#endif

		}
	}

	int n_bytes = num_of_funcs_*sizeof(gpu_libxc_info*);

#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, num_of_funcs_: %d, n_bytes: %d  \n", __FILE__, __LINE__, __func__, num_of_funcs_, n_bytes);
#endif

        //A device array of pointers for gpu_libxc_info data
        gpu_libxc_info** h_glinfo_array;
        cudaHostAlloc((void**)&h_glinfo_array, n_bytes, cudaHostAllocMapped);

#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, num_of_funcs_: %d  \n", __FILE__, __LINE__, __func__, num_of_funcs_);

#endif

        for(int i=0; i<num_of_funcs_; i++){
#ifdef DEBUG                    
                fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, *num_of_funcs: %d, func_id: %d  \n", __FILE__, __LINE__, __func__, *num_of_funcs, arr_func_id_[i]);
#endif

        }


        for(int i=0; i<num_of_funcs_; i++){
#ifdef DEBUG
                //printf("gpu_libxc.cu: init_gpu_libxc(): num_of_funcs: %d, arr_func_id: %d, xc_pol: %d \n", *num_of_funcs, arr_func_id[i], *xc_polarization);
#endif
		xc_func_type func;

                if(*xc_polarization > 0){
                        //xc_func_init(&func, arr_func_id[i], XC_POLARIZED);
                }else{
                        xc_func_init(&func, arr_func_id_[i], XC_UNPOLARIZED);

#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Funational: %d initialized..!, family: %d kind: %d \n", __FILE__, __LINE__, __func__, arr_func_id_[i], func.info->family, func.info->kind);
#endif	

		//Madu: Put this inside a new libxc param init method
		gpu_libxc_info* unkptr;

        	switch(func.info->family){
		case(XC_FAMILY_LDA):
			gpu_lda_work_params *ldawp;
			ldawp = (gpu_lda_work_params*)malloc(sizeof(gpu_lda_work_params));
#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Obtaining paramters \n", __FILE__, __LINE__, __func__);
#endif
			get_gpu_work_params(&func, (void*)ldawp);
#ifdef DEBUG            
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Uploading paramters \n", __FILE__, __LINE__, __func__);
#endif

#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, mix_coeff: %f  \n", __FILE__, __LINE__, __func__, arr_mix_coeffs_[i]);
#endif

			unkptr = gpu_upload_libxc_info(&func, (void*)ldawp, arr_mix_coeffs_[i], 1);
#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s Info uploaded to GPU  \n", __FILE__, __LINE__, __func__);
#endif

			break;
		
        	case(XC_FAMILY_GGA):
		case(XC_FAMILY_HYB_GGA):
                	switch(func.info->kind){
                        	case(XC_EXCHANGE):

#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, CALLING GET_GPU_WORK_PARAMS \n", __FILE__, __LINE__, __func__);
#endif
				gpu_ggax_work_params *ggaxwp;
				ggaxwp = (gpu_ggax_work_params*) malloc(sizeof(gpu_ggax_work_params));
				get_gpu_work_params(&func, (void*)ggaxwp);
				unkptr = gpu_upload_libxc_info(&func, (void*)ggaxwp, arr_mix_coeffs_[i], 1);

				break;

				case(XC_CORRELATION):
				case(XC_EXCHANGE_CORRELATION):
#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, CALLING GET_GPU_WORK_PARAMS \n", __FILE__, __LINE__, __func__);
#endif
				gpu_ggac_work_params *ggacwp;
				ggacwp = (gpu_ggac_work_params*) malloc(sizeof(gpu_ggac_work_params));
				get_gpu_work_params(&func, (void*)ggacwp);

#ifdef DEBUG                    
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, mix_coeff: %f  \n", __FILE__, __LINE__, __func__, arr_mix_coeffs_[i]);
#endif

				unkptr = gpu_upload_libxc_info(&func, (void*)ggacwp, arr_mix_coeffs_[i], 1);

				break;
			}
		break;
		}

#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, TEST ggwp func_id: %d \n", __FILE__, __LINE__, __func__, test_ggwp->func_id);
#endif
		
#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, MEM ADD unkptr: %p \n", __FILE__, __LINE__, __func__, (void*)unkptr);
#endif
        h_glinfo_array[i] = unkptr;

	xc_func_end(&func);
                }
        }

        return h_glinfo_array;

}

void libxc_cleanup(gpu_libxc_info** d_glinfo, int *n_func){

	for(int i=0; i< *n_func; i++){

#ifdef DEBUG
        fprintf(debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, LIBXC CLEANUP: %p \n", __FILE__, __LINE__, __func__, (void*)d_glinfo[i]);
#endif
	 gpu_libxc_cleanup(d_glinfo[i], NULL, NULL);

	}

	cudaFreeHost(d_glinfo);

}
