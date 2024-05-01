/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 04/06/2021                            !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions that initialize and upload      !
  ! libxc data and delete them.                                         ! 
  !---------------------------------------------------------------------!
*/

#include "util.h"
#include "gpu_upload.cu"
#include "gpu_cleanup.cu"

gpu_libxc_info** init_gpu_libxc(int * num_of_funcs, int * arr_func_id, int* xc_polarization){

        xc_func_type hyb_func;

	int num_of_funcs_;
	int * arr_func_id_;
	double * arr_mix_coeffs_;

	if(*num_of_funcs == 1){
		if(*xc_polarization > 0){
			xc_func_init(&hyb_func, arr_func_id[0], XC_POLARIZED);
		}else{
			xc_func_init(&hyb_func, arr_func_id[0], XC_UNPOLARIZED);
		}

		if( hyb_func.info->family == XC_FAMILY_HYB_GGA){
			num_of_funcs_ = hyb_func.n_func_aux;
			//Head-gordon's dispersion functionals has been categorized as HYB but they have no aux_funcs. 
			//This should be fixed with another conditional statement. 
			if(num_of_funcs_ > 0){

				*num_of_funcs = num_of_funcs_;

				arr_func_id_ = (int*) malloc(sizeof(int)*num_of_funcs_);
				arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);			

				for(int i=0;i<num_of_funcs_; i++){
					arr_func_id_[i] = (hyb_func.func_aux[i])->info->number;				
					arr_mix_coeffs_[i] = hyb_func.mix_coef[i];
				
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
		
	  xc_func_end(&hyb_func);
	
	}else{
		num_of_funcs_ = *num_of_funcs;
		arr_func_id_ = arr_func_id;
		arr_mix_coeffs_ = (double*) malloc(sizeof(double)*num_of_funcs_);
		for(int i=0; i<num_of_funcs_; i++){
			arr_mix_coeffs_[i]=1.0;
		}
	}

	int n_bytes = num_of_funcs_*sizeof(gpu_libxc_info*);

        //A device array of pointers for gpu_libxc_info data
        gpu_libxc_info** h_glinfo_array;
        cudaHostAlloc((void**)&h_glinfo_array, n_bytes, cudaHostAllocMapped);


        for(int i=0; i<num_of_funcs_; i++){
		xc_func_type func;

                if(*xc_polarization > 0){
                        xc_func_init(&func, arr_func_id_[i], XC_POLARIZED);
                }else{
                        xc_func_init(&func, arr_func_id_[i], XC_UNPOLARIZED);
                }
		//Madu: Put this inside a new libxc param init method
		gpu_libxc_info* unkptr;

        	switch(func.info->family){
		case(XC_FAMILY_LDA):
			gpu_lda_work_params *ldawp;
			ldawp = (gpu_lda_work_params*)malloc(sizeof(gpu_lda_work_params));
			get_gpu_work_params(&func, (void*)ldawp);
			unkptr = gpu_upload_libxc_info(&func, (void*)ldawp, arr_mix_coeffs_[i], 1);
			break;
		
        	case(XC_FAMILY_GGA):
		case(XC_FAMILY_HYB_GGA):
                	switch(func.info->kind){
                        	case(XC_EXCHANGE):

				gpu_ggax_work_params *ggaxwp;
				ggaxwp = (gpu_ggax_work_params*) malloc(sizeof(gpu_ggax_work_params));
				get_gpu_work_params(&func, (void*)ggaxwp);
				unkptr = gpu_upload_libxc_info(&func, (void*)ggaxwp, arr_mix_coeffs_[i], 1);

				break;

				case(XC_CORRELATION):
				case(XC_EXCHANGE_CORRELATION):
				gpu_ggac_work_params *ggacwp;
				ggacwp = (gpu_ggac_work_params*) malloc(sizeof(gpu_ggac_work_params));
				get_gpu_work_params(&func, (void*)ggacwp);

				unkptr = gpu_upload_libxc_info(&func, (void*)ggacwp, arr_mix_coeffs_[i], 1);

				break;
			}
		break;
		}

                h_glinfo_array[i] = unkptr;
	        xc_func_end(&func);
        }

        return h_glinfo_array;

}

void libxc_cleanup(gpu_libxc_info** d_glinfo, int n_func){

	for(int i=0; i< n_func; i++){

	 gpu_libxc_cleanup(d_glinfo[i]);

	}

	cudaFreeHost(d_glinfo);

}
