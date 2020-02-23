
//Uploads parameters required for kernels. *p is a pointer to libxc functional, gpu_work_params is a pointer
//to host memory location containing worker paramters 
void* gpu_upload_maple2c_params(const xc_func_type *p){

        void *d_maple_params;

        printf("FILE: %s, LINE: %d, FUNCTION: %s, p->params_byte_size: %d \n",
        __FILE__, __LINE__, __func__, p->params_byte_size);

        cudaMalloc((void**)&d_maple_params, p->params_byte_size);
        cudaMemcpy(d_maple_params, (p->params), p->params_byte_size, cudaMemcpyHostToDevice);
        return d_maple_params;

}

//Uploads paramters required for kernels. *p is a pointer to libxc functional, gpu_work_params is a pointer
//to host memory location containing worker paramters 
void* gpu_upload_work_params(const xc_func_type *p, void* gpu_work_params){

        void *d_work_params;
        int work_param_size;

        //check the family
        switch(p->info->family){
	case(XC_FAMILY_LDA):
		work_param_size = sizeof(gpu_lda_work_params);
                if(GPU_DEBUG){
                        printf("FILE: %s, LINE: %d, FUNCTION: %s, lda_work_param_size: %d \n",
                        __FILE__, __LINE__, __func__, work_param_size);
                }
		break;

	case(XC_FAMILY_HYB_GGA):
        case(XC_FAMILY_GGA):
                //Now check the kind. 
                switch(p->info->kind){
                        case(XC_EXCHANGE):
                                 work_param_size = sizeof(gpu_ggax_work_params);
                        break;
				
			case(XC_CORRELATION):
			case(XC_EXCHANGE_CORRELATION):
				work_param_size = sizeof(gpu_ggac_work_params);
			break;
                }
        break;
        }

        printf("FILE: %s, LINE: %d, FUNCTION: %s, gga_work_param_size: %d \n",
        __FILE__, __LINE__, __func__, work_param_size);

        cudaMalloc((void**)&d_work_params, work_param_size);
        cudaMemcpy(d_work_params, gpu_work_params, work_param_size, cudaMemcpyHostToDevice);
        return d_work_params;

}

//This is not required for quick_libxc. Fix this later.. 

/*void* gpu_upload_std_libxc_work_params(const xc_func_type *p, void* std_libxc_work_params, int size){

        void *d_work_params;
        int total_arr_size;
        int element_size;
	void *h_work_params;
	void *tmp_h_arr;

        //check the family
        switch(p->info->family){
        case(XC_FAMILY_GGA):
                //Now check the kind. 
                switch(p->info->kind){
                        case(XC_EXCHANGE):

                                tmp_h_arr = (xc_gga_work_x_t*)std_libxc_work_params;

                                element_size = sizeof(xc_gga_work_x_t);
                                total_arr_size = size*element_size;

                                h_work_params = (xc_gga_work_x_t*)malloc(total_arr_size);

                        break;
			case(XC_CORRELATION):
                                tmp_h_arr = (xc_gga_work_c_t*)std_libxc_work_params;

                                element_size = sizeof(xc_gga_work_c_t);
                                total_arr_size = size*element_size;

                                h_work_params = (xc_gga_work_c_t*)malloc(total_arr_size);

			break;
                }
        break;
        }

	for(int i=0;i<size;i++){
		h_work_params[i] = *tmp_h_arr;
		if(GPU_DEBUG){
			printf("FILE: %s, LINE: %d, FUNCTION: %s, h_work_params[i]: %d \n",
			__FILE__, __LINE__, __func__, h_work_params[i].order);
		}
	}

	cudaMalloc((void**)&d_work_params, total_arr_size);
	cudaMemcpy(d_work_params, h_work_params, total_arr_size, cudaMemcpyHostToDevice);

        return d_work_params;

}*/

//returns a pointer to an empty device array
double* gpu_upload_libxc_out_array(int size){
        double *d_double_arr;
        int arr_size = size * sizeof(double);
        cudaMalloc((void**)&d_double_arr, arr_size);

        return d_double_arr;
}

//returns a pointer to a populated device array 
double* gpu_upload_libxc_input_array(const double *h_input, int size){
        double *d_double_arr;
        int arr_size = size * sizeof(double);
        cudaMalloc((void**)&d_double_arr, arr_size);
        cudaMemcpy(d_double_arr, h_input, arr_size, cudaMemcpyHostToDevice);

        return d_double_arr;
}

//Returns an integer that uniquly identifies the gpu worker
int get_gpu_worker(const xc_func_type *p){

        int gpu_wt = 0;

        //check the family
        switch(p->info->family){
        case(XC_FAMILY_LDA):
                gpu_wt = GPU_WORK_LDA;

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, WORKER: %d \n", __FILE__, __LINE__, __func__, gpu_wt);
        }

		break;
	case(XC_FAMILY_HYB_GGA):
        case(XC_FAMILY_GGA):
                //Now check the kind. 
                switch(p->info->kind){
                        case(XC_EXCHANGE):
                                gpu_wt = GPU_WORK_GGA_X;
                        break;
                        case(XC_CORRELATION):
			case(XC_EXCHANGE_CORRELATION):
                                gpu_wt = GPU_WORK_GGA_C;
                        break;
                }
        break;
        }

        return gpu_wt;

}

gpu_libxc_info* gpu_upload_libxc_info(const xc_func_type *p, void *ggwp, double mix_coeff, int np){
	gpu_libxc_info h_glinfo;

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, mix_coeff: %f \n", __FILE__, __LINE__, __func__, mix_coeff);
        }

	h_glinfo.func_id = p->info->number;
	h_glinfo.gpu_worker = get_gpu_worker(p);	

	h_glinfo.mix_coeff = mix_coeff;

        if(GPU_DEBUG){
                printf("FILE: %s, LINE: %d, FUNCTION: %s, mix_coeff: %f \n", __FILE__, __LINE__, __func__, h_glinfo.mix_coeff);
        }

	h_glinfo.d_maple2c_params = gpu_upload_maple2c_params(p);
	h_glinfo.d_worker_params = gpu_upload_work_params(p, ggwp);
        //allocate device memory for some work params required by gga_x worker.
          h_glinfo.d_gdm = gpu_upload_libxc_out_array(np);
          h_glinfo.d_ds = gpu_upload_libxc_out_array(np);
          h_glinfo.d_rhoLDA = gpu_upload_libxc_out_array(np);
        //h_glinfo.d_zk = gpu_upload_libxc_out(np);
        //h_glinfo.d_vrho = gpu_upload_libxc_out(np);
        //h_glinfo.d_vsigma = gpu_upload_libxc_out(np);
       // h_glinfo.d_std_libxc_work_params = gpu_upload_std_libxc_work_params(p, &h_r, np);

	gpu_libxc_info* d_glinfo;
	cudaMalloc((void**)&d_glinfo, sizeof(gpu_libxc_info));
	cudaMemcpy(d_glinfo, &h_glinfo, sizeof(gpu_libxc_info), cudaMemcpyHostToDevice);

	return d_glinfo;
}

gpu_libxc_out* gpu_upload_libxc_out(int np){
	gpu_libxc_out h_glout;
	h_glout.d_zk = gpu_upload_libxc_out_array(np);
	h_glout.d_vrho = gpu_upload_libxc_out_array(np);
	h_glout.d_vsigma = gpu_upload_libxc_out_array(np);


	gpu_libxc_out* d_glout;
	cudaMalloc((void**)&d_glout, sizeof(gpu_libxc_out));
	cudaMemcpy(d_glout, &h_glout, sizeof(gpu_libxc_out), cudaMemcpyHostToDevice);

	return d_glout;
}

gpu_libxc_in* gpu_upload_libxc_in(const double* h_rho, const double *h_sigma, int np){
        gpu_libxc_in h_glin;
        h_glin.d_rho = gpu_upload_libxc_input_array(h_rho, np);
        h_glin.d_sigma = gpu_upload_libxc_input_array(h_sigma, np);

        gpu_libxc_in* d_glin;
        cudaMalloc((void**)&d_glin, sizeof(gpu_libxc_in));
        cudaMemcpy(d_glin, &h_glin, sizeof(gpu_libxc_in), cudaMemcpyHostToDevice);

        return d_glin;
}

