#ifndef GPU_LIBXC
#define GPU_LIBXC
	

	//paramter struct for b88 functional
	typedef struct{
  		double M_CBRT3, M_CBRT4, beta, gamma, sfact, alpha, beta2, dens_threshold, c_zk;
	}param_struct;	

	typedef struct{
		double hyb_x_coeff;
	}libxc_struct;

//	extern "C" void init_gpu_libxc_(int* num_of_funcs, int* arr_func_id,  bool * xc_polarization, double* hyb_x_coeff);
	//host function that loads parameters into param_struct
//	extern "C" void set_struct_paramters_b88(param_struct *p);

#endif

