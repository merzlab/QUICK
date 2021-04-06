#ifndef GPU_WORKH
#define GPU_WORKH

#define GPU_WORK_LDA 0
#define GPU_WORK_GGA_X 1
#define GPU_WORK_GGA_C 2

typedef struct{
        double *d_rho; //Input array, holds densities
        double *d_sigma; //Input array, holds the gradient of densities
}gpu_libxc_in;

//hosts device pointers reqired for libxc kernels
typedef struct {
	int func_id; //libxc functional id
	int gpu_worker; //0, 1, 2, 3 stands for LDA, GGA_X, GGA_C, HYB_GGA respectively	
	double mix_coeff; //mixing coefficient

        //common worker params
        void *d_maple2c_params;
        void *d_worker_params;

        //Specific for gga_x worker
        double *d_gdm;
        double *d_ds;
        double *d_rhoLDA;
        void *d_std_libxc_work_params;
}gpu_libxc_info;

typedef struct {
        //Output device array pointers 
        double *d_zk; //Output array, holds energy per particle
        double *d_vrho; //Output array
        double *d_vsigma; //Output array
}gpu_libxc_out;

//gpu_libxc_info* gpu_upload_libxc_info(const xc_func_type *p, void *ggwp, double mix_coeff, int np);
//void gpu_libxc_cleanup(gpu_libxc_info* d_glinfo, gpu_libxc_in* d_glin, gpu_libxc_out* d_glout);

#endif
