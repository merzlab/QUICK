#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif
extern int dryrun;
//extern void *unk_gpu_func_params;

//typedef void(*point_kernel)(const void*,  xc_gga_work_x_t*);

//Struct to hold gga_x parameters on gpu
typedef struct{
        double sfact, dens_threshold, alpha, beta, c_zk0, c_vrho0, c_vrho1, c_vrho2, c_vsigma0, c_vsigma1;
	int ggax_maple2c_psize;
	int k_index;	
}gpu_ggax_work_params;

typedef struct{
        double dens_threshold;
	char fname[50];
        int k_index;
}gpu_ggac_work_params;

typedef struct{
	double dens_threshold;
	double cnst_rs;
	int xc_dim;
	char fname[50];
        int k_index;
}gpu_lda_work_params;

//Define a parent struct to carry all the above structs
/*typedef struct{
	int func_id;
	gpu_ggax_work_params gwp; //Parameters required for gpu work function
	void *gpu_func_params; //parameters required for maple2c codes
}gpu_libxc_d_s;
*/

//holds the parent struct containing all libxc params 
//extern gpu_libxc_d_s* glds;
void get_gpu_work_params(xc_func_type* p, void *gpu_work_params);

//This will load gga_x work parameters into a gpu_ggax_work_params struct variable and return it. 

void set_gpu_ggax_work_params(double sfact, double dens_threshold, double alpha,                
        double beta, double c_zk0, double c_vrho0, double c_vrho1, double c_vrho2, double c_vsigma0, double c_vsigma1, int k_index, gpu_ggax_work_params *w);

void set_gpu_ggac_work_params(double dens_threshold, int k_index, gpu_ggac_work_params *w);

void set_gpu_lda_work_params(double dens_threshold, double cnst_rs, int xc_dim, int k_index, gpu_lda_work_params *w);

#ifdef __cplusplus
}
#endif

#endif
