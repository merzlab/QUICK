#ifndef GPU_H
#define GPU_H

#ifdef __cplusplus
extern "C" {
#endif
extern int dryrun;

//Structs to hold work parameters in gpu version
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

void get_gpu_work_params(xc_func_type* p, void *gpu_work_params);

//These will load work parameters into struct variables. 
void set_gpu_ggax_work_params(double sfact, double dens_threshold, double alpha,                
        double beta, double c_zk0, double c_vrho0, double c_vrho1, double c_vrho2, double c_vsigma0, double c_vsigma1, int k_index, gpu_ggax_work_params *w);

void set_gpu_ggac_work_params(double dens_threshold, int k_index, gpu_ggac_work_params *w);

void set_gpu_lda_work_params(double dens_threshold, double cnst_rs, int xc_dim, int k_index, gpu_lda_work_params *w);

#ifdef __cplusplus
}
#endif

#endif
