#include "xc.h"
#include "util.h"
#include "gpu.h"
//#include "a.h"
//#include "gpu_extern.h"

int dryrun;

int main(){
	xc_func_type func;
	dryrun=0;
	int func_id = 106;
	//dryrun = 1;
	xc_func_init(&func, func_id, XC_UNPOLARIZED);

//	get_gga_gpu_params(func);
	//xc_func_type *p = &func;

	gpu_ggax_work_params unkptr;
	unkptr.ggax_maple2c_psize = 0; 
//        get_gga_gpu_params(func, (void*) &unkptr);

//	printf("a.cu: main(): unkptr->beta: %f \n", unkptr.beta);

//	xc_func_end(&func);

// --------------- Standarad Libxc run ------------------

//  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
//  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};

//  double rho[5] = {6.237884, 6.243426, 83.369802, 83.372478, 83.407005};
//  double sigma[5] ={5884.105913, 5880.549111, 1709315.925202, 1709231.328550, 1707281.362805};

  double rho[5] = {1.4617748922e-03, 1.4746881228e-03, 3.3660384191e-02, 3.7027619597e-02, 2.0129741084e-01};
  double sigma[5] ={3.4367064296e-05, 3.5210611200e-05, 3.4893029516e-02, 4.1836954986e-02, 7.7637272900e-01};

  double exc[5];
  double vrho[5];
  double vsigma[5];
  int i = 0;

  xc_gga_exc_vxc(&func, 5 , rho, sigma, exc, vrho, vsigma, (void*) &unkptr);

//  printf("a.cu: main(): unkptr -> beta: %f \n", unkptr.beta);

//-------------------------------------------------------


return 0;
}
