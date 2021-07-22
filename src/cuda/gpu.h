/*
 *  gpu_startup.h
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */
#ifndef QUICK_GPU_H
#define QUICK_GPU_H

#include "gpu_type.h"

//Madu Manathunga 07/01/2019: Libxc header files
#include "util.h"
#include "xc_redistribute.h"

// device initial and shutdown operation
extern "C" void gpu_set_device_(int* gpu_dev_id, int* ierr);
extern "C" void gpu_startup_(int* ierr);
extern "C" void gpu_init_(int* ierr);
extern "C" void gpu_shutdown_(int* ierr);

extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
                                     int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv, int* ierr);


// molecule, basis sets, and some other information
extern "C" void gpu_upload_method_(int* quick_method, bool* is_oshell, double* hyb_coeff);
extern "C" void gpu_upload_atom_and_chg_(int* atom, QUICKDouble* atom_chg);
extern "C" void gpu_upload_cutoff_(QUICKDouble* cutMatrix, QUICKDouble* integralCutoff,QUICKDouble* primLimit, QUICKDouble* DMCutoff);
extern "C" void gpu_upload_cutoff_matrix_(QUICKDouble* YCutoff,QUICKDouble* cutPrim);
extern "C" void gpu_upload_energy_(QUICKDouble* E);
extern "C" void gpu_upload_calculated_(QUICKDouble* o, QUICKDouble* co, QUICKDouble* vec, QUICKDouble* dense);
extern "C" void gpu_upload_calculated_beta_(QUICKDouble* ob, QUICKDouble* denseb);
extern "C" void gpu_upload_basis_(int* nshell, int* nprim, int* jshell, int* jbasis, int* maxcontract, \
                                  int* ncontract, int* itype,     QUICKDouble* aexp,      QUICKDouble* dcoeff,\
                                  int* first_basis_function, int* last_basis_function, int* first_shell_basis_function, int* last_shell_basis_function, \
                                  int* ncenter,   int* kstart,    int* katom,     int* ktype,     int* kprim,  int* kshell, int* Ksumtype, \
                                  int* Qnumber,   int* Qstart,    int* Qfinal,    int* Qsbasis,   int* Qfbasis,\
                                  QUICKDouble* gccoeff,           QUICKDouble* cons,      QUICKDouble* gcexpo, int* KLMN);
extern "C" void gpu_upload_grad_(QUICKDouble* gradCutoff);
extern "C" void gpu_cleanup_();

//Following methods weddre added by Madu Manathunga
extern "C" void gpu_upload_density_matrix_(QUICKDouble* dense);
extern "C" void gpu_delete_dft_grid_();
//void upload_xc_smem();
void upload_pteval();
void reupload_pteval();
void delete_pteval(bool devOnly);
// call subroutine
// Fortran subroutine   --->  c interface    ->   kernel interface   ->    global       ->    kernel
//                            [gpu_get2e]    ->      [get2e]         -> [get2e_kernel]  ->   [iclass]

// c interface one electron integrals
extern "C" void gpu_get_oei_(QUICKDouble* o);
void getOEI(_gpu_type gpu);
void upload_sim_to_constant_oei(_gpu_type gpu);
void upload_para_to_const_oei();

// c interface [gpu_get2e]
extern "C" void get1e_();
extern "C" void get_oneen_grad_();
extern "C" void gpu_get_cshell_eri_(QUICKDouble* o);
extern "C" void gpu_get_oshell_eri_(QUICKDouble* o, QUICKDouble* ob);
extern "C" void gpu_get_cshell_xc_(QUICKDouble* Eelxc, QUICKDouble* aelec, QUICKDouble* belec, QUICKDouble *o);
extern "C" void gpu_get_oshell_xc_(QUICKDouble* Eelxc, QUICKDouble* aelec, QUICKDouble* belec, QUICKDouble *o, QUICKDouble *ob);
extern "C" void gpu_get_oshell_eri_grad_(QUICKDouble* grad);
extern "C" void gpu_get_cshell_eri_grad_(QUICKDouble* grad);
extern "C" void gpu_get_oshell_xcgrad_(QUICKDouble *grad);
extern "C" void gpu_get_cshell_xcgrad_(QUICKDouble *grad);
extern "C" void gpu_aoint_(QUICKDouble* leastIntegralCutoff, QUICKDouble* maxIntegralCutoff, int* intNum, char* intFileName);

// kernel interface [get2e]
void get2e(_gpu_type gpu);
void get_oshell_eri(_gpu_type gpu);
void getAOInt(_gpu_type gpu, QUICKULL intStart, QUICKULL intEnd, cudaStream_t streamI, int streamID,  ERI_entry* aoint_buffer);
void get_ssw(_gpu_type gpu);
void get_primf_contraf_lists(_gpu_type gpu, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight);
void getpteval(_gpu_type gpu);
void getxc(_gpu_type gpu);
void getxc_grad(_gpu_type gpu);
void prune_grid_sswgrad();
void gpu_delete_sswgrad_vars();
void get2e_MP2(_gpu_type gpu);
void getAddInt(_gpu_type gpu, int bufferSize, ERI_entry* aoint_buffer);
void getGrad(_gpu_type gpu);
void get_oshell_eri_grad(_gpu_type gpu);
// global [get2e_kernel]
__global__ void get2e_kernel();
__global__ void get2e_kernel_spdf();
__global__ void get2e_kernel_spdf2();
__global__ void get2e_kernel_spdf3();
__global__ void get2e_kernel_spdf4();
__global__ void get2e_kernel_spdf5();
__global__ void get2e_kernel_spdf6();
__global__ void get2e_kernel_spdf7();
__global__ void get2e_kernel_spdf8();
__global__ void get2e_kernel_spdf9();
__global__ void get2e_kernel_spdf10();


__global__ void getAOInt_kernel(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf2(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf3(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf4(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf5(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf6(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf7(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf8(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf9(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);
__global__ void getAOInt_kernel_spdf10(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID);


__global__ void getGrad_kernel();
__global__ void getGrad_kernel_spdf();
__global__ void getGrad_kernel_spdf2();
__global__ void getGrad_kernel_spdf3();
__global__ void getGrad_kernel_spdf4();
__global__ void getGrad_kernel_spdf5();
__global__ void getGrad_kernel_spdf6();
__global__ void getGrad_kernel_spdf7();
__global__ void getGrad_kernel_spdf8();

__global__ void get_ssw_kernel();
__global__ void get_primf_contraf_lists_kernel(unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight);
__global__ void get_pteval_kernel();
__global__ void get_density_kernel();
/*__device__ void pteval_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* phi, QUICKDouble* dphidx, QUICKDouble* dphidy,  QUICKDouble* dphidz, unsigned char *primf, unsigned int *primf_counter, int ibas, int ibasp);*/
__device__ void pteval_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* phi, QUICKDouble* dphidx, QUICKDouble* dphidy,  QUICKDouble* dphidz, int *primf, int *primf_counter, int ibas, int ibasp);
__global__ void get_sswgrad_kernel();
__global__ void getAddInt_kernel(int bufferSize, ERI_entry* aoint_buffer);


// kernel [iclass]
__device__ void iclass(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf3(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf4(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf5(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf6(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf7(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf8(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf9(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_spdf10(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);

__device__ void iclass_oshell(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf3(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf4(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf5(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf6(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf7(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf8(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf9(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_spdf10(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);

__device__ __forceinline__ void iclass_AOInt(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf3(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf4(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf5(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf6(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf7(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf8(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf9(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);
__device__ __forceinline__ void iclass_AOInt_spdf10(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID);



__device__ void iclass_grad(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf3(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf4(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf5(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf6(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf7(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_grad_spdf8(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);

__device__ void iclass_oshell_grad(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf3(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf4(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf5(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf6(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf7(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);
__device__ void iclass_oshell_grad_spdf8(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax);

void upload_sim_to_constant(_gpu_type gpu);
void upload_sim_to_constant_dft(_gpu_type gpu);

void upload_para_to_const();
char *trim(char *s);


//__device__ void gpu_shell(unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL);
__device__ void addint(QUICKULL* oULL, QUICKDouble Y, int III, int JJJ, int KKK, int LLL,QUICKDouble hybrid_coeff,  QUICKDouble* dense, int nbasis);
__device__ __forceinline__ void addint_oshell(QUICKULL* oULL, QUICKULL* obULL,QUICKDouble Y, int III, int JJJ, int KKK, int LLL,QUICKDouble hybrid_coeff,  QUICKDouble* dense, QUICKDouble* denseb,int nbasis);
__device__ void FmT(int MaxM, QUICKDouble X, QUICKDouble* vals);

__device__ QUICKDouble hrrwhole(int I, int J, int K, int L, \
                                int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ QUICKDouble hrrwhole2(int I, int J, int K, int L, \
                                int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ QUICKDouble hrrwhole2_2(int I, int J, int K, int L, \
                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ QUICKDouble hrrwhole2_3(int I, int J, int K, int L, \
                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_4(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_5(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_6(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_7(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_8(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_9(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);
__device__ QUICKDouble hrrwhole2_10(int I, int J, int K, int L, \
                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);


__device__ __forceinline__ void hrrwholegrad(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             int I, int J, int K, int L, \
                                             int III, int JJJ, int KKK, int LLL, int IJKLTYPE, \
                                             QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC,\
                                             QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                             QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                             QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                             QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ __forceinline__ void hrrwholegrad2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ __forceinline__ void hrrwholegrad2_1(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);

__device__ __forceinline__ void hrrwholegrad2_2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz);


__device__ __forceinline__ QUICKDouble quick_dsqr(QUICKDouble a);

__device__ void vertical(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                         QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                         QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                         QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                         QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                         QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                         QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical2(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                         QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                         QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                         QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                         QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                         QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                         QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical_spdf(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                              QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                              QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                              QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                              QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                              QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                              QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical_spdf2(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical_spdf3(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical_spdf4(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical_spdf5(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical_spdf6(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical_spdf7(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical_spdf8(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical_spdf9(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical_spdf10(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);


__device__ void vertical2_spdf(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                              QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                              QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                              QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                              QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                              QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                              QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical2_spdf2(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical2_spdf3(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);
__device__ void vertical2_spdf4(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical2_spdf5(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical2_spdf6(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical2_spdf7(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);

__device__ void vertical2_spdf8(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store, \
                               QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                               QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                               QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                               QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                               QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                               QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom);


__device__ int lefthrr_s(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                       QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                       int KLMNAx, int KLMNAy, int KLMNAz,
                       int KLMNBx, int KLMNBy, int KLMNBz,
                       int IJTYPE,QUICKDouble* coefAngularL, int* angularL);

__device__ int lefthrr(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                       QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                       int KLMNAx, int KLMNAy, int KLMNAz,
                       int KLMNBx, int KLMNBy, int KLMNBz,
                       int IJTYPE,QUICKDouble* coefAngularL, int* angularL);
__device__ int lefthrr23(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                        int KLMNAx, int KLMNAy, int KLMNAz,
                        int KLMNBx, int KLMNBy, int KLMNBz,
                        int IJTYPE,QUICKDouble* coefAngularL, int* angularL);

__device__ void sswder(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble Exc, QUICKDouble quadwt, QUICKULL* smemGrad, int iparent, int gid);
__device__ QUICKDouble get_unnormalized_weight(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, int iatm);

__device__ QUICKDouble SSW( QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, int atm);

//Madu Manathunga 08/20/2019
//__device__ void pt2der(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* dxdx, QUICKDouble* dxdy,
//                QUICKDouble* dxdz, QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz, int ibas);
/*__device__ void pt2der_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* dxdx, QUICKDouble* dxdy,
                QUICKDouble* dxdz, QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz, unsigned char *primf, unsigned int *primf_counter,
                 int ibas, int ibasp);*/
__device__ void pt2der_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* dxdx, QUICKDouble* dxdy,
                QUICKDouble* dxdz, QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz, int *primf, int *primf_counter,
                 int ibas, int ibasp);
//__device__ void pteval(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, 
//                       QUICKDouble* phi, QUICKDouble* dphidx, QUICKDouble* dphidy,  QUICKDouble* dphidz, 
//                       int ibas);
//__device__ void denspt_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* density, QUICKDouble* densityb,
//            QUICKDouble* gax,   QUICKDouble* gay,   QUICKDouble* gaz,   QUICKDouble* gbx,     QUICKDouble* gby,     QUICKDouble* gbz, int gid);
//__device__ void denspt(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* density, QUICKDouble* densityb, 
//                       QUICKDouble* gax,   QUICKDouble* gay,   QUICKDouble* gaz,   QUICKDouble* gbx,     QUICKDouble* gby,     QUICKDouble* gbz);
__device__ QUICKDouble b3lyp_e(QUICKDouble rho, QUICKDouble sigma);

__device__ QUICKDouble b3lypf(QUICKDouble rho, QUICKDouble sigma, QUICKDouble* dfdr);
__device__ int gen_oh(int code, int num, QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, QUICKDouble a, QUICKDouble b, QUICKDouble v);

__device__ void LD0006(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0014(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0026(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0038(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0050(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0074(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0086(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0110(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0146(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0170(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);
__device__ void LD0194(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N);


__device__ void lyp(QUICKDouble pa, QUICKDouble pb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz, QUICKDouble gbx, QUICKDouble gby, QUICKDouble gbz,
                    QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo);
__device__ void becke(QUICKDouble density, QUICKDouble gx, QUICKDouble gy, QUICKDouble gz, QUICKDouble gotherx, QUICKDouble gothery, QUICKDouble gotherz,
                      QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo);
__device__ QUICKDouble lyp_e(QUICKDouble pa, QUICKDouble pb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
                             QUICKDouble gbx,     QUICKDouble gby,      QUICKDouble gbz);

__device__ QUICKDouble becke_e(QUICKDouble density, QUICKDouble densityb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
                               QUICKDouble gbx,     QUICKDouble gby,      QUICKDouble gbz);

#endif
