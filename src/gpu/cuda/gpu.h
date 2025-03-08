/*
 *  gpu_startup.h
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */
#if !defined(__QUICK_GPU_H_)
#define __QUICK_GPU_H_

#include "gpu_type.h"

//Madu Manathunga 07/01/2019: Libxc header files
#include "util.h"
#include "../xc_redistribute.h"


// device initial and shutdown operation
extern "C" void gpu_set_device_(int* gpu_dev_id, int* ierr);
extern "C" void gpu_startup_(int* ierr);
extern "C" void gpu_init_(int* ierr);
extern "C" void gpu_shutdown_(int* ierr);

extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
        int* gpu_num_proc, double* gpu_core_freq,char* gpu_dev_name,
        int* name_len, int* majorv, int* minorv, int* ierr);

// molecule, basis sets, and some other information
extern "C" void gpu_upload_method_(int* quick_method, bool* is_oshell, double* hyb_coeff);
extern "C" void gpu_upload_atom_and_chg_(int* atom, QUICKDouble* atom_chg);
extern "C" void gpu_upload_cutoff_(QUICKDouble* cutMatrix, QUICKDouble* integralCutoff,
        QUICKDouble* primLimit, QUICKDouble* DMCutoff, QUICKDouble* coreIntegralCutoff);
extern "C" void gpu_upload_cutoff_matrix_(QUICKDouble* YCutoff,QUICKDouble* cutPrim);
extern "C" void gpu_upload_energy_(QUICKDouble* E);
extern "C" void gpu_upload_calculated_(QUICKDouble* o, QUICKDouble* co, QUICKDouble* vec, QUICKDouble* dense);
extern "C" void gpu_upload_beta_density_matrix_(QUICKDouble* denseb);
extern "C" void gpu_upload_calculated_beta_(QUICKDouble* ob, QUICKDouble* denseb);
extern "C" void gpu_upload_basis_(int* nshell, int* nprim, int* jshell, int* jbasis,
        int* maxcontract, int* ncontract, int* itype, QUICKDouble* aexp,
        QUICKDouble* dcoeff, int* first_basis_function, int* last_basis_function,
        int* first_shell_basis_function, int* last_shell_basis_function,
        int* ncenter, int* kstart, int* katom, int* ktype, int* kprim,
        int* kshell, int* Ksumtype, int* Qnumber, int* Qstart, int* Qfinal,
        int* Qsbasis, int* Qfbasis, QUICKDouble* gccoeff, QUICKDouble* cons,
        QUICKDouble* gcexpo, int* KLMN);
extern "C" void gpu_upload_grad_(QUICKDouble* gradCutoff);
extern "C" void gpu_cleanup_();

//Following methods weddre added by Madu Manathunga
extern "C" void gpu_upload_density_matrix_(QUICKDouble* dense);
extern "C" void gpu_delete_dft_grid_();
// call subroutine
// Fortran subroutine   --->  c interface    ->   kernel interface   ->    global       ->    kernel
//                            [gpu_get2e]    ->      [get2e]         -> [get2e_kernel]  ->   [iclass]

// c interface one electron integrals
extern "C" void gpu_get_oei_(QUICKDouble* o);
void getOEI(_gpu_type gpu);
void get_oei_grad(_gpu_type gpu);
void upload_sim_to_constant_oei(_gpu_type gpu);
void upload_para_to_const_oei();

// c interface ESP
extern "C" void gpu_get_oeprop_(QUICKDouble* esp_electronic);
void getOEPROP(_gpu_type gpu);
void upload_sim_to_constant_oeprop(_gpu_type gpu);
void upload_para_to_const_oeprop();

// c interface [gpu_get2e]
extern "C" void get1e_();
extern "C" void get_oneen_grad_();
extern "C" void gpu_get_cshell_eri_(bool *deltaO, QUICKDouble* o);
extern "C" void gpu_get_oshell_eri_(bool *deltaO, QUICKDouble* o, QUICKDouble* ob);
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
#if defined(COMPILE_GPU_AOINT)
void getAOInt(_gpu_type gpu, QUICKULL intStart, QUICKULL intEnd, cudaStream_t streamI, int streamID,  ERI_entry* aoint_buffer);
#endif
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

#if defined(CEW)
void get_lri(_gpu_type gpu);
void get_lri_grad(_gpu_type gpu);
void upload_para_to_const_lri();
void getcew_quad(_gpu_type gpu);
void getcew_quad_grad(_gpu_type gpu);
#endif

void upload_sim_to_constant(_gpu_type gpu);
void upload_sim_to_constant_dft(_gpu_type gpu);
void upload_sim_to_constant_lri(_gpu_type gpu);

void upload_para_to_const();

void bind_eri_texture(_gpu_type gpu);
void unbind_eri_texture();


#endif
