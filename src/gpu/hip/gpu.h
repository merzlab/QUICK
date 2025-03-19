/*
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 */
#if !defined(__QUICK_GPU_H_)
#define __QUICK_GPU_H_

#include "gpu_type.h"

#include "util.h"
#include "../xc_redistribute.h"


extern "C" void gpu_set_device_(int *, int *);
extern "C" void gpu_new(
#if defined(MPIV_GPU)
        int,
#endif
        int *);
extern "C" void gpu_init_device_(int *);
extern "C" void gpu_delete_(int *);
extern "C" void gpu_get_device_info_(int *, int *, int *,
        int *, double *, char *, int *, int *, int *, int *);

// molecule, basis sets, and some other information
extern "C" void gpu_upload_method_(int *, bool *, double *);
extern "C" void gpu_upload_atom_and_chg_(int *, QUICKDouble *);
extern "C" void gpu_upload_cutoff_(QUICKDouble *, QUICKDouble *, QUICKDouble *,
        QUICKDouble *, QUICKDouble *);
extern "C" void gpu_upload_cutoff_matrix_(QUICKDouble *, QUICKDouble *);
extern "C" void gpu_upload_energy_(QUICKDouble *);
extern "C" void gpu_upload_calculated_(QUICKDouble *, QUICKDouble *, QUICKDouble *, QUICKDouble *);
extern "C" void gpu_upload_beta_density_matrix_(QUICKDouble *);
extern "C" void gpu_upload_calculated_beta_(QUICKDouble *, QUICKDouble *);
extern "C" void gpu_upload_basis_(int *, int *, int *, int *,
        int *, int *, int *, QUICKDouble *, QUICKDouble *, int *, int *,
        int *, int *, int *, int *, int *, int *, int *,
        int *, int *, int *, int *, int *, int *, int *, QUICKDouble *, QUICKDouble *,
        QUICKDouble *, int *);
extern "C" void gpu_upload_grad_(QUICKDouble *);
extern "C" void gpu_cleanup_();

extern "C" void gpu_upload_density_matrix_(QUICKDouble *);
extern "C" void gpu_delete_dft_grid_();
// call subroutine
// Fortran subroutine   --->  c interface    ->   kernel interface   ->    global       ->    kernel
//                            [gpu_get2e]    ->      [get2e]         -> [get2e_kernel]  ->   [iclass]

// c interface one electron integrals
extern "C" void gpu_get_oei_(QUICKDouble *);
void getOEI(_gpu_type);
void get_oei_grad(_gpu_type);
void upload_sim_to_constant_oei(_gpu_type);
void upload_para_to_const_oei();

// c interface [gpu_get2e]
extern "C" void get1e_();
extern "C" void get_oneen_grad_();
extern "C" void gpu_get_cshell_eri_(bool *, QUICKDouble *);
extern "C" void gpu_get_oshell_eri_(bool *, QUICKDouble *, QUICKDouble *);
extern "C" void gpu_get_cshell_xc_(QUICKDouble *, QUICKDouble *, QUICKDouble *, QUICKDouble *);
extern "C" void gpu_get_oshell_xc_(QUICKDouble *, QUICKDouble *, QUICKDouble *, QUICKDouble *, QUICKDouble *);
extern "C" void gpu_get_oshell_eri_grad_(QUICKDouble *);
extern "C" void gpu_get_cshell_eri_grad_(QUICKDouble *);
extern "C" void gpu_get_oshell_xcgrad_(QUICKDouble *);
extern "C" void gpu_get_cshell_xcgrad_(QUICKDouble *);
extern "C" void gpu_aoint_(QUICKDouble *, QUICKDouble *, int *, char *);

// kernel interface [get2e]
void get2e(_gpu_type);
void get_oshell_eri(_gpu_type);
#if defined(COMPILE_GPU_AOINT)
void getAOInt(_gpu_type, QUICKULL, QUICKULL, hipStream_t, int, ERI_entry *);
#endif
void get_ssw(_gpu_type gpu);
void get_primf_contraf_lists(_gpu_type, unsigned char *, unsigned int *, unsigned int *);
void getpteval(_gpu_type);
void getxc(_gpu_type);
void getxc_grad(_gpu_type);
void prune_grid_sswgrad();
void gpu_delete_sswgrad_vars();
void get2e_MP2(_gpu_type);
void getAddInt(_gpu_type, int, ERI_entry *);
void getGrad(_gpu_type);
void get_oshell_eri_grad(_gpu_type);

#if defined(CEW)
void get_lri(_gpu_type);
void get_lri_grad(_gpu_type);
void upload_para_to_const_lri();
void getcew_quad(_gpu_type);
void getcew_quad_grad(_gpu_type);
#endif

void upload_sim_to_constant(_gpu_type);
void upload_sim_to_constant_dft(_gpu_type);
void upload_sim_to_constant_lri(_gpu_type);

void upload_para_to_const();

void bind_eri_texture(_gpu_type);
void unbind_eri_texture();


#endif
