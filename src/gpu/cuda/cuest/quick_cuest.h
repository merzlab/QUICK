#ifndef QUICK_CUEST_QUICK_CUEST_H
#define QUICK_CUEST_QUICK_CUEST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

// TODO: update when multigpu
#define MPI_MAX_RANKS 1

#define KTYPE_CART_S  1
#define KTYPE_CART_P  3
#define KTYPE_CART_SP 4
#define KTYPE_CART_D  6
#define KTYPE_CART_F  10

#define WEIGHTSPEC_TOTAL      0
#define WEIGHTSPEC_BECKE      1
#define WEIGHTSPEC_QUADRATURE 2

typedef struct {
    cuestHandle_t               handle;
    cuestWorkspaceDescriptor_t *persistWD;
    cuestWorkspaceDescriptor_t *tmpWD;
    cuestWorkspace_t           *persistAOBasisWorkspace;
    cuestAOBasis_t              basis;
    cuestWorkspace_t           *persistAuxBasisWorkspace;
    cuestAOBasis_t              auxBasis;
    cuestWorkspace_t           *persistAOPairListWorkspace;
    cuestAOPairList_t           AOPairList;
    cuestWorkspace_t           *persistOEIntPlanWorkspace;
    cuestOEIntPlan_t            OEIntPlan;
    cuestWorkspace_t           *persistDFIntPlanWorkspace;
    cuestDFIntPlan_t            DFIntPlan;
    cuestWorkspace_t           *persistXCGridWorkspace;
    cuestMolecularGrid_t        molgrid;
    cuestWorkspace_t           *persistXCIntPlanWorkspace;
    cuestXCIntPlan_t            XCIntPlan;
    uint16_t                    fnl;
} quick_cuest_struct_t;

typedef struct {
    cuestWorkspace_t                 *J_wksp;
    cuestDFCoulombComputeParameters_t J_par;
    void                             *d_J;

    cuestWorkspace_t                           *K_wksp;
    cuestWorkspaceDescriptor_t                 *K_vbs;
    cuestDFSymmetricExchangeComputeParameters_t K_par;
    void                                       *d_K;

    cuestWorkspace_t                      *Vxc_wksp;
    cuestWorkspaceDescriptor_t            *Vxc_vbs;
    cuestXCPotentialRKSComputeParameters_t Vxc_par;
    void                                  *d_Vxc;
    void                                  *d_Vxcb;

    cuestWorkspace_t                               *rho_wksp;
    cuestWorkspaceDescriptor_t                     *rho_vbs;
    cuestXCDensityComputeParameters_t               rho_par;
    cuestXCAdvancedComputeParametersApproximation_t rho_approx;
    void                                           *d_rho;
    uint64_t                                        rho_ndim;

    double *weights_save;
    bool    weights_saved;
} quick_cuest_compute_mem_t;

typedef struct {
    cuestWorkspace_t                         *S_wksp;
    cuestOverlapDerivativeComputeParameters_t S_par;
    void                                     *d_dSdR;

    cuestWorkspace_t                         *T_wksp;
    cuestKineticDerivativeComputeParameters_t T_par;
    void                                     *d_dTdR;

    cuestWorkspace_t                           *V_wksp;
    cuestPotentialDerivativeComputeParameters_t V_par;
    void                                       *d_dVdR_bas;
    void                                       *d_dVdR_ptchg;

    cuestWorkspace_t                             *JK_wksp;
    cuestDFSymmetricDerivativeComputeParameters_t JK_par;
    cuestWorkspaceDescriptor_t                   *JK_vbs;
    void                                         *d_dJKdR;

    cuestWorkspace_t                      *xc_wksp;
    cuestWorkspaceDescriptor_t            *xc_vbs;
    cuestXCPotentialRKSComputeParameters_t xc_par;
    void                                  *d_dxcdR;
} quick_cuest_grad_mem_t;

typedef struct {
    void  *d_P[MPI_MAX_RANKS];
    void  *d_C[MPI_MAX_RANKS];
    size_t P_siz;
    size_t C_siz;
    void  *d_Cb[MPI_MAX_RANKS];
    size_t Cb_siz;
} quick_cuest_PC_buf_t;

typedef struct {
    uint64_t natom;
    uint64_t nshell;
    uint64_t nbasis;
    uint64_t nocc;
    uint64_t noccb;
    uint64_t nauxshell;
    uint64_t maxcontract;
    uint64_t maxcontract_aux;
    uint64_t ntotalatom;
    double  *xyz;
    double  *allxyz_gpu;
    double  *allchg;
    double  *allchg_gpu;
    size_t  *ifshell;
    int8_t  *iattype;
    uint64_t npoint;
} quick_cuest_data_t;

typedef struct {
    uint64_t *chk_katom_ktype_kprim; // initialized in cuest_init
    void     *chk_firstd_firstf; // used for reordering density matrix and molecular coefficients
    size_t    ifd;               // ^^ also
    size_t    iff;               // ^^ also
    double   *tmpbuf_dp;         // ^^ also
} quick_cuest_memchk_t;

typedef struct {
    size_t hostmax;    // max memory allocated in host
    size_t hosttotal;  // total memory allocated in host
    size_t hostallocs; // number of host allocations
    size_t hostcur;    // current memory allocation
    size_t devmax;     // max memory allocated in device
    size_t devtotal;   // total memory allocated in device
    size_t devallocs;  // number of device allocations
    size_t devcur;     // current memory allocation
} quick_cuest_memtrace_t;

extern quick_cuest_struct_t      quick_cuest_struct;
extern quick_cuest_compute_mem_t quick_cuest_compute_mem;
extern quick_cuest_PC_buf_t      quick_cuest_PC_buf;
extern quick_cuest_grad_mem_t    quick_cuest_grad_mem;
extern quick_cuest_data_t        quick_cuest_data;
extern quick_cuest_memchk_t      quick_cuest_memchk;
extern quick_cuest_memtrace_t    quick_cuest_memtrace;
extern FILE                     *quick_cuest_log_fp;

void cuest_init (int64_t natom, int64_t nshell, int64_t nbasis, int64_t nocca, int64_t noccb,
                 int64_t nauxshell, int64_t maxcontract, int64_t maxcontract_aux, int8_t *iattype,
                 double *xyz, double *chg, int64_t nextatom, double *extxyz, double *extchg);
// void cuest_init (int64_t natom, int64_t nshell, int64_t nbasis, int64_t nocc, int64_t nauxshell,
//                  int64_t maxcontract, int64_t maxcontract_aux, int8_t *iattype, double *xyz,
//                  double *chg, int64_t nextatom, double *extxyz, double *extchg);

/** Deinitializes the basis set, pair list, and DF integral plan */
void cuest_deinit ();

void cuest_init_basis (int64_t *ncenter, int64_t *katom_, int64_t *ktype_, int64_t *kprim_,
                       double *aexp, double *dcoeff, bool aux);

void cuest_init_pair_list (double cutoff);

void cuest_init_oei_plan ();
void cuest_deinit_oei_plan ();

void cuest_init_df (double hyb_coeff);
void cuest_deinit_df ();

// operator matrix
void cuest_get_oei_S (double *o);
void cuest_get_oei_T (double *o);
void cuest_get_oei_V (double *o);

void cuest_init_eri_J ();
void cuest_deinit_eri_J ();
void cuest_get_eri_J (double *o, double *P);

void cuest_init_eri_K (int64_t devsiz);
void cuest_deinit_eri_K ();
void cuest_get_cshell_eri_K (double *o, double *C);
void cuest_get_oshell_eri_K (double *o, double *ob, double *C, double *Cb);

// DFT
void cuest_create_atom_grid_setup ();
void cuest_create_atom_grid (int64_t nrad, double *r, double *w, int64_t *nang);
void cuest_destroy_atom_grid ();

void cuest_init_cshell_xc (int8_t fnl, int64_t devsiz);
void cuest_init_oshell_xc (int8_t fnl, int64_t devsiz);
void cuest_deinit_cshell_xc ();
void cuest_deinit_oshell_xc ();
void cuest_get_cshell_xc (double *Vxc, double *Exc, double *C);
void cuest_get_oshell_xc (double *Vxc, double *Vxcb, double *Exc, double *C, double *Cb);

void cuest_get_xc_grid_npoint (int64_t *npoint);
void cuest_get_xc_grid_weight (int8_t weightspec, double *w);
void cuest_init_xc_dense (int64_t devsiz);
void cuest_deinit_xc_dense ();
void cuest_get_xc_dense (double *C, double *rho);
void cuest_get_xc_nelec (double *C, double *nelec);

// ======== //
// Gradient //
// ======== //

void cuest_init_S_grad ();
void cuest_deinit_S_grad ();
void cuest_S_grad (double *dSdR, double *P);
void cuest_init_T_grad ();
void cuest_deinit_T_grad ();
void cuest_T_grad (double *dTdR, double *P);
void cuest_init_V_grad ();
void cuest_deinit_V_grad ();
void cuest_V_grad (double *dVdR_bas, double *dVdR_ptchg, double *P);

void cuest_init_JK_grad (int64_t dev_buf_siz);
void cuest_deinit_JK_grad ();
void cuest_get_JK_grad (double *dJKdR, double *P, double *C);

void cuest_init_xc_grad (int64_t devsiz);
void cuest_deinit_xc_grad ();
void cuest_get_xc_grad (double *grad, double *C);

// Other
void cuest_debuglog (const char *str);
void cuest_debuglog_flush ();

#endif
