#ifndef QUICK_CUEST_QUICK_CUEST_H
#define QUICK_CUEST_QUICK_CUEST_H

#include <stdbool.h>
#include <stdint.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#define KTYPE_CART_S  1
#define KTYPE_CART_P  3
#define KTYPE_CART_SP 4
#define KTYPE_CART_D  6
#define KTYPE_CART_F  10

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
} quick_cuest_struct_t;

typedef struct {
    uint64_t natom;
    uint64_t nshell;
    uint64_t nbasis;
    uint64_t nauxshell;
    uint64_t maxcontract;
    uint64_t maxcontract_aux;
    uint64_t ntotalatom;
    double  *xyz;
    double  *allxyz_gpu;
    double  *allchg;
    double  *allchg_gpu;
    size_t  *ifshell;
} quick_cuest_data_t;

typedef struct {
    double   *tmp_o;                 // initialized to nbasis x nbasis by cuest_init
    double   *tmp_C;                 // initialized to NULL, must be manually malloced
    uint64_t *chk_katom_ktype_kprim; // initialized in cuest_init
    void     *chk_firstd_firstf; // used for reordering density matrix and molecular coefficients
    size_t    ifd;               // ^^ also
    size_t    iff;               // ^^ also
    double   *tmpbuf_dp;         // ^^ also
} quick_cuest_memchk_t;

extern quick_cuest_struct_t quick_cuest_struct;
extern quick_cuest_data_t   quick_cuest_data;
extern quick_cuest_memchk_t quick_cuest_memchk;

void cuest_init (int64_t natom, int64_t nshell, int64_t nbasis, int64_t nauxshell,
                 int64_t maxcontract, int64_t maxcontract_aux, double *xyz, double *chg,
                 int64_t nextatom, double *extxyz, double *extchg);
/** Deinitializes the basis set, pair list, and DF integral plan */
void cuest_deinit ();

/** Must be called after initializing main basis set */
void init_correct ();
void deinit_correct ();

void cuest_init_basis (int64_t *ncenter, int64_t *katom_, int64_t *ktype_, int64_t *kprim_,
                       double *aexp, double *dcoeff, bool aux);

void cuest_init_pair_list (double cutoff);

void cuest_init_oei_plan ();
void cuest_deinit_oei_plan ();

void cuest_init_dfint_plan (double hyb_coeff);

void cuest_get_oei_S (double *o);
void cuest_get_oei_T (double *o);
void cuest_get_oei_V (double *o);
void cuest_get_eri_J (double *o, double *D);
void cuest_get_eri_K (double *o, double *C, int64_t NBSuse);

#endif
