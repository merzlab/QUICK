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
} quick_cuest_struct_t;

typedef struct {
    uint64_t natom;
    uint64_t nshell;
    uint64_t MAXPRIM;
    uint64_t MAXPRIM_AUX;
    uint64_t ntotalatom;
    double  *xyz;
    double  *allxyz_gpu;
    double  *allchg_gpu;
    uint64_t nao;
} quick_cuest_data_t;

extern quick_cuest_struct_t quick_cuest_struct;
extern quick_cuest_data_t   quick_cuest_data;

void cuest_init (int64_t natom, int64_t nshell, int64_t MAXPRIM, int64_t MAXPRIM_AUX, double *xyz,
                 double *chg, int64_t nextatom, double *extxyz, double *extchg);
void cuest_deinit ();

void cuest_init_basis (int64_t *ncenter, int64_t *first_basis_function,
                       int64_t *last_basis_function, int64_t *katom_, int64_t *ktype_,
                       int64_t *kprim_, double *gcexpo, double *gccoeff, bool aux);

void cuest_init_oei_plan (double cutoff);

void cuest_get_oei_S (double *o);
void cuest_get_oei_T (double *o);
void cuest_get_oei_V (double *o);

#endif
