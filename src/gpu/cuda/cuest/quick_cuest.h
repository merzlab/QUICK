#ifndef QUICK_CUEST_QUICK_CUEST_H
#define QUICK_CUEST_QUICK_CUEST_H

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
    cuestWorkspace_t           *persistAOPairListWorkspace;
    cuestAOPairList_t           AOPairList;
    cuestWorkspace_t           *persistOEIntPlanWorkspace;
    cuestOEIntPlan_t            OEIntPlan;
} quick_cuest_struct_t;

typedef struct {
    uint64_t natom;
    uint64_t nshell;
    uint64_t MAXPRIM;
    double  *xyz;
} quick_cuest_data_t;

extern quick_cuest_struct_t quick_cuest_struct;
extern quick_cuest_data_t   quick_cuest_data;

void cuest_init (uint64_t natom, uint64_t nshell, uint64_t MAXPRIM, double *xyz);
void cuest_deinit ();

void cuest_init_basis (uint64_t *ncenter, uint64_t *first_basis_function,
                       uint64_t *last_basis_function, uint64_t *katom_, uint64_t *ktype_,
                       uint64_t *kprim_, double *gcexpo, double *gccoeff);

void cuest_init_oei_plan ();

void cuest_get_oei_S (double *o);
void cuest_get_oei_V (double *o);

#endif
