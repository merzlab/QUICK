#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

/**
 * Initializes the one-electron integrals plan `OEIntPlan` in `quick_cuest_struct`
 *
 * `cuest_init` must have been called before calling `cuest_init_oei_plan`
 */
void
cuest_init_oei_plan ()
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;

    // ========================== //
    // one-electron integral plan //
    // ========================== //

    cuestOEIntPlanParameters_t oeint_plan_params;
    checkCuestErrors (cuestParametersCreate (CUEST_OEINTPLAN_PARAMETERS, &oeint_plan_params));
    checkCuestErrors (cuestOEIntPlanCreateWorkspaceQuery (
        handle, basis, quick_cuest_struct.AOPairList, oeint_plan_params, persistWD, tmpWD,
        &quick_cuest_struct.OEIntPlan));

    MEMLOG ("One-electron Integral Plan");
    quick_cuest_struct.persistOEIntPlanWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpOEIntPlanWorkspace      = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestOEIntPlanCreate (handle, basis, quick_cuest_struct.AOPairList,
                                            oeint_plan_params,
                                            quick_cuest_struct.persistOEIntPlanWorkspace,
                                            tmpOEIntPlanWorkspace, &quick_cuest_struct.OEIntPlan));

    checkCuestErrors (cuestParametersDestroy (CUEST_OEINTPLAN_PARAMETERS, oeint_plan_params));
    freeWorkspace (tmpOEIntPlanWorkspace);
}

/**
 * @param o should be `nao` x `nao`.
 */
void
cuest_get_oei_S (double *o)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;

    double *d_S;
    size_t  d_S_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_S, d_S_siz);

    cuestOverlapComputeParameters_t overlap_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_OVERLAPCOMPUTE_PARAMETERS, &overlap_compute_params));
    checkCuestErrors (cuestOverlapComputeWorkspaceQuery (handle, quick_cuest_struct.OEIntPlan,
                                                         overlap_compute_params, tmpWD, d_S));

    MEMLOG_TMPWD ("Overlap Matrix Compute");
    cuestWorkspace_t *tmpSWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestOverlapCompute (handle, quick_cuest_struct.OEIntPlan,
                                           overlap_compute_params, tmpSWorkspace, d_S));

    freeWorkspace (tmpSWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_OVERLAPCOMPUTE_PARAMETERS, overlap_compute_params));

    // ======================== //
    // copy overlap matrix to o //
    // ======================== //

    cudaMemcpyChecked (o, d_S, d_S_siz, cudaMemcpyDeviceToHost);

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- S --------\n");
    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j)
            DEBUGLOG ("%16.10f", o[i * nbasis + j]);
        DEBUGLOG ("\n");
    }
    DEBUGLOG ("------ END S ------\n");
#endif

    cudaFreeChecked (d_S);
    free_dev_alloc (d_S_siz);
}

void
cuest_get_oei_T (double *o)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;

    double *d_T;
    size_t  d_T_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_T, d_T_siz);

    cuestKineticComputeParameters_t kinetic_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_KINETICCOMPUTE_PARAMETERS, &kinetic_compute_params));
    checkCuestErrors (cuestKineticComputeWorkspaceQuery (handle, quick_cuest_struct.OEIntPlan,
                                                         kinetic_compute_params, tmpWD, d_T));

    MEMLOG_TMPWD ("Kinetic Integral Compute");
    cuestWorkspace_t *tmpTWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestKineticCompute (handle, quick_cuest_struct.OEIntPlan,
                                           kinetic_compute_params, tmpTWorkspace, d_T));

    freeWorkspace (tmpTWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_KINETICCOMPUTE_PARAMETERS, kinetic_compute_params));

    // ======================== //
    // copy kinetic matrix to o //
    // ======================== //

    cudaMemcpyChecked (o, d_T, d_T_siz, cudaMemcpyDeviceToHost);

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- T --------\n");
    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j)
            DEBUGLOG ("%16.10f", o[i * nbasis + j]);
        DEBUGLOG ("\n");
    }
    DEBUGLOG ("------ END T ------\n");
#endif

    cudaFreeChecked (d_T);
    free_dev_alloc (d_T_siz);
}

void
cuest_get_oei_V (double *o)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;

    double *d_V;
    size_t  d_V_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_V, d_V_siz);

    cuestPotentialComputeParameters_t potential_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_POTENTIALCOMPUTE_PARAMETERS, &potential_compute_params));
    checkCuestErrors (cuestPotentialComputeWorkspaceQuery (
        handle, quick_cuest_struct.OEIntPlan, potential_compute_params, tmpWD,
        quick_cuest_data.ntotalatom, quick_cuest_data.allxyz_gpu, quick_cuest_data.allchg_gpu,
        d_V));

    MEMLOG_TMPWD ("Potential Integral Compute");
    cuestWorkspace_t *tmpVWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (
        cuestPotentialCompute (handle, quick_cuest_struct.OEIntPlan, potential_compute_params,
                               tmpVWorkspace, quick_cuest_data.ntotalatom,
                               quick_cuest_data.allxyz_gpu, quick_cuest_data.allchg_gpu, d_V));

    freeWorkspace (tmpVWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_POTENTIALCOMPUTE_PARAMETERS, potential_compute_params));

    // ========================== //
    // copy potential matrix to o //
    // ========================== //

    cudaMemcpyChecked (o, d_V, d_V_siz, cudaMemcpyDeviceToHost);

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- V --------\n");
    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j)
            DEBUGLOG ("%16.10f", o[i * nbasis + j]);
        DEBUGLOG ("\n");
    }
    DEBUGLOG ("------ END V ------\n");
#endif

    cudaFreeChecked (d_V);
    free_dev_alloc (d_V_siz);
}

void
cuest_get_eri_J (double *o, double *P)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;

    double *d_J;
    size_t  d_J_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_J, d_J_siz);

    // P does not need to be corrected because S is correct already

    // density matrix
    double *d_P;
    size_t  d_P_siz = d_J_siz;
    cudaMallocChecked ((void **)&d_P, d_P_siz);

    cudaMemcpyChecked (d_P, P, d_P_siz, cudaMemcpyHostToDevice);

    cuestDFCoulombComputeParameters_t dfj_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, &dfj_compute_params));
    checkCuestErrors (cuestDFCoulombComputeWorkspaceQuery (handle, quick_cuest_struct.DFIntPlan,
                                                           dfj_compute_params, tmpWD, d_P, d_J));

    MEMLOG_TMPWD ("Coulomb Integral Compute");
    cuestWorkspace_t *tmpDFJWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestDFCoulombCompute (handle, quick_cuest_struct.DFIntPlan,
                                             dfj_compute_params, tmpDFJWorkspace, d_P, d_J));

    freeWorkspace (tmpDFJWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, dfj_compute_params));

    // ======================== //
    // copy coulomb matrix to o //
    // ======================== //

    cudaMemcpyChecked (o, d_J, d_J_siz, cudaMemcpyDeviceToHost);

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- J --------\n");
    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j)
            fprintf (quick_cuest_log_fp, "%16.10f", o[i * nbasis + j]);
        DEBUGLOG ("\n");
    }
    DEBUGLOG ("------ END J ------\n");
#endif

    cudaFreeChecked (d_J);
    free_dev_alloc (d_J_siz);
}

void
cuest_get_eri_K (double *o, double *C)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;

    double *d_K;
    size_t  d_K_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_K, d_K_siz);

    double *d_C;
    size_t  d_C_siz = nocc * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_C, d_C_siz);

    // P does not need to be corrected because S is correct already

    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

    cuestDFSymmetricExchangeComputeParameters_t dfk_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, &dfk_compute_params));

    cuestWorkspaceDescriptor_t *varBufSiz = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    varBufSiz->hostBufferSizeInBytes   = 0;
    varBufSiz->deviceBufferSizeInBytes = 2e9; // TODO(michaelyxsun): adapt this. 2 GB right now

    checkCuestErrors (cuestDFSymmetricExchangeComputeWorkspaceQuery (
        handle, quick_cuest_struct.DFIntPlan, dfk_compute_params, varBufSiz, tmpWD, nocc, d_C,
        d_K));

    MEMLOG_TMPWD ("Exchange Integral Compute");
    cuestWorkspace_t *tmpDFKWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestDFSymmetricExchangeCompute (handle, quick_cuest_struct.DFIntPlan,
                                                       dfk_compute_params, varBufSiz,
                                                       tmpDFKWorkspace, nocc, d_C, d_K));

    free (varBufSiz);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (tmpDFKWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, dfk_compute_params));

    // ========================= //
    // copy exchange matrix to o //
    // ========================= //

    cudaMemcpyChecked (o, d_K, d_K_siz, cudaMemcpyDeviceToHost);

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- K --------\n");
    for (int i = 0; i < nbasis; ++i) {
        for (int j = 0; j < nbasis; ++j)
            DEBUGLOG ("%16.10f", o[i * nbasis + j]);
        DEBUGLOG ("\n");
    }
    DEBUGLOG ("------ END K ------\n");
#endif

    cudaFreeChecked (d_K);
    cudaFreeChecked (d_C);
    free_dev_alloc (d_K_siz);
    free_dev_alloc (d_C_siz);
}
