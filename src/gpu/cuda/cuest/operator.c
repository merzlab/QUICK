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

    cudaFreeChecked (d_V);
    free_dev_alloc (d_V_siz);
}

void
cuest_init_eri_J ()
{
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    const uint64_t              nbasis = quick_cuest_data.nbasis;

    // allocate J device buffer
    const size_t d_J_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&quick_cuest_compute_mem.d_J, d_J_siz);

    // create parameters
    checkCuestErrors (
        cuestParametersCreate (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, &quick_cuest_compute_mem.J_par));

    // allocate temp workspace
    checkCuestErrors (cuestDFCoulombComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.J_par,
        tmpWD, NULL, quick_cuest_compute_mem.d_J));

    MEMLOG_TMPWD ("Coulomb Integral Compute");
    quick_cuest_compute_mem.J_wksp = allocateWorkspace (tmpWD);

    // allocate density matrix buffer
    cudaMallocChecked (&quick_cuest_PC_buf.d_P[0], quick_cuest_PC_buf.P_siz);
}

void
cuest_deinit_eri_J ()
{
    const uint64_t nbasis = quick_cuest_data.nbasis;

    freeWorkspace (quick_cuest_compute_mem.J_wksp);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_DFCOULOMBCOMPUTE_PARAMETERS, quick_cuest_compute_mem.J_par));

    cudaFreeChecked (quick_cuest_compute_mem.d_J);
    free_dev_alloc (nbasis * nbasis * sizeof (double));

    // free P buffer
    cudaFreeChecked (quick_cuest_PC_buf.d_P[0]);
    free_dev_alloc (quick_cuest_PC_buf.P_siz);
}

void
cuest_get_eri_J (double *o, double *P)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;
    const uint64_t              nbasis = quick_cuest_data.nbasis;

    void  *d_P     = quick_cuest_PC_buf.d_P[0];
    size_t d_P_siz = quick_cuest_PC_buf.P_siz;
    cudaMemcpyChecked (d_P, P, d_P_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (
        cuestDFCoulombCompute (handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.J_par,
                               quick_cuest_compute_mem.J_wksp, d_P, quick_cuest_compute_mem.d_J));

    // copy coulomb matrix to o
    cudaMemcpyChecked (o, quick_cuest_compute_mem.d_J, d_P_siz, cudaMemcpyDeviceToHost);
}

void
cuest_init_eri_K (int64_t devsiz)
{
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    const uint64_t              nbasis = quick_cuest_data.nbasis;

    // allocate K device buffer
    const size_t d_K_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&quick_cuest_compute_mem.d_K, d_K_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.K_par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes    = 0;
    vbs->deviceBufferSizeInBytes  = devsiz;
    quick_cuest_compute_mem.K_vbs = vbs;

    checkCuestErrors (cuestDFSymmetricExchangeComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.K_par, vbs,
        tmpWD, quick_cuest_data.nocc, NULL, quick_cuest_compute_mem.d_K));

    MEMLOG_TMPWD ("Exchange Integral Compute");
    quick_cuest_compute_mem.K_wksp = allocateWorkspace (tmpWD);

    // allocate device C buffer
    cudaMallocChecked (&quick_cuest_PC_buf.d_C[0], quick_cuest_PC_buf.C_siz);
}

void
cuest_deinit_eri_K ()
{
    const uint64_t nbasis = quick_cuest_data.nbasis;

    free (quick_cuest_compute_mem.K_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));

    freeWorkspace (quick_cuest_compute_mem.K_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.K_par));

    cudaFreeChecked (quick_cuest_compute_mem.d_K);
    free_dev_alloc (nbasis * nbasis * sizeof (double));

    cudaFreeChecked (quick_cuest_PC_buf.d_C[0]);
    free_dev_alloc (quick_cuest_PC_buf.C_siz);
}

void
cuest_get_eri_K (double *o, double *C)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;

    void  *d_C     = quick_cuest_PC_buf.d_C[0];
    size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestDFSymmetricExchangeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.K_par,
        quick_cuest_compute_mem.K_vbs, quick_cuest_compute_mem.K_wksp, nocc, d_C,
        quick_cuest_compute_mem.d_K));

    // ========================= //
    // copy exchange matrix to o //
    // ========================= //

    cudaMemcpyChecked (o, quick_cuest_compute_mem.d_K, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
}
