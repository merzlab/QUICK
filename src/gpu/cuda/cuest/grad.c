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

void
cuest_init_S_grad ()
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMallocChecked (&quick_cuest_grad_mem.d_dSdR, grad_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.S_par));
    checkCuestErrors (cuestOverlapDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.S_par, tmpWD, NULL,
        quick_cuest_grad_mem.d_dSdR));

    quick_cuest_grad_mem.S_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_S_grad ()
{
    freeWorkspace (quick_cuest_grad_mem.S_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.S_par));

    cudaFreeChecked (quick_cuest_grad_mem.d_dSdR);
    const size_t grad_siz = 3 * quick_cuest_data.natom * sizeof (double);
    free_dev_alloc (grad_siz);
}

void
cuest_S_grad (double *dSdR, double *P)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;
    uint64_t                    nbasis    = quick_cuest_data.nbasis;
    uint64_t                    natom     = quick_cuest_data.natom;

    void        *d_P;
    const size_t grad_siz = 3 * natom * sizeof (double);
    const size_t P_siz    = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&d_P, P_siz);
    cudaMemcpyChecked (d_P, P, P_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestOverlapDerivativeCompute (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.S_par,
        quick_cuest_grad_mem.S_wksp, d_P, quick_cuest_grad_mem.d_dSdR));

    cudaFreeChecked (d_P);
    free_dev_alloc (P_siz);

    cudaMemcpyChecked (dSdR, quick_cuest_grad_mem.d_dSdR, grad_siz, cudaMemcpyDeviceToHost);
}

void
cuest_init_T_grad ()
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMallocChecked (&quick_cuest_grad_mem.d_dTdR, grad_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.T_par));
    checkCuestErrors (cuestKineticDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.T_par, tmpWD, NULL,
        quick_cuest_grad_mem.d_dTdR));

    quick_cuest_grad_mem.T_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_T_grad ()
{
    freeWorkspace (quick_cuest_grad_mem.T_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.T_par));

    cudaFreeChecked (quick_cuest_grad_mem.d_dTdR);
    const size_t grad_siz = 3 * quick_cuest_data.natom * sizeof (double);
    free_dev_alloc (grad_siz);
}

void
cuest_T_grad (double *dTdR, double *P)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;
    uint64_t                    nbasis    = quick_cuest_data.nbasis;
    uint64_t                    natom     = quick_cuest_data.natom;

    void        *d_P;
    const size_t grad_siz = 3 * natom * sizeof (double);
    const size_t P_siz    = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&d_P, P_siz);
    cudaMemcpyChecked (d_P, P, P_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestKineticDerivativeCompute (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.T_par,
        quick_cuest_grad_mem.T_wksp, d_P, quick_cuest_grad_mem.d_dTdR));

    cudaFreeChecked (d_P);
    free_dev_alloc (P_siz);

    cudaMemcpyChecked (dTdR, quick_cuest_grad_mem.d_dTdR, grad_siz, cudaMemcpyDeviceToHost);
}

void
cuest_init_V_grad ()
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMallocChecked (&quick_cuest_grad_mem.d_dVdR_bas, grad_siz);
    cudaMallocChecked (&quick_cuest_grad_mem.d_dVdR_ptchg, grad_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.V_par));
    checkCuestErrors (cuestPotentialDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.V_par, tmpWD,
        quick_cuest_data.ntotalatom, quick_cuest_data.allxyz_gpu, quick_cuest_data.allchg_gpu, NULL,
        quick_cuest_grad_mem.d_dVdR_bas, quick_cuest_grad_mem.d_dVdR_ptchg));

    quick_cuest_grad_mem.V_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_V_grad ()
{
    freeWorkspace (quick_cuest_grad_mem.V_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.V_par));

    cudaFreeChecked (quick_cuest_grad_mem.d_dVdR_bas);
    cudaFreeChecked (quick_cuest_grad_mem.d_dVdR_ptchg);
    const size_t grad_siz = 3 * quick_cuest_data.natom * sizeof (double);
    free_dev_alloc (grad_siz);
    free_dev_alloc (grad_siz);
}

void
cuest_V_grad (double *dVdR_bas, double *dVdR_ptchg, double *P)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;
    uint64_t                    nbasis    = quick_cuest_data.nbasis;
    uint64_t                    natom     = quick_cuest_data.natom;

    void        *d_P;
    const size_t grad_siz = 3 * natom * sizeof (double);
    const size_t P_siz    = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&d_P, P_siz);
    cudaMemcpyChecked (d_P, P, P_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestPotentialDerivativeCompute (
        handle, quick_cuest_struct.OEIntPlan, quick_cuest_grad_mem.V_par,
        quick_cuest_grad_mem.V_wksp, quick_cuest_data.ntotalatom, quick_cuest_data.allxyz_gpu,
        quick_cuest_data.allchg_gpu, d_P, quick_cuest_grad_mem.d_dVdR_bas,
        quick_cuest_grad_mem.d_dVdR_ptchg));

    cudaFreeChecked (d_P);
    free_dev_alloc (P_siz);

    cudaMemcpyChecked (dVdR_bas, quick_cuest_grad_mem.d_dVdR_bas, grad_siz, cudaMemcpyDeviceToHost);
    cudaMemcpyChecked (dVdR_ptchg, quick_cuest_grad_mem.d_dVdR_ptchg, grad_siz,
                       cudaMemcpyDeviceToHost);
}

void
cuest_init_JK_grad (int64_t dev_buf_siz)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMallocChecked (&quick_cuest_grad_mem.d_dJKdR, grad_siz);

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = dev_buf_siz;
    quick_cuest_grad_mem.JK_vbs     = vbs;

    checkCuestErrors (cuestParametersCreate (CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.JK_par));
    checkCuestErrors (cuestDFSymmetricDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par, vbs, tmpWD, 0.5, NULL,
        -1.0, 1, &quick_cuest_data.nocc, NULL, quick_cuest_grad_mem.d_dJKdR));

    quick_cuest_grad_mem.JK_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_JK_grad ()
{
    freeWorkspace (quick_cuest_grad_mem.JK_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.JK_par));

    cudaFreeChecked (quick_cuest_grad_mem.d_dJKdR);
    const size_t grad_siz = 3 * quick_cuest_data.natom * sizeof (double);
    free_dev_alloc (grad_siz);
}

void
cuest_JK_grad (double *dJKdR, double *P, double *C)
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;
    uint64_t                    nbasis    = quick_cuest_data.nbasis;
    uint64_t                    natom     = quick_cuest_data.natom;
    uint64_t                    nocc      = quick_cuest_data.nocc;

    void        *d_P, *d_C;
    const size_t grad_siz = 3 * natom * sizeof (double);
    const size_t P_siz    = nbasis * nbasis * sizeof (double);
    const size_t C_siz    = nbasis * nocc * sizeof (double);
    cudaMallocChecked (&d_P, P_siz);
    cudaMallocChecked (&d_C, C_siz);
    cudaMemcpyChecked (d_P, P, P_siz, cudaMemcpyHostToDevice);
    cudaMemcpyChecked (d_C, C, C_siz, cudaMemcpyHostToDevice);

    // 0.5 densityScale because QUICK already includes factor of 2 in P
    checkCuestErrors (cuestDFSymmetricDerivativeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par,
        quick_cuest_grad_mem.JK_vbs, quick_cuest_grad_mem.JK_wksp, 0.5, d_P, -1.0, 1, &nocc, d_C,
        quick_cuest_grad_mem.d_dJKdR));

    cudaFreeChecked (d_P);
    cudaFreeChecked (d_C);
    free_dev_alloc (P_siz);
    free_dev_alloc (C_siz);

    cudaMemcpyChecked (dJKdR, quick_cuest_grad_mem.d_dJKdR, grad_siz, cudaMemcpyDeviceToHost);
}
