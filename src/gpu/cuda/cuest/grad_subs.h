#ifdef LOCAL
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"

#include "cuest_funcs.h"
#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

#define OSHELL
#endif

void
#ifdef OSHELL
cuest_init_oshell_JK_grad (int64_t dev_buf_siz)
#else
cuest_init_cshell_JK_grad (int64_t dev_buf_siz)
#endif
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = dev_buf_siz;
    quick_cuest_grad_mem.JK_vbs     = vbs;

    checkCuestErrors (cuestParametersCreate (CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.JK_par));
#ifdef OSHELL
    uint64_t *nocc_arr = malloc (2 * sizeof (uint64_t));
    nocc_arr[0]        = quick_cuest_data.nocc;
    nocc_arr[1]        = quick_cuest_data.noccb;

    checkCuestErrors (cuestDFSymmetricDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par, vbs, tmpWD, 0.5, NULL,
        -0.5, 2, nocc_arr, NULL, quick_cuest_grad_mem.d_grad));

    free (nocc_arr);
#else
    checkCuestErrors (cuestDFSymmetricDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par, vbs, tmpWD, 0.5, NULL,
        -1.0, 1, &quick_cuest_data.nocc, NULL, quick_cuest_grad_mem.d_grad));
#endif

    MEMLOG_TMPWD ("JK grad");
    quick_cuest_grad_mem.JK_wksp = allocateWorkspace (tmpWD);
}

void
#ifdef OSHELL
cuest_get_oshell_JK_grad (double *grad, double *P, double *C, double *Cb)
#else
cuest_get_cshell_JK_grad (double *grad, double *P, double *C)
#endif
{
    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestAOBasis_t              basis     = quick_cuest_struct.basis;
    uint64_t                    nbasis    = quick_cuest_data.nbasis;
    uint64_t                    natom     = quick_cuest_data.natom;
    uint64_t                    nocc      = quick_cuest_data.nocc;
#ifdef OSHELL
    uint64_t noccb = quick_cuest_data.noccb;
#endif

    const size_t grad_siz = 3 * natom * sizeof (double);
    const size_t d_P_siz  = nbasis * nbasis * sizeof (double);
    const size_t C_siz    = nocc * nbasis * sizeof (double);
    size_t       d_C_siz  = C_siz;
#ifdef OSHELL
    const size_t Cb_siz = noccb * nbasis * sizeof (double);
    d_C_siz += Cb_siz;
#endif

    void *d_P = quick_cuest_PC_buf.d_P[0];
    cudaMemcpyChecked (d_P, P, d_P_siz, cudaMemcpyHostToDevice);

    void *d_C;
    cudaMallocChecked (&d_C, d_C_siz);
    cudaMemcpyChecked (d_C, C, C_siz, cudaMemcpyHostToDevice);
#ifdef OSHELL
    cudaMemcpyChecked ((uint8_t *)d_C + C_siz, Cb, Cb_siz, cudaMemcpyHostToDevice);
#endif

#ifdef OSHELL
    uint64_t *nocc_arr = malloc (2 * sizeof (uint64_t));
    nocc_arr[0]        = nocc;
    nocc_arr[1]        = noccb;

    // 0.5 densityScale because QUICK already includes factor of 2 in P
    // -0.5 coefficientScale because alpha + beta
    checkCuestErrors (cuestDFSymmetricDerivativeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par,
        quick_cuest_grad_mem.JK_vbs, quick_cuest_grad_mem.JK_wksp, 0.5, d_P, -0.5, 2, nocc_arr, d_C,
        quick_cuest_grad_mem.d_grad));
#else
    // 0.5 densityScale because QUICK already includes factor of 2 in P
    checkCuestErrors (cuestDFSymmetricDerivativeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_grad_mem.JK_par,
        quick_cuest_grad_mem.JK_vbs, quick_cuest_grad_mem.JK_wksp, 0.5, d_P, -1.0, 1, &nocc, d_C,
        quick_cuest_grad_mem.d_grad));
#endif

    cudaMemcpyChecked (grad, quick_cuest_grad_mem.d_grad, grad_siz, cudaMemcpyDeviceToHost);

    cudaFreeChecked (d_C);
    free_dev_alloc (d_C_siz);
#ifdef OSHELL
    free (nocc_arr);
    free_host_alloc (2 * sizeof (uint64_t));
#endif
}
