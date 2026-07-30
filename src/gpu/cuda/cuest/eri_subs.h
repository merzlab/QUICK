#ifdef LOCAL // for lsp
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"

#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"
#endif

void
#ifdef OSHELL
cuest_get_oshell_eri_K (double *o, double *ob, double *C, double *Cb)
#else
cuest_get_cshell_eri_K (double *o, double *C)
#endif
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    cuestAOBasis_t              basis  = quick_cuest_struct.basis;

    uint64_t nbasis = quick_cuest_data.nbasis;

    void  *d_C     = quick_cuest_PC_buf.d_C[0];
    size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestDFSymmetricExchangeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.K_par,
        quick_cuest_compute_mem.K_vbs, quick_cuest_compute_mem.K_wksp, quick_cuest_data.nocc, d_C,
        quick_cuest_compute_mem.d_K));

    cudaMemcpyChecked (o, quick_cuest_compute_mem.d_K, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);

#ifdef OSHELL
    cudaMemcpyChecked (d_C, Cb, d_C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestDFSymmetricExchangeCompute (
        handle, quick_cuest_struct.DFIntPlan, quick_cuest_compute_mem.K_par,
        quick_cuest_compute_mem.K_vbs, quick_cuest_compute_mem.K_wksp, quick_cuest_data.noccb, d_C,
        quick_cuest_compute_mem.d_K));

    cudaMemcpyChecked (ob, quick_cuest_compute_mem.d_K, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
#endif
}
