#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "cuest_funcs.h"
#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

static size_t                    i_atom_grids;
static cuestAtomGridParameters_t atom_grid_param;
static cuestAtomGrid_t          *atom_grids;

void
cuest_create_atom_grid_setup ()
{
    i_atom_grids = 0;

    const size_t atom_grids_siz = quick_cuest_data.natom * sizeof (cuestAtomGrid_t);
    atom_grids                  = malloc (atom_grids_siz);
    add_host_alloc (atom_grids_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_ATOMGRID_PARAMETERS, &atom_grid_param));
}

void
cuest_create_atom_grid (int64_t nrad, double *r, double *w, int64_t *nang)
{
    checkCuestErrors (cuestAtomGridCreate (quick_cuest_struct.handle, nrad, r, w, (uint64_t *)nang,
                                           atom_grid_param, &atom_grids[i_atom_grids++]));

#ifdef CUESTDEBUG
    DEBUGLOG ("created atom grid %zu\n", i_atom_grids - 1);
    DEBUGLOG ("\tnrad=%lld\n", nrad);
    DEBUGLOG ("\tr=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%f ", r[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("\tw=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%f ", w[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("\tnang=");
    for (size_t i = 0; i < nrad; ++i)
        DEBUGLOG ("%lld ", nang[i]);
    DEBUGLOG ("\n");
#endif
}

void
cuest_destroy_atom_grid ()
{

    for (int i = 0; i < quick_cuest_data.natom; ++i)
        checkCuestErrors (cuestAtomGridDestroy (atom_grids[i]));
    free (atom_grids);
    free_host_alloc (quick_cuest_data.natom * sizeof (cuestAtomGrid_t));
    checkCuestErrors (cuestParametersDestroy (CUEST_ATOMGRID_PARAMETERS, atom_grid_param));
}

#ifndef OSHELL
#define OSHELL
#include "xc_subs.h"
#endif
#undef OSHELL

#include "xc_subs.h"

void
cuest_init_xc_grad (int64_t devsiz)
{
    cuestWorkspaceDescriptor_t *tmpWD = quick_cuest_struct.tmpWD;
    uint64_t                    natom = quick_cuest_data.natom;

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMallocChecked ((void **)&quick_cuest_grad_mem.d_dxcdR, grad_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.xc_par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes   = 0;
    vbs->deviceBufferSizeInBytes = devsiz;
    quick_cuest_grad_mem.xc_vbs  = vbs;

    checkCuestErrors (cuestXCDerivativeRKSComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par, vbs,
        tmpWD, quick_cuest_data.nocc, NULL, quick_cuest_grad_mem.d_dxcdR));

    MEMLOG_TMPWD ("xc Gradient Compute");
    quick_cuest_grad_mem.xc_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_xc_grad ()
{
    const uint64_t natom = quick_cuest_data.natom;

    cudaFreeChecked (quick_cuest_grad_mem.d_dxcdR);
    free_dev_alloc (3 * natom * sizeof (double));

    free (quick_cuest_grad_mem.xc_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_grad_mem.xc_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.xc_par));
}

void
cuest_get_xc_grad (double *grad, double *C)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    double *d_C;
    size_t  d_C_siz = nocc * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&d_C, d_C_siz);
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestXCDerivativeRKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par,
        quick_cuest_grad_mem.xc_vbs, quick_cuest_grad_mem.xc_wksp, nocc, d_C,
        quick_cuest_grad_mem.d_dxcdR));

    // copy to host
    cudaMemcpyChecked (grad, quick_cuest_grad_mem.d_dxcdR, 3 * natom * sizeof (double),
                       cudaMemcpyDeviceToHost);
}
