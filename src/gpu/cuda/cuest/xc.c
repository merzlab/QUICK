#include <alloca.h>
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

void
cuest_get_xc_grid_npoint (int64_t *npoint)
{
    if (quick_cuest_data.npoint != 0) {
        *npoint = quick_cuest_data.npoint;
        return;
    }

    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_MOLECULARGRID,
                                  quick_cuest_struct.molgrid, CUEST_MOLECULARGRID_NUM_POINT,
                                  &quick_cuest_data.npoint, sizeof (uint64_t)));
    *npoint = quick_cuest_data.npoint;
}

void
cuest_get_xc_grid_weight (int8_t weightspec, double *w)
{
    static bool    called = false;
    static double *wsave;

    if (called) {
        // npoint must have been populated already
        memcpy (w, wsave, quick_cuest_data.npoint * sizeof (double));
        return;
    }

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    cuestXCIntegrationWeightComputeParametersWeightType_t weight_type;
    switch (weightspec) {
        case WEIGHTSPEC_TOTAL:
            weight_type = CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL;
            break;
        case WEIGHTSPEC_BECKE:
            weight_type = CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_BECKE;
            break;
        case WEIGHTSPEC_QUADRATURE:
            weight_type = CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_QUADRATURE;
            break;
        default:
            fprintf (stderr, "cuest_get_xc_grid_weight: invalid weightspec %hhu\n", weightspec);
            return;
    }

    cuestXCIntegrationWeightComputeParameters_t par;
    checkCuestErrors (cuestParametersCreate (CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS, &par));

    void        *d_w;
    const size_t w_siz = npoint * sizeof (double);
    cudaMallocChecked (&d_w, w_siz);

    checkCuestErrors (cuestXCIntegrationWeightComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, weight_type, par, tmpWD, d_w));

    MEMLOG_TMPWD ("XC Integration Weight Compute");
    cuestWorkspace_t *wksp = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestXCIntegrationWeightCompute (handle, quick_cuest_struct.XCIntPlan,
                                                       weight_type, par, wksp, d_w));

    cudaMemcpyChecked (w, d_w, w_siz, cudaMemcpyDeviceToHost);
    memcpy (wsave, w, w_siz);
    called = true;

    cudaFreeChecked (d_w);
    free_dev_alloc (w_siz);
    freeWorkspace (wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS, par));
}

void
cuest_init_xc_dense (int64_t devsiz)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    size_t                                          ndim;
    cuestXCAdvancedComputeParametersApproximation_t approx;
    switch (quick_cuest_struct.fnl) {
        // GGA
        case CUEST_FUNCTIONAL_BLYP:
        case CUEST_FUNCTIONAL_B3LYP:
            ndim   = 4;
            approx = CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA;
    }
    quick_cuest_compute_mem.rho_ndim   = ndim;
    quick_cuest_compute_mem.rho_approx = approx;

    const size_t rho_siz = ndim * npoint * sizeof (double);
    cudaMallocChecked (&quick_cuest_compute_mem.d_rho, rho_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_XCDENSITYCOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.rho_par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = devsiz;
    quick_cuest_compute_mem.rho_vbs = vbs;

    checkCuestErrors (cuestXCDensityComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, approx, quick_cuest_compute_mem.rho_par, vbs, tmpWD,
        quick_cuest_data.nocc, NULL, quick_cuest_compute_mem.d_rho));

    MEMLOG_TMPWD ("XC Density Compute");
    quick_cuest_compute_mem.rho_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_xc_dense ()
{
    cudaFreeChecked (quick_cuest_compute_mem.d_rho);
    free_dev_alloc (quick_cuest_compute_mem.rho_ndim * quick_cuest_data.npoint * sizeof (double));

    free (quick_cuest_compute_mem.rho_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_compute_mem.rho_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCDENSITYCOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.rho_par));
}

void
cuest_get_xc_dense (double *C, double *rho)
{
    uint64_t ndim   = quick_cuest_compute_mem.rho_ndim;
    uint64_t npoint = quick_cuest_data.npoint;
    uint64_t nocc   = quick_cuest_data.nocc;

    cuestHandle_t handle = quick_cuest_struct.handle;

    void *d_C = quick_cuest_PC_buf.d_C[0];
    cudaMemcpyChecked (d_C, C, quick_cuest_PC_buf.C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (
        cuestXCDensityCompute (handle, quick_cuest_struct.XCIntPlan,
                               quick_cuest_compute_mem.rho_approx, quick_cuest_compute_mem.rho_par,
                               quick_cuest_compute_mem.rho_vbs, quick_cuest_compute_mem.rho_wksp,
                               quick_cuest_data.nocc, d_C, quick_cuest_compute_mem.d_rho));

    cudaMemcpyChecked (rho, quick_cuest_compute_mem.d_rho, ndim * npoint * sizeof (double),
                       cudaMemcpyDeviceToHost);
}

/*
 * wrapper to get electron density integrated over the grid
 *
 * must have initialized xc_dense
 */
void
cuest_get_xc_nelec (double *C, double *nelec)
{
    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);
    uint64_t ndim = quick_cuest_compute_mem.rho_ndim; // set by xc_dense init

    double *chk_rho_w = malloc (npoint * (ndim + 1));
    double *w         = chk_rho_w;
    double *rho       = w + npoint;

    cuest_get_xc_dense (C, rho);
    cuest_get_xc_grid_weight (WEIGHTSPEC_TOTAL, w);

    double n = 0;
    for (size_t i = 0; i < npoint; ++i)
        n += w[i] * rho[i];

    *nelec = n;
}
