#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-headers/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "cuest_funcs.h"
#include "helper_status.h"
#include "helper_workspace.h"
#include "quick_cuest.h"
#include "util.h"

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

#ifdef CUESTDEBUG
    DEBUGLOG ("Number of grid points: %lld\n", *npoint);
#endif
}

void
cuest_get_xc_grid_weight (int8_t weightspec, double *w)
{
    if (quick_cuest_compute_mem.weights_saved) {
        // npoint must have been populated already
        memcpy (w, quick_cuest_compute_mem.weights_save, quick_cuest_data.npoint * sizeof (double));
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
    quick_cuest_compute_mem.weights_save = malloc (w_siz);
    memcpy (quick_cuest_compute_mem.weights_save, w, w_siz);
    quick_cuest_compute_mem.weights_saved = true;

    cudaFreeChecked (d_w);
    free_dev_alloc (w_siz);
    freeWorkspace (wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS, par));
}

void
get_cuest_approx (int8_t approxspec, cuestXCAdvancedComputeParametersApproximation_t *cuestapprox,
                  size_t *ndim)
{
    switch (approxspec) {
        case APPROXSPEC_GGA:
            *cuestapprox = CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA;
            *ndim        = 4;
            break;
        case APPROXSPEC_METAGGA:
            *cuestapprox = CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA;
            *ndim        = 5;
            break;
        case APPROXSPEC_LDA:
        default:
            *cuestapprox = CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA;
            *ndim        = 1;
    }
}

void
cuest_init_xc_dense (int8_t approxspec, int64_t devsiz)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    size_t                                          ndim;
    cuestXCAdvancedComputeParametersApproximation_t approx;
    get_cuest_approx (approxspec, &approx, &ndim);
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

    double *chk_rho_w = malloc (npoint * (ndim + 1) * sizeof (double));
    double *w         = chk_rho_w;
    double *rho       = w + npoint;

    cuest_get_xc_dense (C, rho);
    cuest_get_xc_grid_weight (WEIGHTSPEC_TOTAL, w);

    double n = 0;
    for (size_t i = 0; i < npoint; ++i)
        n += w[i] * rho[i * ndim];

    *nelec = n;

    free (chk_rho_w);
}

/**
 * does not use separate init/deinit because we expect this to only be called
 * once
 */
void
cuest_get_grid_xyz (double *xyz)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    cuestXCIntegrationGridComputeParameters_t par;
    checkCuestErrors (cuestParametersCreate (CUEST_XCINTEGRATIONGRIDCOMPUTE_PARAMETERS, &par));

    void        *d_xyz;
    const size_t xyz_siz = 3 * npoint * sizeof (double);
    cudaMallocChecked (&d_xyz, xyz_siz);

    checkCuestErrors (cuestXCIntegrationGridComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, par, tmpWD, d_xyz));

    MEMLOG ("XC Grid Position");
    cuestWorkspace_t *wksp = allocateWorkspace (tmpWD);

    checkCuestErrors (
        cuestXCIntegrationGridCompute (handle, quick_cuest_struct.XCIntPlan, par, wksp, d_xyz));

    cudaMemcpyChecked (xyz, d_xyz, xyz_siz, cudaMemcpyDeviceToHost);

    cudaFree (d_xyz);
    free_dev_alloc (xyz_siz);
    freeWorkspace (wksp);
    cuestParametersDestroy (CUEST_XCINTEGRATIONGRIDCOMPUTE_PARAMETERS, &par);
}

void
cuest_get_xc_potential (double *Vgrid, int64_t devsiz, int8_t approxspec, double *V)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    nbasis = quick_cuest_data.nbasis;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    size_t                                          ndim;
    cuestXCAdvancedComputeParametersApproximation_t approx;
    get_cuest_approx (approxspec, &approx, &ndim);

    cuestXCPotentialComputeParameters_t par;
    checkCuestErrors (cuestParametersCreate (CUEST_XCPOTENTIALCOMPUTE_PARAMETERS, &par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes   = 0;
    vbs->deviceBufferSizeInBytes = devsiz;

    void        *d_Vgrid, *d_V;
    const size_t Vgrid_siz = npoint * ndim * sizeof (double);
    const size_t V_siz     = nbasis * nbasis * sizeof (double);
    cudaMallocChecked (&d_Vgrid, Vgrid_siz);
    cudaMallocChecked (&d_V, V_siz);

    checkCuestErrors (cuestXCPotentialComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, approx, par, vbs, tmpWD, d_Vgrid, d_V));

    MEMLOG ("XC Grid Potential");
    cuestWorkspace_t *wksp = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestXCPotentialCompute (handle, quick_cuest_struct.XCIntPlan, approx, par,
                                               vbs, wksp, d_Vgrid, d_V));

    cudaMemcpyChecked (V, d_V, V_siz, cudaMemcpyDeviceToHost);

    cudaFreeChecked (d_V);
    cudaFreeChecked (d_Vgrid);
    free_dev_alloc (V_siz);
    free_dev_alloc (Vgrid_siz);
    freeWorkspace (wksp);
    cuestParametersDestroy (CUEST_XCPOTENTIALCOMPUTE_PARAMETERS, &par);
}

void
cuest_init_grid_der (int8_t approxspec, int64_t devsiz)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    size_t                                          ndim;
    cuestXCAdvancedComputeParametersApproximation_t approx;
    get_cuest_approx (approxspec, &approx, &ndim);

    quick_cuest_compute_mem.der_ndim   = ndim;
    quick_cuest_compute_mem.der_approx = approx;

    checkCuestErrors (cuestParametersCreate (CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.der_par));
    checkCuestErrors (cuestParametersCreate (CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.gridder_par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = devsiz;
    quick_cuest_compute_mem.der_vbs = vbs;

    void *d_basisgrad;
    void *d_gridgrad;
    void *d_v;
    cudaMallocChecked (&d_basisgrad, 3 * natom * sizeof (double));
    cudaMallocChecked (&d_gridgrad, 3 * npoint * sizeof (double));
    cudaMallocChecked (&d_v, ndim * npoint * sizeof (double));
    quick_cuest_compute_mem.d_basisgrad = d_basisgrad;
    quick_cuest_compute_mem.d_gridgrad  = d_gridgrad;
    quick_cuest_compute_mem.d_der_v     = d_v;

    checkCuestErrors (cuestXCDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, approx, quick_cuest_compute_mem.der_par, vbs, tmpWD,
        quick_cuest_data.nocc, NULL, NULL, d_basisgrad, d_gridgrad));

    MEMLOG_TMPWD ("XC Derivative Compute");
    quick_cuest_compute_mem.der_wksp = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestXCGridDerivativeComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.gridder_par, tmpWD, NULL,
        NULL, d_basisgrad));

    MEMLOG_TMPWD ("XC Grid Derivative Compute");
    quick_cuest_compute_mem.gridder_wksp = allocateWorkspace (tmpWD);
}

void
cuest_get_grid_der (double *C, double *v, double *basisgrad, double *gridgrad)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;
    uint64_t                    ndim   = quick_cuest_compute_mem.der_ndim;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    void *d_C = quick_cuest_PC_buf.d_C[0];
    cudaMemcpyChecked (d_C, C, quick_cuest_PC_buf.C_siz, cudaMemcpyHostToDevice);

    void *d_v = quick_cuest_compute_mem.d_der_v;
    cudaMemcpyChecked (d_v, v, ndim * npoint * sizeof (double), cudaMemcpyHostToDevice);

    void *d_basisgrad = quick_cuest_compute_mem.d_basisgrad;
    void *d_gridgrad  = quick_cuest_compute_mem.d_gridgrad;

    checkCuestErrors (cuestXCDerivativeCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.der_approx,
        quick_cuest_compute_mem.der_par, quick_cuest_compute_mem.der_vbs,
        quick_cuest_compute_mem.der_wksp, quick_cuest_data.nocc, d_C, d_v, d_basisgrad,
        d_gridgrad));

    cudaMemcpyChecked (basisgrad, d_basisgrad, 3 * natom * sizeof (double), cudaMemcpyDeviceToHost);
    cudaMemcpyChecked (gridgrad, d_gridgrad, 3 * npoint * sizeof (double), cudaMemcpyDeviceToHost);
}

void
cuest_get_grid_point_der (double *enedens, double *gridgrad, double *outgrad)
{
    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;
    uint64_t                    natom  = quick_cuest_data.natom;

    int64_t npoint;
    cuest_get_xc_grid_npoint (&npoint);

    // reuse memory
    void *d_v         = quick_cuest_compute_mem.d_der_v;
    void *d_basisgrad = quick_cuest_compute_mem.d_basisgrad;
    void *d_gridgrad  = quick_cuest_compute_mem.d_gridgrad;
    cudaMemcpyChecked (d_v, enedens, npoint * sizeof (double), cudaMemcpyHostToDevice);
    cudaMemcpyChecked (d_gridgrad, gridgrad, 3 * npoint * sizeof (double), cudaMemcpyHostToDevice);

    checkCuestErrors (cuestXCGridDerivativeCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.gridder_par,
        quick_cuest_compute_mem.gridder_wksp, d_v, d_gridgrad, d_basisgrad));

    cudaMemcpyChecked (outgrad, d_basisgrad, 3 * natom * sizeof (double), cudaMemcpyDeviceToHost);
}

void
cuest_deinit_grid_der ()
{
    cudaFreeChecked (quick_cuest_compute_mem.d_der_v);
    cudaFreeChecked (quick_cuest_compute_mem.d_basisgrad);
    cudaFreeChecked (quick_cuest_compute_mem.d_gridgrad);
    free_dev_alloc (quick_cuest_compute_mem.der_ndim * quick_cuest_data.npoint * sizeof (double));
    free_dev_alloc (3 * quick_cuest_data.natom * sizeof (double));
    free_dev_alloc (3 * quick_cuest_data.npoint * sizeof (double));

    free (quick_cuest_compute_mem.der_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_compute_mem.der_wksp);
    freeWorkspace (quick_cuest_compute_mem.gridder_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.der_par));
    checkCuestErrors (cuestParametersDestroy (CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.gridder_par));
}
