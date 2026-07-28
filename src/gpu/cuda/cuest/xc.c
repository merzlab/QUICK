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

void
cuest_init_xc (int8_t fnl, int64_t devsiz)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    double  *xyz    = quick_cuest_data.xyz;
    memcpy (xyz, quick_cuest_data.xyz, 3 * natom * sizeof (double));

    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;

    // cuestAtomGrid_t *agrids = malloc (natom * sizeof (cuestAtomGrid_t));
    // form_agrid (n_rad_pts, n_ang_pts, agrids);

    // ===================== //
    // set up molecular grid //
    // ===================== //

    cuestMolecularGridParameters_t molgrid_params;
    checkCuestErrors (cuestParametersCreate (CUEST_MOLECULARGRID_PARAMETERS, &molgrid_params));
    checkCuestErrors (cuestMolecularGridCreateWorkspaceQuery (handle, natom, atom_grids, xyz,
                                                              molgrid_params, persistWD, tmpWD,
                                                              &quick_cuest_struct.molgrid));

    MEMLOG ("XC Molecular Grid");
    quick_cuest_struct.persistXCGridWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpGridWorkspace        = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestMolecularGridCreate (handle, natom, atom_grids, xyz, molgrid_params,
                                                quick_cuest_struct.persistXCGridWorkspace,
                                                tmpGridWorkspace, &quick_cuest_struct.molgrid));

    checkCuestErrors (cuestParametersDestroy (CUEST_MOLECULARGRID_PARAMETERS, molgrid_params));
    freeWorkspace (tmpGridWorkspace);

    // ===================== //
    // init XC integral plan //
    // ===================== //

    cuestXCIntPlanParameters_t xcIntPlan_param;
    checkCuestErrors (cuestParametersCreate (CUEST_XCINTPLAN_PARAMETERS, &xcIntPlan_param));

    // TODO: add support for other functionals supported by cuEST
    cuestXCIntPlanParametersFunctional_t functional;
    switch (fnl) {
        case CUEST_FUNCTIONAL_B3LYP:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1;
            break;
        case CUEST_FUNCTIONAL_B97:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97;
            break;
        case CUEST_FUNCTIONAL_BLYP:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP;
            break;
        case CUEST_FUNCTIONAL_PBE:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE;
            break;
        case CUEST_FUNCTIONAL_PBE0:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0;
            break;
        default:
            fprintf (stderr, "%s:%d Unknown functional code %hhu\n", __func__, __LINE__, fnl);
            return;
    }

    checkCuestErrors (cuestXCIntPlanCreateWorkspaceQuery (
        handle, quick_cuest_struct.basis, quick_cuest_struct.molgrid, functional, xcIntPlan_param,
        persistWD, tmpWD, &quick_cuest_struct.XCIntPlan));

    MEMLOG ("XC Integral Plan");
    quick_cuest_struct.persistXCIntPlanWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpXCIntPlanWorkspace      = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestXCIntPlanCreate (handle, quick_cuest_struct.basis,
                                            quick_cuest_struct.molgrid, functional, xcIntPlan_param,
                                            quick_cuest_struct.persistXCIntPlanWorkspace,
                                            tmpXCIntPlanWorkspace, &quick_cuest_struct.XCIntPlan));
    checkCuestErrors (cuestParametersDestroy (CUEST_XCINTPLAN_PARAMETERS, xcIntPlan_param));

    freeWorkspace (tmpXCIntPlanWorkspace);

    // ====================== //
    // set up compute buffers //
    // ====================== //

    const size_t Vxc_siz = nbasis * nbasis * sizeof (double);
    cudaMallocChecked ((void **)&quick_cuest_compute_mem.d_Vxc, Vxc_siz);

    checkCuestErrors (cuestParametersCreate (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.Vxc_par));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = devsiz;
    quick_cuest_compute_mem.Vxc_vbs = vbs;

    checkCuestErrors (cuestXCPotentialRKSComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par, vbs, tmpWD,
        quick_cuest_data.nocc, NULL, NULL, quick_cuest_compute_mem.d_Vxc));

    MEMLOG_TMPWD ("V_xc Compute");
    quick_cuest_compute_mem.Vxc_wksp = allocateWorkspace (tmpWD);
}

void
cuest_deinit_xc ()
{
    const uint64_t nbasis = quick_cuest_data.nbasis;

    cudaFreeChecked (quick_cuest_compute_mem.d_Vxc);
    free_dev_alloc (nbasis * nbasis * sizeof (double));

    free (quick_cuest_compute_mem.Vxc_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_compute_mem.Vxc_wksp);
    checkCuestErrors (cuestParametersDestroy (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.Vxc_par));

    checkCuestErrors (cuestMolecularGridDestroy (quick_cuest_struct.molgrid));
    freeWorkspace (quick_cuest_struct.persistXCGridWorkspace);
    checkCuestErrors (cuestXCIntPlanDestroy (quick_cuest_struct.XCIntPlan));
    freeWorkspace (quick_cuest_struct.persistXCIntPlanWorkspace);
}

void
cuest_get_cshell_xc (double *Vxc, double *Exc, double *C)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    void  *d_C     = quick_cuest_PC_buf.d_C[0];
    size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestXCPotentialRKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par,
        quick_cuest_compute_mem.Vxc_vbs, quick_cuest_compute_mem.Vxc_wksp, nocc, d_C, Exc,
        quick_cuest_compute_mem.d_Vxc));

    // copy to host
    cudaMemcpyChecked (Vxc, quick_cuest_compute_mem.d_Vxc, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
}

void
cuest_get_oshell_xc (double *Vxc, double *Vxcb, double *Exc, double *C, double *Cb)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;
    uint64_t noccb  = quick_cuest_data.noccb;

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    void  *d_C     = quick_cuest_PC_buf.d_C[0];
    size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);
    void  *d_Cb     = quick_cuest_PC_buf.d_Cb[0];
    size_t d_Cb_siz = quick_cuest_PC_buf.Cb_siz;
    cudaMemcpyChecked (d_Cb, Cb, d_Cb_siz, cudaMemcpyHostToDevice);

    checkCuestErrors (cuestXCPotentialUKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par,
        quick_cuest_compute_mem.Vxc_vbs, quick_cuest_compute_mem.Vxc_wksp, nocc, noccb, d_C, d_Cb,
        Exc, quick_cuest_compute_mem.d_Vxc, quick_cuest_compute_mem.d_Vxcb));

    // copy to host
    cudaMemcpyChecked (Vxc, quick_cuest_compute_mem.d_Vxc, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
    cudaMemcpyChecked (Vxcb, quick_cuest_compute_mem.d_Vxcb, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
}

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
