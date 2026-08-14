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

static cuestAtomGrid_t *atom_grids;
#endif

// atom_grids is defined in xc.c

void
#ifdef OSHELL
cuest_init_oshell_xc (int8_t fnl, int64_t devsiz)
#else
cuest_init_cshell_xc (int8_t fnl, int64_t devsiz)
#endif
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
    quick_cuest_struct.fnl = fnl;
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
    cudaMallocChecked (&quick_cuest_compute_mem.d_Vxc, Vxc_siz);

#ifdef OSHELL
    cudaMallocChecked (&quick_cuest_compute_mem.d_Vxcb, Vxc_siz);
    checkCuestErrors (cuestParametersCreate (CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.Vxc_par));
#else
    checkCuestErrors (cuestParametersCreate (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_compute_mem.Vxc_par));
#endif

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = devsiz;
    quick_cuest_compute_mem.Vxc_vbs = vbs;

#ifdef OSHELL
    checkCuestErrors (cuestXCPotentialUKSComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par, vbs, tmpWD,
        quick_cuest_data.nocc, quick_cuest_data.noccb, NULL, NULL, NULL,
        quick_cuest_compute_mem.d_Vxc, quick_cuest_compute_mem.d_Vxcb));
    MEMLOG_TMPWD ("V_xc alpha beta compute");
#else
    checkCuestErrors (cuestXCPotentialRKSComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par, vbs, tmpWD,
        quick_cuest_data.nocc, NULL, NULL, quick_cuest_compute_mem.d_Vxc));
    MEMLOG_TMPWD ("V_xc Compute");
#endif

    quick_cuest_compute_mem.Vxc_wksp = allocateWorkspace (tmpWD);

    quick_cuest_compute_mem.weights_saved = false;
}

void
#ifdef OSHELL
cuest_deinit_oshell_xc ()
#else
cuest_deinit_cshell_xc ()
#endif
{
    const uint64_t nbasis = quick_cuest_data.nbasis;

    cudaFreeChecked (quick_cuest_compute_mem.d_Vxc);
    free_dev_alloc (nbasis * nbasis * sizeof (double));

    free (quick_cuest_compute_mem.Vxc_vbs);
    if (quick_cuest_compute_mem.weights_saved)
        free (quick_cuest_compute_mem.weights_save);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_compute_mem.Vxc_wksp);
#ifdef OSHELL
    checkCuestErrors (cuestParametersDestroy (CUEST_XCPOTENTIALUKSCOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.Vxc_par));
#else
    checkCuestErrors (cuestParametersDestroy (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                              quick_cuest_compute_mem.Vxc_par));
#endif

    checkCuestErrors (cuestMolecularGridDestroy (quick_cuest_struct.molgrid));
    freeWorkspace (quick_cuest_struct.persistXCGridWorkspace);
    checkCuestErrors (cuestXCIntPlanDestroy (quick_cuest_struct.XCIntPlan));
    freeWorkspace (quick_cuest_struct.persistXCIntPlanWorkspace);
}

void
#ifdef OSHELL
cuest_get_oshell_xc (double *Vxc, double *Vxcb, double *Exc, double *C, double *Cb)
#else
cuest_get_cshell_xc (double *Vxc, double *Exc, double *C)
#endif
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;
#ifdef OSHELL
    uint64_t noccb = quick_cuest_data.noccb;
#endif

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    void  *d_C     = quick_cuest_PC_buf.d_C[0];
    size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);
#ifdef OSHELL
    void  *d_Cb     = quick_cuest_PC_buf.d_Cb[0];
    size_t d_Cb_siz = quick_cuest_PC_buf.Cb_siz;
    cudaMemcpyChecked (d_Cb, Cb, d_Cb_siz, cudaMemcpyHostToDevice);
#endif

#ifdef OSHELL
    checkCuestErrors (cuestXCPotentialUKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par,
        quick_cuest_compute_mem.Vxc_vbs, quick_cuest_compute_mem.Vxc_wksp, nocc, noccb, d_C, d_Cb,
        Exc, quick_cuest_compute_mem.d_Vxc, quick_cuest_compute_mem.d_Vxcb));
#else
    checkCuestErrors (cuestXCPotentialRKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_par,
        quick_cuest_compute_mem.Vxc_vbs, quick_cuest_compute_mem.Vxc_wksp, nocc, d_C, Exc,
        quick_cuest_compute_mem.d_Vxc));

#endif

    // copy to host
    cudaMemcpyChecked (Vxc, quick_cuest_compute_mem.d_Vxc, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
#ifdef OSHELL
    cudaMemcpyChecked (Vxcb, quick_cuest_compute_mem.d_Vxcb, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
#endif
}
