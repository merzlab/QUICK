#ifdef LOCAL
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-headers/include/cuest.h"

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
cuest_init_oshell_xc (int32_t fnl, int64_t devsiz)
#else
cuest_init_cshell_xc (int32_t fnl, int64_t devsiz)
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

    bool   nl   = false;
    double nl_C = 0, nl_b = 0;

    // TODO: add support for other functionals supported by cuEST
    quick_cuest_struct.fnl = fnl;
    cuestXCIntPlanParametersFunctional_t functional;
    switch (fnl) {
        case CUEST_FUNCTIONAL_B3LYP:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1;
            break;
        case CUEST_FUNCTIONAL_B3LYP5:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5;
            break;
        case CUEST_FUNCTIONAL_B97:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97;
            break;
        case CUEST_FUNCTIONAL_B97MV:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV;
            nl         = true;
            nl_C       = 0.01;
            nl_b       = 6.0;
            break;
        case CUEST_FUNCTIONAL_PBE0:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0;
            break;
        case CUEST_FUNCTIONAL_LCWPBE:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBE;
            break;
        case CUEST_FUNCTIONAL_WB97X:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97X;
            break;
        case CUEST_FUNCTIONAL_WB97XV:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97XV;
            nl         = true;
            nl_C       = 0.01;
            nl_b       = 6.0;
            break;
        case CUEST_FUNCTIONAL_CAMB3LYP:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_CAMB3LYP;
            break;
        case CUEST_FUNCTIONAL_HSE06:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HSE06;
            break;
        case CUEST_FUNCTIONAL_BLYP:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP;
            break;
        case CUEST_FUNCTIONAL_PBE:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE;
            break;
        case CUEST_FUNCTIONAL_M06L:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L;
            break;
        case CUEST_FUNCTIONAL_M06:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06;
            break;
        case CUEST_FUNCTIONAL_M062X:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M062X;
            break;
        case CUEST_FUNCTIONAL_SVWN5:
            functional = CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5;
            break;
        default:
            fprintf (stderr, "%s:%d Unknown functional code %hhu\n", __func__, __LINE__, fnl);
            return;
    }

#ifdef DEBUGLOG
    if (nl) {
        DEBUGLOG ("-------- USING NONLOCAL CORRELATION --------\n");
        DEBUGLOG ("C=%f\nb=%f\n", nl_C, nl_b);
        DEBUGLOG ("--------------------------------------------\n");
    }
#endif

    quick_cuest_data.nl_C = nl_C;
    quick_cuest_data.nl_b = nl_b;

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

    quick_cuest_compute_mem.Vxc_wksp      = allocateWorkspace (tmpWD);
    quick_cuest_compute_mem.weights_saved = false;

    if (nl) {
        double scale = 1.0;
#ifdef OSHELL
        checkCuestErrors (cuestParametersCreate (CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                                                 &quick_cuest_compute_mem.Vxc_nl_par));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_SCALE, &scale, sizeof (double)));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_C, &nl_C, sizeof (double)));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS_VV10_B, &nl_b, sizeof (double)));
#else
        checkCuestErrors (cuestParametersCreate (CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                                 &quick_cuest_compute_mem.Vxc_nl_par));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_SCALE, &scale, sizeof (double)));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_C, &nl_C, sizeof (double)));
        checkCuestErrors (cuestParametersConfigure (
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS, quick_cuest_compute_mem.Vxc_nl_par,
            CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_B, &nl_b, sizeof (double)));
#endif

        quick_cuest_compute_mem.Vxc_nl_vbs = vbs;

#ifdef OSHELL
        checkCuestErrors (cuestNonlocalXCPotentialUKSComputeWorkspaceQuery (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_nl_par, vbs, tmpWD,
            quick_cuest_data.nocc, quick_cuest_data.noccb, NULL, NULL, NULL,
            quick_cuest_compute_mem.d_Vxc));
        MEMLOG_TMPWD ("Nonlocal V_xc alpha beta compute");
#else
        checkCuestErrors (cuestNonlocalXCPotentialRKSComputeWorkspaceQuery (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_nl_par, vbs, tmpWD,
            quick_cuest_data.nocc, NULL, NULL, quick_cuest_compute_mem.d_Vxc));
        MEMLOG_TMPWD ("Nonlocal V_xc Compute");
#endif

        quick_cuest_compute_mem.Vxc_nl_wksp = allocateWorkspace (tmpWD);

        quick_cuest_compute_mem.Vxc_nl = malloc (nbasis * nbasis * sizeof (double));
        add_host_alloc (nbasis * nbasis * sizeof (double));
    }
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
#ifdef OSHELL
    cudaFreeChecked (quick_cuest_compute_mem.d_Vxcb);
    free_dev_alloc (nbasis * nbasis * sizeof (double));
#endif

    // free Vxc/semilocal Vxc

    free (quick_cuest_compute_mem.Vxc_vbs); // also frees nonlocal Vxc vbs
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

    // free nonlocal Vxc
    if (quick_cuest_data.nl_C != 0 && quick_cuest_data.nl_b != 0) {
        free (quick_cuest_compute_mem.Vxc_nl);
        free_host_alloc (nbasis * nbasis * sizeof (double));
        freeWorkspace (quick_cuest_compute_mem.Vxc_nl_wksp);
#ifdef OSHELL
        checkCuestErrors (cuestParametersDestroy (CUEST_NONLOCALXCPOTENTIALUKSCOMPUTE_PARAMETERS,
                                                  quick_cuest_compute_mem.Vxc_nl_par));
#else
        checkCuestErrors (cuestParametersDestroy (CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
                                                  quick_cuest_compute_mem.Vxc_nl_par));
#endif
    }
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

    cudaMemcpyChecked (Vxc, quick_cuest_compute_mem.d_Vxc, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
#ifdef OSHELL
    cudaMemcpyChecked (Vxcb, quick_cuest_compute_mem.d_Vxcb, nbasis * nbasis * sizeof (double),
                       cudaMemcpyDeviceToHost);
#endif

    if (quick_cuest_data.nl_C != 0 || quick_cuest_data.nl_b != 0) {
        double  Exc_nl;
        double *Vxc_nl = quick_cuest_compute_mem.Vxc_nl;

#ifdef OSHELL
        checkCuestErrors (cuestNonlocalXCPotentialUKSCompute (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_nl_par,
            quick_cuest_compute_mem.Vxc_nl_vbs, quick_cuest_compute_mem.Vxc_nl_wksp, nocc, noccb,
            d_C, d_Cb, &Exc_nl, quick_cuest_compute_mem.d_Vxc));
#else
        checkCuestErrors (cuestNonlocalXCPotentialRKSCompute (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_compute_mem.Vxc_nl_par,
            quick_cuest_compute_mem.Vxc_nl_vbs, quick_cuest_compute_mem.Vxc_nl_wksp, nocc, d_C,
            &Exc_nl, quick_cuest_compute_mem.d_Vxc));

#endif

        cudaMemcpyChecked (Vxc_nl, quick_cuest_compute_mem.d_Vxc, nbasis * nbasis * sizeof (double),
                           cudaMemcpyDeviceToHost);

        // accumulate in Vxc/Vxcb/Exc/Excb
        *Exc += Exc_nl;

        for (size_t i = 0, end = nbasis * nbasis; i < end; ++i) {
            Vxc[i] += Vxc_nl[i];
#ifdef OSHELL
            Vxcb[i] += Vxc_nl[i];
#endif
        }
    }
}

void
#ifdef OSHELL
cuest_init_oshell_xc_grad (int64_t devsiz)
#else
cuest_init_cshell_xc_grad (int64_t devsiz)
#endif
{
    cuestWorkspaceDescriptor_t *tmpWD = quick_cuest_struct.tmpWD;
    uint64_t                    natom = quick_cuest_data.natom;

#ifdef OSHELL
    checkCuestErrors (cuestParametersCreate (CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.xc_par));
#else
    checkCuestErrors (cuestParametersCreate (CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                             &quick_cuest_grad_mem.xc_par));
#endif

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    add_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes   = 0;
    vbs->deviceBufferSizeInBytes = devsiz;
    quick_cuest_grad_mem.xc_vbs  = vbs;

#ifdef OSHELL
    checkCuestErrors (cuestXCDerivativeUKSComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par, vbs,
        tmpWD, quick_cuest_data.nocc, quick_cuest_data.noccb, NULL, NULL,
        quick_cuest_grad_mem.d_grad));
#else
    checkCuestErrors (cuestXCDerivativeRKSComputeWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par, vbs,
        tmpWD, quick_cuest_data.nocc, NULL, quick_cuest_grad_mem.d_grad));
#endif

    MEMLOG_TMPWD ("xc Gradient Compute");
    quick_cuest_grad_mem.xc_wksp = allocateWorkspace (tmpWD);

    if (quick_cuest_data.nl_C != 0 || quick_cuest_data.nl_b != 0) {
#ifdef OSHELL
        checkCuestErrors (cuestParametersCreate (CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                                                 &quick_cuest_grad_mem.xc_nl_par));
#else
        checkCuestErrors (cuestParametersCreate (CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                                 &quick_cuest_grad_mem.xc_nl_par));
#endif

        quick_cuest_grad_mem.xc_nl_vbs = vbs;

#ifdef OSHELL
        checkCuestErrors (cuestNonlocalXCDerivativeUKSComputeWorkspaceQuery (
            quick_cuest_struct.handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_nl_par,
            vbs, tmpWD, quick_cuest_data.nocc, quick_cuest_data.noccb, NULL, NULL,
            quick_cuest_grad_mem.d_grad));
#else
        checkCuestErrors (cuestNonlocalXCDerivativeRKSComputeWorkspaceQuery (
            quick_cuest_struct.handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_nl_par,
            vbs, tmpWD, quick_cuest_data.nocc, NULL, quick_cuest_grad_mem.d_grad));
#endif

        MEMLOG_TMPWD ("Nonlocal Correlation Gradient Compute");
        quick_cuest_grad_mem.xc_nl_wksp = allocateWorkspace (tmpWD);

        quick_cuest_grad_mem.grad = malloc (3 * natom * sizeof (double));
        add_host_alloc (3 * natom * sizeof (double));
    }
}

void
#ifdef OSHELL
cuest_deinit_oshell_xc_grad ()
#else
cuest_deinit_cshell_xc_grad ()
#endif
{
    const uint64_t natom = quick_cuest_data.natom;

    free (quick_cuest_grad_mem.xc_vbs);
    free_host_alloc (sizeof (cuestWorkspaceDescriptor_t));
    freeWorkspace (quick_cuest_grad_mem.xc_wksp);
#ifdef OSHELL
    checkCuestErrors (cuestParametersDestroy (CUEST_XCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.xc_par));
#else
    checkCuestErrors (cuestParametersDestroy (CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                              quick_cuest_grad_mem.xc_par));
#endif
    if (quick_cuest_data.nl_C != 0 || quick_cuest_data.nl_b != 0) {
        free (quick_cuest_grad_mem.grad);
        free_host_alloc (3 * natom * sizeof (double));
        freeWorkspace (quick_cuest_grad_mem.xc_nl_wksp);
#ifdef OSHELL
        checkCuestErrors (cuestParametersDestroy (CUEST_NONLOCALXCDERIVATIVEUKSCOMPUTE_PARAMETERS,
                                                  quick_cuest_grad_mem.xc_nl_par));
#else
        checkCuestErrors (cuestParametersDestroy (CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
                                                  quick_cuest_grad_mem.xc_nl_par));
#endif
    }
}

void
#ifdef OSHELL
cuest_get_oshell_xc_grad (double *grad, double *C, double *Cb)
#else
cuest_get_cshell_xc_grad (double *grad, double *C)
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

    void        *d_C     = quick_cuest_PC_buf.d_C[0];
    const size_t d_C_siz = quick_cuest_PC_buf.C_siz;
    cudaMemcpyChecked (d_C, C, d_C_siz, cudaMemcpyHostToDevice);

#ifdef OSHELL
    void        *d_Cb     = quick_cuest_PC_buf.d_Cb[0];
    const size_t d_Cb_siz = quick_cuest_PC_buf.Cb_siz;
    cudaMemcpyChecked (d_Cb, Cb, d_Cb_siz, cudaMemcpyHostToDevice);
#endif

#ifdef OSHELL
    checkCuestErrors (cuestXCDerivativeUKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par,
        quick_cuest_grad_mem.xc_vbs, quick_cuest_grad_mem.xc_wksp, nocc, noccb, d_C, d_Cb,
        quick_cuest_grad_mem.d_grad));
#else
    checkCuestErrors (cuestXCDerivativeRKSCompute (
        handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_par,
        quick_cuest_grad_mem.xc_vbs, quick_cuest_grad_mem.xc_wksp, nocc, d_C,
        quick_cuest_grad_mem.d_grad));
#endif

    const size_t grad_siz = 3 * natom * sizeof (double);
    cudaMemcpyChecked (grad, quick_cuest_grad_mem.d_grad, grad_siz, cudaMemcpyDeviceToHost);

    if (quick_cuest_data.nl_C != 0 || quick_cuest_data.nl_b != 0) {
#ifdef OSHELL
        checkCuestErrors (cuestNonlocalXCDerivativeUKSCompute (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_nl_par,
            quick_cuest_grad_mem.xc_nl_vbs, quick_cuest_grad_mem.xc_nl_wksp, nocc, noccb, d_C, d_Cb,
            quick_cuest_grad_mem.d_grad));
#else
        checkCuestErrors (cuestNonlocalXCDerivativeRKSCompute (
            handle, quick_cuest_struct.XCIntPlan, quick_cuest_grad_mem.xc_nl_par,
            quick_cuest_grad_mem.xc_nl_vbs, quick_cuest_grad_mem.xc_nl_wksp, nocc, d_C,
            quick_cuest_grad_mem.d_grad));
#endif

        double *grad_nl = quick_cuest_grad_mem.grad;
        cudaMemcpyChecked (grad_nl, quick_cuest_grad_mem.d_grad, grad_siz, cudaMemcpyDeviceToHost);

        for (size_t i = 0, end = 3 * natom; i < end; ++i)
            grad[i] += grad_nl[i];
    }
}
