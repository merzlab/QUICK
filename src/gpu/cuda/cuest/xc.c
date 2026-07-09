#include <math.h>
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

// clang-format off
static const double ahlrichs_radii[] = {
    1.00,
    0.80,                                                                                0.90,
    1.80,1.40,                                                  1.30,1.10,0.90,0.90,0.90,0.90,
    1.40,1.30,                                                  1.30,1.20,1.10,1.00,1.00,1.00,
    1.50,1.40,1.30,1.20,1.20,1.20,1.20,1.20,1.20,1.10,1.10,1.10,1.10,1.00,0.90,0.90,0.90,0.90
};
// clang-format on

static void
build_ahlrichs_radial_quadrature (size_t npoint, double R, double *radialNodes,
                                  double *radialWeights)
{
    const double alpha = 0.6;
    for (size_t i = 1; i <= npoint; i++) {
        double z = i * M_PI / (npoint + 1.0);
        double x = cos (z);
        double y = sin (z);
        double u = log ((1.0 - x) / 2.0);
        double v = pow (1.0 + x, alpha) / log (2.0);
        double r = -R * v * u;
        double w = M_PI / (npoint + 1.0) * y * R * v * (-alpha * u / (1.0 + x) + 1.0 / (1.0 - x))
                   * r * r;
        radialNodes[npoint - i]   = r;
        radialWeights[npoint - i] = w;
    }
}

/**
 * Forms a direct product atom grid for each atom, written to `agrid` of length `ntaom`
 */
static void
form_agrid (uint64_t n_rad_pts, uint64_t n_ang_pts, cuestAtomGrid_t *agrids)
{
    n_rad_pts        = 70;
    n_ang_pts        = 300;
    uint64_t natom   = quick_cuest_data.natom;
    int8_t  *iattype = quick_cuest_data.iattype;

    cuestHandle_t handle = quick_cuest_struct.handle;

    cuestAtomGridParameters_t atom_grid_param;
    checkCuestErrors (cuestParametersCreate (CUEST_ATOMGRID_PARAMETERS, &atom_grid_param));

    double *chk_radialnodes_w
        = malloc ((n_rad_pts * sizeof (double) << 1) + natom * sizeof (cuestAtomGrid_t));
    double *radial_nodes = chk_radialnodes_w;
    double *w            = radial_nodes + n_rad_pts;

    // nap[i] is number of angular points of radial point i
    uint64_t *nap = malloc (n_rad_pts * sizeof (uint64_t));
    for (uint64_t i = 0; i < n_rad_pts; ++i)
        nap[i] = n_ang_pts;

    for (uint64_t i = 0; i < natom; ++i) {
        printf ("iattype[%llu]=%hhu\nr=%f\n", i, iattype[i], ahlrichs_radii[iattype[i]]);
        build_ahlrichs_radial_quadrature (n_rad_pts, ahlrichs_radii[iattype[i]], radial_nodes, w);
        printf ("radial_nodes: ");
        for (uint64_t i = 0; i < n_rad_pts; ++i)
            printf ("%f ", radial_nodes[i]);
        printf ("\nw: ");
        for (uint64_t i = 0; i < n_rad_pts; ++i)
            printf ("%f ", w[i]);
        putchar ('\n');
        checkCuestErrors (cuestAtomGridCreate (handle, n_rad_pts, radial_nodes, w, nap,
                                               atom_grid_param, &agrids[i]));
    }

    checkCuestErrors (cuestParametersDestroy (CUEST_ATOMGRID_PARAMETERS, atom_grid_param));

    free (nap);
    free (chk_radialnodes_w);
}

void
cuest_init_xc (int64_t n_rad_pts, int64_t n_ang_pts, int8_t fnl)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    double  *xyz    = quick_cuest_data.xyz;
    memcpy (xyz, quick_cuest_data.xyz, 3 * natom * sizeof (double));

    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;

    cuestAtomGrid_t *agrids = malloc (natom * sizeof (cuestAtomGrid_t));
    form_agrid (n_rad_pts, n_ang_pts, agrids);

    // ===================== //
    // set up molecular grid //
    // ===================== //

    printf ("agrids: ");
    for (int i = 0; i < natom; ++i)
        printf ("%p ", agrids[i]);
    putchar ('\n');

    cuestMolecularGridParameters_t molgrid_params;
    checkCuestErrors (cuestParametersCreate (CUEST_MOLECULARGRID_PARAMETERS, &molgrid_params));
    checkCuestErrors (cuestMolecularGridCreateWorkspaceQuery (
        handle, natom, agrids, xyz, molgrid_params, persistWD, tmpWD, &quick_cuest_struct.molgrid));

    MEMLOG ("XC Molecular Grid");
    // persistWD->deviceBufferSizeInBytes *= 10;
    // tmpWD->deviceBufferSizeInBytes *= 10;
    // MEMLOG ("XC Molecular Grid x10 memory");
    quick_cuest_struct.persistXCGridWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpGridWorkspace        = allocateWorkspace (tmpWD);

    printf ("persist device buffer: %zu\n",
            quick_cuest_struct.persistXCGridWorkspace->deviceBufferSizeInBytes);
    printf ("persist host buffer: %zu\n",
            quick_cuest_struct.persistXCGridWorkspace->hostBufferSizeInBytes);
    printf ("tmp device buffer: %zu\n", tmpGridWorkspace->deviceBufferSizeInBytes);
    printf ("tmp host buffer: %zu\n", tmpGridWorkspace->hostBufferSizeInBytes);

    for (uint64_t i = 0; i < 3 * natom; ++i)
        printf ("%f ", xyz[i]);
    putchar ('\n');

    checkCuestErrors (cuestMolecularGridCreate (handle, natom, agrids, xyz, molgrid_params,
                                                quick_cuest_struct.persistXCGridWorkspace,
                                                tmpGridWorkspace, &quick_cuest_struct.molgrid));

    checkCuestErrors (cuestParametersDestroy (CUEST_MOLECULARGRID_PARAMETERS, molgrid_params));
    freeWorkspace (tmpGridWorkspace);

    for (int i = 0; i < natom; ++i)
        checkCuestErrors (cuestAtomGridDestroy (agrids[i]));
    free (agrids);

    // ===================== //
    // init XC integral plan //
    // ===================== //

    cuestXCIntPlanParameters_t xcIntPlan_param;
    checkCuestErrors (cuestParametersCreate (CUEST_XCINTPLAN_PARAMETERS, &xcIntPlan_param));

    // TODO: add support for libxc functionals
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

    MEMLOG ("XC Molecular Grid");
    quick_cuest_struct.persistXCIntPlanWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpXCIntPlanWorkspace      = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestXCIntPlanCreate (handle, quick_cuest_struct.basis,
                                            quick_cuest_struct.molgrid, functional, xcIntPlan_param,
                                            quick_cuest_struct.persistXCIntPlanWorkspace,
                                            tmpXCIntPlanWorkspace, &quick_cuest_struct.XCIntPlan));
    checkCuestErrors (cuestParametersDestroy (CUEST_XCINTPLAN_PARAMETERS, xcIntPlan_param));

    freeWorkspace (tmpXCIntPlanWorkspace);
}

void
cuest_get_Vxc (double *Vxc, double *C)
{
    uint64_t natom  = quick_cuest_data.natom;
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nocc   = quick_cuest_data.nocc;

    cuestHandle_t               handle = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD  = quick_cuest_struct.tmpWD;

    // ============================== //
    // compute RKS XC Potential (Vxc) //
    // ============================== //

    double Exc = 0;

    double      *d_Vxc;
    const size_t Vxc_siz = nbasis * nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_Vxc, Vxc_siz) != cudaSuccess) {
        fprintf (stderr, "cudaMalloc failed in %s:%d\n", __func__, __LINE__);
        exit (EXIT_FAILURE);
    }

    double *d_C;
    size_t  d_C_siz = nocc * nbasis * sizeof (double);
    if (cudaMalloc ((void **)&d_C, d_C_siz)) {
        fprintf (stderr, "cudaMalloc failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaMemcpy (d_C, C, d_C_siz, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    cuestXCPotentialRKSComputeParameters_t rks_V_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS, &rks_V_params));

    cuestWorkspaceDescriptor_t *vbs = malloc (sizeof (cuestWorkspaceDescriptor_t));
    vbs->hostBufferSizeInBytes      = 0;
    vbs->deviceBufferSizeInBytes    = 2e9; // TODO: update; 2GB now

    checkCuestErrors (cuestXCPotentialRKSComputeWorkspaceQuery (
        handle, quick_cuest_struct.XCIntPlan, rks_V_params, vbs, tmpWD, nocc, d_C, &Exc, d_Vxc));

    MEMLOG_TMPWD ("V_xc Compute");
    cuestWorkspace_t *tmpVxcWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestXCPotentialRKSCompute (handle, quick_cuest_struct.XCIntPlan,
                                                  rks_V_params, vbs, tmpVxcWorkspace, nocc, d_C,
                                                  &Exc, d_Vxc));

    // copy to host
    if (cudaMemcpy (Vxc, d_Vxc, Vxc_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
        exit (EXIT_FAILURE);
    }

    if (cudaFree (d_Vxc) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed at %s:%d\n", __func__, __LINE__);
        exit (EXIT_FAILURE);
    }

    // free memory

    if (cudaFree (d_C) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed at %s:%d\n", __func__, __LINE__);
        exit (EXIT_FAILURE);
    }

    free (vbs);
    freeWorkspace (tmpVxcWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS, rks_V_params));
}
