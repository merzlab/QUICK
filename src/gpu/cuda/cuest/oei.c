#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuda_runtime.h>
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"

#include "quick_cuest.h"

// TODO: pass pair list threshold
/**
 * Initializes the one-electron integrals plan `OEIntPlan` in `quick_cuest_struct`
 *
 * `cuest_init` must have been called before calling `cuest_init_oei_plan`
 */
void
cuest_init_oei_plan ()
{
    // ================ //
    // set up pair list //
    // ================ //

    // xyz is flat already
    //
    // double *xyz_flat = malloc (quick_cuest_data.natom * 3 * sizeof (double));
    //
    // for (size_t i = 0; i < quick_cuest_data.natom; ++i) {
    //     size_t i3        = 3 * i;
    //     xyz_flat[i3]     = xyz[i][0];
    //     xyz_flat[i3 + 1] = xyz[i][1];
    //     xyz_flat[i3 + 2] = xyz[i][2];
    // }

    cuestAOPairListParameters_t pair_list_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOPAIRLIST_PARAMETERS, &pair_list_params));
    checkCuestErrors (cuestAOPairListCreateWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.basis, quick_cuest_data.natom,
        quick_cuest_data.xyz, 1e-14, pair_list_params, quick_cuest_struct.persistWD,
        quick_cuest_struct.tmpWD, &quick_cuest_struct.AOPairList));

    quick_cuest_struct.persistAOPairListWorkspace
        = allocateWorkspace (quick_cuest_struct.persistWD);
    cuestWorkspace_t *tmpAOPairListWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);

    checkCuestErrors (
        cuestAOPairListCreate (quick_cuest_struct.handle, quick_cuest_struct.basis,
                               quick_cuest_data.natom, quick_cuest_data.xyz, 1e-14,
                               pair_list_params, quick_cuest_struct.persistAOPairListWorkspace,
                               tmpAOPairListWorkspace, &quick_cuest_struct.AOPairList));
    checkCuestErrors (cuestParametersDestroy (CUEST_AOPAIRLIST_PARAMETERS, pair_list_params));
    freeWorkspace (tmpAOPairListWorkspace);
    // free (xyz_flat);

    // ========================== //
    // one-electron integral plan //
    // ========================== //

    cuestOEIntPlanParameters_t oeint_plan_params;
    checkCuestErrors (cuestParametersCreate (CUEST_OEINTPLAN_PARAMETERS, &oeint_plan_params));
    checkCuestErrors (cuestOEIntPlanCreateWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_struct.basis, quick_cuest_struct.AOPairList,
        oeint_plan_params, quick_cuest_struct.persistWD, quick_cuest_struct.tmpWD,
        &quick_cuest_struct.OEIntPlan));

    quick_cuest_struct.persistOEIntPlanWorkspace = allocateWorkspace (quick_cuest_struct.persistWD);
    cuestWorkspace_t *tmpOEIntPlanWorkspace      = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestOEIntPlanCreate (quick_cuest_struct.handle, quick_cuest_struct.basis,
                                            quick_cuest_struct.AOPairList, oeint_plan_params,
                                            quick_cuest_struct.persistOEIntPlanWorkspace,
                                            tmpOEIntPlanWorkspace, &quick_cuest_struct.OEIntPlan));

    checkCuestErrors (cuestParametersDestroy (CUEST_OEINTPLAN_PARAMETERS, oeint_plan_params));
    freeWorkspace (tmpOEIntPlanWorkspace);
}

void
cuest_get_oei_S (double *o)
{
    // ============================== //
    // compute one-electron integrals //
    // ============================== //

    uint64_t nao = 0;
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS,
                                  quick_cuest_struct.basis, CUEST_AOBASIS_NUM_AO, &nao,
                                  sizeof (uint64_t)));

    double *d_S;
    size_t  d_S_siz = nao * nao * sizeof (double);
    if (cudaMalloc ((void **)&d_S, d_S_siz)) {
        fprintf (stderr, "Failed to allocate device buffer\n");
        exit (EXIT_FAILURE);
    }

    cuestOverlapComputeParameters_t overlap_compute_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_OVERLAPCOMPUTE_PARAMETERS, &overlap_compute_params));
    checkCuestErrors (
        cuestOverlapComputeWorkspaceQuery (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           overlap_compute_params, quick_cuest_struct.tmpWD, d_S));

    cuestWorkspace_t *tmpSWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
    checkCuestErrors (cuestOverlapCompute (quick_cuest_struct.handle, quick_cuest_struct.OEIntPlan,
                                           overlap_compute_params, tmpSWorkspace, d_S));

    freeWorkspace (tmpSWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_OVERLAPCOMPUTE_PARAMETERS, overlap_compute_params));

    // ==================== //
    // print overlap matrix //
    // ==================== //

    double *buf = malloc (nao * nao * sizeof (double));
    if (!buf) {
        fprintf (stderr, "malloc buf failed\n");
        EXIT_FAILURE;
    }

    cudaMemcpy (buf, d_S, d_S_siz, cudaMemcpyDeviceToHost);

    puts ("-------- S --------");
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j)
            printf ("%16.10f", buf[i * nao + j]);
        putchar ('\n');
    }
    puts ("------ END S ------");

    free (buf);

    if (cudaFree (d_S) != cudaSuccess) {
        fprintf (stderr, "cudaFree failed\n");
        exit (EXIT_FAILURE);
    }
}
