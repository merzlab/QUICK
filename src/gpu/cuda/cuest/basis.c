#include <assert.h>
#include <stdatomic.h>
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
#include "util.h"

#define get(arr, i, j, ncols)      arr[(j) + (i) * (ncols)]
#define get_row_ptr(arr, i, ncols) (arr + (i) * (ncols))

void
cuest_init_basis (uint64_t natom, uint64_t nshell, uint64_t *ncenter,
                  uint64_t *first_basis_function,
                  uint64_t *last_basis_function, uint64_t *katom_,
                  uint64_t *ktype_, uint64_t *kprim_, double *gcexpo,
                  double *gccoeff, double *xyz, const uint64_t MAXPRIM)
{
    freopen ("cuest.log", "w", stdout);

    puts ("-------- DUMP --------");

    printf ("natom=%llu\n", natom);
    printf ("natom=%llu\n", nshell);

    puts ("ncenter:");
    for (int i = 0; i < 7; ++i)
        printf ("%llu ", ncenter[i]);
    putchar ('\n');

    puts ("first_basis_function:");
    for (int i = 0; i < 3; ++i)
        printf ("%llu ", first_basis_function[i]);
    putchar ('\n');

    puts ("last_basis_function:");
    for (int i = 0; i < 3; ++i)
        printf ("%llu ", last_basis_function[i]);
    putchar ('\n');

    puts ("katom_:");
    for (int i = 0; i < 4; ++i)
        printf ("%llu ", katom_[i]);
    putchar ('\n');

    puts ("ktype_:");
    for (int i = 0; i < 4; ++i)
        printf ("%llu ", ktype_[i]);
    putchar ('\n');

    puts ("kprim_:");
    for (int i = 0; i < 4; ++i)
        printf ("%llu ", kprim_[i]);
    putchar ('\n');

    puts ("gcexpo:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < MAXPRIM; ++j)
            printf ("%f ", get (gcexpo, i, j, MAXPRIM));
        putchar ('\n');
    }

    puts ("gccoeff:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < MAXPRIM; ++j)
            printf ("%f ", get (gccoeff, i, j, MAXPRIM));
        putchar ('\n');
    }

    puts ("xyz:");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            printf ("%f ", get (xyz, i, j, 3));
        putchar ('\n');
    }

    puts ("gcexpo flat:");
    for (int i = 0; i < 7 * MAXPRIM; ++i)
        printf ("%f ", gcexpo[i]);
    putchar ('\n');

    puts ("gccoeff flat:");
    for (int i = 0; i < 7 * MAXPRIM; ++i)
        printf ("%f ", gccoeff[i]);
    putchar ('\n');

    puts ("xyz flat:");
    for (int i = 0; i < 3 * 3; ++i)
        printf ("%f ", xyz[i]);
    putchar ('\n');

    puts ("------ END DUMP ------");

    // ============= //
    // set up handle //
    // ============= //

    cuestHandle_t           handle;
    cuestHandleParameters_t handle_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_HANDLE_PARAMETERS, &handle_params));
    checkCuestErrors (cuestCreate (handle_params, &handle));
    checkCuestErrors (
        cuestParametersDestroy (CUEST_HANDLE_PARAMETERS, handle_params));

    // ================ //
    // set up AO shells //
    // ================ //

    // preprocess and correct for SP shells

    size_t nsp = 0;
    for (int i = 0; i < nshell; ++i)
        nsp += (ktype_[i] == 4);

    nshell += nsp;
    uint64_t *chk_katom_ktype_kprim = malloc (3 * nshell * sizeof (uint64_t));
    // expanded arrays (SP -> S and P)
    uint64_t *katom = chk_katom_ktype_kprim;
    uint64_t *ktype = chk_katom_ktype_kprim + nshell;
    uint64_t *kprim = chk_katom_ktype_kprim + (nshell << 1);

    // needed for basis

    // first_basis_function but instead first_basis_shell
    size_t *ifshell = malloc (nshell * sizeof (size_t));
    // nshells_per_atom[a] is number of shells atom `a` has.
    // This is the same as the number of times it appears in `katom`.
    uint64_t *nshells_per_atom = calloc (natom, sizeof (uint64_t));

    for (size_t i = 0, j = 0, jend = nshell - nsp; i < nshell && j < jend;
         ++i, ++j) {
        katom[i] = katom_[j];
        kprim[i] = kprim_[j];

        if (ktype_[i] == 4) {
            ktype[i] = 1;
            ++i;
            ktype[i] = 3;
            katom[i] = katom_[j];
            kprim[i] = kprim_[j];
        } else {
            ktype[i] = ktype_[j];
        }
    }

    for (size_t i = 0; i < nshell; ++i) {
        uint64_t a = katom[i] - 1;
        ifshell[i] = first_basis_function[a]
                     + shell_offset_cart[nshells_per_atom[a]++] - 1;
        // printf ("ifshell[%zu]=%zu\n", j, ifshell[i]);
    }

    // start making shells

    cuestAOShell_t *shells = malloc (nshell * sizeof (cuestAOShell_t));
    if (!shells) {
        fprintf (stderr, "Failed to allocate AO shell array\n");
        checkCuestErrors (cuestDestroy (handle));
        exit (EXIT_FAILURE);
    }

    cuestAOShellParameters_t aoshell_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_AOSHELL_PARAMETERS, &aoshell_params));

    // double *coeff = malloc (3 * sizeof (double));

    for (size_t i = 0; i < nshell; ++i) {
        // // manual normalization, same as pulling from QUEST
        // size_t   ifsh = ifshell[i];
        // uint64_t L    = get_L (ktype[i]);
        // normalize_coeff (dcoeff[ifsh], aexp[ifsh], 3, L, 1.0, coeff);
        // checkCuestErrors (
        //     cuestAOShellCreate (handle, 0, L, kprim[i],
        //     aexp[ifsh],
        //                         coeff, aoshell_params, &shells[i]));

        checkCuestErrors (
            cuestAOShellCreate (handle, 0, get_L (ktype[i]), kprim[i],
                                get_row_ptr (gcexpo, ifshell[i], MAXPRIM),
                                get_row_ptr (gccoeff, ifshell[i], MAXPRIM),
                                aoshell_params, &shells[i]));
    }

    // free (coeff);
    free (ifshell);
    free (chk_katom_ktype_kprim);

    checkCuestErrors (
        cuestParametersDestroy (CUEST_AOSHELL_PARAMETERS, aoshell_params));

    // ============ //
    // set up basis //
    // ============ //

    cuestAOBasis_t           basis;
    cuestAOBasisParameters_t basis_params;
    checkCuestErrors (
        cuestParametersCreate (CUEST_AOBASIS_PARAMETERS, &basis_params));

    cuestWorkspaceDescriptor_t *persistWD
        = malloc (sizeof (cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t *tmpWD
        = malloc (sizeof (cuestWorkspaceDescriptor_t));

    checkCuestErrors (cuestAOBasisCreateWorkspaceQuery (
        handle, natom, nshells_per_atom, shells, basis_params, persistWD,
        tmpWD, &basis));

    cuestWorkspace_t *persistBasisWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpBasisWorkspace     = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestAOBasisCreate (
        handle, natom, nshells_per_atom, shells, basis_params,
        persistBasisWorkspace, tmpBasisWorkspace, &basis));

    freeWorkspace (tmpBasisWorkspace);
    checkCuestErrors (
        cuestParametersDestroy (CUEST_AOBASIS_PARAMETERS, basis_params));

    for (size_t i = 0; i < nshell; ++i)
        checkCuestErrors (cuestAOShellDestroy (shells[i]));

    free (shells);
    free (nshells_per_atom);

    // ================= //
    // query information //
    // ================= //

    uint64_t query_natom      = 0;
    uint64_t query_nshell     = 0;
    uint64_t query_nao        = 0;
    uint64_t query_ncart      = 0;
    uint64_t query_nprimitive = 0;
    uint64_t query_max_L      = 0;
    int32_t  query_is_pure    = 0;

    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_ATOM, &query_natom,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_SHELL, &query_nshell,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_AO, &query_nao,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_CART, &query_ncart,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_PRIMITIVE,
                                  &query_nprimitive, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_MAX_L, &query_max_L,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_IS_PURE, &query_is_pure,
                                  sizeof (int32_t)));

    printf ("AO Basis from handle:\n");
    printf ("%-10s = %6llu\n", "natom", query_natom);
    printf ("%-10s = %6llu\n", "nshell", query_nshell);
    printf ("%-10s = %6llu\n", "nao", query_nao);
    printf ("%-10s = %6llu\n", "ncart", query_ncart);
    printf ("%-10s = %6llu\n", "nprimitive", query_nprimitive);
    printf ("%-10s = %6llu\n", "max_L", query_max_L);
    printf ("%-10s = %6s\n", "is_pure", query_is_pure ? "true" : "false");

    // ================ //
    // set up pair list //
    // ================ //

    // xyz is flat already
    //
    // double *xyz_flat = malloc (natom * 3 * sizeof (double));
    //
    // for (size_t i = 0; i < natom; ++i) {
    //     size_t i3        = 3 * i;
    //     xyz_flat[i3]     = xyz[i][0];
    //     xyz_flat[i3 + 1] = xyz[i][1];
    //     xyz_flat[i3 + 2] = xyz[i][2];
    // }

    cuestAOPairList_t           pair_list;
    cuestAOPairListParameters_t pair_list_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOPAIRLIST_PARAMETERS,
                                             &pair_list_params));
    checkCuestErrors (cuestAOPairListCreateWorkspaceQuery (
        handle, basis, natom, xyz, 1e-14, pair_list_params, persistWD, tmpWD,
        &pair_list));

    cuestWorkspace_t *persistAOPairListWorkspace
        = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpAOPairListWorkspace = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestAOPairListCreate (
        handle, basis, natom, xyz, 1e-14, pair_list_params,
        persistAOPairListWorkspace, tmpAOPairListWorkspace, &pair_list));
    checkCuestErrors (cuestParametersDestroy (CUEST_AOPAIRLIST_PARAMETERS,
                                              pair_list_params));
    freeWorkspace (tmpAOPairListWorkspace);
    // free (xyz_flat);

    // ========================= //
    // one-electron integral plan //
    // ========================= //

    cuestOEIntPlan_t           oeint_plan;
    cuestOEIntPlanParameters_t oeint_plan_params;
    checkCuestErrors (cuestParametersCreate (CUEST_OEINTPLAN_PARAMETERS,
                                             &oeint_plan_params));
    checkCuestErrors (cuestOEIntPlanCreateWorkspaceQuery (
        handle, basis, pair_list, oeint_plan_params, persistWD, tmpWD,
        &oeint_plan));

    cuestWorkspace_t *persistOEIntPlanWorkspace
        = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpOEIntPlanWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestOEIntPlanCreate (
        handle, basis, pair_list, oeint_plan_params, persistOEIntPlanWorkspace,
        tmpOEIntPlanWorkspace, &oeint_plan));

    checkCuestErrors (cuestParametersDestroy (CUEST_OEINTPLAN_PARAMETERS,
                                              oeint_plan_params));
    freeWorkspace (tmpOEIntPlanWorkspace);

    // ============================== //
    // compute one-electron integrals //
    // ============================== //

    uint64_t nao = 0;
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_AO, &nao,
                                  sizeof (uint64_t)));

    double *d_S;
    size_t  d_S_siz = nao * nao * sizeof (double);
    if (cudaMalloc ((void **)&d_S, d_S_siz)) {
        fprintf (stderr, "Failed to allocate device buffer\n");
        goto persist_free;
    }

    cuestOverlapComputeParameters_t overlap_compute_params;
    checkCuestErrors (cuestParametersCreate (CUEST_OVERLAPCOMPUTE_PARAMETERS,
                                             &overlap_compute_params));
    checkCuestErrors (cuestOverlapComputeWorkspaceQuery (
        handle, oeint_plan, overlap_compute_params, tmpWD, d_S));

    cuestWorkspace_t *tmpSWorkspace = allocateWorkspace (tmpWD);
    checkCuestErrors (cuestOverlapCompute (
        handle, oeint_plan, overlap_compute_params, tmpSWorkspace, d_S));

    freeWorkspace (tmpSWorkspace);
    checkCuestErrors (cuestParametersDestroy (CUEST_OVERLAPCOMPUTE_PARAMETERS,
                                              overlap_compute_params));

    // ==================== //
    // print overlap matrix //
    // ==================== //

    double *buf = malloc (nao * nao * sizeof (double));
    if (!buf) {
        fprintf (stderr, "malloc buf failed\n");
        goto persist_free;
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
        goto persist_free;
    }

    // ========================= //
    // clean up persistent stuff //
    // ========================= //

persist_free:

    free (persistWD);
    free (tmpWD);

    checkCuestErrors (cuestOEIntPlanDestroy (oeint_plan));
    freeWorkspace (persistOEIntPlanWorkspace);

    checkCuestErrors (cuestAOPairListDestroy (pair_list));
    freeWorkspace (persistAOPairListWorkspace);

    checkCuestErrors (cuestAOBasisDestroy (basis));
    freeWorkspace (persistBasisWorkspace);

    checkCuestErrors (cuestDestroy (handle));

    puts ("bye from c");
}
