#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/fake_cuda_headers/cuda_runtime.h"
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"
#include "util.h"

#include "quick_cuest.h"

/**
 * `quick_type` should not be 4=sp
 */
static uint64_t
get_L_cart (uint64_t quick_ktype)
{
    switch (quick_ktype) {
        case KTYPE_CART_S:
            return 0;
        case KTYPE_CART_P:
            return 1;
        case KTYPE_CART_D:
            return 2;
        case KTYPE_CART_F:
            return 3;
        default:
            fprintf (stderr, "get_L_cart(%llu): quick_ktype parameter invalid\n", quick_ktype);
            return 0;
    }
}

/**
 * `quick_type` should not be 4=sp
 */
#define get_L_sph(quick_ktype) (((quick_ktype) - 1) >> 1)

static void
reorder_d (double *a)
{
    SWAP (a[2], a[3], double);
}

static void
reorder_f (double *a)
{
    SWAP (a[2], a[4], double);
    SWAP (a[3], a[4], double);
    SWAP (a[4], a[5], double);
    SWAP (a[5], a[7], double);
    SWAP (a[6], a[7], double);
}

void
cuest_init_basis (int64_t *ncenter, int64_t *katom_, int64_t *ktype_, int64_t *kprim_, double *aexp,
                  double *dcoeff, bool aux)
{
    uint64_t maxcontract, nshell;
    if (aux) {
        nshell      = quick_cuest_data.nauxshell;
        maxcontract = quick_cuest_data.maxcontract_aux;
    } else {
        nshell      = quick_cuest_data.nshell;
        maxcontract = quick_cuest_data.maxcontract;
    }

#ifdef CUESTDEBUG
    puts ("-------- DUMP --------");

    printf ("natom=%llu\n", quick_cuest_data.natom);
    printf ("nshell=%llu\n", nshell);

    puts ("ncenter:");
    for (int i = 0; i < 7; ++i)
        printf ("%llu ", ncenter[i]);
    putchar ('\n');

    puts ("katom_:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", katom_[i]);
    putchar ('\n');

    puts ("ktype_:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", ktype_[i]);
    putchar ('\n');

    puts ("kprim_:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", kprim_[i]);
    putchar ('\n');

    puts ("aexp:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxcontract; ++j)
            printf ("%f ", get (aexp, i, j, maxcontract));
        putchar ('\n');
    }

    puts ("dcoeff:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxcontract; ++j)
            printf ("%f ", get (dcoeff, i, j, maxcontract));
        putchar ('\n');
    }

    puts ("xyz:");
    for (int i = 0; i < quick_cuest_data.natom; ++i) {
        for (int j = 0; j < 3; ++j)
            printf ("%f ", get (quick_cuest_data.xyz, i, j, 3));
        putchar ('\n');
    }

    puts ("aexp flat:");
    for (int i = 0; i < 7 * maxcontract; ++i)
        printf ("%f ", aexp[i]);
    putchar ('\n');

    puts ("dcoeff flat:");
    for (int i = 0; i < 7 * maxcontract; ++i)
        printf ("%f ", dcoeff[i]);
    putchar ('\n');

    puts ("xyz flat:");
    for (int i = 0; i < quick_cuest_data.natom * 3; ++i)
        printf ("%f ", quick_cuest_data.xyz[i]);
    putchar ('\n');

    puts ("------ END DUMP ------");

    fflush (stdout);
#endif

    // ================ //
    // set up AO shells //
    // ================ //

    // --------------------------------------- //
    // 1. preprocess and correct for SP shells //
    // --------------------------------------- //

    size_t nsp = 0;
    for (int i = 0; i < nshell; ++i)
        nsp += (ktype_[i] == 4);

    nshell += nsp;
    if (aux)
        quick_cuest_data.nauxshell += nsp;
    else
        quick_cuest_data.nshell += nsp;

    uint64_t *chk_katom_ktype_kprim = malloc (3 * nshell * sizeof (uint64_t));
    // expanded arrays (SP -> S and P)
    uint64_t *katom = chk_katom_ktype_kprim;
    uint64_t *ktype = chk_katom_ktype_kprim + nshell;
    uint64_t *kprim = chk_katom_ktype_kprim + (nshell << 1);

    // needed for basis

    for (size_t i = 0, j = 0, jend = nshell - nsp; i < nshell && j < jend; ++i, ++j) {
        katom[i] = katom_[j];
        kprim[i] = kprim_[j];

        if (ktype_[j] == KTYPE_CART_SP) {
            ktype[i] = KTYPE_CART_S;
            ++i;
            ktype[i] = KTYPE_CART_P;
            katom[i] = katom_[j];
            kprim[i] = kprim_[j];
        } else {
            ktype[i] = ktype_[j];
        }
    }

#ifdef CUESTDEBUG
    puts ("new katom:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", katom[i]);
    putchar ('\n');
    puts ("new ktype:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", ktype[i]);
    putchar ('\n');
    puts ("new kprim:");
    for (int i = 0; i < nshell; ++i)
        printf ("%llu ", kprim[i]);
    putchar ('\n');
    fflush (stdout);
#endif

    // first_basis_function but instead first_basis_shell
    size_t *ifshell = malloc (nshell * sizeof (size_t));
    // nshells_per_atom[a] is number of shells atom `a` has.
    // This is the same as the number of times it appears in `katom`.
    uint64_t *nshells_per_atom = calloc (quick_cuest_data.natom, sizeof (uint64_t));

    ifshell[0] = 0;

    for (size_t i = 0; i < nshell; ++i) {
        uint64_t a = katom[i] - 1;
        ++nshells_per_atom[a];
        if (i > 0)
            ifshell[i] = ifshell[i - 1] + ktype[i - 1]; // ktype stores number of cartesian orbitals
        // printf ("ifshell[%zu]=%zu\n", i, ifshell[i]);
    }

    // fflush (stdout);

    // -------------------- //
    // 2. make cuest shells //
    // -------------------- //

    cuestAOShell_t *shells = malloc (nshell * sizeof (cuestAOShell_t));
    if (!shells) {
        fprintf (stderr, "Failed to allocate AO shell array\n");
        checkCuestErrors (cuestDestroy (quick_cuest_struct.handle));
        exit (EXIT_FAILURE);
    }

    cuestAOShellParameters_t aoshell_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOSHELL_PARAMETERS, &aoshell_params));

    // double *coeff = malloc (3 * sizeof (double));

    for (size_t i = 0; i < nshell; ++i) {
        // convert QUICK ordering to CCA lexical (cuEST) ordering
        double *a = get_row_ptr (aexp, ifshell[i], maxcontract);
        double *c = get_row_ptr (dcoeff, ifshell[i], maxcontract);

        // if (!aux && ktype[i] == KTYPE_CART_D) {
        //     reorder_d (a);
        //     reorder_d (c);
        // } else if (!aux && ktype[i] == KTYPE_CART_F) {
        //     reorder_f (a);
        //     reorder_f (c);
        // }

        checkCuestErrors (cuestAOShellCreate (quick_cuest_struct.handle, aux,
                                              aux ? get_L_sph (ktype[i]) : get_L_cart (ktype[i]),
                                              kprim[i], a, c, aoshell_params, &shells[i]));
    }

    // free (coeff);
    free (ifshell);
    free (chk_katom_ktype_kprim);

    checkCuestErrors (cuestParametersDestroy (CUEST_AOSHELL_PARAMETERS, aoshell_params));

    // ============ //
    // set up basis //
    // ============ //

    cuestAOBasis_t           basis;
    cuestAOBasisParameters_t basis_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOBASIS_PARAMETERS, &basis_params));

    checkCuestErrors (cuestAOBasisCreateWorkspaceQuery (
        quick_cuest_struct.handle, quick_cuest_data.natom, nshells_per_atom, shells, basis_params,
        quick_cuest_struct.persistWD, quick_cuest_struct.tmpWD, &basis));

    cuestWorkspace_t *persistBasisWorkspace = allocateWorkspace (quick_cuest_struct.persistWD);
    cuestWorkspace_t *tmpBasisWorkspace     = allocateWorkspace (quick_cuest_struct.tmpWD);

    checkCuestErrors (cuestAOBasisCreate (quick_cuest_struct.handle, quick_cuest_data.natom,
                                          nshells_per_atom, shells, basis_params,
                                          persistBasisWorkspace, tmpBasisWorkspace, &basis));

    if (aux) {
        quick_cuest_struct.persistAuxBasisWorkspace = persistBasisWorkspace;
        quick_cuest_struct.auxBasis                 = basis;
    } else {
        quick_cuest_struct.persistAOBasisWorkspace = persistBasisWorkspace;
        quick_cuest_struct.basis                   = basis;
    }

    freeWorkspace (tmpBasisWorkspace);
    checkCuestErrors (cuestParametersDestroy (CUEST_AOBASIS_PARAMETERS, basis_params));

    for (size_t i = 0; i < nshell; ++i)
        checkCuestErrors (cuestAOShellDestroy (shells[i]));

    free (shells);
    free (nshells_per_atom);

#ifdef CUESTDEBUG
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

    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_ATOM, &query_natom, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_SHELL, &query_nshell, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_AO, &query_nao, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_CART, &query_ncart, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_NUM_PRIMITIVE, &query_nprimitive,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_MAX_L, &query_max_L, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (quick_cuest_struct.handle, CUEST_AOBASIS, basis,
                                  CUEST_AOBASIS_IS_PURE, &query_is_pure, sizeof (int32_t)));

    printf ("AO Basis from handle:\n");
    printf ("%-10s = %6llu\n", "natom", query_natom);
    printf ("%-10s = %6llu\n", "nshell", query_nshell);
    printf ("%-10s = %6llu\n", "nao", query_nao);
    printf ("%-10s = %6llu\n", "ncart", query_ncart);
    printf ("%-10s = %6llu\n", "nprimitive", query_nprimitive);
    printf ("%-10s = %6llu\n", "max_L", query_max_L);
    printf ("%-10s = %6s\n", "is_pure", query_is_pure ? "true" : "false");

    // for debugging auxiliary basis
    if (aux) {
        cuestAOPairList_t           pair_list;
        cuestAOPairListParameters_t pair_list_params;
        checkCuestErrors (cuestParametersCreate (CUEST_AOPAIRLIST_PARAMETERS, &pair_list_params));
        checkCuestErrors (cuestAOPairListCreateWorkspaceQuery (
            quick_cuest_struct.handle, quick_cuest_struct.auxBasis, quick_cuest_data.natom,
            quick_cuest_data.xyz, 1e-12, pair_list_params, quick_cuest_struct.persistWD,
            quick_cuest_struct.tmpWD, &pair_list));

        cuestWorkspace_t *pair_list_wksp         = allocateWorkspace (quick_cuest_struct.persistWD);
        cuestWorkspace_t *tmpAOPairListWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);

        checkCuestErrors (cuestAOPairListCreate (
            quick_cuest_struct.handle, quick_cuest_struct.auxBasis, quick_cuest_data.natom,
            quick_cuest_data.xyz, 1e-12, pair_list_params, pair_list_wksp, tmpAOPairListWorkspace,
            &pair_list));
        checkCuestErrors (cuestParametersDestroy (CUEST_AOPAIRLIST_PARAMETERS, pair_list_params));
        freeWorkspace (tmpAOPairListWorkspace);
        // free (xyz_flat);

        // ========================== //
        // one-electron integral plan //
        // ========================== //

        cuestOEIntPlan_t           oeint_plan;
        cuestOEIntPlanParameters_t oeint_plan_params;
        checkCuestErrors (cuestParametersCreate (CUEST_OEINTPLAN_PARAMETERS, &oeint_plan_params));
        checkCuestErrors (cuestOEIntPlanCreateWorkspaceQuery (
            quick_cuest_struct.handle, quick_cuest_struct.auxBasis, pair_list, oeint_plan_params,
            quick_cuest_struct.persistWD, quick_cuest_struct.tmpWD, &oeint_plan));

        cuestWorkspace_t *oeint_plan_wksp       = allocateWorkspace (quick_cuest_struct.persistWD);
        cuestWorkspace_t *tmpOEIntPlanWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
        checkCuestErrors (cuestOEIntPlanCreate (
            quick_cuest_struct.handle, quick_cuest_struct.auxBasis, pair_list, oeint_plan_params,
            oeint_plan_wksp, tmpOEIntPlanWorkspace, &oeint_plan));

        checkCuestErrors (cuestParametersDestroy (CUEST_OEINTPLAN_PARAMETERS, oeint_plan_params));
        freeWorkspace (tmpOEIntPlanWorkspace);

        double *d_S;
        size_t  d_S_siz = query_nao * query_nao * sizeof (double);
        if (cudaMalloc ((void **)&d_S, d_S_siz) != cudaSuccess) {
            fprintf (stderr, "Failed to allocate device buffer\n");
            exit (EXIT_FAILURE);
        }

        cuestOverlapComputeParameters_t overlap_compute_params;
        checkCuestErrors (
            cuestParametersCreate (CUEST_OVERLAPCOMPUTE_PARAMETERS, &overlap_compute_params));
        checkCuestErrors (cuestOverlapComputeWorkspaceQuery (quick_cuest_struct.handle, oeint_plan,
                                                             overlap_compute_params,
                                                             quick_cuest_struct.tmpWD, d_S));

        cuestWorkspace_t *tmpSWorkspace = allocateWorkspace (quick_cuest_struct.tmpWD);
        printf ("oeint_plan: %p\noverlap_compute_params: %p\ntmpSWorkspace: %p\nd_S: %p\n",
                oeint_plan, overlap_compute_params, tmpSWorkspace, d_S);
        checkCuestErrors (cuestOverlapCompute (quick_cuest_struct.handle, oeint_plan,
                                               overlap_compute_params, tmpSWorkspace, d_S));

        freeWorkspace (tmpSWorkspace);
        checkCuestErrors (
            cuestParametersDestroy (CUEST_OVERLAPCOMPUTE_PARAMETERS, overlap_compute_params));

        // ==================== //
        // print overlap matrix //
        // ==================== //

        double *buf = malloc (d_S_siz);
        if (cudaMemcpy (buf, d_S, d_S_siz, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf (stderr, "cudaMemcpy failed on line %d\n", __LINE__);
            exit (EXIT_FAILURE);
        }

        puts ("-------- DEBUG aux S --------");
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j)
                printf ("%16.10f", buf[i * 113 + j]);
            putchar ('\n');
        }
        puts ("------ END DEBUG aux S ------");

        if (cudaFree (d_S) != cudaSuccess) {
            fprintf (stderr, "cudaFree failed on line %d\n", __LINE__);
            exit (EXIT_FAILURE);
        }

        free (buf);
        checkCuestErrors (cuestOEIntPlanDestroy (oeint_plan));
        freeWorkspace (oeint_plan_wksp);
        checkCuestErrors (cuestAOPairListDestroy (pair_list));
        freeWorkspace (pair_list_wksp);
    }
#endif
}

#undef get_L_sph
