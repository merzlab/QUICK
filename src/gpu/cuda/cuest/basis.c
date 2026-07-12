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

void
cuest_init_basis (int64_t *ncenter, int64_t *katom_, int64_t *ktype_, int64_t *kprim_, double *aexp,
                  double *dcoeff, bool aux)
{
    uint64_t natom = quick_cuest_data.natom;

    uint64_t maxcontract, nshell;
    if (aux) {
        nshell      = quick_cuest_data.nauxshell;
        maxcontract = quick_cuest_data.maxcontract_aux;
    } else {
        nshell      = quick_cuest_data.nshell;
        maxcontract = quick_cuest_data.maxcontract;
    }

#ifdef CUESTDEBUG
    DEBUGLOG ("-------- DUMP --------\n");

    DEBUGLOG ("natom=%llu\n", natom);
    DEBUGLOG ("nshell=%llu\n", nshell);

    DEBUGLOG ("ncenter:\n");
    for (int i = 0; i < 7; ++i)
        DEBUGLOG ("%llu ", ncenter[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("katom_:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", katom_[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("ktype_:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", ktype_[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("kprim_:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", kprim_[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("aexp:\n");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxcontract; ++j)
            DEBUGLOG ("%f ", get (aexp, i, j, maxcontract));
        DEBUGLOG ("\n");
    }

    DEBUGLOG ("dcoeff:\n");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxcontract; ++j)
            DEBUGLOG ("%f ", get (dcoeff, i, j, maxcontract));
        DEBUGLOG ("\n");
    }

    DEBUGLOG ("xyz:\n");
    for (int i = 0; i < natom; ++i) {
        for (int j = 0; j < 3; ++j)
            DEBUGLOG ("%f ", get (quick_cuest_data.xyz, i, j, 3));
        DEBUGLOG ("\n");
    }

    DEBUGLOG ("aexp flat:\n");
    for (int i = 0; i < 7 * maxcontract; ++i)
        DEBUGLOG ("%f ", aexp[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("dcoeff flat:\n");
    for (int i = 0; i < 7 * maxcontract; ++i)
        DEBUGLOG ("%f ", dcoeff[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("xyz flat:\n");
    for (int i = 0; i < natom * 3; ++i)
        DEBUGLOG ("%f ", quick_cuest_data.xyz[i]);
    DEBUGLOG ("\n");

    DEBUGLOG ("------ END DUMP ------\n");

    fflush (stdout);
#endif

    cuestHandle_t               handle    = quick_cuest_struct.handle;
    cuestWorkspaceDescriptor_t *tmpWD     = quick_cuest_struct.tmpWD;
    cuestWorkspaceDescriptor_t *persistWD = quick_cuest_struct.persistWD;

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
    if (!aux)
        quick_cuest_memchk.chk_katom_ktype_kprim = chk_katom_ktype_kprim;
    // expanded arrays (SP -> S and P)
    uint64_t *katom = chk_katom_ktype_kprim;
    uint64_t *ktype = katom + nshell;
    uint64_t *kprim = ktype + nshell;

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
    DEBUGLOG ("new katom:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", katom[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("new ktype:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", ktype[i]);
    DEBUGLOG ("\n");
    DEBUGLOG ("new kprim:\n");
    for (int i = 0; i < nshell; ++i)
        DEBUGLOG ("%llu ", kprim[i]);
    DEBUGLOG ("\n");
    fflush (stdout);
#endif

    // first_basis_function but instead first_basis_shell
    size_t *ifshell = malloc (nshell * sizeof (size_t));
    if (!aux)
        quick_cuest_data.ifshell = ifshell;
    // nshells_per_atom[a] is number of shells atom `a` has.
    // This is the same as the number of times it appears in `katom`.
    uint64_t *nshells_per_atom = calloc (natom, sizeof (uint64_t));

    ifshell[0] = 0;

    for (size_t i = 0; i < nshell; ++i) {
        uint64_t a = katom[i] - 1;
        ++nshells_per_atom[a];
        if (i > 0)
            ifshell[i] = ifshell[i - 1] + ktype[i - 1];
        // DEBUGLOG ("ifshell[%zu]=%zu\n", i, ifshell[i]);
    }

    // fflush (stdout);

    // -------------------- //
    // 2. make cuest shells //
    // -------------------- //

    cuestAOShell_t *shells = malloc (nshell * sizeof (cuestAOShell_t));
    if (!shells) {
        fprintf (stderr, "Failed to allocate AO shell array\n");
        checkCuestErrors (cuestDestroy (handle));
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

        checkCuestErrors (cuestAOShellCreate (handle, aux,
                                              aux ? get_L_sph (ktype[i]) : get_L_cart (ktype[i]),
                                              kprim[i], a, c, aoshell_params, &shells[i]));
    }

    // free (coeff);
    if (aux) {
        free (ifshell);
        free (chk_katom_ktype_kprim);
    }

    checkCuestErrors (cuestParametersDestroy (CUEST_AOSHELL_PARAMETERS, aoshell_params));

    // ============ //
    // set up basis //
    // ============ //

    cuestAOBasis_t           basis;
    cuestAOBasisParameters_t basis_params;
    checkCuestErrors (cuestParametersCreate (CUEST_AOBASIS_PARAMETERS, &basis_params));

    checkCuestErrors (cuestAOBasisCreateWorkspaceQuery (handle, natom, nshells_per_atom, shells,
                                                        basis_params, persistWD, tmpWD, &basis));

    if (aux)
        MEMLOG ("cuEST Primary Basis");
    else
        MEMLOG ("cuEST Auxiliary (Density Fit) Basis");

    cuestWorkspace_t *persistBasisWorkspace = allocateWorkspace (persistWD);
    cuestWorkspace_t *tmpBasisWorkspace     = allocateWorkspace (tmpWD);

    checkCuestErrors (cuestAOBasisCreate (handle, natom, nshells_per_atom, shells, basis_params,
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

    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_ATOM,
                                  &query_natom, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_SHELL,
                                  &query_nshell, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_AO, &query_nao,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_CART,
                                  &query_ncart, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_PRIMITIVE,
                                  &query_nprimitive, sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_MAX_L, &query_max_L,
                                  sizeof (uint64_t)));
    checkCuestErrors (cuestQuery (handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_IS_PURE,
                                  &query_is_pure, sizeof (int32_t)));

    DEBUGLOG ("AO Basis from handle:\n");
    DEBUGLOG ("%-10s = %6llu\n", "natom", query_natom);
    DEBUGLOG ("%-10s = %6llu\n", "nshell", query_nshell);
    DEBUGLOG ("%-10s = %6llu\n", "nao", query_nao);
    DEBUGLOG ("%-10s = %6llu\n", "ncart", query_ncart);
    DEBUGLOG ("%-10s = %6llu\n", "nprimitive", query_nprimitive);
    DEBUGLOG ("%-10s = %6llu\n", "max_L", query_max_L);
    DEBUGLOG ("%-10s = %6s\n", "is_pure", query_is_pure ? "true" : "false");
#endif
}

#undef get_L_sph
