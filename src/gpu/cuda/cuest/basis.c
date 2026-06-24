#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LOCAL
#include "/Users/msun/rehs2026/cuest/libcuest-linux-sbsa-0.1.1.1_cuda13-archive/include/cuest.h"
#else
#include <cuest.h>
#endif

#include "helper_status.h"
#include "helper_workspace.h"
#include "util.h"

#include "quick_cuest.h"

void
cuest_init_basis (int64_t *ncenter, int64_t *first_basis_function, int64_t *last_basis_function,
                  int64_t *katom_, int64_t *ktype_, int64_t *kprim_, double *gcexpo,
                  double *gccoeff, bool aux)
{
    uint64_t maxprim, nshell;
    if (aux) {
        nshell  = quick_cuest_data.nauxshell;
        maxprim = quick_cuest_data.MAXPRIM_AUX;
    } else {
        nshell  = quick_cuest_data.nshell;
        maxprim = quick_cuest_data.MAXPRIM;
    }

    puts ("-------- DUMP --------");

    printf ("natom=%llu\n", quick_cuest_data.natom);
    printf ("nshell=%llu\n", nshell);

    puts ("ncenter:");
    for (int i = 0; i < 7; ++i)
        printf ("%llu ", ncenter[i]);
    putchar ('\n');

    puts ("first_basis_function:");
    for (int i = 0; i < quick_cuest_data.natom; ++i)
        printf ("%llu ", first_basis_function[i]);
    putchar ('\n');

    puts ("last_basis_function:");
    for (int i = 0; i < quick_cuest_data.natom; ++i)
        printf ("%llu ", last_basis_function[i]);
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

    puts ("gcexpo:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxprim; ++j)
            printf ("%f ", get (gcexpo, i, j, maxprim));
        putchar ('\n');
    }

    puts ("gccoeff:");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < maxprim; ++j)
            printf ("%f ", get (gccoeff, i, j, maxprim));
        putchar ('\n');
    }

    puts ("xyz:");
    for (int i = 0; i < quick_cuest_data.natom; ++i) {
        for (int j = 0; j < 3; ++j)
            printf ("%f ", get (quick_cuest_data.xyz, i, j, 3));
        putchar ('\n');
    }

    puts ("gcexpo flat:");
    for (int i = 0; i < 7 * maxprim; ++i)
        printf ("%f ", gcexpo[i]);
    putchar ('\n');

    puts ("gccoeff flat:");
    for (int i = 0; i < 7 * maxprim; ++i)
        printf ("%f ", gccoeff[i]);
    putchar ('\n');

    puts ("xyz flat:");
    for (int i = 0; i < quick_cuest_data.natom * 3; ++i)
        printf ("%f ", quick_cuest_data.xyz[i]);
    putchar ('\n');

    puts ("------ END DUMP ------");

    fflush (stdout);

    // ================ //
    // set up AO shells //
    // ================ //

    // preprocess and correct for SP shells

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

    // first_basis_function but instead first_basis_shell
    size_t *ifshell = malloc (nshell * sizeof (size_t));
    // nshells_per_atom[a] is number of shells atom `a` has.
    // This is the same as the number of times it appears in `katom`.
    uint64_t *nshells_per_atom = calloc (quick_cuest_data.natom, sizeof (uint64_t));

    ifshell[0] = 0;

    for (size_t i = 0; i < nshell; ++i) {
        uint64_t a = katom[i] - 1;
        ++nshells_per_atom[a];
        // ifshell[i] = first_basis_function[a] + shell_offset_cart[nshells_per_atom[a]++] - 1; //
        // <-- wrong
        if (i > 0)
            ifshell[i] = ifshell[i - 1] + ktype[i - 1]; // ktype stores number of cartesian orbitals
        printf ("ifshell[%zu]=%zu\n", i, ifshell[i]);
    }

    fflush (stdout);

    // start making shells

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
        // // manual normalization, same as pulling from QUICK
        // size_t   ifsh = ifshell[i];
        // uint64_t L    = get_L (ktype[i]);
        // normalize_coeff (dcoeff[ifsh], aexp[ifsh], 3, L, 1.0, coeff);
        // checkCuestErrors (
        //     cuestAOShellCreate (handle, 0, L, kprim[i],
        //     aexp[ifsh],
        //                         coeff, aoshell_params, &shells[i]));

        checkCuestErrors (cuestAOShellCreate (quick_cuest_struct.handle, 0, get_L (ktype[i]),
                                              kprim[i], get_row_ptr (gcexpo, ifshell[i], maxprim),
                                              get_row_ptr (gccoeff, ifshell[i], maxprim),
                                              aoshell_params, &shells[i]));
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
}
