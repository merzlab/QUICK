#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "correction.h"
#include "quick_cuest.h"
#include "util.h"

void
init_correct ()
{
    uint64_t  nbasis = quick_cuest_data.nbasis;
    uint64_t  nshell = quick_cuest_data.nshell;
    uint64_t *ktype  = quick_cuest_memchk.chk_katom_ktype_kprim + nshell;

    double *tmpbuf_dp        = malloc (nbasis * sizeof (double));
    void   *chk_firstdf_mark = malloc (nbasis * (sizeof (size_t) + sizeof (bool)));
    size_t *firstdf          = chk_firstdf_mark;
    bool   *mark             = (bool *)((size_t *)chk_firstdf_mark + nbasis);
    size_t  ifdf             = 0;

    // get index of d and f orbitals
    for (size_t i = 0; i < nshell; ++i)
        if (ktype[i] == KTYPE_CART_D || ktype[i] == KTYPE_CART_F) {
            firstdf[ifdf] = quick_cuest_data.ifshell[i];
            mark[ifdf]    = ktype[i] == KTYPE_CART_D;
            ++ifdf;
        }

    quick_cuest_memchk.tmpbuf_dp        = tmpbuf_dp;
    quick_cuest_memchk.chk_firstdf_mark = chk_firstdf_mark;
    quick_cuest_memchk.ifdf             = ifdf;

#ifdef CUESTDEBUG
    printf ("ifdf=%zu\n", ifdf);
    for (size_t i = 0; i < ifdf; ++i)
        printf ("firstdf[%zu]=%zu\tmark[%zu]=%s\n", i, firstdf[i], i, mark[i] ? "T" : "F");
#endif
}

void
deinit_correct ()
{
    free (quick_cuest_memchk.chk_firstdf_mark);
    free (quick_cuest_memchk.tmpbuf_dp);
}

#define MEMSWP_IND(ii, jj)                                                                         \
    do {                                                                                           \
        memcpy (tmpbuf, get_row_ptr (o, firstdf[i] + (ii), nbasis), rowsiz);                       \
        memcpy (get_row_ptr (o, firstdf[i] + (ii), nbasis),                                        \
                get_row_ptr (o, firstdf[i] + (jj), nbasis), rowsiz);                               \
        memcpy (get_row_ptr (o, firstdf[i] + (jj), nbasis), tmpbuf, rowsiz);                       \
    } while (0)

#define SWP_IND(ii, jj)                                                                            \
    do {                                                                                           \
        SWAP (get (o, i, firstdf[j] + (ii), nbasis), get (o, i, firstdf[j] + (jj), nbasis),        \
              double);                                                                             \
    } while (0)

/**
 * Corrects matrix m by fixing the order of d and f orbitals and optionally their normalization
 * Normalization is applied according to cuEST ordering.
 *
 *     d_xy  type has extra 1/sqrt(3)
 *     f_xxy type has extra 1/sqrt(5)
 *     f_xyz type has extra 1/sqrt(15)
 *
 * QUICK
 *        1   2   3   4   5   6
 *     d: xx  xy  yy  xz  yz  zz
 *     f: xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz
 *
 * cuEST
 *        1   2   3   4   5   6   7   8   9   10
 *     d: xx  xy  xz  yy  yz  zz
 *     f: xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
 */
void
correct_o (double *o, uint8_t qspec)
{
    if (qspec == 0 || qspec > CORRECT_REORDER_AND_NORM)
        return;

    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nshell = quick_cuest_data.nshell;

    size_t *firstdf = quick_cuest_memchk.chk_firstdf_mark;
    bool   *mark    = (bool *)((size_t *)quick_cuest_memchk.chk_firstdf_mark + nbasis);
    size_t  ifdf    = quick_cuest_memchk.ifdf;

    double *tmpbuf = quick_cuest_memchk.tmpbuf_dp;

    bool reorder = qspec & CORRECT_REORDER;
    bool norm    = qspec & CORRECT_NORM;

    if (reorder) {
        // copy row col intersections to other side of diagonal
        // right now only i>=j is filled
        for (size_t i = 0; i < ifdf; ++i)
            for (size_t j = 0; j < i; ++j)
                get (o, firstdf[j], firstdf[i], nbasis) = get (o, firstdf[i], firstdf[j], nbasis);
    }

    const size_t rowsiz = nbasis * sizeof (double);

#define APPLY_NORM_ROW(ii, norm)                                                                   \
    do {                                                                                           \
        for (int j = 0; j < nbasis; ++j)                                                           \
            get (o, firstdf[i] + (ii), j, nbasis) *= (norm);                                       \
    } while (0);

    // normalize then reorder so normalize doesn't depend on if we reorder
    for (int i = 0; i < ifdf; ++i) {
        if (mark[i]) {
            if (norm) {
                APPLY_NORM_ROW (1, SQRT_3);
                APPLY_NORM_ROW (2, SQRT_3);
                APPLY_NORM_ROW (4, SQRT_3);
            }

            if (reorder)
                MEMSWP_IND (2, 3);
        } else {
            if (norm) {
                APPLY_NORM_ROW (1, SQRT_5);
                APPLY_NORM_ROW (2, SQRT_5);
                APPLY_NORM_ROW (3, SQRT_5);
                APPLY_NORM_ROW (4, SQRT_15);
                APPLY_NORM_ROW (5, SQRT_5);
                APPLY_NORM_ROW (7, SQRT_5);
                APPLY_NORM_ROW (8, SQRT_5);
            }

            if (reorder) {
                MEMSWP_IND (2, 4);
                MEMSWP_IND (3, 4);
                MEMSWP_IND (4, 5);
                MEMSWP_IND (5, 7);
                MEMSWP_IND (6, 7);
            }
        }
    }
#undef APPLY_NORM_ROW

    for (int i = 0; i < nbasis; ++i)
        for (int j = 0; j < ifdf; ++j) {
            if (mark[j]) {
                if (norm) {
                    get (o, i, firstdf[j] + 1, nbasis) *= SQRT_3;
                    get (o, i, firstdf[j] + 2, nbasis) *= SQRT_3;
                    get (o, i, firstdf[j] + 4, nbasis) *= SQRT_3;
                }

                if (reorder)
                    SWP_IND (2, 3);
            } else {
                if (norm) {
                    get (o, i, firstdf[j] + 1, nbasis) *= SQRT_5;
                    get (o, i, firstdf[j] + 2, nbasis) *= SQRT_5;
                    get (o, i, firstdf[j] + 3, nbasis) *= SQRT_5;
                    get (o, i, firstdf[j] + 4, nbasis) *= SQRT_15;
                    get (o, i, firstdf[j] + 5, nbasis) *= SQRT_5;
                    get (o, i, firstdf[j] + 7, nbasis) *= SQRT_5;
                    get (o, i, firstdf[j] + 8, nbasis) *= SQRT_5;
                }

                if (reorder) {
                    SWP_IND (2, 4);
                    SWP_IND (3, 4);
                    SWP_IND (4, 5);
                    SWP_IND (5, 7);
                    SWP_IND (6, 7);
                }
            }
        }
}

void
reorder_PC (double *C, size_t nocc)
{
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nshell = quick_cuest_data.nshell;

    size_t *firstdf = quick_cuest_memchk.chk_firstdf_mark;
    bool   *mark    = (bool *)((size_t *)quick_cuest_memchk.chk_firstdf_mark + nbasis);
    size_t  ifdf    = quick_cuest_memchk.ifdf;

#ifdef CUESTDEBUG
    puts ("======== C from QUICK ========");
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%f ", get (C, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    puts ("====== end C from QUICK ======");
#endif

    for (size_t i = 0, end = nocc * quick_cuest_data.nbasis; i < end; i += quick_cuest_data.nbasis)
        for (size_t j = 0; j < ifdf; ++j) {
            if (mark[j])
                reorder_d (C + firstdf[j] + i);
            else
                reorder_f (C + firstdf[j] + i);
        }

#ifdef CUESTDEBUG
    puts ("======== C reordered ========");
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            printf ("%f ", get (C, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    puts ("====== end C reordered ======");
#endif
}

#undef MEMSWP_IND
#undef SWP_IND
