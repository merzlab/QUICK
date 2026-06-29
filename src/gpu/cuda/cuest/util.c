#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "quick_cuest.h"
#include "util.h"

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
 * Corrects matrix m by fixing the order of d and f orbitals and their normalization
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
correct_o (double *o)
{
    uint64_t  nbasis = quick_cuest_data.nbasis;
    uint64_t  nshell = quick_cuest_data.nshell;
    uint64_t *ktype  = quick_cuest_data.chk_katom_ktype_kprim + nshell;

    void *chk_firstdf_mark_tmpbuf
        = malloc (nbasis * (sizeof (size_t) + sizeof (bool) + sizeof (double)));
    size_t *firstdf = chk_firstdf_mark_tmpbuf;
    bool   *mark    = (bool *)((size_t *)chk_firstdf_mark_tmpbuf + nbasis);
    double *tmpbuf  = (double *)((uint8_t *)chk_firstdf_mark_tmpbuf
                                 + nbasis * (sizeof (size_t) + sizeof (bool)));
    size_t  ifdf    = 0;

    // get index of d and f orbitals
    for (size_t i = 0; i < nshell; ++i)
        if (ktype[i] == KTYPE_CART_D || ktype[i] == KTYPE_CART_F) {
            firstdf[ifdf] = quick_cuest_data.ifshell[i];
            mark[ifdf]    = ktype[i] == KTYPE_CART_D;
            ++ifdf;
        }

    // copy row col intersections to other side of diagonal
    // right now only i>=j is filled
    for (size_t i = 0; i < ifdf; ++i)
        for (size_t j = 0; j < i; ++j)
            get (o, firstdf[j], firstdf[i], nbasis) = get (o, firstdf[i], firstdf[j], nbasis);

    const size_t rowsiz = nbasis * sizeof (double);

#define APPLY_NORM_ROW(ii, norm)                                                                   \
    do {                                                                                           \
        for (int j = 0; j < nbasis; ++j)                                                           \
            get (o, firstdf[i] + (ii), j, nbasis) *= (norm);                                       \
    } while (0);

    for (int i = 0; i < ifdf; ++i) {
        if (mark[i]) {
            MEMSWP_IND (2, 3);
            APPLY_NORM_ROW (1, SQRT_3);
            APPLY_NORM_ROW (3, SQRT_3);
            APPLY_NORM_ROW (4, SQRT_3);
        } else {
            MEMSWP_IND (2, 4);
            MEMSWP_IND (3, 4);
            MEMSWP_IND (4, 5);
            MEMSWP_IND (5, 7);
            MEMSWP_IND (6, 7);
            APPLY_NORM_ROW (1, SQRT_5);
            APPLY_NORM_ROW (2, SQRT_5);
            APPLY_NORM_ROW (4, SQRT_5);
            APPLY_NORM_ROW (5, SQRT_15);
            APPLY_NORM_ROW (6, SQRT_5);
            APPLY_NORM_ROW (7, SQRT_5);
            APPLY_NORM_ROW (8, SQRT_5);
        }
    }
#undef APPLY_NORM_ROW

    for (int i = 0; i < nbasis; ++i)
        for (int j = 0; j < ifdf; ++j) {
            if (mark[j]) {
                SWP_IND (2, 3);
                get (o, i, firstdf[j] + 1, nbasis) *= SQRT_3;
                get (o, i, firstdf[j] + 3, nbasis) *= SQRT_3;
                get (o, i, firstdf[j] + 4, nbasis) *= SQRT_3;
            } else {
                SWP_IND (2, 4);
                SWP_IND (3, 4);
                SWP_IND (4, 5);
                SWP_IND (5, 7);
                SWP_IND (6, 7);
                get (o, i, firstdf[j] + 1, nbasis) *= SQRT_5;
                get (o, i, firstdf[j] + 2, nbasis) *= SQRT_5;
                get (o, i, firstdf[j] + 4, nbasis) *= SQRT_5;
                get (o, i, firstdf[j] + 5, nbasis) *= SQRT_15;
                get (o, i, firstdf[j] + 6, nbasis) *= SQRT_5;
                get (o, i, firstdf[j] + 7, nbasis) *= SQRT_5;
                get (o, i, firstdf[j] + 8, nbasis) *= SQRT_5;
            }
        }
}

#undef MEMSWP_IND
#undef SWP_IND
