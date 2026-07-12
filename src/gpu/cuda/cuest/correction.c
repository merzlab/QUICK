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
cuest_init_correct ()
{
    uint64_t  nbasis = quick_cuest_data.nbasis;
    uint64_t  nshell = quick_cuest_data.nshell;
    uint64_t *ktype  = quick_cuest_memchk.chk_katom_ktype_kprim + nshell;

    double *tmpbuf_dp         = malloc (nbasis * sizeof (double));
    void   *chk_firstd_firstf = malloc ((nbasis + 1) * sizeof (size_t));

    // all have length ceil(nbasis/2)
    const size_t len    = ((nbasis + 1) >> 1);
    size_t      *firstd = chk_firstd_firstf;
    size_t      *firstf = firstd + len;
    size_t       ifd = 0, iff = 0;

    // get index of d and f orbitals
    for (size_t i = 0; i < nshell; ++i) {
        if (ktype[i] == KTYPE_CART_D)
            firstd[ifd++] = quick_cuest_data.ifshell[i];
        else if (ktype[i] == KTYPE_CART_D)
            firstf[iff++] = quick_cuest_data.ifshell[i];
    }

    quick_cuest_memchk.tmpbuf_dp         = tmpbuf_dp;
    quick_cuest_memchk.chk_firstd_firstf = chk_firstd_firstf;
    quick_cuest_memchk.ifd               = ifd;
    quick_cuest_memchk.iff               = iff;
}

void
cuest_deinit_correct ()
{
    free (quick_cuest_memchk.chk_firstd_firstf);
    free (quick_cuest_memchk.tmpbuf_dp);
}

#define APPLY_NORM_ROW(basei, ii, normfac)                                                         \
    do {                                                                                           \
        for (int jjj = 0; jjj < nbasis; ++jjj)                                                     \
            get (o, basei + (ii), jjj, nbasis) *= (normfac);                                       \
    } while (0);

#define MEMSWP_IND(basei, ii, jj)                                                                  \
    do {                                                                                           \
        memcpy (tmpbuf, get_row_ptr (o, (basei) + (ii), nbasis), rowsiz);                          \
        memcpy (get_row_ptr (o, (basei) + (ii), nbasis), get_row_ptr (o, (basei) + (jj), nbasis),  \
                rowsiz);                                                                           \
        memcpy (get_row_ptr (o, (basei) + (jj), nbasis), tmpbuf, rowsiz);                          \
    } while (0)

#define SWP_IND(basei, ii, jj)                                                                     \
    do {                                                                                           \
        SWAP (get (o, i, (basei) + (ii), nbasis), get (o, i, (basei) + (jj), nbasis), double);     \
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
cuest_correct_o (double *o, int8_t qspec)
{
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nshell = quick_cuest_data.nshell;

    size_t *firstd = quick_cuest_memchk.chk_firstd_firstf;
    size_t *firstf = firstd + ((nbasis + 1) >> 1);
    size_t  ifd    = quick_cuest_memchk.ifd;
    size_t  iff    = quick_cuest_memchk.iff;

    double *tmpbuf = quick_cuest_memchk.tmpbuf_dp;

    bool reorder            = qspec & CORRECT_REORDER;
    bool norm               = qspec & CORRECT_NORM_;
    bool reorder_from_quick = qspec & CORRECT_FROMQUICK_;
    bool norminv = ((qspec & CORRECT_FROMQUICK_) >> 2) ^ ((qspec & CORRECT_NORM_INV) >> 3);

    double kdxx, kfxxy, kfxyz;
    if (norminv) {
        kdxx  = SQRT_3_INV;
        kfxxy = SQRT_5_INV;
        kfxyz = SQRT_15_INV;
    } else {
        kdxx  = SQRT_3;
        kfxxy = SQRT_5;
        kfxyz = SQRT_15;
    }

    // TODO(michaelyxsun): add flag to operate only on upper or lower half of matrix

    if (reorder) {
        // copy row col intersections to other side of diagonal
        // right now only i>=j is filled
        for (size_t i = 0; i < ifd; ++i)
            for (size_t j = 0; j < i; ++j)
                get (o, firstd[j], firstd[i], nbasis) = get (o, firstd[i], firstd[j], nbasis);

        for (size_t i = 0; i < iff; ++i)
            for (size_t j = 0; j < i; ++j)
                get (o, firstf[j], firstf[i], nbasis) = get (o, firstf[i], firstf[j], nbasis);
    }

    const size_t rowsiz = nbasis * sizeof (double);

    // normalize then reorder so normalize doesn't depend on if we reorder
    if (norm) {
        if (reorder_from_quick) {
            // row normalization
            for (int i = 0; i < ifd; ++i) {
                APPLY_NORM_ROW (firstd[i], 1, kdxx);
                APPLY_NORM_ROW (firstd[i], 3, kdxx);
                APPLY_NORM_ROW (firstd[i], 4, kdxx);
            }

            for (int i = 0; i < iff; ++i) {
                APPLY_NORM_ROW (firstf[i], 1, kfxxy);
                APPLY_NORM_ROW (firstf[i], 2, kfxxy);
                APPLY_NORM_ROW (firstf[i], 4, kfxxy);
                APPLY_NORM_ROW (firstf[i], 5, kfxyz);
                APPLY_NORM_ROW (firstf[i], 6, kfxxy);
                APPLY_NORM_ROW (firstf[i], 7, kfxxy);
                APPLY_NORM_ROW (firstf[i], 8, kfxxy);
            }

            // column normalization
            for (int i = 0; i < nbasis; ++i) {
                for (int j = 0; j < ifd; ++j) {
                    get (o, i, firstd[j] + 1, nbasis) *= kdxx;
                    get (o, i, firstd[j] + 3, nbasis) *= kdxx;
                    get (o, i, firstd[j] + 4, nbasis) *= kdxx;
                }

                for (int j = 0; j < iff; ++j) {
                    get (o, i, firstf[j] + 1, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 2, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 4, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 5, nbasis) *= kfxyz;
                    get (o, i, firstf[j] + 6, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 7, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 8, nbasis) *= kfxxy;
                }
            }
        } else {
            // row normalization
            for (int i = 0; i < ifd; ++i) {
                APPLY_NORM_ROW (firstd[i], 1, kdxx);
                APPLY_NORM_ROW (firstd[i], 2, kdxx);
                APPLY_NORM_ROW (firstd[i], 4, kdxx);
            }

            for (int i = 0; i < iff; ++i) {
                APPLY_NORM_ROW (firstf[i], 1, kfxxy);
                APPLY_NORM_ROW (firstf[i], 2, kfxxy);
                APPLY_NORM_ROW (firstf[i], 3, kfxxy);
                APPLY_NORM_ROW (firstf[i], 4, kfxyz);
                APPLY_NORM_ROW (firstf[i], 5, kfxxy);
                APPLY_NORM_ROW (firstf[i], 7, kfxxy);
                APPLY_NORM_ROW (firstf[i], 8, kfxxy);
            }

            // column normalization
            for (int i = 0; i < nbasis; ++i) {
                for (int j = 0; j < ifd; ++j) {
                    get (o, i, firstd[j] + 1, nbasis) *= kdxx;
                    get (o, i, firstd[j] + 2, nbasis) *= kdxx;
                    get (o, i, firstd[j] + 4, nbasis) *= kdxx;
                }

                for (int j = 0; j < iff; ++j) {
                    get (o, i, firstf[j] + 1, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 2, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 3, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 4, nbasis) *= kfxyz;
                    get (o, i, firstf[j] + 5, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 7, nbasis) *= kfxxy;
                    get (o, i, firstf[j] + 8, nbasis) *= kfxxy;
                }
            }
        }
    }

    if (reorder) {
        for (int i = 0; i < ifd; ++i)
            MEMSWP_IND (firstd[i], 2, 3);

        for (int i = 0; i < iff; ++i) {
            MEMSWP_IND (firstf[i], 2, 4);
            MEMSWP_IND (firstf[i], 3, 4);
            MEMSWP_IND (firstf[i], 4, 5);
            MEMSWP_IND (firstf[i], 5, 7);
            MEMSWP_IND (firstf[i], 6, 7);
        }

        for (int i = 0; i < nbasis; ++i) {
            for (int j = 0; j < ifd; ++j)
                SWP_IND (firstd[j], 2, 3);

            for (int j = 0; j < iff; ++j) {
                SWP_IND (firstf[j], 2, 4);
                SWP_IND (firstf[j], 3, 4);
                SWP_IND (firstf[j], 4, 5);
                SWP_IND (firstf[j], 5, 7);
                SWP_IND (firstf[j], 6, 7);
            }
        }
    }
}

#undef APPLY_NORM_ROW
#undef MEMSWP_IND
#undef SWP_IND

void
cuest_correct_P (double *o, int8_t qspec)
{
    cuest_correct_o (o, qspec | CORRECT_NORM_INV);
}

void
cuest_correct_C (double *C, size_t nocc, int8_t qspec)
{
    uint64_t nbasis = quick_cuest_data.nbasis;
    uint64_t nshell = quick_cuest_data.nshell;

    size_t *firstd = quick_cuest_memchk.chk_firstd_firstf;
    size_t *firstf = firstd + ((nbasis + 1) >> 1);
    size_t  ifd    = quick_cuest_memchk.ifd;
    size_t  iff    = quick_cuest_memchk.iff;

    bool reorder            = qspec & CORRECT_REORDER;
    bool norm               = qspec & CORRECT_NORM_;
    bool reorder_from_quick = qspec & CORRECT_FROMQUICK_;
    bool norminv = ~((qspec & CORRECT_FROMQUICK_) >> 2) ^ ((qspec & CORRECT_NORM_INV) >> 3);

    double kdxx, kfxxy, kfxyz;
    if (norminv) {
        kdxx  = SQRT_3_INV;
        kfxxy = SQRT_5_INV;
        kfxyz = SQRT_15_INV;
    } else {
        kdxx  = SQRT_3;
        kfxxy = SQRT_5;
        kfxyz = SQRT_15;
    }

#ifdef CUESTDEBUG
    fputs ("======== C from QUICK ========", quick_cuest_log_fp);
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            DEBUGLOG ("%f ", get (C, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    fputs ("====== end C from QUICK ======", quick_cuest_log_fp);
#endif

    const size_t endi = nocc * nbasis;

    if (norm) {
        if (reorder_from_quick) {
            for (size_t i = 0; i < endi; i += nbasis) {
                for (size_t j = 0; j < ifd; ++j) {
                    C[firstd[j] + i + 1] *= kdxx;
                    C[firstd[j] + i + 3] *= kdxx;
                    C[firstd[j] + i + 4] *= kdxx;
                }

                for (size_t j = 0; j < iff; ++j) {
                    C[firstf[j] + i + 1] *= kfxxy;
                    C[firstf[j] + i + 2] *= kfxxy;
                    C[firstf[j] + i + 4] *= kfxxy;
                    C[firstf[j] + i + 5] *= kfxyz;
                    C[firstf[j] + i + 6] *= kfxxy;
                    C[firstf[j] + i + 7] *= kfxxy;
                    C[firstf[j] + i + 8] *= kfxxy;
                }
            }
        } else {
            for (size_t i = 0; i < endi; i += nbasis) {
                for (size_t j = 0; j < ifd; ++j) {
                    C[firstd[j] + i + 1] *= kdxx;
                    C[firstd[j] + i + 2] *= kdxx;
                    C[firstd[j] + i + 4] *= kdxx;
                }

                for (size_t j = 0; j < iff; ++j) {
                    C[firstf[j] + i + 1] *= kfxxy;
                    C[firstf[j] + i + 2] *= kfxxy;
                    C[firstf[j] + i + 3] *= kfxxy;
                    C[firstf[j] + i + 4] *= kfxyz;
                    C[firstf[j] + i + 5] *= kfxxy;
                    C[firstf[j] + i + 7] *= kfxxy;
                    C[firstf[j] + i + 8] *= kfxxy;
                }
            }
        }
    }

    if (reorder) {
        for (size_t i = 0; i < endi; i += nbasis) {
            for (size_t j = 0; j < ifd; ++j)
                reorder_d (C + firstd[j] + i);

            for (size_t j = 0; j < iff; ++j)
                reorder_f (C + firstf[j] + i);
        }
    }

#ifdef CUESTDEBUG
    fputs ("======== C reordered ========", quick_cuest_log_fp);
    for (int i = 0; i < nocc; ++i) {
        for (int j = 0; j < quick_cuest_data.nbasis; ++j)
            DEBUGLOG ("%f ", get (C, i, j, quick_cuest_data.nbasis));
        putchar ('\n');
    }
    fputs ("====== end C reordered ======", quick_cuest_log_fp);
#endif
}
