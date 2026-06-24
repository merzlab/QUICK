#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define get(arr, i, j, ncols)      arr[(j) + (i) * (ncols)]
#define get_row_ptr(arr, i, ncols) (arr + (i) * (ncols))

/**
 * `quick_type` should not be 4=sp
 */
uint64_t
get_L (uint64_t quick_ktype)
{
    switch (quick_ktype) {
        case 1:
            return 0;
        case 3:
            return 1;
        case 6:
            return 2;
        case 10:
            return 3;
        default:
            fprintf (stderr, "get_L(%llu): quick_ktype parameter invalid\n", quick_ktype);
            return 0;
    }
}

// done up to row 4
// 1s 2s 2p 3s 3p 4s 3d 4p
uint64_t shell_offset_cart[] = { 0, 1, 2, 5, 6, 9, 10, 16, 19 };
uint64_t shell_offset_sph[]  = { 0, 1, 2, 5, 6, 9, 10, 15, 18 };

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif
