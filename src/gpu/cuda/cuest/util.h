#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdio.h>

#define get(arr, i, j, ncols)      arr[(j) + (i) * (ncols)]
#define get_row_ptr(arr, i, ncols) (arr + (i) * (ncols))

/**
 * `quick_type` should not be 4=sp
 */
uint64_t
get_L_cart (uint64_t quick_ktype)
{
    switch (quick_ktype) {
        case 1: // s
            return 0;
        case 3: // p
            return 1;
        case 6: // d
            return 2;
        case 10: // f
            return 3;
        case 15: // g
            return 4;
        case 21: // h
            return 5;
        default:
            fprintf (stderr, "get_L_cart(%llu): quick_ktype parameter invalid\n", quick_ktype);
            return 0;
    }
}

/**
 * `quick_type` should not be 4=sp
 */
#define get_L_sph(quick_ktype) (((quick_ktype) - 1) >> 1)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif
