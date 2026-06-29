#ifndef UTIL_H
#define UTIL_H

#define SWAP(x, y, type)                                                                           \
    do {                                                                                           \
        type swap_temporary_variable_88888888_ = (x);                                              \
        (x)                                    = (y);                                              \
        (y)                                    = swap_temporary_variable_88888888_;                \
    } while (0);

#define get(arr, i, j, ncols)      arr[(j) + (i) * (ncols)]
#define get_row_ptr(arr, i, ncols) (arr + (i) * (ncols))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

#endif
