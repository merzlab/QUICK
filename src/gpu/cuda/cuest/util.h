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

#endif
