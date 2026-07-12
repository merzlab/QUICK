#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

#include "quick_cuest.h"

#define SWAP(x, y, type)                                                                           \
    do {                                                                                           \
        type swap_temporary_variable_88888888_ = (x);                                              \
        (x)                                    = (y);                                              \
        (y)                                    = swap_temporary_variable_88888888_;                \
    } while (0)

#define get(arr, i, j, ncols)      arr[(j) + (i) * (ncols)]
#define get_row_ptr(arr, i, ncols) (arr + (i) * (ncols))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// From Mathematica
#define SQRT_3      1.7320508075688772935274463415058723669428052538104
#define SQRT_5      2.2360679774997896964091736687312762354406183596115
#define SQRT_15     3.8729833462074168851792653997823996108329217052916
#define SQRT_3_INV  0.57735026918962576450914878050195745564760175127013
#define SQRT_5_INV  0.44721359549995793928183473374625524708812367192231
#define SQRT_15_INV 0.25819888974716112567861769331882664072219478035277

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

#define MEMLOG(desc)                                                                               \
    do {                                                                                           \
        fprintf (quick_cuest_log_fp, "%s: Persistent workspace requires %.2f GB of memory\n",      \
                 desc, quick_cuest_struct.persistWD->deviceBufferSizeInBytes / 1e9);               \
        fprintf (quick_cuest_log_fp, "%s: Temporary workspace requires %.2f GB of memory\n", desc, \
                 quick_cuest_struct.tmpWD->deviceBufferSizeInBytes / 1e9);                         \
    } while (0)

#define MEMLOG_TMPWD(desc)                                                                         \
    do {                                                                                           \
        fprintf (quick_cuest_log_fp, "%s: Temporary workspace requires %.2f GB of memory\n", desc, \
                 quick_cuest_struct.tmpWD->deviceBufferSizeInBytes / 1e9);                         \
    } while (0)

#define DEBUGLOG(...) fprintf (quick_cuest_log_fp, __VA_ARGS__);

#define cudaMallocChecked(ptr, siz)                                                                \
    do {                                                                                           \
        if (cudaMalloc (ptr, siz) != cudaSuccess) {                                                \
            fprintf (stderr, "cudaMalloc failed at %s:%d\n", __func__, __LINE__ - 1);              \
            exit (EXIT_FAILURE);                                                                   \
        }                                                                                          \
        add_dev_alloc (siz);                                                                       \
    } while (0)

#define cudaMemcpyChecked(dst, src, siz, dirspec)                                                  \
    do {                                                                                           \
        if (cudaMemcpy (dst, src, siz, dirspec) != cudaSuccess) {                                  \
            fprintf (stderr, "cudaMemcpy failed at %s:%d\n", __func__, __LINE__ - 1);              \
            exit (EXIT_FAILURE);                                                                   \
        }                                                                                          \
    } while (0)

#define cudaFreeChecked(ptr)                                                                       \
    do {                                                                                           \
        if (cudaFree (ptr) != cudaSuccess) {                                                       \
            fprintf (stderr, "cudaFree failed at %s:%d\n", __func__, __LINE__);                    \
            exit (EXIT_FAILURE);                                                                   \
        }                                                                                          \
    } while (0)

static void
add_host_alloc (size_t bytes)
{
    if ((quick_cuest_memtrace.hostcur += bytes) > quick_cuest_memtrace.hostmax)
        quick_cuest_memtrace.hostmax = quick_cuest_memtrace.hostcur;
    quick_cuest_memtrace.hosttotal += bytes;
    ++quick_cuest_memtrace.hostallocs;
}

static void
free_host_alloc (size_t bytes)
{
    quick_cuest_memtrace.hostcur -= bytes;
}

static void
add_dev_alloc (size_t bytes)
{
    if ((quick_cuest_memtrace.devcur += bytes) > quick_cuest_memtrace.devmax)
        quick_cuest_memtrace.devmax = quick_cuest_memtrace.devcur;
    quick_cuest_memtrace.devtotal += bytes;
    ++quick_cuest_memtrace.devallocs;
}

static void
free_dev_alloc (size_t bytes)
{
    quick_cuest_memtrace.devcur -= bytes;
}

#endif
