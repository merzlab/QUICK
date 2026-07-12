#include <stdio.h>

#include "quick_cuest.h"

quick_cuest_struct_t   quick_cuest_struct;
quick_cuest_data_t     quick_cuest_data;
quick_cuest_memchk_t   quick_cuest_memchk;
quick_cuest_memtrace_t quick_cuest_memtrace;
FILE                  *quick_cuest_log_fp;

void
cuest_get_memtrace (int64_t *hostmax, int64_t *hosttotal, int64_t *hostallocs, int64_t *devmax,
                    int64_t *devtotal, int64_t *devallocs)
{
    *hostmax    = quick_cuest_memtrace.hostmax;
    *hosttotal  = quick_cuest_memtrace.hosttotal;
    *hostallocs = quick_cuest_memtrace.hostallocs;
    *devmax     = quick_cuest_memtrace.devmax;
    *devtotal   = quick_cuest_memtrace.devtotal;
    *devallocs  = quick_cuest_memtrace.devallocs;
}
