#include <stdio.h>
#if defined HIP || defined HIP_MPIV
#include <hip/hip_runtime.h>
#endif
#define DEVICE
#include "../util.h"
#include "../gpu_fstructs.h"

#include "../expint_e1.c"
#include "../maple2c/gga_c_ft97.c"
