#include <stdio.h>
#if defined HIP || defined HIP_MPIV
#include <hip/hip_runtime.h>
#endif
#include "../util.h"
#include "../gpu_fstructs.h"

#define DEVICE

#include "../maple2c/mgga_x_ms.c"
