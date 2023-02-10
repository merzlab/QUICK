#include <stdio.h>
#if defined HIP || defined HIP_MPIV
#include <hip/hip_runtime.h>
#endif
#define DEVICE
#include "../util.h"
#include "../gpu_fstructs.h"

#include "special_functions.c"
#include "../maple2c/gga_x_am05.c"
