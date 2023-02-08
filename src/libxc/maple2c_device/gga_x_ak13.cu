#include <stdio.h>
#if defined HIP || defined HIP_MPIV
#include <hip/hip_runtime.h>
#endif
#include "../util.h"
#include "../gpu_fstructs.h"

#define DEVICE
static const double B1 =  1.74959015598863046792081721182; /* 3*muGE/5 + 8 pi/15 */ 
static const double B2 = -1.62613336586517367779736042170; /* muGE - B1 */ 
#include "../maple2c/gga_x_ak13.c"
