/*
 *  gpu_startup.cu
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#include "gpu_startup.h"
#include "../config.h"

extern "C" void gpu_set_device_(int* gpu_dev_id)
{
    cudaError_t status;
    status=cudaSetDevice(*gpu_dev_id);
    cudaThreadSynchronize();
}