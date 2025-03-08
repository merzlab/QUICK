/*
 *  gpu_info.cu
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#include <stdio.h>

#include "gpu_info.h"
#include "../gpu_common.h"


#define MAX_STR_LEN (256)


extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
        int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv)
{
    cudaDeviceProp prop;
    size_t device_mem;
    cudaError_t status;
    
    *gpu_dev_id = 0;  // currently one single GPU is supported
    gpuGetDeviceCount(gpu_dev_count);

    if (*gpu_dev_count == 0) 
    {
        printf("NO GPU DEVICE FOUND\n");
        exit(-1);
    }

    status = cudaGetDeviceProperties(&prop, *gpu_dev_id);
    PRINTERROR(status, "cudaGetDeviceProperties failed!");

    device_mem = (prop.totalGlobalMem / (1024 * 1024));
    *gpu_dev_mem = (int) device_mem;
    *gpu_num_proc = (int) (prop.multiProcessorCount);
    *gpu_core_freq = (double) (prop.clockRate * 1e-6f);
    strncpy(gpu_dev_name, prop.name, MAX_STR_LEN);
    *name_len = strnlen(gpu_dev_name, MAX_STR_LEN);
    *majorv = prop.major;
    *minorv = prop.minor;
}
