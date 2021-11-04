/*
 *  gpu_info.cu
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#include "gpu_info.h"
#include "gpu_common.h"
#include "stdio.h"

// CUDA-C includes
#include <hip/hip_runtime_api.h>


extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
        int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv)
{
    hipError_t cuda_error;
    hipDeviceProp_t prop;
    size_t device_mem;
    
    *gpu_dev_id = 0;  // currently one single GPU is supported
    cuda_error = hipGetDeviceCount(gpu_dev_count);
    PRINTERROR(cuda_error,"hipGetDeviceCount gpu_get_device_info failed!");
    if (*gpu_dev_count == 0) 
    {
        printf("NO CUDA DEVICE FOUNDED \n");
        hipDeviceReset();
        exit(-1);
    }
    hipGetDeviceProperties(&prop,*gpu_dev_id);
    device_mem = (prop.totalGlobalMem/(1024*1024));
    *gpu_dev_mem = (int) device_mem;
    *gpu_num_proc = (int) (prop.multiProcessorCount);
    *gpu_core_freq = (double) (prop.clockRate * 1e-6f);
    strcpy(gpu_dev_name,prop.name);
    *name_len = strlen(gpu_dev_name);
    *majorv = prop.major;
    *minorv = prop.minor;
    
}
