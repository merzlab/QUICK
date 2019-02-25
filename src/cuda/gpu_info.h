/*
 *  gpu_info.h
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */


#include "../config.h"

extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
                                     int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv);
