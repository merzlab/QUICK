/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 11/08/2020                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions and device kernels required     !
  ! for custom matrix operations.                                       !
  !---------------------------------------------------------------------!
*/
#include "gpu.h"
#include <cuda.h>

static __constant__ gpu_simulation_type devSim_matop;

// upload gpu simulation type to constant memory
void upload_sim_to_constant_matop(_gpu_type gpu){
    cudaError_t status;
    PRINTDEBUG("UPLOAD CONSTANT MATOP");
    status = cudaMemcpyToSymbol(devSim_matop, &gpu->gpu_sim, sizeof(gpu_simulation_type));
    PRINTERROR(status, " cudaMemcpyToSymbol, matop sim copy to constants failed")
    PRINTDEBUG("FINISH UPLOAD CONSTANT MATOP");
}

#ifdef DEBUG
static float totTime;
#endif

void get_dmx(_gpu_type gpu){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

        QUICK_SAFE_CALL((get_dmx_kernel<<< gpu -> blocks, gpu -> threadsPerBlock>>>()));

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    fprintf(gpu->debugFile,"Time to form new density matrix:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

}


