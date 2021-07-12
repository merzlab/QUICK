/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 06/17/2021                            !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions required for QUICK one electron !
  ! integral computation.                                               !
  !---------------------------------------------------------------------!
*/

#include "gpu.h"
#include "gpu_common.h"

static __constant__ gpu_simulation_type devSim;

#define STOREDIM 10

#include "gpu_oei_classes.h"
#include "gpu_oei_definitions.h"
#include "gpu_oei_assembler.h"
#include "gpu_oei.h"

/*
 upload gpu simulation type to constant memory
 */
void upload_sim_to_constant_oei(_gpu_type gpu){
    cudaError_t status;
        status = cudaMemcpyToSymbol(devSim, &gpu->gpu_sim, sizeof(gpu_simulation_type));
        PRINTERROR(status, " cudaMemcpyToSymbol, sim copy to constants failed")
}

#if defined DEBUG || defined DEBUGTIME
static float totTime;
#endif

// interface for kernel launching
void getOEI(_gpu_type gpu){

//  QUICK_SAFE_CALL((getOEI_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));

    getOEI_kernel<<<1, 1>>>();

}


