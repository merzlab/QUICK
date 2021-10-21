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
static __constant__ int devTrans[TRANSDIM*TRANSDIM*TRANSDIM];
static __constant__ int Sumindex[10]={0,0,1,4,10,20,35,56,84,120};

#define STOREDIM 20

#include "gpu_oei_classes.h"
#include "gpu_oei_definitions.h"
#include "gpu_oei_assembler.h"
#include "gpu_oei.h"
#include "gpu_oei_grad_assembler.h"
#include "gpu_oei_grad.h"

/*
  upload trans array to constant memory
*/
void upload_para_to_const_oei(){

    int trans[TRANSDIM*TRANSDIM*TRANSDIM];
    // Data to trans
    {
        LOC3(trans, 0, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   1;
        LOC3(trans, 0, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   4;
        LOC3(trans, 0, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  10;
        LOC3(trans, 0, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  20;
        LOC3(trans, 0, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  35;
        LOC3(trans, 0, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  56;
        LOC3(trans, 0, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) =  84;
        LOC3(trans, 0, 0, 7, TRANSDIM, TRANSDIM, TRANSDIM) = 120;
        LOC3(trans, 0, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   3;
        LOC3(trans, 0, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   6;
        LOC3(trans, 0, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  17;
        LOC3(trans, 0, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  32;
        LOC3(trans, 0, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  48;
        LOC3(trans, 0, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  67;
        LOC3(trans, 0, 1, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 100;
        LOC3(trans, 0, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   9;
        LOC3(trans, 0, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  16;
        LOC3(trans, 0, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  23;
        LOC3(trans, 0, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  42;
        LOC3(trans, 0, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  73;
        LOC3(trans, 0, 2, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 106;
        LOC3(trans, 0, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  19;
        LOC3(trans, 0, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  31;
        LOC3(trans, 0, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  43;
        LOC3(trans, 0, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  79;
        LOC3(trans, 0, 3, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 112;
        LOC3(trans, 0, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  34;
        LOC3(trans, 0, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  49;
        LOC3(trans, 0, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  74;
        LOC3(trans, 0, 4, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 113;
        LOC3(trans, 0, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  55;
        LOC3(trans, 0, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  68;
        LOC3(trans, 0, 5, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 107;
        LOC3(trans, 0, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  83;
        LOC3(trans, 0, 6, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 101;
        LOC3(trans, 0, 7, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 119;
        LOC3(trans, 1, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   2;
        LOC3(trans, 1, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   7;
        LOC3(trans, 1, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  15;
        LOC3(trans, 1, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  28;
        LOC3(trans, 1, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  50;
        LOC3(trans, 1, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  69;
        LOC3(trans, 1, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 102;
        LOC3(trans, 1, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   5;
        LOC3(trans, 1, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  11;
        LOC3(trans, 1, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  26;
        LOC3(trans, 1, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  41;
        LOC3(trans, 1, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  59;
        LOC3(trans, 1, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  87;
        LOC3(trans, 1, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  13;
        LOC3(trans, 1, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  25;
        LOC3(trans, 1, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  36;
        LOC3(trans, 1, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  60;
        LOC3(trans, 1, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  88;
        LOC3(trans, 1, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  30;
        LOC3(trans, 1, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  40;
        LOC3(trans, 1, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  61;
        LOC3(trans, 1, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  94;
        LOC3(trans, 1, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  52;
        LOC3(trans, 1, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  58;
        LOC3(trans, 1, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  89;
        LOC3(trans, 1, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  71;
        LOC3(trans, 1, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  86;
        LOC3(trans, 1, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 104;
        LOC3(trans, 2, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   8;
        LOC3(trans, 2, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  14;
        LOC3(trans, 2, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  22;
        LOC3(trans, 2, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  44;
        LOC3(trans, 2, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  75;
        LOC3(trans, 2, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 108;
        LOC3(trans, 2, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  12;
        LOC3(trans, 2, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  24;
        LOC3(trans, 2, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  37;
        LOC3(trans, 2, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  62;
        LOC3(trans, 2, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  90;
        LOC3(trans, 2, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  21;
        LOC3(trans, 2, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  38;
        LOC3(trans, 2, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  66;
        LOC3(trans, 2, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  99;
        LOC3(trans, 2, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  46;
        LOC3(trans, 2, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  64;
        LOC3(trans, 2, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  98;
        LOC3(trans, 2, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  77;
        LOC3(trans, 2, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  92;
        LOC3(trans, 2, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 110;
        LOC3(trans, 3, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  18;
        LOC3(trans, 3, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  27;
        LOC3(trans, 3, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  45;
        LOC3(trans, 3, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  80;
        LOC3(trans, 3, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 114;
        LOC3(trans, 3, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  29;
        LOC3(trans, 3, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  39;
        LOC3(trans, 3, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  63;
        LOC3(trans, 3, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  95;
        LOC3(trans, 3, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  47;
        LOC3(trans, 3, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  65;
        LOC3(trans, 3, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  97;
        LOC3(trans, 3, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  81;
        LOC3(trans, 3, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  96;
        LOC3(trans, 3, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 116;
        LOC3(trans, 4, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  33;
        LOC3(trans, 4, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  51;
        LOC3(trans, 4, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  76;
        LOC3(trans, 4, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 115;
        LOC3(trans, 4, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  53;
        LOC3(trans, 4, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  57;
        LOC3(trans, 4, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  91;
        LOC3(trans, 4, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  78;
        LOC3(trans, 4, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  93;
        LOC3(trans, 4, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 117;
        LOC3(trans, 5, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  54;
        LOC3(trans, 5, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  70;
        LOC3(trans, 5, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
        LOC3(trans, 5, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  72;
        LOC3(trans, 5, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  85;
        LOC3(trans, 5, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 111;
        LOC3(trans, 6, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  82;
        LOC3(trans, 6, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 103;
        LOC3(trans, 6, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 105;
        LOC3(trans, 7, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 118;
    }
    // upload to trans device location
    cudaError_t status;

    status = cudaMemcpyToSymbol(devTrans, trans, sizeof(int)*TRANSDIM*TRANSDIM*TRANSDIM);
    PRINTERROR(status, " cudaMemcpyToSymbol, Trans copy to constants failed")

}


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

    QUICK_SAFE_CALL((getOEI_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));

}

void get_oei_grad(_gpu_type gpu){

    QUICK_SAFE_CALL((get_oei_grad_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));

}

