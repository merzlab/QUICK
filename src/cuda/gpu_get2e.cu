/*
 *  gpu_get2e.cpp
 *  new_quick
 *
 *  Created by Yipu Miao on 6/17/11.
 *  Copyright 2011 University of Florida.All rights reserved.
 *  
 *  Yipu Miao 9/15/11:  the first draft is released. And the GPUGP QM compuation can 
 *                      achieve as much as 15x faster at double precision level compared with CPU.
 */

#include "gpu.h"
#include <cuda.h>

//#ifdef CUDA_SPDF
//#endif


/*
 Constant Memory in GPU is fast but quite limited and hard to operate, usually not allocatable and 
 readonly. So we put the following variables into constant memory:
 devSim: a gpu simluation type variable. which is to store to location of basic information about molecule and basis
 set. Note it only store the location, so it's mostly a set of pointer to GPU memory. and with some non-pointer
 value like the number of basis set. See gpu_type.h for details.
 devTrans : arrays to save the mapping index, will be elimited by hand writing unrolling code.
 Sumindex: a array to store refect how many temp variable needed in VRR. can be elimited by hand writing code.
 */
static __constant__ gpu_simulation_type devSim;
static __constant__ int devTrans[TRANSDIM*TRANSDIM*TRANSDIM];
static __constant__ int Sumindex[10]={0,0,1,4,10,20,35,56,84,120};


#include "gpu_get2e_subs_hrr.h"
#include "int.h"


#define int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"
#include "gpu_get2e_subs_grad.h"


//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs_grad.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs_grad.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs_grad.h"


#ifdef CUDA_SPDF
//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#define int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#define int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"



#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#define int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"



#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#define int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"



#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#define int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#define int_spdf9
#undef int_spdf10
#include "gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#define int_spdf10
#include "gpu_get2e_subs.h"
#endif

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10



/*
 upload gpu simulation type to constant memory
 */
void upload_sim_to_constant(_gpu_type gpu){
    cudaError_t status;
	status = cudaMemcpyToSymbol(devSim, &gpu->gpu_sim, sizeof(gpu_simulation_type));
	PRINTERROR(status, " cudaMemcpyToSymbol, sim copy to constants failed")
}


// totTime is the timer for GPU 2e time. Only on under debug mode
#ifdef DEBUG
static float totTime;
#endif

// =======   INTERFACE SECTION ===========================
// interface to call Kernel subroutine
void getAOInt(_gpu_type gpu, QUICKULL intStart, QUICKULL intEnd, cudaStream_t streamI, int streamID,  ERI_entry* aoint_buffer)
{
    QUICK_SAFE_CALL((getAOInt_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>(intStart, intEnd, aoint_buffer, streamID)));
#ifdef CUDA_SPDF
    // Part f-1
    QUICK_SAFE_CALL((getAOInt_kernel_spdf<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-2
    QUICK_SAFE_CALL((getAOInt_kernel_spdf2<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-3
    QUICK_SAFE_CALL((getAOInt_kernel_spdf3<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-4
    QUICK_SAFE_CALL((getAOInt_kernel_spdf4<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-5
    QUICK_SAFE_CALL((getAOInt_kernel_spdf5<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-6
    QUICK_SAFE_CALL((getAOInt_kernel_spdf6<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-7
    QUICK_SAFE_CALL((getAOInt_kernel_spdf7<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-8
    QUICK_SAFE_CALL((getAOInt_kernel_spdf8<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-9
    QUICK_SAFE_CALL((getAOInt_kernel_spdf9<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
    // Part f-10
    QUICK_SAFE_CALL((getAOInt_kernel_spdf10<<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>>( intStart, intEnd, aoint_buffer, streamID)));
#endif
}

// interface to call Kernel subroutine
void get2e(_gpu_type gpu)
{
    // Part spd
    QUICK_SAFE_CALL((get2e_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
#ifdef CUDA_SPDF
    if (gpu->maxL >= 3) {
        // Part f-1
        QUICK_SAFE_CALL((get2e_kernel_spdf<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-2
        QUICK_SAFE_CALL((get2e_kernel_spdf2<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-3
        QUICK_SAFE_CALL((get2e_kernel_spdf3<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-4
        QUICK_SAFE_CALL((get2e_kernel_spdf4<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-5
        QUICK_SAFE_CALL((get2e_kernel_spdf5<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-6
        QUICK_SAFE_CALL((get2e_kernel_spdf6<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-7
        QUICK_SAFE_CALL((get2e_kernel_spdf7<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-8
        QUICK_SAFE_CALL((get2e_kernel_spdf8<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-9
        QUICK_SAFE_CALL((get2e_kernel_spdf9<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
        // Part f-10
        QUICK_SAFE_CALL((get2e_kernel_spdf10<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>()));
    }
#endif 
}


// interface to call Kernel subroutine
void getAddInt(_gpu_type gpu, int bufferSize, ERI_entry* aoint_buffer)
{
printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim_dft.method \n", __FILE__, __LINE__, __func__);
    QUICK_SAFE_CALL((getAddInt_kernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(bufferSize, aoint_buffer)));
}

// interface to call Kernel subroutine
void getGrad(_gpu_type gpu)
{
   QUICK_SAFE_CALL((getGrad_kernel<<<gpu->blocks, gpu->gradThreadsPerBlock>>>()));
    if (gpu->maxL >= 2) {
        //#ifdef CUDA_SPDF
        // Part f-1
        QUICK_SAFE_CALL((getGrad_kernel_spdf<<<gpu->blocks, gpu->gradThreadsPerBlock>>>()));
        // Part f-2
        QUICK_SAFE_CALL((getGrad_kernel_spdf2<<<gpu->blocks, gpu->gradThreadsPerBlock>>>()));
        // Part f-3
        //    QUICK_SAFE_CALL((getGrad_kernel_spdf3<<<gpu->blocks, gpu->gradThreadsPerBlock>>>()))
        //#endif
    }
}




// =======   KERNEL SECTION ===========================
__global__ void getAddInt_kernel(int bufferSize, ERI_entry* aoint_buffer){
    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    int const batchSize = 20;
    ERI_entry a[batchSize];
    int j = 0;
 
    QUICKULL myInt = (QUICKULL) (bufferSize) / totalThreads;
    if ((bufferSize - myInt*totalThreads)> offside) myInt++;
    
    for (QUICKULL i = 1; i<=myInt; i++) {
        
        QUICKULL currentInt = totalThreads * (i-1) + offside;
        a[j] = aoint_buffer[currentInt];
        j++;
        if (j == batchSize || i == myInt) {
            
            for (int k = 0; k<j; k++) {
                int III = a[k].IJ / devSim.nbasis + 1;
                int JJJ = a[k].IJ % devSim.nbasis + 1;
                int KKK = a[k].KL / devSim.nbasis + 1;
                int LLL = a[k].KL % devSim.nbasis + 1;
                
                if (III <= devSim.nbasis && III >= 1 && JJJ <= devSim.nbasis && JJJ >= 1 && KKK <= devSim.nbasis && KKK >= 1 && LLL <= devSim.nbasis && LLL >= 1){
                    /*QUICKDouble hybrid_coeff = 0.0;
                    if (devSim.method == HF){
                        hybrid_coeff = 1.0;
                    }else if (devSim.method == B3LYP){
                        hybrid_coeff = 0.2;
                    }else if (devSim.method == DFT){
                        hybrid_coeff = 0.0;
                    }else if(devSim.method == LIBXC){
			hybrid_coeff = devSim.hyb_coeff;			
		    }
                    */

                    addint(devSim.oULL, a[k].value, III, JJJ, KKK, LLL, devSim.hyb_coeff, devSim.dense, devSim.nbasis);
                }
            }
            j = 0;
        }
        
    }
    
}


void upload_para_to_const(){
    
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

