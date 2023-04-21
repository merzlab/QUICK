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

#include <cuda.h>
#include "gpu_common.h"
#include "gpu_type.h"
#include "gpu_get2e_grad_ffff.h"

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
static __constant__ unsigned char devTrans[TRANSDIM*TRANSDIM*TRANSDIM];
static __constant__ int Sumindex[10]={0,0,1,4,10,20,35,56,84,120};

//#define USE_TEXTURE

#ifdef USE_TEXTURE
#define USE_TEXTURE_CUTMATRIX
#define USE_TEXTURE_YCUTOFF
#define USE_TEXTURE_XCOEFF
#endif

#ifdef USE_TEXTURE_CUTMATRIX
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_cutMatrix;
#endif
#ifdef USE_TEXTURE_YCUTOFF
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_YCutoff;
#endif
#ifdef USE_TEXTURE_XCOEFF
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_Xcoeff;
#endif

//#define USE_ERI_GRAD_STOREADD

#ifdef USE_ERI_GRAD_STOREADD
#define STORE_OPERATOR +=
#else
#define STORE_OPERATOR =  
#endif

#define ERI_GRAD_FFFF_TPB 1

#define ERI_GRAD_FFFF_SMEM_INT_SIZE 6
#define ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE 10
#define ERI_GRAD_FFFF_SMEM_DBL_SIZE 3
#define ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE 20
#define ERI_GRAD_FFFF_SMEM_CHAR_SIZE 512
#define ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE 2
#define ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE 1

#define DEV_SIM_INT_PTR_KATOM smem_int_ptr[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_INT_PTR_KPRIM smem_int_ptr[ERI_GRAD_FFFF_TPB*1]
#define DEV_SIM_INT_PTR_KSTART smem_int_ptr[ERI_GRAD_FFFF_TPB*2]
#define DEV_SIM_INT_PTR_KSUMTYPE smem_int_ptr[ERI_GRAD_FFFF_TPB*3]
#define DEV_SIM_INT_PTR_PRIM_START smem_int_ptr[ERI_GRAD_FFFF_TPB*4]
#define DEV_SIM_INT_PTR_QFBASIS smem_int_ptr[ERI_GRAD_FFFF_TPB*5]
#define DEV_SIM_INT_PTR_QSBASIS smem_int_ptr[ERI_GRAD_FFFF_TPB*6]
#define DEV_SIM_INT_PTR_QSTART smem_int_ptr[ERI_GRAD_FFFF_TPB*7]
#define DEV_SIM_INT_PTR_SORTED_Q smem_int_ptr[ERI_GRAD_FFFF_TPB*8]
#define DEV_SIM_INT_PTR_SORTED_QNUMBER smem_int_ptr[ERI_GRAD_FFFF_TPB*9]
#define DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ smem_int2_ptr[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_CHAR_PTR_MPI_BCOMPUTE smem_char_ptr[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_CHAR_PTR_KLMN smem_char_ptr[ERI_GRAD_FFFF_TPB*1]
#define DEV_SIM_DBL_PTR_CONS smem_dbl_ptr[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_DBL_PTR_CUTMATRIX smem_dbl_ptr[ERI_GRAD_FFFF_TPB*1]
#define DEV_SIM_DBL_PTR_CUTPRIM smem_dbl_ptr[ERI_GRAD_FFFF_TPB*2]
#define DEV_SIM_DBL_PTR_DENSE smem_dbl_ptr[ERI_GRAD_FFFF_TPB*3]
#define DEV_SIM_DBL_PTR_DENSEB smem_dbl_ptr[ERI_GRAD_FFFF_TPB*4]
#define DEV_SIM_DBL_PTR_EXPOSUM smem_dbl_ptr[ERI_GRAD_FFFF_TPB*5]
#define DEV_SIM_DBL_PTR_GCEXPO smem_dbl_ptr[ERI_GRAD_FFFF_TPB*6]
#define DEV_SIM_DBL_PTR_GRAD smem_dbl_ptr[ERI_GRAD_FFFF_TPB*7]
#define DEV_SIM_DBL_PTR_STORE smem_dbl_ptr[ERI_GRAD_FFFF_TPB*8]
#define DEV_SIM_DBL_PTR_STORE2 smem_dbl_ptr[ERI_GRAD_FFFF_TPB*9]
#define DEV_SIM_DBL_PTR_STOREAA smem_dbl_ptr[ERI_GRAD_FFFF_TPB*10]
#define DEV_SIM_DBL_PTR_STOREBB smem_dbl_ptr[ERI_GRAD_FFFF_TPB*11]
#define DEV_SIM_DBL_PTR_STORECC smem_dbl_ptr[ERI_GRAD_FFFF_TPB*12]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERX smem_dbl_ptr[ERI_GRAD_FFFF_TPB*13]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERY smem_dbl_ptr[ERI_GRAD_FFFF_TPB*14]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERZ smem_dbl_ptr[ERI_GRAD_FFFF_TPB*15]
#define DEV_SIM_DBL_PTR_XCOEFF smem_dbl_ptr[ERI_GRAD_FFFF_TPB*16]
#define DEV_SIM_DBL_PTR_XYZ smem_dbl_ptr[ERI_GRAD_FFFF_TPB*17]
#define DEV_SIM_DBL_PTR_YCUTOFF smem_dbl_ptr[ERI_GRAD_FFFF_TPB*18]
#define DEV_SIM_DBL_PTR_YVERTICALTEMP smem_dbl_ptr[ERI_GRAD_FFFF_TPB*19]
#define DEV_SIM_DBL_PRIMLIMIT smem_dbl[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_DBL_GRADCUTOFF smem_dbl[ERI_GRAD_FFFF_TPB*1]
#define DEV_SIM_DBL_HYB_COEFF smem_dbl[ERI_GRAD_FFFF_TPB*2]
#define DEV_SIM_INT_NATOM smem_int[ERI_GRAD_FFFF_TPB*0]
#define DEV_SIM_INT_NBASIS smem_int[ERI_GRAD_FFFF_TPB*1]
#define DEV_SIM_INT_NSHELL smem_int[ERI_GRAD_FFFF_TPB*2]
#define DEV_SIM_INT_JBASIS smem_int[ERI_GRAD_FFFF_TPB*3]
#define DEV_SIM_INT_SQRQSHELL smem_int[ERI_GRAD_FFFF_TPB*4]
#define DEV_SIM_INT_PRIM_TOTAL smem_int[ERI_GRAD_FFFF_TPB*5]
#define DEV_SIM_CHAR_TRANS smem_char

#ifdef CUDA_SPDF
//===================================


#define int_spdf4
#include "gpu_eri_grad_vrr_ffff.h"
#include "gpu_get2e_grad_ffff.cuh"

#endif

#undef int_spdf4

//Include the kernels for open shell eri calculations
#define OSHELL

#ifdef CUDA_SPDF
#define int_spdf4
//#include "gpu_get2e_grad_ffff.cuh"
#endif
#undef OSHELL


// totTime is the timer for GPU 2e time. Only on under debug mode
#if defined DEBUG || defined DEBUGTIME
static float totTime;
#endif

/*
void uploadDevSimToSmem_ffff(_gpu_type gpu ){

       cuda_buffer_type<int>* int_buffer = new cuda_buffer_type<int>(ERI_GRAD_FFFF_SMEM_INT_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<int*>* int_ptr_buffer = new cuda_buffer_type<int*>(ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<QUICKDouble>* dbl_buffer = new cuda_buffer_type<QUICKDouble>(ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<QUICKDouble*>* dbl_ptr_buffer = new cuda_buffer_type<QUICKDouble*>(ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE*ERI_GRAD_FFFF_TPB);       
       cuda_buffer_type<int2*>* int2_ptr_buffer = new cuda_buffer_type<int2*>(ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<char*>* char_ptr_buffer = new cuda_buffer_type<char*>(ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE*ERI_GRAD_FFFF_TPB);

       for(int i=0; i<ERI_GRAD_FFFF_TPB; i++){
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.natom;
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*1+i] = &gpu->gpu_sim.nbasis;
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*2+i] = &gpu->gpu_sim.nshell;
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*3+i] = &gpu->gpu_sim.jbasis;
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*4+i] = &gpu->gpu_sim.sqrQshell;
           int_buffer->_hostData[ERI_GRAD_FFFF_TPB*5+i] = &gpu->gpu_sim.prim_total;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.katom;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*1+i] = &gpu->gpu_sim.KLMN;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*2+i] = &gpu->gpu_sim.kprim;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*3+i] = &gpu->gpu_sim.kstart;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*4+i] = &gpu->gpu_sim.Ksumtype;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*5+i] = &gpu->gpu_sim.prim_start;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*6+i] = &gpu->gpu_sim.Qfbasis;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*7+i] = &gpu->gpu_sim.Qsbasis;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*8+i] = &gpu->gpu_sim.Qstart;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*9+i] = &gpu->gpu_sim.sorted_Q;
           int_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*10+i] = &gpu->gpu_sim.sorted_Qnumber;
           dbl_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.primLimit;
           dbl_buffer->_hostData[ERI_GRAD_FFFF_TPB*1+i] = &gpu->gpu_sim.gradCutoff;
           dbl_buffer->_hostData[ERI_GRAD_FFFF_TPB*2+i] = &gpu->gpu_sim.hyb_coeff;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.cons;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*1+i] = &gpu->gpu_sim.cutMatrix;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*2+i] = &gpu->gpu_sim.cutPrim;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*3+i] = &gpu->gpu_sim.dense;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*4+i] = &gpu->gpu_sim.denseb;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*5+i] = &gpu->gpu_sim.expoSum;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*6+i] = &gpu->gpu_sim.gcexpo;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*7+i] = &gpu->gpu_sim.grad;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*8+i] = &gpu->gpu_sim.gradULL;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*9+i] = &gpu->gpu_sim.store;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*10+i] = &gpu->gpu_sim.store2;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*11+i] = &gpu->gpu_sim.storeAA;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*12+i] = &gpu->gpu_sim.storeBB;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*13+i] = &gpu->gpu_sim.storeCC;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*14+i] = &gpu->gpu_sim.weightedCenterX;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*15+i] = &gpu->gpu_sim.weightedCenterY;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*16+i] = &gpu->gpu_sim.weightedCenterZ;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*17+i] = &gpu->gpu_sim.Xcoeff;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*18+i] = &gpu->gpu_sim.xyz;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*19+i] = &gpu->gpu_sim.YCutoff;
           dbl_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*20+i] = &gpu->gpu_sim.YVerticalTemp;
           int2_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.sorted_YCutoffIJ;
           char_ptr_buffer->_hostData[ERI_GRAD_FFFF_TPB*0+i] = &gpu->gpu_sim.mpi_bcompute;
       }

       int_buffer -> Upload();
       int_ptr_buffer -> Upload();
       dbl_buffer -> Upload();
       dbl_ptr_buffer -> Upload();
       int2_ptr_buffer -> Upload();
       char_ptr_buffer -> Upload();

}
*/

void getGrad_ffff(_gpu_type gpu)
{

printf("Allocating ffff memory \n");
/*
       cuda_buffer_type<int>* int_buffer = new cuda_buffer_type<int>(ERI_GRAD_FFFF_SMEM_INT_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<int*>* int_ptr_buffer = new cuda_buffer_type<int*>(ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<QUICKDouble>* dbl_buffer = new cuda_buffer_type<QUICKDouble>(ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<QUICKDouble*>* dbl_ptr_buffer = new  cuda_buffer_type<QUICKDouble*>(ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE*ERI_GRAD_FFFF_TPB);    
       cuda_buffer_type<int2*>* int2_ptr_buffer = new cuda_buffer_type<int2*>(ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE*ERI_GRAD_FFFF_TPB);
       cuda_buffer_type<unsigned char*>* char_ptr_buffer = new cuda_buffer_type<unsigned char*>(ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE*ERI_GRAD_FFFF_TPB);
*/

       int *int_buffer = (int*) malloc(ERI_GRAD_FFFF_SMEM_INT_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int));
       int **int_ptr_buffer = (int**) malloc(ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int*));
       QUICKDouble *dbl_buffer = (QUICKDouble*) malloc(ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble));
       QUICKDouble **dbl_ptr_buffer = (QUICKDouble**) malloc(ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble*));
       int2 **int2_ptr_buffer = (int2**) malloc(ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int2*));
       unsigned char **char_ptr_buffer = (unsigned char**) malloc(ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(unsigned char*));
       unsigned char trans[TRANSDIM*TRANSDIM*TRANSDIM];

printf("Storing data \n");

       for(int i=0; i<ERI_GRAD_FFFF_TPB; i++){
       int_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.natom;
       int_buffer[ERI_GRAD_FFFF_TPB*1+i] = gpu->gpu_sim.nbasis;
       int_buffer[ERI_GRAD_FFFF_TPB*2+i] = gpu->gpu_sim.nshell;
       int_buffer[ERI_GRAD_FFFF_TPB*3+i] = gpu->gpu_sim.jbasis;
       int_buffer[ERI_GRAD_FFFF_TPB*4+i] = gpu->gpu_sim.sqrQshell;
       int_buffer[ERI_GRAD_FFFF_TPB*5+i] = gpu->gpu_sim.prim_total;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.katom;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*1+i] = gpu->gpu_sim.kprim;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*2+i] = gpu->gpu_sim.kstart;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*3+i] = gpu->gpu_sim.Ksumtype;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*4+i] = gpu->gpu_sim.prim_start;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*5+i] = gpu->gpu_sim.Qfbasis;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*6+i] = gpu->gpu_sim.Qsbasis;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*7+i] = gpu->gpu_sim.Qstart;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*8+i] = gpu->gpu_sim.sorted_Q;
       int_ptr_buffer[ERI_GRAD_FFFF_TPB*9+i] = gpu->gpu_sim.sorted_Qnumber;
       dbl_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.primLimit;
       dbl_buffer[ERI_GRAD_FFFF_TPB*1+i] = gpu->gpu_sim.gradCutoff;
       dbl_buffer[ERI_GRAD_FFFF_TPB*2+i] = gpu->gpu_sim.hyb_coeff;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.cons;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*1+i] = gpu->gpu_sim.cutMatrix;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*2+i] = gpu->gpu_sim.cutPrim;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*3+i] = gpu->gpu_sim.dense;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*4+i] = gpu->gpu_sim.denseb;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*5+i] = gpu->gpu_sim.expoSum;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*6+i] = gpu->gpu_sim.gcexpo;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*7+i] = gpu->gpu_sim.grad;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*8+i] = gpu->gpu_sim.store;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*9+i] = gpu->gpu_sim.store2;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*10+i] = gpu->gpu_sim.storeAA;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*11+i] = gpu->gpu_sim.storeBB;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*12+i] = gpu->gpu_sim.storeCC;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*13+i] = gpu->gpu_sim.weightedCenterX;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*14+i] = gpu->gpu_sim.weightedCenterY;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*15+i] = gpu->gpu_sim.weightedCenterZ;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*16+i] = gpu->gpu_sim.Xcoeff;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*17+i] = gpu->gpu_sim.xyz;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*18+i] = gpu->gpu_sim.YCutoff;
       dbl_ptr_buffer[ERI_GRAD_FFFF_TPB*19+i] = gpu->gpu_sim.YVerticalTemp;
       int2_ptr_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.sorted_YCutoffIJ;
       char_ptr_buffer[ERI_GRAD_FFFF_TPB*0+i] = gpu->gpu_sim.mpi_bcompute;
       char_ptr_buffer[ERI_GRAD_FFFF_TPB*1+i] = gpu->gpu_sim.KLMN;
       }


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

printf("Allocating device memory \n");

       int *dev_int_buffer;
       int **dev_int_ptr_buffer;
       QUICKDouble *dev_dbl_buffer;
       QUICKDouble **dev_dbl_ptr_buffer;
       int2 **dev_int2_ptr_buffer;
       unsigned char **dev_char_ptr_buffer;
       unsigned char *dev_char_buffer;

       cudaMalloc((void **)&dev_int_buffer, ERI_GRAD_FFFF_SMEM_INT_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int));
printf("Allocating int ptr device memory %d %d %d %d %d %d\n", sizeof(int), sizeof(int*), sizeof(QUICKDouble), sizeof(QUICKDouble*),
sizeof(int2*), sizeof(unsigned char*));
       cudaMalloc((void **)&dev_int_ptr_buffer, ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int*));
printf("Allocating dbl device memory \n");
       cudaMalloc((void **)&dev_dbl_buffer, ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble));
       cudaMalloc((void **)&dev_dbl_ptr_buffer, ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble*));
       cudaMalloc((void **)&dev_int2_ptr_buffer, ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int2*));
       cudaMalloc((void **)&dev_char_ptr_buffer, ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(unsigned char*));
       cudaMalloc((void **)&dev_char_buffer, ERI_GRAD_FFFF_SMEM_CHAR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(unsigned char));

printf("Uploading data \n");

       cudaMemcpy(dev_int_buffer, int_buffer, ERI_GRAD_FFFF_SMEM_INT_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_int_ptr_buffer, int_ptr_buffer, ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int*), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_dbl_buffer, dbl_buffer, ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_dbl_ptr_buffer, dbl_ptr_buffer, ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(QUICKDouble*), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_int2_ptr_buffer, int2_ptr_buffer, ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(int2*), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_char_ptr_buffer, char_ptr_buffer, ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(unsigned
char*), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_char_buffer, &trans, ERI_GRAD_FFFF_SMEM_CHAR_SIZE*ERI_GRAD_FFFF_TPB*sizeof(unsigned char), cudaMemcpyHostToDevice);

/*
       int_buffer -> Upload();
       int_ptr_buffer -> Upload();
       dbl_buffer -> Upload();
       dbl_ptr_buffer -> Upload();
       int2_ptr_buffer -> Upload();
       char_ptr_buffer -> Upload();
*/
printf("Launching ffff \n");

//   nvtxRangePushA("Gradient 2e");
    
        if (gpu->maxL >= 3) {
        // Part f-3
#ifdef CUDA_SPDF

            //printf("calling getGrad_kernel_spdf4 \n");
            QUICK_SAFE_CALL((getGrad_kernel_ffff<<<gpu->blocks, gpu->twoEThreadsPerBlock, sizeof(int)*ERI_GRAD_FFFF_SMEM_INT_SIZE+
            sizeof(QUICKDouble)*ERI_GRAD_FFFF_SMEM_DBL_SIZE+sizeof(QUICKDouble*)*ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE+sizeof(int*)*ERI_GRAD_FFFF_SMEM_INT_PTR_SIZE+
            sizeof(int2*)*ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE+sizeof(unsigned char*)*ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE+sizeof(unsigned char)*ERI_GRAD_FFFF_SMEM_CHAR_SIZE>>>(dev_int_buffer,
dev_int_ptr_buffer, dev_dbl_buffer, dev_dbl_ptr_buffer, dev_int2_ptr_buffer, dev_char_ptr_buffer, dev_char_buffer)))

#endif  
        }

    cudaDeviceSynchronize();

//    nvtxRangePop();

printf("Deleting data \n");

   free(int_buffer);
   free(int_ptr_buffer);
   free(dbl_buffer);
   free(dbl_ptr_buffer);
   free(int2_ptr_buffer);
   free(char_ptr_buffer);

   cudaFree(dev_int_buffer);
   cudaFree(dev_int_ptr_buffer);
   cudaFree(dev_dbl_buffer);
   cudaFree(dev_dbl_ptr_buffer);
   cudaFree(dev_int2_ptr_buffer);
   cudaFree(dev_char_ptr_buffer);
   cudaFree(dev_char_buffer);
/*
    SAFE_DELETE(int_buffer);
    SAFE_DELETE(int_ptr_buffer);
    SAFE_DELETE(dbl_buffer);
    SAFE_DELETE(dbl_ptr_buffer);
    SAFE_DELETE(int2_ptr_buffer);
    SAFE_DELETE(char_ptr_buffer);
*/

}


// interface to call uscf gradient Kernels
void get_oshell_eri_grad_ffff(_gpu_type gpu)
{

//   nvtxRangePushA("Gradient 2e");


    // compute one electron gradients in the meantime
    //get_oneen_grad_();

    if (gpu->maxL >= 3) {
        // Part f-3
        //    QUICK_SAFE_CALL((getGrad_oshell_kernel_ffff<<<gpu->blocks, gpu->gradThreadsPerBlock>>>()))
        //#endif
    }

    cudaDeviceSynchronize();
//    nvtxRangePop();

}

void upload_para_to_const_ffff(){
    
    unsigned char trans[TRANSDIM*TRANSDIM*TRANSDIM];
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

    status = cudaMemcpyToSymbol(devTrans, trans, sizeof(unsigned char)*TRANSDIM*TRANSDIM*TRANSDIM);
    PRINTERROR(status, " cudaMemcpyToSymbol, Trans copy to constants failed")

}

void upload_sim_to_constant_ffff(_gpu_type gpu){
    cudaError_t status;
        status = cudaMemcpyToSymbol(devSim, &gpu->gpu_sim, sizeof(gpu_simulation_type));
        PRINTERROR(status, " cudaMemcpyToSymbol, sim copy to constants failed")

    upload_para_to_const_ffff();
}
