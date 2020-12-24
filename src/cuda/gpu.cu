/*
 *  gpu_startup.cu
 *  new_quick
 *
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#include <stdio.h>
#include <string>
#include "gpu.h"
#include <ctime>
#include <time.h>

#ifdef CUDA_MPIV
#include "mgpu.h"
#endif
//-----------------------------------------------
// Set up specified device and be ready to ignite
//-----------------------------------------------
extern "C" void gpu_set_device_(int* gpu_dev_id)
{
    gpu->gpu_dev_id = *gpu_dev_id;
#ifdef DEBUG
    fprintf(gpu->debugFile,"using gpu: %i\n", *gpu_dev_id);
#endif
}

//-----------------------------------------------
// create gpu class
//-----------------------------------------------
extern "C" void gpu_startup_(void)
{

#if defined DEBUG || defined DEBUGTIME
        debugFile = fopen("debug.cuda", "w+");
#endif
	PRINTDEBUGNS("BEGIN TO WARM UP")

        gpu = new gpu_type;

#if defined DEBUG || defined DEBUGTIME
        gpu->debugFile = debugFile;
#endif
	PRINTDEBUG("CREATE NEW GPU")
	
}


//-----------------------------------------------
// Initialize the device
//-----------------------------------------------
extern "C" void gpu_init_(void)
{
    
    PRINTDEBUG("BEGIN TO INIT")
    
    int device = -1;
    int gpuCount = 0;
    cudaError_t status;
    cudaDeviceProp deviceProp;
    status = cudaGetDeviceCount(&gpuCount);

#ifdef DEBUG
    fprintf(gpu->debugFile,"Number of gpus %i \n", gpuCount);
#endif

    PRINTERROR(status,"cudaGetDeviceCount gpu_init failed!");
    if (gpuCount == 0)
    {
        printf("NO CUDA-Enabled GPU FOUND.\n");
        cudaDeviceReset();
        exit(-1);
    }
    
    if (gpu->gpu_dev_id == -1){
        device = 0;
        // if gpu count is greater than 1(multi-gpu) select one with bigger free memory, or available.
        if (gpuCount > 1) {
            size_t maxMem = 0;
            for (int i = gpuCount-1; i>=0; i--) {
                /*status = cudaSetDevice(i);
                 size_t free_mem = 0;
                 size_t tot_mem  = 0;
                 
                 status = cudaMemGetInfo(&free_mem, &tot_mem); // If error returns, that is to say this device is unavailable.
                 // Else, use one with larger memory.
                 if (free_mem >= maxMem) {
                 maxMem = free_mem;
                 device = i;
                 }
                 cudaThreadExit();*/
                cudaGetDeviceProperties(&deviceProp, i);
                
                if (((deviceProp.major >= 2) || ((deviceProp.major == 1) && (deviceProp.minor == 3))) &&
                    (deviceProp.totalGlobalMem >= maxMem))
                {
                    maxMem                          = deviceProp.totalGlobalMem;
                    device                          = i;
                }
                
            }
        }
        gpu->gpu_dev_id = device;
        
    }else{
        if (gpu->gpu_dev_id >= gpuCount)
        {
            printf("GPU ID IS ILLEGAL, PLEASE SELECT FROM 0 TO %i.\n", gpuCount-1);
            cudaDeviceReset();
            exit(-1);
        }
        
    	cudaGetDeviceProperties(&deviceProp, gpu->gpu_dev_id);
    	if ( (deviceProp.major >=2) || ((deviceProp.major == 1) && (deviceProp.minor == 3)))
        	device = gpu->gpu_dev_id;
    	else {
        	printf("SELECT GPU HAS CUDA SUPPORTING VERSION UNDER 1.3. EXITING. \n");
        	cudaDeviceReset();
        	exit(-1);
    	}
        device = gpu->gpu_dev_id;
    }
    
#ifdef DEBUG
    fprintf(gpu->debugFile,"using gpu: %i\n", device);
#endif
    
    if (device == -1) {
        printf("NO CUDA 1.3 (OR ABOVE) SUPPORTED GPU IS FOUND\n");
        gpu_shutdown_();
        exit(-1);
    }
    
    status = cudaSetDevice(device);
    cudaGetDeviceProperties(&deviceProp, device);
    PRINTERROR(status, "cudaSetDevice gpu_init failed!");
    cudaDeviceSynchronize();
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    size_t val;
    
    cudaDeviceGetLimit(&val, cudaLimitStackSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"Stack size limit:    %zu\n", val);
#endif    

    cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"Printf fifo limit:   %zu\n", val);
#endif
    
    cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"Heap size limit:     %zu\n", val);
#endif
    
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    
    cudaDeviceGetLimit(&val, cudaLimitStackSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"New Stack size limit:    %zu\n", val);
#endif
    
	gpu->blocks = deviceProp.multiProcessorCount;
    if (deviceProp.major ==1) {
        switch (deviceProp.minor) {
            case 0:
            case 1:
            case 2:
            case 5:
                printf("GPU SM VERSION SHOULD BE HIGHER THAN 1.3\n");
                gpu_shutdown_();
                exit(-1);
                break;
            default:
                gpu -> sm_version           =   SM_13;
                gpu -> threadsPerBlock      =   SM_13_THREADS_PER_BLOCK;
                gpu -> twoEThreadsPerBlock  =   SM_13_2E_THREADS_PER_BLOCK;
                gpu -> XCThreadsPerBlock    =   SM_13_XC_THREADS_PER_BLOCK;
                gpu -> gradThreadsPerBlock  =   SM_13_GRAD_THREADS_PER_BLOCK;
                break;
        }
    }else {
        gpu -> sm_version               = SM_2X;
        gpu -> threadsPerBlock          = SM_2X_THREADS_PER_BLOCK;
        gpu -> twoEThreadsPerBlock      = SM_2X_2E_THREADS_PER_BLOCK;
        gpu -> XCThreadsPerBlock        = SM_2X_XC_THREADS_PER_BLOCK;
        gpu -> gradThreadsPerBlock      = SM_2X_GRAD_THREADS_PER_BLOCK;
    }
    
    PRINTDEBUG("FINISH INIT")
    
    return;
}

extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id,int* gpu_dev_mem,
                                     int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv)
{
    cudaError_t cuda_error;
    cudaDeviceProp prop;
    size_t device_mem;
    
    *gpu_dev_id = gpu->gpu_dev_id;  // currently one GPU is supported
    cuda_error = cudaGetDeviceCount(gpu_dev_count);
    PRINTERROR(cuda_error,"cudaGetDeviceCount gpu_get_device_info failed!");
    if (*gpu_dev_count == 0)
    {
        printf("NO CUDA DEVICE FOUNDED \n");
        cudaDeviceReset();
        exit(-1);
    }
    cudaGetDeviceProperties(&prop,*gpu_dev_id);
    device_mem = (prop.totalGlobalMem/(1024*1024));
    *gpu_dev_mem = (int) device_mem;
    *gpu_num_proc = (int) (prop.multiProcessorCount);
    *gpu_core_freq = (double) (prop.clockRate * 1e-6f);
    strcpy(gpu_dev_name,prop.name);
    *name_len = strlen(gpu_dev_name);
    *majorv = prop.major;
    *minorv = prop.minor;
    
}

//-----------------------------------------------
// shutdonw gpu and terminate gpu calculation part
//-----------------------------------------------
extern "C" void gpu_shutdown_(void)
{
    PRINTDEBUG("BEGIN TO SHUTDOWN")

    delete gpu;
    cudaDeviceReset();

    PRINTDEBUGNS("SHUTDOWN NORMALLY")

#if defined DEBUG || defined DEBUGTIME
    fclose(debugFile);
#endif

    return;
}
;
//-----------------------------------------------
//  Setup up basic infomation of the system
//-----------------------------------------------
extern "C" void gpu_setup_(int* natom, int* nbasis, int* nElec, int* imult, int* molchg, int* iAtomType)
{
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    PRINTDEBUG("BEGIN TO SETUP")

#ifdef CUDA_MPIV
    fprintf(gpu->debugFile,"mpirank %i natoms %i \n", gpu -> mpirank, *natom );    
#endif
#endif

    gpu -> natom                    =   *natom;
    gpu -> nbasis                   =   *nbasis;
    gpu -> nElec                    =   *nElec;
    gpu -> imult                    =   *imult;
    gpu -> molchg                   =   *molchg;
    gpu -> iAtomType                =   *iAtomType;
    gpu -> gpu_calculated           =   new gpu_calculated_type;
    gpu -> gpu_basis                =   new gpu_basis_type;
    gpu -> gpu_cutoff               =   new gpu_cutoff_type;
    gpu -> gpu_xcq                  =   new XC_quadrature_type;
    gpu -> gpu_calculated -> natom  =   *natom;
    gpu -> gpu_basis -> natom       =   *natom;
    gpu -> gpu_calculated -> nbasis =   *nbasis;
    gpu -> gpu_basis -> nbasis      =   *nbasis;
    
    gpu -> gpu_sim.natom            =   *natom;
    gpu -> gpu_sim.nbasis           =   *nbasis;
    gpu -> gpu_sim.nElec            =   *nElec;
    gpu -> gpu_sim.imult            =   *imult;
    gpu -> gpu_sim.molchg           =   *molchg;
    gpu -> gpu_sim.iAtomType        =   *iAtomType;
    
    upload_para_to_const();

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
#ifdef CUDA
    PRINTUSINGTIME("UPLOAD PARA TO CONST",time);
#endif
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    PRINTDEBUG("FINISH SETUP")
#endif
    
}

//Madu Manathunga: 08/31/2019
//-----------------------------------------------
//  upload method and hybrid coefficient
//-----------------------------------------------
extern "C" void gpu_upload_method_(int* quick_method, double* hyb_coeff)
{
    if (*quick_method == 0) {
        gpu -> gpu_sim.method = HF;
	gpu -> gpu_sim.hyb_coeff = 1.0;
    }else if (*quick_method == 1) {
        gpu -> gpu_sim.method = B3LYP;
	gpu -> gpu_sim.hyb_coeff = 0.2;
    }else if (*quick_method == 2) {
        gpu -> gpu_sim.method = DFT;
	gpu -> gpu_sim.hyb_coeff = 0.0;
    }else if (*quick_method == 3) {
	gpu -> gpu_sim.method = LIBXC;
	gpu -> gpu_sim.hyb_coeff = *hyb_coeff;
    }
}

//-----------------------------------------------
//  upload coordinates
//-----------------------------------------------
extern "C" void gpu_upload_xyz_(QUICKDouble* atom_xyz)
{
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO UPLOAD COORDINATES")
    //    gpu -> gpu_basis -> xyz = new cuda_buffer_type<QUICKDouble>(atom_xyz, 3, gpu->natom);
    //	gpu -> gpu_basis -> xyz ->Upload();
    gpu -> gpu_calculated -> distance = new cuda_buffer_type<QUICKDouble>(gpu->natom, gpu->natom);
    
    gpu -> xyz = new cuda_buffer_type<QUICKDouble>(atom_xyz, 3, gpu->natom);
    
    for (int i = 0; i < gpu->natom; i++) {
        for (int j = 0; j < gpu->natom; j++) {
            QUICKDouble distance = 0;
            for (int k = 0; k<3; k++) {
                distance += pow(LOC2(gpu->xyz->_hostData, k, i, 3, gpu->natom)
                                -LOC2(gpu->xyz->_hostData, k, j, 3, gpu->natom),2);
            }
            LOC2(gpu->gpu_calculated->distance->_hostData, i, j, gpu->natom, gpu->natom) = sqrt(distance);
        }
    }
    
    gpu -> xyz -> Upload();
    gpu -> gpu_calculated -> distance -> Upload();
    
    gpu -> gpu_sim.xyz =  gpu -> xyz -> _devData;
    gpu -> gpu_sim.distance = gpu -> gpu_calculated -> distance -> _devData;
    
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD XYZ",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE UPLOADING COORDINATES")
     
}


//-----------------------------------------------
//  upload molecule infomation
//-----------------------------------------------
extern "C" void gpu_upload_atom_and_chg_(int* atom, QUICKDouble* atom_chg)
{
    
    PRINTDEBUG("BEGIN TO UPLOAD ATOM AND CHARGE")
    
    gpu -> iattype = new cuda_buffer_type<int>(atom, gpu->natom);
    gpu -> chg     = new cuda_buffer_type<QUICKDouble>(atom_chg, gpu->natom);
    gpu -> iattype -> Upload();
    gpu -> chg     -> Upload();
    
    
    gpu -> gpu_sim.chg              = gpu -> chg -> _devData;
    gpu -> gpu_sim.iattype          = gpu -> iattype -> _devData;
    
    PRINTDEBUG("COMPLETE UPLOADING ATOM AND CHARGE")
    
}


//-----------------------------------------------
//  upload cutoff criteria, will update every
//  interation
//-----------------------------------------------
extern "C" void gpu_upload_cutoff_(QUICKDouble* cutMatrix, QUICKDouble* integralCutoff,QUICKDouble* primLimit, QUICKDouble* DMCutoff)
{
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO UPLOAD CUTOFF")
    
    gpu -> gpu_cutoff -> integralCutoff = *integralCutoff;
    gpu -> gpu_cutoff -> primLimit      = *primLimit;
    gpu -> gpu_cutoff -> DMCutoff       = 1E-9; //*DMCutoff;
    
    gpu -> gpu_cutoff -> cutMatrix  = new cuda_buffer_type<QUICKDouble>(cutMatrix, gpu->nshell, gpu->nshell);
    
    gpu -> gpu_cutoff -> cutMatrix  -> Upload();
    
    gpu -> gpu_cutoff -> cutMatrix  -> DeleteCPU();
    
    gpu -> gpu_sim.cutMatrix        = gpu -> gpu_cutoff -> cutMatrix -> _devData;
    gpu -> gpu_sim.integralCutoff   = gpu -> gpu_cutoff -> integralCutoff;
    gpu -> gpu_sim.primLimit        = gpu -> gpu_cutoff -> primLimit;
    gpu -> gpu_sim.DMCutoff         = gpu -> gpu_cutoff -> DMCutoff;

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD CUTOFF",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE UPLOADING CUTOFF")
}


//-----------------------------------------------
//  upload cutoff matrix, only update at first
//  interation
//-----------------------------------------------
extern "C" void gpu_upload_cutoff_matrix_(QUICKDouble* YCutoff,QUICKDouble* cutPrim)
{

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO UPLOAD CUTOFF")
    
    gpu -> gpu_cutoff -> natom      = gpu -> natom;
    gpu -> gpu_cutoff -> YCutoff    = new cuda_buffer_type<QUICKDouble>(YCutoff, gpu->nshell, gpu->nshell);
    gpu -> gpu_cutoff -> cutPrim    = new cuda_buffer_type<QUICKDouble>(cutPrim, gpu->jbasis, gpu->jbasis);
    
    gpu -> gpu_cutoff -> YCutoff    -> Upload();
    gpu -> gpu_cutoff -> cutPrim    -> Upload();
    
    gpu -> gpu_cutoff -> sqrQshell  = (gpu -> gpu_basis -> Qshell) * (gpu -> gpu_basis -> Qshell);
    gpu -> gpu_cutoff -> sorted_YCutoffIJ           = new cuda_buffer_type<int2>(gpu->gpu_cutoff->sqrQshell);
    

    int sort_method = 0;
    int a = 0;
    bool flag = true;
    int2 temp;
    int maxL = 0;
    
    for ( int i = 0; i < gpu->gpu_basis->Qshell; i++) {
        if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] > maxL) {
            maxL = gpu->gpu_basis->sorted_Qnumber->_hostData[i];
        }
    }
   
#ifdef DEBUG  
    fprintf(gpu->debugFile,"MAX ANGULAR MOMENT = %i\n", maxL);
#endif    

    gpu -> maxL = maxL;
    gpu -> gpu_sim.maxL = maxL;
    
    gpu -> gpu_basis -> fStart = 0;
    gpu -> gpu_sim.fStart = 0;
    
    if (sort_method == 0) {
        QUICKDouble cut1 = 1E-10;
        QUICKDouble cut2 = 1E-4;
        for (int qp = 0; qp <= 6 ; qp++){
            for (int q = 0; q <= 3; q++) {
                for (int p = 0; p <= 3; p++) {
                    if (p+q==qp){
                        
                        int b=0;
                        for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
                            for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                                if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                    if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > cut2 &&
                                        gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
                                        a++;
                                        b++;
                                    }
                                }
                            }
                        }
                        
                        PRINTDEBUG("FINISH STEP 2")
                        flag = true;
                        for (int i = 0; i < b - 1; i ++)
                        {
                            flag = true;
                            for (int j = 0; j < b - i - 1; j ++)
                            {
                                if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] <
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y]])
                                {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1] = temp;
                                    flag = false;
                                }
                                else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] ==
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].x]] *
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].y]])
                                {
                                    if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]]<
                                        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]]) {
                                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b] = temp;
                                        flag = false;
                                    }
                                }
                            }
                            
                            if (flag == true)
                                break;
                        }
                        flag = true;
                        PRINTDEBUG("FINISH STEP 3")
                        
                        if (b != 0) {
                            
                            if (q==2 && p==3){
#ifdef DEBUG
                                fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                gpu -> gpu_basis -> fStart = a - b;
                                gpu -> gpu_sim.fStart = a - b;
                            }
                            
                            if (p+q==6){
#ifdef DEBUG                                
                                fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                gpu -> gpu_basis -> ffStart = a - b;
                                gpu -> gpu_sim.ffStart = a - b;
                            }

                        }
                        
                        //            if (q + p <= 4) {
                        // First to order ERI type
                        // Second to order primitive Gaussian function number
                        // Third to order Schwartz cutoff upbound
                        
                        
                    }
                }
            }
        }
        for (int qp = 0; qp <= 6 ; qp++){
            for (int q = 0; q <= 3; q++) {
                for (int p = 0; p <= 3; p++) {
                    if (p+q==qp){
                        
                        int b=0;
                        for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
                            for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                                if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                    if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) <= cut2 &&
                                        LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > cut1 &&
                                        gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
                                        a++;
                                        b++;
                                    }
                                }
                            }
                        }
                        
                        PRINTDEBUG("FINISH STEP 1")
#ifdef DEBUG
                        fprintf(gpu->debugFile,"a=%i b=%i\n", a, b);
#endif
                        for (int i = 0; i < b - 1; i ++)
                        {
                            flag = true;
                            for (int j = 0; j < b - i - 1; j ++)
                            {
                                if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x], \
                                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                                     LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x], \
                                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                                    //&&
                                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].x] == q &&  \
                                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].y]== p &&  \
                                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].x] == q && \
                                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].y] == p )
                                {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
                                    flag = false;
                                }
                            }
                            
                            if (flag == true)
                                break;
                        }
                        PRINTDEBUG("FINISH STEP 2")
                        flag = true;
                        for (int i = 0; i < b - 1; i ++)
                        {
                            flag = true;
                            for (int j = 0; j < b - i - 1; j ++)
                            {
                                if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] <
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y]])
                                {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1] = temp;
                                    flag = false;
                                }
                                else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] ==
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].x]] *
                                         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].y]])
                                {
                                    if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]]<
                                        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]]) {
                                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b] = temp;
                                        flag = false;
                                    }
                                }
                            }
                            
                            if (flag == true)
                                break;
                        }
                        flag = true;
                        
                        if (b != 0) {
                            if (q==2 && p==3 && gpu -> gpu_sim.fStart == 0){
#ifdef DEBUG
                                fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                gpu -> gpu_basis -> fStart = a - b;
                                gpu -> gpu_sim.fStart = a - b;
                            }
                            
                            if (p+q==6 && gpu -> gpu_sim.ffStart == 0){
#ifdef DEBUG                                
                                fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                gpu -> gpu_basis -> ffStart = a - b;
                                gpu -> gpu_sim.ffStart = a - b;
                            }
                        }
                        
                        PRINTDEBUG("FINISH STEP 3")
                    }
                }
            }
        }
        
        if (gpu -> gpu_sim.ffStart == 0) {
            gpu -> gpu_sim.ffStart = a;
            gpu -> gpu_basis -> ffStart = a;
        }
        
        
        if (gpu -> gpu_sim.fStart == 0) {
            gpu -> gpu_sim.fStart = a;
            gpu -> gpu_basis -> fStart = a;
        }
        
        
        //  }
        /*
         PRINTDEBUG("WORKING on F Orbital")
         
         gpu -> gpu_basis -> fStart = a;
         gpu -> gpu_sim.fStart = a;
         
         printf("df, fd, or ff starts from %i \n", a);
         
         for (int q = 0; q <= 3; q++) {
         for (int p = 0; p <= 3; p++) {
         
         if (q == 3 && p == 3) {
         gpu -> gpu_basis -> ffStart = a;
         gpu -> gpu_sim.ffStart = a;
         
         printf("ff starts from %i \n", a);
         }
         
         if (q + p > 4) {
         
         // First to order ERI type
         // Second to order primitive Gaussian function number
         // Third to order Schwartz cutoff upbound
         
         int b=0;
         for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
         for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
         if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
         if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > 1E-12 &&
         gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
         a++;
         b++;
         }
         }
         }
         }
         
         PRINTDEBUG("FINISH STEP 1")
         printf("a=%i b=%i\n", a, b);
         for (int i = 0; i < b - 1; i ++)
         {
         flag = true;
         for (int j = 0; j < b - i - 1; j ++)
         {
         if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x], \
         gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
         LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x], \
         gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
         //&&
         //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].x] == q &&  \
         //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].y]== p &&  \
         //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].x] == q && \
         //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].y] == p )
         {
         temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
         flag = false;
         }
         }
         
         if (flag == true)
         break;
         }
         
         PRINTDEBUG("FINISH STEP 2")
         flag = true;
         
         for (int i = 0; i < b - 1; i ++)
         {
         flag = true;
         for (int j = 0; j < b - i - 1; j ++)
         {
         if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] <
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]] *
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y]])
         {
         temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1] = temp;
         flag = false;
         }
         else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] ==
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].x]] *
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].y]])
         {
         if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]]<
         gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]]) {
         temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
         gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
         flag = false;
         }
         }
         }
         
         if (flag == true)
         break;
         }
         
         flag = true;
         PRINTDEBUG("FINISH STEP 3")
         
         }
         }
         }*/
    }
    
    if (sort_method == 1) {
        
        int b=0;
        for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
            for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                //if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > 1E-12 &&
                    gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
                    a++;
                    b++;
                }
                //}
            }
        }
        
        
        for (int i = 0; i < b - 1; i ++)
        {
            flag = true;
            for (int j = 0; j < b - i - 1; j ++)
            {
                if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x], \
                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                     LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x], \
                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                    //&&
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].x] == q &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].y]== p &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].x] == q && \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].y] == p )
                {
                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
                    flag = false;
                }
            }
            
            if (flag == true)
                break;
        }
        
        
        
        flag = true;
        
        
    }
    
    if (sort_method == 2) {
        QUICKDouble cut1 = 1E-8;
        QUICKDouble cut2 = 1E-11;
        
        for (int q = 0; q <= 3; q++) {
            for (int p = 0; p <= 3; p++) {
                
                if (q + p <= 4) {
                    // First to order ERI type
                    // Second to order primitive Gaussian function number
                    // Third to order Schwartz cutoff upbound
                    
                    int b=0;
                    for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
                        for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                            if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > cut1 &&
                                    gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
                                    a++;
                                    b++;
                                }
                            }
                        }
                    }
                    
                    
                    PRINTDEBUG("FINISH STEP 1")
#ifdef DEBUG
                    fprintf(gpu->debugFile,"a=%i b=%i\n", a, b);
#endif
                    for (int i = 0; i < b - 1; i ++)
                    {
                        flag = true;
                        for (int j = 0; j < b - i - 1; j ++)
                        {
                            if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x], \
                                      gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                                 LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x], \
                                      gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                                //&&
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].x] == q &&  \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].y]== p &&  \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].x] == q && \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].y] == p )
                            {
                                temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
                                flag = false;
                            }
                        }
                        
                        if (flag == true)
                            break;
                    }
                    
                    PRINTDEBUG("FINISH STEP 2")
                    flag = true;
                    
                    for (int i = 0; i < b - 1; i ++)
                    {
                        flag = true;
                        for (int j = 0; j < b - i - 1; j ++)
                        {
                            if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] <
                                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]] *
                                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y]])
                            {
                                temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1];
                                gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b + 1] = temp;
                                flag = false;
                            }
                            else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]] *
                                     gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y]] ==
                                     gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].x]] *
                                     gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b+1].y]])
                            {
                                if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x]]<
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x]]) {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b] = temp;
                                    flag = false;
                                }
                            }
                        }
                        
                        if (flag == true)
                            break;
                    }
                    
                    flag = true;
                    PRINTDEBUG("FINISH STEP 3")
                }
            }
        }
        
        int b=0;
        for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
            for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                //if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > cut2 &&
                    LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) < cut1 &&
                    gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].x = i;
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[a].y = j;
                    a++;
                    b++;
                }
                //}
            }
        }
        
        
        for (int i = 0; i < b - 1; i ++)
        {
            flag = true;
            for (int j = 0; j < b - i - 1; j ++)
            {
                if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].x], \
                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                     LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].x], \
                          gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                    //&&
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].x] == q &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+1].y]== p &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].x] == q && \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j].y] == p )
                {
                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[j + 1+a-b] = temp;
                    flag = false;
                }
            }
            
            if (flag == true)
                break;
        }
        
        
        
        flag = true;
    }
    
    
#ifdef DEBUG    
    fprintf(gpu->debugFile,"a = %i, total = %i, pect= %f\n", a, gpu->gpu_basis->Qshell * (gpu->gpu_basis->Qshell+1)/2, (float) 2*a/(gpu->gpu_basis->Qshell*(gpu->gpu_basis->Qshell)));
#endif
    
    gpu->gpu_cutoff->sqrQshell  = a;

#ifdef DEBUG    
    fprintf(gpu->debugFile,"SS = %i\n",a);
    for (int i = 0; i<a; i++) {
        fprintf(gpu->debugFile,"%8i %4i %4i %18.13f Q=%4i %4i %4i %4i prim = %4i %4i\n",i, \
        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x, \
        gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y, \
        LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x], gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y], gpu->nshell, gpu->nshell),\
        gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x], \
        gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y], \
        gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x], \
        gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y], \
        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x]], \
        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y]]);
    }
#endif
    
    gpu -> gpu_cutoff -> sorted_YCutoffIJ  -> Upload();
    gpu -> gpu_sim.sqrQshell        = gpu -> gpu_cutoff -> sqrQshell;
    gpu -> gpu_sim.YCutoff          = gpu -> gpu_cutoff -> YCutoff -> _devData;
    gpu -> gpu_sim.cutPrim          = gpu -> gpu_cutoff -> cutPrim -> _devData;
    gpu -> gpu_sim.sorted_YCutoffIJ = gpu -> gpu_cutoff -> sorted_YCutoffIJ  -> _devData;

#ifdef CUDA_MPIV

   cudaEvent_t t_start, t_end;
   float t_time;
   cudaEventCreate(&t_start);
   cudaEventCreate(&t_end);
   cudaEventRecord(t_start, 0);

   mgpu_eri_greedy_distribute();

   cudaEventRecord(t_end, 0);
   cudaEventSynchronize(t_end);
   cudaEventElapsedTime(&t_time, t_start, t_end);   

   gpu -> timer -> t_2elb += (double) t_time/1000;

   cudaEventDestroy(t_start);
   cudaEventDestroy(t_end);

#endif   
 
    gpu -> gpu_cutoff -> YCutoff -> DeleteCPU();
    gpu -> gpu_cutoff -> cutPrim -> DeleteCPU();
    gpu -> gpu_cutoff -> sorted_YCutoffIJ -> DeleteCPU();
 
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD CUTOFF",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
 
    PRINTDEBUG("COMPLETE UPLOADING CUTOFF")

}

//-----------------------------------------------
//  upload calculated information
//-----------------------------------------------
extern "C" void gpu_upload_calculated_(QUICKDouble* o, QUICKDouble* co, QUICKDouble* vec, QUICKDouble* dense)
{
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO UPLOAD O MATRIX")
    
    gpu -> gpu_calculated -> o        =   new cuda_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);
    gpu -> gpu_calculated -> o        ->  DeleteGPU();
    gpu -> gpu_calculated -> dense    =   new cuda_buffer_type<QUICKDouble>(dense,  gpu->nbasis, gpu->nbasis);
    gpu -> gpu_calculated -> oULL     =   new cuda_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
    
    
    /*
     oULL is the unsigned long long int type of O matrix. The reason to do so is because
     Atomic Operator for CUDA 2.0 is only available for integer. So for double precision type,
     an comprimise way is to multiple a very large number (OSCALE), first and divided it
     after atomic operator.
     */
    /*
    for (int i = 0; i<gpu->nbasis; i++) {
        for (int j = 0; j<gpu->nbasis; j++) {
            QUICKULL valUII = (QUICKULL) (fabs ( LOC2( gpu->gpu_calculated->o->_hostData, i, j, gpu->nbasis, gpu->nbasis)*OSCALE + (QUICKDouble)0.5));
            
            if (LOC2( gpu->gpu_calculated->o->_hostData, i, j, gpu->nbasis, gpu->nbasis)<(QUICKDouble)0.0)
            {
                valUII = 0ull - valUII;
            }
            
            LOC2( gpu->gpu_calculated->oULL->_hostData, i, j, gpu->nbasis, gpu->nbasis) = valUII;
        }
    }
    */
    
    //    gpu -> gpu_calculated -> o        -> Upload();
    gpu -> gpu_calculated -> dense    -> Upload();
    gpu -> gpu_calculated -> oULL     -> Upload();
    
    //    gpu -> gpu_sim.o                 =  gpu -> gpu_calculated -> o -> _devData;
    gpu -> gpu_sim.dense             =  gpu -> gpu_calculated -> dense -> _devData;
    gpu -> gpu_sim.oULL              =  gpu -> gpu_calculated -> oULL -> _devData;
    
    
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD CALCULATE",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE UPLOADING O MATRIX")
}

// Added by Madu Manathunga on 01/07/2020
//This method uploads density matrix onto gpu for XC gradient calculation
extern "C" void gpu_upload_density_matrix_(QUICKDouble* dense)
{
    gpu -> gpu_calculated -> dense    =   new cuda_buffer_type<QUICKDouble>(dense,  gpu->nbasis, gpu->nbasis);
    gpu -> gpu_calculated -> dense    -> Upload();
    gpu -> gpu_sim.dense             =  gpu -> gpu_calculated -> dense -> _devData;
}

//-----------------------------------------------
//  upload basis set information
//-----------------------------------------------
extern "C" void gpu_upload_basis_(int* nshell, int* nprim, int* jshell, int* jbasis, int* maxcontract, \
                                  int* ncontract, int* itype,     QUICKDouble* aexp,      QUICKDouble* dcoeff,\
                                  int* first_basis_function, int* last_basis_function, int* first_shell_basis_function, int* last_shell_basis_function, \
                                  int* ncenter,   int* kstart,    int* katom,     int* ktype,     int* kprim,  int* kshell, int* Ksumtype, \
                                  int* Qnumber,   int* Qstart,    int* Qfinal,    int* Qsbasis,   int* Qfbasis,\
                                  QUICKDouble* gccoeff,           QUICKDouble* cons,      QUICKDouble* gcexpo, int* KLMN)
{
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO UPLOAD BASIS")
    
    gpu -> gpu_basis -> nshell          =   *nshell;
    gpu -> gpu_basis -> nprim           =   *nprim;
    gpu -> gpu_basis -> jshell          =   *jshell;
    gpu -> gpu_basis -> jbasis          =   *jbasis;
    gpu -> gpu_basis -> maxcontract     =   *maxcontract;
    
    gpu -> nshell                       =   *nshell;
    gpu -> nprim                        =   *nprim;
    gpu -> jshell                       =   *jshell;
    gpu -> jbasis                       =   *jbasis;
    
    gpu -> gpu_sim.nshell                   =   *nshell;
    gpu -> gpu_sim.nprim                    =   *nprim;
    gpu -> gpu_sim.jshell                   =   *jshell;
    gpu -> gpu_sim.jbasis                   =   *jbasis;
    gpu -> gpu_sim.maxcontract              =   *maxcontract;
    
    
    gpu -> gpu_basis -> ncontract                   =   new cuda_buffer_type<int>(ncontract, gpu->nbasis);//gpu->nbasis);
    gpu -> gpu_basis -> itype                       =   new cuda_buffer_type<int>(itype, 3,  gpu->nbasis);//3, gpu->nbasis);
    gpu -> gpu_basis -> aexp                        =   new cuda_buffer_type<QUICKDouble>(aexp, gpu->gpu_basis->maxcontract, gpu->nbasis);//gpu->gpu_basis->maxcontract, gpu->nbasis);
    gpu -> gpu_basis -> dcoeff                      =   new cuda_buffer_type<QUICKDouble>(dcoeff, gpu->gpu_basis->maxcontract, gpu->nbasis);//gpu->gpu_basis->maxcontract, gpu->nbasis);
    /*
     gpu -> gpu_basis -> first_basis_function        =   new cuda_buffer_type<int>(first_basis_function, 1);//gpu->natom);
     gpu -> gpu_basis -> last_basis_function         =   new cuda_buffer_type<int>(last_basis_function,  1);//gpu->natom);
     
     gpu -> gpu_basis -> first_shell_basis_function  =   new cuda_buffer_type<int>(first_shell_basis_function, 1);//gpu->gpu_basis->nshell);
     gpu -> gpu_basis -> last_shell_basis_function   =   new cuda_buffer_type<int>(last_shell_basis_function,  1);//gpu->gpu_basis->nshell);
     
     gpu -> gpu_basis -> ktype                       =   new cuda_buffer_type<int>(ktype,    gpu->gpu_basis->nshell);
     gpu -> gpu_basis -> kshell                      =   new cuda_buffer_type<int>(kshell,   93);
     */
    gpu -> gpu_basis -> ncenter                     =   new cuda_buffer_type<int>(ncenter,  gpu->gpu_basis->nbasis);
    
    gpu -> gpu_basis -> kstart                      =   new cuda_buffer_type<int>(kstart,   gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> katom                       =   new cuda_buffer_type<int>(katom,    gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> kprim                       =   new cuda_buffer_type<int>(kprim,    gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> Ksumtype                    =   new cuda_buffer_type<int>(Ksumtype, gpu->gpu_basis->nshell+1);
    
    gpu -> gpu_basis -> Qnumber                     =   new cuda_buffer_type<int>(Qnumber,  gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> Qstart                      =   new cuda_buffer_type<int>(Qstart,   gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> Qfinal                      =   new cuda_buffer_type<int>(Qfinal,   gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> Qsbasis                     =   new cuda_buffer_type<int>(Qsbasis,  gpu->gpu_basis->nshell, 4);
    gpu -> gpu_basis -> Qfbasis                     =   new cuda_buffer_type<int>(Qfbasis,  gpu->gpu_basis->nshell, 4);
    gpu -> gpu_basis -> gccoeff                     =   new cuda_buffer_type<QUICKDouble>(gccoeff, MAXPRIM, gpu->nbasis);
    
    gpu -> gpu_basis -> cons                        =   new cuda_buffer_type<QUICKDouble>(cons, gpu->nbasis);
    gpu -> gpu_basis -> gcexpo                      =   new cuda_buffer_type<QUICKDouble>(gcexpo, MAXPRIM, gpu->nbasis);
    gpu -> gpu_basis -> KLMN                        =   new cuda_buffer_type<int>(KLMN, 3, gpu->nbasis);
    
    gpu -> gpu_basis -> prim_start                  =   new cuda_buffer_type<int>(gpu->gpu_basis->nshell);
    gpu -> gpu_basis -> prim_total = 0;
    
    for (int i = 0 ; i < gpu->gpu_basis->nshell; i++) {
        gpu -> gpu_basis -> prim_start -> _hostData[i] = gpu -> gpu_basis -> prim_total;
        gpu -> gpu_basis -> prim_total += gpu -> gpu_basis -> kprim -> _hostData[i];
    }

#ifdef DEBUG    
    for (int i = 0; i<gpu->gpu_basis->nshell; i++) {
        fprintf(gpu->debugFile,"for %i prim= %i, start= %i\n", i, gpu -> gpu_basis -> kprim -> _hostData[i], gpu -> gpu_basis -> prim_start -> _hostData[i]);
    }
    fprintf(gpu->debugFile,"total=%i\n", gpu -> gpu_basis -> prim_total);
#endif

    int prim_total = gpu -> gpu_basis -> prim_total;
    gpu -> gpu_sim.prim_total = gpu -> gpu_basis -> prim_total;
    
    gpu -> gpu_basis -> Xcoeff                      =   new cuda_buffer_type<QUICKDouble>(2*gpu->jbasis, 2*gpu->jbasis);
    gpu -> gpu_basis -> expoSum                     =   new cuda_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu -> gpu_basis -> weightedCenterX             =   new cuda_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu -> gpu_basis -> weightedCenterY             =   new cuda_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu -> gpu_basis -> weightedCenterZ             =   new cuda_buffer_type<QUICKDouble>(prim_total, prim_total);
    
    
    /*
     After uploading basis set information, we want to do some more things on CPU so that will accelarate GPU.
     The very first is to sort orbital type. In this case, we will calculate s orbitals then p, d, and etc.
     Here Qshell is the number of shell orbtials, for example, sp orbitals account for 2 shell orbitals, and s orbital accounts
     1 shell orbital.
     */
    gpu->gpu_basis->Qshell = 0;
    for (int i = 0; i<gpu->nshell; i++) {
        gpu->gpu_basis->Qshell += gpu->gpu_basis->Qfinal->_hostData[i] - gpu->gpu_basis->Qstart->_hostData[i] + 1;
    }
    
    for (int i = 0; i<gpu->gpu_basis->nshell; i++) {
        for (int j = 0; j<4; j++) {
            LOC2(gpu->gpu_basis->Qsbasis->_hostData, i, j, gpu->gpu_basis->nshell, 4) += gpu->gpu_basis->Ksumtype->_hostData[i];
            LOC2(gpu->gpu_basis->Qfbasis->_hostData, i, j, gpu->gpu_basis->nshell, 4) += gpu->gpu_basis->Ksumtype->_hostData[i];
        }
    }
   
#ifdef DEBUG
    //MGPU_TESTING
    fprintf(gpu->debugFile,"nshell: %i jshell: %i Qshell: %i \n",gpu->gpu_basis->nshell, gpu->gpu_basis->jshell, gpu->gpu_basis->Qshell);
#endif 

    gpu -> gpu_sim.Qshell = gpu->gpu_basis->Qshell;
    
    gpu -> gpu_basis -> sorted_Q                    =   new cuda_buffer_type<int>( gpu->gpu_basis->Qshell);
    gpu -> gpu_basis -> sorted_Qnumber              =   new cuda_buffer_type<int>( gpu->gpu_basis->Qshell);
    
    /*
     Now because to sort, sorted_Q stands for the shell no, and sorted_Qnumber is the shell orbital type (or angular momentum).
     For instance:
     
     original: s sp s s s sp s s
     sorteed : s s  s s s s  s s p p
     
     move p orbital to the end of the sequence. so the Qshell stands for the length of sequence after sorting.
     */
    int a = 0;
    for (int i = 0; i<gpu->gpu_basis->nshell; i++) {
        for (int j = gpu->gpu_basis->Qstart->_hostData[i]; j<= gpu->gpu_basis->Qfinal->_hostData[i]; j++) {
            
            if (a == 0) {
                gpu->gpu_basis->sorted_Q->_hostData[0] = i;
                gpu->gpu_basis->sorted_Qnumber->_hostData[0] = j;
            }else {
                for (int k = 0; k<a; k++) {
                    if (j<gpu->gpu_basis->sorted_Qnumber->_hostData[k]) {
                        
                        int kk = k;
                        for (int l = a; l> kk; l--) {
                            gpu->gpu_basis->sorted_Q->_hostData[l] = gpu->gpu_basis->sorted_Q->_hostData[l-1];
                            gpu->gpu_basis->sorted_Qnumber->_hostData[l] = gpu->gpu_basis->sorted_Qnumber->_hostData[l-1];
                        }
                        
                        gpu->gpu_basis->sorted_Q->_hostData[kk] = i;
                        gpu->gpu_basis->sorted_Qnumber->_hostData[kk] = j;
                        break;
                    }
                    gpu->gpu_basis->sorted_Q->_hostData[a] = i;
                    gpu->gpu_basis->sorted_Qnumber->_hostData[a] = j;
                }
            }
            a++;
        }
    }
    
    
    
    /*
     for (int i = 0; i<gpu->gpu_basis->Qshell; i++) {
     for (int j = i; j<gpu->gpu_basis->Qshell; j++) {
     if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == gpu->gpu_basis->sorted_Qnumber->_hostData[j]) {
     if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[i]] < gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[j]]) {
     int temp = gpu->gpu_basis->sorted_Q->_hostData[j];
     gpu->gpu_basis->sorted_Q->_hostData[j] = gpu->gpu_basis->sorted_Q->_hostData[i];
     gpu->gpu_basis->sorted_Q->_hostData[i] = temp;
     }
     }
     }
     }*/

#ifdef DEBUG    
    fprintf(gpu->debugFile,"Pre-Sorted orbitals:\n");
    fprintf(gpu->debugFile,"Qshell = %i\n", gpu->gpu_basis->Qshell);
    for (int i = 0; i<gpu->gpu_basis->Qshell; i++) {
        fprintf(gpu->debugFile,"i= %i, Q=%i, Qnumber= %i, nprim = %i \n", i, gpu->gpu_basis->sorted_Q->_hostData[i], gpu->gpu_basis->sorted_Qnumber->_hostData[i],
               gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[i]]);
    }
#endif    
    
    /*
     some pre-calculated variables includes
     
     expoSum(i,j) = expo(i)+expo(j)
     ------------->                 ->          ->
     weightedCenter(i,j) = (expo(i)*i + expo(j)*j)/(expo(i)+expo(j))
     */
    for (int i = 0; i<gpu->jshell; i++) {
        for (int j = 0; j<gpu->jshell; j++) {
            int kAtomI = gpu->gpu_basis->katom->_hostData[i];
            int kAtomJ = gpu->gpu_basis->katom->_hostData[j];
            int KsumtypeI = gpu->gpu_basis->Ksumtype->_hostData[i];
            int KsumtypeJ = gpu->gpu_basis->Ksumtype->_hostData[j];
            int kstartI = gpu->gpu_basis->kstart->_hostData[i];
            int kstartJ = gpu->gpu_basis->kstart->_hostData[j];
            
            QUICKDouble distance = 0;
            for (int k = 0; k<3; k++) {
                distance += pow(LOC2(gpu->xyz->_hostData, k, kAtomI-1, 3, gpu->natom)
                                -LOC2(gpu->xyz->_hostData, k, kAtomJ-1, 3, gpu->natom),2);
            }
            
            QUICKDouble DIJ = distance;
            
            for (int ii = 0; ii<gpu->gpu_basis->kprim->_hostData[i]; ii++) {
                for (int jj = 0; jj<gpu->gpu_basis->kprim->_hostData[j]; jj++) {
                    
                    QUICKDouble II = LOC2(gpu->gpu_basis->gcexpo->_hostData, ii , KsumtypeI-1, MAXPRIM, gpu->nbasis);
                    QUICKDouble JJ = LOC2(gpu->gpu_basis->gcexpo->_hostData, jj , KsumtypeJ-1, MAXPRIM, gpu->nbasis);
                    
                    int ii_start = gpu->gpu_basis->prim_start->_hostData[i];
                    int jj_start = gpu->gpu_basis->prim_start->_hostData[j];
                    
                    //expoSum(i,j) = expo(i)+expo(j)
                    LOC2(gpu->gpu_basis->expoSum->_hostData, ii_start+ii, jj_start+jj, prim_total, prim_total) = II + JJ;
                    
                    
                    //        ------------->                 ->          ->
                    //        weightedCenter(i,j) = (expo(i)*i + expo(j)*j)/(expo(i)+expo(j))
                    LOC2(gpu->gpu_basis->weightedCenterX->_hostData, ii_start+ii, jj_start+jj, prim_total, prim_total) = \
                    (LOC2(gpu->xyz->_hostData, 0, kAtomI-1, 3, gpu->natom) * II + LOC2(gpu->xyz->_hostData, 0, kAtomJ-1, 3, gpu->natom)*JJ)/(II+JJ);
                    LOC2(gpu->gpu_basis->weightedCenterY->_hostData, ii_start+ii, jj_start+jj, prim_total, prim_total) = \
                    (LOC2(gpu->xyz->_hostData, 1, kAtomI-1, 3, gpu->natom) * II + LOC2(gpu->xyz->_hostData, 1, kAtomJ-1, 3, gpu->natom)*JJ)/(II+JJ);
                    LOC2(gpu->gpu_basis->weightedCenterZ->_hostData, ii_start+ii, jj_start+jj, prim_total, prim_total) = \
                    (LOC2(gpu->xyz->_hostData, 2, kAtomI-1, 3, gpu->natom) * II + LOC2(gpu->xyz->_hostData, 2, kAtomJ-1, 3, gpu->natom)*JJ)/(II+JJ);
                    
                    
                    // Xcoeff = exp(-II*JJ/(II+JJ) * DIJ) / (II+JJ) * coeff(i) * coeff(j) * X0
                    QUICKDouble X = exp(-II*JJ/(II+JJ)*DIJ)/(II+JJ);
                    
                    for (int itemp = gpu->gpu_basis->Qstart->_hostData[i]; itemp <= gpu->gpu_basis->Qfinal->_hostData[i]; itemp++) {
                        for (int itemp2 = gpu->gpu_basis->Qstart->_hostData[j]; itemp2 <= gpu->gpu_basis->Qfinal->_hostData[j]; itemp2++) {
                            LOC4(gpu->gpu_basis->Xcoeff->_hostData, kstartI+ii-1, kstartJ+jj-1, \
                                 itemp-gpu->gpu_basis->Qstart->_hostData[i], itemp2-gpu->gpu_basis->Qstart->_hostData[j], gpu->jbasis, gpu->jbasis, 2, 2)
                            = X0 * X * LOC2(gpu->gpu_basis->gccoeff->_hostData, ii, KsumtypeI+itemp-1, MAXPRIM, gpu->nbasis) \
                            * LOC2(gpu->gpu_basis->gccoeff->_hostData, jj, KsumtypeJ+itemp2-1, MAXPRIM, gpu->nbasis);
                        }
                    }
                }
            }
        }
    }

//    gpu -> gpu_basis -> upload_all();
    gpu -> gpu_basis -> ncontract -> Upload();
    gpu -> gpu_basis ->itype->Upload();
    gpu -> gpu_basis ->aexp->Upload();
    gpu -> gpu_basis ->dcoeff->Upload();    
    gpu -> gpu_basis ->ncenter->Upload();
    gpu -> gpu_basis ->kstart->Upload();
    gpu -> gpu_basis ->katom->Upload();
    gpu -> gpu_basis ->kprim->Upload();
    gpu -> gpu_basis ->Ksumtype->Upload();
    gpu -> gpu_basis ->Qnumber->Upload();
    gpu -> gpu_basis ->Qstart->Upload();
    gpu -> gpu_basis ->Qfinal->Upload();
    gpu -> gpu_basis ->Qsbasis->Upload();
    gpu -> gpu_basis ->Qfbasis->Upload();
    gpu -> gpu_basis ->gccoeff->Upload();
    gpu -> gpu_basis ->cons->Upload();
    gpu -> gpu_basis ->Xcoeff->Upload();
    gpu -> gpu_basis ->gcexpo->Upload();
    gpu -> gpu_basis ->KLMN->Upload();
    gpu -> gpu_basis ->prim_start->Upload();
    gpu -> gpu_basis ->Xcoeff->Upload();
    gpu -> gpu_basis ->expoSum->Upload();
    gpu -> gpu_basis ->weightedCenterX->Upload();
    gpu -> gpu_basis ->weightedCenterY->Upload();
    gpu -> gpu_basis ->weightedCenterZ->Upload();
    gpu -> gpu_basis ->sorted_Q->Upload();
    gpu -> gpu_basis ->sorted_Qnumber->Upload();

    gpu -> gpu_sim.expoSum                      =   gpu -> gpu_basis -> expoSum -> _devData;
    gpu -> gpu_sim.weightedCenterX              =   gpu -> gpu_basis -> weightedCenterX -> _devData;
    gpu -> gpu_sim.weightedCenterY              =   gpu -> gpu_basis -> weightedCenterY -> _devData;
    gpu -> gpu_sim.weightedCenterZ              =   gpu -> gpu_basis -> weightedCenterZ -> _devData;
    gpu -> gpu_sim.sorted_Q                     =   gpu -> gpu_basis -> sorted_Q -> _devData;
    gpu -> gpu_sim.sorted_Qnumber               =   gpu -> gpu_basis -> sorted_Qnumber -> _devData;
    gpu -> gpu_sim.Xcoeff                       =   gpu -> gpu_basis -> Xcoeff -> _devData;
    gpu -> gpu_sim.ncontract                    =   gpu -> gpu_basis -> ncontract -> _devData;
    gpu -> gpu_sim.dcoeff                       =   gpu -> gpu_basis -> dcoeff -> _devData;
    gpu -> gpu_sim.aexp                         =   gpu -> gpu_basis -> aexp -> _devData;
    gpu -> gpu_sim.ncenter                      =   gpu -> gpu_basis -> ncenter -> _devData;
    gpu -> gpu_sim.itype                        =   gpu -> gpu_basis -> itype -> _devData;
    gpu -> gpu_sim.prim_start                   =   gpu -> gpu_basis -> prim_start -> _devData;
    /*
     gpu -> gpu_sim.first_basis_function         =   gpu -> gpu_basis -> first_basis_function -> _devData;
     gpu -> gpu_sim.last_basis_function          =   gpu -> gpu_basis -> last_basis_function -> _devData;
     gpu -> gpu_sim.first_shell_basis_function   =   gpu -> gpu_basis -> first_shell_basis_function -> _devData;
     gpu -> gpu_sim.last_shell_basis_function    =   gpu -> gpu_basis -> last_shell_basis_function -> _devData;
     gpu -> gpu_sim.ktype                        =   gpu -> gpu_basis -> ktype -> _devData;
     gpu -> gpu_sim.kshell                       =   gpu -> gpu_basis -> kshell -> _devData;
     */
    gpu -> gpu_sim.kstart                       =   gpu -> gpu_basis -> kstart -> _devData;
    gpu -> gpu_sim.katom                        =   gpu -> gpu_basis -> katom -> _devData;
    gpu -> gpu_sim.kprim                        =   gpu -> gpu_basis -> kprim -> _devData;
    gpu -> gpu_sim.Ksumtype                     =   gpu -> gpu_basis -> Ksumtype -> _devData;
    gpu -> gpu_sim.Qnumber                      =   gpu -> gpu_basis -> Qnumber -> _devData;
    gpu -> gpu_sim.Qstart                       =   gpu -> gpu_basis -> Qstart -> _devData;
    gpu -> gpu_sim.Qfinal                       =   gpu -> gpu_basis -> Qfinal -> _devData;
    gpu -> gpu_sim.Qsbasis                      =   gpu -> gpu_basis -> Qsbasis -> _devData;
    gpu -> gpu_sim.Qfbasis                      =   gpu -> gpu_basis -> Qfbasis -> _devData;
    gpu -> gpu_sim.gccoeff                      =   gpu -> gpu_basis -> gccoeff -> _devData;
    gpu -> gpu_sim.cons                         =   gpu -> gpu_basis -> cons -> _devData;
    gpu -> gpu_sim.gcexpo                       =   gpu -> gpu_basis -> gcexpo -> _devData;
    gpu -> gpu_sim.KLMN                         =   gpu -> gpu_basis -> KLMN -> _devData;
    
    
    gpu -> gpu_basis -> expoSum -> DeleteCPU();
    gpu -> gpu_basis -> weightedCenterX -> DeleteCPU();
    gpu -> gpu_basis -> weightedCenterY -> DeleteCPU();
    gpu -> gpu_basis -> weightedCenterZ -> DeleteCPU();
    gpu -> gpu_basis -> Xcoeff -> DeleteCPU();
    
    gpu -> gpu_basis -> ncontract -> DeleteCPU();
//    gpu -> gpu_basis -> dcoeff -> DeleteCPU();
    gpu -> gpu_basis -> aexp -> DeleteCPU();
    gpu -> gpu_basis -> ncenter -> DeleteCPU();
    gpu -> gpu_basis -> itype -> DeleteCPU();
    
    gpu -> gpu_basis -> kstart -> DeleteCPU();
    gpu -> gpu_basis -> katom -> DeleteCPU();
    
    //kprim can not be deleted since it will be used later
    //gpu -> gpu_basis -> kprim -> DeleteCPU();
    
    gpu -> gpu_basis -> Ksumtype -> DeleteCPU();
    gpu -> gpu_basis -> prim_start -> DeleteCPU();
    
    gpu -> gpu_basis -> Qnumber -> DeleteCPU();
    gpu -> gpu_basis -> Qstart -> DeleteCPU();
    gpu -> gpu_basis -> Qfinal -> DeleteCPU();
    
    gpu -> gpu_basis -> Qsbasis -> DeleteCPU();
    gpu -> gpu_basis -> Qfbasis -> DeleteCPU();
    gpu -> gpu_basis -> gccoeff -> DeleteCPU();
    gpu -> gpu_basis -> cons -> DeleteCPU();
    gpu -> gpu_basis -> gcexpo -> DeleteCPU();
    gpu -> gpu_basis -> KLMN -> DeleteCPU();
    
    
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD BASIS",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE UPLOADING BASIS")

}


extern "C" void gpu_upload_grad_(QUICKDouble* gradCutoff)
{
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    
    PRINTDEBUG("BEGIN TO UPLOAD GRAD")
    
    gpu -> grad = new cuda_buffer_type<QUICKDouble>(3 * gpu->natom);
    gpu -> gradULL = new cuda_buffer_type<QUICKULL>(3 * gpu->natom);

    gpu -> grad -> DeleteGPU();
    gpu -> gpu_sim.gradULL =  gpu -> gradULL -> _devData;
   
    gpu -> gradULL -> Upload();
    
    gpu -> gpu_cutoff -> gradCutoff = *gradCutoff;
    gpu -> gpu_sim.gradCutoff         = gpu -> gpu_cutoff -> gradCutoff;
    
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("UPLOAD GRAD",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE UPLOADING GRAD")
    
}


//Computes grid weights before grid point packing
extern "C" void gpu_get_ssw_(QUICKDouble *gridx, QUICKDouble *gridy, QUICKDouble *gridz, QUICKDouble *wtang, QUICKDouble *rwt, QUICKDouble *rad3, QUICKDouble *sswt, QUICKDouble *weight, int *gatm, int *count){

	PRINTDEBUG("BEGIN TO COMPUTE SSW")

	gpu -> gpu_xcq -> npoints       = *count;
        gpu -> xc_threadsPerBlock = SM_2X_XC_THREADS_PER_BLOCK;
	
        gpu -> gpu_xcq -> gridx = new cuda_buffer_type<QUICKDouble>(gridx, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> gridy = new cuda_buffer_type<QUICKDouble>(gridy, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> gridz = new cuda_buffer_type<QUICKDouble>(gridz, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> wtang = new cuda_buffer_type<QUICKDouble>(wtang, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> rwt   = new cuda_buffer_type<QUICKDouble>(rwt, gpu -> gpu_xcq -> npoints);	
	gpu -> gpu_xcq -> rad3  = new cuda_buffer_type<QUICKDouble>(rad3, gpu -> gpu_xcq -> npoints); 
        gpu -> gpu_xcq -> gatm  = new cuda_buffer_type<int>(gatm, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> sswt  = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> weight= new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);

        gpu -> gpu_xcq -> gridx -> Upload();
        gpu -> gpu_xcq -> gridy -> Upload();
        gpu -> gpu_xcq -> gridz -> Upload();
	gpu -> gpu_xcq -> wtang -> Upload();
	gpu -> gpu_xcq -> rwt -> Upload();
	gpu -> gpu_xcq -> rad3 -> Upload();
        gpu -> gpu_xcq -> gatm -> Upload();
        gpu -> gpu_xcq -> sswt -> Upload();
        gpu -> gpu_xcq -> weight -> Upload();

        gpu -> gpu_sim.npoints  = gpu -> gpu_xcq -> npoints;
        gpu ->gpu_sim.gridx     = gpu -> gpu_xcq -> gridx -> _devData;
        gpu ->gpu_sim.gridy     = gpu -> gpu_xcq -> gridy -> _devData;
        gpu ->gpu_sim.gridz     = gpu -> gpu_xcq -> gridz -> _devData;
	gpu ->gpu_sim.wtang     = gpu -> gpu_xcq -> wtang -> _devData;	
	gpu ->gpu_sim.rwt       = gpu -> gpu_xcq -> rwt   -> _devData;
	gpu ->gpu_sim.rad3      = gpu -> gpu_xcq -> rad3  -> _devData;
        gpu ->gpu_sim.gatm      = gpu -> gpu_xcq -> gatm  -> _devData;
        gpu ->gpu_sim.sswt      = gpu -> gpu_xcq -> sswt  -> _devData;
        gpu ->gpu_sim.weight    = gpu -> gpu_xcq -> weight-> _devData;

	upload_sim_to_constant_dft(gpu);

	get_ssw(gpu);	

	gpu -> gpu_xcq -> sswt -> Download();	
	gpu -> gpu_xcq -> weight -> Download();   

	for(int i=0; i<*count;i++){
		sswt[i] = gpu -> gpu_xcq -> sswt -> _hostData[i];
		weight[i] = gpu -> gpu_xcq -> weight -> _hostData[i];
	}

	SAFE_DELETE(gpu -> gpu_xcq -> gridx);
	SAFE_DELETE(gpu -> gpu_xcq -> gridy);
	SAFE_DELETE(gpu -> gpu_xcq -> gridz);
	SAFE_DELETE(gpu -> gpu_xcq -> wtang);
	SAFE_DELETE(gpu -> gpu_xcq -> rwt);
	SAFE_DELETE(gpu -> gpu_xcq -> rad3);
        SAFE_DELETE(gpu -> gpu_xcq -> gatm);
        SAFE_DELETE(gpu -> gpu_xcq -> sswt);
        SAFE_DELETE(gpu -> gpu_xcq -> weight);
	
	PRINTDEBUG("END COMPUTE SSW")

}

void prune_grid_sswgrad(){


        PRINTDEBUG("BEGIN TO UPLOAD DFT GRID FOR SSWGRAD")

        gpu -> gpu_xcq -> dweight_ssd -> Download();
        gpu -> gpu_xcq -> exc -> Download();

        //Get the size of input arrays to sswgrad computation
        int count = 0;
        for(int i=0; i< gpu -> gpu_xcq -> npoints;i++){
                count += gpu -> gpu_xcq -> dweight_ssd -> _hostData[i];
        }

        //Load data into temporary arrays
        QUICKDouble *tmp_gridx, *tmp_gridy, *tmp_gridz, *tmp_exc, *tmp_quadwt;
        int* tmp_gatm;
        int dbyte_size = sizeof(QUICKDouble)*count;

        tmp_gridx = (QUICKDouble*) malloc(dbyte_size);
        tmp_gridy = (QUICKDouble*) malloc(dbyte_size);
        tmp_gridz = (QUICKDouble*) malloc(dbyte_size);
        tmp_exc = (QUICKDouble*) malloc(dbyte_size);
        tmp_quadwt= (QUICKDouble*) malloc(dbyte_size);
        tmp_gatm = (int*) malloc(sizeof(int)*count);

        int j=0;
        for(int i=0; i< gpu -> gpu_xcq -> npoints;i++){
                if(gpu -> gpu_xcq -> dweight_ssd -> _hostData[i] > 0){
                        tmp_gridx[j] = gpu -> gpu_xcq -> gridx -> _hostData[i];
                        tmp_gridy[j] = gpu -> gpu_xcq -> gridy -> _hostData[i];
                        tmp_gridz[j] = gpu -> gpu_xcq -> gridz -> _hostData[i];
                        tmp_exc[j] = gpu -> gpu_xcq -> exc -> _hostData[i];

                        double quadwt = (gpu -> gpu_xcq -> weight -> _hostData[i]) / (gpu -> gpu_xcq -> sswt -> _hostData[i]);
                        tmp_quadwt[j] = quadwt;

                        tmp_gatm[j] = gpu -> gpu_xcq -> gatm -> _hostData[i];
                        j++;
                }
        }

	gpu_delete_dft_grid_();

#ifdef CUDA_MPIV
        cudaEvent_t t_start, t_end;
        float t_time;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_end);
        cudaEventRecord(t_start, 0);


        int netgain = getAdjustment(gpu->mpisize, gpu->mpirank, count);
        count += netgain;

        cudaEventRecord(t_end, 0);
        cudaEventSynchronize(t_end);
        cudaEventElapsedTime(&t_time, t_start, t_end);        

        gpu -> timer -> t_xcrb += (double) t_time/1000;

#endif

         gpu -> gpu_xcq -> npoints_ssd = count;

        //Upload data using templates
#ifdef CUDA_MPIV

        gpu -> gpu_xcq -> gridx_ssd = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gridy_ssd = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gridz_ssd = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> exc_ssd = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> quadwt = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gatm_ssd = new cuda_buffer_type<int>(gpu -> gpu_xcq -> npoints_ssd);

        cudaEventRecord(t_start, 0);
         
        sswderRedistribute(gpu->mpisize, gpu->mpirank, count-netgain, count,
        tmp_gridx, tmp_gridy, tmp_gridz, tmp_exc, tmp_quadwt, tmp_gatm, gpu -> gpu_xcq -> gridx_ssd -> _hostData,
        gpu -> gpu_xcq -> gridy_ssd -> _hostData, gpu -> gpu_xcq -> gridz_ssd -> _hostData,
        gpu -> gpu_xcq -> exc_ssd -> _hostData, gpu -> gpu_xcq -> quadwt -> _hostData,
        gpu -> gpu_xcq -> gatm_ssd -> _hostData);

        cudaEventRecord(t_end, 0);
        cudaEventSynchronize(t_end);
        cudaEventElapsedTime(&t_time, t_start, t_end);

        gpu -> timer -> t_xcrb += (double) t_time/1000;
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_end);

#else

        gpu -> gpu_xcq -> gridx_ssd = new cuda_buffer_type<QUICKDouble>(tmp_gridx, gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gridy_ssd = new cuda_buffer_type<QUICKDouble>(tmp_gridy, gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gridz_ssd = new cuda_buffer_type<QUICKDouble>(tmp_gridz, gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> exc_ssd = new cuda_buffer_type<QUICKDouble>(tmp_exc, gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> quadwt = new cuda_buffer_type<QUICKDouble>(tmp_quadwt, gpu -> gpu_xcq -> npoints_ssd);
        gpu -> gpu_xcq -> gatm_ssd = new cuda_buffer_type<int>(tmp_gatm, gpu -> gpu_xcq -> npoints_ssd);
        
#endif
        gpu -> gpu_xcq -> uw_ssd= new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints_ssd * gpu->natom);

        gpu -> gpu_xcq -> gridx_ssd -> Upload();
        gpu -> gpu_xcq -> gridy_ssd -> Upload();
        gpu -> gpu_xcq -> gridz_ssd -> Upload();
        gpu -> gpu_xcq -> exc_ssd -> Upload();
        gpu -> gpu_xcq -> quadwt -> Upload();
        gpu -> gpu_xcq -> gatm_ssd -> Upload();

        gpu -> gpu_sim.npoints_ssd  = gpu -> gpu_xcq -> npoints_ssd;
        gpu -> gpu_sim.gridx_ssd = gpu -> gpu_xcq -> gridx_ssd -> _devData;
        gpu -> gpu_sim.gridy_ssd = gpu -> gpu_xcq -> gridy_ssd -> _devData;
        gpu -> gpu_sim.gridz_ssd = gpu -> gpu_xcq -> gridz_ssd -> _devData;
        gpu -> gpu_sim.exc_ssd = gpu -> gpu_xcq -> exc_ssd -> _devData;
        gpu -> gpu_sim.quadwt = gpu -> gpu_xcq -> quadwt -> _devData;
	gpu -> gpu_sim.uw_ssd = gpu -> gpu_xcq -> uw_ssd -> _devData;
        gpu -> gpu_sim.gatm_ssd = gpu -> gpu_xcq -> gatm_ssd -> _devData;

        upload_sim_to_constant_dft(gpu);

        PRINTDEBUG("COMPLETE UPLOADING DFT GRID FOR SSWGRAD")

        //Clean up temporary arrays
        free(tmp_gridx);
        free(tmp_gridy);
        free(tmp_gridz);
        free(tmp_exc);
        free(tmp_quadwt);
        free(tmp_gatm);
}	


void gpu_get_octree_info(QUICKDouble *gridx, QUICKDouble *gridy, QUICKDouble *gridz, QUICKDouble *sigrad2, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, int *bin_locator, int count, double DMCutoff, int nbins){

        PRINTDEBUG("BEGIN TO OBTAIN PRIMITIVE & BASIS FUNCTION LISTS ")

        gpu -> gpu_xcq -> npoints       = count;
        gpu -> xc_threadsPerBlock       = SM_2X_XCGRAD_THREADS_PER_BLOCK;

        gpu -> gpu_xcq -> gridx = new cuda_buffer_type<QUICKDouble>(gridx, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> gridy = new cuda_buffer_type<QUICKDouble>(gridy, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> gridz = new cuda_buffer_type<QUICKDouble>(gridz, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_basis -> sigrad2 = new cuda_buffer_type<QUICKDouble>(sigrad2, gpu->nbasis);
        gpu -> gpu_xcq -> bin_locator = new cuda_buffer_type<int>(bin_locator,gpu -> gpu_xcq -> npoints);

        gpu -> gpu_xcq -> gridx -> Upload();
        gpu -> gpu_xcq -> gridy -> Upload();
        gpu -> gpu_xcq -> gridz -> Upload();
	gpu -> gpu_basis -> sigrad2 -> Upload();
        gpu -> gpu_xcq -> bin_locator -> Upload();

        gpu -> gpu_sim.npoints  = gpu -> gpu_xcq -> npoints;
        gpu -> gpu_sim.gridx    = gpu -> gpu_xcq -> gridx -> _devData;
        gpu -> gpu_sim.gridy    = gpu -> gpu_xcq -> gridy -> _devData;
        gpu -> gpu_sim.gridz    = gpu -> gpu_xcq -> gridz -> _devData;
	gpu -> gpu_sim.sigrad2  = gpu->gpu_basis->sigrad2->_devData;
        gpu ->gpu_sim.bin_locator      = gpu -> gpu_xcq -> bin_locator -> _devData;

	gpu -> gpu_cutoff -> DMCutoff   = DMCutoff;
        gpu -> gpu_sim.DMCutoff         = gpu -> gpu_cutoff -> DMCutoff;

	//Define cfweight and pfweight arrays seperately and uplaod to gpu until we solve the problem with atomicAdd
	unsigned char *d_gpweight;
	unsigned int  *d_cfweight, *d_pfweight;

	cudaMalloc((void**)&d_gpweight, gpu -> gpu_xcq -> npoints * sizeof(unsigned char));
	cudaMalloc((void**)&d_cfweight, nbins * gpu -> nbasis * sizeof(unsigned int));	
	cudaMalloc((void**)&d_pfweight, nbins * gpu -> nbasis * gpu -> gpu_basis-> maxcontract * sizeof(unsigned int));

	cudaMemcpy(d_gpweight, gpweight, gpu -> gpu_xcq -> npoints * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cfweight, cfweight, nbins * gpu -> nbasis * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pfweight, pfweight, nbins * gpu -> nbasis * gpu -> gpu_basis-> maxcontract * sizeof(unsigned int), cudaMemcpyHostToDevice);

        upload_sim_to_constant_dft(gpu);

/*        for(int i=0; i<nbins;i++){
                //unsigned int cfweight_sum =0;
                for(int j=0; j<gpu -> nbasis; j++){
                        printf("bin id: %i basis id: %i cfcount: %i \n", i, j, cfweight[(i * gpu -> nbasis) + j]);
                        //cfweight_sum += cfweight[ (nbins*gpu -> nbasis) + j];                 
                }
                //printf("bin id: %i cfweight_sum: %i", i, cfweight_sum);
        }
*/

/*        for(int i=0; i<nbins;i++){
                for(int j=0; j<gpu -> nbasis; j++){
                        for(int k=0; k<gpu -> gpu_basis-> maxcontract;k++){
                                printf("bin id: %i basis id: %i cfcount: %i pf id: %i pfcount: %i \n", i, j, cfweight[(i * gpu -> nbasis) + j], k, pfweight[(i * gpu -> nbasis * gpu -> gpu_basis-> maxcontract) + j*gpu -> gpu_basis-> maxcontract + k]);
                        }
                }
        }
*/
        get_primf_contraf_lists(gpu, d_gpweight, d_cfweight, d_pfweight);

	cudaMemcpy(gpweight, d_gpweight, gpu -> gpu_xcq -> npoints * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(cfweight, d_cfweight, nbins * gpu -> nbasis * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pfweight, d_pfweight, nbins * gpu -> nbasis * gpu -> gpu_basis -> maxcontract * sizeof(unsigned int), cudaMemcpyDeviceToHost);

/*	for(int i=0; i<nbins;i++){
		//unsigned int cfweight_sum =0;
		for(int j=0; j<gpu -> nbasis; j++){
			printf("bin id: %i basis id: %i cfcount: %i \n", i, j, cfweight[(i * gpu -> nbasis) + j]);
			//cfweight_sum += cfweight[ (nbins*gpu -> nbasis) + j];			
		}
		//printf("bin id: %i cfweight_sum: %i", i, cfweight_sum);
	}


	for(int i=0; i<nbins;i++){
		for(int j=0; j<gpu -> nbasis; j++){
			for(int k=0; k<gpu -> gpu_basis-> maxcontract;k++){
				printf("bin id: %i basis id: %i cfcount: %i pf id: %i pfcount: %i \n", i, j, cfweight[(i * gpu -> nbasis) + j], k, pfweight[(i * gpu -> nbasis * gpu -> gpu_basis-> maxcontract) + j*gpu -> gpu_basis-> maxcontract + k]);
			}
		}
	}


        for(int i=0;i<gpu -> gpu_xcq -> npoints;i++){
                gpweight[i] = gpu -> gpu_xcq -> gpweight -> _hostData[i];
                for(int j=0; j< gpu -> nbasis; j++){
                        cfweight[j+i * gpu -> nbasis] = gpu -> gpu_xcq -> cfweight -> _hostData[j+i * gpu -> nbasis];
                        for(int k=0; k<gpu -> gpu_basis-> maxcontract;k++){
                                pfweight[k + j * gpu -> gpu_basis-> maxcontract + i * gpu -> nbasis * gpu -> gpu_basis-> maxcontract] = gpu -> gpu_xcq -> pfweight -> _hostData[k + j * gpu -> gpu_basis-> maxcontract + i * gpu -> nbasis * gpu -> gpu_basis-> maxcontract];
                                //printf("gp: %i gpw: %i cf: %i cfw: %i pf: %i pfw: %i \n", i, gpu -> gpu_xcq -> gpweight -> _hostData[i], j, gpu -> gpu_xcq -> cfweight -> _hostData[j+i * gpu -> nbasis], k, gpu -> gpu_xcq -> pfweight -> _hostData[k + j * gpu -> gpu_basis-> maxcontract + i * gpu -> nbasis * gpu -> gpu_basis-> maxcontract]);

                        }
                }
        }
*/

        SAFE_DELETE(gpu -> gpu_xcq -> gridx);
        SAFE_DELETE(gpu -> gpu_xcq -> gridy);
        SAFE_DELETE(gpu -> gpu_xcq -> gridz);
	SAFE_DELETE(gpu->gpu_basis->sigrad2);
        SAFE_DELETE(gpu -> gpu_xcq -> bin_locator);
	cudaFree(d_gpweight);
	cudaFree(d_cfweight);
	cudaFree(d_pfweight);

        PRINTDEBUG("PRIMITIVE & BASIS FUNCTION LISTS OBTAINED")
}

#ifdef DEBUG
void print_uploaded_dft_info(){

  PRINTDEBUG("PRINTING UPLOADED DFT DATA")

  fprintf(gpu->debugFile,"Number of grid points: %i \n", gpu -> gpu_xcq -> npoints);
  fprintf(gpu->debugFile,"Bin size: %i \n", gpu -> gpu_xcq -> bin_size);
  fprintf(gpu->debugFile,"Number of bins: %i \n", gpu -> gpu_xcq -> nbins);
  fprintf(gpu->debugFile,"Number of total basis functions: %i \n", gpu -> gpu_xcq -> ntotbf);
  fprintf(gpu->debugFile,"Number of total primitive functions: %i \n", gpu -> gpu_xcq -> ntotpf);
  
  PRINTDEBUG("GRID POINTS & WEIGHTS")
  
  for(int i=0; i<gpu -> gpu_xcq -> npoints; i++){
    fprintf(gpu->debugFile,"Grid: %i x=%f y=%f z=%f sswt=%f weight=%f gatm=%i dweight_ssd=%i \n",i,
    gpu -> gpu_xcq -> gridx -> _hostData[i], gpu -> gpu_xcq -> gridy -> _hostData[i], gpu -> gpu_xcq -> gridz -> _hostData[i],
    gpu -> gpu_xcq -> sswt -> _hostData[i], gpu -> gpu_xcq -> weight -> _hostData[i], gpu -> gpu_xcq -> gatm -> _hostData[i],
    gpu -> gpu_xcq -> dweight -> _hostData[i], gpu -> gpu_xcq -> dweight_ssd -> _hostData[i]);
  }

  PRINTDEBUG("BASIS & PRIMITIVE FUNCTION LISTS")

  for(int bin_id=0; bin_id<gpu -> gpu_xcq -> nbins; bin_id++){  

    for(int i=gpu -> gpu_xcq -> basf_locator -> _hostData[bin_id]; i<gpu -> gpu_xcq -> basf_locator -> _hostData[bin_id+1] ; i++){

      for(int j=gpu -> gpu_xcq -> primf_locator -> _hostData[i]; j< gpu -> gpu_xcq -> primf_locator -> _hostData[i+1]; j++){

        fprintf(gpu->debugFile,"Bin ID= %i basf location= %i ibas= %i primf location= %i jprim= %i \n", bin_id, i, 
        gpu -> gpu_xcq -> basf -> _hostData[i], j, gpu -> gpu_xcq -> primf -> _hostData[j]);

      }

    }

  }

  PRINTDEBUG("RADIUS OF SIGNIFICANCE")

  for(int i=0; i<gpu -> nbasis; i++){
    fprintf(gpu->debugFile,"ibas=%i sigrad2=%f \n", i, gpu -> gpu_basis -> sigrad2 -> _hostData[i]);
  }

  for(int i=0; i<gpu -> nbasis; i++){
    for(int j=0; j<gpu -> gpu_basis -> maxcontract; j++){

      fprintf(gpu->debugFile,"ibas=%i jprim=%i dcoeff=%f \n",i,j, gpu -> gpu_basis -> dcoeff -> _hostData[ j + i * gpu -> gpu_basis -> maxcontract]);

    }
  }

  PRINTDEBUG("END PRINTING UPLOADED DFT DATA")

}
#endif

extern "C" void gpu_upload_dft_grid_(QUICKDouble *gridxb, QUICKDouble *gridyb, QUICKDouble *gridzb, QUICKDouble *gridb_sswt, QUICKDouble *gridb_weight, int *gridb_atm, int *bin_locator, int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter,int *gridb_count, int *nbins, int *nbtotbf, int *nbtotpf, int *isg, QUICKDouble *sigrad2){

	PRINTDEBUG("BEGIN TO UPLOAD DFT GRID")

	gpu -> gpu_xcq -> npoints	= *gridb_count;
	gpu -> gpu_xcq -> nbins		= *nbins;
	gpu -> gpu_xcq -> ntotbf	= *nbtotbf;	
	gpu -> gpu_xcq -> ntotpf	= *nbtotpf;
//	gpu -> gpu_xcq -> bin_size	= (int) (*gridb_count / *nbins);
	gpu -> gpu_cutoff -> DMCutoff   = 1E-9; //*DMCutoff;

	gpu -> gpu_xcq -> gridx	= new cuda_buffer_type<QUICKDouble>(gridxb, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gridy	= new cuda_buffer_type<QUICKDouble>(gridyb, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gridz	= new cuda_buffer_type<QUICKDouble>(gridzb, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> sswt	= new cuda_buffer_type<QUICKDouble>(gridb_sswt, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> weight	= new cuda_buffer_type<QUICKDouble>(gridb_weight, gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gatm		= new cuda_buffer_type<int>(gridb_atm, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> dweight_ssd   = new cuda_buffer_type<int>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> basf	= new cuda_buffer_type<int>(basf, gpu -> gpu_xcq -> ntotbf);
	gpu -> gpu_xcq -> primf	= new cuda_buffer_type<int>(primf, gpu -> gpu_xcq -> ntotpf);
	gpu -> gpu_xcq -> basf_locator     = new cuda_buffer_type<int>(basf_counter, gpu -> gpu_xcq -> nbins +1);
	gpu -> gpu_xcq -> primf_locator    = new cuda_buffer_type<int>(primf_counter, gpu -> gpu_xcq -> ntotbf +1);
	gpu -> gpu_basis -> sigrad2 = new cuda_buffer_type<QUICKDouble>(sigrad2, gpu->nbasis);
        gpu -> gpu_xcq -> bin_locator       = new cuda_buffer_type<int>(bin_locator, gpu -> gpu_xcq -> npoints);
        gpu -> gpu_xcq -> bin_counter       = new cuda_buffer_type<int>(bin_counter, gpu -> gpu_xcq -> nbins +1);

        for(int i=0; i< gpu -> gpu_xcq -> npoints; ++i) gpu -> gpu_xcq -> dweight_ssd -> _hostData[i] =1;

#ifdef CUDA_MPIV

        cudaEvent_t t_start, t_end;
        float t_time;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_end);
        cudaEventRecord(t_start, 0);

//        mgpu_xc_naive_distribute();
        mgpu_xc_tpbased_greedy_distribute();
//        mgpu_xc_pbased_greedy_distribute();
        mgpu_xc_repack();

        cudaEventRecord(t_end, 0);
        cudaEventSynchronize(t_end);
        cudaEventElapsedTime(&t_time, t_start, t_end);

        gpu -> timer -> t_xclb += (double) t_time / 1000;

        cudaEventDestroy(t_start);
        cudaEventDestroy(t_end);

        gpu ->gpu_sim.mpirank = gpu -> mpirank;
        gpu ->gpu_sim.mpisize = gpu -> mpisize;
#endif

	gpu -> xc_threadsPerBlock = SM_2X_XC_THREADS_PER_BLOCK;
	gpu -> gpu_xcq -> densa = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> densb = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gax = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gbx = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gay = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gby = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gaz = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> gbz = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
	gpu -> gpu_xcq -> exc = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);


	gpu -> gpu_xcq -> gridx -> Upload();
	gpu -> gpu_xcq -> gridy -> Upload();
	gpu -> gpu_xcq -> gridz -> Upload();
	gpu -> gpu_xcq -> sswt -> Upload();
	gpu -> gpu_xcq -> weight -> Upload();
	gpu -> gpu_xcq -> gatm -> Upload();
        gpu -> gpu_xcq -> dweight_ssd -> Upload();
	gpu -> gpu_xcq -> basf -> Upload();
	gpu -> gpu_xcq -> primf -> Upload();
        gpu -> gpu_xcq -> bin_locator -> Upload();
	gpu -> gpu_xcq -> basf_locator -> Upload();
	gpu -> gpu_xcq -> primf_locator -> Upload();
	gpu -> gpu_basis -> sigrad2 -> Upload();
	gpu -> gpu_xcq -> densa -> Upload(); 
	gpu -> gpu_xcq -> densb -> Upload();
	gpu -> gpu_xcq -> gax -> Upload();
	gpu -> gpu_xcq -> gbx -> Upload();
	gpu -> gpu_xcq -> gay -> Upload();
	gpu -> gpu_xcq -> gby -> Upload();
	gpu -> gpu_xcq -> gaz -> Upload();
	gpu -> gpu_xcq -> gbz -> Upload();
	gpu -> gpu_xcq -> exc -> Upload();


        gpu -> gpu_sim.npoints	= gpu -> gpu_xcq -> npoints;
        gpu -> gpu_sim.nbins	= gpu -> gpu_xcq -> nbins;
        gpu -> gpu_sim.ntotbf	= gpu -> gpu_xcq -> ntotbf;
        gpu -> gpu_sim.ntotpf	= gpu -> gpu_xcq -> ntotpf;
	gpu -> gpu_sim.bin_size = gpu -> gpu_xcq -> bin_size;
	gpu ->gpu_sim.gridx 	= gpu -> gpu_xcq -> gridx -> _devData;
	gpu ->gpu_sim.gridy 	= gpu -> gpu_xcq -> gridy -> _devData;	
	gpu ->gpu_sim.gridz 	= gpu -> gpu_xcq -> gridz -> _devData;
	gpu ->gpu_sim.sswt	= gpu -> gpu_xcq -> sswt -> _devData;
	gpu ->gpu_sim.weight 	= gpu -> gpu_xcq -> weight -> _devData;
	gpu ->gpu_sim.gatm 	= gpu -> gpu_xcq -> gatm -> _devData;
        gpu ->gpu_sim.dweight_ssd   = gpu -> gpu_xcq -> dweight_ssd -> _devData;
	gpu ->gpu_sim.basf 	= gpu -> gpu_xcq -> basf -> _devData;
	gpu ->gpu_sim.primf 	= gpu -> gpu_xcq -> primf -> _devData;
        gpu ->gpu_sim.bin_locator      = gpu -> gpu_xcq -> bin_locator -> _devData;
	gpu ->gpu_sim.basf_locator 	= gpu -> gpu_xcq -> basf_locator -> _devData;
	gpu ->gpu_sim.primf_locator 	= gpu -> gpu_xcq -> primf_locator -> _devData;
	gpu ->gpu_sim.densa     = gpu -> gpu_xcq -> densa -> _devData;
	gpu ->gpu_sim.densb     = gpu -> gpu_xcq -> densb -> _devData;
	gpu ->gpu_sim.gax     = gpu -> gpu_xcq -> gax -> _devData;
	gpu ->gpu_sim.gbx     = gpu -> gpu_xcq -> gbx -> _devData;
	gpu ->gpu_sim.gay     = gpu -> gpu_xcq -> gay -> _devData;
	gpu ->gpu_sim.gby     = gpu -> gpu_xcq -> gby -> _devData;
	gpu ->gpu_sim.gaz     = gpu -> gpu_xcq -> gaz -> _devData;
	gpu ->gpu_sim.gbz     = gpu -> gpu_xcq -> gbz -> _devData;
	gpu ->gpu_sim.exc     = gpu -> gpu_xcq -> exc -> _devData;
	gpu ->gpu_sim.sigrad2 = gpu->gpu_basis->sigrad2->_devData;
	gpu ->gpu_sim.isg = *isg;
        gpu ->gpu_sim.DMCutoff     = gpu -> gpu_cutoff -> DMCutoff;

//        upload_xc_smem();
        upload_pteval();

#ifdef DEBUG
        print_uploaded_dft_info();
#endif

	PRINTDEBUG("COMPLETE UPLOADING DFT GRID")

/*	int nblocks = (int) ((*gridb_count/SM_2X_XC_THREADS_PER_BLOCK)+1);

	test_xc_upload <<<nblocks, SM_2X_XC_THREADS_PER_BLOCK>>>();

	cudaDeviceSynchronize();
*/	
}

//-----------------------------------------------
// Reupload dft data
//-----------------------------------------------
extern "C" void gpu_reupload_dft_grid_(){

        PRINTDEBUG("BEGIN TO UPLOAD DFT GRID")

        gpu -> gpu_xcq -> gridx -> ReallocateGPU();
        gpu -> gpu_xcq -> gridy -> ReallocateGPU();
        gpu -> gpu_xcq -> gridz -> ReallocateGPU();
        gpu -> gpu_xcq -> sswt -> ReallocateGPU();
        gpu -> gpu_xcq -> weight -> ReallocateGPU();
        gpu -> gpu_xcq -> gatm -> ReallocateGPU();
        gpu -> gpu_xcq -> dweight_ssd -> ReallocateGPU();
        gpu -> gpu_xcq -> basf -> ReallocateGPU();
        gpu -> gpu_xcq -> primf -> ReallocateGPU();
        gpu -> gpu_xcq -> bin_locator -> ReallocateGPU();
        gpu -> gpu_xcq -> basf_locator -> ReallocateGPU();
        gpu -> gpu_xcq -> primf_locator -> ReallocateGPU(); 
        gpu -> gpu_xcq -> densa -> ReallocateGPU();
        gpu -> gpu_xcq -> densb -> ReallocateGPU();
        gpu -> gpu_xcq -> gax -> ReallocateGPU();
        gpu -> gpu_xcq -> gbx -> ReallocateGPU();
        gpu -> gpu_xcq -> gay -> ReallocateGPU();
        gpu -> gpu_xcq -> gby -> ReallocateGPU();
        gpu -> gpu_xcq -> gaz -> ReallocateGPU();
        gpu -> gpu_xcq -> gbz -> ReallocateGPU();
        gpu -> gpu_xcq -> exc -> ReallocateGPU();
        gpu -> gpu_basis -> sigrad2 -> ReallocateGPU();

        gpu -> gpu_xcq -> gridx -> Upload();
        gpu -> gpu_xcq -> gridy -> Upload();
        gpu -> gpu_xcq -> gridz -> Upload();
        gpu -> gpu_xcq -> sswt -> Upload();
        gpu -> gpu_xcq -> weight -> Upload();
        gpu -> gpu_xcq -> gatm -> Upload();
        gpu -> gpu_xcq -> dweight_ssd -> Upload();
        gpu -> gpu_xcq -> basf -> Upload();
        gpu -> gpu_xcq -> primf -> Upload();
        gpu -> gpu_xcq -> bin_locator -> Upload();
        gpu -> gpu_xcq -> basf_locator -> Upload();
        gpu -> gpu_xcq -> primf_locator -> Upload();
        gpu -> gpu_basis -> sigrad2 -> Upload();
        gpu -> gpu_xcq -> densa -> Upload();
        gpu -> gpu_xcq -> densb -> Upload();
        gpu -> gpu_xcq -> gax -> Upload();
        gpu -> gpu_xcq -> gbx -> Upload();
        gpu -> gpu_xcq -> gay -> Upload();
        gpu -> gpu_xcq -> gby -> Upload();
        gpu -> gpu_xcq -> gaz -> Upload();
        gpu -> gpu_xcq -> gbz -> Upload();
        gpu -> gpu_xcq -> exc -> Upload();
        gpu -> gpu_basis -> sigrad2 -> Upload();

        gpu ->gpu_sim.gridx     = gpu -> gpu_xcq -> gridx -> _devData;
        gpu ->gpu_sim.gridy     = gpu -> gpu_xcq -> gridy -> _devData;
        gpu ->gpu_sim.gridz     = gpu -> gpu_xcq -> gridz -> _devData;
        gpu ->gpu_sim.sswt      = gpu -> gpu_xcq -> sswt -> _devData;
        gpu ->gpu_sim.weight    = gpu -> gpu_xcq -> weight -> _devData;
        gpu ->gpu_sim.gatm      = gpu -> gpu_xcq -> gatm -> _devData;
        gpu ->gpu_sim.dweight_ssd   = gpu -> gpu_xcq -> dweight_ssd -> _devData;
        gpu ->gpu_sim.basf      = gpu -> gpu_xcq -> basf -> _devData;
        gpu ->gpu_sim.primf     = gpu -> gpu_xcq -> primf -> _devData;
        gpu ->gpu_sim.bin_locator      = gpu -> gpu_xcq -> bin_locator -> _devData;
        gpu ->gpu_sim.basf_locator      = gpu -> gpu_xcq -> basf_locator -> _devData;
        gpu ->gpu_sim.primf_locator     = gpu -> gpu_xcq -> primf_locator -> _devData;
        gpu ->gpu_sim.densa     = gpu -> gpu_xcq -> densa -> _devData;
        gpu ->gpu_sim.densb     = gpu -> gpu_xcq -> densb -> _devData;
        gpu ->gpu_sim.gax     = gpu -> gpu_xcq -> gax -> _devData;
        gpu ->gpu_sim.gbx     = gpu -> gpu_xcq -> gbx -> _devData;
        gpu ->gpu_sim.gay     = gpu -> gpu_xcq -> gay -> _devData;
        gpu ->gpu_sim.gby     = gpu -> gpu_xcq -> gby -> _devData;
        gpu ->gpu_sim.gaz     = gpu -> gpu_xcq -> gaz -> _devData;
        gpu ->gpu_sim.gbz     = gpu -> gpu_xcq -> gbz -> _devData;
        gpu ->gpu_sim.exc     = gpu -> gpu_xcq -> exc -> _devData;
        gpu ->gpu_sim.sigrad2 = gpu->gpu_basis->sigrad2->_devData;

        reupload_pteval();

        PRINTDEBUG("COMPLETE UPLOADING DFT GRID")

}


//-----------------------------------------------
// Delete dft device data
//-----------------------------------------------
extern "C" void gpu_delete_dft_dev_grid_(){

        PRINTDEBUG("DEALLOCATING DFT GRID")

        gpu -> gpu_xcq -> gridx -> DeleteGPU();
        gpu -> gpu_xcq -> gridy -> DeleteGPU();
        gpu -> gpu_xcq -> gridz -> DeleteGPU();
        gpu -> gpu_xcq -> sswt -> DeleteGPU();
        gpu -> gpu_xcq -> weight -> DeleteGPU();
        gpu -> gpu_xcq -> gatm -> DeleteGPU();
        gpu -> gpu_xcq -> dweight_ssd -> DeleteGPU();
        gpu -> gpu_xcq -> basf -> DeleteGPU();
        gpu -> gpu_xcq -> primf -> DeleteGPU();
        gpu -> gpu_xcq -> bin_locator -> DeleteGPU();
        gpu -> gpu_xcq -> basf_locator -> DeleteGPU();
        gpu -> gpu_xcq -> primf_locator -> DeleteGPU();
        gpu -> gpu_xcq -> densa -> DeleteGPU();
        gpu -> gpu_xcq -> densb -> DeleteGPU();
        gpu -> gpu_xcq -> gax -> DeleteGPU();
        gpu -> gpu_xcq -> gbx -> DeleteGPU();
        gpu -> gpu_xcq -> gay -> DeleteGPU();
        gpu -> gpu_xcq -> gby -> DeleteGPU();
        gpu -> gpu_xcq -> gaz -> DeleteGPU();
        gpu -> gpu_xcq -> gbz -> DeleteGPU();
        gpu -> gpu_xcq -> exc -> DeleteGPU();
        gpu -> gpu_basis -> sigrad2 -> DeleteGPU();

        delete_pteval(true);

        PRINTDEBUG("FINISHED DEALLOCATING DFT GRID")

}


extern "C" void gpu_delete_dft_grid_(){

        PRINTDEBUG("DEALLOCATING DFT GRID")

        SAFE_DELETE(gpu -> gpu_xcq -> gridx);
        SAFE_DELETE(gpu -> gpu_xcq -> gridy);
        SAFE_DELETE(gpu -> gpu_xcq -> gridz);
        SAFE_DELETE(gpu -> gpu_xcq -> sswt);
        SAFE_DELETE(gpu -> gpu_xcq -> weight);
        SAFE_DELETE(gpu -> gpu_xcq -> gatm);
        SAFE_DELETE(gpu -> gpu_xcq -> dweight_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> basf);
        SAFE_DELETE(gpu -> gpu_xcq -> primf);
        SAFE_DELETE(gpu -> gpu_xcq -> bin_locator);
        SAFE_DELETE(gpu -> gpu_xcq -> basf_locator);
        SAFE_DELETE(gpu -> gpu_xcq -> primf_locator);
        SAFE_DELETE(gpu -> gpu_xcq -> bin_counter);
	SAFE_DELETE(gpu -> gpu_xcq -> densa);
	SAFE_DELETE(gpu -> gpu_xcq -> densb);
	SAFE_DELETE(gpu -> gpu_xcq -> gax);
	SAFE_DELETE(gpu -> gpu_xcq -> gbx);
	SAFE_DELETE(gpu -> gpu_xcq -> gay);
	SAFE_DELETE(gpu -> gpu_xcq -> gby);
	SAFE_DELETE(gpu -> gpu_xcq -> gaz);
	SAFE_DELETE(gpu -> gpu_xcq -> gbz);
	SAFE_DELETE(gpu -> gpu_xcq -> exc);
	SAFE_DELETE(gpu -> gpu_basis -> sigrad2);
//        SAFE_DELETE(gpu -> gpu_xcq -> primfpbin);
#ifdef CUDA_MPIV
        SAFE_DELETE(gpu -> gpu_xcq -> mpi_bxccompute);
#endif
        delete_pteval(false);
        PRINTDEBUG("FINISHED DEALLOCATING DFT GRID")
}

void gpu_delete_sswgrad_vars(){

        PRINTDEBUG("DEALLOCATING SSWGRAD VARIABLES")

        SAFE_DELETE(gpu -> gpu_xcq -> gridx_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> gridy_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> gridz_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> exc_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> quadwt);
	SAFE_DELETE(gpu -> gpu_xcq -> uw_ssd);
        SAFE_DELETE(gpu -> gpu_xcq -> gatm_ssd);
#ifdef CUDA_MPIV
        SAFE_DELETE(gpu -> gpu_xcq -> mpi_bxccompute);
#endif

        PRINTDEBUG("FINISHED DEALLOCATING SSWGRAD VARIABLES")

}

/*Madu Manathunga 06/25/2019
Integration of libxc GPU version. The included file below contains all libxc methods
*/
#include "gpu_libxc.cu"

extern "C" void gpu_xcgrad_(QUICKDouble *grad, int* nof_functionals, int* functional_id, int* xc_polarization){

        /*The following variable will hold the number of auxilary functionals in case of
        //a hybrid functional. Otherwise, the value will be remained as the num. of functionals 
        //from input. */
        int nof_aux_functionals = *nof_functionals;

#ifdef DEBUG
        fprintf(gpu->debugFile,"Calling init_gpu_libxc.. %d %d %d \n", nof_aux_functionals, functional_id[0], *xc_polarization);
#endif
        //Madu: Initialize gpu libxc and upload information to GPU
        gpu_libxc_info** glinfo = NULL;

        if(nof_aux_functionals > 0) glinfo = init_gpu_libxc(&nof_aux_functionals, functional_id, xc_polarization);

        //libxc_cleanup(glinfo, nof_functionals);

        /*gpu -> grad = new cuda_buffer_type<QUICKDouble>(grad, 3 * gpu->natom);
        gpu -> gradULL = new cuda_buffer_type<QUICKULL>(3 * gpu->natom);
        gpu -> gpu_sim.grad =  gpu -> grad -> _devData;
        gpu -> gpu_sim.gradULL =  gpu -> gradULL -> _devData;

        for (int i = 0; i<gpu->natom * 3; i++) {

          QUICKULL valUII = (QUICKULL) (fabs ( gpu->grad->_hostData[i] * GRADSCALE));

          if ( gpu->grad->_hostData[i] <(QUICKDouble)0.0){
            valUII = 0ull - valUII;
          }

          gpu->gradULL ->_hostData[i] = valUII;
        }

        gpu -> gradULL -> Upload();
        */

        // calculate smem size
        gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

        upload_sim_to_constant_dft(gpu);

	getxc_grad(gpu, glinfo, nof_aux_functionals);

        gpu -> gradULL -> Download();

        for (int i = 0; i< 3 * gpu->natom; i++) {
          QUICKULL valULL = gpu->gradULL->_hostData[i];
          QUICKDouble valDB;

          if (valULL >= 0x8000000000000000ull) {
            valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
          }
          else
          {
            valDB  = (QUICKDouble) valULL;
          }

          gpu->grad->_hostData[i] = (QUICKDouble)valDB*ONEOVERGRADSCALE;
        }

        gpu -> grad -> DownloadSum(grad);

        delete gpu -> grad;
        delete gpu -> gradULL;
        delete gpu->gpu_calculated->dense; 
    
}


extern "C" void gpu_cleanup_(){
    SAFE_DELETE(gpu->gpu_basis->ncontract);
    SAFE_DELETE(gpu->gpu_basis->itype);
    SAFE_DELETE(gpu->gpu_basis->aexp);
    SAFE_DELETE(gpu->gpu_basis->dcoeff);
    SAFE_DELETE(gpu->gpu_basis->ncenter);
    SAFE_DELETE(gpu->gpu_basis->kstart);
    SAFE_DELETE(gpu->gpu_basis->katom);
    SAFE_DELETE(gpu->gpu_basis->kprim);
    SAFE_DELETE(gpu->gpu_basis->Ksumtype);
    SAFE_DELETE(gpu->gpu_basis->Qnumber);
    SAFE_DELETE(gpu->gpu_basis->Qstart);
    SAFE_DELETE(gpu->gpu_basis->Qfinal);
    SAFE_DELETE(gpu->gpu_basis->Qsbasis);
    SAFE_DELETE(gpu->gpu_basis->Qfbasis);
    SAFE_DELETE(gpu->gpu_basis->gccoeff);
    SAFE_DELETE(gpu->gpu_basis->cons);
    SAFE_DELETE(gpu->gpu_basis->gcexpo);
    SAFE_DELETE(gpu->gpu_basis->KLMN);
    SAFE_DELETE(gpu->gpu_basis->prim_start);
    SAFE_DELETE(gpu->gpu_basis->Xcoeff);
    SAFE_DELETE(gpu->gpu_basis->expoSum);
    SAFE_DELETE(gpu->gpu_basis->weightedCenterX);
    SAFE_DELETE(gpu->gpu_basis->weightedCenterY);
    SAFE_DELETE(gpu->gpu_basis->weightedCenterZ);
    SAFE_DELETE(gpu->gpu_calculated->distance);
    SAFE_DELETE(gpu->xyz);
    SAFE_DELETE(gpu->gpu_basis->sorted_Q);
    SAFE_DELETE(gpu->gpu_basis->sorted_Qnumber);
    SAFE_DELETE(gpu->gpu_cutoff->cutMatrix);
    SAFE_DELETE(gpu->gpu_cutoff->sorted_YCutoffIJ);
    SAFE_DELETE(gpu->gpu_cutoff->YCutoff);
    SAFE_DELETE(gpu->gpu_cutoff->cutPrim);
    
}

extern "C" void gpu_grad_(QUICKDouble* grad)
{
    PRINTDEBUG("BEGIN TO RUN GRAD")
    
    upload_sim_to_constant(gpu);
    
    PRINTDEBUG("BEGIN TO RUN KERNEL")
    
    getGrad(gpu);
    
    PRINTDEBUG("COMPLETE KERNEL")
    
    if(gpu -> gpu_sim.method == HF){
    
      gpu -> gradULL -> Download();
    
      for (int i = 0; i< 3 * gpu->natom; i++) {
        QUICKULL valULL = gpu->gradULL->_hostData[i];
        QUICKDouble valDB;
        
        if (valULL >= 0x8000000000000000ull) {
            valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
        }
        else
        {
            valDB  = (QUICKDouble) valULL;
        }
        
        gpu->grad->_hostData[i] = (QUICKDouble)valDB*ONEOVERGRADSCALE;
      }
    }
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    if(gpu -> gpu_sim.method == HF){

      gpu -> grad -> DownloadSum(grad);
    
      delete gpu -> grad;
      delete gpu -> gradULL;
      delete gpu->gpu_calculated->dense;
    }

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("DOWNLOAD GRAD",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("COMPLETE RUNNING GRAD")
}


static bool debut = true;
static bool incoreInt = true;
static ERI_entry* intERIEntry;
static int totalBuffer;

extern "C" void gpu_addint_(QUICKDouble* o, int* intindex, char* intFileName){
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    PRINTDEBUG("BEGIN TO RUN ADD INT")
    
    FILE *intFile;
    int aBuffer[BUFFERSIZE], bBuffer[BUFFERSIZE];
    QUICKDouble intBuffer[BUFFERSIZE];
    //int const bufferERI = BUFFERSIZE;
    
    int bufferPackNum = *intindex / BUFFERSIZE + 1;
    int remainingBuffer = *intindex;
    int thisBuffer = 0;
    int const streamNum = 1;
    size_t ERIRead;
    int const availableMem = 400000000/streamNum;
    int const availableERI = gpu->blocks * gpu->twoEThreadsPerBlock * (int)(availableMem/(gpu->blocks * gpu->twoEThreadsPerBlock)/sizeof(ERI_entry));
    
    int bufferIndex = 0;
    
    intFile = fopen(trim(intFileName), "rb");
    if (! intFile) {
        printf("UNABLE TO OPEN INT FILE\n");
    }
    rewind(intFile);
    
    
    cudaStream_t stream[streamNum];
    for (int i = 0; i<streamNum; i++) {
        cudaStreamCreate( &stream[i] );
    }
    
    PRINTDEBUG("BEGIN TO RUN KERNEL")
    
    upload_sim_to_constant(gpu);
    
#ifdef DEBUG
    fprintf(gpu->debugFile,"int total from addint = %i\n", *intindex);
#endif    
    
    // Now begin to allocate AO INT space
    gpu -> aoint_buffer = new cuda_buffer_type<ERI_entry>*[1];//(cuda_buffer_type<ERI_entry> **) malloc(sizeof(cuda_buffer_type<ERI_entry>*) * streamNum);
    gpu -> gpu_sim.aoint_buffer = new ERI_entry*[1];
    
    
    // Zero them out
    gpu -> aoint_buffer[0]                 = new cuda_buffer_type<ERI_entry>( availableERI, false );
    gpu -> gpu_sim.aoint_buffer[0]         = gpu -> aoint_buffer[0] -> _devData;
    
#ifdef DEBUG
    fprintf(gpu->debugFile,"Total buffer pack = %i\n", bufferPackNum);
#endif
    
    if (incoreInt && debut) {
        ERI_entry* intERIEntry_tmp  = new ERI_entry[*intindex];
        intERIEntry                 = new ERI_entry[*intindex];
        int*    ERIEntryByBasis     = new int[gpu->nbasis];
        
        for (int i = 0; i<gpu->nbasis; i++) {
            ERIEntryByBasis[i] = 0;
        }
        
        for ( int i = 0; i < bufferPackNum; i++) {
            
            if (remainingBuffer > BUFFERSIZE) {
                thisBuffer = BUFFERSIZE;
                remainingBuffer = remainingBuffer - BUFFERSIZE;
            }else{
                thisBuffer = remainingBuffer;
            }

#ifdef DEBUG            
            fprintf(gpu->debugFile," For buffer pack %i, %i Entry is read.\n", i, thisBuffer);
#endif
            
            ERIRead = fread(&aBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&bBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&intBuffer, sizeof(QUICKDouble), thisBuffer, intFile);
            
            for ( int j = 0; j < thisBuffer; j++) {
                intERIEntry_tmp[bufferIndex].IJ = aBuffer[j];
                intERIEntry_tmp[bufferIndex].KL = bBuffer[j];
                intERIEntry_tmp[bufferIndex].value = intBuffer[j];
                /*
                 int III2 = aBuffer[j] / gpu->nbasis + 1;
                 int JJJ = aBuffer[j] % gpu->nbasis + 1;
                 int KKK = bBuffer[j] / gpu->nbasis + 1;
                 int LLL = bBuffer[j] % gpu->nbasis + 1;
                 printf("%i %i %i %i %f\n", III2,JJJ,KKK,LLL, intBuffer[j]);*/
                bufferIndex ++;
                int III = aBuffer[j] / gpu->nbasis;
                ERIEntryByBasis[III] ++;
            }
        }

#ifdef DEBUG        
        for (int i = 0; i<gpu->nbasis; i++) {
            fprintf(gpu->debugFile,"for basis %i = %i\n", i, ERIEntryByBasis[i]);
        }
#endif        

        int* ERIEntryByBasisIndex = new int[gpu->nbasis];
        ERIEntryByBasisIndex[0] = 0;
        for (int i = 1; i < gpu->nbasis; i++) {
            ERIEntryByBasisIndex[i] = ERIEntryByBasisIndex[i-1] + ERIEntryByBasis[i-1] ;
        }
        
#ifdef DEBUG        
        for (int i = 0; i<gpu->nbasis; i++) {
            fprintf(gpu->debugFile,"for basis %i = %i\n", i, ERIEntryByBasisIndex[i]);
        }
#endif        

        for (int i = 0; i < bufferIndex; i++) {
            int III = intERIEntry_tmp[i].IJ / gpu->nbasis;
            intERIEntry[ERIEntryByBasisIndex[III]] = intERIEntry_tmp[i];
            ERIEntryByBasisIndex[III]++;
        }
        
        
        debut = false;
        
        totalBuffer = bufferIndex;
    }
    
    
    if (incoreInt) {
        int startingInt = 0;
        int currentInt = 0;
        for (int i = 0; i<totalBuffer; i++) {
            
            //gpu -> aoint_buffer[0] -> _hostData[currentInt].IJ    = intERIEntry[i].IJ;
            //gpu -> aoint_buffer[0] -> _hostData[currentInt].KL    = intERIEntry[i].KL;
            //gpu -> aoint_buffer[0] -> _hostData[currentInt].value = intERIEntry[i].value;
            
            currentInt++;
            
            if (currentInt >= availableERI) {
                //gpu->aoint_buffer[0]->Upload();
                //cudaMemcpy(gpu->aoint_buffer[0]->_devData, gpu->aoint_buffer[0]->_hostData, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
                cudaMemcpy(gpu->aoint_buffer[0]->_devData, intERIEntry + startingInt, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
                getAddInt(gpu, currentInt, gpu -> gpu_sim.aoint_buffer[0]);
                currentInt = 0;
                startingInt = i;
            }
        }
        
        //gpu->aoint_buffer[0]->Upload();
        //cudaMemcpy(gpu->aoint_buffer[0]->_devData, gpu->aoint_buffer[0]->_hostData, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu->aoint_buffer[0]->_devData, intERIEntry + startingInt, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
        getAddInt(gpu, currentInt, gpu -> gpu_sim.aoint_buffer[0]);
        bufferIndex = 0;
        
    }else{
        
        
        for ( int i = 0; i < bufferPackNum; i++) {
            
            if (remainingBuffer > BUFFERSIZE) {
                thisBuffer = BUFFERSIZE;
                remainingBuffer = remainingBuffer - BUFFERSIZE;
            }else{
                thisBuffer = remainingBuffer;
            }

#ifdef DEBUG            
            fprintf(gpu->debugFile," For buffer pack %i, %i Entry is read.\n", i, thisBuffer);
#endif
            
            ERIRead = fread(&aBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&bBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&intBuffer, sizeof(QUICKDouble), thisBuffer, intFile);
            
            for ( int j = 0; j < thisBuffer; j++) {
                gpu -> aoint_buffer[0] -> _hostData[bufferIndex].IJ    = aBuffer[j];
                gpu -> aoint_buffer[0] -> _hostData[bufferIndex].KL    = bBuffer[j];
                gpu -> aoint_buffer[0] -> _hostData[bufferIndex].value = intBuffer[j];
                bufferIndex ++;
                if (bufferIndex >= availableERI) {
                    
                    //cudaMemcpyAsync(gpu->aoint_buffer[0]->_hostData, gpu->aoint_buffer[0]->_devData, bufferIndex*sizeof(ERI_entry), cudaMemcpyHostToDevice, stream[0]);
                    gpu->aoint_buffer[0]->Upload();
                    getAddInt(gpu, bufferIndex, gpu -> gpu_sim.aoint_buffer[0]);
                    bufferIndex = 0;
                }
            }
            
        }
        
        
        gpu->aoint_buffer[0]->Upload();
        //cudaMemcpyAsync(gpu->aoint_buffer[0]->_hostData, gpu->aoint_buffer[0]->_devData, bufferIndex*sizeof(ERI_entry), cudaMemcpyHostToDevice, stream[0]);
        getAddInt(gpu, bufferIndex, gpu -> gpu_sim.aoint_buffer[0]);
        bufferIndex = 0;
    }
    
    for (int i = 0; i<streamNum; i++) {
        delete gpu->aoint_buffer[i];
    }
    
    PRINTDEBUG("COMPLETE KERNEL")
    gpu -> gpu_calculated -> oULL -> Download();
    
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            QUICKULL valULL = LOC2(gpu->gpu_calculated->oULL->_hostData, j, i, gpu->nbasis, gpu->nbasis);
            QUICKDouble valDB;
            
            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
        }
    }
    
    
    gpu -> gpu_calculated -> o    -> Download(o);
    
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("ADD INT",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("DELETE TEMP VARIABLES")
    
    delete gpu->gpu_calculated->o;
    delete gpu->gpu_calculated->dense;
    delete gpu->gpu_calculated->oULL;
    delete gpu->gpu_cutoff->cutMatrix;
    delete gpu->gpu_cutoff->sorted_YCutoffIJ;
    delete gpu->gpu_cutoff->YCutoff;
    delete gpu->gpu_cutoff->cutPrim;
    
    
    PRINTDEBUG("COMPLETE RUNNING ADDINT")
    
}

//-----------------------------------------------
//  core part, compute 2-e integrals
//-----------------------------------------------
extern "C" void gpu_get2e_(QUICKDouble* o)
{
    PRINTDEBUG("BEGIN TO RUN GET2E")
    
    upload_sim_to_constant(gpu);
    
    PRINTDEBUG("BEGIN TO RUN KERNEL")
    
    get2e(gpu);
 
    PRINTDEBUG("COMPLETE KERNEL")
    gpu -> gpu_calculated -> oULL -> Download();

    
    cudaMemsetAsync(gpu -> gpu_calculated -> oULL -> _devData, 0, sizeof(QUICKULL)*gpu->nbasis*gpu->nbasis);    

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            QUICKULL valULL = LOC2(gpu->gpu_calculated->oULL->_hostData, j, i, gpu->nbasis, gpu->nbasis);
            QUICKDouble valDB;
            
            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
        }
    }
    
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    gpu -> gpu_calculated -> o    -> DownloadSum(o);

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("DOWNLOAD O",time);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
    PRINTDEBUG("DELETE TEMP VARIABLES")
   
    if(gpu -> gpu_sim.method == HF){ 
      delete gpu->gpu_calculated->o;
      delete gpu->gpu_calculated->dense;
      delete gpu->gpu_calculated->oULL;
    }
        

    delete gpu->gpu_cutoff->cutMatrix;
    
    PRINTDEBUG("COMPLETE RUNNING GET2E")
}


extern "C" void gpu_getxc_(QUICKDouble* Eelxc, QUICKDouble* aelec, QUICKDouble* belec, QUICKDouble *o, int* nof_functionals, int* functional_id, int* xc_polarization)
{
    PRINTDEBUG("BEGIN TO RUN GETXC")

    /*The following variable will hold the number of auxilary functionals in case of
    //a hybrid functional. Otherwise, the value will be remained as the num. of functionals 
    //from input. */
    int nof_aux_functionals = *nof_functionals;

#ifdef DEBUG
	fprintf(gpu->debugFile, "Calling init_gpu_libxc.. %d %d %d \n", nof_aux_functionals, functional_id[0], *xc_polarization);
#endif
    //Madu: Initialize gpu libxc and upload information to GPU
    gpu_libxc_info** glinfo = NULL;
    if(nof_aux_functionals > 0) glinfo = init_gpu_libxc(&nof_aux_functionals, functional_id, xc_polarization);

    //libxc_cleanup(glinfo, nof_functionals);

    gpu -> DFT_calculated       = new cuda_buffer_type<DFT_calculated_type>(1, 1);

    QUICKULL valUII = (QUICKULL) (fabs ( *Eelxc * OSCALE + (QUICKDouble)0.5));

    if (*Eelxc<(QUICKDouble)0.0)
    {
        valUII = 0ull - valUII;
    }

    gpu -> DFT_calculated -> _hostData[0].Eelxc = valUII;

    valUII = (QUICKULL) (fabs ( *aelec * OSCALE + (QUICKDouble)0.5));

    if (*aelec<(QUICKDouble)0.0)
    {
        valUII = 0ull - valUII;
    }
    gpu -> DFT_calculated -> _hostData[0].aelec = valUII;

    valUII = (QUICKULL) (fabs ( *belec * OSCALE + (QUICKDouble)0.5));

    if (*belec<(QUICKDouble)0.0)
    {
        valUII = 0ull - valUII;
    }

    gpu -> DFT_calculated -> _hostData[0].belec = valUII;

    gpu -> DFT_calculated -> Upload();
    gpu -> gpu_sim.DFT_calculated= gpu -> DFT_calculated->_devData;

    upload_sim_to_constant_dft(gpu);
    PRINTDEBUG("BEGIN TO RUN KERNEL")

        //Madu Manathunga 07/01/2019 added libxc variable
#ifdef DEBUG
        fprintf(gpu->debugFile,"FILE: %s, LINE: %d, FUNCTION: %s, nof_aux_functionals: %d \n", __FILE__, __LINE__, __func__, nof_aux_functionals);
#endif

    getxc(gpu, glinfo, nof_aux_functionals);
    gpu -> gpu_calculated -> oULL -> Download();
    gpu -> DFT_calculated -> Download();

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            QUICKULL valULL = LOC2(gpu->gpu_calculated->oULL->_hostData, j, i, gpu->nbasis, gpu->nbasis);
            QUICKDouble valDB;

            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
        }
    }
    gpu -> gpu_calculated -> o    -> DownloadSum(o);
    QUICKULL valULL = gpu->DFT_calculated -> _hostData[0].Eelxc;
    QUICKDouble valDB;

    if (valULL >= 0x8000000000000000ull) {
        valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
    }
    else
    {
        valDB  = (QUICKDouble) valULL;
    }
    *Eelxc = (QUICKDouble)valDB*ONEOVEROSCALE;

    valULL = gpu->DFT_calculated -> _hostData[0].aelec;

    if (valULL >= 0x8000000000000000ull) {
        valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
    }
    else
    {
        valDB  = (QUICKDouble) valULL;
    }
    *aelec = (QUICKDouble)valDB*ONEOVEROSCALE;

    valULL = gpu->DFT_calculated -> _hostData[0].belec;

    if (valULL >= 0x8000000000000000ull) {
        valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
    }
    else
    {
        valDB  = (QUICKDouble) valULL;
    }
    *belec = (QUICKDouble)valDB*ONEOVEROSCALE;

    PRINTDEBUG("DELETE TEMP VARIABLES")

    delete gpu->gpu_calculated->o;
    delete gpu->gpu_calculated->dense;
    delete gpu->gpu_calculated->oULL;

}

char *trim(char *s) {
    char *ptr;
    if (!s)
        return NULL;   // handle NULL string
    if (!*s)
        return s;      // handle empty string
    for (ptr = s + strlen(s) - 1; (ptr >= s) && isspace(*ptr); --ptr);
    ptr[1] = '\0';
    return s;
}


extern "C" void gpu_aoint_(QUICKDouble* leastIntegralCutoff, QUICKDouble* maxIntegralCutoff, int* intNum, char* intFileName)
{
    PRINTDEBUG("BEGIN TO RUN AOINT")
    
    ERI_entry a;
    FILE *intFile;
    intFile = fopen(trim(intFileName), "wb");

    if (! intFile) {
#ifdef DEBUG
        fprintf(gpu->debugFile,"UNABLE TO OPEN INT FILE\n");
#endif
    }
 	
    int iBatchCount = 0;
    int const streamNum = 1;
    int const availableMem = 400000000/streamNum;
    int const availableERI = gpu->blocks * gpu->twoEThreadsPerBlock * (int)(availableMem/(gpu->blocks * gpu->twoEThreadsPerBlock)/sizeof(ERI_entry));
    QUICKULL nIntStart[30000], nIntEnd[30000], nIntSize[30000];
    int maxIntCount = 0;
    int currentCount = 0;
    int bufferInt = 0;
    int aBuffer[BUFFERSIZE], bBuffer[BUFFERSIZE];
    QUICKDouble intBuffer[BUFFERSIZE];
    
    nIntStart[0] = 0;
    
    /*
     fill up the GPU memory and if it is full, run another batch
     */
    QUICKULL intTotal = gpu -> gpu_cutoff -> sqrQshell * gpu -> gpu_cutoff -> sqrQshell;
    
    for (int i = 0; i < intTotal; i++) {
        
        int intCount = 20;
        if (currentCount + intCount < availableERI) {
            currentCount = currentCount + intCount;
        }else{
            nIntStart[iBatchCount + 1]  = i + 1;
            nIntEnd[iBatchCount]        = i;
            nIntSize[iBatchCount]       = currentCount;
            iBatchCount ++;
            currentCount = 0;
        }
        
    }
    
    // push last batch
    nIntEnd[iBatchCount]  = intTotal - 1;
    nIntSize[iBatchCount] = currentCount;
    iBatchCount++ ;
    
    for (int i = 0; i < iBatchCount; i++) {
        if (maxIntCount < nIntSize[i]) {
            maxIntCount = nIntSize[i];
        }
    }

#ifdef DEBUG    
    // List all the batches
    fprintf(gpu->debugFile,"batch count = %i\n", iBatchCount);
    fprintf(gpu->debugFile,"max int count = %i\n", maxIntCount * sizeof(ERI_entry));
    for (int i = 0; i<iBatchCount; i++) {
        fprintf(gpu->debugFile," %i from %i to %i %i\n", i, nIntStart[i], nIntEnd[i], nIntSize[i] * sizeof(ERI_entry));
    }
#endif
    
    int nBatchERICount = maxIntCount;
    
    // Now begin to allocate AO INT space
    gpu -> aoint_buffer = new cuda_buffer_type<ERI_entry>*[streamNum];//(cuda_buffer_type<ERI_entry> **) malloc(sizeof(cuda_buffer_type<ERI_entry>*) * streamNum);
    gpu -> gpu_sim.aoint_buffer = new ERI_entry*[streamNum];
    
    
    // Zero them out
    for (int i = 0; i<streamNum; i++) {
        gpu -> aoint_buffer[i]                 = new cuda_buffer_type<ERI_entry>( nBatchERICount, false );
        gpu -> gpu_sim.aoint_buffer[i]         = gpu -> aoint_buffer[i] -> _devData;
    }
    
    
    gpu -> gpu_sim.leastIntegralCutoff  = *leastIntegralCutoff;
    gpu -> gpu_sim.maxIntegralCutoff    = *maxIntegralCutoff;
    gpu -> gpu_sim.iBatchSize           = nBatchERICount;
    gpu -> intCount                     = new cuda_buffer_type<QUICKULL>(streamNum);
    gpu -> gpu_sim.intCount = gpu->intCount->_devData;
    
    upload_sim_to_constant(gpu);
    
#ifdef DEBUG
    float time_downloadERI, time_kernel, time_io;
    time_downloadERI = 0;
    time_io = 0;
    time_kernel = 0;
    cudaEvent_t start_tot,end_tot;
    cudaEventCreate(&start_tot);
    cudaEventCreate(&end_tot);
    cudaEventRecord(start_tot, 0);
	clock_t start_cpu = clock();
#endif
    
    
    cudaStream_t stream[streamNum];
    for (int i = 0; i<streamNum; i++) {
        cudaStreamCreate( &stream[i] );
    }
    
    for (int iBatch = 0; iBatch < iBatchCount; iBatch = iBatch + streamNum) {
      
#ifdef DEBUG  
        fprintf(gpu->debugFile,"batch %i start %i end %i\n", iBatch, nIntStart[iBatch], nIntEnd[iBatch]);
        
        cudaEvent_t start,end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
#endif
        for (int i = 0; i < streamNum; i++) {
            gpu -> intCount -> _hostData[i] = 0;
        }
        
        gpu -> intCount -> Upload();
        
        // calculate ERI, kernel part
        for (int i = 0; i< streamNum && iBatch + i < iBatchCount; i++) {
            getAOInt(gpu, nIntStart[iBatch + i], nIntEnd[iBatch + i], stream[i], i, gpu -> gpu_sim.aoint_buffer[i]);
        }
        
        // download ERI from GPU, this is time-consuming part, that need to be reduced
        for (int i = 0; i< streamNum && iBatch + i < iBatchCount; i++) {
            cudaMemcpyAsync(&gpu->intCount->_hostData[i], &gpu->intCount->_devData[i],  sizeof(QUICKULL), cudaMemcpyDeviceToHost, stream[i]);
            cudaMemcpyAsync(gpu->aoint_buffer[i]->_hostData, gpu->aoint_buffer[i]->_devData, gpu->intCount->_hostData[i]*sizeof(ERI_entry), cudaMemcpyDeviceToHost, stream[i]);
        }
        
        
#ifdef DEBUG
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float time;
        cudaEventElapsedTime(&time, start, end);
        PRINTUSINGTIME("KERNEL",time);
        time_kernel += time;
        cudaEventDestroy(start);
        cudaEventDestroy(end);
#endif
        
        
        
#ifdef DEBUG
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
#endif
        
        gpu -> intCount -> Download();
        
        for ( int i = 0; i<streamNum && iBatch + i < iBatchCount; i++) {
            
            cudaStreamSynchronize(stream[i]);
#ifdef DEBUG
            fprintf(gpu->debugFile,"none-sync intCount = %i\n", gpu->intCount->_hostData[i]);
#endif
            
            // write to in-memory buffer.
            for (int j = 0; j < gpu->intCount->_hostData[i]  ; j++) {
                
                a = gpu -> aoint_buffer[i] -> _hostData[j];
                if (abs(a.value) > *maxIntegralCutoff) {
                    aBuffer[bufferInt] = a.IJ;
                    bBuffer[bufferInt] = a.KL;
                    intBuffer[bufferInt] = a.value;
                    //printf("%i %i %i %18.10f\n",bufferInt, aBuffer[bufferInt], bBuffer[bufferInt], intBuffer[bufferInt]);
                    bufferInt ++;
                    if (bufferInt == BUFFERSIZE) {
                        fwrite(&aBuffer, sizeof(int), BUFFERSIZE, intFile);
                        fwrite(&bBuffer, sizeof(int), BUFFERSIZE, intFile);
                        fwrite(&intBuffer, sizeof(QUICKDouble), BUFFERSIZE, intFile);
                        bufferInt = 0;
                    }
                    
                    *intNum = *intNum + 1;
                }
                
            }
        }
#ifdef DEBUG
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        PRINTUSINGTIME("IO",time);
        time_io += time;
        cudaEventDestroy(start);
        cudaEventDestroy(end);
#endif
    }
    
    
    fwrite(&aBuffer, sizeof(int), bufferInt, intFile);
    fwrite(&bBuffer, sizeof(int), bufferInt, intFile);
    fwrite(&intBuffer, sizeof(QUICKDouble), bufferInt, intFile);
    //for (int k = 0; k<bufferInt; k++) {
    //                      printf("%i %i %i %18.10f\n",k, aBuffer[k], bBuffer[k], intBuffer[k]);
    // }
    
    bufferInt = 0;
    
    for (int i = 0; i<streamNum; i++) {
        delete gpu->aoint_buffer[i];
    }
    
#ifdef DEBUG
    
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    
    fclose(intFile);
    
#ifdef DEBUG
    float time;
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    PRINTUSINGTIME("IO FLUSHING",time);
    time_io += time;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
#ifdef DEBUG 
    fprintf(gpu->debugFile," TOTAL INT = %i \n", *intNum);
#endif
    PRINTDEBUG("END TO RUN AOINT KERNEL")
    
#ifdef DEBUG
	clock_t end_cpu = clock();
    
	float cpu_time = static_cast<float>( end_cpu - start_cpu ) / CLOCKS_PER_SEC * 1000;
	PRINTUSINGTIME("CPU TIME", cpu_time);
    
    cudaEventRecord(end_tot, 0);
    cudaEventSynchronize(end_tot);
    float time_tot = 0;
    cudaEventElapsedTime(&time_tot, start_tot, end_tot);
    PRINTUSINGTIME("KERNEL",time_kernel);
    PRINTUSINGTIME("DOWNLOAD ERI", time_downloadERI);
    PRINTUSINGTIME("IO", time_io);
    PRINTUSINGTIME("TOTAL",time_tot);
    cudaEventDestroy(start_tot);
    cudaEventDestroy(end_tot);
#endif
    
}

//-----------------------------------------------
// calculate the size of shared memory for XC code
//-----------------------------------------------
/*void upload_xc_smem(){

  // First, determine the sizes of prmitive function arrays that will go into smem. This is helpful
  // to copy data from gmem to smem. 
  gpu -> gpu_xcq -> primfpbin          = new cuda_buffer_type<int>(gpu -> gpu_xcq -> nbins);

  // Count how many primitive functions per each bin, also keep track of maximum number of basis and
  // primitive functions
  int maxbfpbin=0;
  int maxpfpbin=0;
  for(int i=0; i<gpu -> gpu_xcq -> nbins; i++){

    int nbasf = gpu -> gpu_xcq -> basf_locator -> _hostData[i+1] - gpu -> gpu_xcq -> basf_locator -> _hostData[i];
    maxbfpbin = maxbfpbin < nbasf ? nbasf : maxbfpbin;

    int tot_primfpb=0;

    for(int j=gpu -> gpu_xcq -> basf_locator -> _hostData[i]; j<gpu -> gpu_xcq -> basf_locator -> _hostData[i+1] ; j++){
      for(int k=gpu -> gpu_xcq -> primf_locator -> _hostData[j]; k< gpu -> gpu_xcq -> primf_locator -> _hostData[j+1]; k++){
        tot_primfpb++;
      }
    }
    gpu -> gpu_xcq -> primfpbin -> _hostData[i] = tot_primfpb;
    maxpfpbin = maxpfpbin < tot_primfpb ? tot_primfpb : maxpfpbin;  

  }

  // In order to avoid memory misalignements, round upto the nearest multiple of 8. 
  maxbfpbin = maxbfpbin + 8 - maxbfpbin % 8;
  maxpfpbin = maxpfpbin + 8 - maxpfpbin % 8;

  // We will store basis and primitive function indices and primitive function locations of each bin in shared memory. 
  gpu -> gpu_xcq -> smem_size = sizeof(char)*maxpfpbin + sizeof(short)*maxbfpbin + sizeof(int)*(maxbfpbin+8);

  gpu ->gpu_sim.maxbfpbin = maxbfpbin;
  gpu ->gpu_sim.maxpfpbin = maxpfpbin;

  gpu -> gpu_xcq -> primfpbin -> Upload();
  gpu ->gpu_sim.primfpbin     = gpu -> gpu_xcq -> primfpbin -> _devData;

  printf("Max number of basis functions: %i primitive functions: %i smem size: %i \n", maxbfpbin, maxpfpbin, gpu -> gpu_xcq -> smem_size); 

  for(int i=0; i<gpu -> gpu_xcq -> nbins;i++) 
    if(gpu -> gpu_xcq -> primfpbin -> _hostData[i] >= maxpfpbin) printf("bin_id= %i nprimf= %i \n", i, gpu -> gpu_xcq -> primfpbin -> _hostData[i]);

}*/

//-----------------------------------------------
// checks available gmem and upload temporary large arrays 
// to store values and gradients of basis functions
//-----------------------------------------------
void upload_pteval(){

  gpu ->gpu_sim.prePtevl = false;

/*  // compute available amount of global memory
  size_t free, total;

  cudaMemGetInfo( &free, &total );
  printf("Total GMEM= %lli Free= %lli \n", total,free);  

  // calculate the size of an array
  int count=0;
  unsigned int phi_loc[gpu -> gpu_xcq -> npoints];
  int oidx=0;
  int bffb = gpu -> gpu_xcq -> basf_locator -> _hostData[1] - gpu -> gpu_xcq -> basf_locator -> _hostData[0];

  for(int i=0; i < gpu -> gpu_xcq -> npoints; i++){
    phi_loc[i]=count;
    int nidx=gpu -> gpu_xcq -> bin_locator -> _hostData[i];
    if(nidx != oidx) bffb = gpu -> gpu_xcq -> basf_locator -> _hostData[nidx+1] - gpu -> gpu_xcq -> basf_locator -> _hostData[nidx];
    count += bffb;     
  }

  // amount of memory in bytes for 4 such arrays
  size_t reqMem = count * 32;

  // estimate memory for future needs, 6 nbasis * nbasis 2D arrays of double type
  size_t estMem = gpu->nbasis * gpu->nbasis * 48 + gpu -> gpu_xcq -> npoints * 4;

  printf("Size of each pteval array= %lli Required memory for pteval= %lli Total avail= %lli\n", count, reqMem,free-estMem);

  if( reqMem < (free-estMem) ){
    gpu ->gpu_sim.prePtevl = true; 
    gpu -> gpu_xcq -> phi_loc          = new cuda_buffer_type<unsigned int>(gpu -> gpu_xcq -> npoints);
    gpu -> gpu_xcq -> phi          = new cuda_buffer_type<QUICKDouble>(count);
    gpu -> gpu_xcq -> dphidx       = new cuda_buffer_type<QUICKDouble>(count);
    gpu -> gpu_xcq -> dphidy       = new cuda_buffer_type<QUICKDouble>(count);
    gpu -> gpu_xcq -> dphidz       = new cuda_buffer_type<QUICKDouble>(count);

    memcpy(gpu -> gpu_xcq -> phi_loc -> _hostData, &phi_loc, sizeof(unsigned int)*gpu -> gpu_xcq -> npoints);

    gpu -> gpu_xcq -> phi_loc -> Upload();
    gpu -> gpu_xcq -> phi -> Upload();
    gpu -> gpu_xcq -> dphidx -> Upload();
    gpu -> gpu_xcq -> dphidy -> Upload();
    gpu -> gpu_xcq -> dphidz -> Upload();

    gpu -> gpu_sim.phi_loc = gpu -> gpu_xcq -> phi_loc -> _devData;
    gpu -> gpu_sim.phi = gpu -> gpu_xcq -> phi -> _devData;   
    gpu -> gpu_sim.dphidx = gpu -> gpu_xcq -> dphidx -> _devData;   
    gpu -> gpu_sim.dphidy = gpu -> gpu_xcq -> dphidy -> _devData;
    gpu -> gpu_sim.dphidz = gpu -> gpu_xcq -> dphidz -> _devData;

    upload_sim_to_constant_dft(gpu);

    getpteval(gpu);

    gpu -> gpu_xcq -> phi -> Download();
    gpu -> gpu_xcq -> dphidx -> Download();
    gpu -> gpu_xcq -> dphidy -> Download();
    gpu -> gpu_xcq -> dphidz -> Download();

  }
*/

}

//-----------------------------------------------
// Check memory and reupload for xc grad calculation
// if there is enough space
//-----------------------------------------------
void reupload_pteval(){

  gpu ->gpu_sim.prePtevl = false;

/*  // compute available amount of global memory
  size_t free, total;

  cudaMemGetInfo( &free, &total );
  printf("Total GMEM= %lli Free= %lli \n", total,free);

  // amount of memory in bytes for 4 such arrays
  size_t reqMem = gpu -> gpu_xcq -> phi -> _length * 32 + gpu -> gpu_xcq -> npoints * 4;

  // estimate memory for future needs, 2 nbasis * nbasis 2D arrays of double type
  // and 2 grad arrays of double type
  size_t estMem = gpu->nbasis * gpu->nbasis * 16 + gpu->natom * 48;

  printf("Required memory for pteval= %lli Total avail= %lli\n", reqMem,free-estMem);

  if( reqMem < (free-estMem) ){
    gpu ->gpu_sim.prePtevl = true;

    gpu -> gpu_xcq -> phi_loc -> ReallocateGPU();
    gpu -> gpu_xcq -> phi -> ReallocateGPU();
    gpu -> gpu_xcq -> dphidx -> ReallocateGPU();
    gpu -> gpu_xcq -> dphidy -> ReallocateGPU();   
    gpu -> gpu_xcq -> dphidz -> ReallocateGPU();

    gpu -> gpu_xcq -> phi_loc -> Upload();
    gpu -> gpu_xcq -> phi -> Upload();
    gpu -> gpu_xcq -> dphidx -> Upload();
    gpu -> gpu_xcq -> dphidy -> Upload();
    gpu -> gpu_xcq -> dphidz -> Upload();

    gpu -> gpu_sim.phi_loc = gpu -> gpu_xcq -> phi_loc -> _devData;
    gpu -> gpu_sim.phi = gpu -> gpu_xcq -> phi -> _devData;
    gpu -> gpu_sim.dphidx = gpu -> gpu_xcq -> dphidx -> _devData;
    gpu -> gpu_sim.dphidy = gpu -> gpu_xcq -> dphidy -> _devData;
    gpu -> gpu_sim.dphidz = gpu -> gpu_xcq -> dphidz -> _devData;
  }
*/
}

//-----------------------------------------------
// Delete both device and host pteval data
//-----------------------------------------------
void delete_pteval(bool devOnly){

/*    if(gpu ->gpu_sim.prePtevl == true){

      if(devOnly){

        gpu -> gpu_xcq -> phi_loc -> DeleteGPU();
        gpu -> gpu_xcq -> phi -> DeleteGPU();
        gpu -> gpu_xcq -> dphidx -> DeleteGPU();
        gpu -> gpu_xcq -> dphidy -> DeleteGPU();
        gpu -> gpu_xcq -> dphidz -> DeleteGPU();

      }else{

        SAFE_DELETE(gpu -> gpu_xcq -> phi_loc);
        SAFE_DELETE(gpu -> gpu_xcq -> phi);
        SAFE_DELETE(gpu -> gpu_xcq -> dphidx);
        SAFE_DELETE(gpu -> gpu_xcq -> dphidy);
        SAFE_DELETE(gpu -> gpu_xcq -> dphidz);

      }
    }
*/
}


