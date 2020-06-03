/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 04/29/2020                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains methods required for QUICK multi GPU      !
  ! implementation. Additional changes have been made in the gpu_type.h !
  ! where we define variables holding device information.               !
  !                                                                     ! 
  !---------------------------------------------------------------------!
*/
/*#include <stdio.h>
#include <string>
#include "gpu.h"
#include <ctime>
#include <time.h>
*/
#ifdef CUDA_MPIV

#include "mgpu.h"
#include "mpi.h"
//-----------------------------------------------
// Get information about available GPUs and prepare
// a list of usable GPUs. Only master cpu runs this.
//-----------------------------------------------

extern "C" void mgpu_query_(int *mpisize)
{

    PRINTDEBUG("BEGIN QUERYING DEVICES")
    int gpuCount = 0;           // Total number of cuda devices available
    size_t minMem = 8000000000; // Threshold  memory (in bytes) for device selection criteria
    cudaError_t status;
    bool isZeroID = false;

    status = cudaGetDeviceCount(&gpuCount);

    if(gpuCount == 0){
        PRINTERROR(status,"cudaGetDeviceCount gpu_init failed!");
        cudaDeviceReset();
        exit(-1);
    }else if(gpuCount == 1){
        isZeroID = true;
        gpuCount = *mpisize; 
    }

    printf("Number of gpus %i \n", gpuCount);

    int tmp_gpu_dev_id[gpuCount];        // Temporarily holds device IDs 
    unsigned int idx_tmp_gpu_dev_id = 0; // Array index counter for tmp_gpu_dev_id
    int devID = 0 ;

    for(int i=0;i<gpuCount;i++){

        cudaDeviceProp devProp;

        if(isZeroID == true){
            cudaGetDeviceProperties(&devProp, 0);
            devID = 0;
        }else{
            cudaGetDeviceProperties(&devProp, i);
            devID = i;
        }
        // Should be fixed to select based on sm value used during the compilation
        // For now, we select Volta and Turing devices only

        if((devProp.major == 7) && (devProp.minor >= 0) && (devProp.totalGlobalMem > minMem)){
            validDevCount++;
            tmp_gpu_dev_id[idx_tmp_gpu_dev_id] = devID;
            idx_tmp_gpu_dev_id++;
        }

    }

    if (validDevCount != gpuCount && validDevCount < *mpisize) {
        printf("MPISIZE AND NUMBER OF AVAILABLE GPUS MUST BE THE SAME.\n");
        gpu_shutdown_();
        exit(-1);
    }else{

        if(validDevCount > *mpisize){validDevCount = *mpisize;}

        // Store usable device IDs to broadcast to slaves

        gpu_dev_id = (int*) malloc(validDevCount*sizeof(int));

        for(int i=0; i<validDevCount; i++){
            gpu_dev_id[i] = tmp_gpu_dev_id[i];
        }

    }
    

    PRINTDEBUG("END QUERYING DEVICES")

    return;
}

//-----------------------------------------------
// create gpu class
//-----------------------------------------------
void mgpu_startup(int mpirank)
{
        PRINTDEBUG("BEGIN TO WARM UP")
#ifdef DEBUG
    if(mpirank == 0){
        debugFile = fopen("DEBUG", "w+");
    }
#endif
    gpu = new gpu_type;
        PRINTDEBUG("CREATE NEW GPU")
}

//-----------------------------------------------
// Finalize the devices
//-----------------------------------------------
extern "C" void mgpu_shutdown_(void)
{ 
    PRINTDEBUG("BEGIN TO SHUTDOWN DEVICES")

#ifdef DEBUG
    if(gpu -> mpirank == 0){
        fclose(debugFile);
    }
#endif

    delete gpu;
    cudaDeviceReset();
    free(gpu_dev_id);

    PRINTDEBUG("END DEVICE SHUTDOWN")    
}
//-----------------------------------------------
// Initialize the devices
//-----------------------------------------------
extern "C" void mgpu_init_(int *mpirank, int *mpisize)
{

    PRINTDEBUG("BEGIN MULTI GPU INITIALIZATION")

    int device = -1;
    cudaError_t status;
    cudaDeviceProp deviceProp;

    // Broadcast device information to slaves
    MPI_Bcast(&validDevCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(*mpirank != 0){
        gpu_dev_id = (int*) malloc(validDevCount*sizeof(int)); 
    }

    MPI_Bcast(gpu_dev_id, validDevCount, MPI_INT, 0, MPI_COMM_WORLD);

    printf("mpirank %i mpisize %i validDevCount %i \n", *mpirank, *mpisize, validDevCount);

    for(int i=0; i<validDevCount; i++){
        printf("mpirank %i %i \n", *mpirank, gpu_dev_id[i]);
    }

    // Each node starts up GPUs
    mgpu_startup(*mpirank);

    gpu -> mpirank = *mpirank;
    gpu -> mpisize = *mpisize;

    device = gpu_dev_id[gpu -> mpirank];

    gpu -> gpu_dev_id = device;

    status = cudaSetDevice(device);
    cudaGetDeviceProperties(&deviceProp, device);
    PRINTERROR(status, "cudaSetDevice gpu_init failed!");
    cudaDeviceSynchronize();

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    size_t val;

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
    printf("mpirank: %i Stack size limit:    %zu\n", gpu -> mpirank,val);

    cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize);
    printf("mpirank: %i Printf fifo limit:   %zu\n", gpu -> mpirank,val);

    cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize);
    printf("mpirank: %i Heap size limit:     %zu\n", gpu -> mpirank,val);

    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
    printf("mpirank: %i New Stack size limit:    %zu\n", gpu -> mpirank,val);

    gpu->blocks = deviceProp.multiProcessorCount;

    gpu -> sm_version               = SM_2X;
    gpu -> threadsPerBlock          = SM_2X_THREADS_PER_BLOCK;
    gpu -> twoEThreadsPerBlock      = SM_2X_2E_THREADS_PER_BLOCK;
    gpu -> XCThreadsPerBlock        = SM_2X_XC_THREADS_PER_BLOCK;
    gpu -> gradThreadsPerBlock      = SM_2X_GRAD_THREADS_PER_BLOCK;

    PRINTDEBUG("FINISH MULTI GPU INITIALIZATION")

    return;
}

//--------------------------------------------------------
// Method to distribute sorted shell information among nodes  
//--------------------------------------------------------
extern "C" void mgpu_distribute_qshell_(int *mpi_qshell, int *mpi_qshelln){    

    int qshellpn=(int)(gpu->gpu_basis->Qshell)/gpu->mpisize;

    for(int i=0;i<gpu->mpisize+1;i++){
        mpi_qshelln[i]=i*qshellpn;
    }

    if((gpu->gpu_basis->Qshell) % gpu->mpisize != 0){
        mpi_qshelln[gpu->mpisize] +=1;
    }
    
    printf("mpi_qshelln: %i %i %i %i \n",mpi_qshelln[0],mpi_qshelln[1],mpi_qshelln[2],mpi_qshelln[3]);

}

//--------------------------------------------------------
// Method to upload qshell work load  
//--------------------------------------------------------
extern "C" void mgpu_upload_qshell_(int *mpi_qshell, int *mpi_qshelln){

    PRINTDEBUG("UPLOADING QSHELL SETUP FOR MULTI GPU")

    gpu -> gpu_basis -> mpi_qshelln  = new cuda_buffer_type<int>(mpi_qshelln, gpu->mpisize+1);
    gpu -> gpu_basis -> mpi_qshelln -> Upload();
    gpu -> gpu_sim.mpi_qshelln = gpu -> gpu_basis -> mpi_qshelln -> _devData;

    PRINTDEBUG("END UPLOADING QSHELL SETUP FOR MULTI GPU")

}

//--------------------------------------------------------
// Method to send number of qshells to f90 side  
//--------------------------------------------------------
extern "C" void mgpu_get_nqshell_(int *nqshell){

    *nqshell = gpu->gpu_basis->Qshell;

}

//--------------------------------------------------------
// Method to upload distributed qshell indices
//--------------------------------------------------------
extern "C" void mgpu_upload_arr_bsd_qshell_(int *mpi_qshell, int *mpi_qshelln){

    for(int i=0;i<mpi_qshell[gpu->mpirank];i++){
        printf("Distributed Qindex: %i %i %i \n ",gpu->mpirank,i,mpi_qshelln[(gpu->gpu_basis->Qshell * gpu->mpirank) + i]);
    }

}

//--------------------------------------------------------
// Method to upload shell and basis information for mpi  
//--------------------------------------------------------

extern "C" void mgpu_upload_basis_setup_(int *mpi_jshelln, int* mpi_jshell, int* mpi_nbasisn, int* mpi_nbasis){

    PRINTDEBUG("UPLOADING MULTI GPU BASIS SETUP")
    printf("mgpu_upload_basis_setup %i %i %i \n ", gpu->mpisize,gpu->jshell,gpu->nbasis);   
    printf("mgpu_upload_basis_setup %i %i %i %i \n ", gpu->mpirank,mpi_jshelln[0],mpi_jshelln[1],mpi_jshelln[2]);
 

    gpu -> gpu_basis -> mpi_jshelln = new cuda_buffer_type<int>(mpi_jshelln, gpu->mpisize);
    gpu -> gpu_basis -> mpi_jshell  = new cuda_buffer_type<int>(mpi_jshell, gpu->mpisize, gpu->jshell);
    gpu -> gpu_basis -> mpi_nbasisn = new cuda_buffer_type<int>(mpi_nbasisn, gpu->mpisize);
    gpu -> gpu_basis -> mpi_nbasis  = new cuda_buffer_type<int>(mpi_nbasis, gpu->mpisize, gpu->nbasis);

    gpu -> gpu_basis -> mpi_jshelln -> Upload();
    gpu -> gpu_basis -> mpi_jshell  -> Upload();
    gpu -> gpu_basis -> mpi_nbasisn -> Upload();
    gpu -> gpu_basis -> mpi_nbasis  -> Upload();

    gpu -> gpu_sim.mpi_jshelln = gpu -> gpu_basis -> mpi_jshelln -> _devData;
    gpu -> gpu_sim.mpi_jshell  = gpu -> gpu_basis -> mpi_jshell  -> _devData;
    gpu -> gpu_sim.mpi_nbasisn = gpu -> gpu_basis -> mpi_nbasisn -> _devData;
    gpu -> gpu_sim.mpi_nbasis  = gpu -> gpu_basis -> mpi_nbasis  -> _devData;

    // Following assignments are more appropriate inside a function that performs
    // initial setup. Lets leave them for the time being. 
    gpu -> gpu_sim.mpirank = gpu -> mpirank;
    gpu -> gpu_sim.mpisize = gpu -> mpisize;
    gpu -> gpu_sim.nqshell = gpu -> gpu_basis -> Qshell;

    PRINTDEBUG("END UPLOADING MULTI GPU BASIS SETUP")

}

//--------------------------------------------------------
// Method to upload shell and basis information for mpi  
//--------------------------------------------------------

extern "C" void mgpu_delete_mpi_setup_(){

    PRINTDEBUG("DELETING MULTI GPU BASIS SETUP")

    SAFE_DELETE(gpu -> gpu_basis -> mpi_jshelln);
    SAFE_DELETE(gpu -> gpu_basis -> mpi_jshell); 
    SAFE_DELETE(gpu -> gpu_basis -> mpi_nbasisn);
    SAFE_DELETE(gpu -> gpu_basis -> mpi_nbasis); 

    PRINTDEBUG("FINISH DELETING MULTI GPU BASIS SETUP")

}

//--------------------------------------------------------
// Method to distribute sorted shell information among nodes  
//--------------------------------------------------------
void mgpu_eri_greedy_distribute(){

    // Total number of items to distribute
    int nitems=gpu->gpu_cutoff->sqrQshell;

    // Array to store total number of items each core would have
    int tot_pcore[gpu->mpisize];

    // Save shell indices for each core
    int2 mpi_qidx[gpu->mpisize][nitems];

    // Keep track of primitive count
    int2 mpi_pidx[gpu->mpisize][nitems];

    // Save a set of flags unique to each core, these will be uploaded 
    // to GPU by responsible cores
    char mpi_flags[gpu->mpisize][nitems];

    // Keep track of shell type
    int2 qtypes[gpu->mpisize][nitems];

    // Keep track of total primitive value of each core
    int tot_pval[gpu->mpisize]; 

    // Keep track of how many shell types each core has
    // ss, sp, sd, ps, pp, pd, dd, dp, dd
    int qtype_pcore[gpu->mpisize][16];

    //set arrays to zero
    memset(tot_pcore,0, sizeof(int)*gpu->mpisize);
    memset(mpi_qidx,0,sizeof(int2)*gpu->mpisize*nitems);
    memset(mpi_pidx,0,sizeof(int2)*gpu->mpisize*nitems);
    memset(mpi_flags,0,sizeof(char)*gpu->mpisize*nitems);
    memset(qtypes,0,sizeof(int2)*gpu->mpisize*nitems);
    memset(tot_pval,0,sizeof(int)*gpu->mpisize);
    memset(qtype_pcore,0,sizeof(int2)*gpu->mpisize*16);

    printf(" Greedy distribute sqrQshells= %i number of GPUs= %i \n", nitems, gpu->mpisize);

    int q1_idx, q2_idx, q1, q2, p1, p2, psum, minp, min_core;
    // Helps to store shell types per each core
    int a=0;

    // Sort s,p,d for the time being, increase the value by one to facilitate sorting
    for(int q1_typ=0; q1_typ<4; q1_typ++){
        for(int q2_typ=0; q2_typ<4; q2_typ++){

            //Go through items
            for (int i = 0; i<nitems; i++) {  

                // Get the shell type
                q1     = gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x];
                q2     = gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y];
 
                // Check if the picked shell types match currently interested shell types              
                if(q1 == q1_typ && q2 == q2_typ){
                    
                    // Find out the core with least number of primitives of the current shell types
                    min_core = 0;       // Assume master has the lowest number of primitives
                    minp = tot_pval[0]; // Set master's primitive count as the lowest
                    for(int impi=0; impi<gpu->mpisize;impi++){
                        if(minp > tot_pval[impi]){
                            minp = tot_pval[impi];
                            min_core = impi;
                        }
                    }

                    // Store the primitive value in the total primitive value counter
                    p1 = gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x]];
                    p2 = gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y]]; 
                    psum=p1+p2;
                    tot_pval[min_core] += psum;

                    //Get the q indices  
                    q1_idx = gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].x];                    
                    q2_idx = gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ ->_hostData[i].y];

                    //Assign the indices for corresponding core
                    mpi_qidx[min_core][tot_pcore[min_core]].x = q1_idx;
                    mpi_qidx[min_core][tot_pcore[min_core]].y = q2_idx;

                    // Save the flag
                    mpi_flags[min_core][i] = 1;

                    // Store shell types for debugging
                    qtype_pcore[min_core][a] +=1;
                    
                    //Store primitve number for debugging
                    mpi_pidx[min_core][tot_pcore[min_core]].x = p1;
                    mpi_pidx[min_core][tot_pcore[min_core]].y = p2;

                    // Store the Qshell type for debugging
                    qtypes[min_core][tot_pcore[min_core]].x = q1;
                    qtypes[min_core][tot_pcore[min_core]].y = q2;                    
                   
                    // Increase the counter for minimum core
                    tot_pcore[min_core] += 1;
                }                
            }

            // Reset the primitive counter for current shell type 
            memset(tot_pval,0,sizeof(int)*gpu->mpisize);
            a++;
        }
    }

    // Print information for debugging
    for(int impi=0; impi<gpu->mpisize; impi++){
        for(int icount=0; icount<tot_pcore[impi]; icount++){
            printf(" Greedy Distribute GPU: %i Qindex= %i %i Qtype= %i %i Prim= %i %i \n ",impi, mpi_qidx[impi][icount].x, mpi_qidx[impi][icount].y, \
            qtypes[impi][icount].x, qtypes[impi][icount].y, mpi_pidx[impi][icount].x, mpi_pidx[impi][icount].y);
        }
    }

    for(int impi=0; impi<gpu->mpisize;impi++){
        printf(" Greedy Distribute GPU: %i ss= %i sp= %i sd= %i sf= %i ps= %i pp= %i pd= %i pf= %i ds= %i dp= %i dd= %i df= %i fs= %i fp=%i fd=%i ff=%i \n",impi, qtype_pcore[impi][0], \
        qtype_pcore[impi][1], qtype_pcore[impi][2], qtype_pcore[impi][3], qtype_pcore[impi][4], qtype_pcore[impi][5], \
        qtype_pcore[impi][6], qtype_pcore[impi][7], qtype_pcore[impi][8], qtype_pcore[impi][9], qtype_pcore[impi][10],\
        qtype_pcore[impi][11], qtype_pcore[impi][12], qtype_pcore[impi][13], qtype_pcore[impi][14], qtype_pcore[impi][15]);
    }

    printf(" Greedy Distribute GPU: %i Total shell pairs for this GPU= %i \n", gpu -> mpirank, tot_pcore[gpu -> mpirank]);

    // Upload the flags to GPU
    gpu -> gpu_basis -> mpi_bcompute = new cuda_buffer_type<char>(nitems);

    memcpy(gpu -> gpu_basis -> mpi_bcompute -> _hostData, &mpi_flags[gpu->mpirank][0], sizeof(char)*nitems);

    gpu -> gpu_basis -> mpi_bcompute -> Upload();
    gpu -> gpu_sim.mpi_bcompute  = gpu -> gpu_basis -> mpi_bcompute  -> _devData;

}

//--------------------------------------------------------
// Method to delete mpi_flags from the GPU
//--------------------------------------------------------

/*extern "C" void mgpu_delete_mpi_setup_(){

    PRINTDEBUG("DELETING MULTI GPU BASIS SETUP")

    SAFE_DELETE(gpu -> gpu_basis -> mpi_bcompute);

    PRINTDEBUG("FINISH DELETING MULTI GPU BASIS SETUP")

}
*/

//--------------------------------------------------------
// Methods passing gpu information to f90 side for printing
//--------------------------------------------------------

extern "C" void mgpu_get_dev_count_(int *gpu_dev_count){
    *gpu_dev_count = validDevCount;
}

extern "C" void mgpu_get_device_info_(int *rank, int* dev_id,int* gpu_dev_mem,
                                     int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv)
{
    cudaDeviceProp prop;
    size_t device_mem;

    *dev_id = gpu_dev_id[*rank];

    cudaGetDeviceProperties(&prop,*dev_id);
    device_mem = (prop.totalGlobalMem/(1024*1024));
    *gpu_dev_mem = (int) device_mem;
    *gpu_num_proc = (int) (prop.multiProcessorCount);
    *gpu_core_freq = (double) (prop.clockRate * 1e-6f);
    strcpy(gpu_dev_name,prop.name);
    *name_len = strlen(gpu_dev_name);
    *majorv = prop.major;
    *minorv = prop.minor;

}
#endif
