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

#ifdef CUDA_MPIV

#include "string.h"

//-----------------------------------------------
// Query the availability of devices.
//-----------------------------------------------

extern "C" void mgpu_query_(int* mpisize, int *mpirank, int *mgpu_id)
{

    int gpuCount = 0;           // Total number of cuda devices available
    size_t minMem = 8000000000; // Threshold  memory (in bytes) for device selection criteria
    cudaError_t status;

    status = cudaGetDeviceCount(&gpuCount);

    if(gpuCount == 0){
        printf("Error: Process %d couldnt find a GPU. Make sure there are enough plugged in GPUs. \n", *mpirank);
        cudaDeviceReset();
        exit(-1);
    }/*else if(gpuCount < *mpisize){
        printf("Error: Number of launched processes is greater than the available number of GPUs. Please relaunch with lower number of processes. \n");
        cudaDeviceReset();
        exit(-1);
    }*/

    int devID = *mpirank % gpuCount;
    cudaDeviceProp devProp;
    status = cudaGetDeviceProperties(&devProp, devID); 

    

    if((devProp.major < 3) || (devProp.totalGlobalMem < minMem)){
      printf("Error: GPU assigned for process %d is too old or already in use. \n", *mpirank);
      cudaDeviceReset();
      exit(-1);
    }

    *mgpu_id = devID;

    return;
}

//-----------------------------------------------
// create gpu class
//-----------------------------------------------
void mgpu_startup(int mpirank)
{

#if defined DEBUG || defined DEBUGTIME
    char fname[16];
    sprintf(fname, "debug.cuda.%i", mpirank);    

    debugFile = fopen(fname, "w+");
#endif

    PRINTDEBUGNS("BEGIN TO WARM UP")

    gpu = new gpu_type;

#if defined DEBUG || defined DEBUGTIME
    gpu->debugFile = debugFile;
#endif

    PRINTDEBUG("CREATE NEW GPU")
}

//-----------------------------------------------
// Finalize the devices
//-----------------------------------------------
extern "C" void mgpu_shutdown_(void)
{ 

    PRINTDEBUG("BEGIN TO SHUTDOWN DEVICES")

    delete gpu;
    cudaDeviceReset();

    PRINTDEBUGNS("END DEVICE SHUTDOWN")    

#if defined DEBUG || defined DEBUGTIME
    fclose(debugFile);
#endif

}
//-----------------------------------------------
// Initialize the devices
//-----------------------------------------------
extern "C" void mgpu_init_(int *mpirank, int *mpisize, int *device)
{

    cudaError_t status;
    cudaDeviceProp deviceProp;

    // Each node starts up GPUs
    mgpu_startup(*mpirank);

    PRINTDEBUG("BEGIN MULTI GPU INITIALIZATION")

    gpu -> mpirank = *mpirank;
    gpu -> mpisize = *mpisize;
    gpu -> gpu_dev_id = *device;

#ifdef DEBUG
    fprintf(gpu->debugFile,"mpirank %i mpisize %i dev_id %i \n", *mpirank, *mpisize, *device);
#endif

    status = cudaSetDevice(gpu -> gpu_dev_id);
    cudaGetDeviceProperties(&deviceProp, gpu -> gpu_dev_id);
    PRINTERROR(status, "cudaSetDevice gpu_init failed!");
    cudaDeviceSynchronize();

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    size_t val;

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"mpirank: %i Stack size limit:    %zu\n", gpu -> mpirank,val);
#endif

    cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"mpirank: %i Printf fifo limit:   %zu\n", gpu -> mpirank,val);
#endif

    cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"mpirank: %i Heap size limit:     %zu\n", gpu -> mpirank,val);
#endif

    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
#ifdef DEBUG
    fprintf(gpu->debugFile,"mpirank: %i New Stack size limit:    %zu\n", gpu -> mpirank,val);
#endif

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

#ifdef DEBUG
    fprintf(gpu->debugFile," Greedy distribute sqrQshells= %i number of GPUs= %i \n", nitems, gpu->mpisize);
#endif

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

#ifdef DEBUG
    // Print information for debugging
    for(int impi=0; impi<gpu->mpisize; impi++){
        for(int icount=0; icount<tot_pcore[impi]; icount++){
            fprintf(gpu->debugFile," Greedy Distribute GPU: %i Qindex= %i %i Qtype= %i %i Prim= %i %i \n ",impi, mpi_qidx[impi][icount].x, mpi_qidx[impi][icount].y, \
            qtypes[impi][icount].x, qtypes[impi][icount].y, mpi_pidx[impi][icount].x, mpi_pidx[impi][icount].y);
        }
    }

    for(int impi=0; impi<gpu->mpisize;impi++){
        fprintf(gpu->debugFile," Greedy Distribute GPU: %i ss= %i sp= %i sd= %i sf= %i ps= %i pp= %i pd= %i pf= %i ds= %i dp= %i dd= %i df= %i fs= %i fp=%i fd=%i ff=%i \n",impi, qtype_pcore[impi][0], \
        qtype_pcore[impi][1], qtype_pcore[impi][2], qtype_pcore[impi][3], qtype_pcore[impi][4], qtype_pcore[impi][5], \
        qtype_pcore[impi][6], qtype_pcore[impi][7], qtype_pcore[impi][8], qtype_pcore[impi][9], qtype_pcore[impi][10],\
        qtype_pcore[impi][11], qtype_pcore[impi][12], qtype_pcore[impi][13], qtype_pcore[impi][14], qtype_pcore[impi][15]);
    }

    fprintf(gpu->debugFile," Greedy Distribute GPU: %i Total shell pairs for this GPU= %i \n", gpu -> mpirank, tot_pcore[gpu -> mpirank]);
#endif

    // Upload the flags to GPU
    gpu -> gpu_basis -> mpi_bcompute = new cuda_buffer_type<char>(nitems);

    memcpy(gpu -> gpu_basis -> mpi_bcompute -> _hostData, &mpi_flags[gpu->mpirank][0], sizeof(char)*nitems);

    gpu -> gpu_basis -> mpi_bcompute -> Upload();
    gpu -> gpu_sim.mpi_bcompute  = gpu -> gpu_basis -> mpi_bcompute  -> _devData;

}

//--------------------------------------------------------
// Function to distribute XC quadrature bins among nodes. 
// Method 1: Naively distribute packed bins. 
//--------------------------------------------------------
void mgpu_xc_naive_distribute(){

    // due to grid point packing, npoints is always a multiple of bin_size
    int nbins    = gpu -> gpu_xcq -> nbins;
    int bin_size = gpu -> gpu_xcq -> bin_size;

#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i nbins= %i bin_size= %i \n", gpu->mpirank, nbins, bin_size);
#endif

    // array to keep track of how many bins per core
    int bins_pcore[gpu->mpisize];

    memset(bins_pcore,0, sizeof(int)*gpu->mpisize);

    int dividend  = (int) (nbins/gpu->mpisize);  
    int remainder = nbins - (dividend * gpu->mpisize);

#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i dividend= %i remainder= %i \n", gpu->mpirank, dividend, remainder);
#endif

    for(int i=0; i< gpu->mpisize; i++){
        bins_pcore[i] = dividend;
    }

#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i bins_pcore[0]= %i bins_pcore[1]= %i \n", gpu->mpirank, bins_pcore[0], bins_pcore[1]);
#endif

    // distribute the remainder among cores
    int cremainder = remainder;
    for(int i=0; i<remainder; i+=gpu->mpisize ){
        for(int j=0; j< gpu->mpisize; j++){
            bins_pcore[j] += 1;
            cremainder--;

            if(cremainder < 1) {
                break;
            }
        } 
    }

#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i bins_pcore[0]= %i bins_pcore[1]= %i \n", gpu->mpirank, bins_pcore[0], bins_pcore[1]);
#endif

    // compute lower and upper grid point limits
    int xcstart, xcend, count;
    count = 0;

    if(gpu->mpirank == 0){
        xcstart = 0;
        xcend   = bins_pcore[gpu->mpirank] * bin_size;
    }else{

#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i setting borders.. \n", gpu -> mpirank);
#endif

        for(int i=0; i < gpu->mpirank; i++){
            count += bins_pcore[i];
#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i count= %i \n", gpu -> mpirank, count);
#endif
        }
     
        xcstart = count * bin_size;
        xcend   = (count + bins_pcore[gpu->mpirank]) * bin_size;
#ifdef DEBUG
    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i start and end points= %i %i \n", gpu -> mpirank, xcstart, xcend);
#endif

    }

    gpu -> gpu_sim.mpi_xcstart = xcstart;
    gpu -> gpu_sim.mpi_xcend   = xcend;

#ifdef DEBUG
    // print information for debugging

    for(int i=0; i<gpu->mpisize; i++){
        fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i number of bins for gpu %i = %i \n", gpu -> mpirank, i, bins_pcore[i]);
    }

    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i start and end points= %i %i \n", gpu -> mpirank, xcstart, xcend);

    fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i start and end points= %i %i \n", gpu -> mpirank, gpu -> gpu_sim.mpi_xcstart, gpu -> gpu_sim.mpi_xcend);

#endif

}


//--------------------------------------------------------
// Function to distribute XC quadrature points among nodes.  
// Method 2: Consider number of true grid points in 
// each bin during the distribution.
//--------------------------------------------------------
void mgpu_xc_tpbased_greedy_distribute(){

    PRINTDEBUG("BEGIN TO DISTRIBUTE XC GRID POINTS")

    // due to grid point packing, npoints is always a multiple of bin_size
    int nbins    = gpu -> gpu_xcq -> nbins;
    int bin_size = gpu -> gpu_xcq -> bin_size;

#ifdef DEBUG
    fprintf(gpu->debugFile,"GPU: %i nbins= %i bin_size= %i \n", gpu->mpirank, nbins, bin_size);
#endif

    // array to keep track of how many true grid points per bin
    int2 tpoints[nbins];

    // save a set of flags to indicate if a given node should work on a particular bin
    char mpi_xcflags[gpu->mpisize][nbins];

    // array to keep track of how many bins per gpu
    int bins_pcore[gpu->mpisize];

    // array to keep track of how many true grid points per core
    int tpts_pcore[gpu->mpisize];

    // initialize all arrays to zero
    //memset(tpoints,0, sizeof(int)*nbins);
    memset(mpi_xcflags,0, sizeof(char)*nbins*gpu->mpisize);
    memset(bins_pcore,0, sizeof(int)*gpu->mpisize);
    memset(tpts_pcore,0, sizeof(int)*gpu->mpisize);

    // count how many true grid point in each bin and store in tpoints
    int tot_tpts=0;
    for(int i=0; i<nbins; i++){
        tpoints[i].x=i;
        tpoints[i].y=0;
        for(int j=0; j<bin_size; j++){
            if(gpu -> gpu_xcq -> dweight -> _hostData[i*bin_size + j] > 0 ){
                tpoints[i].y++;
                tot_tpts++;
            }
        }
    }

#ifdef DEBUG
    for(int i=0; i<nbins; i++){
        fprintf(gpu->debugFile,"GPU: %i bin= %i true points= %i \n", gpu->mpirank, i, tpoints[i].y);
    }
#endif

    // sort tpoints array based on the number of true points
    bool swapped;
    int sort_end=nbins-1;
    int2 aux;
    do{
      swapped=false;
      for(int i=0;i<sort_end;++i){
        if(tpoints[i].y < tpoints[i+1].y){
          aux=tpoints[i+1];
          tpoints[i+1]=tpoints[i];
          tpoints[i]=aux;
          swapped=true;
        }
      }
      --sort_end;
    }while(swapped);



    // now distribute the bins considering the total number of true grid points each core would receive 

    int mincore, min_tpts;

    for(int i=0; i<nbins; i++){

        // find out the core with minimum number of true grid points
        mincore  = 0;             // assume master has the lowest number of points
        min_tpts = tpts_pcore[0]; // set master's point count as default

        for(int impi=0; impi< gpu->mpisize; impi++){
            if(min_tpts > tpts_pcore[impi]){
                mincore  = impi;
                min_tpts = tpts_pcore[impi];
            }
        }

        // increase the point counter by the amount in current bin
        tpts_pcore[mincore] += tpoints[i].y;

        // assign the bin to corresponding core        
        mpi_xcflags[mincore][tpoints[i].x] = 1;

    }

    printf(" XC Greedy Distribute GPU: %i number of points for gpu %i = %i \n", gpu -> mpirank, gpu -> mpirank, tpts_pcore[gpu -> mpirank]);

#ifdef DEBUG

    // print information for debugging
    for(int i=0; i<gpu->mpisize; i++){
        fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i number of points for gpu %i = %i \n", gpu -> mpirank, i, tpts_pcore[i]);
    }

#endif

    // upload flags to gpu
    gpu -> gpu_xcq -> mpi_bxccompute = new cuda_buffer_type<char>(nbins);

    memcpy(gpu -> gpu_xcq -> mpi_bxccompute -> _hostData, &mpi_xcflags[gpu->mpirank][0], sizeof(char)*nbins);

    gpu -> gpu_xcq -> mpi_bxccompute -> Upload();

    gpu -> gpu_sim.mpi_bxccompute  = gpu -> gpu_xcq -> mpi_bxccompute  -> _devData;

    PRINTDEBUG("END DISTRIBUTING XC GRID POINTS")

}

//--------------------------------------------------------
// Function to distribute XC quadrature points among nodes.  
// Method 3: Consider the number of true grid point-primitive 
// funtcion product for each bin during the distribution.
//--------------------------------------------------------
void mgpu_xc_pbased_greedy_distribute(){

    PRINTDEBUG("BEGIN TO DISTRIBUTE XC GRID POINTS")

    // due to grid point packing, npoints is always a multiple of bin_size
    int nbins    = gpu -> gpu_xcq -> nbins;
    int bin_size = gpu -> gpu_xcq -> bin_size;

#ifdef DEBUG
    fprintf(gpu->debugFile,"GPU: %i nbins= %i bin_size= %i \n", gpu->mpirank, nbins, bin_size);
#endif

    // array to keep track of how many true grid points per bin
    int tpoints[nbins];

    // array to keep track of how many primitive functions per bin
    int primfpb[nbins];

    // array to keep track of true grid point primitive function product per bin
    int2 ptpf_pb[nbins];

    // save a set of flags to indicate if a given node should work on a particular bin
    char mpi_xcflags[gpu->mpisize][nbins];

    // array to keep track of how many bins per gpu
    int bins_pcore[gpu->mpisize];

    // array to keep track of how many true grid points per gpu
    int tpts_pcore[gpu->mpisize];

    // array to keep track of how many primitive functions per gpu
    int primf_pcore[gpu->mpisize];

    // array to keep track of the true grid point primf product per gpu
    int ptpf_pcore[gpu->mpisize];

   // initialize all arrays to zero
    memset(tpoints,0, sizeof(int)*nbins);
    memset(primfpb,0, sizeof(int)*nbins);
    memset(mpi_xcflags,0, sizeof(char)*nbins*gpu->mpisize);
    memset(bins_pcore,0, sizeof(int)*gpu->mpisize);
    memset(tpts_pcore,0, sizeof(int)*gpu->mpisize);
    memset(primf_pcore,0, sizeof(int)*gpu->mpisize);
    memset(ptpf_pcore,0, sizeof(int)*gpu->mpisize);

    // count how many true grid point in each bin and store in tpoints
    int tot_tpts=0;
    for(int i=0; i<nbins; i++){
        for(int j=0; j<bin_size; j++){
            if(gpu -> gpu_xcq -> dweight -> _hostData[i*bin_size + j] > 0 ){
                tpoints[i]++;
                tot_tpts++;
            }
        }
    }

    // count how many primitive functions per each bin
    for(int i=0; i<nbins; i++){

        int tot_primfpb=0;

        for(int j=gpu -> gpu_xcq -> basf_locator -> _hostData[i]; j<gpu -> gpu_xcq -> basf_locator -> _hostData[i+1] ; j++){
            for(int k=gpu -> gpu_xcq -> primf_locator -> _hostData[j]; k< gpu -> gpu_xcq -> primf_locator -> _hostData[j+1]; k++){
                tot_primfpb++;
            }
        }

        primfpb[i] = tot_primfpb;
        //printf("bin_id:= %i npoints= %i primf= %i \n", i, tpoints[i], primfpb[i]);
    }

    // compute the number of true grid point - primitive function product per bin
    for(int i=0; i<nbins; i++){
        ptpf_pb[i].x = i;
        ptpf_pb[i].y = tpoints[i] * primfpb[i];
    }

    // sort tpoints array based on the number of true points
    bool swapped;
    int sort_end=nbins-1;
    int2 aux;
    do{
      swapped=false;
      for(int i=0;i<sort_end;++i){
        if(ptpf_pb[i].y < ptpf_pb[i+1].y){
          aux=ptpf_pb[i+1];
          ptpf_pb[i+1]=ptpf_pb[i];
          ptpf_pb[i]=aux;
          swapped=true;
        }
      }
      --sort_end;
    }while(swapped);

#ifdef DEBUG
    for(int i=0; i<nbins; i++){
        fprintf(gpu->debugFile,"GPU: %i bin= %i true points= %i \n", gpu->mpirank, i, tpoints[i]);
    }
#endif

    // now distribute the bins considering the total number of true grid points each core would receive 

    int mincore, min_tpts, min_primf, min_ptpf;

    // distribute bins based on true grid point-primitive function product per bin criteria
    for(int i=0; i<nbins; i++){
        mincore   = 0;
        min_ptpf = ptpf_pcore[0];

        for(int impi=0; impi< gpu->mpisize; impi++){
            if(min_ptpf > ptpf_pcore[impi]){
                mincore  = impi;
                min_ptpf = ptpf_pcore[impi];
            }
        }

        // increase the point-primf counter by the amount in current bin
        ptpf_pcore[mincore] += ptpf_pb[i].y;

        // increase the point counter by the amount in current bin
        tpts_pcore[mincore] += tpoints[ptpf_pb[i].x];

        // assign the bin to corresponding core
        mpi_xcflags[mincore][ptpf_pb[i].x] = 1;
    }


    printf(" XC Greedy Distribute GPU: %i number of points = %i point-primitive product= %i \n", gpu -> mpirank, tpts_pcore[gpu -> mpirank], ptpf_pcore[gpu -> mpirank]);

//#ifdef DEBUG

    // print information for debugging
    //for(int i=0; i<gpu->mpisize; i++){
        //fprintf(gpu->debugFile," XC Greedy Distribute GPU: %i number of points for gpu %i = %i \n", gpu -> mpirank, i, tpts_pcore[i]);
        //printf(" XC Greedy Distribute GPU: %i number of points for gpu %i = %i \n", gpu -> mpirank, i, tpts_pcore[i]);
        //printf(" XC Greedy Distribute GPU: %i number of points for gpu %i = %i number of primf= %i \n", gpu -> mpirank, i, tpts_pcore[i], primf_pcore[i]);    
    //    printf(" XC Greedy Distribute GPU: %i number of points for gpu %i = %i number of primf= %i \n", gpu -> mpirank, i, tpts_pcore[i], primf_pcore[i]);
    //}

//#endif

    // upload flags to gpu
    gpu -> gpu_xcq -> mpi_bxccompute = new cuda_buffer_type<char>(nbins);

    memcpy(gpu -> gpu_xcq -> mpi_bxccompute -> _hostData, &mpi_xcflags[gpu->mpirank][0], sizeof(char)*nbins);

    gpu -> gpu_xcq -> mpi_bxccompute -> Upload();

    gpu -> gpu_sim.mpi_bxccompute  = gpu -> gpu_xcq -> mpi_bxccompute  -> _devData;

    PRINTDEBUG("END DISTRIBUTING XC GRID POINTS")

}

//--------------------------------------------------------
// Function to re-pack XC grid information based on flags
// generated by mgpu_xc_greedy_distribute function. 
//-------------------------------------------------------- 
void mgpu_xc_repack(){

// get the total number of true bins for this mpi rank 
int nbtr = 0;
for(int i = 0; i < gpu -> gpu_xcq -> nbins; i++){
  nbtr += gpu -> gpu_xcq -> mpi_bxccompute -> _hostData[i];
}

// array to keep track of how many true grid points per bin
int tpoints[gpu -> gpu_xcq -> nbins];

// count how many true grid point in each bin and store in tpoints
int ntot_tpts=0;
memset(tpoints,0, sizeof(int)*gpu -> gpu_xcq -> nbins);

for(int i=0; i< gpu -> gpu_xcq -> nbins; i++){

    int tpts = 0;
    for(int j=0; j< gpu -> gpu_xcq -> bin_size; j++){
        if(gpu -> gpu_xcq -> dweight -> _hostData[i*gpu -> gpu_xcq -> bin_size + j] > 0 ){
            tpoints[i]++;
            tpts++;
        }
    }

    if(gpu -> gpu_xcq -> mpi_bxccompute -> _hostData[i] > 0 ) ntot_tpts += tpts;
}

// create a temporary XC_quadrature_type object and store packed data
XC_quadrature_type* mgpu_xcq = new XC_quadrature_type;

// set properties
mgpu_xcq -> nbins    = nbtr;  
mgpu_xcq -> bin_size = MAX_POINTS_PER_CLUSTER;
mgpu_xcq -> npoints  = ntot_tpts;

mgpu_xcq -> gridx       = new cuda_buffer_type<QUICKDouble>(mgpu_xcq -> npoints);
mgpu_xcq -> gridy       = new cuda_buffer_type<QUICKDouble>(mgpu_xcq -> npoints);
mgpu_xcq -> gridz       = new cuda_buffer_type<QUICKDouble>(mgpu_xcq -> npoints);
mgpu_xcq -> sswt        = new cuda_buffer_type<QUICKDouble>(mgpu_xcq -> npoints);
mgpu_xcq -> weight      = new cuda_buffer_type<QUICKDouble>(mgpu_xcq -> npoints);
mgpu_xcq -> gatm        = new cuda_buffer_type<int>(mgpu_xcq -> npoints);
mgpu_xcq -> dweight     = new cuda_buffer_type<int>(mgpu_xcq -> npoints);
mgpu_xcq -> dweight_ssd = new cuda_buffer_type<int>(mgpu_xcq -> npoints);

mgpu_xcq -> bin_locator  = new cuda_buffer_type<int>(mgpu_xcq -> npoints);
mgpu_xcq -> basf_locator  = new cuda_buffer_type<int>(mgpu_xcq -> nbins +1);

// at this point we are still unsure about number of basis and primitive functions.
// create the arrays with original sizes
mgpu_xcq -> primf_locator = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotbf +1);
mgpu_xcq -> basf          = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotbf);
mgpu_xcq -> primf         = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotpf);

// load data, where obidx and nbidx are old and new bin indices, npidx is the new primitive index.
// nbfidx_ul and npfidx_ul variables keep track of upper bounds of basis & primitive function
// locator arrays.
int nbidx = 0, nidx=0, npidx = 0, nbfidx_ul = 0, npfidx_ul = 0;

// set the first element of the locator arrays to zero
mgpu_xcq -> basf_locator -> _hostData[0] = 0;
mgpu_xcq -> primf_locator -> _hostData[0] = 0;

for(int obidx = 0; obidx < gpu -> gpu_xcq -> nbins; obidx++){
  if(gpu -> gpu_xcq -> mpi_bxccompute -> _hostData[obidx] > 0){


    // transfer basis function info, where bffb is the number of basis functions for bin. 
    int bffb = gpu -> gpu_xcq -> basf_locator -> _hostData[obidx+1] - gpu -> gpu_xcq -> basf_locator -> _hostData[obidx];

    memcpy(&mgpu_xcq -> basf -> _hostData[nbfidx_ul], &gpu -> gpu_xcq -> basf -> _hostData[gpu -> gpu_xcq -> basf_locator -> _hostData[obidx]], sizeof(int) * bffb);

    nbfidx_ul += bffb;
    mgpu_xcq -> basf_locator -> _hostData[nbidx+1] = nbfidx_ul;

    // transfer primitive function info, where pffb is the number of primitive functions for a given basis function..
    for(int i=gpu -> gpu_xcq -> basf_locator -> _hostData[obidx]; i < gpu -> gpu_xcq -> basf_locator -> _hostData[obidx+1]; i++){
      int pffb = gpu -> gpu_xcq -> primf_locator -> _hostData[i+1] - gpu -> gpu_xcq -> primf_locator -> _hostData[i];

      memcpy(&mgpu_xcq -> primf -> _hostData[npfidx_ul], &gpu -> gpu_xcq -> primf -> _hostData[gpu -> gpu_xcq -> primf_locator -> _hostData[i]], sizeof(int) * pffb);

      npfidx_ul += pffb;
      mgpu_xcq -> primf_locator -> _hostData[npidx+1] = npfidx_ul;

      npidx++;
    }

    nbidx++;
  }
}

// set the number of basis and primitive function for the current mpi rank
mgpu_xcq -> ntotbf = nbfidx_ul;
mgpu_xcq -> ntotpf = npfidx_ul;

#ifdef DEBUG
/*fprintf(gpu->debugFile, " Repack XC data for GPU: original: %i, number of bins= %i, number of points= %i \n", gpu -> mpirank, gpu -> gpu_xcq -> nbins, gpu -> gpu_xcq -> npoints);
fprintf(gpu->debugFile, " Repack XC data for GPU: original: %i, number of basis functions= %i, number of primitive functions= %i \n", gpu -> mpirank, gpu -> gpu_xcq -> ntotbf, gpu -> gpu_xcq -> ntotpf);
*/
for(int i=0; i < gpu -> gpu_xcq -> npoints; i++){
  fprintf(gpu->debugFile, " Repack XC data: original: point= %i x= %f, y= %f, z= %f, sswt= %f, weight= %f, gatm= %i, dweight= %i, dweight_ssd= %i \n", i, gpu -> gpu_xcq -> gridx -> _hostData[i], gpu -> gpu_xcq -> gridy -> _hostData[i], gpu -> gpu_xcq -> gridz -> _hostData[i], gpu -> gpu_xcq -> sswt  -> _hostData[i], gpu -> gpu_xcq -> weight -> _hostData[i], gpu -> gpu_xcq -> gatm   -> _hostData[i], gpu -> gpu_xcq -> dweight -> _hostData[i], gpu -> gpu_xcq -> dweight_ssd -> _hostData[i]);
}

for(int i=0; i <= gpu -> gpu_xcq -> nbins; i++){
  fprintf(gpu->debugFile, " Repack XC data: original: location= %i, bf loc= %i \n", i, gpu -> gpu_xcq -> basf_locator -> _hostData[i]);
}

for(int i=0; i <= gpu -> gpu_xcq -> ntotbf; i++){
  fprintf(gpu->debugFile, " Repack XC data: original: location= %i, pf loc= %i \n", i, gpu -> gpu_xcq -> primf_locator -> _hostData[i]);
}

for(int i = 0; i < gpu -> gpu_xcq -> nbins; i++){
  for(int j = gpu -> gpu_xcq -> basf_locator -> _hostData[i]; j < gpu -> gpu_xcq -> basf_locator -> _hostData[i+1]; j++){
    for(int k = gpu -> gpu_xcq -> primf_locator -> _hostData[j]; k < gpu -> gpu_xcq -> primf_locator -> _hostData[j+1]; k++){
      fprintf(gpu->debugFile, "Repack XC data: original: bin= %i, bf location= %i, pf location= %i, bf index= %i, pf index= %i \n", i, j, k, gpu -> gpu_xcq -> basf -> _hostData[j], gpu -> gpu_xcq -> primf -> _hostData[k]);
    }
  }
}
#endif

// delete existing arrays from gpu_xcq object
SAFE_DELETE(gpu -> gpu_xcq -> gridx);
SAFE_DELETE(gpu -> gpu_xcq -> gridy);
SAFE_DELETE(gpu -> gpu_xcq -> gridz);
SAFE_DELETE(gpu -> gpu_xcq -> sswt);
SAFE_DELETE(gpu -> gpu_xcq -> weight);
SAFE_DELETE(gpu -> gpu_xcq -> gatm);
SAFE_DELETE(gpu -> gpu_xcq -> dweight);
SAFE_DELETE(gpu -> gpu_xcq -> dweight_ssd);
SAFE_DELETE(gpu -> gpu_xcq -> basf);
SAFE_DELETE(gpu -> gpu_xcq -> primf);
SAFE_DELETE(gpu -> gpu_xcq -> basf_locator);
SAFE_DELETE(gpu -> gpu_xcq -> primf_locator);

// reset properties of gpu_xcq object and reallocate arrays
gpu -> gpu_xcq -> npoints  = mgpu_xcq -> npoints;
gpu -> gpu_xcq -> nbins    = mgpu_xcq -> nbins;
gpu -> gpu_xcq -> ntotbf   = mgpu_xcq -> ntotbf;
gpu -> gpu_xcq -> ntotpf   = mgpu_xcq -> ntotpf;
gpu -> gpu_xcq -> bin_size = mgpu_xcq -> bin_size;

gpu -> gpu_xcq -> gridx         = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> gridy         = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> gridz         = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> sswt          = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> weight        = new cuda_buffer_type<QUICKDouble>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> gatm          = new cuda_buffer_type<int>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> dweight       = new cuda_buffer_type<int>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> dweight_ssd   = new cuda_buffer_type<int>(gpu -> gpu_xcq -> npoints);
gpu -> gpu_xcq -> basf          = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotbf);
gpu -> gpu_xcq -> primf         = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotpf);
gpu -> gpu_xcq -> basf_locator  = new cuda_buffer_type<int>(gpu -> gpu_xcq -> nbins +1);
gpu -> gpu_xcq -> primf_locator = new cuda_buffer_type<int>(gpu -> gpu_xcq -> ntotbf +1);

// copy content from mgpu_xcq into gpu_xcq object
memcpy(gpu -> gpu_xcq -> gridx -> _hostData, mgpu_xcq -> gridx -> _hostData, sizeof(QUICKDouble) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> gridy -> _hostData, mgpu_xcq -> gridy -> _hostData, sizeof(QUICKDouble) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> gridz -> _hostData, mgpu_xcq -> gridz -> _hostData, sizeof(QUICKDouble) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> sswt  -> _hostData, mgpu_xcq -> sswt  -> _hostData, sizeof(QUICKDouble) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> weight -> _hostData, mgpu_xcq -> weight -> _hostData, sizeof(QUICKDouble) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> gatm   -> _hostData, mgpu_xcq -> gatm   -> _hostData, sizeof(int) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> dweight -> _hostData, mgpu_xcq -> dweight -> _hostData, sizeof(int) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> dweight_ssd -> _hostData, mgpu_xcq -> dweight_ssd -> _hostData, sizeof(int) * gpu -> gpu_xcq -> npoints);
memcpy(gpu -> gpu_xcq -> basf_locator -> _hostData, mgpu_xcq -> basf_locator -> _hostData, sizeof(int) * (gpu -> gpu_xcq -> nbins + 1));
memcpy(gpu -> gpu_xcq -> primf_locator -> _hostData, mgpu_xcq -> primf_locator -> _hostData, sizeof(int) * (gpu -> gpu_xcq -> ntotbf + 1));
memcpy(gpu -> gpu_xcq -> basf -> _hostData, mgpu_xcq -> basf -> _hostData, sizeof(int) * gpu -> gpu_xcq -> ntotbf);
memcpy(gpu -> gpu_xcq -> primf -> _hostData, mgpu_xcq -> primf -> _hostData, sizeof(int) * gpu -> gpu_xcq -> ntotpf);

printf(" Repack XC data for GPU: new: %i, number of bins= %i, number of points= %i \n", gpu -> mpirank, mgpu_xcq -> nbins, mgpu_xcq -> npoints);
printf(" Repack XC data for GPU: new: %i, number of basis functions= %i, number of primitive functions= %i \n", gpu -> mpirank, mgpu_xcq -> ntotbf, mgpu_xcq -> ntotpf);

#ifdef DEBUG
// print information for debugging

fprintf(gpu->debugFile, " Repack XC data for GPU: new: %i, number of bins= %i, number of points= %i \n", gpu -> mpirank, mgpu_xcq -> nbins, mgpu_xcq -> npoints);
fprintf(gpu->debugFile, " Repack XC data for GPU: new: %i, number of basis functions= %i, number of primitive functions= %i \n", gpu -> mpirank, mgpu_xcq -> ntotbf, mgpu_xcq -> ntotpf);

for(int i=0; i < mgpu_xcq -> npoints; i++){
  fprintf(gpu->debugFile, " Repack XC data: new: point= %i x= %f, y= %f, z= %f, sswt= %f, weight= %f, gatm= %i, dweight= %i, dweight_ssd= %i \n", i, mgpu_xcq -> gridx -> _hostData[i], mgpu_xcq -> gridy -> _hostData[i], mgpu_xcq -> gridz -> _hostData[i], mgpu_xcq -> sswt  -> _hostData[i], mgpu_xcq -> weight -> _hostData[i], mgpu_xcq -> gatm   -> _hostData[i], mgpu_xcq -> dweight -> _hostData[i], mgpu_xcq -> dweight_ssd -> _hostData[i]);
}

for(int i=0; i <= mgpu_xcq -> nbins; i++){
  fprintf(gpu->debugFile, " Repack XC data: new: location= %i, bf loc= %i \n", i, mgpu_xcq -> basf_locator -> _hostData[i]);
}

for(int i=0; i <= mgpu_xcq -> ntotbf; i++){
  fprintf(gpu->debugFile, " Repack XC data: new: location= %i, pf loc= %i \n", i, mgpu_xcq -> primf_locator -> _hostData[i]);
}

for(int i = 0; i < mgpu_xcq -> nbins; i++){
  for(int j = mgpu_xcq -> basf_locator -> _hostData[i]; j < mgpu_xcq -> basf_locator -> _hostData[i+1]; j++){
    for(int k = mgpu_xcq -> primf_locator -> _hostData[j]; k < mgpu_xcq -> primf_locator -> _hostData[j+1]; k++){
      fprintf(gpu->debugFile, "Repack XC data: new: bin= %i, bf location= %i, pf location= %i, bf index= %i, pf index= %i \n", i, j, k, mgpu_xcq -> basf -> _hostData[j], mgpu_xcq -> primf -> _hostData[k]);
    }
  }
}

/*fprintf(gpu->debugFile, " Repack XC data for GPU: %i, number of bins= %i, number of points= %i \n", gpu -> mpirank, gpu -> gpu_xcq -> nbins, gpu -> gpu_xcq -> npoints);
fprintf(gpu->debugFile, " Repack XC data for GPU: %i, number of basis functions= %i, number of primitive functions= %i \n", gpu -> mpirank, gpu -> gpu_xcq -> ntotbf, gpu -> gpu_xcq -> ntotpf);

for(int i=0; i < gpu -> gpu_xcq -> npoints; i++){
  fprintf(gpu->debugFile, " Repack XC data: point= %i x= %f, y= %f, z= %f, sswt= %f, weight= %f, gatm= %i, dweight= %i, dweight_ssd= %i \n", i, gpu -> gpu_xcq -> gridx -> _hostData[i], gpu -> gpu_xcq -> gridy -> _hostData[i], gpu -> gpu_xcq -> gridz -> _hostData[i], gpu -> gpu_xcq -> sswt  -> _hostData[i], gpu -> gpu_xcq -> weight -> _hostData[i], gpu -> gpu_xcq -> gatm   -> _hostData[i], gpu -> gpu_xcq -> dweight -> _hostData[i], gpu -> gpu_xcq -> dweight_ssd -> _hostData[i]);
}

for(int i = 0; i < gpu -> gpu_xcq -> nbins; i++){
  for(int j = gpu -> gpu_xcq -> basf_locator -> _hostData[i]; j < gpu -> gpu_xcq -> basf_locator -> _hostData[i+1]; j++){
    for(int k = gpu -> gpu_xcq -> primf_locator -> _hostData[j]; k < gpu -> gpu_xcq -> primf_locator -> _hostData[j+1]; k++){
      fprintf(gpu->debugFile, "Repack XC data: bin= %i, bf location= %i, pf location= %i, bf index= %i, pf index= %i \n", i, j, k, gpu -> gpu_xcq -> basf -> _hostData[j], gpu -> gpu_xcq -> primf -> _hostData[k]);
    }
  }
}*/
#endif

// delete arrays from mgpu_xcq object
SAFE_DELETE(mgpu_xcq -> gridx);
SAFE_DELETE(mgpu_xcq -> gridy);
SAFE_DELETE(mgpu_xcq -> gridz);
SAFE_DELETE(mgpu_xcq -> sswt);
SAFE_DELETE(mgpu_xcq -> weight);
SAFE_DELETE(mgpu_xcq -> gatm);
SAFE_DELETE(mgpu_xcq -> dweight);
SAFE_DELETE(mgpu_xcq -> dweight_ssd);
SAFE_DELETE(mgpu_xcq -> basf);
SAFE_DELETE(mgpu_xcq -> primf);
SAFE_DELETE(mgpu_xcq -> basf_locator);
SAFE_DELETE(mgpu_xcq -> primf_locator);

}


//--------------------------------------------------------
// Methods passing gpu information to f90 side for printing
//--------------------------------------------------------

extern "C" void mgpu_get_device_info_(int* dev_id,int* gpu_dev_mem,
                                     int* gpu_num_proc,double* gpu_core_freq,char* gpu_dev_name,int* name_len, int* majorv, int* minorv)
{
    cudaDeviceProp prop;
    size_t device_mem;

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
