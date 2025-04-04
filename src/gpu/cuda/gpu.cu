/*
 *  Created by Yipu Miao on 4/20/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 */

#include <stdio.h>
#include <string>
#include <ctime>
#include <time.h>

#include "gpu.h"
#include "gpu_get2e_grad_ffff.h"
#include "gpu_libxc.h"
#if defined(MPIV_GPU)
  #include "mgpu.h"
#endif
#if defined(CEW)
  #include "iface.hpp"
#endif
#include "../gpu_get2e_getxc_drivers.h"
#define OSHELL
#include "../gpu_get2e_getxc_drivers.h"
#undef OSHELL


#if defined(COMPILE_GPU_AOINT)
static char *trim(char *s)
{
    char *ptr;
    if (!s)
        return NULL;   // handle NULL string
    if (!*s)
        return s;      // handle empty string
    for (ptr = s + strlen(s) - 1; (ptr >= s) && isspace(*ptr); --ptr);
    ptr[1] = '\0';
    return s;
}
#endif


//-----------------------------------------------
// checks available gmem and upload temporary large arrays
// to store values and gradients of basis functions
//-----------------------------------------------
static void upload_pteval()
{
    gpu->gpu_sim.prePtevl = false;

//    // compute available amount of global memory
//    size_t free, total;
//
//    cudaMemGetInfo(&free, &total);
//    printf("Total GMEM= %lli Free= %lli \n", total,free);
//
//    // calculate the size of an array
//    int count=0;
//    unsigned int phi_loc[gpu->gpu_xcq->npoints];
//    int oidx=0;
//    int bffb = gpu->gpu_xcq->basf_locator->_hostData[1] - gpu->gpu_xcq->basf_locator->_hostData[0];
//
//    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
//        phi_loc[i]=count;
//        int nidx=gpu->gpu_xcq->bin_locator->_hostData[i];
//        if(nidx != oidx) bffb = gpu->gpu_xcq->basf_locator->_hostData[nidx+1] - gpu->gpu_xcq->basf_locator->_hostData[nidx];
//        count += bffb;
//    }
//
//    // amount of memory in bytes for 4 such arrays
//    size_t reqMem = count * 32;
//
//    // estimate memory for future needs, 6 nbasis * nbasis 2D arrays of double type
//    size_t estMem = gpu->nbasis * gpu->nbasis * 48 + gpu->gpu_xcq->npoints * 4;
//
//    printf("Size of each pteval array= %lli Required memory for pteval= %lli Total avail= %lli\n", count, reqMem,free-estMem);
//
//    if (reqMem < free - estMem) {
//        gpu->gpu_sim.prePtevl = true;
//        gpu->gpu_xcq->phi_loc          = new gpu_buffer_type<unsigned int>(gpu->gpu_xcq->npoints);
//        gpu->gpu_xcq->phi          = new gpu_buffer_type<QUICKDouble>(count);
//        gpu->gpu_xcq->dphidx       = new gpu_buffer_type<QUICKDouble>(count);
//        gpu->gpu_xcq->dphidy       = new gpu_buffer_type<QUICKDouble>(count);
//        gpu->gpu_xcq->dphidz       = new gpu_buffer_type<QUICKDouble>(count);
//
//        memcpy(gpu->gpu_xcq->phi_loc->_hostData, &phi_loc, sizeof(unsigned int)*gpu->gpu_xcq->npoints);
//
//        gpu->gpu_xcq->phi_loc->Upload();
//        gpu->gpu_xcq->phi->Upload();
//        gpu->gpu_xcq->dphidx->Upload();
//        gpu->gpu_xcq->dphidy->Upload();
//        gpu->gpu_xcq->dphidz->Upload();
//
//        gpu->gpu_sim.phi_loc = gpu->gpu_xcq->phi_loc->_devData;
//        gpu->gpu_sim.phi = gpu->gpu_xcq->phi->_devData;
//        gpu->gpu_sim.dphidx = gpu->gpu_xcq->dphidx->_devData;
//        gpu->gpu_sim.dphidy = gpu->gpu_xcq->dphidy->_devData;
//        gpu->gpu_sim.dphidz = gpu->gpu_xcq->dphidz->_devData;
//
//        upload_sim_to_constant_dft(gpu);
//
//        getpteval(gpu);
//
//        gpu->gpu_xcq->phi->Download();
//        gpu->gpu_xcq->dphidx->Download();
//        gpu->gpu_xcq->dphidy->Download();
//        gpu->gpu_xcq->dphidz->Download();
//    }
}


//-----------------------------------------------
// Check memory and reupload for xc grad calculation
// if there is enough space
//-----------------------------------------------
static void reupload_pteval()
{
    gpu->gpu_sim.prePtevl = false;

//    // compute available amount of global memory
//    size_t free, total;
//
//    cudaMemGetInfo(&free, &total);
//    printf("Total GMEM= %lli Free= %lli \n", total,free);
//
//    // amount of memory in bytes for 4 such arrays
//    size_t reqMem = gpu->gpu_xcq->phi->_length * 32 + gpu->gpu_xcq->npoints * 4;
//
//    // estimate memory for future needs, 2 nbasis * nbasis 2D arrays of double type
//    // and 2 grad arrays of double type
//    size_t estMem = gpu->nbasis * gpu->nbasis * 16 + gpu->natom * 48;
//
//    printf("Required memory for pteval= %lli Total avail= %lli\n", reqMem,free-estMem);
//
//    if (reqMem < free - estMem ) {
//        gpu->gpu_sim.prePtevl = true;
//
//        gpu->gpu_xcq->phi_loc->ReallocateGPU();
//        gpu->gpu_xcq->phi->ReallocateGPU();
//        gpu->gpu_xcq->dphidx->ReallocateGPU();
//        gpu->gpu_xcq->dphidy->ReallocateGPU();
//        gpu->gpu_xcq->dphidz->ReallocateGPU();
//
//        gpu->gpu_xcq->phi_loc->Upload();
//        gpu->gpu_xcq->phi->Upload();
//        gpu->gpu_xcq->dphidx->Upload();
//        gpu->gpu_xcq->dphidy->Upload();
//        gpu->gpu_xcq->dphidz->Upload();
//
//        gpu->gpu_sim.phi_loc = gpu->gpu_xcq->phi_loc->_devData;
//        gpu->gpu_sim.phi = gpu->gpu_xcq->phi->_devData;
//        gpu->gpu_sim.dphidx = gpu->gpu_xcq->dphidx->_devData;
//        gpu->gpu_sim.dphidy = gpu->gpu_xcq->dphidy->_devData;
//        gpu->gpu_sim.dphidz = gpu->gpu_xcq->dphidz->_devData;
//    }
}


//-----------------------------------------------
// Delete both device and host pteval data
//-----------------------------------------------
static void delete_pteval(bool devOnly) {
    /*    if(gpu->gpu_sim.prePtevl == true){

          if(devOnly){
          gpu->gpu_xcq->phi_loc->DeleteGPU();
          gpu->gpu_xcq->phi->DeleteGPU();
          gpu->gpu_xcq->dphidx->DeleteGPU();
          gpu->gpu_xcq->dphidy->DeleteGPU();
          gpu->gpu_xcq->dphidz->DeleteGPU();
          }else{
          SAFE_DELETE(gpu->gpu_xcq->phi_loc);
          SAFE_DELETE(gpu->gpu_xcq->phi);
          SAFE_DELETE(gpu->gpu_xcq->dphidx);
          SAFE_DELETE(gpu->gpu_xcq->dphidy);
          SAFE_DELETE(gpu->gpu_xcq->dphidz);
          }
          }
     */
}


//-----------------------------------------------
// Set up specified device and be ready to ignite
//-----------------------------------------------
extern "C" void gpu_set_device_(int* gpu_dev_id, int* ierr)
{
    gpu->gpu_dev_id = *gpu_dev_id;
#ifdef DEBUG
    fprintf(gpu->debugFile,"using gpu: %i\n", *gpu_dev_id);
#endif
}


//-----------------------------------------------
// Allocates and initializes top-level GPU
// data structures
//-----------------------------------------------
extern "C" void gpu_new_(
#if defined(MPIV_GPU)
        int mpirank,
#endif
        int* ierr)
{
#if defined(DEBUG) || defined(DEBUGTIME)
#if defined(MPIV_GPU)
    char fname[16];

    sprintf(fname, "debug.gpu.%i", mpirank);
    debugFile = fopen(fname, "w+");
#else
    debugFile = fopen("debug.gpu", "w+");
#endif
#endif
    PRINTDEBUGNS("BEGIN NEW GPU ALLOC AND INIT")

    gpu = new gpu_type;

#if defined(DEBUG) || defined(DEBUGTIME)
    gpu->debugFile = debugFile;
#endif
    gpu->totalCPUMemory = 0;
    gpu->totalGPUMemory = 0;
    gpu->gpu_dev_id = -1;
    gpu->blocks = 0;
    gpu->threadsPerBlock = 0;
    gpu->twoEThreadsPerBlock = 0;
    gpu->XCThreadsPerBlock = 0;
    gpu->gradThreadsPerBlock = 0;
    gpu->xc_blocks = 0;
    gpu->xc_threadsPerBlock = 0;
    gpu->sswGradThreadsPerBlock = 0;
    gpu->mpirank = -1;
    gpu->mpisize = 0;    
    gpu->timer = NULL;
    gpu->natom = 0;
    gpu->nextatom = 0;
    gpu->nbasis = 0;
    gpu->nElec = 0;
    gpu->imult = 0;
    gpu->molchg = 0;
    gpu->iAtomType = 0;
    gpu->nshell = 0;
    gpu->nprim = 0;
    gpu->jshell = 0;
    gpu->jbasis = 0;
    gpu->maxL = 0;
    gpu->iattype = NULL;
    gpu->xyz = NULL;
    gpu->allxyz = NULL;
    gpu->chg = NULL;
    gpu->allchg = NULL;
    gpu->DFT_calculated = NULL;
    gpu->grad = NULL;
    gpu->ptchg_grad = NULL;
    gpu->gradULL = NULL;
    gpu->ptchg_gradULL = NULL;
    gpu->cew_grad = NULL;
    gpu->gpu_calculated = NULL;
    gpu->gpu_basis = NULL;
    gpu->gpu_cutoff = NULL;
    gpu->gpu_xcq = NULL;
    gpu->aoint_buffer = NULL;
    gpu->intCount = NULL;
    gpu->scratch = NULL;
    gpu->lri_data = NULL;

#if defined(MPIV_GPU)
    gpu->timer = new gpu_timer_type;
    gpu->timer->t_2elb = 0.0;
    gpu->timer->t_xclb = 0.0;
    gpu->timer->t_xcrb = 0.0;
#endif

    PRINTDEBUG("END NEW GPU ALLOC AND INIT");
}


//-----------------------------------------------
// Determines of num. of capable GPU devices and
// assigns the GPU to be used for calculations
//-----------------------------------------------
extern "C" void gpu_init_device_(int* ierr)
{
    int gpuCount, device;
    cudaError_t status;
    cudaDeviceProp deviceProp;

    PRINTDEBUG("BEGIN GPU INIT");

    status = cudaGetDeviceCount(&gpuCount);
    PRINTERROR(status, "cudaGetDeviceCount gpu_init failed!");

#if defined(DEBUG)
    fprintf(gpu->debugFile, "Number of gpus %i \n", gpuCount);
#endif

    if (gpuCount == 0) {
        *ierr = 24;
        return;
    }

    device = -1;
    if (gpu->gpu_dev_id == -1) {
        // if gpu count is greater than 1 (multi-gpu), select capable GPU with large available memory
        size_t maxMem = 0;
        for (int i = 0; i < gpuCount; ++i) {
            status = cudaGetDeviceProperties(&deviceProp, i);
            PRINTERROR(status, "cudaGetDeviceProperties gpu_init failed!");

            if ((deviceProp.major >= 2 || (deviceProp.major == 1 && deviceProp.minor == 3))
                    && deviceProp.totalGlobalMem >= maxMem) {
                maxMem = deviceProp.totalGlobalMem;
                device = i;
            }
        }
    } else {
        if (gpu->gpu_dev_id >= gpuCount) {
            *ierr = 25;
            return;
        }

        cudaGetDeviceProperties(&deviceProp, gpu->gpu_dev_id);

        if (deviceProp.major >= 2 || (deviceProp.major == 1 && deviceProp.minor == 3)) {
            device = gpu->gpu_dev_id;
        } else {
            *ierr = 26;
            return;
        }
    }

    gpu->gpu_dev_id = device;

#if defined(DEBUG)
    fprintf(gpu->debugFile, "using gpu: %i\n", device);
#endif

    if (device == -1) {
        gpu_delete_(ierr);
        *ierr = 27;
        return;
    }

    status = cudaSetDevice(device);
    PRINTERROR(status, "cudaSetDevice gpu_init failed!");
    status = cudaGetDeviceProperties(&deviceProp, device);
    PRINTERROR(status, "cudaGetDeviceProperties gpu_init failed!");

#if defined(DEBUG)
    size_t val;

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
    fprintf(gpu->debugFile, "Stack size limit:    %zu\n", val);

    cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize);
    fprintf(gpu->debugFile, "Printf fifo limit:   %zu\n", val);

    cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize);
    fprintf(gpu->debugFile, "Heap size limit:     %zu\n", val);

    cudaDeviceGetLimit(&val, cudaLimitStackSize);
    fprintf(gpu->debugFile, "New Stack size limit:    %zu\n", val);
#endif

    gpu->blocks = deviceProp.multiProcessorCount;
    if (deviceProp.major == 1) {
        switch (deviceProp.minor) {
            case 0:
            case 1:
            case 2:
            case 5:
                gpu_delete_(ierr);
                *ierr = 28;
                return;
                break;
            default:
                gpu->sm_version = SM_13;
                gpu->threadsPerBlock = SM_13_THREADS_PER_BLOCK;
                gpu->twoEThreadsPerBlock = SM_13_2E_THREADS_PER_BLOCK;
                gpu->XCThreadsPerBlock = SM_13_XC_THREADS_PER_BLOCK;
                gpu->gradThreadsPerBlock = SM_13_GRAD_THREADS_PER_BLOCK;
                break;
        }
    } else {
        gpu->sm_version = SM_2X;
        gpu->threadsPerBlock = SM_2X_THREADS_PER_BLOCK;
        gpu->twoEThreadsPerBlock = SM_2X_2E_THREADS_PER_BLOCK;
        gpu->XCThreadsPerBlock = SM_2X_XC_THREADS_PER_BLOCK;
        gpu->gradThreadsPerBlock = SM_2X_GRAD_THREADS_PER_BLOCK;
        gpu->sswGradThreadsPerBlock = SM_2X_SSW_GRAD_THREADS_PER_BLOCK;
    }

    PRINTDEBUG("FINISH GPU INIT");
}


extern "C" void gpu_get_device_info_(int* gpu_dev_count, int* gpu_dev_id, int* gpu_dev_mem,
        int* gpu_num_proc, double* gpu_core_freq, char* gpu_dev_name, int* name_len,
        int* majorv, int* minorv, int* ierr)
{
    cudaError_t error;
    cudaDeviceProp prop;
    size_t device_mem;

    *gpu_dev_id = gpu->gpu_dev_id;  // currently one GPU is supported
    error = cudaGetDeviceCount(gpu_dev_count);
    PRINTERROR(error,"cudaGetDeviceCount gpu_get_device_info failed!");
    if (*gpu_dev_count == 0)
    {
        *ierr = 24;
        return;
    }
    cudaGetDeviceProperties(&prop,*gpu_dev_id);
    device_mem = (prop.totalGlobalMem / (1024 * 1024));
    *gpu_dev_mem = (int) device_mem;
    *gpu_num_proc = (int) (prop.multiProcessorCount);
    *gpu_core_freq = (double) (prop.clockRate * 1e-6f);
    strcpy(gpu_dev_name,prop.name);
    *name_len = strlen(gpu_dev_name);
    *majorv = prop.major;
    *minorv = prop.minor;
}


//-----------------------------------------------
// allocate memory for device scratch
//-----------------------------------------------
extern "C" void gpu_allocate_scratch_(bool* allocate_gradient_scratch)
{
    gpu->scratch = new gpu_scratch;

    /* The sizes of these arrays must be (# blocks * # threads per block * store dimension).
       Note 1: that store dimension would be 35*35 in OEI code and 120*120 in ERI code when we have F functions. We will choose the max here.
       Note 2: We may have different threads/block for OEI and ERI. Choose the max of them.
     */
    unsigned int store_size = gpu->blocks * gpu->twoEThreadsPerBlock * STOREDIM_XL * STOREDIM_XL;

    gpu->scratch->store = new gpu_buffer_type<QUICKDouble>(store_size);
    gpu->scratch->store->DeleteCPU( );

    gpu->scratch->store2 = new gpu_buffer_type<QUICKDouble>(store_size);
    gpu->scratch->store2->DeleteCPU( );

    gpu->scratch->YVerticalTemp = new gpu_buffer_type<QUICKDouble>(
            gpu->blocks * gpu->twoEThreadsPerBlock * VDIM1 * VDIM2 * VDIM3);
    gpu->scratch->YVerticalTemp->DeleteCPU( );

    gpu->gpu_sim.store = gpu->scratch->store->_devData;
    gpu->gpu_sim.store2 = gpu->scratch->store2->_devData;
    gpu->gpu_sim.YVerticalTemp = gpu->scratch->YVerticalTemp->_devData;

    if (*allocate_gradient_scratch) {
        gpu->scratch->storeAA = new gpu_buffer_type<QUICKDouble>(store_size);
        gpu->scratch->storeAA->DeleteCPU( );

        gpu->scratch->storeBB = new gpu_buffer_type<QUICKDouble>(store_size);
        gpu->scratch->storeBB->DeleteCPU( );

        gpu->scratch->storeCC = new gpu_buffer_type<QUICKDouble>(store_size);
        gpu->scratch->storeCC->DeleteCPU( );

        gpu->gpu_sim.storeAA = gpu->scratch->storeAA->_devData;
        gpu->gpu_sim.storeBB = gpu->scratch->storeBB->_devData;
        gpu->gpu_sim.storeCC = gpu->scratch->storeCC->_devData;
    } else {
        gpu->scratch->storeAA = NULL;
        gpu->scratch->storeBB = NULL;
        gpu->scratch->storeCC = NULL;
        gpu->gpu_sim.storeAA = NULL;
        gpu->gpu_sim.storeBB = NULL;
        gpu->gpu_sim.storeCC = NULL;
    }

    gpuMemsetAsync(gpu->gpu_sim.store, 0, sizeof(QUICKDouble) * gpu->scratch->store->_length, 0);
    gpuMemsetAsync(gpu->gpu_sim.store2, 0, sizeof(QUICKDouble) * gpu->scratch->store2->_length, 0);
    if (*allocate_gradient_scratch) {
        gpuMemsetAsync(gpu->gpu_sim.storeAA, 0, sizeof(QUICKDouble) * gpu->scratch->storeAA->_length, 0);
        gpuMemsetAsync(gpu->gpu_sim.storeBB, 0, sizeof(QUICKDouble) * gpu->scratch->storeBB->_length, 0);
        gpuMemsetAsync(gpu->gpu_sim.storeCC, 0, sizeof(QUICKDouble) * gpu->scratch->storeCC->_length, 0);
    }
}


//-----------------------------------------------
// deallocate device scratch
//-----------------------------------------------
extern "C" void gpu_deallocate_scratch_(bool* deallocate_gradient_scratch)
{
    SAFE_DELETE(gpu->scratch->store);
    SAFE_DELETE(gpu->scratch->YVerticalTemp);
    SAFE_DELETE(gpu->scratch->store2);

    if (*deallocate_gradient_scratch) {
        SAFE_DELETE(gpu->scratch->storeAA);
        SAFE_DELETE(gpu->scratch->storeBB);
        SAFE_DELETE(gpu->scratch->storeCC);
    }
}


//-----------------------------------------------
// Deallocate top-level GPU data structures
// and reset assigned GPU device
//-----------------------------------------------
extern "C" void gpu_delete_(int* ierr)
{
    cudaError_t status;

    PRINTDEBUG("BEGIN GPU DELETE");

#if defined(MPIV_GPU)
    delete gpu->timer;
#endif
    delete gpu;

    status = cudaDeviceReset( );
    PRINTERROR(status, "cudaDeviceReset gpu_delete failed!");

    PRINTDEBUGNS("END GPU DELETE");

#if defined(DEBUG) || defined(DEBUGTIME)
    fclose(debugFile);
#endif
}


//-----------------------------------------------
//  Setup up basic infomation of the system
//-----------------------------------------------
extern "C" void gpu_setup_(int* natom, int* nbasis, int* nElec, int* imult, int* molchg,
        int* iAtomType)
{
#if defined(DEBUG)
    PRINTDEBUG("BEGIN TO SETUP");
  #if defined(MPIV_GPU)
    fprintf(gpu->debugFile,"mpirank %i natoms %i \n", gpu->mpirank, *natom );
  #endif
#endif

    gpu->natom = *natom;
    gpu->nbasis = *nbasis;
    gpu->nElec = *nElec;
    gpu->imult = *imult;
    gpu->molchg = *molchg;
    gpu->iAtomType = *iAtomType;

    gpu->gpu_calculated = new gpu_calculated_type;
    gpu->gpu_calculated->natom = *natom;
    gpu->gpu_calculated->nbasis = *nbasis;
    gpu->gpu_calculated->o = NULL;
    gpu->gpu_calculated->ob = NULL;
    gpu->gpu_calculated->dense = NULL;
    gpu->gpu_calculated->denseb = NULL;
#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->oULL = NULL;
    gpu->gpu_calculated->obULL = NULL;
#endif
    gpu->gpu_calculated->distance = NULL;
    gpu->gpu_calculated->esp_electronic = NULL;
#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->esp_electronicULL = NULL;
#endif

    gpu->gpu_basis = new gpu_basis_type;
    gpu->gpu_basis->natom = *natom;
    gpu->gpu_basis->nbasis = *nbasis;
    gpu->gpu_basis->nshell = 0;
    gpu->gpu_basis->nprim = 0;
    gpu->gpu_basis->jshell = 0;
    gpu->gpu_basis->jbasis = 0;
    gpu->gpu_basis->Qshell = 0;
    gpu->gpu_basis->maxcontract = 0;
    gpu->gpu_basis->prim_total = 0;
    gpu->gpu_basis->fStart = 0;
    gpu->gpu_basis->ffStart = 0;
    gpu->gpu_basis->ncontract = NULL;
    gpu->gpu_basis->itype = NULL;
    gpu->gpu_basis->aexp = NULL;
    gpu->gpu_basis->dcoeff = NULL;
    gpu->gpu_basis->ncenter = NULL;
    gpu->gpu_basis->sigrad2 = NULL;
    gpu->gpu_basis->kstart = NULL;
    gpu->gpu_basis->katom  = NULL;
    gpu->gpu_basis->kprim = NULL;
    gpu->gpu_basis->Ksumtype = NULL;
    gpu->gpu_basis->Qnumber = NULL;
    gpu->gpu_basis->Qstart = NULL;
    gpu->gpu_basis->Qfinal = NULL;
    gpu->gpu_basis->Qsbasis = NULL;
    gpu->gpu_basis->Qfbasis = NULL;
    gpu->gpu_basis->sorted_Qnumber = NULL;
    gpu->gpu_basis->sorted_Q = NULL;
    gpu->gpu_basis->gccoeff = NULL;
    gpu->gpu_basis->Xcoeff = NULL;
    gpu->gpu_basis->Xcoeff_oei = NULL;
    gpu->gpu_basis->expoSum = NULL;
    gpu->gpu_basis->weightedCenterX = NULL;
    gpu->gpu_basis->weightedCenterY = NULL;
    gpu->gpu_basis->weightedCenterZ = NULL;
    gpu->gpu_basis->cons = NULL;
    gpu->gpu_basis->gcexpo = NULL;
    gpu->gpu_basis->KLMN = NULL;
    gpu->gpu_basis->Apri = NULL;
    gpu->gpu_basis->Kpri = NULL;
    gpu->gpu_basis->PpriX = NULL;
    gpu->gpu_basis->PpriY = NULL;
    gpu->gpu_basis->PpriZ = NULL;
    gpu->gpu_basis->prim_start = NULL;
    gpu->gpu_basis->mpi_bcompute = NULL;
    gpu->gpu_basis->mpi_boeicompute = NULL;

    gpu->gpu_cutoff = new gpu_cutoff_type;
    gpu->gpu_cutoff->natom = 0;
    gpu->gpu_cutoff->nbasis = 0;
    gpu->gpu_cutoff->nshell = 0;
    gpu->gpu_cutoff->sqrQshell = 0;
    gpu->gpu_cutoff->sorted_YCutoffIJ = NULL;
    gpu->gpu_cutoff->cutMatrix = NULL;
    gpu->gpu_cutoff->YCutoff = NULL;
    gpu->gpu_cutoff->cutPrim = NULL;
    gpu->gpu_cutoff->integralCutoff = 0.0;
    gpu->gpu_cutoff->coreIntegralCutoff = 0.0;
    gpu->gpu_cutoff->primLimit = 0.0;
    gpu->gpu_cutoff->DMCutoff = 0.0;
    gpu->gpu_cutoff->XCCutoff = 0.0;
    gpu->gpu_cutoff->gradCutoff = 0.0;
    gpu->gpu_cutoff->sorted_OEICutoffIJ = NULL;

    gpu->gpu_sim.natom = *natom;
    gpu->gpu_sim.nbasis = *nbasis;
    gpu->gpu_sim.nElec = *nElec;
    gpu->gpu_sim.imult = *imult;
    gpu->gpu_sim.molchg = *molchg;
    gpu->gpu_sim.iAtomType = *iAtomType;
    gpu->gpu_sim.use_cew = false;

    gpu->gpu_xcq = new XC_quadrature_type;
    gpu->gpu_xcq->npoints = 0;
    gpu->gpu_xcq->nbins = 0;
    gpu->gpu_xcq->ntotbf = 0;
    gpu->gpu_xcq->ntotpf = 0;
    gpu->gpu_xcq->bin_size = 0;
    gpu->gpu_xcq->gridy = NULL;
    gpu->gpu_xcq->gridz = NULL;
    gpu->gpu_xcq->sswt = NULL;
    gpu->gpu_xcq->weight = NULL;
    gpu->gpu_xcq->gatm = NULL;
    gpu->gpu_xcq->bin_counter = NULL;
    gpu->gpu_xcq->dweight_ssd = NULL;
    gpu->gpu_xcq->basf = NULL;
    gpu->gpu_xcq->primf = NULL;
    gpu->gpu_xcq->primfpbin = NULL;
    gpu->gpu_xcq->basf_locator = NULL;
    gpu->gpu_xcq->primf_locator = NULL;
    gpu->gpu_xcq->bin_locator = NULL;
    gpu->gpu_xcq->densa = NULL;
    gpu->gpu_xcq->densb = NULL;
    gpu->gpu_xcq->gax = NULL;
    gpu->gpu_xcq->gbx = NULL;
    gpu->gpu_xcq->gay = NULL;
    gpu->gpu_xcq->gby = NULL;
    gpu->gpu_xcq->gaz = NULL;
    gpu->gpu_xcq->gbz = NULL;
    gpu->gpu_xcq->exc = NULL;
    gpu->gpu_xcq->xc_grad = NULL;
    gpu->gpu_xcq->gxc_grad = NULL;
    gpu->gpu_xcq->phi = NULL;
    gpu->gpu_xcq->dphidx = NULL;
    gpu->gpu_xcq->dphidy = NULL;
    gpu->gpu_xcq->dphidz = NULL;
    gpu->gpu_xcq->phi_loc = NULL;
    gpu->gpu_xcq->npoints_ssd = 0;
    gpu->gpu_xcq->gridy_ssd = NULL;
    gpu->gpu_xcq->gridz_ssd = NULL;
    gpu->gpu_xcq->exc_ssd = NULL;
    gpu->gpu_xcq->quadwt = NULL;
    gpu->gpu_xcq->gatm_ssd = NULL;
    gpu->gpu_xcq->uw_ssd = NULL;
    gpu->gpu_xcq->wtang = NULL;
    gpu->gpu_xcq->rwt = NULL;
    gpu->gpu_xcq->rad3 = NULL;
    gpu->gpu_xcq->gpweight = NULL;
    gpu->gpu_xcq->cfweight = NULL;
    gpu->gpu_xcq->pfweight = NULL;
    gpu->gpu_xcq->mpi_bxccompute = NULL;
    gpu->gpu_xcq->smem_size = 0;

#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    upload_para_to_const( );

#if defined(DEBUG)
    GPU_TIMER_STOP();
  #if defined(GPU)
    PRINTUSINGTIME("UPLOAD PARA TO CONST", time);
  #endif
    GPU_TIMER_DESTROY();

    PRINTDEBUG("FINISH SETUP");
#endif
}


//Madu Manathunga: 08/31/2019
//-----------------------------------------------
//  upload method and hybrid coefficient
//-----------------------------------------------
extern "C" void gpu_upload_method_(int* quick_method, bool* is_oshell, double* hyb_coeff)
{
    if (*quick_method == 0) {
        gpu->gpu_sim.method = HF;
        gpu->gpu_sim.hyb_coeff = 1.0;
    } else if (*quick_method == 1) {
        gpu->gpu_sim.method = B3LYP;
        gpu->gpu_sim.hyb_coeff = 0.2;
    } else if (*quick_method == 2) {
        gpu->gpu_sim.method = BLYP;
        gpu->gpu_sim.hyb_coeff = 0.0;
    } else if (*quick_method == 3) {
        gpu->gpu_sim.method = LIBXC;
        gpu->gpu_sim.hyb_coeff = *hyb_coeff;
    }

    gpu->gpu_sim.is_oshell = (*is_oshell != 0);
}


#if defined(CEW)
//-----------------------------------------------
//  set cew variables
//-----------------------------------------------
extern "C" void gpu_set_cew_(bool *use_cew)
{
    gpu->gpu_sim.use_cew = *use_cew;

}
#endif


//-----------------------------------------------
//  upload libxc information
//-----------------------------------------------
extern "C" void gpu_upload_libxc_(int* nof_functionals, int* functional_id, int* xc_polarization, int *ierr)
{
    int nof_aux_functionals = *nof_functionals;

#if defined(DEBUG)
    fprintf(gpu->debugFile, "Calling init_gpu_libxc.. %d %d %d \n", nof_aux_functionals, functional_id[0], *xc_polarization);
#endif

    //Madu: Initialize gpu libxc and upload information to GPU
    if (nof_aux_functionals > 0) {
        gpu->gpu_sim.glinfo = init_gpu_libxc(&nof_aux_functionals, functional_id, xc_polarization);
        gpu->gpu_sim.nauxfunc = nof_aux_functionals;
    }
}


//-----------------------------------------------
//  upload coordinates
//-----------------------------------------------
extern "C" void gpu_upload_xyz_(QUICKDouble* atom_xyz)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD COORDINATES");
//    gpu->gpu_basis->xyz = new gpu_buffer_type<QUICKDouble>(atom_xyz, 3, gpu->natom);
//    gpu->gpu_basis->xyz->Upload( );
    gpu->gpu_calculated->distance = new gpu_buffer_type<QUICKDouble>(gpu->natom, gpu->natom);
    gpu->xyz = new gpu_buffer_type<QUICKDouble>(atom_xyz, 3, gpu->natom);

    for (int i = 0; i < gpu->natom; i++) {
        for (int j = 0; j < gpu->natom; j++) {
            QUICKDouble distance = 0.0;
            for (int k = 0; k < 3; k++) {
                distance += SQR(LOC2(gpu->xyz->_hostData, k, i, 3, gpu->natom)
                        - LOC2(gpu->xyz->_hostData, k, j, 3, gpu->natom));
            }

            LOC2(gpu->gpu_calculated->distance->_hostData, i, j, gpu->natom, gpu->natom) = sqrt(distance);
        }
    }

    gpu->xyz->Upload( );
    gpu->gpu_calculated->distance->Upload( );

    gpu->gpu_sim.xyz = gpu->xyz->_devData;
    gpu->gpu_sim.distance = gpu->gpu_calculated->distance->_devData;

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD XYZ", time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING COORDINATES");
}


//-----------------------------------------------
//  upload molecule infomation
//-----------------------------------------------
extern "C" void gpu_upload_atom_and_chg_(int* atom, QUICKDouble* atom_chg)
{
    PRINTDEBUG("BEGIN TO UPLOAD ATOM AND CHARGE");

    gpu->iattype = new gpu_buffer_type<int>(atom, gpu->natom);
    gpu->chg = new gpu_buffer_type<QUICKDouble>(atom_chg, gpu->natom);
    gpu->iattype->Upload( );
    gpu->chg->Upload( );

    gpu->gpu_sim.chg = gpu->chg->_devData;
    gpu->gpu_sim.iattype = gpu->iattype->_devData;

    PRINTDEBUG("COMPLETE UPLOADING ATOM AND CHARGE");
}


//-----------------------------------------------
//  upload cutoff criteria, will update every
//  interation
//-----------------------------------------------
extern "C" void gpu_upload_cutoff_(QUICKDouble* cutMatrix, QUICKDouble* integralCutoff,
        QUICKDouble* primLimit, QUICKDouble* DMCutoff, QUICKDouble* coreIntegralCutoff)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD CUTOFF");

    gpu->gpu_cutoff->integralCutoff = *integralCutoff;
    gpu->gpu_cutoff->coreIntegralCutoff = *coreIntegralCutoff;
    gpu->gpu_cutoff->primLimit = *primLimit;
    gpu->gpu_cutoff->DMCutoff = *DMCutoff;

    gpu->gpu_cutoff->cutMatrix = new gpu_buffer_type<QUICKDouble>(cutMatrix, gpu->nshell, gpu->nshell);
    gpu->gpu_cutoff->cutMatrix->Upload( );
    gpu->gpu_cutoff->cutMatrix->DeleteCPU( );

    gpu->gpu_sim.cutMatrix = gpu->gpu_cutoff->cutMatrix->_devData;
    gpu->gpu_sim.integralCutoff = gpu->gpu_cutoff->integralCutoff;
    gpu->gpu_sim.coreIntegralCutoff = gpu->gpu_cutoff->coreIntegralCutoff;
    gpu->gpu_sim.primLimit = gpu->gpu_cutoff->primLimit;
    gpu->gpu_sim.DMCutoff = gpu->gpu_cutoff->DMCutoff;

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD CUTOFF", time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING CUTOFF");
}


//-----------------------------------------------
//  upload cutoff matrix, only update at first
//  interation
//-----------------------------------------------
extern "C" void gpu_upload_cutoff_matrix_(QUICKDouble* YCutoff,QUICKDouble* cutPrim)
{
#if defined(MPIV_GPU)
    GPU_TIMER_CREATE();
#endif
    PRINTDEBUG("BEGIN TO UPLOAD CUTOFF");

    gpu->gpu_cutoff->natom = gpu->natom;
    gpu->gpu_cutoff->YCutoff = new gpu_buffer_type<QUICKDouble>(YCutoff, gpu->nshell, gpu->nshell);
    gpu->gpu_cutoff->cutPrim = new gpu_buffer_type<QUICKDouble>(cutPrim, gpu->jbasis, gpu->jbasis);

    gpu->gpu_cutoff->YCutoff->Upload( );
    gpu->gpu_cutoff->cutPrim->Upload( );

    gpu->gpu_cutoff->sqrQshell = (gpu->gpu_basis->Qshell) * (gpu->gpu_basis->Qshell);
    gpu->gpu_cutoff->sorted_YCutoffIJ = new gpu_buffer_type<int2>(gpu->gpu_cutoff->sqrQshell);

    int a = 0;
    bool flag = true;
    int2 temp;
    int maxL = 0;

    for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
        if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] > maxL) {
            maxL = gpu->gpu_basis->sorted_Qnumber->_hostData[i];
        }
    }

#if defined(DEBUG)
    fprintf(gpu->debugFile,"MAX ANGULAR MOMENT = %i\n", maxL);
#endif

    gpu->maxL = maxL;
    gpu->gpu_sim.maxL = maxL;

    gpu->gpu_basis->fStart = 0;
    gpu->gpu_sim.fStart = 0;

    gpu->gpu_basis->ffStart = 0;
    gpu->gpu_sim.ffStart = 0;

    int sort_method = 0;

#ifdef GPU_SPDF
    if(maxL >= 3){
        sort_method = 1;
    }
#endif

    if (sort_method == 0) {
        QUICKDouble cut1 = 1E-10;
        QUICKDouble cut2 = 1E-4;
        for (int qp = 0; qp <= 6; qp++){
            for (int q = 0; q <= 3; q++) {
                for (int p = 0; p <= 3; p++) {
                    if (p + q == qp) {
                        int b = 0;
                        for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
                            for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                                if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q
                                        && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                    if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i],
                                                gpu->gpu_basis->sorted_Q->_hostData[j], gpu->nshell, gpu->nshell) > cut2
                                            && gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
                                        a++;
                                        b++;
                                    }
                                }
                            }
                        }

                        PRINTDEBUG("FINISH STEP 2");
                        flag = true;
                        for (int i = 0; i < b - 1; i++) {
                            flag = true;
                            for (int j = 0; j < b - i - 1; j ++) {
                                if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b].x]]
                                        * gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b].y]] <
                                        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1 + a - b].x]]
                                        * gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1 + a - b].y]])
                                {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b]
                                        = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b + 1];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b + 1] = temp;
                                    flag = false;
                                } else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]]
                                        * gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] ==
                                        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].x]]
                                        * gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].y]]) {
                                    if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] <
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]]) {
                                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + a - b]
                                            = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1 + a - b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1 + a - b] = temp;
                                        flag = false;
                                    }
                                }
                            }

                            if (flag == true)
                                break;
                        }
                        flag = true;
                        PRINTDEBUG("FINISH STEP 3");

                            if (b != 0) {
                                if (q == 2 && p == 3) {
#ifdef DEBUG
                                    fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                    gpu->gpu_basis->fStart = a - b;
                                    gpu->gpu_sim.fStart = a - b;
                                }

                                if (p + q == 6) {
#ifdef DEBUG
                                    fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                    gpu->gpu_basis->ffStart = a - b;
                                    gpu->gpu_sim.ffStart = a - b;
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
                        if (p + q == qp) {
                            int b = 0;
                            for (int i = 0; i < gpu->gpu_basis->Qshell; i++) {
                                for (int j = 0; j<gpu->gpu_basis->Qshell; j++) {
                                    if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q
                                            && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                        if (LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i],
                                                    gpu->gpu_basis->sorted_Q->_hostData[j],
                                                    gpu->nshell, gpu->nshell) <= cut2
                                                && LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[i],
                                                    gpu->gpu_basis->sorted_Q->_hostData[j],
                                                    gpu->nshell, gpu->nshell) > cut1
                                                && gpu->gpu_basis->sorted_Q->_hostData[i] <= gpu->gpu_basis->sorted_Q->_hostData[j]) {
                                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
                                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
                                            a++;
                                            b++;
                                        }
                                    }
                                }
                            }

                            PRINTDEBUG("FINISH STEP 1");
#ifdef DEBUG
                            fprintf(gpu->debugFile,"a=%i b=%i\n", a, b);
#endif

                            for (int i = 0; i < b - 1; i++) {
                                flag = true;
                                for (int j = 0; j < b - i - 1; j ++) {
                                    if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x],
                                                    gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y],
                                                    gpu->nshell, gpu->nshell) <
                                                LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x],
                                                    gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y],
                                                    gpu->nshell, gpu->nshell)))
                                        //&&
                                        //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].x] == q &&  \
                                        //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].y]== p &&  \
                                        //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].x] == q && \
                                        //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].y] == p )
                                    {
                                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
                                        flag = false;
                                    }
                                }

                                if (flag == true)
                                    break;
                            }

                            PRINTDEBUG("FINISH STEP 2");
                            flag = true;

                            for (int i = 0; i < b - 1; i ++) {
                                flag = true;
                                for (int j = 0; j < b - i - 1; j ++) {
                                    if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] <
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]] *
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y]])
                                    {
                                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1];
                                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1] = temp;
                                        flag = false;
                                    }
                                    else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] ==
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].x]] *
                                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].y]])
                                    {
                                        if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]]<
                                                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]]) {
                                            temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b] = temp;
                                            flag = false;
                                        }
                                    }
                                }

                                if (flag == true)
                                    break;
                            }
                            flag = true;

                            if (b != 0) {
                                if (q==2 && p==3 && gpu->gpu_sim.fStart == 0){
#ifdef DEBUG
                                    fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                    gpu->gpu_basis->fStart = a - b;
                                    gpu->gpu_sim.fStart = a - b;
                                }

                                if (p+q==6 && gpu->gpu_sim.ffStart == 0){
#ifdef DEBUG
                                    fprintf(gpu->debugFile,"df, fd, or ff starts from %i \n", a);
#endif
                                    gpu->gpu_basis->ffStart = a - b;
                                    gpu->gpu_sim.ffStart = a - b;
                                }
                            }

                            PRINTDEBUG("FINISH STEP 3");
                        }
                    }
                }
            }

            if (gpu->gpu_sim.ffStart == 0) {
                gpu->gpu_sim.ffStart = a;
                gpu->gpu_basis->ffStart = a;
            }


            if (gpu->gpu_sim.fStart == 0) {
                gpu->gpu_sim.fStart = a;
                gpu->gpu_basis->fStart = a;
            }


            //  }
            /*
               PRINTDEBUG("WORKING on F Orbital");

               gpu->gpu_basis->fStart = a;
               gpu->gpu_sim.fStart = a;

               printf("df, fd, or ff starts from %i \n", a);

               for (int q = 0; q <= 3; q++) {
               for (int p = 0; p <= 3; p++) {

               if (q == 3 && p == 3) {
               gpu->gpu_basis->ffStart = a;
               gpu->gpu_sim.ffStart = a;

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
            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
            a++;
            b++;
            }
            }
            }
            }

            PRINTDEBUG("FINISH STEP 1");
            printf("a=%i b=%i\n", a, b);
            for (int i = 0; i < b - 1; i ++)
            {
            flag = true;
            for (int j = 0; j < b - i - 1; j ++)
            {
            if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x], \
            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
            LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x], \
            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
            //&&
            //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].x] == q &&  \
            //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].y]== p &&  \
            //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].x] == q && \
            //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].y] == p )
            {
            temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
            flag = false;
            }
            }

            if (flag == true)
            break;
            }

            PRINTDEBUG("FINISH STEP 2");
            flag = true;

            for (int i = 0; i < b - 1; i ++)
            {
                flag = true;
                for (int j = 0; j < b - i - 1; j ++)
                {
                    if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] <
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]] *
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y]])
                    {
                        temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1];
                        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1] = temp;
                        flag = false;
                    }
                    else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] ==
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].x]] *
                            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].y]])
                    {
                        if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]]<
                                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]]) {
                            temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
                            flag = false;
                        }
                    }
                }

                if (flag == true)
                    break;
            }

            flag = true;
            PRINTDEBUG("FINISH STEP 3");

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
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
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
                if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x], \
                                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                            LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x], \
                                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                    //&&
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].x] == q &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].y]== p &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].x] == q && \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].y] == p )
                {
                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
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
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
                                    a++;
                                    b++;
                                }
                            }
                        }
                    }

                    PRINTDEBUG("FINISH STEP 1");
#ifdef DEBUG
                    fprintf(gpu->debugFile,"a=%i b=%i\n", a, b);
#endif

                    for (int i = 0; i < b - 1; i ++) {
                        flag = true;
                        for (int j = 0; j < b - i - 1; j ++) {
                            if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x], \
                                            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                                        LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x], \
                                            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                                //&&
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].x] == q &&  \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].y]== p &&  \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].x] == q && \
                                //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].y] == p )
                            {
                                temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
                                flag = false;
                            }
                        }

                        if (flag == true)
                            break;
                    }

                    PRINTDEBUG("FINISH STEP 2");
                    flag = true;

                    for (int i = 0; i < b - 1; i ++) {
                        flag = true;
                        for (int j = 0; j < b - i - 1; j ++) {
                            if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] <
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y]])
                            {
                                temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1];
                                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b + 1] = temp;
                                flag = false;
                            } else if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y]] ==
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].x]] *
                                    gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b+1].y]]) {
                                if (gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x]]<
                                        gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x]]) {
                                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b]
                                        = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b] = temp;
                                    flag = false;
                                }
                            }
                        }

                        if (flag == true)
                            break;
                    }

                    flag = true;
                    PRINTDEBUG("FINISH STEP 3");
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
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].x = i;
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[a].y = j;
                    a++;
                    b++;
                }
                //}
            }
        }

        for (int i = 0; i < b - 1; i ++) {
            flag = true;
            for (int j = 0; j < b - i - 1; j ++) {
                if ((LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].x], \
                                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b].y], gpu->nshell, gpu->nshell) < \
                            LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].x], \
                                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1+a-b].y], gpu->nshell, gpu->nshell)))
                    //&&
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].x] == q &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+1].y]== p &&  \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].x] == q && \
                    //gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j].y] == p )
                {
                    temp = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j+a-b] = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b];
                    gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[j + 1+a-b] = temp;
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

    gpu->gpu_cutoff->sqrQshell = a;

#ifdef DEBUG
    fprintf(gpu->debugFile,"SS = %i\n",a);
    for (int i = 0; i<a; i++) {
        fprintf(gpu->debugFile,"%8i %4i %4i %18.13f Q=%4i %4i %4i %4i prim = %4i %4i\n",i, \
                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x, \
                gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y, \
                LOC2(YCutoff, gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x], gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y], gpu->nshell, gpu->nshell),\
                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x], \
                gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y], \
                gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x], \
                gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y], \
                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x]], \
                gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y]]);
    }
#endif

    gpu->gpu_cutoff->sorted_YCutoffIJ->Upload();
    gpu->gpu_sim.sqrQshell = gpu->gpu_cutoff->sqrQshell;
    gpu->gpu_sim.YCutoff = gpu->gpu_cutoff->YCutoff->_devData;
    gpu->gpu_sim.cutPrim = gpu->gpu_cutoff->cutPrim->_devData;
    gpu->gpu_sim.sorted_YCutoffIJ = gpu->gpu_cutoff->sorted_YCutoffIJ->_devData;

#if defined(MPIV_GPU)
    GPU_TIMER_START();

    mgpu_eri_greedy_distribute();

    GPU_TIMER_STOP();
    gpu->timer->t_2elb += (double) time / 1000.0;
    GPU_TIMER_DESTROY();
#endif

//    gpu->gpu_cutoff->YCutoff->DeleteCPU();
//    gpu->gpu_cutoff->cutPrim->DeleteCPU();
//    gpu->gpu_cutoff->sorted_YCutoffIJ->DeleteCPU();

    PRINTDEBUG("COMPLETE UPLOADING CUTOFF");
}


//-----------------------------------------------
//  upload information for OEI calculation
//-----------------------------------------------
extern "C" void gpu_upload_oei_(int* nextatom, QUICKDouble* extxyz, QUICKDouble* extchg, int *ierr)
{

    // store coordinates and charges for oei calculation
    gpu->nextatom = *nextatom;
    gpu->allxyz = new gpu_buffer_type<QUICKDouble>(3, gpu->natom+gpu->nextatom);
    gpu->allchg = new gpu_buffer_type<QUICKDouble>(gpu->natom+gpu->nextatom);

    memcpy(gpu->allxyz->_hostData, gpu->xyz->_hostData, sizeof(QUICKDouble) * 3 * gpu->natom);
    memcpy(gpu->allchg->_hostData, gpu->chg->_hostData, sizeof(QUICKDouble) * gpu->natom);

    // manually append f90 data
    unsigned int idxf90data = 0;
    for (unsigned int i = 0; i < gpu->nextatom; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            gpu->allxyz->_hostData[(gpu->natom + i) * 3 + j] = extxyz[idxf90data++];

    idxf90data = 0;
    for (unsigned int i = 0; i < gpu->nextatom; ++i)
        gpu->allchg->_hostData[gpu->natom + i] = extchg[idxf90data++];

    gpu->allxyz->Upload();
    gpu->allchg->Upload();

    gpu->gpu_sim.nextatom = *nextatom;
    gpu->gpu_sim.allxyz = gpu->allxyz->_devData;
    gpu->gpu_sim.allchg = gpu->allchg->_devData;

    // precompute the product of overlap prefactor and contraction coefficients and store
    gpu->gpu_basis->Xcoeff_oei = new gpu_buffer_type<QUICKDouble>(2 * gpu->jbasis, 2 * gpu->jbasis);

    for (int i = 0; i < gpu->jshell; i++) {
        for (int j = 0; j < gpu->jshell; j++) {
            int kAtomI = gpu->gpu_basis->katom->_hostData[i];
            int kAtomJ = gpu->gpu_basis->katom->_hostData[j];
            int KsumtypeI = gpu->gpu_basis->Ksumtype->_hostData[i];
            int KsumtypeJ = gpu->gpu_basis->Ksumtype->_hostData[j];
            int kstartI = gpu->gpu_basis->kstart->_hostData[i];
            int kstartJ = gpu->gpu_basis->kstart->_hostData[j];

            QUICKDouble DIJ = 0.0;
            for (int k = 0; k < 3; k++) {
                DIJ += SQR(LOC2(gpu->xyz->_hostData, k, kAtomI - 1, 3, gpu->natom)
                        - LOC2(gpu->xyz->_hostData, k, kAtomJ - 1, 3, gpu->natom));
            }

            for (int ii = 0; ii < gpu->gpu_basis->kprim->_hostData[i]; ii++) {
                for (int jj = 0; jj < gpu->gpu_basis->kprim->_hostData[j]; jj++) {
                    QUICKDouble II = LOC2(gpu->gpu_basis->gcexpo->_hostData, ii, KsumtypeI - 1, MAXPRIM, gpu->nbasis);
                    QUICKDouble JJ = LOC2(gpu->gpu_basis->gcexpo->_hostData, jj, KsumtypeJ - 1, MAXPRIM, gpu->nbasis);

                    QUICKDouble X = 2.0 * PI_TO_3HALF * sqrt((II + JJ) / PI)
                        * pow((II + JJ), -1.5) * exp((-II * JJ * DIJ) / (II + JJ));

                    for (int itemp = gpu->gpu_basis->Qstart->_hostData[i]; itemp <= gpu->gpu_basis->Qfinal->_hostData[i]; itemp++) {
                        for (int itemp2 = gpu->gpu_basis->Qstart->_hostData[j]; itemp2 <= gpu->gpu_basis->Qfinal->_hostData[j]; itemp2++) {
                            LOC4(gpu->gpu_basis->Xcoeff_oei->_hostData, kstartI + ii - 1, kstartJ + jj - 1,
                                    itemp-gpu->gpu_basis->Qstart->_hostData[i],
                                    itemp2-gpu->gpu_basis->Qstart->_hostData[j],
                                    gpu->jbasis, gpu->jbasis, 2, 2)
                                = X * LOC2(gpu->gpu_basis->gccoeff->_hostData, ii, KsumtypeI + itemp - 1, MAXPRIM, gpu->nbasis)
                                * LOC2(gpu->gpu_basis->gccoeff->_hostData, jj, KsumtypeJ + itemp2 - 1, MAXPRIM, gpu->nbasis);
                        }
                    }
                }
            }
        }
    }

    gpu->gpu_basis->Xcoeff_oei->Upload();
    gpu->gpu_sim.Xcoeff_oei = gpu->gpu_basis->Xcoeff_oei->_devData;

    // allocate array for sorted shell pair info
    gpu->gpu_cutoff->sorted_OEICutoffIJ = new gpu_buffer_type<int2>(gpu->gpu_basis->Qshell * gpu->gpu_basis->Qshell);

    unsigned char sort_method = 0;
    unsigned int a = 0;

    if (sort_method == 0) {
        // store Qshell indices, at this point we already have Qshells sorted according to type.
        for (int qp = 0; qp <= 6 ; ++qp) {
            for (int q = 0; q <= 3; ++q) {
                for (int p = 0; p <= 3; ++p) {
                    if (p + q == qp) {
                        unsigned int b = 0;
                        for (int i = 0; i < gpu->gpu_basis->Qshell; ++i) {
                            for (int j = 0; j < gpu->gpu_basis->Qshell; ++j) {
                                if (gpu->gpu_basis->sorted_Qnumber->_hostData[i] == q
                                        && gpu->gpu_basis->sorted_Qnumber->_hostData[j] == p) {
                                    // check if the product of overlap prefactor and contraction coefficients is greater than the threshold
                                    // if a given basis function pair has at least one primitive pair that satify this condition, we will add it

                                    /*bool bSignificant=false;

                                      int kPrimI = gpu->gpu_basis->kprim->_hostData[i];
                                      int kPrimJ = gpu->gpu_basis->kprim->_hostData[j];

                                      int kStartI = gpu->gpu_basis->kstart->_hostData[i]-1;
                                      int kStartJ = gpu->gpu_basis->kstart->_hostData[j]-1;

                                      for(int iprim=0; iprim < kPrimI * kPrimJ ; ++iprim){
                                      int JJJ = (int) iprim/kPrimI;
                                      int III = (int) iprim-kPrimI*JJJ;

                                      QUICKDouble Xcoeff_oei = LOC4(gpu->gpu_basis->Xcoeff_oei->_hostData, kStartI+III, kStartJ+JJJ, q - gpu->gpu_basis->Qstart->_hostData[gpu->gpu_basis->sorted_Q->_hostData[i]], \
                                      p - gpu->gpu_basis->Qstart->_hostData[gpu->gpu_basis->sorted_Q->_hostData[j]], gpu->jbasis, gpu->jbasis, 2, 2);

                                      printf("xcoeff_oei: %d %d %d %d %.10e \n", kStartI+III, kStartJ+JJJ, q - gpu->gpu_basis->Qstart->_hostData[gpu->gpu_basis->sorted_Q->_hostData[i]], \
                                      p - gpu->gpu_basis->Qstart->_hostData[gpu->gpu_basis->sorted_Q->_hostData[j]], Xcoeff_oei);


                                    //if(Xcoeff_oei > gpu->gpu_cutoff->integralCutoff){
                                    if(abs(Xcoeff_oei) > 0.0 ){
                                    bSignificant=true;
                                    break;
                                    }

                                    }

                                    if(bSignificant){*/
                                    gpu->gpu_cutoff->sorted_OEICutoffIJ->_hostData[a].x = i;
                                    gpu->gpu_cutoff->sorted_OEICutoffIJ->_hostData[a].y = j;
                                    ++a;
                                    ++b;
                                    //}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    gpu->gpu_cutoff->sorted_OEICutoffIJ->Upload();
    gpu->gpu_sim.sorted_OEICutoffIJ = gpu->gpu_cutoff->sorted_OEICutoffIJ->_devData;
    gpu->gpu_sim.Qshell_OEI = a;

#ifdef MPIV_GPU
    mgpu_oei_greedy_distribute();
#endif

    /*  for(int i=0; i<gpu->gpu_basis->Qshell * gpu->gpu_basis->Qshell; ++i) {

        int II = gpu->gpu_cutoff->sorted_OEICutoffIJ->_hostData[i].x;
        int JJ = gpu->gpu_cutoff->sorted_OEICutoffIJ->_hostData[i].y;

        int ii = gpu->gpu_basis->sorted_Q->_hostData[II];
        int jj = gpu->gpu_basis->sorted_Q->_hostData[JJ];

        int iii = gpu->gpu_basis->sorted_Qnumber->_hostData[II];
        int jjj = gpu->gpu_basis->sorted_Qnumber->_hostData[JJ];

        printf("%i II JJ ii jj iii jjj %d %d %d %d %d %d nprim_i: %d nprim_j: %d \n",i, II, JJ, ii, jj, iii, jjj, gpu->gpu_basis->kprim->_hostData[ii], gpu->gpu_basis->kprim->_hostData[jj]);
        }
     */
    gpu->gpu_cutoff->sorted_OEICutoffIJ->DeleteCPU();
    gpu->gpu_basis->Xcoeff_oei->DeleteCPU();
    gpu->gpu_basis->kstart->DeleteCPU();
    gpu->gpu_basis->katom->DeleteCPU();
    gpu->gpu_basis->Ksumtype->DeleteCPU();
    gpu->gpu_basis->Qstart->DeleteCPU();
    gpu->gpu_basis->Qfinal->DeleteCPU();
    gpu->gpu_basis->gccoeff->DeleteCPU();
    gpu->gpu_basis->gcexpo->DeleteCPU();
}


//-----------------------------------------------
//  upload information for OEPROP calculation
//-----------------------------------------------
extern "C" void gpu_upload_oeprop_(int * nextpoint, QUICKDouble * extpointxyz,
        QUICKDouble * esp_electronic, int *ierr)
{
    // store coordinates and charges for oeprop calculation
    gpu->nextpoint = *nextpoint;
    gpu->extpointxyz = new gpu_buffer_type<QUICKDouble>(extpointxyz, 3, gpu->nextpoint);

    gpu->extpointxyz->Upload();

    gpu->gpu_sim.nextpoint = *nextpoint;
    gpu->gpu_sim.extpointxyz = gpu->extpointxyz->_devData;

    gpu->gpu_calculated->esp_electronic = new gpu_buffer_type<QUICKDouble>(1, gpu->nextpoint);

#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->esp_electronic->DeleteGPU();
    gpu->gpu_calculated->esp_electronicULL = new gpu_buffer_type<QUICKULL>(1, gpu->nextpoint);
    gpu->gpu_calculated->esp_electronicULL->Upload();
    gpu->gpu_sim.esp_electronicULL = gpu->gpu_calculated->esp_electronicULL->_devData;
#else
    gpu->gpu_calculated->esp_electronic->Upload();
    gpu->gpu_sim.esp_electronic = gpu->gpu_calculated->esp_electronic->_devData;
#endif
}


//-----------------------------------------------
//  upload calculated information
//-----------------------------------------------
extern "C" void gpu_upload_calculated_(QUICKDouble* o, QUICKDouble* co, QUICKDouble* vec, QUICKDouble* dense)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD O MATRIX");

    gpu->gpu_calculated->o = new gpu_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);
    gpu->gpu_calculated->dense = new gpu_buffer_type<QUICKDouble>(dense, gpu->nbasis, gpu->nbasis);

#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->o->DeleteGPU();
    gpu->gpu_calculated->oULL = new gpu_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
    gpu->gpu_calculated->oULL->Upload();
    gpu->gpu_sim.oULL = gpu->gpu_calculated->oULL->_devData;
#else
    gpu->gpu_calculated->o->Upload();
    gpu->gpu_sim.o = gpu->gpu_calculated->o->_devData;
#endif

    gpu->gpu_calculated->dense->Upload();
    gpu->gpu_sim.dense = gpu->gpu_calculated->dense->_devData;

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD CALCULATE", time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING O MATRIX");
}


//-----------------------------------------------
//  upload calculated information for uscf
//-----------------------------------------------
extern "C" void gpu_upload_calculated_beta_(QUICKDouble* ob, QUICKDouble* denseb)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD BETA O MATRIX");

    gpu->gpu_calculated->ob = new gpu_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->ob->DeleteGPU();
    gpu->gpu_calculated->obULL = new gpu_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
    gpu->gpu_calculated->obULL->Upload();
    gpu->gpu_sim.obULL = gpu->gpu_calculated->obULL->_devData;
#else
    gpu->gpu_calculated->ob->Upload();
    gpu->gpu_sim.ob = gpu->gpu_calculated->ob->_devData;
#endif

    gpu_upload_beta_density_matrix_(denseb);

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD CALCULATE",time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING BETA O MATRIX");
}


// Added by Madu Manathunga on 01/07/2020
//This method uploads density matrix onto gpu for XC gradient calculation
extern "C" void gpu_upload_density_matrix_(QUICKDouble* dense)
{
    gpu->gpu_calculated->dense = new gpu_buffer_type<QUICKDouble>(dense,  gpu->nbasis, gpu->nbasis);
    gpu->gpu_calculated->dense->Upload();
    gpu->gpu_sim.dense = gpu->gpu_calculated->dense->_devData;
}


extern "C" void gpu_upload_beta_density_matrix_(QUICKDouble* denseb)
{
    gpu->gpu_calculated->denseb = new gpu_buffer_type<QUICKDouble>(denseb,  gpu->nbasis, gpu->nbasis);
    gpu->gpu_calculated->denseb->Upload();
    gpu->gpu_sim.denseb = gpu->gpu_calculated->denseb->_devData;
}


//-----------------------------------------------
//  upload basis set information
//-----------------------------------------------
extern "C" void gpu_upload_basis_(int* nshell, int* nprim, int* jshell, int* jbasis, int* maxcontract,
        int* ncontract, int* itype, QUICKDouble* aexp, QUICKDouble* dcoeff,
        int* first_basis_function, int* last_basis_function, int* first_shell_basis_function, int* last_shell_basis_function,
        int* ncenter, int* kstart, int* katom, int* ktype, int* kprim, int* kshell, int* Ksumtype,
        int* Qnumber, int* Qstart, int* Qfinal,int* Qsbasis, int* Qfbasis,
        QUICKDouble* gccoeff, QUICKDouble* cons, QUICKDouble* gcexpo, int* KLMN)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD BASIS");

    gpu->gpu_basis->nshell          =   *nshell;
    gpu->gpu_basis->nprim           =   *nprim;
    gpu->gpu_basis->jshell          =   *jshell;
    gpu->gpu_basis->jbasis          =   *jbasis;
    gpu->gpu_basis->maxcontract     =   *maxcontract;

    gpu->nshell                       =   *nshell;
    gpu->nprim                        =   *nprim;
    gpu->jshell                       =   *jshell;
    gpu->jbasis                       =   *jbasis;

    gpu->gpu_sim.nshell                   =   *nshell;
    gpu->gpu_sim.nprim                    =   *nprim;
    gpu->gpu_sim.jshell                   =   *jshell;
    gpu->gpu_sim.jbasis                   =   *jbasis;
    gpu->gpu_sim.maxcontract              =   *maxcontract;

    gpu->gpu_basis->ncontract                   =   new gpu_buffer_type<int>(ncontract, gpu->nbasis);//gpu->nbasis);
    gpu->gpu_basis->itype                       =   new gpu_buffer_type<int>(itype, 3,  gpu->nbasis);//3, gpu->nbasis);
    gpu->gpu_basis->aexp                        =   new gpu_buffer_type<QUICKDouble>(aexp, gpu->gpu_basis->maxcontract, gpu->nbasis);//gpu->gpu_basis->maxcontract, gpu->nbasis);
    gpu->gpu_basis->dcoeff                      =   new gpu_buffer_type<QUICKDouble>(dcoeff, gpu->gpu_basis->maxcontract, gpu->nbasis);//gpu->gpu_basis->maxcontract, gpu->nbasis);
    /*
       gpu->gpu_basis->first_basis_function        =   new gpu_buffer_type<int>(first_basis_function, 1);//gpu->natom);
       gpu->gpu_basis->last_basis_function         =   new gpu_buffer_type<int>(last_basis_function,  1);//gpu->natom);

       gpu->gpu_basis->first_shell_basis_function  =   new gpu_buffer_type<int>(first_shell_basis_function, 1);//gpu->gpu_basis->nshell);
       gpu->gpu_basis->last_shell_basis_function   =   new gpu_buffer_type<int>(last_shell_basis_function,  1);//gpu->gpu_basis->nshell);

       gpu->gpu_basis->ktype                       =   new gpu_buffer_type<int>(ktype,    gpu->gpu_basis->nshell);
       gpu->gpu_basis->kshell                      =   new gpu_buffer_type<int>(kshell,   93);
     */
    gpu->gpu_basis->ncenter                     =   new gpu_buffer_type<int>(ncenter,  gpu->gpu_basis->nbasis);

    gpu->gpu_basis->kstart                      =   new gpu_buffer_type<int>(kstart,   gpu->gpu_basis->nshell);
    gpu->gpu_basis->katom                       =   new gpu_buffer_type<int>(katom,    gpu->gpu_basis->nshell);
    gpu->gpu_basis->kprim                       =   new gpu_buffer_type<int>(kprim,    gpu->gpu_basis->nshell);
    gpu->gpu_basis->Ksumtype                    =   new gpu_buffer_type<int>(Ksumtype, gpu->gpu_basis->nshell+1);

    gpu->gpu_basis->Qnumber                     =   new gpu_buffer_type<int>(Qnumber,  gpu->gpu_basis->nshell);
    gpu->gpu_basis->Qstart                      =   new gpu_buffer_type<int>(Qstart,   gpu->gpu_basis->nshell);
    gpu->gpu_basis->Qfinal                      =   new gpu_buffer_type<int>(Qfinal,   gpu->gpu_basis->nshell);
    gpu->gpu_basis->Qsbasis                     =   new gpu_buffer_type<int>(Qsbasis,  gpu->gpu_basis->nshell, 4);
    gpu->gpu_basis->Qfbasis                     =   new gpu_buffer_type<int>(Qfbasis,  gpu->gpu_basis->nshell, 4);
    gpu->gpu_basis->gccoeff                     =   new gpu_buffer_type<QUICKDouble>(gccoeff, MAXPRIM, gpu->nbasis);

    gpu->gpu_basis->cons                        =   new gpu_buffer_type<QUICKDouble>(cons, gpu->nbasis);
    gpu->gpu_basis->gcexpo                      =   new gpu_buffer_type<QUICKDouble>(gcexpo, MAXPRIM, gpu->nbasis);
    gpu->gpu_basis->KLMN                        =   new gpu_buffer_type<unsigned char>(3, gpu->nbasis);

    size_t index_c = 0;
    size_t index_f = 0;
    for (size_t j=0; j<gpu->nbasis; j++) {
        for (size_t i=0; i<3; i++) {
            index_c = j * 3 + i;
            gpu->gpu_basis->KLMN->_hostData[index_c] = (unsigned char) KLMN[index_f++];
        }
    }

    gpu->gpu_basis->prim_start                  =   new gpu_buffer_type<int>(gpu->gpu_basis->nshell);
    gpu->gpu_basis->prim_total = 0;

    for (int i = 0 ; i < gpu->gpu_basis->nshell; i++) {
        gpu->gpu_basis->prim_start->_hostData[i] = gpu->gpu_basis->prim_total;
        gpu->gpu_basis->prim_total += gpu->gpu_basis->kprim->_hostData[i];
    }

#ifdef DEBUG
    for (int i = 0; i<gpu->gpu_basis->nshell; i++) {
        fprintf(gpu->debugFile,"for %i prim= %i, start= %i\n", i, gpu->gpu_basis->kprim->_hostData[i], gpu->gpu_basis->prim_start->_hostData[i]);
    }
    fprintf(gpu->debugFile,"total=%i\n", gpu->gpu_basis->prim_total);
#endif

    int prim_total = gpu->gpu_basis->prim_total;
    gpu->gpu_sim.prim_total = gpu->gpu_basis->prim_total;

    gpu->gpu_basis->Xcoeff                      =   new gpu_buffer_type<QUICKDouble>(2*gpu->jbasis, 2*gpu->jbasis);
    gpu->gpu_basis->expoSum                     =   new gpu_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu->gpu_basis->weightedCenterX             =   new gpu_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu->gpu_basis->weightedCenterY             =   new gpu_buffer_type<QUICKDouble>(prim_total, prim_total);
    gpu->gpu_basis->weightedCenterZ             =   new gpu_buffer_type<QUICKDouble>(prim_total, prim_total);

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

    gpu->gpu_sim.Qshell = gpu->gpu_basis->Qshell;

    gpu->gpu_basis->sorted_Q                    =   new gpu_buffer_type<int>( gpu->gpu_basis->Qshell);
    gpu->gpu_basis->sorted_Qnumber              =   new gpu_buffer_type<int>( gpu->gpu_basis->Qshell);

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

            QUICKDouble distance = 0.0;
            for (int k = 0; k < 3; k++) {
                distance += SQR(LOC2(gpu->xyz->_hostData, k, kAtomI - 1, 3, gpu->natom)
                        - LOC2(gpu->xyz->_hostData, k, kAtomJ - 1, 3, gpu->natom));
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

    //    gpu->gpu_basis->upload_all();
    gpu->gpu_basis->ncontract->Upload();
    gpu->gpu_basis->itype->Upload();
    gpu->gpu_basis->aexp->Upload();
    gpu->gpu_basis->dcoeff->Upload();
    gpu->gpu_basis->ncenter->Upload();
    gpu->gpu_basis->kstart->Upload();
    gpu->gpu_basis->katom->Upload();
    gpu->gpu_basis->kprim->Upload();
    gpu->gpu_basis->Ksumtype->Upload();
    gpu->gpu_basis->Qnumber->Upload();
    gpu->gpu_basis->Qstart->Upload();
    gpu->gpu_basis->Qfinal->Upload();
    gpu->gpu_basis->Qsbasis->Upload();
    gpu->gpu_basis->Qfbasis->Upload();
    gpu->gpu_basis->gccoeff->Upload();
    gpu->gpu_basis->cons->Upload();
    gpu->gpu_basis->Xcoeff->Upload();
    gpu->gpu_basis->gcexpo->Upload();
    gpu->gpu_basis->KLMN->Upload();
    gpu->gpu_basis->prim_start->Upload();
    gpu->gpu_basis->Xcoeff->Upload();
    gpu->gpu_basis->expoSum->Upload();
    gpu->gpu_basis->weightedCenterX->Upload();
    gpu->gpu_basis->weightedCenterY->Upload();
    gpu->gpu_basis->weightedCenterZ->Upload();
    gpu->gpu_basis->sorted_Q->Upload();
    gpu->gpu_basis->sorted_Qnumber->Upload();

    gpu->gpu_sim.expoSum                      =   gpu->gpu_basis->expoSum->_devData;
    gpu->gpu_sim.weightedCenterX              =   gpu->gpu_basis->weightedCenterX->_devData;
    gpu->gpu_sim.weightedCenterY              =   gpu->gpu_basis->weightedCenterY->_devData;
    gpu->gpu_sim.weightedCenterZ              =   gpu->gpu_basis->weightedCenterZ->_devData;
    gpu->gpu_sim.sorted_Q                     =   gpu->gpu_basis->sorted_Q->_devData;
    gpu->gpu_sim.sorted_Qnumber               =   gpu->gpu_basis->sorted_Qnumber->_devData;
    gpu->gpu_sim.Xcoeff                       =   gpu->gpu_basis->Xcoeff->_devData;
    gpu->gpu_sim.ncontract                    =   gpu->gpu_basis->ncontract->_devData;
    gpu->gpu_sim.dcoeff                       =   gpu->gpu_basis->dcoeff->_devData;
    gpu->gpu_sim.aexp                         =   gpu->gpu_basis->aexp->_devData;
    gpu->gpu_sim.ncenter                      =   gpu->gpu_basis->ncenter->_devData;
    gpu->gpu_sim.itype                        =   gpu->gpu_basis->itype->_devData;
    gpu->gpu_sim.prim_start                   =   gpu->gpu_basis->prim_start->_devData;
    /*
       gpu->gpu_sim.first_basis_function         =   gpu->gpu_basis->first_basis_function->_devData;
       gpu->gpu_sim.last_basis_function          =   gpu->gpu_basis->last_basis_function->_devData;
       gpu->gpu_sim.first_shell_basis_function   =   gpu->gpu_basis->first_shell_basis_function->_devData;
       gpu->gpu_sim.last_shell_basis_function    =   gpu->gpu_basis->last_shell_basis_function->_devData;
       gpu->gpu_sim.ktype                        =   gpu->gpu_basis->ktype->_devData;
       gpu->gpu_sim.kshell                       =   gpu->gpu_basis->kshell->_devData;
     */
    gpu->gpu_sim.kstart                       =   gpu->gpu_basis->kstart->_devData;
    gpu->gpu_sim.katom                        =   gpu->gpu_basis->katom->_devData;
    gpu->gpu_sim.kprim                        =   gpu->gpu_basis->kprim->_devData;
    gpu->gpu_sim.Ksumtype                     =   gpu->gpu_basis->Ksumtype->_devData;
    gpu->gpu_sim.Qnumber                      =   gpu->gpu_basis->Qnumber->_devData;
    gpu->gpu_sim.Qstart                       =   gpu->gpu_basis->Qstart->_devData;
    gpu->gpu_sim.Qfinal                       =   gpu->gpu_basis->Qfinal->_devData;
    gpu->gpu_sim.Qsbasis                      =   gpu->gpu_basis->Qsbasis->_devData;
    gpu->gpu_sim.Qfbasis                      =   gpu->gpu_basis->Qfbasis->_devData;
    gpu->gpu_sim.gccoeff                      =   gpu->gpu_basis->gccoeff->_devData;
    gpu->gpu_sim.cons                         =   gpu->gpu_basis->cons->_devData;
    gpu->gpu_sim.gcexpo                       =   gpu->gpu_basis->gcexpo->_devData;
    gpu->gpu_sim.KLMN                         =   gpu->gpu_basis->KLMN->_devData;

    gpu->gpu_basis->expoSum->DeleteCPU();
    gpu->gpu_basis->weightedCenterX->DeleteCPU();
    gpu->gpu_basis->weightedCenterY->DeleteCPU();
    gpu->gpu_basis->weightedCenterZ->DeleteCPU();
    gpu->gpu_basis->Xcoeff->DeleteCPU();

    gpu->gpu_basis->ncontract->DeleteCPU();
    //    gpu->gpu_basis->dcoeff->DeleteCPU();
    gpu->gpu_basis->aexp->DeleteCPU();
    gpu->gpu_basis->ncenter->DeleteCPU();
    gpu->gpu_basis->itype->DeleteCPU();

    //kprim can not be deleted since it will be used later
    //gpu->gpu_basis->kprim->DeleteCPU();

    gpu->gpu_basis->prim_start->DeleteCPU();

    gpu->gpu_basis->Qnumber->DeleteCPU();

    gpu->gpu_basis->Qsbasis->DeleteCPU();
    gpu->gpu_basis->Qfbasis->DeleteCPU();
    gpu->gpu_basis->cons->DeleteCPU();
    gpu->gpu_basis->KLMN->DeleteCPU();

    // the following will be deleted inside gpu_upload_oei function
    //gpu->gpu_basis->kstart->DeleteCPU();
    //gpu->gpu_basis->katom->DeleteCPU();
    //gpu->gpu_basis->Ksumtype->DeleteCPU();
    //gpu->gpu_basis->Qstart->DeleteCPU();
    //gpu->gpu_basis->Qfinal->DeleteCPU();
    //gpu->gpu_basis->gccoeff->DeleteCPU();
    //gpu->gpu_basis->gcexpo->DeleteCPU();

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD BASIS", time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING BASIS");
}


extern "C" void gpu_upload_grad_(QUICKDouble* gradCutoff)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO UPLOAD GRAD");

    gpu->grad = new gpu_buffer_type<QUICKDouble>(3 * gpu->natom);

#if defined(USE_LEGACY_ATOMICS)
    gpu->gradULL = new gpu_buffer_type<QUICKULL>(3 * gpu->natom);
    gpu->gpu_sim.gradULL = gpu->gradULL->_devData;
    gpu->gradULL->Upload();
#endif

    //gpu->grad->DeleteGPU();
    gpu->gpu_sim.grad = gpu->grad->_devData;
    gpu->grad->Upload();


    gpu->gpu_cutoff->gradCutoff = *gradCutoff;
    gpu->gpu_sim.gradCutoff = gpu->gpu_cutoff->gradCutoff;

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("UPLOAD GRAD", time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("COMPLETE UPLOADING GRAD");
}


//-----------------------------------------------
//  upload information for LRI calculation
//-----------------------------------------------
extern "C" void gpu_upload_lri_(QUICKDouble* zeta, QUICKDouble* cc, int *ierr)
{
    gpu->lri_data = new lri_data_type;

    gpu->lri_data->zeta = 0;
    gpu->lri_data->cc = new gpu_buffer_type<QUICKDouble>(cc, gpu->natom+gpu->nextatom);
    gpu->lri_data->cc->Upload();
    gpu->lri_data->vrecip = NULL;

    gpu->gpu_sim.lri_zeta = *zeta;

    /*    printf("zeta %f \n", gpu->gpu_sim.lri_zeta);

          for(int i=0; i < (gpu->natom+gpu->nextatom); i++)
          printf("cc %d %f \n", i, gpu->lri_data->cc->_hostData[i]);

          for(int iatom=0; iatom < (gpu->natom+gpu->nextatom); iatom++)
          printf("allxyz %d %f %f %f \n", iatom, LOC2( gpu->allxyz->_hostData, 0, iatom, 3, devSim.natom+devSim.nextatom),\
          LOC2( gpu->allxyz->_hostData, 1, iatom, 3, devSim.natom+devSim.nextatom), LOC2( gpu->allxyz->_hostData, 2, iatom, 3, devSim.natom+devSim.nextatom));
     */
    gpu->gpu_sim.lri_cc = gpu->lri_data->cc->_devData;
}


//-----------------------------------------------
//  upload information for CEW quad calculation
//-----------------------------------------------
extern "C" void gpu_upload_cew_vrecip_(int *ierr)
{
    gpu->lri_data->vrecip = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);

    QUICKDouble *gridpt = new QUICKDouble[3];

    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
        gridpt[0] = gpu->gpu_xcq->gridx->_hostData[i];
        gridpt[1] = gpu->gpu_xcq->gridy->_hostData[i];
        gridpt[2] = gpu->gpu_xcq->gridz->_hostData[i];

        QUICKDouble vrecip = 0.0;

#ifdef CEW
        cew_getpotatpt_(gridpt, &vrecip);
#endif

        gpu->lri_data->vrecip->_hostData[i] = -vrecip;
    }

    gpu->lri_data->vrecip->Upload();
    gpu->gpu_sim.cew_vrecip = gpu->lri_data->vrecip->_devData;
}


//Computes grid weights before grid point packing
extern "C" void gpu_get_ssw_(QUICKDouble *gridx, QUICKDouble *gridy, QUICKDouble *gridz,
        QUICKDouble *wtang, QUICKDouble *rwt, QUICKDouble *rad3, QUICKDouble *sswt,
        QUICKDouble *weight, int *gatm, int *count) {
    PRINTDEBUG("BEGIN TO COMPUTE SSW");

    gpu->gpu_xcq->npoints = *count;
    gpu->xc_threadsPerBlock = SM_2X_XC_THREADS_PER_BLOCK;

    gpu->gpu_xcq->gridx = new gpu_buffer_type<QUICKDouble>(gridx, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridy = new gpu_buffer_type<QUICKDouble>(gridy, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridz = new gpu_buffer_type<QUICKDouble>(gridz, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->wtang = new gpu_buffer_type<QUICKDouble>(wtang, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->rwt   = new gpu_buffer_type<QUICKDouble>(rwt, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->rad3  = new gpu_buffer_type<QUICKDouble>(rad3, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gatm  = new gpu_buffer_type<int>(gatm, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->sswt  = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->weight= new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);

    gpu->gpu_xcq->gridx->Upload();
    gpu->gpu_xcq->gridy->Upload();
    gpu->gpu_xcq->gridz->Upload();
    gpu->gpu_xcq->wtang->Upload();
    gpu->gpu_xcq->rwt->Upload();
    gpu->gpu_xcq->rad3->Upload();
    gpu->gpu_xcq->gatm->Upload();
    gpu->gpu_xcq->sswt->Upload();
    gpu->gpu_xcq->weight->Upload();

    gpu->gpu_sim.npoints = gpu->gpu_xcq->npoints;
    gpu->gpu_sim.gridx = gpu->gpu_xcq->gridx->_devData;
    gpu->gpu_sim.gridy = gpu->gpu_xcq->gridy->_devData;
    gpu->gpu_sim.gridz = gpu->gpu_xcq->gridz->_devData;
    gpu->gpu_sim.wtang = gpu->gpu_xcq->wtang->_devData;
    gpu->gpu_sim.rwt = gpu->gpu_xcq->rwt->_devData;
    gpu->gpu_sim.rad3 = gpu->gpu_xcq->rad3->_devData;
    gpu->gpu_sim.gatm = gpu->gpu_xcq->gatm->_devData;
    gpu->gpu_sim.sswt = gpu->gpu_xcq->sswt->_devData;
    gpu->gpu_sim.weight = gpu->gpu_xcq->weight->_devData;

    upload_sim_to_constant_dft(gpu);

    get_ssw(gpu);

    gpu->gpu_xcq->sswt->Download();
    gpu->gpu_xcq->weight->Download();

    for (int i = 0; i < *count; i++) {
        sswt[i] = gpu->gpu_xcq->sswt->_hostData[i];
        weight[i] = gpu->gpu_xcq->weight->_hostData[i];
    }

    SAFE_DELETE(gpu->gpu_xcq->gridx);
    SAFE_DELETE(gpu->gpu_xcq->gridy);
    SAFE_DELETE(gpu->gpu_xcq->gridz);
    SAFE_DELETE(gpu->gpu_xcq->wtang);
    SAFE_DELETE(gpu->gpu_xcq->rwt);
    SAFE_DELETE(gpu->gpu_xcq->rad3);
    SAFE_DELETE(gpu->gpu_xcq->gatm);
    SAFE_DELETE(gpu->gpu_xcq->sswt);
    SAFE_DELETE(gpu->gpu_xcq->weight);

    PRINTDEBUG("END COMPUTE SSW");
}


void prune_grid_sswgrad()
{
    PRINTDEBUG("BEGIN TO UPLOAD DFT GRID FOR SSWGRAD");
#if defined(MPIV_GPU)
    GPU_TIMER_CREATE();
#endif

    gpu->gpu_xcq->dweight_ssd->Download();
    gpu->gpu_xcq->exc->Download();

    //Get the size of input arrays to sswgrad computation
    int count = 0;
    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
        count += gpu->gpu_xcq->dweight_ssd->_hostData[i];
    }

    //Load data into temporary arrays
    QUICKDouble *tmp_gridx, *tmp_gridy, *tmp_gridz, *tmp_exc, *tmp_quadwt;
    int* tmp_gatm;
    int dbyte_size = sizeof(QUICKDouble)*count;

    tmp_gridx = (QUICKDouble *) malloc(dbyte_size);
    tmp_gridy = (QUICKDouble *) malloc(dbyte_size);
    tmp_gridz = (QUICKDouble *) malloc(dbyte_size);
    tmp_exc = (QUICKDouble *) malloc(dbyte_size);
    tmp_quadwt= (QUICKDouble *) malloc(dbyte_size);
    tmp_gatm = (int *) malloc(sizeof(int) * count);

    int j = 0;
    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
        if (gpu->gpu_xcq->dweight_ssd->_hostData[i] > 0) {
            tmp_gridx[j] = gpu->gpu_xcq->gridx->_hostData[i];
            tmp_gridy[j] = gpu->gpu_xcq->gridy->_hostData[i];
            tmp_gridz[j] = gpu->gpu_xcq->gridz->_hostData[i];
            tmp_exc[j] = gpu->gpu_xcq->exc->_hostData[i];
            tmp_quadwt[j] = gpu->gpu_xcq->weight->_hostData[i] / gpu->gpu_xcq->sswt->_hostData[i];
            tmp_gatm[j] = gpu->gpu_xcq->gatm->_hostData[i];
            j++;
        }
    }

    gpu_delete_dft_grid_();

#if defined(MPIV_GPU)
    GPU_TIMER_START();

    int netgain = getAdjustment(gpu->mpisize, gpu->mpirank, count);
    count += netgain;

    GPU_TIMER_STOP();
    gpu->timer->t_xcrb += (double) time / 1000.0;
#endif

    gpu->gpu_xcq->npoints_ssd = count;

    //Upload data using templates
#if defined(MPIV_GPU)
    gpu->gpu_xcq->gridx_ssd = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gridy_ssd = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gridz_ssd = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->exc_ssd = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->quadwt = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gatm_ssd = new gpu_buffer_type<int>(gpu->gpu_xcq->npoints_ssd);

    GPU_TIMER_START();

    sswderRedistribute(gpu->mpisize, gpu->mpirank, count-netgain, count,
            tmp_gridx, tmp_gridy, tmp_gridz, tmp_exc, tmp_quadwt, tmp_gatm, gpu->gpu_xcq->gridx_ssd->_hostData,
            gpu->gpu_xcq->gridy_ssd->_hostData, gpu->gpu_xcq->gridz_ssd->_hostData,
            gpu->gpu_xcq->exc_ssd->_hostData, gpu->gpu_xcq->quadwt->_hostData,
            gpu->gpu_xcq->gatm_ssd->_hostData);

    GPU_TIMER_STOP();
    gpu->timer->t_xcrb += (double) time / 1000.0;

    GPU_TIMER_START();
#else
    gpu->gpu_xcq->gridx_ssd = new gpu_buffer_type<QUICKDouble>(tmp_gridx, gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gridy_ssd = new gpu_buffer_type<QUICKDouble>(tmp_gridy, gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gridz_ssd = new gpu_buffer_type<QUICKDouble>(tmp_gridz, gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->exc_ssd = new gpu_buffer_type<QUICKDouble>(tmp_exc, gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->quadwt = new gpu_buffer_type<QUICKDouble>(tmp_quadwt, gpu->gpu_xcq->npoints_ssd);
    gpu->gpu_xcq->gatm_ssd = new gpu_buffer_type<int>(tmp_gatm, gpu->gpu_xcq->npoints_ssd);
#endif
    gpu->gpu_xcq->uw_ssd= new gpu_buffer_type<QUICKDouble>(gpu->blocks * gpu->xc_threadsPerBlock * gpu->natom * 3);

    gpu->gpu_xcq->gridx_ssd->Upload();
    gpu->gpu_xcq->gridy_ssd->Upload();
    gpu->gpu_xcq->gridz_ssd->Upload();
    gpu->gpu_xcq->exc_ssd->Upload();
    gpu->gpu_xcq->quadwt->Upload();
    gpu->gpu_xcq->gatm_ssd->Upload();

    gpu->gpu_sim.npoints_ssd  = gpu->gpu_xcq->npoints_ssd;
    gpu->gpu_sim.gridx_ssd = gpu->gpu_xcq->gridx_ssd->_devData;
    gpu->gpu_sim.gridy_ssd = gpu->gpu_xcq->gridy_ssd->_devData;
    gpu->gpu_sim.gridz_ssd = gpu->gpu_xcq->gridz_ssd->_devData;
    gpu->gpu_sim.exc_ssd = gpu->gpu_xcq->exc_ssd->_devData;
    gpu->gpu_sim.quadwt = gpu->gpu_xcq->quadwt->_devData;
    gpu->gpu_sim.uw_ssd = gpu->gpu_xcq->uw_ssd->_devData;
    gpu->gpu_sim.gatm_ssd = gpu->gpu_xcq->gatm_ssd->_devData;

    upload_sim_to_constant_dft(gpu);

    PRINTDEBUG("COMPLETE UPLOADING DFT GRID FOR SSWGRAD");

    //Clean up temporary arrays
    free(tmp_gridx);
    free(tmp_gridy);
    free(tmp_gridz);
    free(tmp_exc);
    free(tmp_quadwt);
    free(tmp_gatm);

#if defined(MPIV_GPU)
    GPU_TIMER_STOP();
    gpu->timer->t_xcpg += (double) time / 1000.0;
    GPU_TIMER_DESTROY();
#endif
}


void gpu_get_octree_info(QUICKDouble *gridx, QUICKDouble *gridy, QUICKDouble *gridz,
        QUICKDouble *sigrad2, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight,
        int *bin_locator, int count, double DMCutoff, double XCCutoff, int nbins)
{
    PRINTDEBUG("BEGIN TO OBTAIN PRIMITIVE & BASIS FUNCTION LISTS ");

    gpu->gpu_xcq->npoints = count;
    gpu->xc_threadsPerBlock = SM_2X_XCGRAD_THREADS_PER_BLOCK;

    gpu->gpu_xcq->gridx = new gpu_buffer_type<QUICKDouble>(gridx, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridy = new gpu_buffer_type<QUICKDouble>(gridy, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridz = new gpu_buffer_type<QUICKDouble>(gridz, gpu->gpu_xcq->npoints);
    gpu->gpu_basis->sigrad2 = new gpu_buffer_type<QUICKDouble>(sigrad2, gpu->nbasis);
    gpu->gpu_xcq->bin_locator = new gpu_buffer_type<int>(bin_locator,gpu->gpu_xcq->npoints);

    gpu->gpu_xcq->gridx->Upload();
    gpu->gpu_xcq->gridy->Upload();
    gpu->gpu_xcq->gridz->Upload();
    gpu->gpu_basis->sigrad2->Upload();
    gpu->gpu_xcq->bin_locator->Upload();

    gpu->gpu_sim.npoints = gpu->gpu_xcq->npoints;
    gpu->gpu_sim.gridx = gpu->gpu_xcq->gridx->_devData;
    gpu->gpu_sim.gridy = gpu->gpu_xcq->gridy->_devData;
    gpu->gpu_sim.gridz = gpu->gpu_xcq->gridz->_devData;
    gpu->gpu_sim.sigrad2 = gpu->gpu_basis->sigrad2->_devData;
    gpu->gpu_sim.bin_locator = gpu->gpu_xcq->bin_locator->_devData;

    gpu->gpu_cutoff->DMCutoff = DMCutoff;
    gpu->gpu_sim.DMCutoff = gpu->gpu_cutoff->DMCutoff;
    gpu->gpu_cutoff->XCCutoff = XCCutoff;
    gpu->gpu_sim.XCCutoff = gpu->gpu_cutoff->XCCutoff;

    //Define cfweight and pfweight arrays seperately and uplaod to gpu until we solve the problem with atomicAdd
    unsigned char *d_gpweight;
    unsigned int *d_cfweight, *d_pfweight;

    gpuMalloc((void**) &d_gpweight, sizeof(unsigned char) * gpu->gpu_xcq->npoints);
    gpuMalloc((void**) &d_cfweight, sizeof(unsigned int) * nbins * gpu->nbasis);
    gpuMalloc((void**) &d_pfweight, sizeof(unsigned int) * nbins * gpu->nbasis * gpu->gpu_basis->maxcontract);

    gpuMemcpy(d_gpweight, gpweight, sizeof(unsigned char) * gpu->gpu_xcq->npoints, cudaMemcpyHostToDevice);
    gpuMemcpy(d_cfweight, cfweight, sizeof(unsigned int) * nbins * gpu->nbasis, cudaMemcpyHostToDevice);
    gpuMemcpy(d_pfweight, pfweight, sizeof(unsigned int) * nbins * gpu->nbasis * gpu->gpu_basis->maxcontract, cudaMemcpyHostToDevice);

    upload_sim_to_constant_dft(gpu);

    /*        for(int i=0; i<nbins;i++){
    //unsigned int cfweight_sum =0;
    for(int j=0; j<gpu->nbasis; j++){
    printf("bin id: %i basis id: %i cfcount: %i \n", i, j, cfweight[(i * gpu->nbasis) + j]);
    //cfweight_sum += cfweight[ (nbins*gpu->nbasis) + j];
    }
    //printf("bin id: %i cfweight_sum: %i", i, cfweight_sum);
    }
     */

    /*        for(int i=0; i<nbins;i++){
              for(int j=0; j<gpu->nbasis; j++){
              for(int k=0; k<gpu->gpu_basis->maxcontract;k++){
              printf("bin id: %i basis id: %i cfcount: %i pf id: %i pfcount: %i \n", i, j, cfweight[(i * gpu->nbasis) + j], k, pfweight[(i * gpu->nbasis * gpu->gpu_basis->maxcontract) + j*gpu->gpu_basis->maxcontract + k]);
              }
              }
              }
     */
    get_primf_contraf_lists(gpu, d_gpweight, d_cfweight, d_pfweight);

    gpuMemcpy(gpweight, d_gpweight, gpu->gpu_xcq->npoints * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    gpuMemcpy(cfweight, d_cfweight, nbins * gpu->nbasis * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    gpuMemcpy(pfweight, d_pfweight, nbins * gpu->nbasis * gpu->gpu_basis->maxcontract * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /*      for(int i=0; i<nbins;i++){
    //unsigned int cfweight_sum =0;
    for(int j=0; j<gpu->nbasis; j++){
    printf("bin id: %i basis id: %i cfcount: %i \n", i, j, cfweight[(i * gpu->nbasis) + j]);
    //cfweight_sum += cfweight[ (nbins*gpu->nbasis) + j];
    }
    //printf("bin id: %i cfweight_sum: %i", i, cfweight_sum);
    }


    for(int i=0; i<nbins;i++){
    for(int j=0; j<gpu->nbasis; j++){
    for(int k=0; k<gpu->gpu_basis->maxcontract;k++){
    printf("bin id: %i basis id: %i cfcount: %i pf id: %i pfcount: %i \n", i, j, cfweight[(i * gpu->nbasis) + j], k, pfweight[(i * gpu->nbasis * gpu->gpu_basis->maxcontract) + j*gpu->gpu_basis->maxcontract + k]);
    }
    }
    }


    for(int i=0;i<gpu->gpu_xcq->npoints;i++){
    gpweight[i] = gpu->gpu_xcq->gpweight->_hostData[i];
    for(int j=0; j< gpu->nbasis; j++){
    cfweight[j+i * gpu->nbasis] = gpu->gpu_xcq->cfweight->_hostData[j+i * gpu->nbasis];
    for(int k=0; k<gpu->gpu_basis->maxcontract;k++){
    pfweight[k + j * gpu->gpu_basis->maxcontract + i * gpu->nbasis * gpu->gpu_basis->maxcontract] = gpu->gpu_xcq->pfweight->_hostData[k + j * gpu->gpu_basis->maxcontract + i * gpu->nbasis * gpu->gpu_basis->maxcontract];
    //printf("gp: %i gpw: %i cf: %i cfw: %i pf: %i pfw: %i \n", i, gpu->gpu_xcq->gpweight->_hostData[i], j, gpu->gpu_xcq->cfweight->_hostData[j+i * gpu->nbasis], k, gpu->gpu_xcq->pfweight->_hostData[k + j * gpu->gpu_basis->maxcontract + i * gpu->nbasis * gpu->gpu_basis->maxcontract]);

    }
    }
    }
     */

    SAFE_DELETE(gpu->gpu_xcq->gridx);
    SAFE_DELETE(gpu->gpu_xcq->gridy);
    SAFE_DELETE(gpu->gpu_xcq->gridz);
    SAFE_DELETE(gpu->gpu_basis->sigrad2);
    SAFE_DELETE(gpu->gpu_xcq->bin_locator);
    gpuFree(d_gpweight);
    gpuFree(d_cfweight);
    gpuFree(d_pfweight);

    PRINTDEBUG("PRIMITIVE & BASIS FUNCTION LISTS OBTAINED");
}


#ifdef DEBUG
void print_uploaded_dft_info()
{
    PRINTDEBUG("PRINTING UPLOADED DFT DATA");

    fprintf(gpu->debugFile,"Number of grid points: %i \n", gpu->gpu_xcq->npoints);
    fprintf(gpu->debugFile,"Bin size: %i \n", gpu->gpu_xcq->bin_size);
    fprintf(gpu->debugFile,"Number of bins: %i \n", gpu->gpu_xcq->nbins);
    fprintf(gpu->debugFile,"Number of total basis functions: %i \n", gpu->gpu_xcq->ntotbf);
    fprintf(gpu->debugFile,"Number of total primitive functions: %i \n", gpu->gpu_xcq->ntotpf);

    PRINTDEBUG("GRID POINTS & WEIGHTS");

    for (int i = 0; i < gpu->gpu_xcq->npoints; i++) {
        fprintf(gpu->debugFile, "Grid: %i x=%f y=%f z=%f sswt=%f weight=%f gatm=%i dweight_ssd=%i \n", i,
                gpu->gpu_xcq->gridx->_hostData[i], gpu->gpu_xcq->gridy->_hostData[i], gpu->gpu_xcq->gridz->_hostData[i],
                gpu->gpu_xcq->sswt->_hostData[i], gpu->gpu_xcq->weight->_hostData[i], gpu->gpu_xcq->gatm->_hostData[i],
                gpu->gpu_xcq->dweight_ssd->_hostData[i]);
    }

    PRINTDEBUG("BASIS & PRIMITIVE FUNCTION LISTS");

    for (int bin_id = 0; bin_id < gpu->gpu_xcq->nbins; bin_id++) {
        for (int i = gpu->gpu_xcq->basf_locator->_hostData[bin_id]; i < gpu->gpu_xcq->basf_locator->_hostData[bin_id + 1]; i++) {
            for (int j = gpu->gpu_xcq->primf_locator->_hostData[i]; j < gpu->gpu_xcq->primf_locator->_hostData[i + 1]; j++) {
                fprintf(gpu->debugFile, "Bin ID= %i basf location= %i ibas= %i primf location= %i jprim= %i \n", bin_id, i,
                        gpu->gpu_xcq->basf->_hostData[i], j, gpu->gpu_xcq->primf->_hostData[j]);
            }
        }
    }

    PRINTDEBUG("RADIUS OF SIGNIFICANCE");

    for (int i = 0; i < gpu->nbasis; i++) {
        fprintf(gpu->debugFile, "ibas=%i sigrad2=%f \n", i, gpu->gpu_basis->sigrad2->_hostData[i]);
    }

    for (int i = 0; i < gpu->nbasis; i++) {
        for (int j = 0; j < gpu->gpu_basis->maxcontract; j++) {
            fprintf(gpu->debugFile, "ibas=%i jprim=%i dcoeff=%f \n", i, j,
                    gpu->gpu_basis->dcoeff->_hostData[j + i * gpu->gpu_basis->maxcontract]);
        }
    }

    PRINTDEBUG("END PRINTING UPLOADED DFT DATA");
}
#endif


extern "C" void gpu_upload_dft_grid_(QUICKDouble *gridxb, QUICKDouble *gridyb, QUICKDouble *gridzb,
        QUICKDouble *gridb_sswt, QUICKDouble *gridb_weight, int *gridb_atm, int *bin_locator,
        int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter,
        int *gridb_count, int *nbins, int *nbtotbf, int *nbtotpf, int *isg, QUICKDouble *sigrad2,
        QUICKDouble *DMCutoff, QUICKDouble *XCCutoff) {
    PRINTDEBUG("BEGIN TO UPLOAD DFT GRID");

    gpu->gpu_xcq->npoints       = *gridb_count;
    gpu->gpu_xcq->nbins         = *nbins;
    gpu->gpu_xcq->ntotbf        = *nbtotbf;
    gpu->gpu_xcq->ntotpf        = *nbtotpf;
    //      gpu->gpu_xcq->bin_size      = (int) (*gridb_count / *nbins);
    gpu->gpu_cutoff->DMCutoff   = *DMCutoff;
    gpu->gpu_cutoff->XCCutoff   = *XCCutoff;

    gpu->gpu_xcq->gridx = new gpu_buffer_type<QUICKDouble>(gridxb, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridy = new gpu_buffer_type<QUICKDouble>(gridyb, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gridz = new gpu_buffer_type<QUICKDouble>(gridzb, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->sswt  = new gpu_buffer_type<QUICKDouble>(gridb_sswt, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->weight        = new gpu_buffer_type<QUICKDouble>(gridb_weight, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gatm          = new gpu_buffer_type<int>(gridb_atm, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->dweight_ssd   = new gpu_buffer_type<int>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->basf  = new gpu_buffer_type<int>(basf, gpu->gpu_xcq->ntotbf);
    gpu->gpu_xcq->primf = new gpu_buffer_type<int>(primf, gpu->gpu_xcq->ntotpf);
    gpu->gpu_xcq->basf_locator     = new gpu_buffer_type<int>(basf_counter, gpu->gpu_xcq->nbins +1);
    gpu->gpu_xcq->primf_locator    = new gpu_buffer_type<int>(primf_counter, gpu->gpu_xcq->ntotbf +1);
    gpu->gpu_basis->sigrad2 = new gpu_buffer_type<QUICKDouble>(sigrad2, gpu->nbasis);
    gpu->gpu_xcq->bin_locator       = new gpu_buffer_type<int>(bin_locator, gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->bin_counter       = new gpu_buffer_type<int>(bin_counter, gpu->gpu_xcq->nbins +1);

    for(int i=0; i< gpu->gpu_xcq->npoints; ++i)
        gpu->gpu_xcq->dweight_ssd->_hostData[i] =1;

#if defined(MPIV_GPU)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();

//    mgpu_xc_naive_distribute();
    mgpu_xc_tpbased_greedy_distribute();
//    mgpu_xc_pbased_greedy_distribute();
    mgpu_xc_repack();

    GPU_TIMER_STOP();
    gpu->timer->t_xclb += (double) time / 1000.0;
    GPU_TIMER_DESTROY();

    gpu->gpu_sim.mpirank = gpu->mpirank;
    gpu->gpu_sim.mpisize = gpu->mpisize;
#endif

    gpu->xc_threadsPerBlock = SM_2X_XC_THREADS_PER_BLOCK;
    gpu->gpu_xcq->densa = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->densb = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gax = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gbx = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gay = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gby = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gaz = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->gbz = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);
    gpu->gpu_xcq->exc = new gpu_buffer_type<QUICKDouble>(gpu->gpu_xcq->npoints);

    gpu->gpu_xcq->gridx->UploadAsync();
    gpu->gpu_xcq->gridy->UploadAsync();
    gpu->gpu_xcq->gridz->UploadAsync();
    gpu->gpu_xcq->sswt->UploadAsync();
    gpu->gpu_xcq->weight->UploadAsync();
    gpu->gpu_xcq->gatm->UploadAsync();
    gpu->gpu_xcq->dweight_ssd->UploadAsync();
    gpu->gpu_xcq->basf->UploadAsync();
    gpu->gpu_xcq->primf->UploadAsync();
    gpu->gpu_xcq->bin_locator->UploadAsync();
    gpu->gpu_xcq->basf_locator->UploadAsync();
    gpu->gpu_xcq->primf_locator->UploadAsync();
    gpu->gpu_basis->sigrad2->UploadAsync();
    gpu->gpu_xcq->densa->UploadAsync();
    gpu->gpu_xcq->densb->UploadAsync();
    gpu->gpu_xcq->gax->UploadAsync();
    gpu->gpu_xcq->gbx->UploadAsync();
    gpu->gpu_xcq->gay->UploadAsync();
    gpu->gpu_xcq->gby->UploadAsync();
    gpu->gpu_xcq->gaz->UploadAsync();
    gpu->gpu_xcq->gbz->UploadAsync();
    gpu->gpu_xcq->exc->UploadAsync();

    gpu->gpu_sim.npoints = gpu->gpu_xcq->npoints;
    gpu->gpu_sim.nbins = gpu->gpu_xcq->nbins;
    gpu->gpu_sim.ntotbf = gpu->gpu_xcq->ntotbf;
    gpu->gpu_sim.ntotpf = gpu->gpu_xcq->ntotpf;
    gpu->gpu_sim.bin_size = gpu->gpu_xcq->bin_size;
    gpu->gpu_sim.gridx = gpu->gpu_xcq->gridx->_devData;
    gpu->gpu_sim.gridy = gpu->gpu_xcq->gridy->_devData;
    gpu->gpu_sim.gridz = gpu->gpu_xcq->gridz->_devData;
    gpu->gpu_sim.sswt = gpu->gpu_xcq->sswt->_devData;
    gpu->gpu_sim.weight = gpu->gpu_xcq->weight->_devData;
    gpu->gpu_sim.gatm = gpu->gpu_xcq->gatm->_devData;
    gpu->gpu_sim.dweight_ssd = gpu->gpu_xcq->dweight_ssd->_devData;
    gpu->gpu_sim.basf = gpu->gpu_xcq->basf->_devData;
    gpu->gpu_sim.primf = gpu->gpu_xcq->primf->_devData;
    gpu->gpu_sim.bin_locator = gpu->gpu_xcq->bin_locator->_devData;
    gpu->gpu_sim.basf_locator = gpu->gpu_xcq->basf_locator->_devData;
    gpu->gpu_sim.primf_locator = gpu->gpu_xcq->primf_locator->_devData;
    gpu->gpu_sim.densa = gpu->gpu_xcq->densa->_devData;
    gpu->gpu_sim.densb = gpu->gpu_xcq->densb->_devData;
    gpu->gpu_sim.gax = gpu->gpu_xcq->gax->_devData;
    gpu->gpu_sim.gbx = gpu->gpu_xcq->gbx->_devData;
    gpu->gpu_sim.gay = gpu->gpu_xcq->gay->_devData;
    gpu->gpu_sim.gby = gpu->gpu_xcq->gby->_devData;
    gpu->gpu_sim.gaz = gpu->gpu_xcq->gaz->_devData;
    gpu->gpu_sim.gbz = gpu->gpu_xcq->gbz->_devData;
    gpu->gpu_sim.exc = gpu->gpu_xcq->exc->_devData;
    gpu->gpu_sim.sigrad2 = gpu->gpu_basis->sigrad2->_devData;
    gpu->gpu_sim.isg = *isg;
    gpu->gpu_sim.DMCutoff = gpu->gpu_cutoff->DMCutoff;
    gpu->gpu_sim.XCCutoff = gpu->gpu_cutoff->XCCutoff;

//    upload_xc_smem();
    upload_pteval();

#ifdef DEBUG
    print_uploaded_dft_info();
#endif

    PRINTDEBUG("COMPLETE UPLOADING DFT GRID");
}


//-----------------------------------------------
// Reupload dft data
//-----------------------------------------------
extern "C" void gpu_reupload_dft_grid_()
{
    PRINTDEBUG("BEGIN TO UPLOAD DFT GRID");

    gpu->gpu_xcq->gridx->ReallocateGPU();
    gpu->gpu_xcq->gridy->ReallocateGPU();
    gpu->gpu_xcq->gridz->ReallocateGPU();
    gpu->gpu_xcq->sswt->ReallocateGPU();
    gpu->gpu_xcq->weight->ReallocateGPU();
    gpu->gpu_xcq->gatm->ReallocateGPU();
    gpu->gpu_xcq->dweight_ssd->ReallocateGPU();
    gpu->gpu_xcq->basf->ReallocateGPU();
    gpu->gpu_xcq->primf->ReallocateGPU();
    gpu->gpu_xcq->bin_locator->ReallocateGPU();
    gpu->gpu_xcq->basf_locator->ReallocateGPU();
    gpu->gpu_xcq->primf_locator->ReallocateGPU();
    gpu->gpu_xcq->densa->ReallocateGPU();
    gpu->gpu_xcq->densb->ReallocateGPU();
    gpu->gpu_xcq->gax->ReallocateGPU();
    gpu->gpu_xcq->gbx->ReallocateGPU();
    gpu->gpu_xcq->gay->ReallocateGPU();
    gpu->gpu_xcq->gby->ReallocateGPU();
    gpu->gpu_xcq->gaz->ReallocateGPU();
    gpu->gpu_xcq->gbz->ReallocateGPU();
    gpu->gpu_xcq->exc->ReallocateGPU();
    gpu->gpu_basis->sigrad2->ReallocateGPU();

    gpu->gpu_xcq->gridx->Upload();
    gpu->gpu_xcq->gridy->Upload();
    gpu->gpu_xcq->gridz->Upload();
    gpu->gpu_xcq->sswt->Upload();
    gpu->gpu_xcq->weight->Upload();
    gpu->gpu_xcq->gatm->Upload();
    gpu->gpu_xcq->dweight_ssd->Upload();
    gpu->gpu_xcq->basf->Upload();
    gpu->gpu_xcq->primf->Upload();
    gpu->gpu_xcq->bin_locator->Upload();
    gpu->gpu_xcq->basf_locator->Upload();
    gpu->gpu_xcq->primf_locator->Upload();
    gpu->gpu_basis->sigrad2->Upload();
    gpu->gpu_xcq->densa->Upload();
    gpu->gpu_xcq->densb->Upload();
    gpu->gpu_xcq->gax->Upload();
    gpu->gpu_xcq->gbx->Upload();
    gpu->gpu_xcq->gay->Upload();
    gpu->gpu_xcq->gby->Upload();
    gpu->gpu_xcq->gaz->Upload();
    gpu->gpu_xcq->gbz->Upload();
    gpu->gpu_xcq->exc->Upload();
    gpu->gpu_basis->sigrad2->Upload();

    gpu->gpu_sim.gridx     = gpu->gpu_xcq->gridx->_devData;
    gpu->gpu_sim.gridy     = gpu->gpu_xcq->gridy->_devData;
    gpu->gpu_sim.gridz     = gpu->gpu_xcq->gridz->_devData;
    gpu->gpu_sim.sswt      = gpu->gpu_xcq->sswt->_devData;
    gpu->gpu_sim.weight    = gpu->gpu_xcq->weight->_devData;
    gpu->gpu_sim.gatm      = gpu->gpu_xcq->gatm->_devData;
    gpu->gpu_sim.dweight_ssd   = gpu->gpu_xcq->dweight_ssd->_devData;
    gpu->gpu_sim.basf      = gpu->gpu_xcq->basf->_devData;
    gpu->gpu_sim.primf     = gpu->gpu_xcq->primf->_devData;
    gpu->gpu_sim.bin_locator      = gpu->gpu_xcq->bin_locator->_devData;
    gpu->gpu_sim.basf_locator      = gpu->gpu_xcq->basf_locator->_devData;
    gpu->gpu_sim.primf_locator     = gpu->gpu_xcq->primf_locator->_devData;
    gpu->gpu_sim.densa     = gpu->gpu_xcq->densa->_devData;
    gpu->gpu_sim.densb     = gpu->gpu_xcq->densb->_devData;
    gpu->gpu_sim.gax     = gpu->gpu_xcq->gax->_devData;
    gpu->gpu_sim.gbx     = gpu->gpu_xcq->gbx->_devData;
    gpu->gpu_sim.gay     = gpu->gpu_xcq->gay->_devData;
    gpu->gpu_sim.gby     = gpu->gpu_xcq->gby->_devData;
    gpu->gpu_sim.gaz     = gpu->gpu_xcq->gaz->_devData;
    gpu->gpu_sim.gbz     = gpu->gpu_xcq->gbz->_devData;
    gpu->gpu_sim.exc     = gpu->gpu_xcq->exc->_devData;
    gpu->gpu_sim.sigrad2 = gpu->gpu_basis->sigrad2->_devData;

    reupload_pteval( );

    PRINTDEBUG("COMPLETE UPLOADING DFT GRID");
}


//-----------------------------------------------
// Delete dft device data
//-----------------------------------------------
extern "C" void gpu_delete_dft_dev_grid_()
{
    PRINTDEBUG("DEALLOCATING DFT GRID");

    gpu->gpu_xcq->gridx->DeleteGPU();
    gpu->gpu_xcq->gridy->DeleteGPU();
    gpu->gpu_xcq->gridz->DeleteGPU();
    gpu->gpu_xcq->sswt->DeleteGPU();
    gpu->gpu_xcq->weight->DeleteGPU();
    gpu->gpu_xcq->gatm->DeleteGPU();
    gpu->gpu_xcq->dweight_ssd->DeleteGPU();
    gpu->gpu_xcq->basf->DeleteGPU();
    gpu->gpu_xcq->primf->DeleteGPU();
    gpu->gpu_xcq->bin_locator->DeleteGPU();
    gpu->gpu_xcq->basf_locator->DeleteGPU();
    gpu->gpu_xcq->primf_locator->DeleteGPU();
    gpu->gpu_xcq->densa->DeleteGPU();
    gpu->gpu_xcq->densb->DeleteGPU();
    gpu->gpu_xcq->gax->DeleteGPU();
    gpu->gpu_xcq->gbx->DeleteGPU();
    gpu->gpu_xcq->gay->DeleteGPU();
    gpu->gpu_xcq->gby->DeleteGPU();
    gpu->gpu_xcq->gaz->DeleteGPU();
    gpu->gpu_xcq->gbz->DeleteGPU();
    gpu->gpu_xcq->exc->DeleteGPU();
    gpu->gpu_basis->sigrad2->DeleteGPU();

    delete_pteval(true);

    PRINTDEBUG("FINISHED DEALLOCATING DFT GRID");
}


extern "C" void gpu_delete_dft_grid_()
{
    PRINTDEBUG("DEALLOCATING DFT GRID");

    SAFE_DELETE(gpu->gpu_xcq->gridx);
    SAFE_DELETE(gpu->gpu_xcq->gridy);
    SAFE_DELETE(gpu->gpu_xcq->gridz);
    SAFE_DELETE(gpu->gpu_xcq->sswt);
    SAFE_DELETE(gpu->gpu_xcq->weight);
    SAFE_DELETE(gpu->gpu_xcq->gatm);
    SAFE_DELETE(gpu->gpu_xcq->dweight_ssd);
    SAFE_DELETE(gpu->gpu_xcq->basf);
    SAFE_DELETE(gpu->gpu_xcq->primf);
    SAFE_DELETE(gpu->gpu_xcq->bin_locator);
    SAFE_DELETE(gpu->gpu_xcq->basf_locator);
    SAFE_DELETE(gpu->gpu_xcq->primf_locator);
    SAFE_DELETE(gpu->gpu_xcq->bin_counter);
    SAFE_DELETE(gpu->gpu_xcq->densa);
    SAFE_DELETE(gpu->gpu_xcq->densb);
    SAFE_DELETE(gpu->gpu_xcq->gax);
    SAFE_DELETE(gpu->gpu_xcq->gbx);
    SAFE_DELETE(gpu->gpu_xcq->gay);
    SAFE_DELETE(gpu->gpu_xcq->gby);
    SAFE_DELETE(gpu->gpu_xcq->gaz);
    SAFE_DELETE(gpu->gpu_xcq->gbz);
    SAFE_DELETE(gpu->gpu_xcq->exc);
    SAFE_DELETE(gpu->gpu_basis->sigrad2);
//    SAFE_DELETE(gpu->gpu_xcq->primfpbin);
#ifdef MPIV_GPU
    SAFE_DELETE(gpu->gpu_xcq->mpi_bxccompute);
#endif
    delete_pteval(false);

    PRINTDEBUG("FINISHED DEALLOCATING DFT GRID");
}


void gpu_delete_sswgrad_vars()
{
    PRINTDEBUG("DEALLOCATING SSWGRAD VARIABLES");

    SAFE_DELETE(gpu->gpu_xcq->gridx_ssd);
    SAFE_DELETE(gpu->gpu_xcq->gridy_ssd);
    SAFE_DELETE(gpu->gpu_xcq->gridz_ssd);
    SAFE_DELETE(gpu->gpu_xcq->exc_ssd);
    SAFE_DELETE(gpu->gpu_xcq->quadwt);
    SAFE_DELETE(gpu->gpu_xcq->uw_ssd);
    SAFE_DELETE(gpu->gpu_xcq->gatm_ssd);
#ifdef MPIV_GPU
    SAFE_DELETE(gpu->gpu_xcq->mpi_bxccompute);
#endif

    PRINTDEBUG("FINISHED DEALLOCATING SSWGRAD VARIABLES");
}


extern "C" void gpu_cleanup_()
{
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
    SAFE_DELETE(gpu->gpu_basis->Xcoeff_oei);
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

    SAFE_DELETE(gpu->allxyz);
    SAFE_DELETE(gpu->allchg);
    SAFE_DELETE(gpu->gpu_cutoff->sorted_OEICutoffIJ);
}


#if defined(COMPILE_GPU_AOINT)
static bool debut = true;
static bool incoreInt = true;
static ERI_entry* intERIEntry;
static int totalBuffer;


extern "C" void gpu_addint_(QUICKDouble* o, int* intindex, char* intFileName)
{
#if defined(DEBUG)
    GPU_TIMER_CREATE();
    GPU_TIMER_START();
#endif

    PRINTDEBUG("BEGIN TO RUN ADD INT");

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
    int const availableERI = gpu->blocks * gpu->twoEThreadsPerBlock
        * (int) (availableMem/(gpu->blocks * gpu->twoEThreadsPerBlock) / sizeof(ERI_entry));

    int bufferIndex = 0;

    intFile = fopen(trim(intFileName), "rb");
    if (!intFile) {
        printf("UNABLE TO OPEN INT FILE\n");
    }
    rewind(intFile);

    cudaStream_t stream[streamNum];
    for (int i = 0; i<streamNum; i++) {
        cudaStreamCreate( &stream[i] );
    }

    PRINTDEBUG("BEGIN TO RUN KERNEL");

    upload_sim_to_constant(gpu);

#ifdef DEBUG
    fprintf(gpu->debugFile,"int total from addint = %i\n", *intindex);
#endif

    // Now begin to allocate AO INT space
    gpu->aoint_buffer = new gpu_buffer_type<ERI_entry>*[1];//(gpu_buffer_type<ERI_entry> **) malloc(sizeof(gpu_buffer_type<ERI_entry>*) * streamNum);
    gpu->gpu_sim.aoint_buffer = new ERI_entry*[1];

    // Zero them out
    gpu->aoint_buffer[0]                 = new gpu_buffer_type<ERI_entry>(availableERI, false);
    gpu->gpu_sim.aoint_buffer[0]         = gpu->aoint_buffer[0]->_devData;

#ifdef DEBUG
    fprintf(gpu->debugFile,"Total buffer pack = %i\n", bufferPackNum);
#endif

    if (incoreInt && debut) {
        ERI_entry* intERIEntry_tmp = new ERI_entry[*intindex];
        intERIEntry = new ERI_entry[*intindex];
        int* ERIEntryByBasis = new int[gpu->nbasis];

        for (int i = 0; i<gpu->nbasis; i++) {
            ERIEntryByBasis[i] = 0;
        }

        for ( int i = 0; i < bufferPackNum; i++) {
            if (remainingBuffer > BUFFERSIZE) {
                thisBuffer = BUFFERSIZE;
                remainingBuffer = remainingBuffer - BUFFERSIZE;
            } else {
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

            //gpu->aoint_buffer[0]->_hostData[currentInt].IJ    = intERIEntry[i].IJ;
            //gpu->aoint_buffer[0]->_hostData[currentInt].KL    = intERIEntry[i].KL;
            //gpu->aoint_buffer[0]->_hostData[currentInt].value = intERIEntry[i].value;

            currentInt++;

            if (currentInt >= availableERI) {
                //gpu->aoint_buffer[0]->Upload();
                //gpuMemcpy(gpu->aoint_buffer[0]->_devData, gpu->aoint_buffer[0]->_hostData, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
                gpuMemcpy(gpu->aoint_buffer[0]->_devData, intERIEntry + startingInt, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
                getAddInt(gpu, currentInt, gpu->gpu_sim.aoint_buffer[0]);
                currentInt = 0;
                startingInt = i;
            }
        }

        //gpu->aoint_buffer[0]->Upload();
        //gpuMemcpy(gpu->aoint_buffer[0]->_devData, gpu->aoint_buffer[0]->_hostData, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
        gpuMemcpy(gpu->aoint_buffer[0]->_devData, intERIEntry + startingInt, currentInt*sizeof(ERI_entry), cudaMemcpyHostToDevice);
        getAddInt(gpu, currentInt, gpu->gpu_sim.aoint_buffer[0]);
        bufferIndex = 0;
    } else {
        for ( int i = 0; i < bufferPackNum; i++) {
            if (remainingBuffer > BUFFERSIZE) {
                thisBuffer = BUFFERSIZE;
                remainingBuffer = remainingBuffer - BUFFERSIZE;
            } else {
                thisBuffer = remainingBuffer;
            }

#ifdef DEBUG
            fprintf(gpu->debugFile," For buffer pack %i, %i Entry is read.\n", i, thisBuffer);
#endif

            ERIRead = fread(&aBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&bBuffer,   sizeof(int),         thisBuffer, intFile);
            ERIRead = fread(&intBuffer, sizeof(QUICKDouble), thisBuffer, intFile);

            for (int j = 0; j < thisBuffer; j++) {
                gpu->aoint_buffer[0]->_hostData[bufferIndex].IJ    = aBuffer[j];
                gpu->aoint_buffer[0]->_hostData[bufferIndex].KL    = bBuffer[j];
                gpu->aoint_buffer[0]->_hostData[bufferIndex].value = intBuffer[j];
                bufferIndex ++;

                if (bufferIndex >= availableERI) {
                    //gpuMemcpyAsync(gpu->aoint_buffer[0]->_hostData, gpu->aoint_buffer[0]->_devData, bufferIndex*sizeof(ERI_entry), cudaMemcpyHostToDevice, stream[0]);
                    gpu->aoint_buffer[0]->Upload();
                    getAddInt(gpu, bufferIndex, gpu->gpu_sim.aoint_buffer[0]);
                    bufferIndex = 0;
                }
            }
        }

        gpu->aoint_buffer[0]->Upload();
        //gpuMemcpyAsync(gpu->aoint_buffer[0]->_hostData, gpu->aoint_buffer[0]->_devData, bufferIndex*sizeof(ERI_entry), cudaMemcpyHostToDevice, stream[0]);
        getAddInt(gpu, bufferIndex, gpu->gpu_sim.aoint_buffer[0]);
        bufferIndex = 0;
    }

    for (int i = 0; i<streamNum; i++) {
        delete gpu->aoint_buffer[i];
    }

    PRINTDEBUG("COMPLETE KERNEL");

#if defined(USE_LEGACY_ATOMICS)
    gpu->gpu_calculated->oULL->Download();
    
    for (int i = 0; i < gpu->nbasis; i++) {
        for (int j = i; j < gpu->nbasis; j++) {
            QUICKDouble val = ULLTODOUBLE(LOC2(gpu->gpu_calculated->oULL->_hostData, j, i, gpu->nbasis, gpu->nbasis))
                * ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->o->_hostData, i, j, gpu->nbasis, gpu->nbasis) = val;
            LOC2(gpu->gpu_calculated->o->_hostData, j, i, gpu->nbasis, gpu->nbasis) = val;
        }
    }
#else
    gpu->gpu_calculated->o->Download();

    for (int i = 0; i < gpu->nbasis; i++) {
        for (int j = i; j < gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData, i, j, gpu->nbasis, gpu->nbasis)
                = LOC2(gpu->gpu_calculated->o->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }
#endif
    gpu->gpu_calculated->o->Download(o);

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("ADD INT",time);
    GPU_TIMER_DESTROY();
#endif

    PRINTDEBUG("DELETE TEMP VARIABLES");

    delete gpu->gpu_calculated->o;
    delete gpu->gpu_calculated->dense;
    delete gpu->gpu_cutoff->cutMatrix;
    delete gpu->gpu_cutoff->sorted_YCutoffIJ;
    delete gpu->gpu_cutoff->YCutoff;
    delete gpu->gpu_cutoff->cutPrim;
#if defined(USE_LEGACY_ATOMICS)
    delete gpu->gpu_calculated->oULL;
#endif    

    PRINTDEBUG("COMPLETE RUNNING ADDINT");
}
#endif


#if defined(COMPILE_GPU_AOINT)
extern "C" void gpu_aoint_(QUICKDouble* leastIntegralCutoff, QUICKDouble* maxIntegralCutoff, int* intNum, char* intFileName)
{
    PRINTDEBUG("BEGIN TO RUN AOINT");

    ERI_entry a;
    FILE *intFile;
    intFile = fopen(trim(intFileName), "wb");

#if defined(DEBUG)
    if (!intFile) {
        fprintf(gpu->debugFile, "UNABLE TO OPEN INT FILE\n");
    }
#endif

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
    QUICKULL intTotal = gpu->gpu_cutoff->sqrQshell * gpu->gpu_cutoff->sqrQshell;

    for (int i = 0; i < intTotal; i++) {
        int intCount = 20;
        if (currentCount + intCount < availableERI) {
            currentCount = currentCount + intCount;
        } else {
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

#if defined(DEBUG)
    // List all the batches
    fprintf(gpu->debugFile,"batch count = %i\n", iBatchCount);
    fprintf(gpu->debugFile,"max int count = %i\n", maxIntCount * sizeof(ERI_entry));
    for (int i = 0; i<iBatchCount; i++) {
        fprintf(gpu->debugFile," %i from %i to %i %i\n", i, nIntStart[i], nIntEnd[i], nIntSize[i] * sizeof(ERI_entry));
    }
#endif

    int nBatchERICount = maxIntCount;

    // Now begin to allocate AO INT space
    gpu->aoint_buffer = new gpu_buffer_type<ERI_entry>*[streamNum];//(gpu_buffer_type<ERI_entry> **) malloc(sizeof(gpu_buffer_type<ERI_entry>*) * streamNum);
    gpu->gpu_sim.aoint_buffer = new ERI_entry*[streamNum];

    // Zero them out
    for (int i = 0; i<streamNum; i++) {
        gpu->aoint_buffer[i]                 = new gpu_buffer_type<ERI_entry>( nBatchERICount, false );
        gpu->gpu_sim.aoint_buffer[i]         = gpu->aoint_buffer[i]->_devData;
    }

    gpu->gpu_sim.leastIntegralCutoff  = *leastIntegralCutoff;
    gpu->gpu_sim.maxIntegralCutoff    = *maxIntegralCutoff;
    gpu->gpu_sim.iBatchSize           = nBatchERICount;
    gpu->intCount                     = new gpu_buffer_type<QUICKULL>(streamNum);
    gpu->gpu_sim.intCount = gpu->intCount->_devData;

    upload_sim_to_constant(gpu);

#if defined(DEBUG)
    float time_downloadERI, time_kernel, time_io;
    time_downloadERI = 0;
    time_io = 0;
    time_kernel = 0;
    GPU_TIMER_CREATE();
    clock_t start_cpu = clock();
#endif

    cudaStream_t stream[streamNum];
    for (int i = 0; i<streamNum; i++) {
        cudaStreamCreate( &stream[i] );
    }

    for (int iBatch = 0; iBatch < iBatchCount; iBatch = iBatch + streamNum) {
#if defined(DEBUG)
        fprintf(gpu->debugFile,"batch %i start %i end %i\n", iBatch, nIntStart[iBatch], nIntEnd[iBatch]);
        GPU_TIMER_START();
#endif
        for (int i = 0; i < streamNum; i++) {
            gpu->intCount->_hostData[i] = 0;
        }

        gpu->intCount->Upload();

        // calculate ERI, kernel part
        for (int i = 0; i< streamNum && iBatch + i < iBatchCount; i++) {
            getAOInt(gpu, nIntStart[iBatch + i], nIntEnd[iBatch + i], stream[i], i, gpu->gpu_sim.aoint_buffer[i]);
        }

        // download ERI from GPU, this is time-consuming part, that need to be reduced
        for (int i = 0; i < streamNum && iBatch + i < iBatchCount; i++) {
            gpuMemcpyAsync(&gpu->intCount->_hostData[i], &gpu->intCount->_devData[i],
                    sizeof(QUICKULL), cudaMemcpyDeviceToHost, stream[i]);
            gpuMemcpyAsync(gpu->aoint_buffer[i]->_hostData, gpu->aoint_buffer[i]->_devData,
                    sizeof(ERI_entry) * gpu->intCount->_hostData[i], cudaMemcpyDeviceToHost, stream[i]);
        }

#if defined(DEBUG)
        GPU_TIMER_STOP();
        PRINTUSINGTIME("KERNEL", time);
        time_kernel += time;

        GPU_TIMER_START();
#endif

        gpu->intCount->Download();

        for (int i = 0; i < streamNum && iBatch + i < iBatchCount; i++) {
            cudaStreamSynchronize(stream[i]);
#if defined(DEBUG)
            fprintf(gpu->debugFile, "none-sync intCount = %i\n", gpu->intCount->_hostData[i]);
#endif

            // write to in-memory buffer.
            for (int j = 0; j < gpu->intCount->_hostData[i]; j++) {
                a = gpu->aoint_buffer[i]->_hostData[j];
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

#if defined(DEBUG)
        GPU_TIMER_STOP();
        PRINTUSINGTIME("IO", time);
        time_io += time;
#endif
    }

    fwrite(&aBuffer, sizeof(int), bufferInt, intFile);
    fwrite(&bBuffer, sizeof(int), bufferInt, intFile);
    fwrite(&intBuffer, sizeof(QUICKDouble), bufferInt, intFile);
//    for (int k = 0; k < bufferInt; k++) {
//        printf("%i %i %i %18.10f\n", k, aBuffer[k], bBuffer[k], intBuffer[k]);
//    }

    bufferInt = 0;

    for (int i = 0; i < streamNum; i++) {
        delete gpu->aoint_buffer[i];
    }

#if defined(DEBUG)
    GPU_TIMER_START();
#endif

    fclose(intFile);

#if defined(DEBUG)
    GPU_TIMER_STOP();
    PRINTUSINGTIME("IO FLUSHING", time);
    time_io += time;
    fprintf(gpu->debugFile, " TOTAL INT = %i \n", *intNum);
#endif

    PRINTDEBUG("END TO RUN AOINT KERNEL");

#if defined(DEBUG)
    GPU_TIMER_DESTROY();
#endif
}
#endif


//-----------------------------------------------
// calculate the size of shared memory for XC code
//-----------------------------------------------
//void upload_xc_smem() {
//    // First, determine the sizes of prmitive function arrays that will go into smem. This is helpful
//    // to copy data from gmem to smem.
//    gpu->gpu_xcq->primfpbin          = new gpu_buffer_type<int>(gpu->gpu_xcq->nbins);
//
//    // Count how many primitive functions per each bin, also keep track of maximum number of basis and
//    // primitive functions
//    int maxbfpbin=0;
//    int maxpfpbin=0;
//    for(int i=0; i<gpu->gpu_xcq->nbins; i++){
//        int nbasf = gpu->gpu_xcq->basf_locator->_hostData[i+1] - gpu->gpu_xcq->basf_locator->_hostData[i];
//        maxbfpbin = maxbfpbin < nbasf ? nbasf : maxbfpbin;
//
//        int tot_primfpb=0;
//
//        for(int j=gpu->gpu_xcq->basf_locator->_hostData[i]; j<gpu->gpu_xcq->basf_locator->_hostData[i+1] ; j++){
//            for(int k=gpu->gpu_xcq->primf_locator->_hostData[j]; k< gpu->gpu_xcq->primf_locator->_hostData[j+1]; k++){
//                tot_primfpb++;
//            }
//        }
//        gpu->gpu_xcq->primfpbin->_hostData[i] = tot_primfpb;
//        maxpfpbin = maxpfpbin < tot_primfpb ? tot_primfpb : maxpfpbin;
//    }
//
//    // In order to avoid memory misalignements, round upto the nearest multiple of 8.
//    maxbfpbin = maxbfpbin + 8 - maxbfpbin % 8;
//    maxpfpbin = maxpfpbin + 8 - maxpfpbin % 8;
//
//    // We will store basis and primitive function indices and primitive function locations of each bin in shared memory.
//    gpu->gpu_xcq->smem_size = sizeof(char)*maxpfpbin + sizeof(short)*maxbfpbin + sizeof(int)*(maxbfpbin+8);
//
//    gpu->gpu_sim.maxbfpbin = maxbfpbin;
//    gpu->gpu_sim.maxpfpbin = maxpfpbin;
//
//    gpu->gpu_xcq->primfpbin->Upload();
//    gpu->gpu_sim.primfpbin     = gpu->gpu_xcq->primfpbin->_devData;
//
//    printf("Max number of basis functions: %i primitive functions: %i smem size: %i \n", maxbfpbin, maxpfpbin, gpu->gpu_xcq->smem_size);
//
//    for(int i=0; i<gpu->gpu_xcq->nbins;i++)
//        if(gpu->gpu_xcq->primfpbin->_hostData[i] >= maxpfpbin)
//            printf("bin_id= %i nprimf= %i \n", i, gpu->gpu_xcq->primfpbin->_hostData[i]);
//}


//-----------------------------------------------
//  delete libxc information
//-----------------------------------------------
extern "C" void gpu_delete_libxc_(int *ierr)
{
    libxc_cleanup(gpu->gpu_sim.glinfo, gpu->gpu_sim.nauxfunc);
}


//-------------------------------------------------
//  delete information uploaded for LRI calculation
//-------------------------------------------------
extern "C" void gpu_delete_lri_(int *ierr)
{
    SAFE_DELETE(gpu->lri_data->cc);
}


//-------------------------------------------------
//  delete info uploaded for CEW quad calculation
//-------------------------------------------------
extern "C" void gpu_delete_cew_vrecip_(int *ierr)
{
    SAFE_DELETE(gpu->lri_data->vrecip);
}
