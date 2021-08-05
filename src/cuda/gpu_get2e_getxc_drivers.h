/*
  !---------------------------------------------------------------------!
  ! Created by Madu Manathunga on 04/07/2021                            !
  !                                                                     !
  ! Previous contributors: Yipu Miao                                    !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains preprocessable get2e and getxc C functions!
  ! that can be called from f90 subroutines.                            !
  !---------------------------------------------------------------------!
*/

//-----------------------------------------------
//  core part, compute 2-e integrals
//-----------------------------------------------
#ifdef OSHELL
extern "C" void gpu_get_oshell_eri_(QUICKDouble* o, QUICKDouble* ob)
#else
extern "C" void gpu_get_cshell_eri_(QUICKDouble* o)
#endif
{
    PRINTDEBUG("BEGIN TO RUN GET ERI")

    upload_sim_to_constant(gpu);

    PRINTDEBUG("BEGIN TO RUN KERNEL")

#ifdef OSHELL
    get_oshell_eri(gpu);
#else
    get2e(gpu);
#endif

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

#ifdef OSHELL
    gpu -> gpu_calculated -> obULL -> Download();
    cudaMemsetAsync(gpu -> gpu_calculated -> obULL -> _devData, 0, sizeof(QUICKULL)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            QUICKULL valULL = LOC2(gpu->gpu_calculated->obULL->_hostData, j, i, gpu->nbasis, gpu->nbasis);
            QUICKDouble valDB;

            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }
            LOC2(gpu->gpu_calculated->ob->_hostData,i,j,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->ob->_hostData,j,i,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
        }
    }
#endif

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

    gpu -> gpu_calculated -> o    -> DownloadSum(o);

#ifdef OSHELL
    gpu -> gpu_calculated -> ob   -> DownloadSum(ob);
#endif

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
#ifdef OSHELL
      delete gpu->gpu_calculated->ob;
      delete gpu->gpu_calculated->denseb;
      delete gpu->gpu_calculated->obULL;
#endif

    }

    delete gpu->gpu_cutoff->cutMatrix;

    PRINTDEBUG("COMPLETE RUNNING GET2E")
}

#ifdef OSHELL
extern "C" void gpu_get_oshell_eri_grad_(QUICKDouble* grad)
#else
extern "C" void gpu_get_cshell_eri_grad_(QUICKDouble* grad)
#endif
{
    PRINTDEBUG("BEGIN TO RUN GRAD")

    upload_sim_to_constant(gpu);

    PRINTDEBUG("BEGIN TO RUN KERNEL")

    if(gpu -> gpu_sim.is_oshell == true){
        get_oshell_eri_grad(gpu);
    }else{
        getGrad(gpu);
    }

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

#ifdef OSHELL
      delete gpu->gpu_calculated->denseb;
#endif

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



#ifdef OSHELL
extern "C" void gpu_get_oshell_xc_(QUICKDouble* Eelxc, QUICKDouble* aelec, QUICKDouble* belec, QUICKDouble *o, QUICKDouble *ob)
#else
extern "C" void gpu_get_cshell_xc_(QUICKDouble* Eelxc, QUICKDouble* aelec, QUICKDouble* belec, QUICKDouble *o)
#endif
{
    PRINTDEBUG("BEGIN TO RUN GETXC")

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

    getxc(gpu);

    gpu -> DFT_calculated -> Download();

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
    gpu -> gpu_calculated -> o    -> DownloadSum(o);

#ifdef OSHELL
    gpu -> gpu_calculated -> obULL -> Download();
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            QUICKULL valULL = LOC2(gpu->gpu_calculated->obULL->_hostData, j, i, gpu->nbasis, gpu->nbasis);
            QUICKDouble valDB;

            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }
            LOC2(gpu->gpu_calculated->ob->_hostData,i,j,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
            LOC2(gpu->gpu_calculated->ob->_hostData,j,i,gpu->nbasis, gpu->nbasis) = (QUICKDouble)valDB*ONEOVEROSCALE;
        }
    }
    gpu -> gpu_calculated -> ob    -> DownloadSum(ob);
#endif

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

#ifdef OSHELL
    delete gpu->gpu_calculated->ob;
    delete gpu->gpu_calculated->denseb;
    delete gpu->gpu_calculated->obULL;
#endif

}

#ifdef OSHELL
extern "C" void gpu_get_oshell_xcgrad_(QUICKDouble *grad)
#else
extern "C" void gpu_get_cshell_xcgrad_(QUICKDouble *grad)
#endif
{
        // calculate smem size
        gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

        upload_sim_to_constant_dft(gpu);

        getxc_grad(gpu);

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

#ifdef OSHELL
        delete gpu->gpu_calculated->denseb;
#endif
}


#ifndef OSHELL
extern "C" void gpu_get_oei_(QUICKDouble* o)
{

    gpu -> gpu_calculated -> o        =   new cuda_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

#ifdef LEGACY_ATOMIC_ADD
    gpu -> gpu_calculated -> o        ->  DeleteGPU();
    gpu -> gpu_calculated -> oULL     =   new cuda_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
    gpu -> gpu_calculated -> oULL     -> Upload();
    gpu -> gpu_sim.oULL              =  gpu -> gpu_calculated -> oULL -> _devData;
#else
    gpu -> gpu_calculated -> o     -> Upload();
    gpu -> gpu_sim.o = gpu -> gpu_calculated -> o -> _devData;
#endif
        
    upload_sim_to_constant_oei(gpu);
 
    upload_para_to_const_oei();

    getOEI(gpu);

#ifdef LEGACY_ATOMIC_ADD
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
#else
    gpu -> gpu_calculated -> o -> Download();

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis);
        }
    }

#endif


    /*for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {            
            printf("OEI host O: %d %d %f %f \n", i, j, LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis), o[idxf90++]);
        }
    }*/

    gpu -> gpu_calculated -> o    -> DownloadSum(o);

    SAFE_DELETE(gpu -> gpu_calculated -> o);

#ifdef LEGACY_ATOMIC_ADD
    SAFE_DELETE(gpu -> gpu_calculated -> oULL);
#endif

}

extern "C" void gpu_get_oei_grad_(QUICKDouble* grad, QUICKDouble* ptchg_grad)
{

    // upload point charge grad vector
    if(gpu -> nextatom > 0) {
        gpu -> ptchg_grad = new cuda_buffer_type<QUICKDouble>(3 * gpu -> nextatom);

#ifdef LEGACY_ATOMIC_ADD
        gpu -> ptchg_gradULL = new cuda_buffer_type<QUICKULL>(3 * gpu -> nextatom);       
        gpu -> ptchg_gradULL -> Upload();
        gpu -> gpu_sim.ptchg_gradULL =  gpu -> ptchg_gradULL -> _devData;
        gpu -> ptchg_grad -> DeleteGPU();
#else
        gpu -> ptchg_grad -> Upload();
        gpu -> gpu_sim.ptchg_grad =  gpu -> ptchg_grad -> _devData;
#endif
    }

    upload_sim_to_constant_oei(gpu);

    get_oei_grad(gpu);

    // download gradients
#ifdef LEGACY_ATOMIC_ADD
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
#else

    gpu->grad->Download();

#endif

/*  for(int i=0; i<3*gpu->natom; ++i){
    printf("grad: %d %f \n", i, gpu->grad->_hostData[i]);

  }
*/
    // download point charge gradients
    if(gpu -> nextatom > 0) {

#ifdef LEGACY_ATOMIC_ADD
        gpu -> ptchg_gradULL -> Download();

        for (int i = 0; i< 3 * gpu->nextatom; i++) {
            QUICKULL valULL = gpu->ptchg_gradULL->_hostData[i];
            QUICKDouble valDB;

            if (valULL >= 0x8000000000000000ull) {
                valDB  = -(QUICKDouble)(valULL ^ 0xffffffffffffffffull);
            }
            else
            {
                valDB  = (QUICKDouble) valULL;
            }

            gpu->ptchg_grad->_hostData[i] = (QUICKDouble)valDB*ONEOVERGRADSCALE;
      }
#else

      gpu->ptchg_grad->Download();

#endif

/*      for(int i=0; i<3*gpu->nextatom; ++i){
          printf("ptchg_grad: %d %f \n", i, gpu->ptchg_grad->_hostData[i]);
      }
*/
      gpu->ptchg_grad->DownloadSum(ptchg_grad);
  
    }

    gpu->grad->DownloadSum(grad);

  // ptchg_grad is no longer needed. reclaim the memory.
  if(gpu -> nextatom > 0) {
#ifdef LEGACY_ATOMIC_ADD
      SAFE_DELETE(gpu -> ptchg_gradULL);
#endif
      SAFE_DELETE(gpu -> ptchg_grad);
  }
}
#endif
