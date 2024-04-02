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
extern "C" void gpu_get_oshell_eri_(bool *deltaO, QUICKDouble* o, QUICKDouble* ob)
#else
extern "C" void gpu_get_cshell_eri_(bool *deltaO, QUICKDouble* o)
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

#ifdef USE_LEGACY_ATOMICS
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

#else
    gpu -> gpu_calculated -> o -> Download();
    cudaMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData, j, i, gpu->nbasis, gpu->nbasis); 
        }
    }
#ifdef OSHELL
    gpu -> gpu_calculated -> ob -> Download();
    cudaMemsetAsync(gpu -> gpu_calculated -> ob -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->ob->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->ob->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }
#endif

#endif

    gpu -> gpu_calculated -> o    -> DownloadSum(o);

#ifdef OSHELL
    gpu -> gpu_calculated -> ob   -> DownloadSum(ob);
#endif

    PRINTDEBUG("DELETE TEMP VARIABLES")

    if(gpu -> gpu_sim.method == HF){
      delete gpu->gpu_calculated->o;
      delete gpu->gpu_calculated->dense;

#ifdef USE_LEGACY_ATOMICS
      delete gpu->gpu_calculated->oULL;
#ifdef OSHELL
      delete gpu->gpu_calculated->obULL;
#endif
#endif

#ifdef OSHELL
      delete gpu->gpu_calculated->ob;
      delete gpu->gpu_calculated->denseb;
#endif

    }else if(*deltaO != 0){
      delete gpu->gpu_calculated->dense;
#ifdef OSHELL
      delete gpu->gpu_calculated->denseb;
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

#ifdef CUDA_SPDF
    if (gpu->maxL >= 3) {
        upload_sim_to_constant_ffff(gpu);
       
        if(gpu -> gpu_sim.is_oshell == true){
            get_oshell_eri_grad_ffff(gpu);
        }else{
            getGrad_ffff(gpu);
        }
    }
#endif

    PRINTDEBUG("COMPLETE KERNEL")

    if(gpu -> gpu_sim.method == HF){

#ifdef USE_LEGACY_ATOMICS
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
      gpu -> grad -> Download();

#endif
    }

    if(gpu -> gpu_sim.method == HF){

      gpu -> grad -> DownloadSum(grad);

      delete gpu -> grad;
#ifdef USE_LEGACY_ATOMICS
      delete gpu -> gradULL;
#endif
      delete gpu->gpu_calculated->dense;

#ifdef OSHELL
      delete gpu->gpu_calculated->denseb;
#endif

    }


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

#ifdef USE_LEGACY_ATOMICS
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
#else
    gpu -> DFT_calculated -> _hostData[0].Eelxc = 0.0;
    gpu -> DFT_calculated -> _hostData[0].aelec = 0.0;
    gpu -> DFT_calculated -> _hostData[0].belec = 0.0;
#endif
    gpu -> DFT_calculated -> Upload();
    gpu -> gpu_sim.DFT_calculated= gpu -> DFT_calculated->_devData;

    upload_sim_to_constant_dft(gpu);
    PRINTDEBUG("BEGIN TO RUN KERNEL")

    getxc(gpu);

    gpu -> DFT_calculated -> Download();

#ifdef USE_LEGACY_ATOMICS
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
#else

    gpu -> gpu_calculated -> o -> Download();
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }

#ifdef OSHELL
    gpu -> gpu_calculated -> ob -> Download();
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->ob->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->ob->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }

#endif

    *Eelxc = gpu->DFT_calculated -> _hostData[0].Eelxc; 
    *aelec = gpu->DFT_calculated -> _hostData[0].aelec;
    *belec = gpu->DFT_calculated -> _hostData[0].belec;

#endif

    gpu -> gpu_calculated -> o    -> DownloadSum(o);
#ifdef OSHELL
    gpu -> gpu_calculated -> ob    -> DownloadSum(ob);
#endif

    PRINTDEBUG("DELETE TEMP VARIABLES")

    delete gpu->gpu_calculated->o;
    delete gpu->gpu_calculated->dense;

#ifdef USE_LEGACY_ATOMICS
    delete gpu->gpu_calculated->oULL;
#ifdef OSHELL
    delete gpu->gpu_calculated->obULL;
#endif
#endif

#ifdef OSHELL
    delete gpu->gpu_calculated->ob;
    delete gpu->gpu_calculated->denseb;
#endif

}

#ifdef OSHELL
extern "C" void gpu_get_oshell_xcgrad_(QUICKDouble *grad)
#else
extern "C" void gpu_get_cshell_xcgrad_(QUICKDouble *grad)
#endif
{

#if (defined CEW) && !(defined USE_LEGACY_ATOMICS)
    gpu -> cew_grad = new cuda_buffer_type<QUICKDouble>(3 * gpu -> nextatom);
#endif

        // calculate smem size
        gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

        upload_sim_to_constant_dft(gpu);

        memset(gpu->grad->_hostData, 0, gpu -> gpu_xcq -> smem_size);

        getxc_grad(gpu);

#ifdef USE_LEGACY_ATOMICS
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

          gpu->grad->_hostData[i] += (QUICKDouble)valDB*ONEOVERGRADSCALE;
        }
#else
        gpu -> grad -> Download();
#endif

        gpu -> grad -> DownloadSum(grad);

#if (defined CEW) && !(defined USE_LEGACY_ATOMICS)
        gpu -> cew_grad->DownloadSum(grad);
        delete gpu -> cew_grad;
#endif

        delete gpu -> grad;
#ifdef USE_LEGACY_ATOMICS
        delete gpu -> gradULL;
#endif
        delete gpu->gpu_calculated->dense;

#ifdef OSHELL
        delete gpu->gpu_calculated->denseb;
#endif
}



#ifndef OSHELL
extern "C" void gpu_get_oei_(QUICKDouble* o)
{

//    gpu -> gpu_calculated -> o        =   new cuda_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

//#ifdef LEGACY_ATOMIC_ADD
//    gpu -> gpu_calculated -> o        ->  DeleteGPU();
//    gpu -> gpu_calculated -> oULL     =   new cuda_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
//    gpu -> gpu_calculated -> oULL     -> Upload();
//    gpu -> gpu_sim.oULL              =  gpu -> gpu_calculated -> oULL -> _devData;
/*#else
    gpu -> gpu_calculated -> o     -> Upload();
    gpu -> gpu_sim.o = gpu -> gpu_calculated -> o -> _devData;
#endif
*/        
    upload_sim_to_constant_oei(gpu);
 
    upload_para_to_const_oei();

    getOEI(gpu);

#ifdef USE_LEGACY_ATOMICS
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
#else
    gpu -> gpu_calculated -> o -> Download();
    cudaMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis);
        }
    }

#endif

/*
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {            
            printf("OEI host O: %d %d %f %f \n", i, j, LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis), o[idxf90++]);
        }
    }
*/
    gpu -> gpu_calculated -> o    -> DownloadSum(o);

//    SAFE_DELETE(gpu -> gpu_calculated -> o);

//#ifdef LEGACY_ATOMIC_ADD
//    SAFE_DELETE(gpu -> gpu_calculated -> oULL);
//#endif

}

extern "C" void gpu_get_oei_grad_(QUICKDouble* grad, QUICKDouble* ptchg_grad)
{

    // upload point charge grad vector
    if(gpu -> nextatom > 0) {
        gpu -> ptchg_grad = new cuda_buffer_type<QUICKDouble>(3 * gpu -> nextatom);

#ifdef USE_LEGACY_ATOMICS
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
#ifdef USE_LEGACY_ATOMICS
      gpu -> gradULL -> Download();
      cudaMemsetAsync(gpu -> gradULL -> _devData, 0, sizeof(QUICKULL)*3*gpu->natom);
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
    cudaMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

#endif

    gpu->grad->DownloadSum(grad);

/*    for(int i=0; i<3*gpu->natom; ++i){
      printf("grad: %d %f %f \n", i, grad[i], gpu->grad->_hostData[i]);
  
    }
*/
    // download point charge gradients
    if(gpu -> nextatom > 0) {

#ifdef USE_LEGACY_ATOMICS
        gpu -> ptchg_gradULL -> Download();
   
        cudaMemsetAsync(gpu -> ptchg_gradULL -> _devData, 0, sizeof(QUICKULL)*3*gpu->nextatom);

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
      cudaMemsetAsync(gpu -> ptchg_grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->nextatom);

#endif

/*      for(int i=0; i<3*gpu->nextatom; ++i){
          printf("ptchg_grad: %d %f \n", i, gpu->ptchg_grad->_hostData[i]);
      }
*/
      gpu->ptchg_grad->DownloadSum(ptchg_grad);
  
    }

  // ptchg_grad is no longer needed. reclaim the memory.
  if(gpu -> nextatom > 0 && !gpu->gpu_sim.use_cew) {
#ifdef USE_LEGACY_ATOMICS
      SAFE_DELETE(gpu -> ptchg_gradULL);
#endif
      SAFE_DELETE(gpu -> ptchg_grad);
  }
}

#ifdef CEW

extern "C" void gpu_get_lri_(QUICKDouble* o)
{
  
//    gpu -> gpu_calculated -> o        =   new cuda_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

//#ifdef LEGACY_ATOMIC_ADD
//    gpu -> gpu_calculated -> o        ->  DeleteGPU();
//    gpu -> gpu_calculated -> oULL     =   new cuda_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
//    gpu -> gpu_calculated -> oULL     -> Upload();
//    gpu -> gpu_sim.oULL              =  gpu -> gpu_calculated -> oULL -> _devData;
/*#else
    gpu -> gpu_calculated -> o     -> Upload();
    gpu -> gpu_sim.o = gpu -> gpu_calculated -> o -> _devData;
#endif
*/

    upload_sim_to_constant_lri(gpu);

    upload_para_to_const_lri();

    get_lri(gpu);  

    //compute xc quad potential
    upload_sim_to_constant_dft(gpu);
    getcew_quad(gpu);

#ifdef USE_LEGACY_ATOMICS
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
#else
    gpu -> gpu_calculated -> o -> Download();
    cudaMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis);
        }
    }

#endif


/*    int idxf90=0;
    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {            
            printf("OEI host O: %d %d %f %f \n", i, j, LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis), o[idxf90++]);
        }
    }
*/
    gpu -> gpu_calculated -> o    -> DownloadSum(o);

//    SAFE_DELETE(gpu -> gpu_calculated -> o);

//#ifdef LEGACY_ATOMIC_ADD
//    SAFE_DELETE(gpu -> gpu_calculated -> oULL);
//#endif

}


extern "C" void gpu_get_lri_grad_(QUICKDouble* grad, QUICKDouble* ptchg_grad)
{

    upload_sim_to_constant_lri(gpu);

    upload_para_to_const_lri();

    get_lri_grad(gpu);

    // download gradients
#ifdef USE_LEGACY_ATOMICS
      gpu -> gradULL -> Download();
      cudaMemsetAsync(gpu -> gradULL -> _devData, 0, sizeof(QUICKULL)*3*gpu->natom);
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
    cudaMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

#endif

    gpu->grad->DownloadSum(grad);

/*    for(int i=0; i<3*gpu->natom; ++i){
      printf("grad: %d %f %f \n", i, grad[i], gpu->grad->_hostData[i]);
  
    }
*/
    // download point charge gradients
    if(gpu -> nextatom > 0) {

#ifdef USE_LEGACY_ATOMICS
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

  // ptchg_grad is no longer needed. reclaim the memory.
  if(gpu -> nextatom > 0) {
#ifdef USE_LEGACY_ATOMICS
      SAFE_DELETE(gpu -> ptchg_gradULL);
#endif
      SAFE_DELETE(gpu -> ptchg_grad);
  }  

}

extern "C" void gpu_getcew_grad_quad_(QUICKDouble* grad)
{

#ifndef USE_LEGACY_ATOMICS
    gpu -> cew_grad = new cuda_buffer_type<QUICKDouble>(3 * gpu -> nextatom);
#else
    memset(gpu -> grad -> _hostData, 0, sizeof(QUICKDouble)*3*gpu->natom);
#endif

    // calculate smem size
    gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

    //compute xc quad potential
    upload_sim_to_constant_dft(gpu);

    getcew_quad_grad(gpu);

    // download gradients
#ifdef USE_LEGACY_ATOMICS
      gpu -> gradULL -> Download();
      cudaMemsetAsync(gpu -> gradULL -> _devData, 0, sizeof(QUICKULL)*3*gpu->natom);
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

        // make sure to add rather than assign. we already computed one part of the cew
        // gradients on host asynchronously.
        gpu->grad->_hostData[i] += (QUICKDouble)valDB*ONEOVERGRADSCALE;
      }
#else
    gpu->grad->Download();
    cudaMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

#endif

    gpu->grad->DownloadSum(grad);

#ifndef USE_LEGACY_ATOMICS
    gpu -> cew_grad ->DownloadSum(grad);
    SAFE_DELETE(gpu -> cew_grad);
#endif

}
#endif
#endif
