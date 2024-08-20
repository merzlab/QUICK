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

    gpu -> gpu_calculated -> o -> Download();
    hipMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }
#ifdef OSHELL
    gpu -> gpu_calculated -> ob -> Download();
    hipMemsetAsync(gpu -> gpu_calculated -> ob -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->ob->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->ob->_hostData, j, i, gpu->nbasis, gpu->nbasis);
        }
    }
#endif

    gpu -> gpu_calculated -> o    -> DownloadSum(o);

#ifdef OSHELL
    gpu -> gpu_calculated -> ob   -> DownloadSum(ob);
#endif

    PRINTDEBUG("DELETE TEMP VARIABLES")

    if(gpu -> gpu_sim.method == HF){
        delete gpu->gpu_calculated->o;
        delete gpu->gpu_calculated->dense;

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

#ifdef GPU_SPDF
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
        gpu -> grad -> Download();
    }

    if(gpu -> gpu_sim.method == HF){
        gpu -> grad -> DownloadSum(grad);

        delete gpu -> grad;
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

    gpu->DFT_calculated = new gpu_buffer_type<DFT_calculated_type>(1, 1);
    gpu->DFT_calculated->_hostData[0].Eelxc = 0.0;
    gpu->DFT_calculated->_hostData[0].aelec = 0.0;
    gpu->DFT_calculated->_hostData[0].belec = 0.0;
    gpu->DFT_calculated->Upload();
    gpu->gpu_sim.DFT_calculated = gpu->DFT_calculated->_devData;

    upload_sim_to_constant_dft(gpu);
    PRINTDEBUG("BEGIN TO RUN KERNEL")

    getxc(gpu);

    gpu->DFT_calculated->Download();

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

    gpu -> gpu_calculated -> o    -> DownloadSum(o);
#ifdef OSHELL
    gpu -> gpu_calculated -> ob    -> DownloadSum(ob);
#endif

    PRINTDEBUG("DELETE TEMP VARIABLES")

    delete gpu->gpu_calculated->o;
    delete gpu->gpu_calculated->dense;

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
#if defined(CEW)
    gpu -> cew_grad = new gpu_buffer_type<QUICKDouble>(3 * gpu -> nextatom);
#endif

    // calculate smem size
    gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

    upload_sim_to_constant_dft(gpu);

    memset(gpu->grad->_hostData, 0, gpu -> gpu_xcq -> smem_size);

    getxc_grad(gpu);
    gpu -> grad -> Download();

    gpu -> grad -> DownloadSum(grad);

#if defined(CEW)
    gpu -> cew_grad->DownloadSum(grad);
    delete gpu -> cew_grad;
#endif

    delete gpu -> grad;
    delete gpu->gpu_calculated->dense;

#ifdef OSHELL
    delete gpu->gpu_calculated->denseb;
#endif
}


#ifndef OSHELL
extern "C" void gpu_get_oei_(QUICKDouble* o)
{
    //    gpu -> gpu_calculated -> o        =   new gpu_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

    //#ifdef LEGACY_ATOMIC_ADD
    //    gpu -> gpu_calculated -> o        ->  DeleteGPU();
    //    gpu -> gpu_calculated -> oULL     =   new gpu_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
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

    gpu -> gpu_calculated -> o -> Download();
    hipMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis);
        }
    }

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
        gpu -> ptchg_grad = new gpu_buffer_type<QUICKDouble>(3 * gpu -> nextatom);
        gpu -> ptchg_grad -> Upload();
        gpu -> gpu_sim.ptchg_grad =  gpu -> ptchg_grad -> _devData;
    }

    upload_sim_to_constant_oei(gpu);

    get_oei_grad(gpu);

    // download gradients
    gpu->grad->Download();
    hipMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

    gpu->grad->DownloadSum(grad);

    /*    for(int i=0; i<3*gpu->natom; ++i){
          printf("grad: %d %f %f \n", i, grad[i], gpu->grad->_hostData[i]);

          }
          */
    // download point charge gradients
    if(gpu -> nextatom > 0) {
        gpu->ptchg_grad->Download();
        hipMemsetAsync(gpu -> ptchg_grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->nextatom);

        /*      for(int i=0; i<3*gpu->nextatom; ++i){
                printf("ptchg_grad: %d %f \n", i, gpu->ptchg_grad->_hostData[i]);
                }
                */
        gpu->ptchg_grad->DownloadSum(ptchg_grad);
    }

    // ptchg_grad is no longer needed. reclaim the memory.
    if(gpu -> nextatom > 0 && !gpu->gpu_sim.use_cew) {
        SAFE_DELETE(gpu -> ptchg_grad);
    }
}


#if defined(CEW)
extern "C" void gpu_get_lri_(QUICKDouble* o)
{
    //    gpu -> gpu_calculated -> o        =   new gpu_buffer_type<QUICKDouble>(gpu->nbasis, gpu->nbasis);

    //#ifdef LEGACY_ATOMIC_ADD
    //    gpu -> gpu_calculated -> o        ->  DeleteGPU();
    //    gpu -> gpu_calculated -> oULL     =   new gpu_buffer_type<QUICKULL>(gpu->nbasis, gpu->nbasis);
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

    gpu -> gpu_calculated -> o -> Download();
    hipMemsetAsync(gpu -> gpu_calculated -> o -> _devData, 0, sizeof(QUICKDouble)*gpu->nbasis*gpu->nbasis);

    for (int i = 0; i< gpu->nbasis; i++) {
        for (int j = i; j< gpu->nbasis; j++) {
            LOC2(gpu->gpu_calculated->o->_hostData,i,j,gpu->nbasis, gpu->nbasis) = LOC2(gpu->gpu_calculated->o->_hostData,j,i,gpu->nbasis, gpu->nbasis);
        }
    }

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
    gpu->grad->Download();
    hipMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

    gpu->grad->DownloadSum(grad);

    /*    for(int i=0; i<3*gpu->natom; ++i){
          printf("grad: %d %f %f \n", i, grad[i], gpu->grad->_hostData[i]);

          }
          */
    // download point charge gradients
    if(gpu -> nextatom > 0) {
        gpu->ptchg_grad->Download();

        /*      for(int i=0; i<3*gpu->nextatom; ++i){
                printf("ptchg_grad: %d %f \n", i, gpu->ptchg_grad->_hostData[i]);
                }
                */
        gpu->ptchg_grad->DownloadSum(ptchg_grad);
    }

    // ptchg_grad is no longer needed. reclaim the memory.
    if(gpu -> nextatom > 0) {
        SAFE_DELETE(gpu -> ptchg_grad);
    }
}


extern "C" void gpu_getcew_grad_quad_(QUICKDouble* grad)
{
    gpu->cew_grad = new gpu_buffer_type<QUICKDouble>(3 * gpu -> nextatom);

    // calculate smem size
    gpu -> gpu_xcq -> smem_size = gpu->natom * 3 * sizeof(QUICKULL);

    //compute xc quad potential
    upload_sim_to_constant_dft(gpu);

    getcew_quad_grad(gpu);

    // download gradients
    gpu->grad->Download();
    hipMemsetAsync(gpu -> grad -> _devData, 0, sizeof(QUICKDouble)*3*gpu->natom);

    gpu->grad->DownloadSum(grad);

    gpu -> cew_grad ->DownloadSum(grad);
    SAFE_DELETE(gpu -> cew_grad);
}
#endif
#endif
