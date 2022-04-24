/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 09/29/2021                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains preprocessable functions required for     ! 
  ! QUICK GPU version.                                                  !
  !---------------------------------------------------------------------!
*/

#ifdef CEW
#include "iface.hpp"

#ifndef OSHELL
void getcew_quad(_gpu_type gpu){

    QUICK_SAFE_CALL((getcew_quad_kernel<<< gpu -> blocks, gpu -> xc_threadsPerBlock>>>()));

    cudaDeviceSynchronize();
}


void getcew_quad_grad(_gpu_type gpu){

    if(gpu -> gpu_sim.is_oshell == true){

        QUICK_SAFE_CALL((get_oshell_density_kernel<<<gpu->blocks, gpu->xc_threadsPerBlock>>>()));

        cudaDeviceSynchronize();

        QUICK_SAFE_CALL((oshell_getcew_quad_grad_kernel<<< gpu -> blocks, gpu -> xc_threadsPerBlock, gpu -> gpu_xcq -> smem_size>>>()));

    }else{

        QUICK_SAFE_CALL((get_cshell_density_kernel<<<gpu->blocks, gpu->xc_threadsPerBlock>>>()));

        cudaDeviceSynchronize();

        QUICK_SAFE_CALL((cshell_getcew_quad_grad_kernel<<< gpu -> blocks, gpu -> xc_threadsPerBlock, gpu -> gpu_xcq -> smem_size>>>()));
        //QUICK_SAFE_CALL((cshell_getcew_quad_grad_kernel<<< 1,1, gpu -> gpu_xcq -> smem_size>>>()));

    }

    cudaDeviceSynchronize();

    get_cew_accdens(gpu);

    prune_grid_sswgrad();

    get_sswnumgrad_kernel<<< gpu->blocks, gpu->sswGradThreadsPerBlock, gpu -> gpu_xcq -> smem_size>>>();

    //get_sswnumgrad_kernel<<< 1,1, gpu -> gpu_xcq -> smem_size>>>();

    cudaDeviceSynchronize();

    gpu_delete_sswgrad_vars();

}


void get_cew_accdens(_gpu_type gpu){

    QUICKDouble *gridpt = new QUICKDouble[3]; 
    QUICKDouble *cewGrad= new QUICKDouble[3];   

    gpu -> gpu_xcq -> densa -> Download();

    if(gpu -> gpu_sim.is_oshell == true) gpu -> gpu_xcq -> densb -> Download();
    

    for(int i=0; i< gpu -> gpu_xcq -> npoints;i++){
        
        QUICKDouble weight = gpu -> gpu_xcq -> weight -> _hostData[i];
        QUICKDouble densea = gpu -> gpu_xcq -> densa -> _hostData[i];
        QUICKDouble denseb = densea;

        if(gpu -> gpu_sim.is_oshell == true) denseb = gpu -> gpu_xcq -> densb -> _hostData[i];   

        gridpt[0] = gpu -> gpu_xcq -> gridx -> _hostData[i];
        gridpt[1] = gpu -> gpu_xcq -> gridy -> _hostData[i];
        gridpt[2] = gpu -> gpu_xcq -> gridz -> _hostData[i];

        const QUICKDouble charge_density = -weight * (densea+denseb);

        for(int j=0; j<3; j++) cewGrad[j]=0.0;

        QUICKDouble const *cnst_gridpt = gridpt;

        // this function comes from cew library in amber       
        cew_accdensatpt_(cnst_gridpt, &charge_density, cewGrad);

//printf("cew_accdensatpt %f %f %f %f %f %f %f \n", gridpt[0], gridpt[1], gridpt[2], charge_density\
,cewGrad[0], cewGrad[1], cewGrad[2]);

        int Istart = (gpu -> gpu_xcq -> gatm -> _hostData[i]-1) * 3;

        for(int j=0; j<3; j++)
            gpu->grad->_hostData[Istart+j] += cewGrad[j];

    }

    delete gridpt;
    delete cewGrad;

}




__global__ void getcew_quad_kernel()
{
  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  int totalThreads = blockDim.x*gridDim.x;

  for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {

    int bin_id    = devSim_dft.bin_locator[gid];
    int bfloc_st  = devSim_dft.basf_locator[bin_id];
    int bfloc_end = devSim_dft.basf_locator[bin_id+1];

    QUICKDouble gridx = devSim_dft.gridx[gid];
    QUICKDouble gridy = devSim_dft.gridy[gid];
    QUICKDouble gridz = devSim_dft.gridz[gid];

    QUICKDouble weight = devSim_dft.weight[gid];

    QUICKDouble dfdr = devSim_dft.cew_vrecip[gid];

    for (int i = bfloc_st; i< bfloc_end; ++i) {

      int ibas = devSim_dft.basf[i];
      QUICKDouble phi, dphidx, dphidy, dphidz;

      pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
      if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {
        for (int j = bfloc_st; j < bfloc_end; j++) {

          int jbas = devSim_dft.basf[j];
          QUICKDouble phi2, dphidx2, dphidy2, dphidz2;           

          pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

          QUICKDouble _tmp = phi * phi2 * dfdr * weight;

          QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
          if ( _tmp * weight < (QUICKDouble)0.0) val1 = 0ull - val1;
          QUICKADD(LOC2(devSim_dft.oULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), val1);

        }
      }
    }
  }
    
}
#endif

#ifdef OSHELL
__global__ void oshell_getcew_quad_grad_kernel()
#else
__global__ void cshell_getcew_quad_grad_kernel()
#endif
{

  //declare smem grad vector
  extern __shared__ QUICKULL smem_buffer[];
  QUICKULL* smemGrad=(QUICKULL*)smem_buffer;

  // initialize smem grad
  for(int i = threadIdx.x; i< devSim_dft.natom * 3; i+=blockDim.x)
    smemGrad[i]=0ull;

  __syncthreads();

  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  int totalThreads = blockDim.x*gridDim.x;

  for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {

    int bin_id    = devSim_dft.bin_locator[gid];
    int bfloc_st  = devSim_dft.basf_locator[bin_id];
    int bfloc_end = devSim_dft.basf_locator[bin_id+1];


    QUICKDouble gridx = devSim_dft.gridx[gid];
    QUICKDouble gridy = devSim_dft.gridy[gid];
    QUICKDouble gridz = devSim_dft.gridz[gid];
    QUICKDouble weight = devSim_dft.weight[gid];
#ifdef OSHELL
    QUICKDouble densitysum = devSim_dft.densa[gid]+devSim_dft.densb[gid];
#else
    QUICKDouble densitysum = 2*devSim_dft.densa[gid];
#endif

    QUICKDouble dfdr = devSim_dft.cew_vrecip[gid];

    if(densitysum >devSim_dft.DMCutoff){

    QUICKDouble _tmp = ((QUICKDouble) (dfdr * densitysum));

    devSim_dft.exc[gid] = _tmp;

    QUICKDouble sumGradx = 0.0;
    QUICKDouble sumGrady = 0.0;
    QUICKDouble sumGradz = 0.0;

      for (int i = bfloc_st; i< bfloc_end; i++) {
        int ibas = devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;
        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

          //QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

          //pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

          int Istart = (devSim_dft.ncenter[ibas]-1) * 3;

          for (int j = bfloc_st; j < bfloc_end; j++) {

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

            QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

#ifdef OSHELL
            denseij += (QUICKDouble) LOC2(devSim_dft.denseb, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
#endif

            QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2);
            QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2);
            QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2);
//printf("test quad grad %f %f %f %f %f %f %f %f %f %f\n", gridx, gridy, gridz, denseij, weight, dfdr, dphidx, dphidy, dphidz, phi2);

            GRADADD(smemGrad[Istart], Gradx);
            GRADADD(smemGrad[Istart+1], Grady);
            GRADADD(smemGrad[Istart+2], Gradz);

            sumGradx += Gradx;
            sumGrady += Grady;
            sumGradz += Gradz;

          }
        }
      }

      int Istart = (devSim_dft.gatm[gid]-1)*3;      

      GRADADD(smemGrad[Istart], -sumGradx);
      GRADADD(smemGrad[Istart+1], -sumGrady);
      GRADADD(smemGrad[Istart+2], -sumGradz);

    }

    //Set weights for sswder calculation
    if(densitysum < devSim_dft.DMCutoff){
            devSim_dft.dweight_ssd[gid] = 0;
    }

    if(devSim_dft.sswt[gid] == 1){
            devSim_dft.dweight_ssd[gid] = 0;
    }
    
  }

  __syncthreads();

  // update gmem grad vector
  for(int i = threadIdx.x; i< devSim_dft.natom * 3; i+=blockDim.x)
    atomicAdd(&devSim_dft.gradULL[i],smemGrad[i]);

  __syncthreads();

}
#endif
