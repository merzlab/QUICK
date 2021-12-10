#include "hip/hip_runtime.h"
/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 12/03/2020                            !
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

#ifdef OSHELL
#define NSPIN 2
#else
#define NSPIN 1
#endif

//-----------------------------------------------
// Calculate the density and gradients of density at
// each grid point.
//-----------------------------------------------
#ifdef OSHELL
__global__ void get_oshell_density_kernel()
#else
__global__ void get_cshell_density_kernel()
#endif
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

      QUICKDouble density = 0.0;
      QUICKDouble gax = 0.0;
      QUICKDouble gay = 0.0;
      QUICKDouble gaz = 0.0;

#ifdef OSHELL
      QUICKDouble densityb = 0.0;
      QUICKDouble gbx = 0.0;
      QUICKDouble gby = 0.0;
      QUICKDouble gbz = 0.0;
#endif

      for(int i=bfloc_st; i < bfloc_end; i++){

        int ibas = (int) devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;

        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

        if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) {

          QUICKDouble denseii = LOC2(devSim_dft.dense, ibas, ibas, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
#ifdef OSHELL
          QUICKDouble densebii = LOC2(devSim_dft.denseb, ibas, ibas, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
#endif

#ifdef OSHELL
          density  = density  + denseii * phi;
          densityb = densityb + densebii * phi;
#else
          density = density + denseii * phi / 2.0;
#endif
          gax = gax + denseii * dphidx;
          gay = gay + denseii * dphidy;
          gaz = gaz + denseii * dphidz;

#ifdef OSHELL
          gbx = gbx + densebii * dphidx;
          gby = gby + densebii * dphidy;
          gbz = gbz + densebii * dphidz;
#endif

          for(int j=i+1; j< bfloc_end; j++){

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

            QUICKDouble denseij = LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
#ifdef OSHELL
            QUICKDouble densebij = LOC2(devSim_dft.denseb, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
#endif

#ifdef OSHELL
            density  = density  + 2.0 * denseij * phi * phi2;
            densityb = densityb + 2.0 * densebij * phi * phi2;
#else
            density = density + denseij * phi * phi2;
#endif
            gax = gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
            gay = gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
            gaz = gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );
#ifdef OSHELL
            gbx = gbx + densebij * ( phi * dphidx2 + phi2 * dphidx );
            gby = gby + densebij * ( phi * dphidy2 + phi2 * dphidy );
            gbz = gbz + densebij * ( phi * dphidz2 + phi2 * dphidz );
#endif
          }
        }
      }
#ifdef OSHELL
      devSim_dft.densa[gid] = density;
      devSim_dft.densb[gid] = densityb;
      devSim_dft.gax[gid] = 2.0 * gax;
      devSim_dft.gbx[gid] = 2.0 * gbx;
      devSim_dft.gay[gid] = 2.0 * gay;
      devSim_dft.gby[gid] = 2.0 * gby;
      devSim_dft.gaz[gid] = 2.0 * gaz;
      devSim_dft.gbz[gid] = 2.0 * gbz;
#else
      devSim_dft.densa[gid] = density;
      devSim_dft.densb[gid] = density;
      devSim_dft.gax[gid] = gax;
      devSim_dft.gbx[gid] = gax;
      devSim_dft.gay[gid] = gay;
      devSim_dft.gby[gid] = gay;
      devSim_dft.gaz[gid] = gaz;
      devSim_dft.gbz[gid] = gaz;
#endif
  }
}

#ifdef OSHELL
__global__ void oshell_getxc_kernel()
#else
__global__ void cshell_getxc_kernel()
#endif
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
    QUICKDouble density = devSim_dft.densa[gid];
    QUICKDouble densityb = devSim_dft.densb[gid];
    QUICKDouble gax = devSim_dft.gax[gid];
    QUICKDouble gay = devSim_dft.gay[gid];
    QUICKDouble gaz = devSim_dft.gaz[gid];
    QUICKDouble gbx = devSim_dft.gbx[gid];
    QUICKDouble gby = devSim_dft.gby[gid];
    QUICKDouble gbz = devSim_dft.gbz[gid];

    if(density >devSim_dft.DMCutoff){

      QUICKDouble dfdr;
      QUICKDouble xdot, ydot, zdot;
      QUICKDouble _tmp ;


#ifdef OSHELL
      QUICKDouble dfdrb;
      QUICKDouble xdotb, ydotb, zdotb;

      QUICKDouble gaa = (gax * gax + gay * gay + gaz * gaz);
      QUICKDouble gab = (gax * gbx + gay * gby + gaz * gbz);
      QUICKDouble gbb = (gbx * gbx + gby * gby + gbz * gbz);
#else

      QUICKDouble dot;
      QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);

      if (devSim_dft.method == B3LYP) {
        _tmp = b3lyp_e(2.0*density, sigma) * weight;
      }else if(devSim_dft.method == BLYP){
         _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
         + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)) * weight;
      }


      if (devSim_dft.method == B3LYP) {
         dot = b3lypf(2.0*density, sigma, &dfdr);
         xdot = dot * gax;
         ydot = dot * gay;
         zdot = dot * gaz;
      }else if(devSim_dft.method == BLYP){
         QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
         QUICKDouble dfdr2;

         becke(density, gax, gay, gaz, gbx, gby, gbz, &dfdr, &dfdgaa, &dfdgab);
         lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
         dfdr += dfdr2;
         dfdgaa += dfdgaa2;
         dfdgab += dfdgab2;
         //Calculate the first term in the dot product shown above,i.e.:
         //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
         xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
         ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
         zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;
     }else if(devSim_dft.method == LIBXC){
#endif
         //Prepare in/out for libxc call
         double d_rhoa = (double) density;
         double d_rhob = (double) densityb;

         // array d_sigma stores gaa, gab and gbb respectively
         QUICKDouble d_sigma[3]  = {0.0, 0.0, 0.0};
         // array d_vrho stores dfdra and dfdrb respectively
         QUICKDouble d_vrho[2]   = {0.0, 0.0};
         // array d_vsigma carries dfdgaa, dfdgab and dfdgbb respectively
         QUICKDouble d_vsigma[3] = {0.0, 0.0, 0.0};
         QUICKDouble d_zk = 0.0;

#ifdef OSHELL
            d_sigma[0] = gaa;
            d_sigma[1] = gab;
            d_sigma[2] = gbb;
#else
            d_sigma[0] = sigma;
#endif

         int nof_functionals = devSim_dft.nauxfunc;
         gpu_libxc_info** glinfo = devSim_dft.glinfo;

         for(int i=0; i<nof_functionals; i++){
           QUICKDouble tmp_d_zk = 0.0;
           QUICKDouble tmp_d_vrho[2]   = {0.0, 0.0};
           QUICKDouble tmp_d_vsigma[3] = {0.0, 0.0, 0.0};

           gpu_libxc_info* tmp_glinfo = glinfo[i];

           switch(tmp_glinfo->gpu_worker){
             case GPU_WORK_LDA:
                     gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, NSPIN);
                     break;

             case GPU_WORK_GGA_X:
                     
                     gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble*)&d_sigma, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, (QUICKDouble*)&tmp_d_vsigma, NSPIN);
                     break;

             case GPU_WORK_GGA_C:
                     gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble*)&d_sigma, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, (QUICKDouble*)&tmp_d_vsigma, NSPIN);
                     break;
           }
           d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
           d_vrho[0] += (tmp_d_vrho[0]*tmp_glinfo->mix_coeff);
           d_vsigma[0] += (tmp_d_vsigma[0]*tmp_glinfo->mix_coeff);
#ifdef OSHELL
           d_vrho[1]   += (tmp_d_vrho[1] * tmp_glinfo->mix_coeff);
           d_vsigma[1] += (tmp_d_vsigma[1] * tmp_glinfo->mix_coeff);
           d_vsigma[2] += (tmp_d_vsigma[2] * tmp_glinfo->mix_coeff);
#endif

         }

         _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)) * weight);
         dfdr = (QUICKDouble) d_vrho[0];
#ifdef OSHELL
         dfdrb= (QUICKDouble) d_vrho[1];

         xdot  =  2.0 * d_vsigma[0] * gax + d_vsigma[1] * gbx;
         ydot  =  2.0 * d_vsigma[0] * gay + d_vsigma[1] * gby;
         zdot  =  2.0 * d_vsigma[0] * gaz + d_vsigma[1] * gbz;

         xdotb =  2.0 * d_vsigma[2] * gbx + d_vsigma[1] * gax;
         ydotb =  2.0 * d_vsigma[2] * gby + d_vsigma[1] * gay;
         zdotb =  2.0 * d_vsigma[2] * gbz + d_vsigma[1] * gaz;
#else
         xdot = 4.0 * d_vsigma[0] * gax;
         ydot = 4.0 * d_vsigma[0] * gay;
         zdot = 4.0 * d_vsigma[0] * gaz;
#endif

#ifndef OSHELL
      }
#endif
      QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
      if ( _tmp * weight < (QUICKDouble)0.0)
          val1 = 0ull - val1;
      QUICKADD(devSim_dft.DFT_calculated[0].Eelxc, val1);

      _tmp = weight*density;
      val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
      if ( _tmp * weight < (QUICKDouble)0.0)
          val1 = 0ull - val1;
      QUICKADD(devSim_dft.DFT_calculated[0].aelec, val1);


      _tmp = weight*densityb;
      val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
      if ( _tmp * weight < (QUICKDouble)0.0)
          val1 = 0ull - val1;
      QUICKADD(devSim_dft.DFT_calculated[0].belec, val1);

      for (int i = bfloc_st; i< bfloc_end; ++i) {

        int ibas = devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;

        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {
          for (int j = bfloc_st; j < bfloc_end; j++) {

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;           

            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

            QUICKDouble _tmp = (phi * phi2 * dfdr + xdot * (phi*dphidx2 + phi2*dphidx) \
            + ydot * (phi*dphidy2 + phi2*dphidy) + zdot * (phi*dphidz2 + phi2*dphidz))*weight;

            QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
            if ( _tmp * weight < (QUICKDouble)0.0) val1 = 0ull - val1;
            QUICKADD(LOC2(devSim_dft.oULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), val1);

#ifdef OSHELL
            QUICKDouble _tmpb = (phi * phi2 * dfdrb + xdotb * (phi*dphidx2 + phi2*dphidx)
              + ydotb * (phi*dphidy2 + phi2*dphidy) + zdotb * (phi*dphidz2 + phi2*dphidz))*weight;

            QUICKULL val2 = (QUICKULL) (fabs( _tmpb * OSCALE) + (QUICKDouble)0.5);
              if ( _tmpb * weight < (QUICKDouble)0.0) val2 = 0ull - val2;
            QUICKADD(LOC2(devSim_dft.obULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), val2);
#endif
          }
        }
      }
    }
  }
    
}

#ifdef OSHELL
__global__ void oshell_getxcgrad_kernel()
#else
__global__ void cshell_getxcgrad_kernel()
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
    QUICKDouble density = devSim_dft.densa[gid];
    QUICKDouble densityb = devSim_dft.densb[gid];
    QUICKDouble gax = devSim_dft.gax[gid];
    QUICKDouble gay = devSim_dft.gay[gid];
    QUICKDouble gaz = devSim_dft.gaz[gid];
    QUICKDouble gbx = devSim_dft.gbx[gid];
    QUICKDouble gby = devSim_dft.gby[gid];
    QUICKDouble gbz = devSim_dft.gbz[gid];

#ifdef CEW
    QUICKDouble dfdr_cew = 0.0;
    if(devSim_dft.use_cew) dfdr_cew = devSim_dft.cew_vrecip[gid];
#endif

    if(density >devSim_dft.DMCutoff){

      QUICKDouble dfdr;
      QUICKDouble xdot, ydot, zdot;
      QUICKDouble _tmp ;

#ifdef OSHELL
      QUICKDouble dfdrb;
      QUICKDouble xdotb, ydotb, zdotb;

      QUICKDouble gaa = (gax * gax + gay * gay + gaz * gaz);
      QUICKDouble gab = (gax * gbx + gay * gby + gaz * gbz);
      QUICKDouble gbb = (gbx * gbx + gby * gby + gbz * gbz);
#else
      QUICKDouble dot;
      QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);

      if (devSim_dft.method == B3LYP) {
        _tmp = b3lyp_e(2.0*density, sigma);
      }else if(devSim_dft.method == BLYP){
         _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
              + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
      }


      if (devSim_dft.method == B3LYP) {
        dot = b3lypf(2.0*density, sigma, &dfdr);
        xdot = dot * gax;
        ydot = dot * gay;
        zdot = dot * gaz;
      }else if(devSim_dft.method == BLYP){
        QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
        QUICKDouble dfdr2;
        
        becke(density, gax, gay, gaz, gbx, gby, gbz, &dfdr, &dfdgaa, &dfdgab);
        lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
        dfdr   += dfdr2;
        dfdgaa += dfdgaa2;
        dfdgab += dfdgab2;

        //Calculate the first term in the dot product shown above,i.e.:
        //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
        xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
        ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
        zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;

      }else if(devSim_dft.method == LIBXC){
#endif
        //Prepare in/out for libxc call
        QUICKDouble d_rhoa = (QUICKDouble) density;
        QUICKDouble d_rhob = (QUICKDouble) densityb;
        // array d_sigma stores gaa, gab and gbb respectively
        QUICKDouble d_sigma[3]  = {0.0, 0.0, 0.0};
        // array d_vrho stores dfdra and dfdrb respectively
        QUICKDouble d_vrho[2]   = {0.0, 0.0};
        // array d_vsigma carries dfdgaa, dfdgab and dfdgbb respectively
        QUICKDouble d_vsigma[3] = {0.0, 0.0, 0.0};
        QUICKDouble d_zk = 0.0;

#ifdef OSHELL
        d_sigma[0] = gaa;
        d_sigma[1] = gab;
        d_sigma[2] = gbb;
#else
        d_sigma[0] = sigma;
#endif

        int nof_functionals = devSim_dft.nauxfunc;
        gpu_libxc_info** glinfo = devSim_dft.glinfo;

        for(int i=0; i<nof_functionals; i++){
          QUICKDouble tmp_d_zk = 0.0;
          QUICKDouble tmp_d_vrho[2]   = {0.0, 0.0};
          QUICKDouble tmp_d_vsigma[3] = {0.0, 0.0, 0.0};

          gpu_libxc_info* tmp_glinfo = glinfo[i];

          switch(tmp_glinfo->gpu_worker){
            case GPU_WORK_LDA:
                    gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, NSPIN);
                    break;

            case GPU_WORK_GGA_X:
                    gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble*)&d_sigma, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, (QUICKDouble*)&tmp_d_vsigma, NSPIN);
                    break;

            case GPU_WORK_GGA_C:
                    gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble*)&d_sigma, &tmp_d_zk, (QUICKDouble*)&tmp_d_vrho, (QUICKDouble*)&tmp_d_vsigma, NSPIN);
                    break;
          }
          d_zk        += (tmp_d_zk * tmp_glinfo->mix_coeff);
          d_vrho[0]   += (tmp_d_vrho[0] * tmp_glinfo->mix_coeff);
          d_vsigma[0] += (tmp_d_vsigma[0] * tmp_glinfo->mix_coeff);
#ifdef OSHELL
          d_vrho[1]   += (tmp_d_vrho[1] * tmp_glinfo->mix_coeff);
          d_vsigma[1] += (tmp_d_vsigma[1] * tmp_glinfo->mix_coeff);
          d_vsigma[2] += (tmp_d_vsigma[2] * tmp_glinfo->mix_coeff);
#endif

        }

        _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));
        dfdr = (QUICKDouble) d_vrho[0];

#ifdef OSHELL
        dfdrb= (QUICKDouble) d_vrho[1];

        xdot  =  2.0 * d_vsigma[0] * gax + d_vsigma[1] * gbx;
        ydot  =  2.0 * d_vsigma[0] * gay + d_vsigma[1] * gby;
        zdot  =  2.0 * d_vsigma[0] * gaz + d_vsigma[1] * gbz;

        xdotb =  2.0 * d_vsigma[2] * gbx + d_vsigma[1] * gax;
        ydotb =  2.0 * d_vsigma[2] * gby + d_vsigma[1] * gay;
        zdotb =  2.0 * d_vsigma[2] * gbz + d_vsigma[1] * gaz;
#else
        xdot = 4.0 * d_vsigma[0] * gax;
        ydot = 4.0 * d_vsigma[0] * gay;
        zdot = 4.0 * d_vsigma[0] * gaz;
#endif

#ifndef OSHELL
      }
#endif

#ifdef CEW
      devSim_dft.exc[gid] = _tmp + (dfdr_cew * (density+densityb));
#else
      devSim_dft.exc[gid] = _tmp;
#endif

      QUICKDouble sumGradx=0.0, sumGrady=0.0, sumGradz=0.0;

      for (int i = bfloc_st; i< bfloc_end; i++) {
        int ibas = devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;
        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

          QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

          pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

          int Istart = (devSim_dft.ncenter[ibas]-1) * 3;

          for (int j = bfloc_st; j < bfloc_end; j++) {

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

            QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

            QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2
                    + xdot * (dxdx * phi2 + dphidx * dphidx2)
                    + ydot * (dxdy * phi2 + dphidx * dphidy2)
                    + zdot * (dxdz * phi2 + dphidx * dphidz2));

            QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2
                    + xdot * (dxdy * phi2 + dphidy * dphidx2)
                    + ydot * (dydy * phi2 + dphidy * dphidy2)
                    + zdot * (dydz * phi2 + dphidy * dphidz2));

            QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2
                    + xdot * (dxdz * phi2 + dphidz * dphidx2)
                    + ydot * (dydz * phi2 + dphidz * dphidy2)
                    + zdot * (dzdz * phi2 + dphidz * dphidz2));
#ifdef OSHELL
            QUICKDouble densebij = (QUICKDouble) LOC2(devSim_dft.denseb, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

            Gradx += - 2.0 * densebij * weight * (dfdrb * dphidx * phi2
                    + xdotb * (dxdx * phi2 + dphidx * dphidx2)
                    + ydotb * (dxdy * phi2 + dphidx * dphidy2)
                    + zdotb * (dxdz * phi2 + dphidx * dphidz2));

            Grady += - 2.0 * densebij * weight * (dfdrb * dphidy * phi2
                    + xdotb * (dxdy * phi2 + dphidy * dphidx2)
                    + ydotb * (dydy * phi2 + dphidy * dphidy2)
                    + zdotb * (dydz * phi2 + dphidy * dphidz2));

            Gradz += - 2.0 * densebij * weight * (dfdrb * dphidz * phi2
                    + xdotb * (dxdz * phi2 + dphidz * dphidx2)
                    + ydotb * (dydz * phi2 + dphidz * dphidy2)
                    + zdotb * (dzdz * phi2 + dphidz * dphidz2));
#endif

#ifdef CEW

            if(devSim_dft.use_cew){
#ifdef OSHELL
              denseij += densebij;
#endif

              Gradx -= 2.0 * denseij * weight * dfdr_cew * dphidx * phi2;
              Grady -= 2.0 * denseij * weight * dfdr_cew * dphidy * phi2;
              Gradz -= 2.0 * denseij * weight * dfdr_cew * dphidz * phi2;

            }
#endif

            GRADADD(smemGrad[Istart], Gradx);
            GRADADD(smemGrad[Istart+1], Grady);
            GRADADD(smemGrad[Istart+2], Gradz);

            sumGradx += Gradx;
            sumGrady += Grady;
            sumGradz += Gradz;

          }
        }
      }

      int Istart = (devSim_dft.gatm[gid]-1) * 3;
      GRADADD(smemGrad[Istart], -sumGradx);
      GRADADD(smemGrad[Istart+1], -sumGrady);
      GRADADD(smemGrad[Istart+2], -sumGradz);

    }
    //Set weights for sswder calculation
    if(density < devSim_dft.DMCutoff){
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

#undef NSPIN
