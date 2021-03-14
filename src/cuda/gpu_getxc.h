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

//-----------------------------------------------
// Calculate the density and gradients of density at
// each grid point. Huge memory (hmem) version will 
// use precomputed basis function values and gradients
// while the other will compute them. 
//-----------------------------------------------
#ifdef HMEM
__global__ void get_density_hmem_kernel()
#else
__global__ void get_density_kernel()
#endif
{
  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  int totalThreads = blockDim.x*gridDim.x;

  for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {

    int bin_id    = devSim_dft.bin_locator[gid];
    int bfloc_st  = devSim_dft.basf_locator[bin_id];
    int bfloc_end = devSim_dft.basf_locator[bin_id+1];

#ifdef HMEM
      int phii = devSim_dft.phi_loc[gid];
#else
      QUICKDouble gridx = devSim_dft.gridx[gid];
      QUICKDouble gridy = devSim_dft.gridy[gid];
      QUICKDouble gridz = devSim_dft.gridz[gid];
#endif

      QUICKDouble density = 0.0;
      QUICKDouble gax = 0.0;
      QUICKDouble gay = 0.0;
      QUICKDouble gaz = 0.0;

      for(int i=bfloc_st; i < bfloc_end; i++){

        int ibas = (int) devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;

#ifdef HMEM
        phi    = devSim_dft.phi[phii];
        dphidx = devSim_dft.dphidx[phii];
        dphidy = devSim_dft.dphidy[phii];
        dphidz = devSim_dft.dphidz[phii];
#else
        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
#endif

        if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) {

          QUICKDouble denseii = LOC2(devSim_dft.dense, ibas, ibas, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
          density = density + denseii * phi / 2.0;
          gax = gax + denseii * dphidx;
          gay = gay + denseii * dphidy;
          gaz = gaz + denseii * dphidz;

#ifdef HMEM
          int phij = phii+1;
#endif
          for(int j=i+1; j< bfloc_end; j++){

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

#ifdef HMEM
            phi2    = devSim_dft.phi[phij];
            dphidx2 = devSim_dft.dphidx[phij];
            dphidy2 = devSim_dft.dphidy[phij];
            dphidz2 = devSim_dft.dphidz[phij];
#else
            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);
#endif

            QUICKDouble denseij = LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
            density = density + denseij * phi * phi2;
            gax = gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
            gay = gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
            gaz = gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );
#ifdef HMEM
            ++phij;
#endif
          }
        }
#ifdef HMEM
        ++phii;
#endif
      }

      devSim_dft.densa[gid] = density;
      devSim_dft.densb[gid] = density;
      devSim_dft.gax[gid] = gax;
      devSim_dft.gbx[gid] = gax;
      devSim_dft.gay[gid] = gay;
      devSim_dft.gby[gid] = gay;
      devSim_dft.gaz[gid] = gaz;
      devSim_dft.gbz[gid] = gaz;
  }
}


//-----------------------------------------------
// Calculate the density and gradients of density at
// each grid point. Huge memory (hmem) version will 
// use precomputed basis function values and gradients
// while the other will compute them. 
//-----------------------------------------------
#ifdef HMEM
__global__ void getxc_hmem_kernel(gpu_libxc_info** glinfo, int nof_functionals)
#else
__global__ void getxc_kernel(gpu_libxc_info** glinfo, int nof_functionals)
#endif
{
  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  int totalThreads = blockDim.x*gridDim.x;

  for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {

    int bin_id    = devSim_dft.bin_locator[gid];
    int bfloc_st  = devSim_dft.basf_locator[bin_id];
    int bfloc_end = devSim_dft.basf_locator[bin_id+1];

#ifdef HMEM
    int phi_st = devSim_dft.phi_loc[gid];
#else
    QUICKDouble gridx = devSim_dft.gridx[gid];
    QUICKDouble gridy = devSim_dft.gridy[gid];
    QUICKDouble gridz = devSim_dft.gridz[gid];
#endif
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
      QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
      QUICKDouble _tmp ;

      if (devSim_dft.method == B3LYP) {
        _tmp = b3lyp_e(2.0*density, sigma) * weight;
      }else if(devSim_dft.method == DFT){
         _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
         + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)) * weight;
      }

      QUICKDouble dfdr;
      QUICKDouble dot, xdot, ydot, zdot;

      if (devSim_dft.method == B3LYP) {
         dot = b3lypf(2.0*density, sigma, &dfdr);
         xdot = dot * gax;
         ydot = dot * gay;
         zdot = dot * gaz;
      }else if(devSim_dft.method == DFT){
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
         //Prepare in/out for libxc call
         double d_rhoa = (double) density;
         double d_rhob = (double) densityb;
         double d_sigma = (double)sigma;
         double d_zk, d_vrho, d_vsigma;
         d_zk = d_vrho = d_vsigma = 0.0;

         for(int i=0; i<nof_functionals; i++){
           double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
           tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

           gpu_libxc_info* tmp_glinfo = glinfo[i];

           switch(tmp_glinfo->gpu_worker){
             case GPU_WORK_LDA:
                     gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                     break;

             case GPU_WORK_GGA_X:
                     gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                     break;

             case GPU_WORK_GGA_C:
                     gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                     break;
           }
           d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
           d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
           d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
         }

         _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)) * weight);

         QUICKDouble dfdgaa;
         //QUICKDouble dfdgab, dfdgaa2, dfdgab2;
         //QUICKDouble dfdr2;
         dfdr = (QUICKDouble)d_vrho;
         dfdgaa = (QUICKDouble)d_vsigma*4.0;

         xdot = dfdgaa * gax;
         ydot = dfdgaa * gay;
         zdot = dfdgaa * gaz;
      }
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

#ifdef HMEM
      int phii = phi_st;
#endif
      for (int i = bfloc_st; i< bfloc_end; ++i) {

        int ibas = devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;

#ifdef HMEM
        int phij = phi_st;
        phi    = devSim_dft.phi[phii];
        dphidx = devSim_dft.dphidx[phii];
        dphidy = devSim_dft.dphidy[phii];
        dphidz = devSim_dft.dphidz[phii];
#else
        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
#endif
        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {
          for (int j = bfloc_st; j < bfloc_end; j++) {

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;           

#ifdef HMEM
            phi2    = devSim_dft.phi[phij];
            dphidx2 = devSim_dft.dphidx[phij];
            dphidy2 = devSim_dft.dphidy[phij];
            dphidz2 = devSim_dft.dphidz[phij];
#else
            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);
#endif
            QUICKDouble _tmp = (phi * phi2 * dfdr + xdot * (phi*dphidx2 + phi2*dphidx) \
            + ydot * (phi*dphidy2 + phi2*dphidy) + zdot * (phi*dphidz2 + phi2*dphidz))*weight;

            QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
            if ( _tmp * weight < (QUICKDouble)0.0)
                   val1 = 0ull - val1;
            QUICKADD(LOC2(devSim_dft.oULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), val1);
#ifdef HMEM
            ++phij;
#endif
          }
        }
#ifdef HMEM 
        ++phii;
#endif
      }
    }
  }
    
}

//-----------------------------------------------
// Calculate the density and gradients of density at
// each grid point. Huge memory (hmem) version will 
// use precomputed basis function values and gradients
// while the other will compute them. 
//-----------------------------------------------
#ifdef HMEM
__global__ void get_xcgrad_hmem_kernel(gpu_libxc_info** glinfo, int nof_functionals)
#else
__global__ void get_xcgrad_kernel(gpu_libxc_info** glinfo, int nof_functionals)
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

#ifdef HMEM
    int phi_st = devSim_dft.phi_loc[gid];
#endif

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
      QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
      QUICKDouble _tmp ;

      if (devSim_dft.method == B3LYP) {
        _tmp = b3lyp_e(2.0*density, sigma);
      }else if(devSim_dft.method == DFT){
         _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
              + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
      }

      QUICKDouble dfdr;
      QUICKDouble dot, xdot, ydot, zdot;

      if (devSim_dft.method == B3LYP) {
        dot = b3lypf(2.0*density, sigma, &dfdr);
        xdot = dot * gax;
        ydot = dot * gay;
        zdot = dot * gaz;
      }else if(devSim_dft.method == DFT){
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
        //Prepare in/out for libxc call
        double d_rhoa = (double) density;
        double d_rhob = (double) densityb;
        double d_sigma = (double)sigma;
        double d_zk, d_vrho, d_vsigma;
        d_zk = d_vrho = d_vsigma = 0.0;

        for(int i=0; i<nof_functionals; i++){
          double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
          tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

          gpu_libxc_info* tmp_glinfo = glinfo[i];

          switch(tmp_glinfo->gpu_worker){
            case GPU_WORK_LDA:
                    gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                    break;

            case GPU_WORK_GGA_X:
                    gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                    break;

            case GPU_WORK_GGA_C:
                    gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                    break;
          }
          d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
          d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
          d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
        }

        _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));

        QUICKDouble dfdgaa;
        dfdr = (QUICKDouble)d_vrho;
        dfdgaa = (QUICKDouble)d_vsigma*4.0;

        xdot = dfdgaa * gax;
        ydot = dfdgaa * gay;
        zdot = dfdgaa * gaz;
      }
      devSim_dft.exc[gid] = _tmp;

#ifdef HMEM
      int phii = phi_st;
#endif

      for (int i = bfloc_st; i< bfloc_end; i++) {
        int ibas = devSim_dft.basf[i];
        QUICKDouble phi, dphidx, dphidy, dphidz;

#ifdef HMEM
        int phij = phi_st;
        phi    = devSim_dft.phi[phii];
        dphidx = devSim_dft.dphidx[phii];
        dphidy = devSim_dft.dphidy[phii];
        dphidz = devSim_dft.dphidz[phii];
#else
        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
#endif

        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

          QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

          pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

          int Istart = (devSim_dft.ncenter[ibas]-1) * 3;

          for (int j = bfloc_st; j < bfloc_end; j++) {

            int jbas = devSim_dft.basf[j];
            QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

#ifdef HMEM
            phi2    = devSim_dft.phi[phij];
            dphidx2 = devSim_dft.dphidx[phij];
            dphidy2 = devSim_dft.dphidy[phij];
            dphidz2 = devSim_dft.dphidz[phij];
#else
            pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);
#endif

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

            GRADADD(smemGrad[Istart], Gradx);
            GRADADD(smemGrad[Istart+1], Grady);
            GRADADD(smemGrad[Istart+2], Gradz);
#ifdef HMEM
            ++phij;
#endif
          }
        }
#ifdef HMEM
        ++phii;
#endif
      }
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
