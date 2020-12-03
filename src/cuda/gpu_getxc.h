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
// each grid point.
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
#endif

      QUICKDouble density = 0.0;
      QUICKDouble gax = 0.0;
      QUICKDouble gay = 0.0;
      QUICKDouble gaz = 0.0;

      QUICKDouble gridx = devSim_dft.gridx[gid];
      QUICKDouble gridy = devSim_dft.gridy[gid];
      QUICKDouble gridz = devSim_dft.gridz[gid];

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
            ++phij;
          }
        }
        ++phii;
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
