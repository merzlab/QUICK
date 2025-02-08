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

#if !defined(OSHELL)
__global__ void getcew_quad_kernel()
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {
        int bin_id = devSim_dft.bin_locator[gid];
        int bfloc_st = devSim_dft.basf_locator[bin_id];
        int bfloc_end = devSim_dft.basf_locator[bin_id+1];

        QUICKDouble gridx = devSim_dft.gridx[gid];
        QUICKDouble gridy = devSim_dft.gridy[gid];
        QUICKDouble gridz = devSim_dft.gridz[gid];

        QUICKDouble weight = devSim_dft.weight[gid];

        QUICKDouble dfdr = devSim_dft.cew_vrecip[gid];

        for (int i = bfloc_st; i < bfloc_end; ++i) {
            int ibas = devSim_dft.basf[i];
            QUICKDouble phi, dphidx, dphidy, dphidz;

            pteval_new(gridx, gridy, gridz,
                    &phi, &dphidx, &dphidy, &dphidz,
                    devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

            if (abs(phi + dphidx + dphidy + dphidz) > devSim_dft.DMCutoff) {
                for (int j = bfloc_st; j < bfloc_end; j++) {
                    int jbas = devSim_dft.basf[j];
                    QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                    pteval_new(gridx, gridy, gridz,
                            &phi2, &dphidx2, &dphidy2, &dphidz2,
                            devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

                    QUICKDouble _tmp = phi * phi2 * dfdr * weight;

#if defined(USE_LEGACY_ATOMICS)
                    GPUATOMICADD(&LOC2(devSim_dft.oULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), _tmp, OSCALE);
#else
                    atomicAdd(&LOC2(devSim_dft.o, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), _tmp);
#endif
                }
            }
        }
    }
}
#endif


#if defined(OSHELL)
__global__ void oshell_getcew_quad_grad_kernel()
#else
__global__ void cshell_getcew_quad_grad_kernel()
#endif
{
#if defined(USE_LEGACY_ATOMICS)
    //declare smem grad vector
    extern __shared__ QUICKULL smem_buffer[];
    QUICKULL* smemGrad = (QUICKULL *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
        smemGrad[i] = 0ull;
    }
#else
    //declare smem grad vector
    extern __shared__ QUICKDouble smem_buffer[];
    QUICKDouble* smemGrad = (QUICKDouble *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
        smemGrad[i] = 0.0;
    }
#endif

    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < devSim_dft.npoints; gid += totalThreads) {
        int bin_id = devSim_dft.bin_locator[gid];
        int bfloc_st = devSim_dft.basf_locator[bin_id];
        int bfloc_end = devSim_dft.basf_locator[bin_id+1];

        QUICKDouble gridx = devSim_dft.gridx[gid];
        QUICKDouble gridy = devSim_dft.gridy[gid];
        QUICKDouble gridz = devSim_dft.gridz[gid];
        QUICKDouble weight = devSim_dft.weight[gid];
#if defined(OSHELL)
        QUICKDouble densitysum = devSim_dft.densa[gid] + devSim_dft.densb[gid];
#else
        QUICKDouble densitysum = 2 * devSim_dft.densa[gid];
#endif

        QUICKDouble dfdr = devSim_dft.cew_vrecip[gid];

        if (densitysum > devSim_dft.DMCutoff) {
            QUICKDouble _tmp = (QUICKDouble) (dfdr * densitysum);

            devSim_dft.exc[gid] = _tmp;

            QUICKDouble sumGradx = 0.0;
            QUICKDouble sumGrady = 0.0;
            QUICKDouble sumGradz = 0.0;

            for (int i = bfloc_st; i < bfloc_end; i++) {
                int ibas = devSim_dft.basf[i];
                QUICKDouble phi, dphidx, dphidy, dphidz;
                pteval_new(gridx, gridy, gridz,
                        &phi, &dphidx, &dphidy, &dphidz,
                        devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

                if (abs(phi + dphidx + dphidy + dphidz) > devSim_dft.DMCutoff) {
                    //QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;
                    //pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

                    int Istart = (devSim_dft.ncenter[ibas] - 1) * 3;

                    for (int j = bfloc_st; j < bfloc_end; j++) {
                        int jbas = devSim_dft.basf[j];
                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                        pteval_new(gridx, gridy, gridz,
                                &phi2, &dphidx2, &dphidy2, &dphidz2,
                                devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

                        QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

#if defined(OSHELL)
                        denseij += (QUICKDouble) LOC2(devSim_dft.denseb, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
#endif

                        QUICKDouble Gradx = -2.0 * denseij * weight * (dfdr * dphidx * phi2);
                        QUICKDouble Grady = -2.0 * denseij * weight * (dfdr * dphidy * phi2);
                        QUICKDouble Gradz = -2.0 * denseij * weight * (dfdr * dphidz * phi2);
                        //printf("test quad grad %f %f %f %f %f %f %f %f %f %f\n", gridx, gridy, gridz, denseij, weight, dfdr, dphidx, dphidy, dphidz, phi2);

                        GPUATOMICADD(&smemGrad[Istart], Gradx, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 1], Grady, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 2], Gradz, GRADSCALE);
                        sumGradx += Gradx;
                        sumGrady += Grady;
                        sumGradz += Gradz;
                    }
                }
            }

            int Istart = (devSim_dft.gatm[gid] - 1) * 3;

            GPUATOMICADD(&smemGrad[Istart], -sumGradx, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 1], -sumGrady, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 2], -sumGradz, GRADSCALE);
        }

        //Set weights for sswder calculation
        if (densitysum < devSim_dft.DMCutoff) {
            devSim_dft.dweight_ssd[gid] = 0;
        }

        if (devSim_dft.sswt[gid] == 1) {
            devSim_dft.dweight_ssd[gid] = 0;
        }
    }

    __syncthreads();

    // update gmem grad vector
    for (int i = threadIdx.x; i < devSim_dft.natom * 3; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&devSim_dft.gradULL[i], smemGrad[i]);
#else
        atomicAdd(&devSim_dft.grad[i], smemGrad[i]);
#endif
    }

    __syncthreads();
}
