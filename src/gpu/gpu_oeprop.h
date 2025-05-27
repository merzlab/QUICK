/*
   !---------------------------------------------------------------------!
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!

   !---------------------------------------------------------------------!
   ! This source file contains functions required for QUICK one electron !
   ! integral computation.                                               !
   !---------------------------------------------------------------------!
*/

#if !defined(__QUICK_GPU_OEPROP_H_)
#define __QUICK_GPU_OEPROP_H_

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"


__device__ static inline void addint_oeprop(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ,
        unsigned int ipoint, QUICKDouble * const store2)
{
    // obtain the start and final basis function indices for given shells II and JJ for
    // contribution into correct location in Fock matrix.
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (int III = III1; III <= III2; III++) {
        // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
        int i = (int) LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);

        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            int j = (int) LOC3(devTrans, 
                    LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // multiply the integral value by normalization constants.
            QUICKDouble dense_sym_factor;
            if (III != JJJ) {
                dense_sym_factor = 2.0;
            } else {
                dense_sym_factor = 1.0;
            }
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            if (devSim.is_oshell) {
                DENSEJI = DENSEJI + (QUICKDouble) LOC2(devSim.denseb, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis);
            }
            QUICKDouble Y = dense_sym_factor * DENSEJI * devSim.cons[III - 1] * devSim.cons[JJJ - 1]
                * LOCSTORE(store2, i - 1, j - 1, STOREDIM, STOREDIM);

#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&devSim.esp_electronicULL[ipoint], Y, OSCALE);
#else
            atomicAdd(&devSim.esp_electronic[ipoint], Y);
#endif
        }
    }
}


__device__ static inline void iclass_oeprop(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ,
        unsigned int ipoint, unsigned int totalpoint, unsigned int totalatom,
        QUICKDouble * const YVerticalTemp, QUICKDouble * const store, QUICKDouble * const store2)
{
    /*
       kAtom A, B  is the coresponding atom for shell II, JJ
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.
       Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
       which means they are corrosponding coorinates for shell II, JJ and nuclei.
   */
    QUICKDouble Ax = LOC2(devSim.allxyz, 0, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Ay = LOC2(devSim.allxyz, 1, devSim.katom[II] - 1, 3, totalatom);
    QUICKDouble Az = LOC2(devSim.allxyz, 2, devSim.katom[II] - 1, 3, totalatom);

    QUICKDouble Bx = LOC2(devSim.allxyz, 0, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble By = LOC2(devSim.allxyz, 1, devSim.katom[JJ] - 1, 3, totalatom);
    QUICKDouble Bz = LOC2(devSim.allxyz, 2, devSim.katom[JJ] - 1, 3, totalatom);

    /*
       kPrimI and kPrimJ indicates the number of primitives in shell II and JJ.
       kStartI, J indicates the starting guassian function for shell II, JJ.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
   */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];

    int kStartI = devSim.kstart[II] - 1;
    int kStartJ = devSim.kstart[JJ] - 1;

    /*
       Store array holds contracted integral values computed using VRR algorithm.
       See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
    */
    // initialize store2 array
    for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = 0; i < kPrimI * kPrimJ ; ++i) {
        int JJJ = (int) i / kPrimI;
        int III = (int) i - kPrimI * JJJ;

        /*
           In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
           for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
           we use I to express the corresponding index.
           Zeta = expo(I)+expo(J)
           --->                --->
           ->     expo(I) * xyz (I) + expo(J) * xyz(J)
           P  = ---------------------------------------
           expo(I) + expo(J)
           Those two are pre-calculated in CPU stage.

        */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];

        QUICKDouble Zeta = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ,
                devSim.prim_total, devSim.prim_total);

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        QUICKDouble Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        if (abs(Xcoeff_oei) > devSim.coreIntegralCutoff) {
            QUICKDouble Cx = LOC2(devSim.extpointxyz, 0, ipoint, 3, totalpoint);
            QUICKDouble Cy = LOC2(devSim.extpointxyz, 1, ipoint, 3, totalpoint);
            QUICKDouble Cz = LOC2(devSim.extpointxyz, 2, ipoint, 3, totalpoint);

            FmT(I + J, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)), YVerticalTemp);

            // compute all auxilary integrals and store
            for (int n = 0; n <= I + J; n++) {
                VY(0, 0, n) *= -1.0 * Xcoeff_oei;
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme.
            OEint_vertical(I, J,
#if defined(DEBUG_OEI)
                    II, JJ,
#endif
                    Px - Ax, Py - Ay, Pz - Az,
                    Px - Bx, Py - By, Pz - Bz,
                    Px - Cx, Py - Cy, Pz - Cz,
                    1.0 / (2.0 * Zeta), store, YVerticalTemp);

            // sum up primitive integral contributions
            for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(store2, j, i, STOREDIM, STOREDIM) +=  LOCSTORE(store, j, i, STOREDIM, STOREDIM);
                    }
                }
            }
        }
    }

    // retrive computed integral values from store array and update the Fock matrix
    addint_oeprop(I, J, II, JJ, ipoint, store2);
}


__global__ void getOEPROP_kernel()
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int jshell = devSim.Qshell;
    unsigned int jshell2 = jshell * jshell;
    unsigned int totalatom = devSim.natom + devSim.nextatom;
    unsigned int totalpoint = devSim.nextpoint;

    for (unsigned int ipoint = offset; ipoint < totalpoint; ipoint += totalThreads) {
      for (unsigned int idx = 0; idx < jshell2; idx++){
        // use the global index to obtain shell pair. Note that here we obtain
        // a couple of indices that helps us to obtain
        // shell number (ii and jj) and quantum numbers (iii, jjj).
        // Each point is assigned to a thread.

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
            int II = devSim.sorted_OEICutoffIJ[idx].x;
            int JJ = devSim.sorted_OEICutoffIJ[idx].y;

            // get the shell numbers of selected shell pair
            int ii = devSim.sorted_Q[II];
            int jj = devSim.sorted_Q[JJ];

            // Only choose the unique shell pairs
            if (jj >= ii) {
                // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];

                // compute coulomb attraction for the selected shell pair.
                iclass_oeprop(iii, jjj, ii, jj, ipoint, totalpoint, totalatom, devSim.YVerticalTemp + offset,
                        devSim.store + offset, devSim.store2 + offset);
            }
#if defined(MPIV_GPU)
        }
#endif
      }
    }
}


#endif
