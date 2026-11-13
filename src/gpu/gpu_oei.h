/*
   !---------------------------------------------------------------------!
   ! Written by Madu Manathunga on 06/17/2021                            !
   !                                                                     !
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

#if !defined(__QUICK_GPU_OEI_H_)
#define __QUICK_GPU_OEI_H_

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"

/*
 On-the-fly replacement for the expoSum/weightedCenterX/Y/Z (Zeta/P) lookup
 tables. Only used when devSim.expoSum == NULL (quick_method%gpuOnTheFly
 DnC jobs); mirrors the host-side precompute loop in gpu.cu (and the
 identically-named helper in gpu_get2e_subs.h) exactly, using arrays that
 are always uploaded (gcexpo, xyz, katom, Ksumtype). Duplicated here
 (rather than shared) because this file is compiled into a separate
 translation unit (gpu_oei.cu) from gpu_get2e_subs.h's (gpu_get2e.cu).
 */
__device__ static inline void onthefly_AB_P(const int ISH, const int JSH, const int primI, const int primJ,
        QUICKDouble &AB, QUICKDouble &Px, QUICKDouble &Py, QUICKDouble &Pz)
{
    int kAtomI = devSim.katom[ISH];
    int kAtomJ = devSim.katom[JSH];
    int KsumtypeI = devSim.Ksumtype[ISH];
    int KsumtypeJ = devSim.Ksumtype[JSH];

    QUICKDouble expoI = LOC2(devSim.gcexpo, primI, KsumtypeI - 1, MAXPRIM, devSim.nbasis);
    QUICKDouble expoJ = LOC2(devSim.gcexpo, primJ, KsumtypeJ - 1, MAXPRIM, devSim.nbasis);

    AB = expoI + expoJ;
    Px = (LOC2(devSim.xyz, 0, kAtomI - 1, 3, devSim.natom) * expoI
            + LOC2(devSim.xyz, 0, kAtomJ - 1, 3, devSim.natom) * expoJ) / AB;
    Py = (LOC2(devSim.xyz, 1, kAtomI - 1, 3, devSim.natom) * expoI
            + LOC2(devSim.xyz, 1, kAtomJ - 1, 3, devSim.natom) * expoJ) / AB;
    Pz = (LOC2(devSim.xyz, 2, kAtomI - 1, 3, devSim.natom) * expoI
            + LOC2(devSim.xyz, 2, kAtomJ - 1, 3, devSim.natom) * expoJ) / AB;
}

/*
 On-the-fly replacement for the Xcoeff_oei lookup table. Only used when
 devSim.Xcoeff_oei == NULL (quick_method%gpuOnTheFly DnC jobs); mirrors
 the host-side precompute loop in gpu.cu's gpu_upload_oei_ exactly.
 angI/angJ are the absolute angular-momentum class indices (I/J, not
 offset by Qstart).
 */
__device__ static inline QUICKDouble onthefly_Xcoeff_oei(const int ISH, const int JSH, const int primI, const int primJ,
        const int angI, const int angJ)
{
    int kAtomI = devSim.katom[ISH];
    int kAtomJ = devSim.katom[JSH];
    int KsumtypeI = devSim.Ksumtype[ISH];
    int KsumtypeJ = devSim.Ksumtype[JSH];

    QUICKDouble expoI = LOC2(devSim.gcexpo, primI, KsumtypeI - 1, MAXPRIM, devSim.nbasis);
    QUICKDouble expoJ = LOC2(devSim.gcexpo, primJ, KsumtypeJ - 1, MAXPRIM, devSim.nbasis);

    QUICKDouble DIJ = 0.0;
    for (int k = 0; k < 3; k++) {
        QUICKDouble d = LOC2(devSim.xyz, k, kAtomI - 1, 3, devSim.natom)
                - LOC2(devSim.xyz, k, kAtomJ - 1, 3, devSim.natom);
        DIJ += d * d;
    }

    QUICKDouble X = 2.0 * PI_TO_3HALF * sqrt((expoI + expoJ) / PI)
            * pow((expoI + expoJ), -1.5) * exp((-expoI * expoJ * DIJ) / (expoI + expoJ));

    return X * LOC2(devSim.gccoeff, primI, KsumtypeI + angI - 1, MAXPRIM, devSim.nbasis)
            * LOC2(devSim.gccoeff, primJ, KsumtypeJ + angJ - 1, MAXPRIM, devSim.nbasis);
}

__device__ static inline void iclass_oei(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ,
        unsigned int iatom, unsigned int totalatom, QUICKDouble * const YVerticalTemp,
        QUICKDouble * const store, QUICKDouble * const store2) {
    /*
       kAtom A, B  is the coresponding atom for shell II, JJ
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.
       Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
       which means they are corrosponding coorinates for shell II, JJ and nuclei.
    */
    const QUICKDouble Ax = LOC2(devSim.allxyz, 0, devSim.katom[II] - 1, 3, totalatom);
    const QUICKDouble Ay = LOC2(devSim.allxyz, 1, devSim.katom[II] - 1, 3, totalatom);
    const QUICKDouble Az = LOC2(devSim.allxyz, 2, devSim.katom[II] - 1, 3, totalatom);
    const QUICKDouble Bx = LOC2(devSim.allxyz, 0, devSim.katom[JJ] - 1, 3, totalatom);
    const QUICKDouble By = LOC2(devSim.allxyz, 1, devSim.katom[JJ] - 1, 3, totalatom);
    const QUICKDouble Bz = LOC2(devSim.allxyz, 2, devSim.katom[JJ] - 1, 3, totalatom);

    /*
       kPrimI and kPrimJ indicates the number of primitives in shell II and JJ.
       kStartI, J indicates the starting guassian function for shell II, JJ.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
    */
    const int kPrimI = devSim.kprim[II];
    const int kPrimJ = devSim.kprim[JJ];
    const int kStartI = devSim.kstart[II] - 1;
    const int kStartJ = devSim.kstart[JJ] - 1;

    /*
       Store array holds contracted integral values computed using VRR algorithm.
       See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
    */
    for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (int j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    const int ii_start = devSim.prim_start[II];
    const int jj_start = devSim.prim_start[JJ];

    for (int i = 0; i < kPrimI * kPrimJ; ++i) {
        const int JJJ = (int) i / kPrimI;
        const int III = (int) i - kPrimI * JJJ;

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
        QUICKDouble Zeta, Px, Py, Pz;
        if (devSim.expoSum != NULL) {
            Zeta = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
            Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
            Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
            Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        } else {
            onthefly_AB_P(II, JJ, III, JJJ, Zeta, Px, Py, Pz);
        }

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        QUICKDouble Xcoeff_oei;
        if (devSim.Xcoeff_oei != NULL) {
            Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                    I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        } else {
            Xcoeff_oei = onthefly_Xcoeff_oei(II, JJ, III, JJJ, I, J);
        }

        if (abs(Xcoeff_oei) > devSim.coreIntegralCutoff) {
            const QUICKDouble Cx = LOC2(devSim.allxyz, 0, iatom, 3, totalatom);
            const QUICKDouble Cy = LOC2(devSim.allxyz, 1, iatom, 3, totalatom);
            const QUICKDouble Cz = LOC2(devSim.allxyz, 2, iatom, 3, totalatom);
            const QUICKDouble chg = -1.0 * devSim.allchg[iatom];

            // compute boys function values, the third term of OS A20
            FmT(I + J, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)), YVerticalTemp);

            // compute all auxilary integrals and store
            for (int n = 0; n <= I + J; n++) {
                VY(0, 0, n) *= Xcoeff_oei * chg;
                //printf("aux: %d %f \n", i, VY(0, 0, i));
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
                        LOCSTORE(store2, j, i, STOREDIM, STOREDIM) += LOCSTORE(store, j, i, STOREDIM, STOREDIM);
                    }
                }
            }
        }
    }

    // retrive computed integral values from store array and update the Fock matrix
    
    // obtain the start and final basis function indices for given shells II and JJ. They will help us to save the integral
    // contribution into correct location in Fock matrix.
    const int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    const int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    const int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    const int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (int III = III1; III <= III2; III++) {
        // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
        const int i = (int) LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III - 1, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III - 1, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);

        for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
            const int j = (int) LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, JJJ - 1, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, JJJ - 1, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // multiply the integral value by normalization constants.
            const QUICKDouble Y = devSim.cons[III - 1] * devSim.cons[JJJ - 1]
                * LOCSTORE(store2, i - 1, j - 1, STOREDIM, STOREDIM);

//            if (III == 10 && JJJ == 50) {
//                printf("OEI debug: III JJJ I J iatm i j c1 c2 store2 Y %d %d %d %d %d %d %d %f %f %f %f\n",
//                        III, JJJ, I, J, iatom, i - 1,
//                        j - 1, devSim.cons[III - 1], devSim.cons[JJJ - 1],
//                        LOCSTORE(store2, i - 1, j - 1, STOREDIM, STOREDIM), Y);
//                printf("OEI debug: dt1 dt2 dt3 dt4 dt5 dt6:  %d %d %d %d %d %d \n",
//                        LOC2(devSim.KLMN, 0, III - 1, 3,devSim.nbasis),
//                        LOC2(devSim.KLMN, 1, III - 1, 3,devSim.nbasis),\
//                        LOC2(devSim.KLMN, 2, III - 1, 3,devSim.nbasis),
//                        LOC2(devSim.KLMN, 0, JJJ - 1, 3,devSim.nbasis),
//                        LOC2(devSim.KLMN, 1, JJJ - 1, 3,devSim.nbasis),
//                        LOC2(devSim.KLMN, 2, JJJ - 1, 3,devSim.nbasis));
//            }

            // Now add the contribution into Fock matrix.
#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(devSim.oULL, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis), Y, OSCALE);
#else
            atomicAdd(&LOC2(devSim.o, JJJ - 1, III - 1, devSim.nbasis, devSim.nbasis), Y);
#endif

            //printf("addint_oei: %d %d %f %f %f \n", III, JJJ, devSim.cons[III-1], devSim.cons[JJJ-1], LOCSTORE(store2, i-1, j-1, STOREDIM, STOREDIM));
        }
    }
}


__global__ void getOEI_kernel() {
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    const unsigned int jshell = devSim.Qshell;
    const unsigned int totalatom = devSim.natom + devSim.nextatom;

    // The work is one item per (shell pair, atom), and all three counts are
    // 32 bit. Their product is not: trypsin in STO-3G has 4929 shells and 3263
    // atoms, so the true count is 7.9e10 against a 4.29e9 limit. Computed in 32
    // bit it wrapped 18 times and left a bound of 1.97e9, which is 2.5% of the
    // work -- the nuclear attraction was summed over the first 80 atoms and no
    // others. The operator was then kinetic energy plus a fragment of the
    // attraction, which is positive definite, so every orbital energy came out
    // positive and no SCF built on it could converge. Nothing failed loudly.
    //
    // The two electron kernels already widen their shell counts the same way
    // (see gpu_get2e_subs.h); this one did not. The index arithmetic below has
    // to be widened too, or i / (jshell * jshell) truncates once i passes the
    // range of int, which it now does.
    const QUICKULL npair = (QUICKULL) jshell * (QUICKULL) jshell;

    for (QUICKULL i = offset; i < npair * (QUICKULL) totalatom; i += totalThreads) {
        // use the global index to obtain shell pair. Note that here we obtain a couple of indices that helps us to obtain
        // shell number (ii and jj) and quantum numbers (iii, jjj).
        const unsigned int iatom = (unsigned int) (i / npair);
        const unsigned int idx = (unsigned int) (i - (QUICKULL) iatom * npair);

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
            const int II = devSim.sorted_OEICutoffIJ[idx].x;
            const int JJ = devSim.sorted_OEICutoffIJ[idx].y;

            // get the shell numbers of selected shell pair
            const int ii = devSim.sorted_Q[II];
            const int jj = devSim.sorted_Q[JJ];

            // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
            const int iii = devSim.sorted_Qnumber[II];
            const int jjj = devSim.sorted_Qnumber[JJ];

            //printf(" tid: %d II JJ ii jj iii jjj %d  %d  %d  %d  %d  %d \n", (int) i, II, JJ, ii, jj, iii, jjj);

            // compute coulomb attraction for the selected shell pair.
            iclass_oei(iii, jjj, ii, jj, iatom, totalatom, devSim.YVerticalTemp + offset,
                    devSim.store + offset, devSim.store2 + offset);
#if defined(MPIV_GPU)
        }
#endif
    }
}


#endif
