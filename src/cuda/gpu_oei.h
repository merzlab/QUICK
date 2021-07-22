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

#include "gpu_fmt.h"

__device__ void addint_oei(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ, unsigned int iatom, QUICKDouble* store){

    // obtain the start and final basis function indices for given shells II and JJ. They will help us to save the integral
    // contribution into correct location in Fock matrix. 
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {

            // devTrans maps a basis function with certain angular momentum to store array. Get the correct indices now.  
            int i = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis),\
                           LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

            int j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis),\
                           LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM);
             
            // multiply the integral value by normalization constants. 
            QUICKDouble Y = devSim.cons[III-1] * devSim.cons[JJJ-1] * LOC2(store, i-1, j-1, STOREDIM, STOREDIM);

            /*if( III == 10 && JJJ == 50 ) {

            printf("OEI debug: III JJJ I J iatm i j c1 c2 store Y %d %d %d %d %d %d %d %f %f %f %f\n", III, JJJ, I, J, iatom, i-1, j-1, devSim.cons[III-1], \
            devSim.cons[JJJ-1], LOC2(store, i-1, j-1, STOREDIM, STOREDIM), Y);
            printf("OEI debug: dt1 dt2 dt3 dt4 dt5 dt6:  %d %d %d %d %d %d \n", LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis),\
            LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis));
            }*/


#ifdef LEGACY_ATOMIC_ADD
            QUICKULL Yull = (QUICKULL) (fabs( Y * OSCALE) + (QUICKDouble)0.5);

            if (Y < (QUICKDouble)0.0) Yull = 0ull - Yull;

            // Now add the contribution into Fock matrix.
            QUICKADD(LOC2(devSim.oULL, JJJ-1, III-1, devSim.nbasis, devSim.nbasis), Yull);
#else
            QUICKADD(LOC2(devSim.o, JJJ-1, III-1, devSim.nbasis, devSim.nbasis), Y);
#endif
            //printf("addint_oei: %d %d %f %f %f \n", III, JJJ, devSim.cons[III-1], devSim.cons[JJJ-1], LOC2(store, i-1, j-1, STOREDIM, STOREDIM));

        }
    }

}


__device__ void iclass_oei(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ , unsigned int iatom, unsigned int totalatom){

    /*
     kAtom A, B  is the coresponding atom for shell II, JJ
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
     which means they are corrosponding coorinates for shell II, JJ and nuclei.
     */

    QUICKDouble Ax = LOC2(devSim.allxyz, 0 , devSim.katom[II]-1, 3, totalatom);
    QUICKDouble Ay = LOC2(devSim.allxyz, 1 , devSim.katom[II]-1, 3, totalatom);
    QUICKDouble Az = LOC2(devSim.allxyz, 2 , devSim.katom[II]-1, 3, totalatom);

    QUICKDouble Bx = LOC2(devSim.allxyz, 0 , devSim.katom[JJ]-1, 3, totalatom);
    QUICKDouble By = LOC2(devSim.allxyz, 1 , devSim.katom[JJ]-1, 3, totalatom);
    QUICKDouble Bz = LOC2(devSim.allxyz, 2 , devSim.katom[JJ]-1, 3, totalatom);

    /*
     kPrimI and kPrimJ indicates the number of primitives in shell II and JJ. 
     kStartI, J indicates the starting guassian function for shell II, JJ.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];

    //int kStartI = devSim.kstart[II]-1;
    //int kStartJ = devSim.kstart[JJ]-1;

    /* 
    sum of basis functions for shell II and JJ. For eg, for a p shell, KsumtypeI or KsumtypeJ
    would be 3.
    */ 
    int KsumtypeI = devSim.Ksumtype[II]-1;
    int KsumtypeJ = devSim.Ksumtype[JJ]-1;

    /*
     Store array holds contracted integral values computed using VRR algorithm. 
     See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
     */
    QUICKDouble store[STOREDIM*STOREDIM];

    // initialize store array

    for(int i=0; i<STOREDIM; ++i){
        for(int j=0; j<STOREDIM; ++j){
            LOC2(store, j, i, STOREDIM, STOREDIM) = 0.0;
        }
    }

    for(int i=0; i < kPrimI * kPrimJ ; ++i){

        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;

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

        QUICKDouble Zeta = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);

        // calculation below may be reduced by precomputing 

        QUICKDouble ssoverlap = PI_TO_3HALF * pow(Zeta, -1.5) * exp( - LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis) *\
        LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis) * (pow(Ax-Bx, 2) + pow(Ay-By, 2) + pow(Az-Bz, 2)) / Zeta);

        // compute the first two terms of OS eqn A20
        QUICKDouble _tmp = 2.0 * sqrt(Zeta/PI) * ssoverlap;            

        // multiply by contraction coefficients
        _tmp *= LOC2(devSim.gccoeff, III, KsumtypeI, MAXPRIM, gpu->nbasis) * LOC2(devSim.gccoeff, JJJ, KsumtypeJ, MAXPRIM, gpu->nbasis); 

//        for(int iatom=0; iatom<totalatom; ++iatom){

            QUICKDouble Cx = LOC2(devSim.allxyz, 0 , iatom, 3, totalatom);
            QUICKDouble Cy = LOC2(devSim.allxyz, 1 , iatom, 3, totalatom);
            QUICKDouble Cz = LOC2(devSim.allxyz, 2 , iatom, 3, totalatom);           
            QUICKDouble chg = -1.0 * devSim.allchg[iatom];

            // compute OS A21
            QUICKDouble U = Zeta * ( pow(Px-Cx, 2) + pow(Py-Cy, 2) + pow(Pz-Cz, 2) );

            // compute boys function values, the third term of OS A20
            QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];

            FmT(I+J, U, YVerticalTemp);

            // compute all auxilary integrals and store
            for (int i = 0; i<=I+J; i++) {
                VY(0, 0, i) = VY(0, 0, i) * _tmp * chg;
                //printf("aux: %d %f \n", i, VY(0, 0, i));
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme. 
            OEint_vertical(I, J, II, JJ, Px-Ax, Py-Ay, Pz-Az, Px-Bx, Py-By, Pz-Bz, Px-Cx, Py-Cy, Pz-Cz, Zeta, store, YVerticalTemp);
//        }

    }

    // retrive computed integral values from store array and update the Fock matrix 
    addint_oei(I, J, II, JJ, iatom, store);

}

__global__ void getOEI_kernel(){

  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int totalThreads = blockDim.x*gridDim.x;

  unsigned int jshell = devSim.Qshell;

  unsigned int totalatom = devSim.natom+devSim.nextatom;

  for (QUICKULL i = offset; i < jshell * jshell * totalatom; i+= totalThreads) {

    // use the global index to obtain shell pair. Note that here we obtain a couple of indices that helps us to obtain
    // shell number (ii and jj) and quantum numbers (iii, jjj).
    
    unsigned int iatom = (int) i/(jshell * jshell);
    unsigned int idx   = i - iatom * jshell * jshell;

    int II = devSim.sorted_OEICutoffIJ[idx].x;
    int JJ = devSim.sorted_OEICutoffIJ[idx].y;

    // get the shell numbers of selected shell pair
    int ii = devSim.sorted_Q[II];
    int jj = devSim.sorted_Q[JJ];

    // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
    int iii = devSim.sorted_Qnumber[II];
    int jjj = devSim.sorted_Qnumber[JJ];
    
    //printf(" tid: %d II JJ ii jj iii jjj %d  %d  %d  %d  %d  %d \n", (int) i, II, JJ, ii, jj, iii, jjj);

    // compute coulomb attraction for the selected shell pair.  
    iclass_oei(iii, jjj, ii, jj, iatom, totalatom);

  }

}
