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

__device__ void iclass_oei(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ ){

    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
     which means they are corrosponding coorinates for shell II, JJ and nuclei.
     */

    QUICKDouble Ax = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble Ay = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble Az = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);

    QUICKDouble Bx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    QUICKDouble By = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    QUICKDouble Bz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);

    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];

    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;


    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    QUICKDouble store[STOREDIM*STOREDIM];

    /*
     Initial the neccessary element for
     */

    for(int i=0; i<STOREDIM; ++i){
        for(int j=0; j<STOREDIM; ++j){
            LOC2(store, j, i, STOREDIM, STOREDIM) = 0;
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

        // calculation below may be reduced by using Xcoeff

        QUICKDouble ssoverlap = PI_TO_3HALF * pow(Zeta, -1.5) * exp( - LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis) *\
        LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis) * (pow(Ax-Bx, 2) + pow(Ay-By, 2) + pow(Az-Bz, 2)) / Zeta);

        // compute the first two terms of OS eqn A20
        QUICKDouble _tmp = 2.0 * sqrt(Zeta/PI) * ssoverlap;            

        for(int iatom=0; iatom<devSim.natom; ++iatom){

            QUICKDouble Cx = LOC2(devSim.xyz, 0 , iatom, 3, devSim.natom);
            QUICKDouble Cy = LOC2(devSim.xyz, 1 , iatom, 3, devSim.natom);
            QUICKDouble Cz = LOC2(devSim.xyz, 2 , iatom, 3, devSim.natom);           
            QUICKDouble chg = -1.0 * devSim.chg[iatom];

            // compute OS A21
            QUICKDouble U = Zeta * ( pow(Px-Cx, 2) + pow(Py-Cy, 2) + pow(Pz-Cz, 2) );

            // compute boys function values, the third term of OS A20
            QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];

            FmT(I+J, U, YVerticalTemp);

            // compute all auxilary integrals and store
            for (int i = 0; i<=I+J; i++) {
                VY(0, 0, i) = VY(0, 0, i) * _tmp * chg;
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme. 
            OEint_vertical(I, J, Px-Ax, Py-Ay, Pz-Az, Px-Bx, Py-By, Pz-Bz, Px-Cx, Py-Cy, Pz-Cz, Zeta, store, YVerticalTemp);
        }

    }

}

__global__ void getOEI_kernel(){

  unsigned int offset = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int totalThreads = blockDim.x*gridDim.x;

  for (QUICKULL i = offset; i < devSim.nshell * devSim.nshell; i+= totalThreads) {

    /************* assume nshell = nbasis for testing *************/
    QUICKULL a = (QUICKULL) i/devSim.nshell;
    QUICKULL b = (QUICKULL) (i - a*devSim.nshell);

    printf("I %d J %d \n", (int) a, (int) b);

    iclass_oei(a, b, a, b);

  }

}
