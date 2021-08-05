/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 07/29/2021                            !
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


__device__ void add_oei_grad(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ, \
                             unsigned int iatom, QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB){

    // obtain the start and final basis function indices for given shells II and JJ. They will help us to save the integral
    // contribution into correct location in Fock matrix. 
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;

    int nbasis = devSim.nbasis;

    for (int III = III1; III <= III2; III++) {

        int i = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis), LOC2(devSim.KLMN,1,III-1,3,nbasis),\
                      LOC2(devSim.KLMN,2,III-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {

            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);

            if( III != JJJ ) DENSEJI *= 2.0;

            QUICKDouble constant =  devSim.cons[III-1] * devSim.cons[JJJ-1] * DENSEJI;

                // devTrans maps a basis function with certain angular momentum to store array. Get the correct indices now.  
                int j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,nbasis),\
                               LOC2(devSim.KLMN,2,JJJ-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

                // sum up gradient wrt x-coordinate of first center                 
                int itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis)+1, LOC2(devSim.KLMN,1,III-1,3,nbasis),\
                               LOC2(devSim.KLMN,2,III-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

                   AGradx += constant * LOC2(storeAA, itemp-1, j-1, STOREDIM, STOREDIM);

                if(LOC2(devSim.KLMN,0,III-1,3,nbasis) >= 1){
           
                    itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis)-1, LOC2(devSim.KLMN,1,III-1,3,nbasis),\
                                  LOC2(devSim.KLMN,2,III-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);
                 
                    AGradx -= constant * LOC2(devSim.KLMN,0,III-1,3,nbasis) * LOC2(store, itemp-1, j-1, STOREDIM, STOREDIM);
                }
           
                // sum up gradient wrt y-coordinate of first center
                itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis), LOC2(devSim.KLMN,1,III-1,3,nbasis)+1,\
                               LOC2(devSim.KLMN,2,III-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

                   AGrady += constant * LOC2(storeAA, itemp-1, j-1, STOREDIM, STOREDIM);


                if(LOC2(devSim.KLMN,1,III-1,3,nbasis) >= 1){
              
                    itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis), LOC2(devSim.KLMN,1,III-1,3,nbasis)-1,\
                                  LOC2(devSim.KLMN,2,III-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);
              
                    AGrady -= constant * LOC2(devSim.KLMN,1,III-1,3,nbasis) * LOC2(store, itemp-1, j-1, STOREDIM, STOREDIM);
            
                }
           
                // sum up gradient wrt z-coordinate of first center
                itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis), LOC2(devSim.KLMN,1,III-1,3,nbasis),\
                               LOC2(devSim.KLMN,2,III-1,3,nbasis)+1, TRANSDIM, TRANSDIM, TRANSDIM);


                   AGradz += constant * LOC2(storeAA, itemp-1, j-1, STOREDIM, STOREDIM);


                if(LOC2(devSim.KLMN,2,III-1,3,nbasis) >= 1){

                    itemp = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,nbasis), LOC2(devSim.KLMN,1,III-1,3,nbasis),\
                                  LOC2(devSim.KLMN,2,III-1,3,nbasis)-1, TRANSDIM, TRANSDIM, TRANSDIM);

                    AGradz -= constant * LOC2(devSim.KLMN,2,III-1,3,nbasis) * LOC2(store, itemp-1, j-1, STOREDIM, STOREDIM);
                }

                // sum up gradient wrt x-coordinate of second center 
                j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis)+1, LOC2(devSim.KLMN,1,JJJ-1,3,nbasis),\
                               LOC2(devSim.KLMN,2,JJJ-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);


                   BGradx += constant * LOC2(storeBB, i-1, j-1, STOREDIM, STOREDIM);
 

                if(LOC2(devSim.KLMN,0,JJJ-1,3,nbasis) >= 1){

                    j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis)-1, LOC2(devSim.KLMN,1,JJJ-1,3,nbasis),\
                              LOC2(devSim.KLMN,2,JJJ-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM); 
 
                    BGradx -= constant * LOC2(devSim.KLMN,0,JJJ-1,3,nbasis) * LOC2(store, i-1, j-1, STOREDIM, STOREDIM);
 
                }


                // sum up gradient wrt y-coordinate of second center 
                j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,nbasis)+1,\
                               LOC2(devSim.KLMN,2,JJJ-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);

                   BGrady += constant * LOC2(storeBB, i-1, j-1, STOREDIM, STOREDIM);
 
                if(LOC2(devSim.KLMN,1,JJJ-1,3,nbasis) >= 1){
 
                    j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,nbasis)-1,\
                              LOC2(devSim.KLMN,2,JJJ-1,3,nbasis), TRANSDIM, TRANSDIM, TRANSDIM);
 
                    BGrady -= constant * LOC2(devSim.KLMN,1,JJJ-1,3,nbasis) * LOC2(store, i-1, j-1, STOREDIM, STOREDIM);
                }

                // sum up gradient wrt z-coordinate of second center 
                j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,nbasis),\
                               LOC2(devSim.KLMN,2,JJJ-1,3,nbasis)+1, TRANSDIM, TRANSDIM, TRANSDIM);

                   BGradz += constant * LOC2(storeBB, i-1, j-1, STOREDIM, STOREDIM);
            
                if(LOC2(devSim.KLMN,2,JJJ-1,3,nbasis) >= 1){
 
                    j = (int) LOC3(devTrans, LOC2(devSim.KLMN,0,JJJ-1,3,nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,nbasis),\
                              LOC2(devSim.KLMN,2,JJJ-1,3,nbasis)-1, TRANSDIM, TRANSDIM, TRANSDIM);
 
                    BGradz -= constant * LOC2(devSim.KLMN,2,JJJ-1,3,nbasis) * LOC2(store, i-1, j-1, STOREDIM, STOREDIM);
                }

        }
    }

    int AStart = (devSim.katom[II]-1) * 3;
    int BStart = (devSim.katom[JJ]-1) * 3;
    int CStart = iatom * 3;

#ifdef LEGACY_ATOMIC_ADD
    GRADADD(devSim.gradULL[AStart], AGradx);
    GRADADD(devSim.gradULL[AStart + 1], AGrady);
    GRADADD(devSim.gradULL[AStart + 2], AGradz);

    GRADADD(devSim.gradULL[BStart], BGradx);
    GRADADD(devSim.gradULL[BStart + 1], BGrady);
    GRADADD(devSim.gradULL[BStart + 2], BGradz);

if(iatom < devSim.natom){
    GRADADD(devSim.gradULL[CStart], (-AGradx-BGradx));
    GRADADD(devSim.gradULL[CStart + 1], (-AGrady-BGrady));
    GRADADD(devSim.gradULL[CStart + 2], (-AGradz-BGradz));
}else{
    CStart = (iatom-devSim.natom) * 3;
    GRADADD(devSim.ptchg_gradULL[CStart], (-AGradx-BGradx));
    GRADADD(devSim.ptchg_gradULL[CStart + 1], (-AGrady-BGrady));
    GRADADD(devSim.ptchg_gradULL[CStart + 2], (-AGradz-BGradz));
}

#else

    QUICKADD(devSim.grad[AStart], AGradx);
    QUICKADD(devSim.grad[AStart + 1], AGrady);
    QUICKADD(devSim.grad[AStart + 2], AGradz);

    QUICKADD(devSim.grad[BStart], BGradx);
    QUICKADD(devSim.grad[BStart + 1], BGrady);
    QUICKADD(devSim.grad[BStart + 2], BGradz);

if(iatom < devSim.natom){
    QUICKADD(devSim.grad[CStart], (-AGradx-BGradx));
    QUICKADD(devSim.grad[CStart + 1], (-AGrady-BGrady));
    QUICKADD(devSim.grad[CStart + 2], (-AGradz-BGradz));
}else{
    CStart = (iatom-devSim.natom) * 3;
    QUICKADD(devSim.ptchg_grad[CStart], (-AGradx-BGradx));
    QUICKADD(devSim.ptchg_grad[CStart + 1], (-AGrady-BGrady));
    QUICKADD(devSim.ptchg_grad[CStart + 2], (-AGradz-BGradz));
}

#endif

}


__device__ void iclass_oei_grad(unsigned int I, unsigned int J, unsigned int II, unsigned int JJ , unsigned int iatom, unsigned int totalatom){

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

    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;

    /* 
    sum of basis functions for shell II and JJ. For eg, for a p shell, KsumtypeI or KsumtypeJ
    would be 3.
    */ 
    //int KsumtypeI = devSim.Ksumtype[II]-1;
    //int KsumtypeJ = devSim.Ksumtype[JJ]-1;

    /*
    At this point, we will need 3 arrays. The first, store, will keep the sum of primitive integral 
    values as in oei code. Gradient calculation also requires scaling certain primitive integral values
    by the exponents on each center. It is possible to eliminate the second and third arrays, by looping
    through the primitives and updating the grad vector during each cycle. But this incurs a huge performance
    penalty.
     */

    QUICKDouble store[STOREDIM*STOREDIM];
    QUICKDouble storeAA[STOREDIM*STOREDIM];
    QUICKDouble storeBB[STOREDIM*STOREDIM];

    /*
     initialize the region of store array that we will be using. This region is determined by looking at the
     Sumindex array with angular momentums of the shells.
    */  

    for(int i=Sumindex[J]; i< Sumindex[J+2]; ++i){
        for(int j=Sumindex[I]; j<Sumindex[I+2]; ++j){
            if (i < STOREDIM && j < STOREDIM) {
                LOC2(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for(int i=Sumindex[J]; i< Sumindex[J+2]; ++i){
        for(int j=Sumindex[I+1]; j<Sumindex[I+3]; ++j){
           if (i < STOREDIM && j < STOREDIM) {
               LOC2(storeAA, j, i, STOREDIM, STOREDIM) = 0.0;
           }
        }
    }


    for(int i=Sumindex[J+1]; i< Sumindex[J+3]; ++i){
        for(int j=Sumindex[I]; j<Sumindex[I+2]; ++j){
           if (i < STOREDIM && j < STOREDIM) {
               LOC2(storeBB, j, i, STOREDIM, STOREDIM) = 0.0;
           }
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


        QUICKDouble AA = LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, nbasis);

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients 
        QUICKDouble Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        if(abs(Xcoeff_oei) > devSim.integralCutoff){
//        for(int iatom=0; iatom<totalatom; ++iatom){

            QUICKDouble Cx = LOC2(devSim.allxyz, 0 , iatom, 3, totalatom);
            QUICKDouble Cy = LOC2(devSim.allxyz, 1 , iatom, 3, totalatom);
            QUICKDouble Cz = LOC2(devSim.allxyz, 2 , iatom, 3, totalatom);           
            QUICKDouble chg = -1.0 * devSim.allchg[iatom];

            // compute OS A21
            QUICKDouble U = Zeta * ( pow(Px-Cx, 2) + pow(Py-Cy, 2) + pow(Pz-Cz, 2) );

            // compute boys function values, the third term of OS A20
            QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];

            FmT(I+J+2, U, YVerticalTemp);

            // compute all auxilary integrals and store
            for (int n = 0; n<=I+J+2; n++) {
                VY(0, 0, n) = VY(0, 0, n) * Xcoeff_oei * chg;
                //printf("aux: %d %f \n", i, VY(0, 0, i));
            }

            /*
            oei grad assembler will save primitive integral values into the following temporary array. 
            Note that we dont worry about the initialization since we store integral values rather than 
            summing up values.
            */
            QUICKDouble store2[STOREDIM*STOREDIM];

            // decompose all attraction integrals to their auxilary integrals through VRR scheme. 
            oei_grad_vertical(I, J, II, JJ, Px-Ax, Py-Ay, Pz-Az, Px-Bx, Py-By, Pz-Bz, Px-Cx, Py-Cy, Pz-Cz, Zeta, store2, YVerticalTemp);

            // sum up primitive integral values into store array
            for(int i=Sumindex[J]; i< Sumindex[J+2]; ++i){
              for(int j=Sumindex[I]; j<Sumindex[I+2]; ++j){
                if (i < STOREDIM && j < STOREDIM) {
                  LOC2(store, j, i , STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM);
                }
              }
            }

            // scale primitive integral values with exponent of the first center and add up into storeAA
            for(int i=Sumindex[J]; i< Sumindex[J+2]; ++i){
              for(int j=Sumindex[I+1]; j<Sumindex[I+3]; ++j){
                if (i < STOREDIM && j < STOREDIM) {
                  LOC2(storeAA, j, i , STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM) * AA * 2.0;
                }
              }
            }
             
            // scale primitive integral values with exponent of the second center and add up into storeBB
            for(int i=Sumindex[J+1]; i< Sumindex[J+3]; ++i){
              for(int j=Sumindex[I]; j<Sumindex[I+2]; ++j){
                if (i < STOREDIM && j < STOREDIM) {
                  LOC2(storeBB, j, i , STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM) * BB * 2.0;
                }
              }
            }

        }

    }

    // retrive computed integral values from store array and update the Fock matrix 
    add_oei_grad(I, J, II, JJ, iatom, store, storeAA, storeBB);

}

__global__ void get_oei_grad_kernel(){

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
    iclass_oei_grad(iii, jjj, ii, jj, iatom, totalatom);

  }

}
