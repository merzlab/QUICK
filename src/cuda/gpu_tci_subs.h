
/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 08/31/2021                            !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains driver functions required for computing 3 !
  ! center integrals necessary for CEW method.                          !
  !---------------------------------------------------------------------!
*/
#include "gpu_common.h"

#undef STOREDIM

#ifdef int_spd
#define STOREDIM STOREDIM_S
#else
#define STOREDIM STOREDIM_L
#endif


/*
Note that this driver implementations are very similar to the ones implemented by Yipu Miao in gpu_get2e_subs.h.
To understand the following comments better, please refer to Figure 2(b) and 2(d) in Miao and Merz 2015 paper. 

 In the following kernel, we treat f orbital into 5 parts.
 
 type:   ss sp ps sd ds pp dd sf pf | df ff |
 ss                                 |       |
 sp                                 |       |
 ps                                 | zone  |
 sd                                 |  2    |
 ds         zone 0                  |       |
 pp                                 |       |
 dd                                 |       |
 sf                                 |       |
 pf                                 |       |
 -------------------------------------------
 df         zone 1                  | z | z |
 ff                                 | 3 | 4 |
 -------------------------------------------
 
 
 because the single f orbital kernel is impossible to compile completely, we treat VRR as:
 
 
 I+J  0 1 2 3 4 | 5 | 6 |
 0 ----------------------
 1|             |       |
 2|   Kernel    |  K2   |
 3|     0       |       |
 4|             |       |
 -----------------------|
 5|   Kernel    | K | K |
 6|     1       | 3 | 4 |
 ------------------------
 
 Their responses for
              I+J          K+L
 Kernel 0:   0-4           0-4
 Kernel 1:   0-4           5,6
 Kernel 2:   5,6           0-4
 Kernel 3:   5             5,6
 Kernel 4:   6             5,6
 
 Integrals in zone need kernel:
 zone 0: kernel 0
 zone 1: kernel 0,1
 zone 2: kernel 0,2
 zone 3: kernel 0,1,2,3
 zone 4: kernel 0,1,2,3,4
 
 so first, kernel 0: zone 0,1,2,3,4 (get_tci_kernel()), if no f, then that's it.
 second,   kernel 1: zone 1,3,4(get_tci_kernel_spdf())
 then,     kernel 2: zone 2,3,4(get_tci_kernel_spdf2())
 then,     kernel 3: zone 3,4(get_tci_kernel_spdf3())
 finally,  kernel 4: zone 4(get_tci_kernel_spdf4())

 */
#ifdef int_spd
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_tci_kernel()
#elif defined int_spdf2
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_tci_kernel_spdf2()
#endif
{
    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    
    // jshell and jshell2 defines the regions in i+j and k+l axes respectively.    
    // sqrQshell= Qshell x Qshell; where Qshell is the number of sorted shells (see gpu_upload_basis_ in gpu.cu)
    // for details on sorting. 
 
#ifdef int_spd
/*
 Here we walk through full cutoff matrix.

 --sqrQshell --
 _______________ 
 |             |  |
 |             |  |
 |             | sqrQshell
 |             |  |
 |             |  |
 |_____________|  |

*/

    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;

#elif defined int_spdf2

    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;

#endif

    unsigned int totalatom = devSim.natom+devSim.nextatom;

    for (QUICKULL i = offside; i < jshell * totalatom; i+= totalThreads) {
        
#ifdef int_spd
       /* 
        QUICKULL a, b;
        
        double aa = (double)((i+1)*1E-4);
        QUICKULL t = (QUICKULL)(sqrt(aa)*1E2);
        if ((i+1)==t*t) {
            t--;
        }
        
        QUICKULL k = i-t*t;
        if (k<=t) {
            a = k;
            b = t;
        }else {
            a = t;
            b = 2*t-k;
        }
        */
        // Zone 0

        QUICKULL iatom = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - iatom*jshell);

#elif defined int_spdf2
        
        // Zone 2
        QUICKULL iatom = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - iatom*jshell);
        //a = a + devSim.fStart;

#endif

#ifdef CUDA_MPIV
        if(devSim.mpi_bcompute[b] > 0){
#endif 

        int II = devSim.sorted_YCutoffIJ[b].x;
        int JJ = devSim.sorted_YCutoffIJ[b].y;

        int ii = devSim.sorted_Q[II];
        int jj = devSim.sorted_Q[JJ];

        
//        printf("b II JJ ii jj %lu %lu %d %d %d %d \n", jshell, b, II, JJ, ii, jj);

//        int kk = 0;
//        int ll = 0;    
            
            
//            int nshell = devSim.nshell;
/*            QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim.cutMatrix, ii, jj, nshell, nshell), 4.0*LOC2(devSim.cutMatrix, kk, ll, nshell, nshell)),
                                    MAX(MAX(LOC2(devSim.cutMatrix, ii, ll, nshell, nshell),     LOC2(devSim.cutMatrix, ii, kk, nshell, nshell)),
                                        MAX(LOC2(devSim.cutMatrix, jj, kk, nshell, nshell),     LOC2(devSim.cutMatrix, jj, ll, nshell, nshell))));
            
            if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))> devSim.coreIntegralCutoff && \
                (LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell) * DNMax) > devSim.coreIntegralCutoff) {
*/                
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];

                // assign values to dummy variables, to be cleaned up eventually
                //for(int iatom=0; iatom < devSim.natom+devSim.nextatom; iatom++ ){ 
#ifdef int_spd
                    iclass_tci(iii, jjj, ii, jj, iatom, totalatom, devSim.YVerticalTemp, devSim.store);
                
#elif defined int_spdf2
                if ( (iii + jjj) > 4 && (iii + jjj) <= 6 ) {
                    iclass_tci_spdf2(iii, jjj, ii, jj, iatom, totalatom, devSim.YVerticalTemp, devSim.store);
                }
                
#endif
                //}
/*                
            }
*/
#ifdef CUDA_MPIV
        }      
#endif        
    }
}

/*
 iclass_tci subroutine is to generate 3 center intergrals using HRR and VRR method.
*/
#ifdef int_spd
__device__ __forceinline__ void iclass_tci
#elif defined int_spdf2
__device__ __forceinline__ void iclass_tci_spdf2
#endif
                                      (int I, int J, unsigned int II, unsigned int JJ, int iatom, \
                                      unsigned int totalatom, QUICKDouble* YVerticalTemp, QUICKDouble* store)
{
    
    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);
    
    QUICKDouble RCx = LOC2(devSim.allxyz, 0, iatom, 3, totalatom);
    QUICKDouble RCy = LOC2(devSim.allxyz, 1, iatom, 3, totalatom);
    QUICKDouble RCz = LOC2(devSim.allxyz, 2, iatom, 3, totalatom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = 1;
    int kPrimL = 1;
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    //int kStartK = 0;
    //int kStartL = 0;
    
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    //QUICKDouble store[STOREDIM*STOREDIM];
    
    /*
     Initial the neccessary element for
     */

     int K=0;
     int L=0;
    
#ifdef int_spd
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf2
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#endif
    
    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage.
         
         */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
       // QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
 
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int)j/kPrimK;
           // int KKK = (int) j-kPrimK*LLL;
            
           // if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                
                //int kk_start = devSim.prim_start[KK];
                //int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = devSim.lri_zeta;
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1/devSim.lri_zeta) * devSim.lri_cc[iatom] * pow(devSim.lri_zeta/PI, 1.5);
                
                /*
                 Q' is the weighting center of K and L
                 --->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 --->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 expo(I) + expo(J) + expo(K) + expo(L)
                 ->  ->  2
                 RPQ =| P - Q |
                 
                 ->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = RCx;
                QUICKDouble Qy = RCy;
                QUICKDouble Qz = RCz;

                
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                //QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
                for (int i = 0; i<=I+J+K+L; i++) {

                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
#ifdef int_spd
                tci::vertical(I, J, K, L, YVerticalTemp, store, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
         
#elif defined int_spdf2
                
                tci::vertical_spdf2(I, J, K, L, YVerticalTemp, store, \
                              Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                              Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                              0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
         
#endif
                
           // }
        }
    }
    
    
    // IJKLTYPE is the I, J, K,L type
    int IJKLTYPE = (int) (1000 * I + 100 *J + 10 * K + L);
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
    
    
    RDx = 0.0;
    RDy = 0.0;
    RDz = 0.0;
    
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    //int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    //int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    //int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    //int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    
    // maxIJKL is the max of I,J,K,L
    int maxIJKL = (int)MAX(MAX(I,J),MAX(K,L));
    
    if (((maxIJKL == 2)&&(J != 0 || L!=0)) || (maxIJKL >= 3)) {
        IJKLTYPE = 999;
    }
    
    /*QUICKDouble hybrid_coeff = 0.0;
    if (devSim.method == HF){
        hybrid_coeff = 1.0;
    }else if (devSim.method == B3LYP){
        hybrid_coeff = 0.2;
    }else if (devSim.method == DFT){
        hybrid_coeff = 0.0;
    }else if(devSim.method == LIBXC){
        hybrid_coeff = devSim.hyb_coeff;                        
    }*/
    
    
    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
           int KKK=0;
           int LLL=0; 
/*            for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    
                    if (III < KKK ||
                        ((III == JJJ) && (III == LLL)) ||
                        ((III == JJJ) && (III  < LLL)) ||
                        ((JJJ == LLL) && (III  < JJJ)) ||
                        ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
*/                        

#ifdef int_spd
                        QUICKDouble Y = (QUICKDouble) hrrwhole_tci
#elif defined int_spdf2
                        QUICKDouble Y = (QUICKDouble) hrrwhole_tci_2_2
#else                        
                        QUICKDouble Y = (QUICKDouble) hrrwhole_tci_2
                       
#endif
                                                               (I, J, K, L,\
                                                               III, JJJ, KKK, LLL, IJKLTYPE, store, \
                                                               RAx, RAy, RAz, RBx, RBy, RBz, \
                                                               RCx, RCy, RCz, RDx, RDy, RDz);

//printf("II JJ III JJJ Y %d %d %d %d %f \n", II, JJ, III, JJJ, Y);
#ifdef int_spd
                        if (abs(Y) > 0.0e0)
#else
                        if (abs(Y) > DEFAULT_INTEGRAL_CUTOFF)
#endif
                        {
                            addint_tci(devSim.oULL, Y, III, JJJ, KKK, LLL, devSim.hyb_coeff, devSim.dense, devSim.nbasis);
                        }
/*                        
                    }
                }
            }
*/
        }
    }
    return;
}


#ifndef new_quick_2_gpu_tci_subs_h
#define new_quick_2_gpu_tci_subs_h

__device__ __forceinline__ void addint_tci(QUICKULL* oULL, QUICKDouble Y, int III, int JJJ, int KKK, int LLL,QUICKDouble hybrid_coeff,  QUICKDouble* dense, int nbasis)
{
    
    QUICKULL val1 = (QUICKULL) (fabs(Y*OSCALE) + (QUICKDouble)0.5);
    if ( Y < (QUICKDouble)0.0) val1 = 0ull - val1;
    QUICKADD(LOC2(oULL, JJJ-1, III-1, nbasis, nbasis), val1);    

}

#include "gpu_fmt.h"

/*
 sqr for double precision. there no internal function to do that in fast-math-lib of CUDA
 */
__device__ __forceinline__ QUICKDouble quick_dsqr(QUICKDouble a)
{
    return a*a;
}

#endif

#undef STOREDIM
