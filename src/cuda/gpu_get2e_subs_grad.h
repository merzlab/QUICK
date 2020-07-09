//
//  gpu_get2e_subs_grad.h
//  new_quick
//
//  Created by Yipu Miao on 1/22/14.
//
//

#include "gpu_common.h"

#undef STOREDIM

#ifdef int_spd
#define STOREDIM STOREDIM_S
#else
#define STOREDIM STOREDIM_L
#endif

#ifdef int_spd
__global__ void getGrad_kernel()
#elif defined int_spdf
__global__ void getGrad_kernel_spdf()
#elif defined int_spdf2
__global__ void getGrad_kernel_spdf2()
#elif defined int_spdf3
__global__ void getGrad_kernel_spdf3()
#elif defined int_spdf4
__global__ void getGrad_kernel_spdf4()
#elif defined int_spdf5
__global__ void getGrad_kernel_spdf5()
#elif defined int_spdf6
__global__ void getGrad_kernel_spdf6()
#elif defined int_spdf7
__global__ void getGrad_kernel_spdf7()
#elif defined int_spdf8
__global__ void getGrad_kernel_spdf8()
#endif
{

    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    
    
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell;
    
    for (QUICKULL i = offside; i<jshell2*jshell; i+= totalThreads) {
        
        /*
        QUICKULL a, b;
        
        // That's simply because no sqrt for ULL
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
        
        QUICKULL a = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - a*jshell);

#ifdef CUDA_MPIV
        if(devSim.mpi_bcompute[a] > 0){
#endif
                
        int II = devSim.sorted_YCutoffIJ[a].x;
        int KK = devSim.sorted_YCutoffIJ[b].x;
        
        int ii = devSim.sorted_Q[II];
        int kk = devSim.sorted_Q[KK];
        
        if (ii<=kk){
            
            int JJ = devSim.sorted_YCutoffIJ[a].y;
            int LL = devSim.sorted_YCutoffIJ[b].y;
            
            int jj = devSim.sorted_Q[JJ];
            int ll = devSim.sorted_Q[LL];
            
            
            if ( !((devSim.katom[ii] == devSim.katom[jj]) &&
                   (devSim.katom[ii] == devSim.katom[kk]) &&
                   (devSim.katom[ii] == devSim.katom[ll]))     // In case 4 indices are in the same atom
                ) {
                
                int nshell = devSim.nshell;
                
                QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim.cutMatrix, ii, jj, nshell, nshell), 4.0*LOC2(devSim.cutMatrix, kk, ll, nshell, nshell)),
                                        MAX(MAX(LOC2(devSim.cutMatrix, ii, ll, nshell, nshell),     LOC2(devSim.cutMatrix, ii, kk, nshell, nshell)),
                                            MAX(LOC2(devSim.cutMatrix, jj, kk, nshell, nshell),     LOC2(devSim.cutMatrix, jj, ll, nshell, nshell))));
                
                
                if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))> devSim.integralCutoff && \
                    (LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell) * DNMax) > devSim.integralCutoff) {
                    
                    int iii = devSim.sorted_Qnumber[II];
                    int jjj = devSim.sorted_Qnumber[JJ];
                    int kkk = devSim.sorted_Qnumber[KK];
                    int lll = devSim.sorted_Qnumber[LL];
#ifdef int_spd
                    iclass_grad(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf
                    if ( (kkk + lll) >= 4 ) {
                        iclass_grad_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                    }
#elif defined int_spdf2
                    if ( (iii + jjj) >= 4 ) {
                        iclass_grad_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                    }
#elif defined int_spdf3
                    iclass_grad_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf4
                    iclass_grad_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf5
                    iclass_grad_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf6
                    iclass_grad_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf7
                    iclass_grad_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#elif defined int_spdf8
                    iclass_grad_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
#endif
                    
                }
            }
        }

#ifdef CUDA_MPIV
        }
#endif

    }
}


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef int_spd
__device__ __forceinline__ void iclass_grad

(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax)
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
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    QUICKDouble store[STOREDIM*STOREDIM];
    QUICKDouble storeAA[STOREDIM*STOREDIM];
    QUICKDouble storeBB[STOREDIM*STOREDIM];
    QUICKDouble storeCC[STOREDIM*STOREDIM];
    
    
    for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
        for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                if (j < Sumindex[I+J+2] && i < Sumindex[K+L+2]) {
                    LOC2(store, j, i, STOREDIM, STOREDIM) = 0;
                }
                
                if (j >= Sumindex[I+1]) {
                    LOC2(storeAA, j, i, STOREDIM, STOREDIM) = 0;
                    LOC2(storeBB, j, i, STOREDIM, STOREDIM) = 0;
                }
                
                if (i >= Sumindex[K+1]) {
                    LOC2(storeCC, j, i, STOREDIM, STOREDIM) = 0;
                }
            }
        }
    }
    
    
    
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
        
        QUICKDouble AA = LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int) j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                
                QUICKDouble CC = LOC2(devSim.gcexpo, KKK , devSim.Ksumtype[KK] - 1, MAXPRIM, devSim.nbasis);
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
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
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
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L+1, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
                
                for (int i = 0; i<=I+J+K+L+1; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
                
                QUICKDouble store2[STOREDIM*STOREDIM];
                
                
                vertical2(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                          Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                          Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                          0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
                
                
                
                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOC2(store, j, i , STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM);
                        }
                    }
                }
                
                
                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I+1]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOC2(storeAA, j, i, STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM) * AA * 2 ;
                            LOC2(storeBB, j, i, STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM) * BB * 2 ;
                        }
                        
                    }
                }
                
                for (int i = Sumindex[K+1]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOC2(storeCC, j, i, STOREDIM, STOREDIM) += LOC2(store2, j, i, STOREDIM, STOREDIM) * CC * 2 ;
                        }
                    }
                }
                
                
            }
        }
    }
    
    
    
    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;
    
    int         AStart = (devSim.katom[II]-1) * 3;
    int         BStart = (devSim.katom[JJ]-1) * 3;
    int         CStart = (devSim.katom[KK]-1) * 3;
    int         DStart = (devSim.katom[LL]-1) * 3;
    
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
    
    
    RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
    RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
    RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
    
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    
    int  IJKLTYPE = 999;
    
    int  nbasis = devSim.nbasis;

    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    
                    if (III < KKK ||
                        ((III == JJJ) && (III == LLL)) ||
                        ((III == JJJ) && (III  < LLL)) ||
                        ((JJJ == LLL) && (III  < JJJ)) ||
                        ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                        
                        QUICKDouble Yaax, Yaay, Yaaz;
                        QUICKDouble Ybbx, Ybby, Ybbz;
                        QUICKDouble Yccx, Yccy, Yccz;
                        
                        hrrwholegrad(&Yaax, &Yaay, &Yaaz, \
                                     &Ybbx, &Ybby, &Ybbz, \
                                     &Yccx, &Yccy, &Yccz, \
                                     I, J, K, L,\
                                     III, JJJ, KKK, LLL, IJKLTYPE, \
                                     store, storeAA, storeBB, storeCC, \
                                     RAx, RAy, RAz, RBx, RBy, RBz, \
                                     RCx, RCy, RCz, RDx, RDy, RDz);
                        
                        QUICKDouble constant = 0.0 ;
                        
                        QUICKDouble DENSEKI = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEKJ = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELJ = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELI = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSELK = (QUICKDouble) LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis);
                        QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);
                        
                        if (II < JJ && II < KK && KK < LL ||
                            ( III < KKK && III < JJJ && KKK < LLL)) {
                            constant = ( 4.0 * DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSELJ - devSim.hyb_coeff * DENSELI * DENSEKJ);
                        }else{
                            if (III < KKK) {
                                if( III == JJJ && KKK == LLL){
                                    constant = (DENSEJI * DENSELK - 0.5 * devSim.hyb_coeff * DENSEKI * DENSEKI);
                                }else if (JJJ == KKK && JJJ == LLL){
                                    constant = 2.0 * DENSELJ * DENSEJI - devSim.hyb_coeff * DENSELJ * DENSEJI;
                                }else if (KKK == LLL && III < JJJ && JJJ != KKK){
                                    constant = (2.0* DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSEKJ);
                                }else if ( III == JJJ && KKK < LLL){
                                    constant = (2.0* DENSELK * DENSEJI - devSim.hyb_coeff * DENSEKI * DENSELI);
                                }
                            }
                            else{
                                if (JJJ <= LLL) {
                                    if (III == JJJ && III == KKK && III == LLL) {
                                        // Do nothing
                                    }else if (III==JJJ && III==KKK && III < LLL){
                                        //constant = DENSELI * DENSEJI;
					constant = 2.0 * DENSELI * DENSEJI - devSim.hyb_coeff * DENSELI * DENSEJI;
                                    }else if (III==KKK && JJJ==LLL && III < JJJ){
                                        //constant = (1.5 * DENSEJI * DENSEJI - 0.5 * DENSELJ * DENSEKI);
					constant = (2.0 * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSELJ * DENSEKI);
                                    }else if (III== KKK && III < JJJ && JJJ < LLL){
                                        //constant = (3.0 * DENSEJI * DENSELI - DENSELJ * DENSEKI);
					constant = (4.0 * DENSEJI * DENSELI - devSim.hyb_coeff * DENSEJI * DENSELI - devSim.hyb_coeff * DENSELJ * DENSEKI);
                                    }
                                }
                            }
                        }
                        
                        
                        AGradx += constant * Yaax;
                        AGrady += constant * Yaay;
                        AGradz += constant * Yaaz;
                        
                        BGradx += constant * Ybbx;
                        BGrady += constant * Ybby;
                        BGradz += constant * Ybbz;
                        
                        CGradx += constant * Yccx;
                        CGrady += constant * Yccy;
                        CGradz += constant * Yccz;
                    }
                }
            }
        }
    }
    
    
    
    GRADADD(devSim.gradULL[AStart], AGradx);
    GRADADD(devSim.gradULL[AStart + 1], AGrady);
    GRADADD(devSim.gradULL[AStart + 2], AGradz);
    
    
    GRADADD(devSim.gradULL[BStart], BGradx);
    GRADADD(devSim.gradULL[BStart + 1], BGrady);
    GRADADD(devSim.gradULL[BStart + 2], BGradz);
    
    
    GRADADD(devSim.gradULL[CStart], CGradx);
    GRADADD(devSim.gradULL[CStart + 1], CGrady);
    GRADADD(devSim.gradULL[CStart + 2], CGradz);
    
    
    GRADADD(devSim.gradULL[DStart], (-AGradx-BGradx-CGradx));
    GRADADD(devSim.gradULL[DStart + 1], (-AGrady-BGrady-CGrady));
    GRADADD(devSim.gradULL[DStart + 2], (-AGradz-BGradz-CGradz));
    
    return;
}
#else


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef int_spdf
__device__ __forceinline__ void iclass_grad_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_grad_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_grad_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_grad_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_grad_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_grad_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_grad_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_grad_spdf8
#endif

(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax)
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
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
    
    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;
    
    int         AStart = (devSim.katom[II]-1) * 3;
    int         BStart = (devSim.katom[JJ]-1) * 3;
    int         CStart = (devSim.katom[KK]-1) * 3;
    int         DStart = (devSim.katom[LL]-1) * 3;
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    
    
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
        
        QUICKDouble AA = LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int) j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.integralCutoff) {
                
                QUICKDouble CC = LOC2(devSim.gcexpo, KKK , devSim.Ksumtype[KK] - 1, MAXPRIM, devSim.nbasis);
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
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
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
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L+2, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
                
                for (int i = 0; i<=I+J+K+L+2; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
                QUICKDouble store2[STOREDIM*STOREDIM];
                
                
                for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM && !(i >= Sumindex[I+J+2] && j >= Sumindex[K+L+2])) {
                            LOC2(store2, j, i, STOREDIM, STOREDIM) = 0;
                        }
                    }
                }
                
                
#ifdef int_spdf
                vertical2_spdf(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf2
                vertical2_spdf2(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf3
                vertical2_spdf3(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf4
                vertical2_spdf4(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf5
                vertical2_spdf5(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf6
                vertical2_spdf6(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf7
                vertical2_spdf7(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf8
                vertical2_spdf8(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#endif
                
                
                
                
                QUICKDouble RBx, RBy, RBz;
                QUICKDouble RDx, RDy, RDz;
                
                RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
                RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
                RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
                
                
                RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
                RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
                RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
                
                int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
                int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
                int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
                int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
                int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
                int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
                int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
                int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
                
                
                int  IJKLTYPE = 999;
                
                int  nbasis = devSim.nbasis;
                
                for (int III = III1; III <= III2; III++) {
                    for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
                        for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                            for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                                
                                if (III < KKK ||
                                    ((III == JJJ) && (III == LLL)) ||
                                    ((III == JJJ) && (III  < LLL)) ||
                                    ((JJJ == LLL) && (III  < JJJ)) ||
                                    ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                                    
                                    QUICKDouble Yaax, Yaay, Yaaz;
                                    QUICKDouble Ybbx, Ybby, Ybbz;
                                    QUICKDouble Yccx, Yccy, Yccz;
#ifdef  int_spdf
                                    hrrwholegrad2_1
#elif defined int_spdf2
                                    hrrwholegrad2_2
#else
                                    hrrwholegrad2
#endif
                                    (&Yaax, &Yaay, &Yaaz, \
                                                  &Ybbx, &Ybby, &Ybbz, \
                                                  &Yccx, &Yccy, &Yccz, \
                                                  I, J, K, L,\
                                                  III, JJJ, KKK, LLL, IJKLTYPE, \
                                                  store2, AA, BB, CC, \
                                                  RAx, RAy, RAz, RBx, RBy, RBz, \
                                                  RCx, RCy, RCz, RDx, RDy, RDz);
                                    
                                    QUICKDouble constant = 0.0 ;
                                    
                                    QUICKDouble DENSEKI = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEKJ = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELJ = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELI = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSELK = (QUICKDouble) LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis);
                                    QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);
                                    
                                    if (II < JJ && II < KK && KK < LL ||
                                        ( III < KKK && III < JJJ && KKK < LLL)) {
                                        //constant = ( 4.0 * DENSEJI * DENSELK - DENSEKI * DENSELJ - DENSELI * DENSEKJ);
					constant = ( 4.0 * DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSELJ - devSim.hyb_coeff * DENSELI * DENSEKJ);
                                    }else{
                                        if (III < KKK) {
                                            if( III == JJJ && KKK == LLL){
                                                //constant = (DENSEJI * DENSELK - 0.5 * DENSEKI * DENSEKI);
                                                constant = (DENSEJI * DENSELK - 0.5 * devSim.hyb_coeff * DENSEKI * DENSEKI);
                                            }else if (JJJ == KKK && JJJ == LLL){
                                                //constant = DENSELJ * DENSEJI;
                                                constant = 2.0 * DENSELJ * DENSEJI - devSim.hyb_coeff * DENSELJ * DENSEJI;
                                            }else if (KKK == LLL && III < JJJ && JJJ != KKK){
                                                //constant = (2.0* DENSEJI * DENSELK - DENSEKI * DENSEKJ);
                                                constant = (2.0* DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSEKJ);
                                            }else if ( III == JJJ && KKK < LLL){
                                                //constant = (2.0* DENSELK * DENSEJI - DENSEKI * DENSELI);
                                                constant = (2.0* DENSELK * DENSEJI - devSim.hyb_coeff * DENSEKI * DENSELI);
                                            }
                                        }
                                        else{
                                            if (JJJ <= LLL) {
                                                if (III == JJJ && III == KKK && III == LLL) {
                                                    // Do nothing
                                                }else if (III==JJJ && III==KKK && III < LLL){
                                                    //constant = DENSELI * DENSEJI;
						    constant = 2.0 * DENSELI * DENSEJI - devSim.hyb_coeff * DENSELI * DENSEJI;
                                                }else if (III==KKK && JJJ==LLL && III < JJJ){
                                                    //constant = (1.5 * DENSEJI * DENSEJI - 0.5 * DENSELJ * DENSEKI);
                                                    constant = (2.0 * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSELJ * DENSEKI);

                                                }else if (III== KKK && III < JJJ && JJJ < LLL){
                                                    //constant = (3.0 * DENSEJI * DENSELI - DENSELJ * DENSEKI);
                                                    constant = (4.0 * DENSEJI * DENSELI - devSim.hyb_coeff * DENSEJI * DENSELI - devSim.hyb_coeff * DENSELJ * DENSEKI);
                                                }
                                            }
                                        }
                                    }
                                    
                                    AGradx += constant * Yaax;
                                    AGrady += constant * Yaay;
                                    AGradz += constant * Yaaz;
                                    
                                    BGradx += constant * Ybbx;
                                    BGrady += constant * Ybby;
                                    BGradz += constant * Ybbz;
                                    
                                    CGradx += constant * Yccx;
                                    CGrady += constant * Yccy;
                                    CGradz += constant * Yccz;
                                    
                                    
                                    
                                }
                            }
                        }
                    }
                }
                
                
                
                /*
                if ( abs(AGradx) > 0 || abs(AGrady) > 0 || abs(AGradz) > 0 ||
                    abs(BGradx) > 0 || abs(BGrady) > 0 || abs(BGradz) > 0 ||
                    abs(CGradx) > 0 || abs(CGrady) > 0 || abs(CGradz) > 0) {
                    
                    printf("%i %i %i %i %i %i %i %i %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e \n", II, JJ, KK, LL, \
                           I, J, K, L, AGradx, AGrady, AGradz, BGradx, BGrady, BGradz, CGradx, CGrady, CGradz);
                }*/
                
                
                
            }
        }
    }
    
    
#ifdef DEBUG
    //printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim.hyb_coeff \n", __FILE__, __LINE__, __func__);
#endif    
    
    GRADADD(devSim.gradULL[AStart], AGradx);
    GRADADD(devSim.gradULL[AStart + 1], AGrady);
    GRADADD(devSim.gradULL[AStart + 2], AGradz);
    
    
    GRADADD(devSim.gradULL[BStart], BGradx);
    GRADADD(devSim.gradULL[BStart + 1], BGrady);
    GRADADD(devSim.gradULL[BStart + 2], BGradz);
    
    
    GRADADD(devSim.gradULL[CStart], CGradx);
    GRADADD(devSim.gradULL[CStart + 1], CGrady);
    GRADADD(devSim.gradULL[CStart + 2], CGradz);
    
    
    GRADADD(devSim.gradULL[DStart], (-AGradx-BGradx-CGradx));
    GRADADD(devSim.gradULL[DStart + 1], (-AGrady-BGrady-CGrady));
    GRADADD(devSim.gradULL[DStart + 2], (-AGradz-BGradz-CGradz));
    
    return;
}

#endif


#ifndef new_quick_2_gpu_get2e_subs_grad_h
#define new_quick_2_gpu_get2e_subs_grad_h


#undef STOREDIM
#define STOREDIM STOREDIM_S

__device__ __forceinline__ void hrrwholegrad(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             int I, int J, int K, int L, \
                                             int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                             QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                             QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                             QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                             QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                             QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOC2(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOC2(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOC2(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOC2(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOC2(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOC2(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
        
    }
    
    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOC2(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccx = *Yccx - LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    // Part C - y
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOC2(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccy = *Yccy - LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOC2(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccz = *Yccz - LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}

#undef STOREDIM
#define STOREDIM STOREDIM_L

__device__ __forceinline__ void hrrwholegrad2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             int I, int J, int K, int L, \
                                             int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                             QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                             QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                             QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                             QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                             QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
        
    }
    
    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                    
                }
            }
        }
    }
    
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    if (LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccx = *Yccx - LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    // Part C - y
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    if (LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccy = *Yccy - LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccz = *Yccz - LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}


__device__ __forceinline__ void hrrwholegrad2_1(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    // Part C - y
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * CC;
            }
        }
    }
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}


__device__ __forceinline__ void hrrwholegrad2_2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble AA, QUICKDouble BB, QUICKDouble CC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * AA;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM) * 2 * BB;
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}

#endif
