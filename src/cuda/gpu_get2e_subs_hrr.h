//
//  gpu_get2e_subs_hrr.h
//  new_quick
//
//  Created by Yipu Miao on 3/18/14.
//
//

#ifndef new_quick_2_gpu_get2e_subs_hrr_h
#define new_quick_2_gpu_get2e_subs_hrr_h


#undef STOREDIM
#define STOREDIM STOREDIM_S

__device__ __forceinline__ QUICKDouble hrrwhole(int I, int J, int K, int L, \
                                                int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y;
    
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    Y = (QUICKDouble) 0.0;
    
    int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
    int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
    }
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    //#endif
    return Y;
}

#undef STOREDIM
#define STOREDIM STOREDIM_L

__device__ __forceinline__ QUICKDouble hrrwhole2(int I, int J, int K, int L, \
                                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y;
    
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    Y = (QUICKDouble) 0.0;
    
    int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
    int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM ) {
                Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
    }
    
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    //#endif
    return Y;
}


__device__ __forceinline__ QUICKDouble hrrwhole2_1(int I, int J, int K, int L, \
                                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y;
    
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    Y = (QUICKDouble) 0.0;
    
    int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
    
    if (L == 2) {
        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM ) {
                Y += coefAngularL[i] * LOC2(store, angularL[i]-1,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                                            STOREDIM, STOREDIM);
            }
        }
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
    }
    
    int numAngularR = lefthrr23(RCx, RCy, RCz, RDx, RDy, RDz,
                                LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                                LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                                L, coefAngularR, angularR);
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM ) {
                Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
    }
    
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    //#endif
    return Y;
}


__device__ __forceinline__ QUICKDouble hrrwhole2_2(int I, int J, int K, int L, \
                                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y;
    
    int angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    Y = (QUICKDouble) 0.0;
    
    int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
    
    if (J == 2) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularR[j] <= STOREDIM ) {
                Y += coefAngularR[j] * LOC2(store,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1, \
                                            angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
        
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        
        return Y;
    }
    
    
    int numAngularL = lefthrr23(RAx, RAy, RAz, RBx, RBy, RBz,
                                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                                J, coefAngularL, angularL);
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM ) {
                Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
    }
    
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    //#endif
    return Y;
}



__device__ __forceinline__ QUICKDouble hrrwhole2_5(int I, int J, int K, int L, \
                                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    /*
     When this subroutine is called, (ij|kl) where i+j = 4 and k+l = 6 is computed, but (i+j) >=4 and k+l = 6 entering this subroutine
     therefore, k = 3 and l = 3 is confirmed.
     */
    
    if ((K+L)== 6 && (I+J)==4) // k+l = 6, and i+j = 4
    {
        
        QUICKDouble Y = LOC2(store,  \
                             (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                             (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                             STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
        
    }
    
    
    // else case, j can be 2 or 3, and k = 3 and l = 3
    int angularL[12];
    QUICKDouble coefAngularL[12];
    QUICKDouble Y = 0.0;
    
    int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
    
    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM ) {
            Y += coefAngularL[i] * LOC2(store, angularL[i]-1,
                                        (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                                   LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                                   LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                                        STOREDIM, STOREDIM);
        }
    }
    
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    return Y;
    
}

__device__ __forceinline__ QUICKDouble hrrwhole2_6(int I, int J, int K, int L, \
                                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    /*
     When this subroutine is called, (ij|kl) where i+j = 6 and k+l = 4 is computed, but (i+j) ==6 and k+l >= 4 entering this subroutine
     therefore, i = 3 and j = 3 is confirmed.
     */
    
    if ((K+L)== 4 && (I+J)==6) // k+l = 4, and i+j = 6
    {
        
        QUICKDouble Y = LOC2(store,  \
                             (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                             (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                        LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                             STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
        
    }
    
    int angularR[12];
    QUICKDouble coefAngularR[12];
    QUICKDouble Y = 0.0;
    
    // For hrr, only k+l need hrr, but can be simplified to only consider k+l=5 contibution
    int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
    
    for (int j = 0; j<numAngularR; j++) {
        if (angularR[j] <= STOREDIM ) {
            Y += coefAngularR[j] * LOC2(store,
                                        (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                                   LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                                   LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1, \
                                        angularR[j]-1 , STOREDIM, STOREDIM);
        }
    }
    
    
    Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    
    return Y;
    
}





__device__ __forceinline__ QUICKDouble hrrwhole2_3(int I, int J, int K, int L, \
                                                 int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                 QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                 QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                 QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                 QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    /*
     when this subroutine is called, only (ij|kl) k+l = 5 and i+j = 5 is computed, but (k+l)>=5 and (i+J)>=5 is entering this subroutine
     */
    
    if ((K+L)== 5 && (I+J)==5) // k+l = 5, and i+j = 5
    {
        
        QUICKDouble Y = LOC2(store,  \
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                 STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
        
    }
    
    if ((K+L)== 5 && (I+J)==6) // k+l = 5, and i = 3, j = 3
    {
        
        int angularL[12];
        QUICKDouble coefAngularL[12];
        QUICKDouble Y = 0.0;
        
        int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                                  LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                                  3, coefAngularL, angularL);
        
        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM ) {
                Y += coefAngularL[i] * LOC2(store, angularL[i]-1,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                                            STOREDIM, STOREDIM);
            }
        }
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
        
    }
    
    if ((I+J) == 5 && (K+L) == 6) {  // i+j = 5 and k=3 and l = 3
        int angularR[12];
        QUICKDouble coefAngularR[12];
        
        QUICKDouble Y = 0.0;
        int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                                  LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                                  3, coefAngularR, angularR);
        
        for (int j = 0; j<numAngularR; j++) {
            if (angularR[j] <= STOREDIM ) {
                Y += coefAngularR[j] * LOC2(store,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1, \
                                            angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
        
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
    }
    
    if ((I+J) == 6 && (K+L) == 6) { // i,j,k,l = 3
        
        int angularL[12], angularR[12];
        QUICKDouble coefAngularL[12], coefAngularR[12];
        QUICKDouble Y = (QUICKDouble) 0.0;
        
        int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                                  LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                                  3, coefAngularL, angularL);
        int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                                  LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                                  3, coefAngularR, angularR);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM ) {
                    Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
                }
            }
        }
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
    }
    
    return 0.0;
}



__device__ __forceinline__ QUICKDouble hrrwhole2_4(int I, int J, int K, int L, \
                                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    /*
     When this subroutine is called, only (ij|kl) k+l=5 and i+j=6 integral is computed, but (k+l)>=5 and i+j=6 is entering this subroutine
     since i+j = 6, i=3 and j= 3
     so if (k+l) = 5, then, the highest integral is used, and no selection hrr.
     if (k+l) = 6, then, k=3 and l=3
     */
    
    QUICKDouble Y = 0.0;
    
    if ((K+L)== 5) // k+l = 5, and i = 3 and j = 3
    {
        
        Y = LOC2(store,  \
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                 STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
        
    }else{ //k=3 and l = 3, for i and j , i = 3 and j = 3
        int angularR[12];
        QUICKDouble coefAngularR[12];
        
        
        // For hrr, only k+l need hrr, but can be simplified to only consider k+l=5 contibution
        int numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                                  LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                                  3, coefAngularR, angularR);
        
        for (int j = 0; j<numAngularR; j++) {
            if (angularR[j] <= STOREDIM ) {
                Y += coefAngularR[j] * LOC2(store,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1, \
                                            angularR[j]-1 , STOREDIM, STOREDIM);
            }
        }
        
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    }
    return Y;
}

// For this subroutine, the basic idea is the same with hrrwhole2_4, just swap i to k and j to l.

__device__ __forceinline__ QUICKDouble hrrwhole2_7(int I, int J, int K, int L, \
                                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    if ((I+J) == 5) {
        Y = LOC2(store,  \
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                 (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                            LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                 STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        return Y;
    }else{
        
        int angularL[12];
        QUICKDouble coefAngularL[12];
        
        int numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                                  LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                                  LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                                  3, coefAngularL, angularL);
        
        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM ) {
                Y += coefAngularL[i] * LOC2(store, angularL[i]-1,
                                            (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                                       LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                                            STOREDIM, STOREDIM);
            }
        }
        
        Y = Y * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
        
        return Y;
    }
    
}


// For hrrwhole2_8,9,10, the situation is much simple, i=3, j=3, k=3, l=3
__device__ __forceinline__ QUICKDouble hrrwhole2_8(int I, int J, int K, int L, \
                                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y = LOC2(store,  \
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                         STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    
    return Y;
}

__device__ __forceinline__ QUICKDouble hrrwhole2_9(int I, int J, int K, int L, \
                                                   int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                   QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                   QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                   QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                   QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y = LOC2(store,  \
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                         STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    
    return Y;
}

__device__ __forceinline__ QUICKDouble hrrwhole2_10(int I, int J, int K, int L, \
                                                    int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                                    QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                                    QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                                    QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                                    QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    QUICKDouble Y = LOC2(store,  \
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1,
                         (int) LOC3(devTrans, LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), \
                                    LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), TRANSDIM, TRANSDIM, TRANSDIM)-1 , \
                         STOREDIM, STOREDIM) * devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    
    return Y;
}


__device__ __forceinline__ int lefthrr1(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                        int KLMNAx, int KLMNAy, int KLMNAz,
                                        int KLMNBx, int KLMNBy, int KLMNBz,
                                        int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    int numAngularL = 2;
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    
    if (KLMNBx != 0) {
        coefAngularL[1] = RAx-RBx;
    }else if(KLMNBy !=0 ){
        coefAngularL[1] = RAy-RBy;
    }else if (KLMNBz != 0) {
        coefAngularL[1] = RAz-RBz;
    }
    angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    return numAngularL;
}

__device__ __forceinline__ int lefthrr2(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                        int KLMNAx, int KLMNAy, int KLMNAz,
                                        int KLMNBx, int KLMNBy, int KLMNBz,
                                        int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    int numAngularL;
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    
    
    if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
        numAngularL = 3;
        QUICKDouble tmp;
        
        
        if (KLMNBx == 2) {
            tmp = RAx - RBx;
            angularL[1] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if(KLMNBy == 2) {
            tmp = RAy - RBy;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBz == 2 ){
            tmp = RAz - RBz;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        coefAngularL[1] = 2 * tmp;
        coefAngularL[2]= tmp * tmp;
    }else{
        
        numAngularL = 4;
        QUICKDouble tmp, tmp2;
        
        if(KLMNBx == 1 && KLMNBy == 1){
            tmp = RAx - RBx;
            tmp2 = RAy - RBy;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
        }else if (KLMNBx == 1 && KLMNBz == 1) {
            tmp = RAx - RBx;
            tmp2 = RAz - RBz;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBy == 1 && KLMNBz == 1) {
            tmp = RAy - RBy;
            tmp2 = RAz - RBz;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        
        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp * tmp2;
        
    }
    
    
    angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    return numAngularL;
}



__device__ __forceinline__ int lefthrr3(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                        int KLMNAx, int KLMNAy, int KLMNAz,
                                        int KLMNBx, int KLMNBy, int KLMNBz,
                                        int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    int numAngularL;
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
        numAngularL = 4;
        QUICKDouble tmp;
        
        if (KLMNBx == 3) {
            tmp = RAx - RBx;
            angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBy == 3) {
            tmp = RAy - RBy;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBz == 3) {
            tmp = RAz - RBz;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        
        coefAngularL[1] = 3 * tmp;
        coefAngularL[2] = 3 * tmp * tmp;
        coefAngularL[3] = tmp * tmp * tmp;
    }else if (KLMNBx == 1 && KLMNBy == 1) {
        numAngularL = 8;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;
        
        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp3;
        coefAngularL[4] = tmp * tmp2;
        coefAngularL[5] = tmp * tmp3;
        coefAngularL[6] = tmp2 * tmp3;
        coefAngularL[7] = tmp * tmp2 * tmp3;
        
        angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = (int) LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
    }else{
        
        numAngularL = 6;
        QUICKDouble tmp;
        QUICKDouble tmp2;
        
        if (KLMNBx == 1) {
            tmp = RAx - RBx;
        }else if (KLMNBy == 1){
            tmp = RAy - RBy;
        }else if (KLMNBz == 1){
            tmp = RAz - RBz;
        }
        
        if (KLMNBx == 2) {
            tmp2 = RAx - RBx;
        }else if (KLMNBy == 2){
            tmp2 = RAy - RBy;
        }else if (KLMNBz == 2){
            tmp2 = RAz - RBz;
        }
        
        coefAngularL[1] = tmp;
        coefAngularL[2] = 2 * tmp2;
        coefAngularL[3] = 2 * tmp * tmp2;
        coefAngularL[4] = tmp2 * tmp2;
        coefAngularL[5] = tmp * tmp2 * tmp2;
        
        
        if (KLMNBx == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBy == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBz == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBx == 1) {
            if (KLMNBy == 2) {  //120
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              //102
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        if (KLMNBy == 1) {
            if (KLMNBx == 2) {  // 210
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              // 012
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        if (KLMNBz == 1) {
            if (KLMNBx == 2) {  // 201
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              // 021
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        
        if (KLMNBx == 1) {
            angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBy == 1) {
            angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBz == 1) {
            angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
    }
    
    angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    return numAngularL;
    
}


__device__ __forceinline__ QUICKDouble quick_pow(QUICKDouble a, int power)
{
    /* 
        notice 0^0 = 1 for this subroutine but is invalid mathmatically
     */
    if (power == 0) {
        return 1;
    }
    if (power == 1) {
        return a;
    }
    if (power == 2) {
        return a*a;
    }
    
    if (power ==3) {
        return a*a*a;
    }
    return 0;
}



__device__ __forceinline__ int lefthrr_r(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                       QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                       int KLMNAx, int KLMNAy, int KLMNAz,
                                       int KLMNBx, int KLMNBy, int KLMNBz,
                                       int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    if (IJTYPE == 0) {
        return 1;
    }
    
    QUICKDouble tmpx = RAx - RBx;
    QUICKDouble tmpy = RAy - RBy;
    QUICKDouble tmpz = RAz - RBz;
    
    QUICKDouble tmpx2 = 1.0;
    QUICKDouble tmpy2 = 1.0;
    QUICKDouble tmpz2 = 1.0;
    
    if (KLMNBx > 0) {
        tmpx2 = quick_pow(tmpx, KLMNBx);
    }
    
    if (KLMNBy > 0) {
        tmpy2 = quick_pow(tmpy, KLMNBy);
    }
    
    if (KLMNBz > 0) {
        tmpz2 = quick_pow(tmpz, KLMNBz);
    }
    
    angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    coefAngularL[1] = tmpx2 * tmpy2 * tmpz2;
    
    if (IJTYPE == 1) {
        return 2;
    }
    
    int numAngularL = 2;
    
    if (KLMNBx >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = KLMNBx * quick_pow(tmpx, KLMNBx-1) * tmpy2 * tmpz2;
        numAngularL++;
    }
    
    if (KLMNBy >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = KLMNBy * quick_pow(tmpy, KLMNBy-1) * tmpx2 * tmpz2;
        numAngularL++;
    }
    if (KLMNBz >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = KLMNBz * quick_pow(tmpz, KLMNBz-1) * tmpx2 * tmpy2;
        numAngularL++;
    }
    
    
    if (IJTYPE == 2) {
        return numAngularL;
    }
    
    if (KLMNBx >= 2) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * (KLMNBx - 1) / 2 * quick_pow(tmpx, KLMNBx-2) * tmpy2 * tmpz2;
        numAngularL++;
    }
    
    
    if (KLMNBy >= 2) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * (KLMNBy - 1) / 2 * quick_pow(tmpy, KLMNBy-2) * tmpx2 * tmpz2;
        numAngularL++;
    }
    
    if (KLMNBz >= 2) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBz * (KLMNBz - 1) / 2 * quick_pow(tmpz, KLMNBz-2) * tmpx2 * tmpy2;
        numAngularL++;
    }
    
    if (KLMNBx >= 1 && KLMNBy >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBy * quick_pow(tmpx, KLMNBx-1) * quick_pow(tmpy, KLMNBy-1) * tmpz2;
        numAngularL++;
    }
    
    if (KLMNBx >= 1 && KLMNBz >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBz * quick_pow(tmpx, KLMNBx-1) * quick_pow(tmpz, KLMNBz-1) * tmpy2;
        numAngularL++;
    }
    
    
    if (KLMNBy >= 1 && KLMNBz >= 1) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * KLMNBz * quick_pow(tmpy, KLMNBy-1) * quick_pow(tmpz, KLMNBz-1) * tmpx2;
        numAngularL++;
    }
    
    /* the last case is IJTYPE = 3 */
    return numAngularL;
    
}

__device__ __forceinline__ int lefthrr(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                       QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                       int KLMNAx, int KLMNAy, int KLMNAz,
                                       int KLMNBx, int KLMNBy, int KLMNBz,
                                       int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    
    
    int numAngularL;
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    if (IJTYPE == 0) {
        numAngularL = 1;
        return numAngularL;
    }
    else if (IJTYPE == 1)
    {
        numAngularL = 2;
        
        if (KLMNBx != 0) {
            coefAngularL[1] = RAx-RBx;
        }else if(KLMNBy !=0 ){
            coefAngularL[1] = RAy-RBy;
        }else if (KLMNBz != 0) {
            coefAngularL[1] = RAz-RBz;
        }
        
        angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        
        return numAngularL;
    }
    else if (IJTYPE == 2)
    {
        
        if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
            numAngularL = 3;
            QUICKDouble tmp;
            
            
            if (KLMNBx == 2) {
                tmp = RAx - RBx;
                angularL[1] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if(KLMNBy == 2) {
                tmp = RAy - RBy;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBz == 2 ){
                tmp = RAz - RBz;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            coefAngularL[1] = 2 * tmp;
            coefAngularL[2]= tmp * tmp;
            
            
            angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            return numAngularL;
            
        }else{
            
            numAngularL = 4;
            QUICKDouble tmp, tmp2;
            
            if(KLMNBx == 1 && KLMNBy == 1){
                tmp = RAx - RBx;
                tmp2 = RAy - RBy;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                
            }else if (KLMNBx == 1 && KLMNBz == 1) {
                tmp = RAx - RBx;
                tmp2 = RAz - RBz;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBy == 1 && KLMNBz == 1) {
                tmp = RAy - RBy;
                tmp2 = RAz - RBz;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            
            coefAngularL[1] = tmp;
            coefAngularL[2] = tmp2;
            coefAngularL[3] = tmp * tmp2;
            
            
            angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            return numAngularL;
        }
    }
    else if (IJTYPE == 3)
    {
        if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
            numAngularL = 4;
            QUICKDouble tmp;
            
            if (KLMNBx == 3) {
                tmp = RAx - RBx;
                angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBy == 3) {
                tmp = RAy - RBy;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBz == 3) {
                tmp = RAz - RBz;
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            
            coefAngularL[1] = 3 * tmp;
            coefAngularL[2] = 3 * tmp * tmp;
            coefAngularL[3] = tmp * tmp * tmp;
            
            
            angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            return numAngularL;
        }else if (KLMNBx == 1 && KLMNBy == 1) {
            numAngularL = 8;
            QUICKDouble tmp = RAx - RBx;
            QUICKDouble tmp2 = RAy - RBy;
            QUICKDouble tmp3 = RAz - RBz;
            
            coefAngularL[1] = tmp;
            coefAngularL[2] = tmp2;
            coefAngularL[3] = tmp3;
            coefAngularL[4] = tmp * tmp2;
            coefAngularL[5] = tmp * tmp3;
            coefAngularL[6] = tmp2 * tmp3;
            coefAngularL[7] = tmp * tmp2 * tmp3;
            
            angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = (int) LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            
            angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            return numAngularL;
        }else{
            
            numAngularL = 6;
            QUICKDouble tmp;
            QUICKDouble tmp2;
            
            if (KLMNBx == 1) {
                tmp = RAx - RBx;
            }else if (KLMNBy == 1){
                tmp = RAy - RBy;
            }else if (KLMNBz == 1){
                tmp = RAz - RBz;
            }
            
            if (KLMNBx == 2) {
                tmp2 = RAx - RBx;
            }else if (KLMNBy == 2){
                tmp2 = RAy - RBy;
            }else if (KLMNBz == 2){
                tmp2 = RAz - RBz;
            }
            
            coefAngularL[1] = tmp;
            coefAngularL[2] = 2 * tmp2;
            coefAngularL[3] = 2 * tmp * tmp2;
            coefAngularL[4] = tmp2 * tmp2;
            coefAngularL[5] = tmp * tmp2 * tmp2;
            
            
            if (KLMNBx == 2) {
                angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            if (KLMNBy == 2) {
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            if (KLMNBz == 2) {
                angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            if (KLMNBx == 1) {
                if (KLMNBy == 2) {  //120
                    angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                }else{              //102
                    angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }
            
            if (KLMNBy == 1) {
                if (KLMNBx == 2) {  // 210
                    angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                }else{              // 012
                    angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }
            
            if (KLMNBz == 1) {
                if (KLMNBx == 2) {  // 201
                    angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }else{              // 021
                    angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }
            
            
            if (KLMNBx == 1) {
                angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            if (KLMNBy == 1) {
                angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            if (KLMNBz == 1) {
                angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            
            angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            return numAngularL;
            
            
            
        }
    }    
    
    /*    else if (IJTYPE == 4)
     {
     if (KLMNBx == 4) {
     numAngularL = 5;
     QUICKDouble tmp = RAx - RBx;
     
     coefAngularL[1] = 4 * tmp;
     coefAngularL[2] = 6 * tmp * tmp;
     coefAngularL[3] = 4 * tmp * tmp * tmp;
     coefAngularL[4] = tmp * tmp * tmp * tmp;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx+3, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBy == 4) {
     numAngularL = 5;
     QUICKDouble tmp = RAy - RBy;
     coefAngularL[1] = 4 * tmp;
     coefAngularL[2] = 6 * tmp * tmp;
     coefAngularL[3] = 4 * tmp * tmp * tmp;
     coefAngularL[4] = tmp * tmp * tmp * tmp;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBz == 4) {
     numAngularL = 5;
     
     QUICKDouble tmp = RAz - RBz;
     coefAngularL[1] = 4 * tmp;
     coefAngularL[2] = 6 * tmp * tmp;
     coefAngularL[3] = 4 * tmp * tmp * tmp;
     coefAngularL[4] = tmp * tmp * tmp * tmp;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBx == 1 && KLMNBy == 3) {
     numAngularL = 8;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAy - RBy;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBx == 3 && KLMNBy == 1) {
     numAngularL = 8;
     QUICKDouble tmp = RAy - RBy;
     QUICKDouble tmp2 = RAx - RBx;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx+3, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     }
     
     else if (KLMNBx == 1 && KLMNBz ==3) {
     numAngularL = 8;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAz - RBz;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBx == 3 && KLMNBz == 1) {
     numAngularL = 8;
     QUICKDouble tmp = RAz - RBz;
     QUICKDouble tmp2 = RAx - RBx;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx+3, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBy == 1 && KLMNBz == 3) {
     numAngularL = 8;
     QUICKDouble tmp = RAy - RBy;
     QUICKDouble tmp2 = RAz - RBz;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBy == 3 && KLMNBz == 1) {
     numAngularL = 8;
     QUICKDouble tmp = RAz - RBz;
     QUICKDouble tmp2 = RAy - RBy;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = 3 * tmp2;
     coefAngularL[3] = 3 * tmp * tmp2;
     coefAngularL[4] = 3 * tmp2 * tmp2;
     coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
     coefAngularL[6] = tmp2 * tmp2 * tmp2;
     coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+3, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     
     }else if (KLMNBx == 2 && KLMNBy == 2) {
     numAngularL = 9;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAy - RBy;
     
     coefAngularL[1] = 2 * tmp;
     coefAngularL[2] = 2 * tmp2;
     coefAngularL[3] = 4 * tmp * tmp2;
     coefAngularL[4] = tmp * tmp;
     coefAngularL[5] = tmp2 * tmp2;
     coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
     coefAngularL[7] = 2 * tmp * tmp * tmp2;
     coefAngularL[8] = tmp * tmp * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBx == 2 && KLMNBz == 2) {
     numAngularL = 9;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAz - RBz;
     
     coefAngularL[1] = 2 * tmp;
     coefAngularL[2] = 2 * tmp2;
     coefAngularL[3] = 4 * tmp * tmp2;
     coefAngularL[4] = tmp * tmp;
     coefAngularL[5] = tmp2 * tmp2;
     coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
     coefAngularL[7] = 2 * tmp * tmp * tmp2;
     coefAngularL[8] = tmp * tmp * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int) LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBy == 2 && KLMNBz == 2) {
     numAngularL = 9;
     QUICKDouble tmp = RAy - RBy;
     QUICKDouble tmp2 = RAz - RBz;
     
     coefAngularL[1] = 2 * tmp;
     coefAngularL[2] = 2 * tmp2;
     coefAngularL[3] = 4 * tmp * tmp2;
     coefAngularL[4] = tmp * tmp;
     coefAngularL[5] = tmp2 * tmp2;
     coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
     coefAngularL[7] = 2 * tmp * tmp * tmp2;
     coefAngularL[8] = tmp * tmp * tmp2 * tmp2;
     
     angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBx == 1 && KLMNBy == 1 && KLMNBz == 2) {
     numAngularL = 12;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAy - RBy;
     QUICKDouble tmp3 = RAz - RBz;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = tmp2;
     coefAngularL[3] = 2 * tmp3;
     coefAngularL[4] = tmp * tmp2;
     coefAngularL[5] = 2 * tmp * tmp3;
     coefAngularL[6] = 2 * tmp2 * tmp3;
     coefAngularL[7] = tmp3 * tmp3;
     coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
     coefAngularL[9] = tmp * tmp3 * tmp3;
     coefAngularL[10] = tmp2 * tmp3 * tmp3;
     coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;
     
     angularL[1] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[8] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[9] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[10] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBx == 1 && KLMNBz == 1 && KLMNBy == 2) {
     numAngularL = 12;
     QUICKDouble tmp = RAx - RBx;
     QUICKDouble tmp2 = RAz - RBz;
     QUICKDouble tmp3 = RAy - RBy;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = tmp2;
     coefAngularL[3] = 2 * tmp3;
     coefAngularL[4] = tmp * tmp2;
     coefAngularL[5] = 2 * tmp * tmp3;
     coefAngularL[6] = 2 * tmp2 * tmp3;
     coefAngularL[7] = tmp3 * tmp3;
     coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
     coefAngularL[9] = tmp * tmp3 * tmp3;
     coefAngularL[10] = tmp2 * tmp3 * tmp3;
     coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;
     
     angularL[1] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[8] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[9] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[10] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     }else if (KLMNBy == 1 && KLMNBz == 1 && KLMNBx == 2) {
     numAngularL = 12;
     QUICKDouble tmp = RAy - RBy;
     QUICKDouble tmp2 = RAz - RBz;
     QUICKDouble tmp3 = RAx - RBx;
     
     coefAngularL[1] = tmp;
     coefAngularL[2] = tmp2;
     coefAngularL[3] = 2 * tmp3;
     coefAngularL[4] = tmp * tmp2;
     coefAngularL[5] = 2 * tmp * tmp3;
     coefAngularL[6] = 2 * tmp2 * tmp3;
     coefAngularL[7] = tmp3 * tmp3;
     coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
     coefAngularL[9] = tmp * tmp3 * tmp3;
     coefAngularL[10] = tmp2 * tmp3 * tmp3;
     coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;
     
     angularL[1] = (int)  LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[2] = (int)  LOC3(devTrans, KLMNAx+2, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[3] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[4] = (int)  LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[5] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[6] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[7] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[8] = (int)  LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[9] = (int)  LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
     angularL[10] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
     }
     }
     
     */
    //angularL[numAngularL - 1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    return numAngularL;
}

__device__ __forceinline__ int lefthrr23(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                    QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                    int KLMNAx, int KLMNAy, int KLMNAz,
                                    int KLMNBx, int KLMNBy, int KLMNBz,
                                    int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    int numAngularL;
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    
    /*
     if this subroutine is called, (ij|kl) for (k+l)>=5 is computed, but (k+l)>=5 entering this subroutine
     here ijtype is the value of l
     */
    //if (IJTYPE == 2) // If l = 2, then k = 3
    //{
    //    return 1;
    //}
    //else if (IJTYPE == 3) // If l = 3, then k = 2 or 3, only k+l = 5 or 6 is valid, so only keep k+2 items
    //{
    if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
        numAngularL = 2;
        QUICKDouble tmp;
        
        if (KLMNBx == 3) {
            tmp = RAx - RBx;
            angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBy == 3) {
            tmp = RAy - RBy;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }else if (KLMNBz == 3) {
            tmp = RAz - RBz;
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        
        coefAngularL[1] = 3 * tmp;
        
        return numAngularL;
    }else if (KLMNBx == 1 && KLMNBy == 1) {
        numAngularL = 4;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;
        
        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp3;
        
        angularL[1] = (int) LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        
        
        return numAngularL;
    }else{
        
        numAngularL = 3;
        QUICKDouble tmp;
        QUICKDouble tmp2;
        
        if (KLMNBx == 1) {
            tmp = RAx - RBx;
        }else if (KLMNBy == 1){
            tmp = RAy - RBy;
        }else if (KLMNBz == 1){
            tmp = RAz - RBz;
        }
        
        if (KLMNBx == 2) {
            tmp2 = RAx - RBx;
        }else if (KLMNBy == 2){
            tmp2 = RAy - RBy;
        }else if (KLMNBz == 2){
            tmp2 = RAz - RBz;
        }
        
        coefAngularL[1] = tmp;
        coefAngularL[2] = 2 * tmp2;
        
        
        if (KLMNBx == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBy == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBz == 2) {
            angularL[1] = (int) LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        }
        
        if (KLMNBx == 1) {
            if (KLMNBy == 2) {  //120
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              //102
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        if (KLMNBy == 1) {
            if (KLMNBx == 2) {  // 210
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              // 012
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        if (KLMNBz == 1) {
            if (KLMNBx == 2) {  // 201
                angularL[2] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }else{              // 021
                angularL[2] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
        
        
        
        return numAngularL;
        
        
        
        //}
        
    }
    
    //return 0;
}


__device__ __forceinline__ int lefthrr23_new(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
                                         QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                                         int KLMNAx, int KLMNAy, int KLMNAz,
                                         int KLMNBx, int KLMNBy, int KLMNBz,
                                         int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{
    
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    
    /*
     if this subroutine is called, (ij|kl) for (k+l)>=5 is computed, but (k+l)>=5 entering this subroutine
     here ijtype is the value of l
     */
    //if (IJTYPE == 2) // If l = 2, then k = 3
    //{
    //    return 1;
    //}
    //else if (IJTYPE == 3) // If l = 3, then k = 2 or 3, only k+l = 5 or 6 is valid, so only keep k+2 items
    //{
    coefAngularL[0] = 1.0;
    angularL[0] = (int) LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
    
    QUICKDouble tmp4 = 1.0;
    
    QUICKDouble tmpx = RAx - RBx;
    QUICKDouble tmpy = RAy - RBy;
    QUICKDouble tmpz = RAz - RBz;
    
    if (KLMNBx > 0) {
        tmp4 = tmp4 * quick_pow(tmpx, KLMNBx);
    }
    
    if (KLMNBy > 0) {
        tmp4 = tmp4 * quick_pow(tmpy, KLMNBy);
    }
    
    if (KLMNBz > 0) {
        tmp4 = tmp4 * quick_pow(tmpz, KLMNBz);
    }
    
    
    int numAngularL = 1;
    
    if (KLMNBx >= 2 && tmpx != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * (KLMNBx - 1) / 2 * tmp4 / (tmpx * tmpx);
        numAngularL++;
    }
    
    
    if (KLMNBy >= 2 && tmpy != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * (KLMNBy - 1) / 2 * tmp4 / (tmpy * tmpy);
        numAngularL++;
    }
    
    if (KLMNBz >= 2 && tmpz != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBz * (KLMNBz - 1) / 2 * tmp4 / (tmpz * tmpz);
        numAngularL++;
    }
    
    if (KLMNBx >= 1 && KLMNBy >= 1 && tmpx != 0 && tmpy != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBy * tmp4 / (tmpx * tmpy);
        numAngularL++;
    }
    
    if (KLMNBx >= 1 && KLMNBz >= 1 && tmpx != 0 && tmpz != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBz * tmp4 / (tmpx * tmpz);
        numAngularL++;
    }
    
    
    if (KLMNBy >= 1 && KLMNBz >= 1 && tmpy != 0 && tmpz != 0) {
        angularL[numAngularL] = (int) LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * KLMNBz * tmp4 / (tmpy * tmpz);
        numAngularL++;
    }
    
    /* the last case is IJTYPE = 3 */
    return numAngularL;
    
    //return 0;
}

#endif
