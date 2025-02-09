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


#if !defined(__gpu_get2e_subs_grad_h_)
  #define __gpu_get2e_subs_grad_h_
  #undef STOREDIM
  #define STOREDIM STOREDIM_S
__device__ static inline void hrrwholegrad_lri(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        int J, int III, int JJJ,
        QUICKDouble * const store, QUICKDouble * const storeAA, QUICKDouble * const storeBB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz)
{
    unsigned char angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1];
    int numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(storeAA, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(storeAA, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1,
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(storeAA, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(storeBB, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(storeBB, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(storeBB, angularL[i]-1, 0, STOREDIM, STOREDIM);
        }
    }

    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}


  #undef STOREDIM
  #define STOREDIM STOREDIM_L
__device__ static inline void hrrwholegrad_lri2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        int J, int III, int JJJ,
        QUICKDouble * const store, QUICKDouble AA, QUICKDouble BB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz)
{
    unsigned char angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1];
    int numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1,
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                J - 1, coefAngularL, angularL);

        for (int i = 0; i<numAngularL; i++) {
            if (angularL[i] <= STOREDIM) {
                *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM);

            }
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}


__device__ static inline void hrrwholegrad_lri2_2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        int J, int III, int JJJ,
        QUICKDouble * const store, QUICKDouble AA, QUICKDouble BB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz)
{
    unsigned char angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1];
    int numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1,
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
            LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (int i = 0; i<numAngularL; i++) {
        if (angularL[i] <= STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(store, angularL[i]-1, 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}
#endif


#undef STOREDIM
#if defined(int_spd)
  #define STOREDIM STOREDIM_S
#else
  #define STOREDIM STOREDIM_L
#endif


/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
   */
#if defined(int_spd)
  #if defined(OSHELL)
__device__ static inline void iclass_oshell_lri_grad
  #else
__device__ static inline void iclass_lri_grad
  #endif
(int I, int J, unsigned int II, unsigned int JJ, int iatom, unsigned int totalatom,
 QUICKDouble * const YVerticalTemp, QUICKDouble * const store, QUICKDouble * const store2,
 QUICKDouble * const storeAA, QUICKDouble * const storeBB) {
    /*
       kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.

       RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
       which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
       And we don't need the coordinates now, so we will not retrieve the data now.
   */
    QUICKDouble RAx = LOC2(devSim.xyz, 0, devSim.katom[II] - 1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1, devSim.katom[II] - 1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2, devSim.katom[II] - 1, 3, devSim.natom);

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

    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;

    /*
       store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
       of GPU limitation, we can not do that now.

       See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
    */
    for (int i = Sumindex[0]; i < Sumindex[2]; i++) {
        for (int j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = Sumindex[0]; i < Sumindex[3]; i++) {
        for (int j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = Sumindex[0]; i < Sumindex[3]; i++) {
        for (int j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (int i = 0; i < kPrimI * kPrimJ; i++) {
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

        QUICKDouble AA = LOC2(devSim.gcexpo, III, devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ, devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);

        QUICKDouble AB = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
        */
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

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
        QUICKDouble CD = devSim.lri_zeta;
        QUICKDouble ABCD = 1.0 / (AB + CD);

        /*
           X2 is the multiplication of four indices normalized coeffecient
        */
        QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1.0 / devSim.lri_zeta)
            * devSim.lri_cc[iatom] * pow(devSim.lri_zeta / PI, 1.5);
        
        //printf("lngr grad itt, x0,xcoeff1,x2,xcoeff2,x44: %f %f %f %f %f %f \n", X0, X1, devSim.lri_zeta, devSim.lri_cc[iatom], (1/devSim.lri_zeta) * devSim.lri_cc[iatom] * pow(devSim.lri_zeta/PI, 1.5), X1 * X0 * (1/devSim.lri_zeta) * devSim.lri_cc[iatom] * pow(devSim.lri_zeta/PI, 1.5));

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

        FmT(I + J + 1, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

        for (int i = 0; i <= I + J + 1; i++) {
            //printf("FmT %f Fm*sqrt %f X2 %f \n", VY(0, 0, i), VY(0, 0, i)*sqrt(ABCD), X2/sqrt(ABCD));
            VY(0, 0, i) *= X2;
        }

        lri::vertical2(I, J + 1, 0, 1, YVerticalTemp, store2,
                Px - RAx, Py - RAy, Pz - RAz,
                (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                Qx - RCx, Qy - RCy, Qz - RCz,
                (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);

        for (int i = Sumindex[0]; i < Sumindex[2]; i++) {
            for (int j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                if (i < STOREDIM && j < STOREDIM) {
                    LOCSTORE(store, j, i , STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                    //printf("store2 %d %d %f \n", i, j, LOCSTORE(store2, j, i, STOREDIM, STOREDIM));
                }
            }
        }

        for (int i = Sumindex[0]; i < Sumindex[2]; i++) {
            for (int j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
                if (i < STOREDIM && j < STOREDIM) {
                    LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2.0;
                    LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2.0;
                    //printf("storeAA storeBB %d %d %f %f \n", j, i, LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM), LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM));
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

    int AStart = (devSim.katom[II] - 1) * 3;
    int BStart = (devSim.katom[JJ] - 1) * 3;
    int CStart = iatom * 3;

    QUICKDouble RBx, RBy, RBz;

    RBx = LOC2(devSim.xyz, 0, devSim.katom[JJ] - 1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1, devSim.katom[JJ] - 1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2, devSim.katom[JJ] - 1, 3, devSim.natom);

    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    int  nbasis = devSim.nbasis;

    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            QUICKDouble Yaax, Yaay, Yaaz;
            QUICKDouble Ybbx, Ybby, Ybbz;

            hrrwholegrad_lri(&Yaax, &Yaay, &Yaaz,
                    &Ybbx, &Ybby, &Ybbz,
                    J, III, JJJ,
                    store, storeAA, storeBB,
                    RAx, RAy, RAz, RBx, RBy, RBz);

  #if defined(OSHELL)
            QUICKDouble DENSEJI = (QUICKDouble) (LOC2(devSim.dense, JJJ - 1, III - 1, nbasis, nbasis)
                    + LOC2(devSim.denseb, JJJ - 1, III - 1, nbasis, nbasis));
  #else
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1, nbasis, nbasis);
  #endif

            QUICKDouble constant;
            if (III != JJJ) {
                constant = 2.0 * DENSEJI;
            } else {
                constant = DENSEJI;
            }

            //printf("iatom %d %d %d %d %d dmx: %f Y: %f %f %f %f %f %f %f %f %f \n",iatom, II, JJ, III, JJJ, constant, RAx, RBx, RCx, Yaax, Yaay, Yaaz, Ybbx, Ybby, Ybbz);

            AGradx += constant * Yaax;
            AGrady += constant * Yaay;
            AGradz += constant * Yaaz;

            BGradx += constant * Ybbx;
            BGrady += constant * Ybby;
            BGradz += constant * Ybbz;
        }
    }

#if defined(USE_LEGACY_ATOMICS)
    GPUATOMICADD(&devSim.gradULL[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[AStart + 2], AGradz, GRADSCALE);

    GPUATOMICADD(&devSim.gradULL[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[BStart + 2], BGradz, GRADSCALE);

    if (iatom < devSim.natom) {
        GPUATOMICADD(&devSim.gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&devSim.gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&devSim.gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    } else {
        CStart = (iatom - devSim.natom) * 3;
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    }    
#else
    atomicAdd(&devSim.grad[AStart], AGradx);
    atomicAdd(&devSim.grad[AStart + 1], AGrady);
    atomicAdd(&devSim.grad[AStart + 2], AGradz);

    atomicAdd(&devSim.grad[BStart], BGradx);
    atomicAdd(&devSim.grad[BStart + 1], BGrady);
    atomicAdd(&devSim.grad[BStart + 2], BGradz);

    if (iatom < devSim.natom) {
        atomicAdd(&devSim.grad[CStart], -AGradx - BGradx);
        atomicAdd(&devSim.grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&devSim.grad[CStart + 2], -AGradz - BGradz);
    } else {
        CStart = (iatom - devSim.natom) * 3;
        atomicAdd(&devSim.ptchg_grad[CStart], -AGradx - BGradx);
        atomicAdd(&devSim.ptchg_grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&devSim.ptchg_grad[CStart + 2], -AGradz - BGradz);
    }    
#endif
}


#else
/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
*/
  #if defined(OSHELL)
    #if defined(int_spdf2)
__device__ static inline void iclass_oshell_lri_grad_spdf2
    #endif
  #else
    #if defined(int_spdf2)
__device__ static inline void iclass_lri_grad_spdf2
    #endif
  #endif
(int I, int J, unsigned int II, unsigned int JJ, int iatom, unsigned int totalatom,
 QUICKDouble* YVerticalTemp, QUICKDouble* store, QUICKDouble* store2,
 QUICKDouble* storeAA, QUICKDouble* storeBB) {
    /*
       kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.


       RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
       which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
       And we don't need the coordinates now, so we will not retrieve the data now.
    */
    QUICKDouble RAx = LOC2(devSim.xyz, 0, devSim.katom[II] - 1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1, devSim.katom[II] - 1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2, devSim.katom[II] - 1, 3, devSim.natom);

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

    int kStartI = devSim.kstart[II] - 1;
    int kStartJ = devSim.kstart[JJ] - 1;

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;

    int AStart = (devSim.katom[II] - 1) * 3;
    int BStart = (devSim.katom[JJ] - 1) * 3;
    int CStart = iatom * 3;

    /*
       store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
       of GPU limitation, we can not do that now.

       See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
   */

    for (int i = 0; i < kPrimI * kPrimJ; i++) {
        int JJJ = (int) i / kPrimI;
        int III = (int) i - kPrimI * JJJ;
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

        QUICKDouble AA = LOC2(devSim.gcexpo, III, devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ, devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);

        QUICKDouble AB = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
        */
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        for (int j = 0; j < kPrimK * kPrimL; j++) {
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
            QUICKDouble CD = devSim.lri_zeta;
            QUICKDouble ABCD = 1.0 / (AB + CD);

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

            FmT(I + J + 2, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

            for (int i = 0; i <= I + J + 2; i++) {
                VY(0, 0, i) *= X2;
            }

            for (int i = Sumindex[0]; i < Sumindex[3]; i++) {
                for (int j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
                    if (i < STOREDIM && j < STOREDIM && !(i >= Sumindex[I + J + 2] && j >= Sumindex[2])) {
                        LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
                    }
                }
            }

  #if defined(int_spdf2)
            lri::vertical2_spdf2(I, J + 1, 0, 1, YVerticalTemp, store2,
                    Px - RAx, Py - RAy, Pz - RAz,
                    (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                    Qx - RCx, Qy - RCy, Qz - RCz,
                    (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
  #endif

            QUICKDouble RBx, RBy, RBz;

            RBx = LOC2(devSim.xyz, 0, devSim.katom[JJ] - 1, 3, devSim.natom);
            RBy = LOC2(devSim.xyz, 1, devSim.katom[JJ] - 1, 3, devSim.natom);
            RBz = LOC2(devSim.xyz, 2, devSim.katom[JJ] - 1, 3, devSim.natom);

            int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
            int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
            int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
            int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

            int  nbasis = devSim.nbasis;

            for (int III = III1; III <= III2; III++) {
                for (int JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
                    QUICKDouble Yaax, Yaay, Yaaz;
                    QUICKDouble Ybbx, Ybby, Ybbz;

  #if defined(int_spdf2)
                    hrrwholegrad_lri2_2
  #else
                    hrrwholegrad_lri2
  #endif
                    (&Yaax, &Yaay, &Yaaz,
                     &Ybbx, &Ybby, &Ybbz,
                     J, III, JJJ,
                     store2, AA, BB,
                     RAx, RAy, RAz, RBx, RBy, RBz);

  #if defined(OSHELL)
                    QUICKDouble DENSEJI = (QUICKDouble) (LOC2(devSim.dense, JJJ - 1, III - 1, nbasis, nbasis)
                            + LOC2(devSim.denseb, JJJ - 1, III - 1, nbasis, nbasis));
  #else
                    QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ - 1, III - 1, nbasis, nbasis);
  #endif

                    QUICKDouble constant;
                    if (III != JJJ) {
                        constant = 2.0 * DENSEJI;
                    } else {
                        constant = DENSEJI;
                    }

                    AGradx += constant * Yaax;
                    AGrady += constant * Yaay;
                    AGradz += constant * Yaaz;

                    BGradx += constant * Ybbx;
                    BGrady += constant * Ybby;
                    BGradz += constant * Ybbz;
                }
            }
        }
    }

  #if defined(DEBUG)
    //printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim.hyb_coeff \n", __FILE__, __LINE__, __func__);
  #endif

  #if defined(USE_LEGACY_ATOMICS)
    GPUATOMICADD(&devSim.gradULL[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[AStart + 2], AGradz, GRADSCALE);

    GPUATOMICADD(&devSim.gradULL[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&devSim.gradULL[BStart + 2], BGradz, GRADSCALE);

    if (iatom < devSim.natom) {
        GPUATOMICADD(&devSim.gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&devSim.gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&devSim.gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    } else {
        CStart = (iatom - devSim.natom) * 3;
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&devSim.ptchg_gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    }
  #else
    atomicAdd(&devSim.grad[AStart], AGradx);
    atomicAdd(&devSim.grad[AStart + 1], AGrady);
    atomicAdd(&devSim.grad[AStart + 2], AGradz);

    atomicAdd(&devSim.grad[BStart], BGradx);
    atomicAdd(&devSim.grad[BStart + 1], BGrady);
    atomicAdd(&devSim.grad[BStart + 2], BGradz);

    if (iatom < devSim.natom) {
        atomicAdd(&devSim.grad[CStart], -AGradx - BGradx);
        atomicAdd(&devSim.grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&devSim.grad[CStart + 2], -AGradz - BGradz);
    } else {
        CStart = (iatom - devSim.natom) * 3;
        atomicAdd(&devSim.ptchg_grad[CStart], -AGradx - BGradx);
        atomicAdd(&devSim.ptchg_grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&devSim.ptchg_grad[CStart + 2], -AGradz - BGradz);
    }
  #endif
}
#endif


#if defined(int_spd)
__global__ void
//__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) get_lri_grad_kernel()
get_lri_grad_kernel()
#elif defined(int_spdf2)
__global__ void
//__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) get_lri_grad_kernel_spdf2()
get_lri_grad_kernel_spdf2()
#endif
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    unsigned int totalatom = devSim.natom + devSim.nextatom;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;

    for (QUICKULL i = offset; i < totalatom * jshell; i += totalThreads) {
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);

#if defined(MPIV_GPU)
        if (devSim.mpi_bcompute[b] > 0) {
#endif
        int II = devSim.sorted_YCutoffIJ[b].x;
        int JJ = devSim.sorted_YCutoffIJ[b].y;

        int ii = devSim.sorted_Q[II];
        int jj = devSim.sorted_Q[JJ];

        int iii = devSim.sorted_Qnumber[II];
        int jjj = devSim.sorted_Qnumber[JJ];

#if defined(int_spd)
        iclass_lri_grad(iii, jjj, ii, jj, iatom, totalatom,
                devSim.YVerticalTemp + offset, devSim.store + offset,
                devSim.store2 + offset, devSim.storeAA + offset,
                devSim.storeBB + offset);
#elif defined(int_spdf2)
        if (iii + jjj >= 4) {
            iclass_lri_grad_spdf2(iii, jjj, ii, jj, iatom, totalatom,
                    devSim.YVerticalTemp + offset, devSim.store + offset,
                    devSim.store2 + offset, devSim.storeAA + offset,
                    devSim.storeBB + offset);
        }
#endif
#if defined(MPIV_GPU)
        }
#endif
    }
}
