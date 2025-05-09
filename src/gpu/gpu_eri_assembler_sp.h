/*
   !---------------------------------------------------------------------!
   ! Written by QUICK-GenInt code generator on 03/27/2023                !
   !                                                                     !
   ! Copyright (C) 2023-2024 Merz lab                                    !
   ! Copyright (C) 2023-2024 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!
   */

#undef STOREDIM
#undef VDIM3
#undef VY
#undef LOCSTORE
#define STOREDIM STOREDIM_T
#define VDIM3 VDIM3_T
#define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) LOCVY(YVerticalTemp, (a), (b), (c), VDIM1, VDIM2, VDIM3)


__device__ static inline void ERint_vertical_sp(const int I, const int J, const int K, const int L,
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz,
        const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz,
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz,
        const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz,
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp,
        const QUICKDouble ABcom, const QUICKDouble CDcom,
        QUICKDouble * const store, QUICKDouble * const YVerticalTemp) {
#include "iclass_ssss.h"
    if ((I + J) >= 0 && (K + L) >= 1) {
        if (K <= 1 && I <= 0) {
#include "iclass_ssps.h"
        }
        if ((I + J) >= 0 && (K + L) >= 2) {
            if (K <= 2 && I <= 0)
            {
#include "iclass_ssds.h"
            }
        }
    }
    if ((I + J) >= 1 && (K + L) >= 0) {
        if (I <= 1) {
#include "iclass_psss.h"
        }
        if ((I + J) >= 1 && (K + L) >= 1) {
            if (K <= 1 && I <= 1) {
#include "iclass_psps.h"
            }
            if ((I + J) >= 1 && (K + L) >= 2) {
                if (K <= 2 && I <= 1) {
#include "iclass_psds.h"
                }
            }
            if ((I + J) >= 2 && (K + L) >= 1) {
                if (K <= 1 && I <= 2) {
#include "iclass_dsps.h"
                }
                if ((I + J) >= 2 && (K + L) >= 2) {
                    if (K <= 2 && I <= 2) {
#include "iclass_dsds.h"
                    }
                }
            }
        }
        if ((I + J) >= 2 && (K + L) >= 0) {
            if (K <= 0 && I <= 2) {
#include "iclass_dsss.h"
            }
        }
    }
}
