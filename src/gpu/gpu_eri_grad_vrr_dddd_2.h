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
#undef STORE_OPERATOR
#define STOREDIM STOREDIM_GRAD_S
#define VDIM3 VDIM3_L
#define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) LOCVY(YVerticalTemp, (a), (b), (c), VDIM1, VDIM2, VDIM3)
#define STORE_OPERATOR =

__device__ __inline__ void ERint_grad_vertical_dddd_2(const int I, const int J, const int K, const int L,
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz,
        const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz,
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz,
        const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz,
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp,
        const QUICKDouble ABcom, const QUICKDouble CDcom,
        QUICKDouble * const store, QUICKDouble * const YVerticalTemp)
{
    if ((I + J) >= 5 && (K + L) >= 0)
    {
#include "iclass_hsss.h"
    }
    if ((I + J) >= 5 && (K + L) >= 1)
    {
#include "iclass_hsps.h"
    }
    if ((I + J) >= 5 && (K + L) >= 2)
    {
#include "iclass_hsds.h"
    }
    if ((I + J) >= 5 && (K + L) >= 3)
    {
#include "iclass_hsfs.h"
    }
    if ((I + J) >= 5 && (K + L) >= 4)
    {
#include "iclass_hsgs.h"
    }
}

#undef STORE_OPERATOR
#define STORE_OPERATOR +=
