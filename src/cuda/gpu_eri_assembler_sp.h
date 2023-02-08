/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 02/23/2022                !
 !                                                                     !
 ! Copyright (C) 2020-2021 Merz lab                                    !
 ! Copyright (C) 2020-2021 Götz lab                                    !
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
#define LOCSTORE(A,i1,i2,d1,d2)  A[(i1+(i2)*(d1))*gridDim.x*blockDim.x] 
#define VY(a,b,c) LOCVY(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3) 

__device__ __inline__ void ERint_vertical_sp(const int I, const int J, const int K, const int L, const int II, const int JJ, const int KK, const int LL, 
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz, const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz, 
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz, const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz, 
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp, const QUICKDouble ABcom, const QUICKDouble CDcom, 
        QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

    // [SS|SS] integral - Start
    LOCSTORE(store, 0, 0, STOREDIM, STOREDIM) += VY(0, 0, 0);
    // [SS|SS] integral - End 

    if ((I + J) >= 0 && (K + L) >= 1)
    {
        if (K <= 1 && I <= 0)
        {

            // [SS|PS] integral - Start
            QUICKDouble VY_0 = VY(0, 0, 0);
            QUICKDouble VY_1 = VY(0, 0, 1);
            LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) += Qtempx * VY_0 + WQtempx * VY_1;
            LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) += Qtempy * VY_0 + WQtempy * VY_1;
            LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) += Qtempz * VY_0 + WQtempz * VY_1;
            // [SS|PS] integral - End 

        }
        if ((I + J) >= 0 && (K + L) >= 2)
        {
            if (K <= 2 && I <= 0)
            {

                // [SS|DS] integral - Start
                QUICKDouble VY_0 = VY(0, 0, 0);
                QUICKDouble VY_1 = VY(0, 0, 1);
                QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                QUICKDouble VY_2 = VY(0, 0, 2);
                QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                LOCSTORE(store, 0, 7, STOREDIM, STOREDIM) += Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                LOCSTORE(store, 0, 8, STOREDIM, STOREDIM) += Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                LOCSTORE(store, 0, 4, STOREDIM, STOREDIM) += Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                LOCSTORE(store, 0, 9, STOREDIM, STOREDIM) += Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                LOCSTORE(store, 0, 6, STOREDIM, STOREDIM) += Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                LOCSTORE(store, 0, 5, STOREDIM, STOREDIM) += Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                // [SS|DS] integral - End 

            }
        }
    }
    if ((I + J) >= 1 && (K + L) >= 0)
    {
        if (I <= 1)
        {

            // [PS|SS] integral - Start
            QUICKDouble VY_0 = VY(0, 0, 0);
            QUICKDouble VY_1 = VY(0, 0, 1);
            LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) += Ptempx * VY_0 + WPtempx * VY_1;
            LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) += Ptempy * VY_0 + WPtempy * VY_1;
            LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) += Ptempz * VY_0 + WPtempz * VY_1;
            // [PS|SS] integral - End 

        }
        if ((I + J) >= 1 && (K + L) >= 1)
        {
            if (K <= 1 && I <= 1)
            {

                // [PS|PS] integral - Start
                QUICKDouble VY_0 = VY(0, 0, 0);
                QUICKDouble VY_1 = VY(0, 0, 1);
                QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                QUICKDouble VY_2 = VY(0, 0, 2);
                QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
                LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
                LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY_1;
                QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
                LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY_1;
                LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
                QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY_1;
                LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
                LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
                // [PS|PS] integral - End 

            }
            if ((I + J) >= 1 && (K + L) >= 2)
            {
                if (K <= 2 && I <= 1)
                {

                    // [PS|DS] integral - Start
                    QUICKDouble VY_1 = VY(0, 0, 1);
                    QUICKDouble VY_2 = VY(0, 0, 2);
                    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                    QUICKDouble VY_0 = VY(0, 0, 0);
                    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                    QUICKDouble VY_3 = VY(0, 0, 3);
                    QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
                    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                    LOCSTORE(store, 3, 4, STOREDIM, STOREDIM) += Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
                    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                    LOCSTORE(store, 2, 4, STOREDIM, STOREDIM) += Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
                    LOCSTORE(store, 1, 4, STOREDIM, STOREDIM) += Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
                    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                    LOCSTORE(store, 3, 5, STOREDIM, STOREDIM) += Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
                    LOCSTORE(store, 2, 5, STOREDIM, STOREDIM) += Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;

//if(I==1 && J==1 && K == 1 && L==1 && II==4 && JJ==4 && KK==4 && LL==5){
//printf("I %d J %d K %d L %d II %d JJ %d KK %d LL %d vrrstore %f \n", I, J, K, L, II, JJ, KK, LL, LOCSTORE(store, 2, 5, STOREDIM, STOREDIM));
//}
                    LOCSTORE(store, 1, 5, STOREDIM, STOREDIM) += Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
                    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                    LOCSTORE(store, 3, 6, STOREDIM, STOREDIM) += Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
                    LOCSTORE(store, 2, 6, STOREDIM, STOREDIM) += Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
                    LOCSTORE(store, 1, 6, STOREDIM, STOREDIM) += Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
                    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    LOCSTORE(store, 3, 7, STOREDIM, STOREDIM) += Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
                    LOCSTORE(store, 2, 7, STOREDIM, STOREDIM) += Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
                    LOCSTORE(store, 1, 7, STOREDIM, STOREDIM) += Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + 2.000000 * ABCDtemp * x_0_1_1;
                    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    LOCSTORE(store, 3, 8, STOREDIM, STOREDIM) += Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
                    LOCSTORE(store, 2, 8, STOREDIM, STOREDIM) += Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + 2.000000 * ABCDtemp * x_0_2_1;
                    LOCSTORE(store, 1, 8, STOREDIM, STOREDIM) += Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
                    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    LOCSTORE(store, 3, 9, STOREDIM, STOREDIM) += Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + 2.000000 * ABCDtemp * x_0_3_1;
                    LOCSTORE(store, 2, 9, STOREDIM, STOREDIM) += Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
                    LOCSTORE(store, 1, 9, STOREDIM, STOREDIM) += Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
                    // [PS|DS] integral - End 

                }
            }
            if ((I + J) >= 2 && (K + L) >= 1)
            {
                if (K <= 1 && I <= 2)
                {

                    // [DS|PS] integral - Start
                    QUICKDouble VY_1 = VY(0, 0, 1);
                    QUICKDouble VY_2 = VY(0, 0, 2);
                    QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
                    QUICKDouble VY_0 = VY(0, 0, 0);
                    QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
                    QUICKDouble VY_3 = VY(0, 0, 3);
                    QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
                    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
                    LOCSTORE(store, 4, 3, STOREDIM, STOREDIM) += Qtempz * x_4_0_0 + WQtempz * x_4_0_1;
                    QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                    LOCSTORE(store, 4, 2, STOREDIM, STOREDIM) += Qtempy * x_4_0_0 + WQtempy * x_4_0_1 + ABCDtemp * x_1_0_1;
                    LOCSTORE(store, 4, 1, STOREDIM, STOREDIM) += Qtempx * x_4_0_0 + WQtempx * x_4_0_1 + ABCDtemp * x_2_0_1;
                    QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                    QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                    QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
                    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                    LOCSTORE(store, 5, 3, STOREDIM, STOREDIM) += Qtempz * x_5_0_0 + WQtempz * x_5_0_1 + ABCDtemp * x_2_0_1;
                    LOCSTORE(store, 5, 2, STOREDIM, STOREDIM) += Qtempy * x_5_0_0 + WQtempy * x_5_0_1 + ABCDtemp * x_3_0_1;

                    LOCSTORE(store, 5, 1, STOREDIM, STOREDIM) += Qtempx * x_5_0_0 + WQtempx * x_5_0_1;
                    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                    LOCSTORE(store, 6, 3, STOREDIM, STOREDIM) += Qtempz * x_6_0_0 + WQtempz * x_6_0_1 + ABCDtemp * x_1_0_1;
                    LOCSTORE(store, 6, 2, STOREDIM, STOREDIM) += Qtempy * x_6_0_0 + WQtempy * x_6_0_1;
                    LOCSTORE(store, 6, 1, STOREDIM, STOREDIM) += Qtempx * x_6_0_0 + WQtempx * x_6_0_1 + ABCDtemp * x_3_0_1;
                    QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                    QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
                    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    LOCSTORE(store, 7, 3, STOREDIM, STOREDIM) += Qtempz * x_7_0_0 + WQtempz * x_7_0_1;
                    LOCSTORE(store, 7, 2, STOREDIM, STOREDIM) += Qtempy * x_7_0_0 + WQtempy * x_7_0_1;
                    LOCSTORE(store, 7, 1, STOREDIM, STOREDIM) += Qtempx * x_7_0_0 + WQtempx * x_7_0_1 + 2.000000 * ABCDtemp * x_1_0_1;
                    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    LOCSTORE(store, 8, 3, STOREDIM, STOREDIM) += Qtempz * x_8_0_0 + WQtempz * x_8_0_1;
                    LOCSTORE(store, 8, 2, STOREDIM, STOREDIM) += Qtempy * x_8_0_0 + WQtempy * x_8_0_1 + 2.000000 * ABCDtemp * x_2_0_1;
                    LOCSTORE(store, 8, 1, STOREDIM, STOREDIM) += Qtempx * x_8_0_0 + WQtempx * x_8_0_1;
                    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    LOCSTORE(store, 9, 3, STOREDIM, STOREDIM) += Qtempz * x_9_0_0 + WQtempz * x_9_0_1 + 2.000000 * ABCDtemp * x_3_0_1;
                    LOCSTORE(store, 9, 2, STOREDIM, STOREDIM) += Qtempy * x_9_0_0 + WQtempy * x_9_0_1;
                    LOCSTORE(store, 9, 1, STOREDIM, STOREDIM) += Qtempx * x_9_0_0 + WQtempx * x_9_0_1;
                    // [DS|PS] integral - End 

                }
                if ((I + J) >= 2 && (K + L) >= 2)
                {
                    if (K <= 2 && I <= 2)
                    {

                        // [DS|DS] integral - Start
                        QUICKDouble VY_0 = VY(0, 0, 0);
                        QUICKDouble VY_1 = VY(0, 0, 1);
                        QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                        QUICKDouble VY_2 = VY(0, 0, 2);
                        QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                        QUICKDouble VY_3 = VY(0, 0, 3);
                        QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
                        QUICKDouble VY_4 = VY(0, 0, 4);
                        QUICKDouble x_0_2_3 = Qtempy * VY_3 + WQtempy * VY_4;
                        QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                        QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                        QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
                        QUICKDouble x_3_4_0 = Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
                        QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
                        LOCSTORE(store, 9, 4, STOREDIM, STOREDIM) += Ptempz * x_3_4_0 + WPtempz * x_3_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1);
                        QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                        QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                        QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                        QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
                        QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                        QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                        QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                        QUICKDouble x_1_5_0 = Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
                        QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                        LOCSTORE(store, 7, 5, STOREDIM, STOREDIM) += Ptempx * x_1_5_0 + WPtempx * x_1_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1);
                        QUICKDouble x_2_5_0 = Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
                        QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                        LOCSTORE(store, 4, 5, STOREDIM, STOREDIM) += Ptempx * x_2_5_0 + WPtempx * x_2_5_1;
                        QUICKDouble x_3_5_0 = Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
                        QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                        LOCSTORE(store, 6, 5, STOREDIM, STOREDIM) += Ptempx * x_3_5_0 + WPtempx * x_3_5_1;
                        QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                        QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                        QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                        QUICKDouble x_2_6_0 = Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
                        QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                        LOCSTORE(store, 8, 6, STOREDIM, STOREDIM) += Ptempy * x_2_6_0 + WPtempy * x_2_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1);
                        QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                        QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                        QUICKDouble x_3_6_0 = Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
                        QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                        LOCSTORE(store, 5, 6, STOREDIM, STOREDIM) += Ptempy * x_3_6_0 + WPtempy * x_3_6_1;
                        QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                        QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
                        QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_2_7_0 = Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
                        QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                        LOCSTORE(store, 8, 7, STOREDIM, STOREDIM) += Ptempy * x_2_7_0 + WPtempy * x_2_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                        QUICKDouble x_3_7_0 = Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
                        QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                        LOCSTORE(store, 9, 7, STOREDIM, STOREDIM) += Ptempz * x_3_7_0 + WPtempz * x_3_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                        LOCSTORE(store, 5, 7, STOREDIM, STOREDIM) += Ptempy * x_3_7_0 + WPtempy * x_3_7_1;
                        QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_1_8_0 = Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
                        QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                        LOCSTORE(store, 7, 8, STOREDIM, STOREDIM) += Ptempx * x_1_8_0 + WPtempx * x_1_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                        QUICKDouble x_2_8_0 = Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + 2.000000 * ABCDtemp * x_0_2_1;
                        QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + 2.000000 * ABCDtemp * x_0_2_2;
                        LOCSTORE(store, 4, 8, STOREDIM, STOREDIM) += Ptempx * x_2_8_0 + WPtempx * x_2_8_1;
                        QUICKDouble x_3_8_0 = Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
                        QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                        LOCSTORE(store, 9, 8, STOREDIM, STOREDIM) += Ptempz * x_3_8_0 + WPtempz * x_3_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                        LOCSTORE(store, 6, 8, STOREDIM, STOREDIM) += Ptempx * x_3_8_0 + WPtempx * x_3_8_1;
                        QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_1_9_0 = Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
                        QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                        LOCSTORE(store, 7, 9, STOREDIM, STOREDIM) += Ptempx * x_1_9_0 + WPtempx * x_1_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                        QUICKDouble x_2_9_0 = Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
                        QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                        LOCSTORE(store, 8, 9, STOREDIM, STOREDIM) += Ptempy * x_2_9_0 + WPtempy * x_2_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                        LOCSTORE(store, 4, 9, STOREDIM, STOREDIM) += Ptempx * x_2_9_0 + WPtempx * x_2_9_1;
                        QUICKDouble x_3_9_0 = Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + 2.000000 * ABCDtemp * x_0_3_1;
                        QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + 2.000000 * ABCDtemp * x_0_3_2;
                        LOCSTORE(store, 6, 9, STOREDIM, STOREDIM) += Ptempx * x_3_9_0 + WPtempx * x_3_9_1;
                        LOCSTORE(store, 5, 9, STOREDIM, STOREDIM) += Ptempy * x_3_9_0 + WPtempy * x_3_9_1;
                        QUICKDouble x_1_7_0 = Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + 2.000000 * ABCDtemp * x_0_1_1;
                        QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + 2.000000 * ABCDtemp * x_0_1_2;
                        QUICKDouble x_1_1_1 = Ptempx * x_0_1_1 + WPtempx * x_0_1_2 + ABCDtemp * VY_2;
                        LOCSTORE(store, 7, 7, STOREDIM, STOREDIM) += Ptempx * x_1_7_0 + WPtempx * x_1_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1) + 2.000000 * ABCDtemp * x_1_1_1;
                        QUICKDouble x_1_4_0 = Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
                        QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                        QUICKDouble x_1_2_1 = Ptempx * x_0_2_1 + WPtempx * x_0_2_2;
                        LOCSTORE(store, 7, 4, STOREDIM, STOREDIM) += Ptempx * x_1_4_0 + WPtempx * x_1_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_1_2_1;
                        QUICKDouble x_1_6_0 = Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
                        QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                        QUICKDouble x_1_3_1 = Ptempx * x_0_3_1 + WPtempx * x_0_3_2;
                        LOCSTORE(store, 7, 6, STOREDIM, STOREDIM) += Ptempx * x_1_6_0 + WPtempx * x_1_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_1_3_1;
                        QUICKDouble x_2_4_0 = Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
                        QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                        QUICKDouble x_2_1_1 = Ptempy * x_0_1_1 + WPtempy * x_0_1_2;
                        LOCSTORE(store, 8, 4, STOREDIM, STOREDIM) += Ptempy * x_2_4_0 + WPtempy * x_2_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_2_1_1;
                        LOCSTORE(store, 4, 7, STOREDIM, STOREDIM) += Ptempx * x_2_7_0 + WPtempx * x_2_7_1 + 2.000000 * ABCDtemp * x_2_1_1;
                        QUICKDouble x_2_2_1 = Ptempy * x_0_2_1 + WPtempy * x_0_2_2 + ABCDtemp * VY_2;
                        LOCSTORE(store, 8, 8, STOREDIM, STOREDIM) += Ptempy * x_2_8_0 + WPtempy * x_2_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1) + 2.000000 * ABCDtemp * x_2_2_1;
                        LOCSTORE(store, 4, 4, STOREDIM, STOREDIM) += Ptempx * x_2_4_0 + WPtempx * x_2_4_1 + ABCDtemp * x_2_2_1;
                        QUICKDouble x_2_3_1 = Ptempy * x_0_3_1 + WPtempy * x_0_3_2;
                        LOCSTORE(store, 8, 5, STOREDIM, STOREDIM) += Ptempy * x_2_5_0 + WPtempy * x_2_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_2_3_1;
                        LOCSTORE(store, 4, 6, STOREDIM, STOREDIM) += Ptempx * x_2_6_0 + WPtempx * x_2_6_1 + ABCDtemp * x_2_3_1;
                        QUICKDouble x_3_1_1 = Ptempz * x_0_1_1 + WPtempz * x_0_1_2;
                        LOCSTORE(store, 9, 6, STOREDIM, STOREDIM) += Ptempz * x_3_6_0 + WPtempz * x_3_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_3_1_1;
                        LOCSTORE(store, 6, 7, STOREDIM, STOREDIM) += Ptempx * x_3_7_0 + WPtempx * x_3_7_1 + 2.000000 * ABCDtemp * x_3_1_1;
                        LOCSTORE(store, 5, 4, STOREDIM, STOREDIM) += Ptempy * x_3_4_0 + WPtempy * x_3_4_1 + ABCDtemp * x_3_1_1;
                        QUICKDouble x_3_2_1 = Ptempz * x_0_2_1 + WPtempz * x_0_2_2;
                        LOCSTORE(store, 9, 5, STOREDIM, STOREDIM) += Ptempz * x_3_5_0 + WPtempz * x_3_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_3_2_1;
                        LOCSTORE(store, 6, 4, STOREDIM, STOREDIM) += Ptempx * x_3_4_0 + WPtempx * x_3_4_1 + ABCDtemp * x_3_2_1;
                        LOCSTORE(store, 5, 8, STOREDIM, STOREDIM) += Ptempy * x_3_8_0 + WPtempy * x_3_8_1 + 2.000000 * ABCDtemp * x_3_2_1;
                        QUICKDouble x_3_3_1 = Ptempz * x_0_3_1 + WPtempz * x_0_3_2 + ABCDtemp * VY_2;
                        LOCSTORE(store, 9, 9, STOREDIM, STOREDIM) += Ptempz * x_3_9_0 + WPtempz * x_3_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1) + 2.000000 * ABCDtemp * x_3_3_1;
                        LOCSTORE(store, 6, 6, STOREDIM, STOREDIM) += Ptempx * x_3_6_0 + WPtempx * x_3_6_1 + ABCDtemp * x_3_3_1;
                        LOCSTORE(store, 5, 5, STOREDIM, STOREDIM) += Ptempy * x_3_5_0 + WPtempy * x_3_5_1 + ABCDtemp * x_3_3_1;
                        // [DS|DS] integral - End 

                    }
                }
            }
        }
        if ((I + J) >= 2 && (K + L) >= 0)
        {
            if (K <= 0 && I <= 2)
            {

                // [DS|SS] integral - Start
                QUICKDouble VY_0 = VY(0, 0, 0);
                QUICKDouble VY_1 = VY(0, 0, 1);
                QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                QUICKDouble VY_2 = VY(0, 0, 2);
                QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                LOCSTORE(store, 7, 0, STOREDIM, STOREDIM) += Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
                QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
                LOCSTORE(store, 8, 0, STOREDIM, STOREDIM) += Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                LOCSTORE(store, 4, 0, STOREDIM, STOREDIM) += Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                LOCSTORE(store, 9, 0, STOREDIM, STOREDIM) += Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                LOCSTORE(store, 6, 0, STOREDIM, STOREDIM) += Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                LOCSTORE(store, 5, 0, STOREDIM, STOREDIM) += Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                // [DS|SS] integral - End 

            }
        }
    }

} 
