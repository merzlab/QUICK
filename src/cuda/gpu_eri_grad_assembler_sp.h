/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 08/02/2022                !
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
#define STOREDIM STOREDIM_GRAD_T 
#define VDIM3 VDIM3_GRAD_T 
#define LOCSTORE(A,i1,i2,d1,d2)  A[(i1+(i2)*(d1))*gridDim.x*blockDim.x] 
#define VY(a,b,c) LOCVY(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3) 

__device__ __inline__ void ERint_grad_vertical_sp(const int I, const int J, const int K, const int L, const int II, const int JJ, const int KK, const int LL, 
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz, const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz, 
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz, const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz, 
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp, const QUICKDouble ABcom, const QUICKDouble CDcom, 
        QUICKDouble* store, QUICKDouble* YVerticalTemp){ 

    // [SS|SS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    LOCSTORE(store, 0, 0, STOREDIM, STOREDIM) STORE_OPERATOR VY_0;
    // [SS|SS] integral - End 

    if ((I+J) >=  0 && (K+L)>= 1)
    {

        // [SS|PS] integral - Start
        QUICKDouble VY_0 = VY(0, 0, 0);
        QUICKDouble VY_1 = VY(0, 0, 1);
        LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * VY_0 + WQtempx * VY_1;
        LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * VY_0 + WQtempy * VY_1;
        LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * VY_0 + WQtempz * VY_1;
        // [SS|PS] integral - End 

       if ((I+J) >=  0 && (K+L)>= 2)
       {

           // [SS|DS] integral - Start
           QUICKDouble VY_0 = VY(0, 0, 0);
           QUICKDouble VY_1 = VY(0, 0, 1);
           QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
           QUICKDouble VY_2 = VY(0, 0, 2);
           QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
           LOCSTORE(store, 0, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
           QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
           QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
           LOCSTORE(store, 0, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
           LOCSTORE(store, 0, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
           QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
           QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
           LOCSTORE(store, 0, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
           LOCSTORE(store, 0, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
           LOCSTORE(store, 0, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
           // [SS|DS] integral - End 

           if ((I+J) >=  0 && (K+L)>= 3)
           {

               // [SS|FS] integral - Start
               QUICKDouble VY_0 = VY(0, 0, 0);
               QUICKDouble VY_1 = VY(0, 0, 1);
               QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
               QUICKDouble VY_2 = VY(0, 0, 2);
               QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
               QUICKDouble VY_3 = VY(0, 0, 3);
               QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
               QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
               QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
               LOCSTORE(store, 0, 11, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
               QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
               QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
               QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
               QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
               QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
               LOCSTORE(store, 0, 15, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
               LOCSTORE(store, 0, 10, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
               QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
               QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
               LOCSTORE(store, 0, 13, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
               QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
               QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
               QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
               QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
               QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
               LOCSTORE(store, 0, 17, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
               QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
               QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
               LOCSTORE(store, 0, 18, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
               LOCSTORE(store, 0, 12, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
               QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
               QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
               LOCSTORE(store, 0, 19, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
               LOCSTORE(store, 0, 16, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
               LOCSTORE(store, 0, 14, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
               // [SS|FS] integral - End 

           }
       }
   }
   if ((I+J) >=  1 && (K+L)>= 0)
   {

        // [PS|SS] integral - Start
        QUICKDouble VY_0 = VY(0, 0, 0);
        QUICKDouble VY_1 = VY(0, 0, 1);
        LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * VY_0 + WPtempx * VY_1;
        LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * VY_0 + WPtempy * VY_1;
        LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * VY_0 + WPtempz * VY_1;
        // [PS|SS] integral - End 

       if ((I+J) >=  1 && (K+L)>= 1)
       {

            // [PS|PS] integral - Start
            QUICKDouble VY_0 = VY(0, 0, 0);
            QUICKDouble VY_1 = VY(0, 0, 1);
            QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
            QUICKDouble VY_2 = VY(0, 0, 2);
            QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
            LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
            LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
            LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY_1;
            QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
            QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
            LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
            LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY_1;
            LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
            QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
            QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
            LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY_1;
            LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
            LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
            // [PS|PS] integral - End 

           if ((I+J) >=  1 && (K+L)>= 2)
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
                LOCSTORE(store, 3, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
                QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                LOCSTORE(store, 2, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
                LOCSTORE(store, 1, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
                QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                LOCSTORE(store, 3, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
                LOCSTORE(store, 2, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
                LOCSTORE(store, 1, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
                QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                LOCSTORE(store, 3, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
                LOCSTORE(store, 2, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
                LOCSTORE(store, 1, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
                QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                LOCSTORE(store, 3, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
                LOCSTORE(store, 2, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
                LOCSTORE(store, 1, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + 2.000000 * ABCDtemp * x_0_1_1;
                QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                LOCSTORE(store, 3, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
                LOCSTORE(store, 2, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + 2.000000 * ABCDtemp * x_0_2_1;
                LOCSTORE(store, 1, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
                QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                LOCSTORE(store, 3, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + 2.000000 * ABCDtemp * x_0_3_1;
                LOCSTORE(store, 2, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
                LOCSTORE(store, 1, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
                // [PS|DS] integral - End 

               if ((I+J) >=  1 && (K+L)>= 3)
               {

                    // [PS|FS] integral - Start
                    QUICKDouble VY_1 = VY(0, 0, 1);
                    QUICKDouble VY_2 = VY(0, 0, 2);
                    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                    QUICKDouble VY_3 = VY(0, 0, 3);
                    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                    QUICKDouble VY_0 = VY(0, 0, 0);
                    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                    QUICKDouble VY_4 = VY(0, 0, 4);
                    QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
                    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                    QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
                    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                    QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
                    QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
                    LOCSTORE(store, 3, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_10_0 + WPtempz * x_0_10_1 + ABCDtemp * x_0_4_1;
                    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                    LOCSTORE(store, 2, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_10_0 + WPtempy * x_0_10_1 + ABCDtemp * x_0_6_1;
                    LOCSTORE(store, 1, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_10_0 + WPtempx * x_0_10_1 + ABCDtemp * x_0_5_1;
                    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                    QUICKDouble x_0_2_3 = Qtempy * VY_3 + WQtempy * VY_4;
                    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                    QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
                    QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
                    QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
                    LOCSTORE(store, 3, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_11_0 + WPtempz * x_0_11_1;
                    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    LOCSTORE(store, 2, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_11_0 + WPtempy * x_0_11_1 + ABCDtemp * x_0_7_1;
                    LOCSTORE(store, 1, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_11_0 + WPtempx * x_0_11_1 + 2.000000 * ABCDtemp * x_0_4_1;
                    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
                    QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
                    LOCSTORE(store, 3, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_12_0 + WPtempz * x_0_12_1;
                    LOCSTORE(store, 2, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_12_0 + WPtempy * x_0_12_1 + 2.000000 * ABCDtemp * x_0_4_1;
                    LOCSTORE(store, 1, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_12_0 + WPtempx * x_0_12_1 + ABCDtemp * x_0_8_1;
                    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                    QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                    QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                    LOCSTORE(store, 3, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_13_0 + WPtempz * x_0_13_1 + ABCDtemp * x_0_7_1;
                    LOCSTORE(store, 2, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_13_0 + WPtempy * x_0_13_1;
                    LOCSTORE(store, 1, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_13_0 + WPtempx * x_0_13_1 + 2.000000 * ABCDtemp * x_0_6_1;
                    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
                    QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
                    LOCSTORE(store, 3, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_14_0 + WPtempz * x_0_14_1 + 2.000000 * ABCDtemp * x_0_6_1;
                    LOCSTORE(store, 2, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_14_0 + WPtempy * x_0_14_1;
                    LOCSTORE(store, 1, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_14_0 + WPtempx * x_0_14_1 + ABCDtemp * x_0_9_1;
                    QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                    QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                    LOCSTORE(store, 3, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_15_0 + WPtempz * x_0_15_1 + ABCDtemp * x_0_8_1;
                    LOCSTORE(store, 2, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_15_0 + WPtempy * x_0_15_1 + 2.000000 * ABCDtemp * x_0_5_1;
                    LOCSTORE(store, 1, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_15_0 + WPtempx * x_0_15_1;
                    QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
                    QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
                    LOCSTORE(store, 3, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_16_0 + WPtempz * x_0_16_1 + 2.000000 * ABCDtemp * x_0_5_1;
                    LOCSTORE(store, 2, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_16_0 + WPtempy * x_0_16_1 + ABCDtemp * x_0_9_1;
                    LOCSTORE(store, 1, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_16_0 + WPtempx * x_0_16_1;
                    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                    QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
                    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
                    QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2.000000 * CDtemp * (x_0_1_1 - ABcom * x_0_1_2);
                    LOCSTORE(store, 3, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_17_0 + WPtempz * x_0_17_1;
                    LOCSTORE(store, 2, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_17_0 + WPtempy * x_0_17_1;
                    LOCSTORE(store, 1, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_17_0 + WPtempx * x_0_17_1 + 3.000000 * ABCDtemp * x_0_7_1;
                    QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
                    QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2.000000 * CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
                    LOCSTORE(store, 3, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_18_0 + WPtempz * x_0_18_1;
                    LOCSTORE(store, 2, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_18_0 + WPtempy * x_0_18_1 + 3.000000 * ABCDtemp * x_0_8_1;
                    LOCSTORE(store, 1, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_18_0 + WPtempx * x_0_18_1;
                    QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                    QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2.000000 * CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                    LOCSTORE(store, 3, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_0_19_0 + WPtempz * x_0_19_1 + 3.000000 * ABCDtemp * x_0_9_1;
                    LOCSTORE(store, 2, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_0_19_0 + WPtempy * x_0_19_1;
                    LOCSTORE(store, 1, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_0_19_0 + WPtempx * x_0_19_1;
                    // [PS|FS] integral - End 

               }
           }
           if ((I+J) >=  2 && (K+L)>= 1)
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
                LOCSTORE(store, 4, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_4_0_0 + WQtempz * x_4_0_1;
                QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                LOCSTORE(store, 4, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_4_0_0 + WQtempy * x_4_0_1 + ABCDtemp * x_1_0_1;
                LOCSTORE(store, 4, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_4_0_0 + WQtempx * x_4_0_1 + ABCDtemp * x_2_0_1;
                QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
                QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                LOCSTORE(store, 5, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_5_0_0 + WQtempz * x_5_0_1 + ABCDtemp * x_2_0_1;
                LOCSTORE(store, 5, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_5_0_0 + WQtempy * x_5_0_1 + ABCDtemp * x_3_0_1;
                LOCSTORE(store, 5, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_5_0_0 + WQtempx * x_5_0_1;
                QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                LOCSTORE(store, 6, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_6_0_0 + WQtempz * x_6_0_1 + ABCDtemp * x_1_0_1;
                LOCSTORE(store, 6, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_6_0_0 + WQtempy * x_6_0_1;
                LOCSTORE(store, 6, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_6_0_0 + WQtempx * x_6_0_1 + ABCDtemp * x_3_0_1;
                QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
                QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 7, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_7_0_0 + WQtempz * x_7_0_1;
                LOCSTORE(store, 7, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_7_0_0 + WQtempy * x_7_0_1;
                LOCSTORE(store, 7, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_7_0_0 + WQtempx * x_7_0_1 + 2.000000 * ABCDtemp * x_1_0_1;
                QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 8, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_8_0_0 + WQtempz * x_8_0_1;
                LOCSTORE(store, 8, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_8_0_0 + WQtempy * x_8_0_1 + 2.000000 * ABCDtemp * x_2_0_1;
                LOCSTORE(store, 8, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_8_0_0 + WQtempx * x_8_0_1;
                QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 9, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_9_0_0 + WQtempz * x_9_0_1 + 2.000000 * ABCDtemp * x_3_0_1;
                LOCSTORE(store, 9, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_9_0_0 + WQtempy * x_9_0_1;
                LOCSTORE(store, 9, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_9_0_0 + WQtempx * x_9_0_1;
                // [DS|PS] integral - End 

               if ((I+J) >=  2 && (K+L)>= 2)
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
                    LOCSTORE(store, 9, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_4_0 + WPtempz * x_3_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1);
                    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                    QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
                    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                    QUICKDouble x_1_5_0 = Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
                    QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                    LOCSTORE(store, 7, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_5_0 + WPtempx * x_1_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1);
                    QUICKDouble x_2_5_0 = Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
                    QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                    LOCSTORE(store, 4, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_5_0 + WPtempx * x_2_5_1;
                    QUICKDouble x_3_5_0 = Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
                    QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                    LOCSTORE(store, 6, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_5_0 + WPtempx * x_3_5_1;
                    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                    QUICKDouble x_2_6_0 = Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
                    QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                    LOCSTORE(store, 8, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_6_0 + WPtempy * x_2_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1);
                    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                    QUICKDouble x_3_6_0 = Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
                    QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                    LOCSTORE(store, 5, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_6_0 + WPtempy * x_3_6_1;
                    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                    QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
                    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_2_7_0 = Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
                    QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                    LOCSTORE(store, 8, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_7_0 + WPtempy * x_2_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                    QUICKDouble x_3_7_0 = Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
                    QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                    LOCSTORE(store, 9, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_7_0 + WPtempz * x_3_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                    LOCSTORE(store, 5, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_7_0 + WPtempy * x_3_7_1;
                    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_1_8_0 = Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
                    QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                    LOCSTORE(store, 7, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_8_0 + WPtempx * x_1_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                    QUICKDouble x_2_8_0 = Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + 2.000000 * ABCDtemp * x_0_2_1;
                    QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + 2.000000 * ABCDtemp * x_0_2_2;
                    LOCSTORE(store, 4, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_8_0 + WPtempx * x_2_8_1;
                    QUICKDouble x_3_8_0 = Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
                    QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                    LOCSTORE(store, 9, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_8_0 + WPtempz * x_3_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                    LOCSTORE(store, 6, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_8_0 + WPtempx * x_3_8_1;
                    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
                    QUICKDouble x_1_9_0 = Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
                    QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                    LOCSTORE(store, 7, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_9_0 + WPtempx * x_1_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                    QUICKDouble x_2_9_0 = Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
                    QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                    LOCSTORE(store, 8, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_9_0 + WPtempy * x_2_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                    LOCSTORE(store, 4, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_9_0 + WPtempx * x_2_9_1;
                    QUICKDouble x_3_9_0 = Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + 2.000000 * ABCDtemp * x_0_3_1;
                    QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + 2.000000 * ABCDtemp * x_0_3_2;
                    LOCSTORE(store, 6, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_9_0 + WPtempx * x_3_9_1;
                    LOCSTORE(store, 5, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_9_0 + WPtempy * x_3_9_1;
                    QUICKDouble x_1_7_0 = Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + 2.000000 * ABCDtemp * x_0_1_1;
                    QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + 2.000000 * ABCDtemp * x_0_1_2;
                    QUICKDouble x_1_1_1 = Ptempx * x_0_1_1 + WPtempx * x_0_1_2 + ABCDtemp * VY_2;
                    LOCSTORE(store, 7, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_7_0 + WPtempx * x_1_7_1 + ABtemp * (x_0_7_0 - CDcom * x_0_7_1) + 2.000000 * ABCDtemp * x_1_1_1;
                    QUICKDouble x_1_4_0 = Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
                    QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                    QUICKDouble x_1_2_1 = Ptempx * x_0_2_1 + WPtempx * x_0_2_2;
                    LOCSTORE(store, 7, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_4_0 + WPtempx * x_1_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_1_2_1;
                    QUICKDouble x_1_6_0 = Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
                    QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                    QUICKDouble x_1_3_1 = Ptempx * x_0_3_1 + WPtempx * x_0_3_2;
                    LOCSTORE(store, 7, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_6_0 + WPtempx * x_1_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_1_3_1;
                    QUICKDouble x_2_4_0 = Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
                    QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                    QUICKDouble x_2_1_1 = Ptempy * x_0_1_1 + WPtempy * x_0_1_2;
                    LOCSTORE(store, 8, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_4_0 + WPtempy * x_2_4_1 + ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_2_1_1;
                    LOCSTORE(store, 4, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_7_0 + WPtempx * x_2_7_1 + 2.000000 * ABCDtemp * x_2_1_1;
                    QUICKDouble x_2_2_1 = Ptempy * x_0_2_1 + WPtempy * x_0_2_2 + ABCDtemp * VY_2;
                    LOCSTORE(store, 8, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_8_0 + WPtempy * x_2_8_1 + ABtemp * (x_0_8_0 - CDcom * x_0_8_1) + 2.000000 * ABCDtemp * x_2_2_1;
                    LOCSTORE(store, 4, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_4_0 + WPtempx * x_2_4_1 + ABCDtemp * x_2_2_1;
                    QUICKDouble x_2_3_1 = Ptempy * x_0_3_1 + WPtempy * x_0_3_2;
                    LOCSTORE(store, 8, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_5_0 + WPtempy * x_2_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_2_3_1;
                    LOCSTORE(store, 4, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_6_0 + WPtempx * x_2_6_1 + ABCDtemp * x_2_3_1;
                    QUICKDouble x_3_1_1 = Ptempz * x_0_1_1 + WPtempz * x_0_1_2;
                    LOCSTORE(store, 9, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_6_0 + WPtempz * x_3_6_1 + ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_3_1_1;
                    LOCSTORE(store, 6, 7, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_7_0 + WPtempx * x_3_7_1 + 2.000000 * ABCDtemp * x_3_1_1;
                    LOCSTORE(store, 5, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_4_0 + WPtempy * x_3_4_1 + ABCDtemp * x_3_1_1;
                    QUICKDouble x_3_2_1 = Ptempz * x_0_2_1 + WPtempz * x_0_2_2;
                    LOCSTORE(store, 9, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_5_0 + WPtempz * x_3_5_1 + ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_3_2_1;
                    LOCSTORE(store, 6, 4, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_4_0 + WPtempx * x_3_4_1 + ABCDtemp * x_3_2_1;
                    LOCSTORE(store, 5, 8, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_8_0 + WPtempy * x_3_8_1 + 2.000000 * ABCDtemp * x_3_2_1;
                    QUICKDouble x_3_3_1 = Ptempz * x_0_3_1 + WPtempz * x_0_3_2 + ABCDtemp * VY_2;
                    LOCSTORE(store, 9, 9, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_9_0 + WPtempz * x_3_9_1 + ABtemp * (x_0_9_0 - CDcom * x_0_9_1) + 2.000000 * ABCDtemp * x_3_3_1;
                    LOCSTORE(store, 6, 6, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_6_0 + WPtempx * x_3_6_1 + ABCDtemp * x_3_3_1;
                    LOCSTORE(store, 5, 5, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_5_0 + WPtempy * x_3_5_1 + ABCDtemp * x_3_3_1;
                    // [DS|DS] integral - End 

                   if ((I+J) >=  2 && (K+L)>= 3)
                   {

                        // [DS|FS] integral - Start
                        QUICKDouble VY_0 = VY(0, 0, 0);
                        QUICKDouble VY_1 = VY(0, 0, 1);
                        QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
                        QUICKDouble VY_2 = VY(0, 0, 2);
                        QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
                        QUICKDouble VY_3 = VY(0, 0, 3);
                        QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
                        QUICKDouble VY_4 = VY(0, 0, 4);
                        QUICKDouble x_0_2_3 = Qtempy * VY_3 + WQtempy * VY_4;
                        QUICKDouble VY_5 = VY(0, 0, 5);
                        QUICKDouble x_0_2_4 = Qtempy * VY_4 + WQtempy * VY_5;
                        QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                        QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                        QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
                        QUICKDouble x_0_4_3 = Qtempx * x_0_2_3 + WQtempx * x_0_2_4;
                        QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
                        QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
                        QUICKDouble x_0_11_2 = Qtempx * x_0_4_2 + WQtempx * x_0_4_3 + CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
                        QUICKDouble x_3_11_0 = Ptempz * x_0_11_0 + WPtempz * x_0_11_1;
                        QUICKDouble x_3_11_1 = Ptempz * x_0_11_1 + WPtempz * x_0_11_2;
                        LOCSTORE(store, 9, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_11_0 + WPtempz * x_3_11_1 + ABtemp * (x_0_11_0 - CDcom * x_0_11_1);
                        QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_0_8_3 = Qtempy * x_0_2_3 + WQtempy * x_0_2_4 + CDtemp * (VY_3 - ABcom * VY_4);
                        QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
                        QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
                        QUICKDouble x_0_12_2 = Qtempx * x_0_8_2 + WQtempx * x_0_8_3;
                        QUICKDouble x_3_12_0 = Ptempz * x_0_12_0 + WPtempz * x_0_12_1;
                        QUICKDouble x_3_12_1 = Ptempz * x_0_12_1 + WPtempz * x_0_12_2;
                        LOCSTORE(store, 9, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_12_0 + WPtempz * x_3_12_1 + ABtemp * (x_0_12_0 - CDcom * x_0_12_1);
                        QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
                        QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
                        QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
                        QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
                        QUICKDouble x_0_3_4 = Qtempz * VY_4 + WQtempz * VY_5;
                        QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                        QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                        QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                        QUICKDouble x_0_6_3 = Qtempx * x_0_3_3 + WQtempx * x_0_3_4;
                        QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                        QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                        QUICKDouble x_0_13_2 = Qtempx * x_0_6_2 + WQtempx * x_0_6_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
                        QUICKDouble x_2_13_0 = Ptempy * x_0_13_0 + WPtempy * x_0_13_1;
                        QUICKDouble x_2_13_1 = Ptempy * x_0_13_1 + WPtempy * x_0_13_2;
                        LOCSTORE(store, 8, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_13_0 + WPtempy * x_2_13_1 + ABtemp * (x_0_13_0 - CDcom * x_0_13_1);
                        QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
                        QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
                        QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
                        QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_3_13_0 = Ptempz * x_0_13_0 + WPtempz * x_0_13_1 + ABCDtemp * x_0_7_1;
                        QUICKDouble x_3_13_1 = Ptempz * x_0_13_1 + WPtempz * x_0_13_2 + ABCDtemp * x_0_7_2;
                        LOCSTORE(store, 5, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_13_0 + WPtempy * x_3_13_1;
                        QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
                        QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
                        QUICKDouble x_0_9_3 = Qtempz * x_0_3_3 + WQtempz * x_0_3_4 + CDtemp * (VY_3 - ABcom * VY_4);
                        QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
                        QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
                        QUICKDouble x_0_14_2 = Qtempx * x_0_9_2 + WQtempx * x_0_9_3;
                        QUICKDouble x_2_14_0 = Ptempy * x_0_14_0 + WPtempy * x_0_14_1;
                        QUICKDouble x_2_14_1 = Ptempy * x_0_14_1 + WPtempy * x_0_14_2;
                        LOCSTORE(store, 8, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_14_0 + WPtempy * x_2_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1);
                        QUICKDouble x_3_14_0 = Ptempz * x_0_14_0 + WPtempz * x_0_14_1 + 2.000000 * ABCDtemp * x_0_6_1;
                        QUICKDouble x_3_14_1 = Ptempz * x_0_14_1 + WPtempz * x_0_14_2 + 2.000000 * ABCDtemp * x_0_6_2;
                        LOCSTORE(store, 5, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_14_0 + WPtempy * x_3_14_1;
                        QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                        QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                        QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                        QUICKDouble x_0_5_3 = Qtempy * x_0_3_3 + WQtempy * x_0_3_4;
                        QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                        QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                        QUICKDouble x_0_15_2 = Qtempy * x_0_5_2 + WQtempy * x_0_5_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
                        QUICKDouble x_1_15_0 = Ptempx * x_0_15_0 + WPtempx * x_0_15_1;
                        QUICKDouble x_1_15_1 = Ptempx * x_0_15_1 + WPtempx * x_0_15_2;
                        LOCSTORE(store, 7, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_15_0 + WPtempx * x_1_15_1 + ABtemp * (x_0_15_0 - CDcom * x_0_15_1);
                        QUICKDouble x_2_15_0 = Ptempy * x_0_15_0 + WPtempy * x_0_15_1 + 2.000000 * ABCDtemp * x_0_5_1;
                        QUICKDouble x_2_15_1 = Ptempy * x_0_15_1 + WPtempy * x_0_15_2 + 2.000000 * ABCDtemp * x_0_5_2;
                        LOCSTORE(store, 4, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_15_0 + WPtempx * x_2_15_1;
                        QUICKDouble x_3_15_0 = Ptempz * x_0_15_0 + WPtempz * x_0_15_1 + ABCDtemp * x_0_8_1;
                        QUICKDouble x_3_15_1 = Ptempz * x_0_15_1 + WPtempz * x_0_15_2 + ABCDtemp * x_0_8_2;
                        LOCSTORE(store, 6, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_15_0 + WPtempx * x_3_15_1;
                        QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
                        QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
                        QUICKDouble x_0_16_2 = Qtempy * x_0_9_2 + WQtempy * x_0_9_3;
                        QUICKDouble x_1_16_0 = Ptempx * x_0_16_0 + WPtempx * x_0_16_1;
                        QUICKDouble x_1_16_1 = Ptempx * x_0_16_1 + WPtempx * x_0_16_2;
                        LOCSTORE(store, 7, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_16_0 + WPtempx * x_1_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1);
                        QUICKDouble x_2_16_0 = Ptempy * x_0_16_0 + WPtempy * x_0_16_1 + ABCDtemp * x_0_9_1;
                        QUICKDouble x_2_16_1 = Ptempy * x_0_16_1 + WPtempy * x_0_16_2 + ABCDtemp * x_0_9_2;
                        LOCSTORE(store, 4, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_16_0 + WPtempx * x_2_16_1;
                        QUICKDouble x_3_16_0 = Ptempz * x_0_16_0 + WPtempz * x_0_16_1 + 2.000000 * ABCDtemp * x_0_5_1;
                        QUICKDouble x_3_16_1 = Ptempz * x_0_16_1 + WPtempz * x_0_16_2 + 2.000000 * ABCDtemp * x_0_5_2;
                        LOCSTORE(store, 6, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_16_0 + WPtempx * x_3_16_1;
                        QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
                        QUICKDouble x_0_1_4 = Qtempx * VY_4 + WQtempx * VY_5;
                        QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
                        QUICKDouble x_0_7_3 = Qtempx * x_0_1_3 + WQtempx * x_0_1_4 + CDtemp * (VY_3 - ABcom * VY_4);
                        QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
                        QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2.000000 * CDtemp * (x_0_1_1 - ABcom * x_0_1_2);
                        QUICKDouble x_0_17_2 = Qtempx * x_0_7_2 + WQtempx * x_0_7_3 + 2.000000 * CDtemp * (x_0_1_2 - ABcom * x_0_1_3);
                        QUICKDouble x_2_17_0 = Ptempy * x_0_17_0 + WPtempy * x_0_17_1;
                        QUICKDouble x_2_17_1 = Ptempy * x_0_17_1 + WPtempy * x_0_17_2;
                        LOCSTORE(store, 8, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_17_0 + WPtempy * x_2_17_1 + ABtemp * (x_0_17_0 - CDcom * x_0_17_1);
                        QUICKDouble x_3_17_0 = Ptempz * x_0_17_0 + WPtempz * x_0_17_1;
                        QUICKDouble x_3_17_1 = Ptempz * x_0_17_1 + WPtempz * x_0_17_2;
                        LOCSTORE(store, 9, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_17_0 + WPtempz * x_3_17_1 + ABtemp * (x_0_17_0 - CDcom * x_0_17_1);
                        LOCSTORE(store, 5, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_17_0 + WPtempy * x_3_17_1;
                        QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
                        QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2.000000 * CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
                        QUICKDouble x_0_18_2 = Qtempy * x_0_8_2 + WQtempy * x_0_8_3 + 2.000000 * CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
                        QUICKDouble x_1_18_0 = Ptempx * x_0_18_0 + WPtempx * x_0_18_1;
                        QUICKDouble x_1_18_1 = Ptempx * x_0_18_1 + WPtempx * x_0_18_2;
                        LOCSTORE(store, 7, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_18_0 + WPtempx * x_1_18_1 + ABtemp * (x_0_18_0 - CDcom * x_0_18_1);
                        QUICKDouble x_2_18_0 = Ptempy * x_0_18_0 + WPtempy * x_0_18_1 + 3.000000 * ABCDtemp * x_0_8_1;
                        QUICKDouble x_2_18_1 = Ptempy * x_0_18_1 + WPtempy * x_0_18_2 + 3.000000 * ABCDtemp * x_0_8_2;
                        LOCSTORE(store, 4, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_18_0 + WPtempx * x_2_18_1;
                        QUICKDouble x_3_18_0 = Ptempz * x_0_18_0 + WPtempz * x_0_18_1;
                        QUICKDouble x_3_18_1 = Ptempz * x_0_18_1 + WPtempz * x_0_18_2;
                        LOCSTORE(store, 9, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_18_0 + WPtempz * x_3_18_1 + ABtemp * (x_0_18_0 - CDcom * x_0_18_1);
                        LOCSTORE(store, 6, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_18_0 + WPtempx * x_3_18_1;
                        QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
                        QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2.000000 * CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
                        QUICKDouble x_0_19_2 = Qtempz * x_0_9_2 + WQtempz * x_0_9_3 + 2.000000 * CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
                        QUICKDouble x_1_19_0 = Ptempx * x_0_19_0 + WPtempx * x_0_19_1;
                        QUICKDouble x_1_19_1 = Ptempx * x_0_19_1 + WPtempx * x_0_19_2;
                        LOCSTORE(store, 7, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_19_0 + WPtempx * x_1_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1);
                        QUICKDouble x_2_19_0 = Ptempy * x_0_19_0 + WPtempy * x_0_19_1;
                        QUICKDouble x_2_19_1 = Ptempy * x_0_19_1 + WPtempy * x_0_19_2;
                        LOCSTORE(store, 8, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_19_0 + WPtempy * x_2_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1);
                        LOCSTORE(store, 4, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_19_0 + WPtempx * x_2_19_1;
                        QUICKDouble x_3_19_0 = Ptempz * x_0_19_0 + WPtempz * x_0_19_1 + 3.000000 * ABCDtemp * x_0_9_1;
                        QUICKDouble x_3_19_1 = Ptempz * x_0_19_1 + WPtempz * x_0_19_2 + 3.000000 * ABCDtemp * x_0_9_2;
                        LOCSTORE(store, 6, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_19_0 + WPtempx * x_3_19_1;
                        LOCSTORE(store, 5, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_19_0 + WPtempy * x_3_19_1;
                        QUICKDouble x_1_11_0 = Ptempx * x_0_11_0 + WPtempx * x_0_11_1 + 2.000000 * ABCDtemp * x_0_4_1;
                        QUICKDouble x_1_11_1 = Ptempx * x_0_11_1 + WPtempx * x_0_11_2 + 2.000000 * ABCDtemp * x_0_4_2;
                        QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                        LOCSTORE(store, 7, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_11_0 + WPtempx * x_1_11_1 + ABtemp * (x_0_11_0 - CDcom * x_0_11_1) + 2.000000 * ABCDtemp * x_1_4_1;
                        QUICKDouble x_2_12_0 = Ptempy * x_0_12_0 + WPtempy * x_0_12_1 + 2.000000 * ABCDtemp * x_0_4_1;
                        QUICKDouble x_2_12_1 = Ptempy * x_0_12_1 + WPtempy * x_0_12_2 + 2.000000 * ABCDtemp * x_0_4_2;
                        QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                        LOCSTORE(store, 8, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_12_0 + WPtempy * x_2_12_1 + ABtemp * (x_0_12_0 - CDcom * x_0_12_1) + 2.000000 * ABCDtemp * x_2_4_1;
                        QUICKDouble x_2_11_0 = Ptempy * x_0_11_0 + WPtempy * x_0_11_1 + ABCDtemp * x_0_7_1;
                        QUICKDouble x_2_11_1 = Ptempy * x_0_11_1 + WPtempy * x_0_11_2 + ABCDtemp * x_0_7_2;
                        LOCSTORE(store, 4, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_11_0 + WPtempx * x_2_11_1 + 2.000000 * ABCDtemp * x_2_4_1;
                        QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
                        LOCSTORE(store, 5, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_12_0 + WPtempy * x_3_12_1 + 2.000000 * ABCDtemp * x_3_4_1;
                        LOCSTORE(store, 6, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_11_0 + WPtempx * x_3_11_1 + 2.000000 * ABCDtemp * x_3_4_1;
                        QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
                        QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
                        QUICKDouble x_0_10_2 = Qtempx * x_0_5_2 + WQtempx * x_0_5_3;
                        QUICKDouble x_3_10_0 = Ptempz * x_0_10_0 + WPtempz * x_0_10_1 + ABCDtemp * x_0_4_1;
                        QUICKDouble x_3_10_1 = Ptempz * x_0_10_1 + WPtempz * x_0_10_2 + ABCDtemp * x_0_4_2;
                        LOCSTORE(store, 9, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_10_0 + WPtempz * x_3_10_1 + ABtemp * (x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_3_4_1;
                        QUICKDouble x_1_10_0 = Ptempx * x_0_10_0 + WPtempx * x_0_10_1 + ABCDtemp * x_0_5_1;
                        QUICKDouble x_1_10_1 = Ptempx * x_0_10_1 + WPtempx * x_0_10_2 + ABCDtemp * x_0_5_2;
                        QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                        LOCSTORE(store, 7, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_10_0 + WPtempx * x_1_10_1 + ABtemp * (x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_1_5_1;
                        QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                        LOCSTORE(store, 8, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_15_0 + WPtempy * x_2_15_1 + ABtemp * (x_0_15_0 - CDcom * x_0_15_1) + 2.000000 * ABCDtemp * x_2_5_1;
                        QUICKDouble x_2_10_0 = Ptempy * x_0_10_0 + WPtempy * x_0_10_1 + ABCDtemp * x_0_6_1;
                        QUICKDouble x_2_10_1 = Ptempy * x_0_10_1 + WPtempy * x_0_10_2 + ABCDtemp * x_0_6_2;
                        LOCSTORE(store, 4, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_10_0 + WPtempx * x_2_10_1 + ABCDtemp * x_2_5_1;
                        QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                        LOCSTORE(store, 9, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_16_0 + WPtempz * x_3_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1) + 2.000000 * ABCDtemp * x_3_5_1;
                        LOCSTORE(store, 5, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_15_0 + WPtempy * x_3_15_1 + 2.000000 * ABCDtemp * x_3_5_1;
                        LOCSTORE(store, 6, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_10_0 + WPtempx * x_3_10_1 + ABCDtemp * x_3_5_1;
                        QUICKDouble x_1_13_0 = Ptempx * x_0_13_0 + WPtempx * x_0_13_1 + 2.000000 * ABCDtemp * x_0_6_1;
                        QUICKDouble x_1_13_1 = Ptempx * x_0_13_1 + WPtempx * x_0_13_2 + 2.000000 * ABCDtemp * x_0_6_2;
                        QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                        LOCSTORE(store, 7, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_13_0 + WPtempx * x_1_13_1 + ABtemp * (x_0_13_0 - CDcom * x_0_13_1) + 2.000000 * ABCDtemp * x_1_6_1;
                        QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                        LOCSTORE(store, 4, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_13_0 + WPtempx * x_2_13_1 + 2.000000 * ABCDtemp * x_2_6_1;
                        LOCSTORE(store, 8, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_10_0 + WPtempy * x_2_10_1 + ABtemp * (x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_2_6_1;
                        QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                        LOCSTORE(store, 9, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_14_0 + WPtempz * x_3_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1) + 2.000000 * ABCDtemp * x_3_6_1;
                        LOCSTORE(store, 6, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_13_0 + WPtempx * x_3_13_1 + 2.000000 * ABCDtemp * x_3_6_1;
                        LOCSTORE(store, 5, 10, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_10_0 + WPtempy * x_3_10_1 + ABCDtemp * x_3_6_1;
                        QUICKDouble x_1_17_0 = Ptempx * x_0_17_0 + WPtempx * x_0_17_1 + 3.000000 * ABCDtemp * x_0_7_1;
                        QUICKDouble x_1_17_1 = Ptempx * x_0_17_1 + WPtempx * x_0_17_2 + 3.000000 * ABCDtemp * x_0_7_2;
                        QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + 2.000000 * ABCDtemp * x_0_1_2;
                        LOCSTORE(store, 7, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_17_0 + WPtempx * x_1_17_1 + ABtemp * (x_0_17_0 - CDcom * x_0_17_1) + 3.000000 * ABCDtemp * x_1_7_1;
                        QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                        LOCSTORE(store, 4, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_17_0 + WPtempx * x_2_17_1 + 3.000000 * ABCDtemp * x_2_7_1;
                        LOCSTORE(store, 8, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_11_0 + WPtempy * x_2_11_1 + ABtemp * (x_0_11_0 - CDcom * x_0_11_1) + ABCDtemp * x_2_7_1;
                        QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                        LOCSTORE(store, 6, 17, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_17_0 + WPtempx * x_3_17_1 + 3.000000 * ABCDtemp * x_3_7_1;
                        LOCSTORE(store, 9, 13, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_13_0 + WPtempz * x_3_13_1 + ABtemp * (x_0_13_0 - CDcom * x_0_13_1) + ABCDtemp * x_3_7_1;
                        LOCSTORE(store, 5, 11, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_11_0 + WPtempy * x_3_11_1 + ABCDtemp * x_3_7_1;
                        QUICKDouble x_1_12_0 = Ptempx * x_0_12_0 + WPtempx * x_0_12_1 + ABCDtemp * x_0_8_1;
                        QUICKDouble x_1_12_1 = Ptempx * x_0_12_1 + WPtempx * x_0_12_2 + ABCDtemp * x_0_8_2;
                        QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                        LOCSTORE(store, 7, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_12_0 + WPtempx * x_1_12_1 + ABtemp * (x_0_12_0 - CDcom * x_0_12_1) + ABCDtemp * x_1_8_1;
                        QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + 2.000000 * ABCDtemp * x_0_2_2;
                        LOCSTORE(store, 8, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_18_0 + WPtempy * x_2_18_1 + ABtemp * (x_0_18_0 - CDcom * x_0_18_1) + 3.000000 * ABCDtemp * x_2_8_1;
                        LOCSTORE(store, 4, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_12_0 + WPtempx * x_2_12_1 + ABCDtemp * x_2_8_1;
                        QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                        LOCSTORE(store, 5, 18, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_18_0 + WPtempy * x_3_18_1 + 3.000000 * ABCDtemp * x_3_8_1;
                        LOCSTORE(store, 9, 15, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_15_0 + WPtempz * x_3_15_1 + ABtemp * (x_0_15_0 - CDcom * x_0_15_1) + ABCDtemp * x_3_8_1;
                        LOCSTORE(store, 6, 12, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_12_0 + WPtempx * x_3_12_1 + ABCDtemp * x_3_8_1;
                        QUICKDouble x_1_14_0 = Ptempx * x_0_14_0 + WPtempx * x_0_14_1 + ABCDtemp * x_0_9_1;
                        QUICKDouble x_1_14_1 = Ptempx * x_0_14_1 + WPtempx * x_0_14_2 + ABCDtemp * x_0_9_2;
                        QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                        LOCSTORE(store, 7, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_14_0 + WPtempx * x_1_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1) + ABCDtemp * x_1_9_1;
                        QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                        LOCSTORE(store, 8, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_16_0 + WPtempy * x_2_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1) + ABCDtemp * x_2_9_1;
                        LOCSTORE(store, 4, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_14_0 + WPtempx * x_2_14_1 + ABCDtemp * x_2_9_1;
                        QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + 2.000000 * ABCDtemp * x_0_3_2;
                        LOCSTORE(store, 9, 19, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_19_0 + WPtempz * x_3_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1) + 3.000000 * ABCDtemp * x_3_9_1;
                        LOCSTORE(store, 5, 16, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_16_0 + WPtempy * x_3_16_1 + ABCDtemp * x_3_9_1;
                        LOCSTORE(store, 6, 14, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_14_0 + WPtempx * x_3_14_1 + ABCDtemp * x_3_9_1;
                        // [DS|FS] integral - End 

                   }
                   if ((I+J) >=  3 && (K+L)>= 2)
                   {

                        // [FS|DS] integral - Start
                        QUICKDouble VY_2 = VY(0, 0, 2);
                        QUICKDouble VY_3 = VY(0, 0, 3);
                        QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
                        QUICKDouble VY_1 = VY(0, 0, 1);
                        QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                        QUICKDouble VY_0 = VY(0, 0, 0);
                        QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                        QUICKDouble VY_4 = VY(0, 0, 4);
                        QUICKDouble x_3_0_3 = Ptempz * VY_3 + WPtempz * VY_4;
                        QUICKDouble VY_5 = VY(0, 0, 5);
                        QUICKDouble x_3_0_4 = Ptempz * VY_4 + WPtempz * VY_5;
                        QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                        QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                        QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
                        QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
                        QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
                        QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
                        QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
                        QUICKDouble x_5_1_1 = Qtempx * x_5_0_1 + WQtempx * x_5_0_2;
                        QUICKDouble x_10_1_0 = Qtempx * x_10_0_0 + WQtempx * x_10_0_1 + ABCDtemp * x_5_0_1;
                        QUICKDouble x_10_1_1 = Qtempx * x_10_0_1 + WQtempx * x_10_0_2 + ABCDtemp * x_5_0_2;
                        LOCSTORE(store, 10, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_10_1_0 + WQtempx * x_10_1_1 + CDtemp * (x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_5_1_1;
                        QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                        QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
                        QUICKDouble x_6_2_1 = Qtempy * x_6_0_1 + WQtempy * x_6_0_2;
                        QUICKDouble x_10_2_0 = Qtempy * x_10_0_0 + WQtempy * x_10_0_1 + ABCDtemp * x_6_0_1;
                        QUICKDouble x_10_2_1 = Qtempy * x_10_0_1 + WQtempy * x_10_0_2 + ABCDtemp * x_6_0_2;
                        LOCSTORE(store, 10, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_10_2_0 + WQtempy * x_10_2_1 + CDtemp * (x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_6_2_1;
                        QUICKDouble x_5_2_1 = Qtempy * x_5_0_1 + WQtempy * x_5_0_2 + ABCDtemp * x_3_0_2;
                        LOCSTORE(store, 10, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_10_2_0 + WQtempx * x_10_2_1 + ABCDtemp * x_5_2_1;
                        QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
                        QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
                        QUICKDouble x_2_0_3 = Ptempy * VY_3 + WPtempy * VY_4;
                        QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
                        QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
                        QUICKDouble x_4_3_1 = Qtempz * x_4_0_1 + WQtempz * x_4_0_2;
                        QUICKDouble x_10_3_0 = Qtempz * x_10_0_0 + WQtempz * x_10_0_1 + ABCDtemp * x_4_0_1;
                        QUICKDouble x_10_3_1 = Qtempz * x_10_0_1 + WQtempz * x_10_0_2 + ABCDtemp * x_4_0_2;
                        LOCSTORE(store, 10, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_10_3_0 + WQtempz * x_10_3_1 + CDtemp * (x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_4_3_1;
                        QUICKDouble x_5_3_1 = Qtempz * x_5_0_1 + WQtempz * x_5_0_2 + ABCDtemp * x_2_0_2;
                        LOCSTORE(store, 10, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_10_3_0 + WQtempx * x_10_3_1 + ABCDtemp * x_5_3_1;
                        QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
                        QUICKDouble x_6_3_1 = Qtempz * x_6_0_1 + WQtempz * x_6_0_2 + ABCDtemp * x_1_0_2;
                        LOCSTORE(store, 10, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_10_3_0 + WQtempy * x_10_3_1 + ABCDtemp * x_6_3_1;
                        QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
                        QUICKDouble x_2_0_4 = Ptempy * VY_4 + WPtempy * VY_5;
                        QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                        QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
                        QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                        QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
                        QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
                        QUICKDouble x_4_1_1 = Qtempx * x_4_0_1 + WQtempx * x_4_0_2 + ABCDtemp * x_2_0_2;
                        QUICKDouble x_11_1_0 = Qtempx * x_11_0_0 + WQtempx * x_11_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
                        QUICKDouble x_11_1_1 = Qtempx * x_11_0_1 + WQtempx * x_11_0_2 + 2.000000 * ABCDtemp * x_4_0_2;
                        LOCSTORE(store, 11, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_11_1_0 + WQtempx * x_11_1_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1) + 2.000000 * ABCDtemp * x_4_1_1;
                        QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                        QUICKDouble x_1_0_3 = Ptempx * VY_3 + WPtempx * VY_4;
                        QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                        QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                        QUICKDouble x_7_2_1 = Qtempy * x_7_0_1 + WQtempy * x_7_0_2;
                        QUICKDouble x_11_2_0 = Qtempy * x_11_0_0 + WQtempy * x_11_0_1 + ABCDtemp * x_7_0_1;
                        QUICKDouble x_11_2_1 = Qtempy * x_11_0_1 + WQtempy * x_11_0_2 + ABCDtemp * x_7_0_2;
                        LOCSTORE(store, 11, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_11_2_0 + WQtempy * x_11_2_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1) + ABCDtemp * x_7_2_1;
                        QUICKDouble x_4_2_1 = Qtempy * x_4_0_1 + WQtempy * x_4_0_2 + ABCDtemp * x_1_0_2;
                        LOCSTORE(store, 11, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_11_2_0 + WQtempx * x_11_2_1 + 2.000000 * ABCDtemp * x_4_2_1;
                        QUICKDouble x_11_3_0 = Qtempz * x_11_0_0 + WQtempz * x_11_0_1;
                        QUICKDouble x_11_3_1 = Qtempz * x_11_0_1 + WQtempz * x_11_0_2;
                        LOCSTORE(store, 11, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_11_3_0 + WQtempz * x_11_3_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1);
                        LOCSTORE(store, 11, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_11_3_0 + WQtempx * x_11_3_1 + 2.000000 * ABCDtemp * x_4_3_1;
                        QUICKDouble x_7_3_1 = Qtempz * x_7_0_1 + WQtempz * x_7_0_2;
                        LOCSTORE(store, 11, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_11_3_0 + WQtempy * x_11_3_1 + ABCDtemp * x_7_3_1;
                        QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                        QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                        QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                        QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
                        QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
                        QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
                        QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
                        QUICKDouble x_8_1_1 = Qtempx * x_8_0_1 + WQtempx * x_8_0_2;
                        QUICKDouble x_12_1_0 = Qtempx * x_12_0_0 + WQtempx * x_12_0_1 + ABCDtemp * x_8_0_1;
                        QUICKDouble x_12_1_1 = Qtempx * x_12_0_1 + WQtempx * x_12_0_2 + ABCDtemp * x_8_0_2;
                        LOCSTORE(store, 12, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_12_1_0 + WQtempx * x_12_1_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1) + ABCDtemp * x_8_1_1;
                        QUICKDouble x_12_2_0 = Qtempy * x_12_0_0 + WQtempy * x_12_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
                        QUICKDouble x_12_2_1 = Qtempy * x_12_0_1 + WQtempy * x_12_0_2 + 2.000000 * ABCDtemp * x_4_0_2;
                        LOCSTORE(store, 12, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_12_2_0 + WQtempy * x_12_2_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1) + 2.000000 * ABCDtemp * x_4_2_1;
                        QUICKDouble x_8_2_1 = Qtempy * x_8_0_1 + WQtempy * x_8_0_2 + 2.000000 * ABCDtemp * x_2_0_2;
                        LOCSTORE(store, 12, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_12_2_0 + WQtempx * x_12_2_1 + ABCDtemp * x_8_2_1;
                        QUICKDouble x_12_3_0 = Qtempz * x_12_0_0 + WQtempz * x_12_0_1;
                        QUICKDouble x_12_3_1 = Qtempz * x_12_0_1 + WQtempz * x_12_0_2;
                        LOCSTORE(store, 12, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_12_3_0 + WQtempz * x_12_3_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1);
                        QUICKDouble x_8_3_1 = Qtempz * x_8_0_1 + WQtempz * x_8_0_2;
                        LOCSTORE(store, 12, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_12_3_0 + WQtempx * x_12_3_1 + ABCDtemp * x_8_3_1;
                        LOCSTORE(store, 12, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_12_3_0 + WQtempy * x_12_3_1 + 2.000000 * ABCDtemp * x_4_3_1;
                        QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                        QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
                        QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                        QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                        QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
                        QUICKDouble x_6_1_1 = Qtempx * x_6_0_1 + WQtempx * x_6_0_2 + ABCDtemp * x_3_0_2;
                        QUICKDouble x_13_1_0 = Qtempx * x_13_0_0 + WQtempx * x_13_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
                        QUICKDouble x_13_1_1 = Qtempx * x_13_0_1 + WQtempx * x_13_0_2 + 2.000000 * ABCDtemp * x_6_0_2;
                        LOCSTORE(store, 13, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_13_1_0 + WQtempx * x_13_1_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1) + 2.000000 * ABCDtemp * x_6_1_1;
                        QUICKDouble x_13_2_0 = Qtempy * x_13_0_0 + WQtempy * x_13_0_1;
                        QUICKDouble x_13_2_1 = Qtempy * x_13_0_1 + WQtempy * x_13_0_2;
                        LOCSTORE(store, 13, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_13_2_0 + WQtempy * x_13_2_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1);
                        LOCSTORE(store, 13, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_13_2_0 + WQtempx * x_13_2_1 + 2.000000 * ABCDtemp * x_6_2_1;
                        QUICKDouble x_13_3_0 = Qtempz * x_13_0_0 + WQtempz * x_13_0_1 + ABCDtemp * x_7_0_1;
                        QUICKDouble x_13_3_1 = Qtempz * x_13_0_1 + WQtempz * x_13_0_2 + ABCDtemp * x_7_0_2;
                        LOCSTORE(store, 13, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_13_3_0 + WQtempz * x_13_3_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1) + ABCDtemp * x_7_3_1;
                        LOCSTORE(store, 13, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_13_3_0 + WQtempx * x_13_3_1 + 2.000000 * ABCDtemp * x_6_3_1;
                        LOCSTORE(store, 13, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_13_3_0 + WQtempy * x_13_3_1;
                        QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                        QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                        QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                        QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
                        QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
                        QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
                        QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
                        QUICKDouble x_9_1_1 = Qtempx * x_9_0_1 + WQtempx * x_9_0_2;
                        QUICKDouble x_14_1_0 = Qtempx * x_14_0_0 + WQtempx * x_14_0_1 + ABCDtemp * x_9_0_1;
                        QUICKDouble x_14_1_1 = Qtempx * x_14_0_1 + WQtempx * x_14_0_2 + ABCDtemp * x_9_0_2;
                        LOCSTORE(store, 14, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_14_1_0 + WQtempx * x_14_1_1 + CDtemp * (x_14_0_0 - ABcom * x_14_0_1) + ABCDtemp * x_9_1_1;
                        QUICKDouble x_14_2_0 = Qtempy * x_14_0_0 + WQtempy * x_14_0_1;
                        QUICKDouble x_14_2_1 = Qtempy * x_14_0_1 + WQtempy * x_14_0_2;
                        LOCSTORE(store, 14, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_14_2_0 + WQtempy * x_14_2_1 + CDtemp * (x_14_0_0 - ABcom * x_14_0_1);
                        QUICKDouble x_9_2_1 = Qtempy * x_9_0_1 + WQtempy * x_9_0_2;
                        LOCSTORE(store, 14, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_14_2_0 + WQtempx * x_14_2_1 + ABCDtemp * x_9_2_1;
                        QUICKDouble x_14_3_0 = Qtempz * x_14_0_0 + WQtempz * x_14_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
                        QUICKDouble x_14_3_1 = Qtempz * x_14_0_1 + WQtempz * x_14_0_2 + 2.000000 * ABCDtemp * x_6_0_2;
                        LOCSTORE(store, 14, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_14_3_0 + WQtempz * x_14_3_1 + CDtemp * (x_14_0_0 - ABcom * x_14_0_1) + 2.000000 * ABCDtemp * x_6_3_1;
                        QUICKDouble x_9_3_1 = Qtempz * x_9_0_1 + WQtempz * x_9_0_2 + 2.000000 * ABCDtemp * x_3_0_2;
                        LOCSTORE(store, 14, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_14_3_0 + WQtempx * x_14_3_1 + ABCDtemp * x_9_3_1;
                        LOCSTORE(store, 14, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_14_3_0 + WQtempy * x_14_3_1;
                        QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                        QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                        QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
                        QUICKDouble x_15_1_0 = Qtempx * x_15_0_0 + WQtempx * x_15_0_1;
                        QUICKDouble x_15_1_1 = Qtempx * x_15_0_1 + WQtempx * x_15_0_2;
                        LOCSTORE(store, 15, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_15_1_0 + WQtempx * x_15_1_1 + CDtemp * (x_15_0_0 - ABcom * x_15_0_1);
                        QUICKDouble x_15_2_0 = Qtempy * x_15_0_0 + WQtempy * x_15_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
                        QUICKDouble x_15_2_1 = Qtempy * x_15_0_1 + WQtempy * x_15_0_2 + 2.000000 * ABCDtemp * x_5_0_2;
                        LOCSTORE(store, 15, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_15_2_0 + WQtempy * x_15_2_1 + CDtemp * (x_15_0_0 - ABcom * x_15_0_1) + 2.000000 * ABCDtemp * x_5_2_1;
                        LOCSTORE(store, 15, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_15_2_0 + WQtempx * x_15_2_1;
                        QUICKDouble x_15_3_0 = Qtempz * x_15_0_0 + WQtempz * x_15_0_1 + ABCDtemp * x_8_0_1;
                        QUICKDouble x_15_3_1 = Qtempz * x_15_0_1 + WQtempz * x_15_0_2 + ABCDtemp * x_8_0_2;
                        LOCSTORE(store, 15, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_15_3_0 + WQtempz * x_15_3_1 + CDtemp * (x_15_0_0 - ABcom * x_15_0_1) + ABCDtemp * x_8_3_1;
                        LOCSTORE(store, 15, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_15_3_0 + WQtempx * x_15_3_1;
                        LOCSTORE(store, 15, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_15_3_0 + WQtempy * x_15_3_1 + 2.000000 * ABCDtemp * x_5_3_1;
                        QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
                        QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
                        QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
                        QUICKDouble x_16_1_0 = Qtempx * x_16_0_0 + WQtempx * x_16_0_1;
                        QUICKDouble x_16_1_1 = Qtempx * x_16_0_1 + WQtempx * x_16_0_2;
                        LOCSTORE(store, 16, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_16_1_0 + WQtempx * x_16_1_1 + CDtemp * (x_16_0_0 - ABcom * x_16_0_1);
                        QUICKDouble x_16_2_0 = Qtempy * x_16_0_0 + WQtempy * x_16_0_1 + ABCDtemp * x_9_0_1;
                        QUICKDouble x_16_2_1 = Qtempy * x_16_0_1 + WQtempy * x_16_0_2 + ABCDtemp * x_9_0_2;
                        LOCSTORE(store, 16, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_16_2_0 + WQtempy * x_16_2_1 + CDtemp * (x_16_0_0 - ABcom * x_16_0_1) + ABCDtemp * x_9_2_1;
                        LOCSTORE(store, 16, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_16_2_0 + WQtempx * x_16_2_1;
                        QUICKDouble x_16_3_0 = Qtempz * x_16_0_0 + WQtempz * x_16_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
                        QUICKDouble x_16_3_1 = Qtempz * x_16_0_1 + WQtempz * x_16_0_2 + 2.000000 * ABCDtemp * x_5_0_2;
                        LOCSTORE(store, 16, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_16_3_0 + WQtempz * x_16_3_1 + CDtemp * (x_16_0_0 - ABcom * x_16_0_1) + 2.000000 * ABCDtemp * x_5_3_1;
                        LOCSTORE(store, 16, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_16_3_0 + WQtempx * x_16_3_1;
                        LOCSTORE(store, 16, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_16_3_0 + WQtempy * x_16_3_1 + ABCDtemp * x_9_3_1;
                        QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                        QUICKDouble x_1_0_4 = Ptempx * VY_4 + WPtempx * VY_5;
                        QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                        QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
                        QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
                        QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 + 2.000000 * ABtemp * (x_1_0_1 - CDcom * x_1_0_2);
                        QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 + 2.000000 * ABtemp * (x_1_0_2 - CDcom * x_1_0_3);
                        QUICKDouble x_7_1_1 = Qtempx * x_7_0_1 + WQtempx * x_7_0_2 + 2.000000 * ABCDtemp * x_1_0_2;
                        QUICKDouble x_17_1_0 = Qtempx * x_17_0_0 + WQtempx * x_17_0_1 + 3.000000 * ABCDtemp * x_7_0_1;
                        QUICKDouble x_17_1_1 = Qtempx * x_17_0_1 + WQtempx * x_17_0_2 + 3.000000 * ABCDtemp * x_7_0_2;
                        LOCSTORE(store, 17, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_17_1_0 + WQtempx * x_17_1_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1) + 3.000000 * ABCDtemp * x_7_1_1;
                        QUICKDouble x_17_2_0 = Qtempy * x_17_0_0 + WQtempy * x_17_0_1;
                        QUICKDouble x_17_2_1 = Qtempy * x_17_0_1 + WQtempy * x_17_0_2;
                        LOCSTORE(store, 17, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_17_2_0 + WQtempy * x_17_2_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1);
                        LOCSTORE(store, 17, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_17_2_0 + WQtempx * x_17_2_1 + 3.000000 * ABCDtemp * x_7_2_1;
                        QUICKDouble x_17_3_0 = Qtempz * x_17_0_0 + WQtempz * x_17_0_1;
                        QUICKDouble x_17_3_1 = Qtempz * x_17_0_1 + WQtempz * x_17_0_2;
                        LOCSTORE(store, 17, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_17_3_0 + WQtempz * x_17_3_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1);
                        LOCSTORE(store, 17, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_17_3_0 + WQtempx * x_17_3_1 + 3.000000 * ABCDtemp * x_7_3_1;
                        LOCSTORE(store, 17, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_17_3_0 + WQtempy * x_17_3_1;
                        QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                        QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 + 2.000000 * ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
                        QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 + 2.000000 * ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
                        QUICKDouble x_18_1_0 = Qtempx * x_18_0_0 + WQtempx * x_18_0_1;
                        QUICKDouble x_18_1_1 = Qtempx * x_18_0_1 + WQtempx * x_18_0_2;
                        LOCSTORE(store, 18, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_18_1_0 + WQtempx * x_18_1_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1);
                        QUICKDouble x_18_2_0 = Qtempy * x_18_0_0 + WQtempy * x_18_0_1 + 3.000000 * ABCDtemp * x_8_0_1;
                        QUICKDouble x_18_2_1 = Qtempy * x_18_0_1 + WQtempy * x_18_0_2 + 3.000000 * ABCDtemp * x_8_0_2;
                        LOCSTORE(store, 18, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_18_2_0 + WQtempy * x_18_2_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1) + 3.000000 * ABCDtemp * x_8_2_1;
                        LOCSTORE(store, 18, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_18_2_0 + WQtempx * x_18_2_1;
                        QUICKDouble x_18_3_0 = Qtempz * x_18_0_0 + WQtempz * x_18_0_1;
                        QUICKDouble x_18_3_1 = Qtempz * x_18_0_1 + WQtempz * x_18_0_2;
                        LOCSTORE(store, 18, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_18_3_0 + WQtempz * x_18_3_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1);
                        LOCSTORE(store, 18, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_18_3_0 + WQtempx * x_18_3_1;
                        LOCSTORE(store, 18, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_18_3_0 + WQtempy * x_18_3_1 + 3.000000 * ABCDtemp * x_8_3_1;
                        QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                        QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 + 2.000000 * ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                        QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 + 2.000000 * ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
                        QUICKDouble x_19_1_0 = Qtempx * x_19_0_0 + WQtempx * x_19_0_1;
                        QUICKDouble x_19_1_1 = Qtempx * x_19_0_1 + WQtempx * x_19_0_2;
                        LOCSTORE(store, 19, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_19_1_0 + WQtempx * x_19_1_1 + CDtemp * (x_19_0_0 - ABcom * x_19_0_1);
                        QUICKDouble x_19_2_0 = Qtempy * x_19_0_0 + WQtempy * x_19_0_1;
                        QUICKDouble x_19_2_1 = Qtempy * x_19_0_1 + WQtempy * x_19_0_2;
                        LOCSTORE(store, 19, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_19_2_0 + WQtempy * x_19_2_1 + CDtemp * (x_19_0_0 - ABcom * x_19_0_1);
                        LOCSTORE(store, 19, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_19_2_0 + WQtempx * x_19_2_1;
                        QUICKDouble x_19_3_0 = Qtempz * x_19_0_0 + WQtempz * x_19_0_1 + 3.000000 * ABCDtemp * x_9_0_1;
                        QUICKDouble x_19_3_1 = Qtempz * x_19_0_1 + WQtempz * x_19_0_2 + 3.000000 * ABCDtemp * x_9_0_2;
                        LOCSTORE(store, 19, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_19_3_0 + WQtempz * x_19_3_1 + CDtemp * (x_19_0_0 - ABcom * x_19_0_1) + 3.000000 * ABCDtemp * x_9_3_1;
                        LOCSTORE(store, 19, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_19_3_0 + WQtempx * x_19_3_1;
                        LOCSTORE(store, 19, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_19_3_0 + WQtempy * x_19_3_1;
                        // [FS|DS] integral - End 

                   }
               }
               if ((I+J) >=  3 && (K+L)>= 1)
               {

                    // [FS|PS] integral - Start
                    QUICKDouble VY_1 = VY(0, 0, 1);
                    QUICKDouble VY_2 = VY(0, 0, 2);
                    QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                    QUICKDouble VY_3 = VY(0, 0, 3);
                    QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
                    QUICKDouble VY_0 = VY(0, 0, 0);
                    QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                    QUICKDouble VY_4 = VY(0, 0, 4);
                    QUICKDouble x_3_0_3 = Ptempz * VY_3 + WPtempz * VY_4;
                    QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
                    QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
                    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                    QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
                    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
                    QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
                    QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
                    LOCSTORE(store, 10, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_10_0_0 + WQtempz * x_10_0_1 + ABCDtemp * x_4_0_1;
                    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                    LOCSTORE(store, 10, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_10_0_0 + WQtempy * x_10_0_1 + ABCDtemp * x_6_0_1;
                    LOCSTORE(store, 10, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_10_0_0 + WQtempx * x_10_0_1 + ABCDtemp * x_5_0_1;
                    QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
                    QUICKDouble x_2_0_3 = Ptempy * VY_3 + WPtempy * VY_4;
                    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                    QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
                    QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                    QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
                    LOCSTORE(store, 11, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_11_0_0 + WQtempz * x_11_0_1;
                    QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                    QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
                    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    LOCSTORE(store, 11, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_11_0_0 + WQtempy * x_11_0_1 + ABCDtemp * x_7_0_1;
                    LOCSTORE(store, 11, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_11_0_0 + WQtempx * x_11_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
                    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                    QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
                    QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
                    LOCSTORE(store, 12, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_12_0_0 + WQtempz * x_12_0_1;
                    LOCSTORE(store, 12, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_12_0_0 + WQtempy * x_12_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
                    LOCSTORE(store, 12, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_12_0_0 + WQtempx * x_12_0_1 + ABCDtemp * x_8_0_1;
                    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                    QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
                    QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                    QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                    LOCSTORE(store, 13, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_13_0_0 + WQtempz * x_13_0_1 + ABCDtemp * x_7_0_1;
                    LOCSTORE(store, 13, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_13_0_0 + WQtempy * x_13_0_1;
                    LOCSTORE(store, 13, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_13_0_0 + WQtempx * x_13_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
                    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                    QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
                    QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
                    LOCSTORE(store, 14, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_14_0_0 + WQtempz * x_14_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
                    LOCSTORE(store, 14, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_14_0_0 + WQtempy * x_14_0_1;
                    LOCSTORE(store, 14, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_14_0_0 + WQtempx * x_14_0_1 + ABCDtemp * x_9_0_1;
                    QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                    QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                    LOCSTORE(store, 15, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_15_0_0 + WQtempz * x_15_0_1 + ABCDtemp * x_8_0_1;
                    LOCSTORE(store, 15, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_15_0_0 + WQtempy * x_15_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
                    LOCSTORE(store, 15, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_15_0_0 + WQtempx * x_15_0_1;
                    QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
                    QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
                    LOCSTORE(store, 16, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_16_0_0 + WQtempz * x_16_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
                    LOCSTORE(store, 16, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_16_0_0 + WQtempy * x_16_0_1 + ABCDtemp * x_9_0_1;
                    LOCSTORE(store, 16, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_16_0_0 + WQtempx * x_16_0_1;
                    QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                    QUICKDouble x_1_0_3 = Ptempx * VY_3 + WPtempx * VY_4;
                    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                    QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
                    QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
                    QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 + 2.000000 * ABtemp * (x_1_0_1 - CDcom * x_1_0_2);
                    LOCSTORE(store, 17, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_17_0_0 + WQtempz * x_17_0_1;
                    LOCSTORE(store, 17, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_17_0_0 + WQtempy * x_17_0_1;
                    LOCSTORE(store, 17, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_17_0_0 + WQtempx * x_17_0_1 + 3.000000 * ABCDtemp * x_7_0_1;
                    QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                    QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 + 2.000000 * ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
                    LOCSTORE(store, 18, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_18_0_0 + WQtempz * x_18_0_1;
                    LOCSTORE(store, 18, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_18_0_0 + WQtempy * x_18_0_1 + 3.000000 * ABCDtemp * x_8_0_1;
                    LOCSTORE(store, 18, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_18_0_0 + WQtempx * x_18_0_1;
                    QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                    QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 + 2.000000 * ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
                    LOCSTORE(store, 19, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_19_0_0 + WQtempz * x_19_0_1 + 3.000000 * ABCDtemp * x_9_0_1;
                    LOCSTORE(store, 19, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_19_0_0 + WQtempy * x_19_0_1;
                    LOCSTORE(store, 19, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_19_0_0 + WQtempx * x_19_0_1;
                    // [FS|PS] integral - End 

               }
           }
       }
       if ((I+J) >=  2 && (K+L)>= 0)
       {

            // [DS|SS] integral - Start
            QUICKDouble VY_0 = VY(0, 0, 0);
            QUICKDouble VY_1 = VY(0, 0, 1);
            QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
            QUICKDouble VY_2 = VY(0, 0, 2);
            QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
            LOCSTORE(store, 7, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
            QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
            QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
            LOCSTORE(store, 8, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
            LOCSTORE(store, 4, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
            QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
            QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
            LOCSTORE(store, 9, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
            LOCSTORE(store, 6, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
            LOCSTORE(store, 5, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
            // [DS|SS] integral - End 

           if ((I+J) >=  3 && (K+L)>= 0)
           {

                // [FS|SS] integral - Start
                QUICKDouble VY_0 = VY(0, 0, 0);
                QUICKDouble VY_1 = VY(0, 0, 1);
                QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
                QUICKDouble VY_2 = VY(0, 0, 2);
                QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
                QUICKDouble VY_3 = VY(0, 0, 3);
                QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
                QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
                LOCSTORE(store, 11, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
                QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
                QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
                QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                LOCSTORE(store, 15, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                LOCSTORE(store, 10, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
                QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                LOCSTORE(store, 13, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
                QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
                QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
                QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 17, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
                QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 18, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
                LOCSTORE(store, 12, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
                QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
                QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
                LOCSTORE(store, 19, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
                LOCSTORE(store, 16, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
                LOCSTORE(store, 14, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
                // [FS|SS] integral - End 

           }
       }
   }

 } 
