/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 03/25/2023                !
 !                                                                     !
 ! Copyright (C) 2023-2024 Merz lab                                    !
 ! Copyright (C) 2023-2024 Götz lab                                    !
 !                                                                     !
 ! This Source Code Form is subject to the terms of the Mozilla Public !
 ! License, v. 2.0. If a copy of the MPL was not distributed with this !
 ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
 !_____________________________________________________________________!
*/

{

    // [SS|HS] integral - Start
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
    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_8_3 = Qtempy * x_0_2_3 + WQtempy * x_0_2_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
    QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
    QUICKDouble x_0_12_2 = Qtempx * x_0_8_2 + WQtempx * x_0_8_3;
    QUICKDouble x_0_20_0 = Qtempx * x_0_12_0 + WQtempx * x_0_12_1 + CDtemp * (x_0_8_0 - ABcom * x_0_8_1);
    QUICKDouble x_0_20_1 = Qtempx * x_0_12_1 + WQtempx * x_0_12_2 + CDtemp * (x_0_8_1 - ABcom * x_0_8_2);
    LOCSTORE(store, 0, 46, STOREDIM, STOREDIM) += Qtempx * x_0_20_0 + WQtempx * x_0_20_1 + 2.000000 * CDtemp * (x_0_12_0 - ABcom * x_0_12_1);
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
    QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
    QUICKDouble x_0_3_4 = Qtempz * VY_4 + WQtempz * VY_5;
    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_9_3 = Qtempz * x_0_3_3 + WQtempz * x_0_3_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
    QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
    QUICKDouble x_0_14_2 = Qtempx * x_0_9_2 + WQtempx * x_0_9_3;
    QUICKDouble x_0_21_0 = Qtempx * x_0_14_0 + WQtempx * x_0_14_1 + CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    QUICKDouble x_0_21_1 = Qtempx * x_0_14_1 + WQtempx * x_0_14_2 + CDtemp * (x_0_9_1 - ABcom * x_0_9_2);
    LOCSTORE(store, 0, 44, STOREDIM, STOREDIM) += Qtempx * x_0_21_0 + WQtempx * x_0_21_1 + 2.000000 * CDtemp * (x_0_14_0 - ABcom * x_0_14_1);
    QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
    QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
    QUICKDouble x_0_16_2 = Qtempy * x_0_9_2 + WQtempy * x_0_9_3;
    QUICKDouble x_0_22_0 = Qtempy * x_0_16_0 + WQtempy * x_0_16_1 + CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    QUICKDouble x_0_22_1 = Qtempy * x_0_16_1 + WQtempy * x_0_16_2 + CDtemp * (x_0_9_1 - ABcom * x_0_9_2);
    LOCSTORE(store, 0, 42, STOREDIM, STOREDIM) += Qtempy * x_0_22_0 + WQtempy * x_0_22_1 + 2.000000 * CDtemp * (x_0_16_0 - ABcom * x_0_16_1);
    LOCSTORE(store, 0, 35, STOREDIM, STOREDIM) += Qtempx * x_0_22_0 + WQtempx * x_0_22_1;
    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
    QUICKDouble x_0_5_3 = Qtempy * x_0_3_3 + WQtempy * x_0_3_4;
    QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
    QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
    QUICKDouble x_0_10_2 = Qtempx * x_0_5_2 + WQtempx * x_0_5_3;
    QUICKDouble x_0_23_0 = Qtempx * x_0_10_0 + WQtempx * x_0_10_1 + CDtemp * (x_0_5_0 - ABcom * x_0_5_1);
    QUICKDouble x_0_23_1 = Qtempx * x_0_10_1 + WQtempx * x_0_10_2 + CDtemp * (x_0_5_1 - ABcom * x_0_5_2);
    LOCSTORE(store, 0, 38, STOREDIM, STOREDIM) += Qtempx * x_0_23_0 + WQtempx * x_0_23_1 + 2.000000 * CDtemp * (x_0_10_0 - ABcom * x_0_10_1);
    QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_0_15_2 = Qtempy * x_0_5_2 + WQtempy * x_0_5_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_0_24_0 = Qtempx * x_0_15_0 + WQtempx * x_0_15_1;
    QUICKDouble x_0_24_1 = Qtempx * x_0_15_1 + WQtempx * x_0_15_2;
    LOCSTORE(store, 0, 37, STOREDIM, STOREDIM) += Qtempx * x_0_24_0 + WQtempx * x_0_24_1 + CDtemp * (x_0_15_0 - ABcom * x_0_15_1);
    QUICKDouble x_0_25_0 = Qtempx * x_0_16_0 + WQtempx * x_0_16_1;
    QUICKDouble x_0_25_1 = Qtempx * x_0_16_1 + WQtempx * x_0_16_2;
    LOCSTORE(store, 0, 36, STOREDIM, STOREDIM) += Qtempx * x_0_25_0 + WQtempx * x_0_25_1 + CDtemp * (x_0_16_0 - ABcom * x_0_16_1);
    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
    QUICKDouble x_0_6_3 = Qtempx * x_0_3_3 + WQtempx * x_0_3_4;
    QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_0_13_2 = Qtempx * x_0_6_2 + WQtempx * x_0_6_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_0_26_0 = Qtempx * x_0_13_0 + WQtempx * x_0_13_1 + 2.000000 * CDtemp * (x_0_6_0 - ABcom * x_0_6_1);
    QUICKDouble x_0_26_1 = Qtempx * x_0_13_1 + WQtempx * x_0_13_2 + 2.000000 * CDtemp * (x_0_6_1 - ABcom * x_0_6_2);
    LOCSTORE(store, 0, 50, STOREDIM, STOREDIM) += Qtempx * x_0_26_0 + WQtempx * x_0_26_1 + 3.000000 * CDtemp * (x_0_13_0 - ABcom * x_0_13_1);
    QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2.000000 * CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_0_19_2 = Qtempz * x_0_9_2 + WQtempz * x_0_9_3 + 2.000000 * CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_0_27_0 = Qtempx * x_0_19_0 + WQtempx * x_0_19_1;
    QUICKDouble x_0_27_1 = Qtempx * x_0_19_1 + WQtempx * x_0_19_2;
    LOCSTORE(store, 0, 43, STOREDIM, STOREDIM) += Qtempx * x_0_27_0 + WQtempx * x_0_27_1 + CDtemp * (x_0_19_0 - ABcom * x_0_19_1);
    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
    QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
    QUICKDouble x_0_4_3 = Qtempx * x_0_2_3 + WQtempx * x_0_2_4;
    QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    QUICKDouble x_0_11_2 = Qtempx * x_0_4_2 + WQtempx * x_0_4_3 + CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
    QUICKDouble x_0_28_0 = Qtempx * x_0_11_0 + WQtempx * x_0_11_1 + 2.000000 * CDtemp * (x_0_4_0 - ABcom * x_0_4_1);
    QUICKDouble x_0_28_1 = Qtempx * x_0_11_1 + WQtempx * x_0_11_2 + 2.000000 * CDtemp * (x_0_4_1 - ABcom * x_0_4_2);
    LOCSTORE(store, 0, 52, STOREDIM, STOREDIM) += Qtempx * x_0_28_0 + WQtempx * x_0_28_1 + 3.000000 * CDtemp * (x_0_11_0 - ABcom * x_0_11_1);
    QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2.000000 * CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    QUICKDouble x_0_18_2 = Qtempy * x_0_8_2 + WQtempy * x_0_8_3 + 2.000000 * CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
    QUICKDouble x_0_29_0 = Qtempx * x_0_18_0 + WQtempx * x_0_18_1;
    QUICKDouble x_0_29_1 = Qtempx * x_0_18_1 + WQtempx * x_0_18_2;
    LOCSTORE(store, 0, 45, STOREDIM, STOREDIM) += Qtempx * x_0_29_0 + WQtempx * x_0_29_1 + CDtemp * (x_0_18_0 - ABcom * x_0_18_1);
    QUICKDouble x_0_30_0 = Qtempy * x_0_15_0 + WQtempy * x_0_15_1 + 2.000000 * CDtemp * (x_0_5_0 - ABcom * x_0_5_1);
    QUICKDouble x_0_30_1 = Qtempy * x_0_15_1 + WQtempy * x_0_15_2 + 2.000000 * CDtemp * (x_0_5_1 - ABcom * x_0_5_2);
    LOCSTORE(store, 0, 48, STOREDIM, STOREDIM) += Qtempy * x_0_30_0 + WQtempy * x_0_30_1 + 3.000000 * CDtemp * (x_0_15_0 - ABcom * x_0_15_1);
    LOCSTORE(store, 0, 39, STOREDIM, STOREDIM) += Qtempx * x_0_30_0 + WQtempx * x_0_30_1;
    QUICKDouble x_0_31_0 = Qtempy * x_0_19_0 + WQtempy * x_0_19_1;
    QUICKDouble x_0_31_1 = Qtempy * x_0_19_1 + WQtempy * x_0_19_2;
    LOCSTORE(store, 0, 41, STOREDIM, STOREDIM) += Qtempy * x_0_31_0 + WQtempy * x_0_31_1 + CDtemp * (x_0_19_0 - ABcom * x_0_19_1);
    LOCSTORE(store, 0, 40, STOREDIM, STOREDIM) += Qtempx * x_0_31_0 + WQtempx * x_0_31_1;
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
    QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
    QUICKDouble x_0_1_4 = Qtempx * VY_4 + WQtempx * VY_5;
    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_7_3 = Qtempx * x_0_1_3 + WQtempx * x_0_1_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
    QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2.000000 * CDtemp * (x_0_1_1 - ABcom * x_0_1_2);
    QUICKDouble x_0_17_2 = Qtempx * x_0_7_2 + WQtempx * x_0_7_3 + 2.000000 * CDtemp * (x_0_1_2 - ABcom * x_0_1_3);
    QUICKDouble x_0_32_0 = Qtempx * x_0_17_0 + WQtempx * x_0_17_1 + 3.000000 * CDtemp * (x_0_7_0 - ABcom * x_0_7_1);
    QUICKDouble x_0_32_1 = Qtempx * x_0_17_1 + WQtempx * x_0_17_2 + 3.000000 * CDtemp * (x_0_7_1 - ABcom * x_0_7_2);
    LOCSTORE(store, 0, 53, STOREDIM, STOREDIM) += Qtempx * x_0_32_0 + WQtempx * x_0_32_1 + 4.000000 * CDtemp * (x_0_17_0 - ABcom * x_0_17_1);
    QUICKDouble x_0_33_0 = Qtempy * x_0_18_0 + WQtempy * x_0_18_1 + 3.000000 * CDtemp * (x_0_8_0 - ABcom * x_0_8_1);
    QUICKDouble x_0_33_1 = Qtempy * x_0_18_1 + WQtempy * x_0_18_2 + 3.000000 * CDtemp * (x_0_8_1 - ABcom * x_0_8_2);
    LOCSTORE(store, 0, 54, STOREDIM, STOREDIM) += Qtempy * x_0_33_0 + WQtempy * x_0_33_1 + 4.000000 * CDtemp * (x_0_18_0 - ABcom * x_0_18_1);
    LOCSTORE(store, 0, 51, STOREDIM, STOREDIM) += Qtempx * x_0_33_0 + WQtempx * x_0_33_1;
    QUICKDouble x_0_34_0 = Qtempz * x_0_19_0 + WQtempz * x_0_19_1 + 3.000000 * CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    QUICKDouble x_0_34_1 = Qtempz * x_0_19_1 + WQtempz * x_0_19_2 + 3.000000 * CDtemp * (x_0_9_1 - ABcom * x_0_9_2);
    LOCSTORE(store, 0, 55, STOREDIM, STOREDIM) += Qtempz * x_0_34_0 + WQtempz * x_0_34_1 + 4.000000 * CDtemp * (x_0_19_0 - ABcom * x_0_19_1);
    LOCSTORE(store, 0, 49, STOREDIM, STOREDIM) += Qtempx * x_0_34_0 + WQtempx * x_0_34_1;
    LOCSTORE(store, 0, 47, STOREDIM, STOREDIM) += Qtempy * x_0_34_0 + WQtempy * x_0_34_1;
    // [SS|HS] integral - End 

}

