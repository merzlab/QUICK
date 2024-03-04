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

    // [DS|FS] integral - Start
    QUICKDouble VY_5 = VY(0, 0, 5);
    QUICKDouble VY_4 = VY(0, 0, 4);
    QUICKDouble x_0_1_4 = Qtempx * VY_4 + WQtempx * VY_5;
    QUICKDouble x_0_2_4 = Qtempy * VY_4 + WQtempy * VY_5;
    QUICKDouble x_0_3_4 = Qtempz * VY_4 + WQtempz * VY_5;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
    QUICKDouble x_0_7_3 = Qtempx * x_0_1_3 + WQtempx * x_0_1_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble x_0_2_3 = Qtempy * VY_3 + WQtempy * VY_4;
    QUICKDouble x_0_4_3 = Qtempx * x_0_2_3 + WQtempx * x_0_2_4;
    QUICKDouble x_0_8_3 = Qtempy * x_0_2_3 + WQtempy * x_0_2_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
    QUICKDouble x_0_5_3 = Qtempy * x_0_3_3 + WQtempy * x_0_3_4;
    QUICKDouble x_0_9_3 = Qtempz * x_0_3_3 + WQtempz * x_0_3_4 + CDtemp * (VY_3 - ABcom * VY_4);
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble x_0_6_3 = Qtempx * x_0_3_3 + WQtempx * x_0_3_4;
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_17_2 = Qtempx * x_0_7_2 + WQtempx * x_0_7_3 + 2.000000 * CDtemp * (x_0_1_2 - ABcom * x_0_1_3);
    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
    QUICKDouble x_0_10_2 = Qtempx * x_0_5_2 + WQtempx * x_0_5_3;
    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_12_2 = Qtempx * x_0_8_2 + WQtempx * x_0_8_3;
    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
    QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
    QUICKDouble x_0_11_2 = Qtempx * x_0_4_2 + WQtempx * x_0_4_3 + CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_19_2 = Qtempz * x_0_9_2 + WQtempz * x_0_9_3 + 2.000000 * CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_18_2 = Qtempy * x_0_8_2 + WQtempy * x_0_8_3 + 2.000000 * CDtemp * (x_0_2_2 - ABcom * x_0_2_3);
    QUICKDouble x_0_16_2 = Qtempy * x_0_9_2 + WQtempy * x_0_9_3;
    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
    QUICKDouble x_0_15_2 = Qtempy * x_0_5_2 + WQtempy * x_0_5_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_14_2 = Qtempx * x_0_9_2 + WQtempx * x_0_9_3;
    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
    QUICKDouble x_0_13_2 = Qtempx * x_0_6_2 + WQtempx * x_0_6_3 + CDtemp * (x_0_3_2 - ABcom * x_0_3_3);
    QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + 2.000000 * ABCDtemp * x_0_3_2;
    QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
    QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
    QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
    QUICKDouble x_1_14_1 = Ptempx * x_0_14_1 + WPtempx * x_0_14_2 + ABCDtemp * x_0_9_2;
    QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
    QUICKDouble x_1_14_0 = Ptempx * x_0_14_0 + WPtempx * x_0_14_1 + ABCDtemp * x_0_9_1;
    LOCSTORE(store, 7, 14, STOREDIM, STOREDIM) += Ptempx * x_1_14_0 + WPtempx * x_1_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1) + ABCDtemp * x_1_9_1;
    QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + 2.000000 * ABCDtemp * x_0_2_2;
    QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
    QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
    QUICKDouble x_1_12_1 = Ptempx * x_0_12_1 + WPtempx * x_0_12_2 + ABCDtemp * x_0_8_2;
    QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
    QUICKDouble x_1_12_0 = Ptempx * x_0_12_0 + WPtempx * x_0_12_1 + ABCDtemp * x_0_8_1;
    LOCSTORE(store, 7, 12, STOREDIM, STOREDIM) += Ptempx * x_1_12_0 + WPtempx * x_1_12_1 + ABtemp * (x_0_12_0 - CDcom * x_0_12_1) + ABCDtemp * x_1_8_1;
    QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + 2.000000 * ABCDtemp * x_0_1_2;
    QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2.000000 * CDtemp * (x_0_1_1 - ABcom * x_0_1_2);
    QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
    QUICKDouble x_1_17_1 = Ptempx * x_0_17_1 + WPtempx * x_0_17_2 + 3.000000 * ABCDtemp * x_0_7_2;
    QUICKDouble x_1_17_0 = Ptempx * x_0_17_0 + WPtempx * x_0_17_1 + 3.000000 * ABCDtemp * x_0_7_1;
    LOCSTORE(store, 7, 17, STOREDIM, STOREDIM) += Ptempx * x_1_17_0 + WPtempx * x_1_17_1 + ABtemp * (x_0_17_0 - CDcom * x_0_17_1) + 3.000000 * ABCDtemp * x_1_7_1;
    QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_3_13_1 = Ptempz * x_0_13_1 + WPtempz * x_0_13_2 + ABCDtemp * x_0_7_2;
    QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_3_13_0 = Ptempz * x_0_13_0 + WPtempz * x_0_13_1 + ABCDtemp * x_0_7_1;
    QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
    LOCSTORE(store, 6, 13, STOREDIM, STOREDIM) += Ptempx * x_3_13_0 + WPtempx * x_3_13_1 + 2.000000 * ABCDtemp * x_3_6_1;
    QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
    QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
    QUICKDouble x_1_13_1 = Ptempx * x_0_13_1 + WPtempx * x_0_13_2 + 2.000000 * ABCDtemp * x_0_6_2;
    QUICKDouble x_1_13_0 = Ptempx * x_0_13_0 + WPtempx * x_0_13_1 + 2.000000 * ABCDtemp * x_0_6_1;
    LOCSTORE(store, 7, 13, STOREDIM, STOREDIM) += Ptempx * x_1_13_0 + WPtempx * x_1_13_1 + ABtemp * (x_0_13_0 - CDcom * x_0_13_1) + 2.000000 * ABCDtemp * x_1_6_1;
    QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
    QUICKDouble x_3_10_1 = Ptempz * x_0_10_1 + WPtempz * x_0_10_2 + ABCDtemp * x_0_4_2;
    QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
    QUICKDouble x_3_10_0 = Ptempz * x_0_10_0 + WPtempz * x_0_10_1 + ABCDtemp * x_0_4_1;
    LOCSTORE(store, 5, 10, STOREDIM, STOREDIM) += Ptempy * x_3_10_0 + WPtempy * x_3_10_1 + ABCDtemp * x_3_6_1;
    QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
    LOCSTORE(store, 6, 10, STOREDIM, STOREDIM) += Ptempx * x_3_10_0 + WPtempx * x_3_10_1 + ABCDtemp * x_3_5_1;
    QUICKDouble x_2_10_1 = Ptempy * x_0_10_1 + WPtempy * x_0_10_2 + ABCDtemp * x_0_6_2;
    QUICKDouble x_2_10_0 = Ptempy * x_0_10_0 + WPtempy * x_0_10_1 + ABCDtemp * x_0_6_1;
    LOCSTORE(store, 8, 10, STOREDIM, STOREDIM) += Ptempy * x_2_10_0 + WPtempy * x_2_10_1 + ABtemp * (x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_2_6_1;
    QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
    LOCSTORE(store, 4, 10, STOREDIM, STOREDIM) += Ptempx * x_2_10_0 + WPtempx * x_2_10_1 + ABCDtemp * x_2_5_1;
    QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
    QUICKDouble x_1_10_1 = Ptempx * x_0_10_1 + WPtempx * x_0_10_2 + ABCDtemp * x_0_5_2;
    QUICKDouble x_1_10_0 = Ptempx * x_0_10_0 + WPtempx * x_0_10_1 + ABCDtemp * x_0_5_1;
    LOCSTORE(store, 7, 10, STOREDIM, STOREDIM) += Ptempx * x_1_10_0 + WPtempx * x_1_10_1 + ABtemp * (x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_1_5_1;
    QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    QUICKDouble x_2_11_1 = Ptempy * x_0_11_1 + WPtempy * x_0_11_2 + ABCDtemp * x_0_7_2;
    QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_2_11_0 = Ptempy * x_0_11_0 + WPtempy * x_0_11_1 + ABCDtemp * x_0_7_1;
    QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
    LOCSTORE(store, 4, 11, STOREDIM, STOREDIM) += Ptempx * x_2_11_0 + WPtempx * x_2_11_1 + 2.000000 * ABCDtemp * x_2_4_1;
    QUICKDouble x_2_12_1 = Ptempy * x_0_12_1 + WPtempy * x_0_12_2 + 2.000000 * ABCDtemp * x_0_4_2;
    QUICKDouble x_2_12_0 = Ptempy * x_0_12_0 + WPtempy * x_0_12_1 + 2.000000 * ABCDtemp * x_0_4_1;
    LOCSTORE(store, 4, 12, STOREDIM, STOREDIM) += Ptempx * x_2_12_0 + WPtempx * x_2_12_1 + ABCDtemp * x_2_8_1;
    LOCSTORE(store, 8, 12, STOREDIM, STOREDIM) += Ptempy * x_2_12_0 + WPtempy * x_2_12_1 + ABtemp * (x_0_12_0 - CDcom * x_0_12_1) + 2.000000 * ABCDtemp * x_2_4_1;
    QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
    QUICKDouble x_1_11_1 = Ptempx * x_0_11_1 + WPtempx * x_0_11_2 + 2.000000 * ABCDtemp * x_0_4_2;
    QUICKDouble x_1_11_0 = Ptempx * x_0_11_0 + WPtempx * x_0_11_1 + 2.000000 * ABCDtemp * x_0_4_1;
    LOCSTORE(store, 7, 11, STOREDIM, STOREDIM) += Ptempx * x_1_11_0 + WPtempx * x_1_11_1 + ABtemp * (x_0_11_0 - CDcom * x_0_11_1) + 2.000000 * ABCDtemp * x_1_4_1;
    QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2.000000 * CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_3_19_1 = Ptempz * x_0_19_1 + WPtempz * x_0_19_2 + 3.000000 * ABCDtemp * x_0_9_2;
    QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_3_19_0 = Ptempz * x_0_19_0 + WPtempz * x_0_19_1 + 3.000000 * ABCDtemp * x_0_9_1;
    LOCSTORE(store, 9, 19, STOREDIM, STOREDIM) += Ptempz * x_3_19_0 + WPtempz * x_3_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1) + 3.000000 * ABCDtemp * x_3_9_1;
    LOCSTORE(store, 5, 19, STOREDIM, STOREDIM) += Ptempy * x_3_19_0 + WPtempy * x_3_19_1;
    LOCSTORE(store, 6, 19, STOREDIM, STOREDIM) += Ptempx * x_3_19_0 + WPtempx * x_3_19_1;
    QUICKDouble x_2_19_1 = Ptempy * x_0_19_1 + WPtempy * x_0_19_2;
    QUICKDouble x_2_19_0 = Ptempy * x_0_19_0 + WPtempy * x_0_19_1;
    LOCSTORE(store, 4, 19, STOREDIM, STOREDIM) += Ptempx * x_2_19_0 + WPtempx * x_2_19_1;
    LOCSTORE(store, 8, 19, STOREDIM, STOREDIM) += Ptempy * x_2_19_0 + WPtempy * x_2_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1);
    QUICKDouble x_1_19_1 = Ptempx * x_0_19_1 + WPtempx * x_0_19_2;
    QUICKDouble x_1_19_0 = Ptempx * x_0_19_0 + WPtempx * x_0_19_1;
    LOCSTORE(store, 7, 19, STOREDIM, STOREDIM) += Ptempx * x_1_19_0 + WPtempx * x_1_19_1 + ABtemp * (x_0_19_0 - CDcom * x_0_19_1);
    QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2.000000 * CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    QUICKDouble x_2_18_1 = Ptempy * x_0_18_1 + WPtempy * x_0_18_2 + 3.000000 * ABCDtemp * x_0_8_2;
    QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_2_18_0 = Ptempy * x_0_18_0 + WPtempy * x_0_18_1 + 3.000000 * ABCDtemp * x_0_8_1;
    LOCSTORE(store, 8, 18, STOREDIM, STOREDIM) += Ptempy * x_2_18_0 + WPtempy * x_2_18_1 + ABtemp * (x_0_18_0 - CDcom * x_0_18_1) + 3.000000 * ABCDtemp * x_2_8_1;
    LOCSTORE(store, 4, 18, STOREDIM, STOREDIM) += Ptempx * x_2_18_0 + WPtempx * x_2_18_1;
    QUICKDouble x_1_18_1 = Ptempx * x_0_18_1 + WPtempx * x_0_18_2;
    QUICKDouble x_1_18_0 = Ptempx * x_0_18_0 + WPtempx * x_0_18_1;
    LOCSTORE(store, 7, 18, STOREDIM, STOREDIM) += Ptempx * x_1_18_0 + WPtempx * x_1_18_1 + ABtemp * (x_0_18_0 - CDcom * x_0_18_1);
    QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
    QUICKDouble x_3_16_1 = Ptempz * x_0_16_1 + WPtempz * x_0_16_2 + 2.000000 * ABCDtemp * x_0_5_2;
    QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
    QUICKDouble x_3_16_0 = Ptempz * x_0_16_0 + WPtempz * x_0_16_1 + 2.000000 * ABCDtemp * x_0_5_1;
    LOCSTORE(store, 5, 16, STOREDIM, STOREDIM) += Ptempy * x_3_16_0 + WPtempy * x_3_16_1 + ABCDtemp * x_3_9_1;
    LOCSTORE(store, 9, 16, STOREDIM, STOREDIM) += Ptempz * x_3_16_0 + WPtempz * x_3_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1) + 2.000000 * ABCDtemp * x_3_5_1;
    LOCSTORE(store, 6, 16, STOREDIM, STOREDIM) += Ptempx * x_3_16_0 + WPtempx * x_3_16_1;
    QUICKDouble x_2_16_1 = Ptempy * x_0_16_1 + WPtempy * x_0_16_2 + ABCDtemp * x_0_9_2;
    QUICKDouble x_2_16_0 = Ptempy * x_0_16_0 + WPtempy * x_0_16_1 + ABCDtemp * x_0_9_1;
    LOCSTORE(store, 8, 16, STOREDIM, STOREDIM) += Ptempy * x_2_16_0 + WPtempy * x_2_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1) + ABCDtemp * x_2_9_1;
    LOCSTORE(store, 4, 16, STOREDIM, STOREDIM) += Ptempx * x_2_16_0 + WPtempx * x_2_16_1;
    QUICKDouble x_1_16_1 = Ptempx * x_0_16_1 + WPtempx * x_0_16_2;
    QUICKDouble x_1_16_0 = Ptempx * x_0_16_0 + WPtempx * x_0_16_1;
    LOCSTORE(store, 7, 16, STOREDIM, STOREDIM) += Ptempx * x_1_16_0 + WPtempx * x_1_16_1 + ABtemp * (x_0_16_0 - CDcom * x_0_16_1);
    QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    QUICKDouble x_3_15_1 = Ptempz * x_0_15_1 + WPtempz * x_0_15_2 + ABCDtemp * x_0_8_2;
    QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_3_15_0 = Ptempz * x_0_15_0 + WPtempz * x_0_15_1 + ABCDtemp * x_0_8_1;
    LOCSTORE(store, 5, 15, STOREDIM, STOREDIM) += Ptempy * x_3_15_0 + WPtempy * x_3_15_1 + 2.000000 * ABCDtemp * x_3_5_1;
    LOCSTORE(store, 6, 15, STOREDIM, STOREDIM) += Ptempx * x_3_15_0 + WPtempx * x_3_15_1;
    QUICKDouble x_2_15_1 = Ptempy * x_0_15_1 + WPtempy * x_0_15_2 + 2.000000 * ABCDtemp * x_0_5_2;
    QUICKDouble x_2_15_0 = Ptempy * x_0_15_0 + WPtempy * x_0_15_1 + 2.000000 * ABCDtemp * x_0_5_1;
    LOCSTORE(store, 8, 15, STOREDIM, STOREDIM) += Ptempy * x_2_15_0 + WPtempy * x_2_15_1 + ABtemp * (x_0_15_0 - CDcom * x_0_15_1) + 2.000000 * ABCDtemp * x_2_5_1;
    LOCSTORE(store, 4, 15, STOREDIM, STOREDIM) += Ptempx * x_2_15_0 + WPtempx * x_2_15_1;
    QUICKDouble x_1_15_1 = Ptempx * x_0_15_1 + WPtempx * x_0_15_2;
    QUICKDouble x_1_15_0 = Ptempx * x_0_15_0 + WPtempx * x_0_15_1;
    LOCSTORE(store, 7, 15, STOREDIM, STOREDIM) += Ptempx * x_1_15_0 + WPtempx * x_1_15_1 + ABtemp * (x_0_15_0 - CDcom * x_0_15_1);
    QUICKDouble x_3_14_1 = Ptempz * x_0_14_1 + WPtempz * x_0_14_2 + 2.000000 * ABCDtemp * x_0_6_2;
    QUICKDouble x_3_14_0 = Ptempz * x_0_14_0 + WPtempz * x_0_14_1 + 2.000000 * ABCDtemp * x_0_6_1;
    LOCSTORE(store, 6, 14, STOREDIM, STOREDIM) += Ptempx * x_3_14_0 + WPtempx * x_3_14_1 + ABCDtemp * x_3_9_1;
    LOCSTORE(store, 9, 14, STOREDIM, STOREDIM) += Ptempz * x_3_14_0 + WPtempz * x_3_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1) + 2.000000 * ABCDtemp * x_3_6_1;
    LOCSTORE(store, 5, 14, STOREDIM, STOREDIM) += Ptempy * x_3_14_0 + WPtempy * x_3_14_1;
    QUICKDouble x_2_14_1 = Ptempy * x_0_14_1 + WPtempy * x_0_14_2;
    QUICKDouble x_2_14_0 = Ptempy * x_0_14_0 + WPtempy * x_0_14_1;
    LOCSTORE(store, 4, 14, STOREDIM, STOREDIM) += Ptempx * x_2_14_0 + WPtempx * x_2_14_1 + ABCDtemp * x_2_9_1;
    LOCSTORE(store, 8, 14, STOREDIM, STOREDIM) += Ptempy * x_2_14_0 + WPtempy * x_2_14_1 + ABtemp * (x_0_14_0 - CDcom * x_0_14_1);
    QUICKDouble x_2_13_1 = Ptempy * x_0_13_1 + WPtempy * x_0_13_2;
    QUICKDouble x_2_13_0 = Ptempy * x_0_13_0 + WPtempy * x_0_13_1;
    LOCSTORE(store, 4, 13, STOREDIM, STOREDIM) += Ptempx * x_2_13_0 + WPtempx * x_2_13_1 + 2.000000 * ABCDtemp * x_2_6_1;
    LOCSTORE(store, 8, 13, STOREDIM, STOREDIM) += Ptempy * x_2_13_0 + WPtempy * x_2_13_1 + ABtemp * (x_0_13_0 - CDcom * x_0_13_1);
    // [DS|FS] integral - End 

}

