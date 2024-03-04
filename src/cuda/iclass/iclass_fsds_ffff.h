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

    // [FS|DS] integral - Start
    QUICKDouble VY_5 = VY(0, 0, 5);
    QUICKDouble VY_4 = VY(0, 0, 4);
    QUICKDouble x_1_0_4 = Ptempx * VY_4 + WPtempx * VY_5;
    QUICKDouble x_2_0_4 = Ptempy * VY_4 + WPtempy * VY_5;
    QUICKDouble x_3_0_4 = Ptempz * VY_4 + WPtempz * VY_5;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_1_0_3 = Ptempx * VY_3 + WPtempx * VY_4;
    QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_3_0_3 = Ptempz * VY_3 + WPtempz * VY_4;
    QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
    QUICKDouble x_2_0_3 = Ptempy * VY_3 + WPtempy * VY_4;
    QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
    QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
    QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 + 2.000000 * ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
    QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 + 2.000000 * ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
    QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
    QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 + 2.000000 * ABtemp * (x_1_0_2 - CDcom * x_1_0_3);
    QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
    QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
    QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
    QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
    QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
    QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
    QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
    QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
    QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
    QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 + 2.000000 * ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_19_3_1 = Qtempz * x_19_0_1 + WQtempz * x_19_0_2 + 3.000000 * ABCDtemp * x_9_0_2;
    QUICKDouble x_19_3_0 = Qtempz * x_19_0_0 + WQtempz * x_19_0_1 + 3.000000 * ABCDtemp * x_9_0_1;
    QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 + 2.000000 * ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_18_3_1 = Qtempz * x_18_0_1 + WQtempz * x_18_0_2;
    QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_18_3_0 = Qtempz * x_18_0_0 + WQtempz * x_18_0_1;
    LOCSTORE(store, 18, 6, STOREDIM, STOREDIM) += Qtempx * x_18_3_0 + WQtempx * x_18_3_1;
    LOCSTORE(store, 18, 9, STOREDIM, STOREDIM) += Qtempz * x_18_3_0 + WQtempz * x_18_3_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1);
    QUICKDouble x_18_2_1 = Qtempy * x_18_0_1 + WQtempy * x_18_0_2 + 3.000000 * ABCDtemp * x_8_0_2;
    QUICKDouble x_18_2_0 = Qtempy * x_18_0_0 + WQtempy * x_18_0_1 + 3.000000 * ABCDtemp * x_8_0_1;
    LOCSTORE(store, 18, 4, STOREDIM, STOREDIM) += Qtempx * x_18_2_0 + WQtempx * x_18_2_1;
    QUICKDouble x_18_1_1 = Qtempx * x_18_0_1 + WQtempx * x_18_0_2;
    QUICKDouble x_18_1_0 = Qtempx * x_18_0_0 + WQtempx * x_18_0_1;
    LOCSTORE(store, 18, 7, STOREDIM, STOREDIM) += Qtempx * x_18_1_0 + WQtempx * x_18_1_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1);
    QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 + 2.000000 * ABtemp * (x_1_0_1 - CDcom * x_1_0_2);
    QUICKDouble x_17_3_1 = Qtempz * x_17_0_1 + WQtempz * x_17_0_2;
    QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
    QUICKDouble x_17_3_0 = Qtempz * x_17_0_0 + WQtempz * x_17_0_1;
    LOCSTORE(store, 17, 5, STOREDIM, STOREDIM) += Qtempy * x_17_3_0 + WQtempy * x_17_3_1;
    LOCSTORE(store, 17, 9, STOREDIM, STOREDIM) += Qtempz * x_17_3_0 + WQtempz * x_17_3_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1);
    QUICKDouble x_17_2_1 = Qtempy * x_17_0_1 + WQtempy * x_17_0_2;
    QUICKDouble x_17_2_0 = Qtempy * x_17_0_0 + WQtempy * x_17_0_1;
    LOCSTORE(store, 17, 8, STOREDIM, STOREDIM) += Qtempy * x_17_2_0 + WQtempy * x_17_2_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1);
    QUICKDouble x_7_1_1 = Qtempx * x_7_0_1 + WQtempx * x_7_0_2 + 2.000000 * ABCDtemp * x_1_0_2;
    QUICKDouble x_17_1_1 = Qtempx * x_17_0_1 + WQtempx * x_17_0_2 + 3.000000 * ABCDtemp * x_7_0_2;
    QUICKDouble x_17_1_0 = Qtempx * x_17_0_0 + WQtempx * x_17_0_1 + 3.000000 * ABCDtemp * x_7_0_1;
    LOCSTORE(store, 17, 7, STOREDIM, STOREDIM) += Qtempx * x_17_1_0 + WQtempx * x_17_1_1 + CDtemp * (x_17_0_0 - ABcom * x_17_0_1) + 3.000000 * ABCDtemp * x_7_1_1;
    QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
    QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
    QUICKDouble x_16_3_1 = Qtempz * x_16_0_1 + WQtempz * x_16_0_2 + 2.000000 * ABCDtemp * x_5_0_2;
    QUICKDouble x_16_3_0 = Qtempz * x_16_0_0 + WQtempz * x_16_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
    QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_15_3_1 = Qtempz * x_15_0_1 + WQtempz * x_15_0_2 + ABCDtemp * x_8_0_2;
    QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_15_3_0 = Qtempz * x_15_0_0 + WQtempz * x_15_0_1 + ABCDtemp * x_8_0_1;
    LOCSTORE(store, 15, 6, STOREDIM, STOREDIM) += Qtempx * x_15_3_0 + WQtempx * x_15_3_1;
    QUICKDouble x_15_2_1 = Qtempy * x_15_0_1 + WQtempy * x_15_0_2 + 2.000000 * ABCDtemp * x_5_0_2;
    QUICKDouble x_15_2_0 = Qtempy * x_15_0_0 + WQtempy * x_15_0_1 + 2.000000 * ABCDtemp * x_5_0_1;
    QUICKDouble x_9_3_1 = Qtempz * x_9_0_1 + WQtempz * x_9_0_2 + 2.000000 * ABCDtemp * x_3_0_2;
    LOCSTORE(store, 19, 9, STOREDIM, STOREDIM) += Qtempz * x_19_3_0 + WQtempz * x_19_3_1 + CDtemp * (x_19_0_0 - ABcom * x_19_0_1) + 3.000000 * ABCDtemp * x_9_3_1;
    LOCSTORE(store, 16, 5, STOREDIM, STOREDIM) += Qtempy * x_16_3_0 + WQtempy * x_16_3_1 + ABCDtemp * x_9_3_1;
    QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
    QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
    QUICKDouble x_14_3_1 = Qtempz * x_14_0_1 + WQtempz * x_14_0_2 + 2.000000 * ABCDtemp * x_6_0_2;
    QUICKDouble x_14_3_0 = Qtempz * x_14_0_0 + WQtempz * x_14_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
    LOCSTORE(store, 14, 5, STOREDIM, STOREDIM) += Qtempy * x_14_3_0 + WQtempy * x_14_3_1;
    LOCSTORE(store, 14, 6, STOREDIM, STOREDIM) += Qtempx * x_14_3_0 + WQtempx * x_14_3_1 + ABCDtemp * x_9_3_1;
    QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_13_3_1 = Qtempz * x_13_0_1 + WQtempz * x_13_0_2 + ABCDtemp * x_7_0_2;
    QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_13_3_0 = Qtempz * x_13_0_0 + WQtempz * x_13_0_1 + ABCDtemp * x_7_0_1;
    LOCSTORE(store, 13, 5, STOREDIM, STOREDIM) += Qtempy * x_13_3_0 + WQtempy * x_13_3_1;
    QUICKDouble x_13_2_1 = Qtempy * x_13_0_1 + WQtempy * x_13_0_2;
    QUICKDouble x_13_2_0 = Qtempy * x_13_0_0 + WQtempy * x_13_0_1;
    LOCSTORE(store, 13, 8, STOREDIM, STOREDIM) += Qtempy * x_13_2_0 + WQtempy * x_13_2_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1);
    QUICKDouble x_6_1_1 = Qtempx * x_6_0_1 + WQtempx * x_6_0_2 + ABCDtemp * x_3_0_2;
    QUICKDouble x_13_1_1 = Qtempx * x_13_0_1 + WQtempx * x_13_0_2 + 2.000000 * ABCDtemp * x_6_0_2;
    QUICKDouble x_13_1_0 = Qtempx * x_13_0_0 + WQtempx * x_13_0_1 + 2.000000 * ABCDtemp * x_6_0_1;
    LOCSTORE(store, 13, 7, STOREDIM, STOREDIM) += Qtempx * x_13_1_0 + WQtempx * x_13_1_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1) + 2.000000 * ABCDtemp * x_6_1_1;
    QUICKDouble x_8_3_1 = Qtempz * x_8_0_1 + WQtempz * x_8_0_2;
    LOCSTORE(store, 18, 5, STOREDIM, STOREDIM) += Qtempy * x_18_3_0 + WQtempy * x_18_3_1 + 3.000000 * ABCDtemp * x_8_3_1;
    LOCSTORE(store, 15, 9, STOREDIM, STOREDIM) += Qtempz * x_15_3_0 + WQtempz * x_15_3_1 + CDtemp * (x_15_0_0 - ABcom * x_15_0_1) + ABCDtemp * x_8_3_1;
    QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
    QUICKDouble x_12_3_1 = Qtempz * x_12_0_1 + WQtempz * x_12_0_2;
    QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
    QUICKDouble x_12_3_0 = Qtempz * x_12_0_0 + WQtempz * x_12_0_1;
    LOCSTORE(store, 12, 6, STOREDIM, STOREDIM) += Qtempx * x_12_3_0 + WQtempx * x_12_3_1 + ABCDtemp * x_8_3_1;
    LOCSTORE(store, 12, 9, STOREDIM, STOREDIM) += Qtempz * x_12_3_0 + WQtempz * x_12_3_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1);
    QUICKDouble x_8_2_1 = Qtempy * x_8_0_1 + WQtempy * x_8_0_2 + 2.000000 * ABCDtemp * x_2_0_2;
    LOCSTORE(store, 18, 8, STOREDIM, STOREDIM) += Qtempy * x_18_2_0 + WQtempy * x_18_2_1 + CDtemp * (x_18_0_0 - ABcom * x_18_0_1) + 3.000000 * ABCDtemp * x_8_2_1;
    QUICKDouble x_12_2_1 = Qtempy * x_12_0_1 + WQtempy * x_12_0_2 + 2.000000 * ABCDtemp * x_4_0_2;
    QUICKDouble x_12_2_0 = Qtempy * x_12_0_0 + WQtempy * x_12_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
    LOCSTORE(store, 12, 4, STOREDIM, STOREDIM) += Qtempx * x_12_2_0 + WQtempx * x_12_2_1 + ABCDtemp * x_8_2_1;
    QUICKDouble x_8_1_1 = Qtempx * x_8_0_1 + WQtempx * x_8_0_2;
    QUICKDouble x_12_1_1 = Qtempx * x_12_0_1 + WQtempx * x_12_0_2 + ABCDtemp * x_8_0_2;
    QUICKDouble x_12_1_0 = Qtempx * x_12_0_0 + WQtempx * x_12_0_1 + ABCDtemp * x_8_0_1;
    LOCSTORE(store, 12, 7, STOREDIM, STOREDIM) += Qtempx * x_12_1_0 + WQtempx * x_12_1_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1) + ABCDtemp * x_8_1_1;
    QUICKDouble x_7_3_1 = Qtempz * x_7_0_1 + WQtempz * x_7_0_2;
    LOCSTORE(store, 17, 6, STOREDIM, STOREDIM) += Qtempx * x_17_3_0 + WQtempx * x_17_3_1 + 3.000000 * ABCDtemp * x_7_3_1;
    LOCSTORE(store, 13, 9, STOREDIM, STOREDIM) += Qtempz * x_13_3_0 + WQtempz * x_13_3_1 + CDtemp * (x_13_0_0 - ABcom * x_13_0_1) + ABCDtemp * x_7_3_1;
    QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_11_3_1 = Qtempz * x_11_0_1 + WQtempz * x_11_0_2;
    QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_11_3_0 = Qtempz * x_11_0_0 + WQtempz * x_11_0_1;
    LOCSTORE(store, 11, 5, STOREDIM, STOREDIM) += Qtempy * x_11_3_0 + WQtempy * x_11_3_1 + ABCDtemp * x_7_3_1;
    LOCSTORE(store, 11, 9, STOREDIM, STOREDIM) += Qtempz * x_11_3_0 + WQtempz * x_11_3_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1);
    QUICKDouble x_4_2_1 = Qtempy * x_4_0_1 + WQtempy * x_4_0_2 + ABCDtemp * x_1_0_2;
    LOCSTORE(store, 12, 8, STOREDIM, STOREDIM) += Qtempy * x_12_2_0 + WQtempy * x_12_2_1 + CDtemp * (x_12_0_0 - ABcom * x_12_0_1) + 2.000000 * ABCDtemp * x_4_2_1;
    QUICKDouble x_7_2_1 = Qtempy * x_7_0_1 + WQtempy * x_7_0_2;
    LOCSTORE(store, 17, 4, STOREDIM, STOREDIM) += Qtempx * x_17_2_0 + WQtempx * x_17_2_1 + 3.000000 * ABCDtemp * x_7_2_1;
    QUICKDouble x_11_2_1 = Qtempy * x_11_0_1 + WQtempy * x_11_0_2 + ABCDtemp * x_7_0_2;
    QUICKDouble x_11_2_0 = Qtempy * x_11_0_0 + WQtempy * x_11_0_1 + ABCDtemp * x_7_0_1;
    LOCSTORE(store, 11, 4, STOREDIM, STOREDIM) += Qtempx * x_11_2_0 + WQtempx * x_11_2_1 + 2.000000 * ABCDtemp * x_4_2_1;
    LOCSTORE(store, 11, 8, STOREDIM, STOREDIM) += Qtempy * x_11_2_0 + WQtempy * x_11_2_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1) + ABCDtemp * x_7_2_1;
    QUICKDouble x_4_1_1 = Qtempx * x_4_0_1 + WQtempx * x_4_0_2 + ABCDtemp * x_2_0_2;
    QUICKDouble x_11_1_1 = Qtempx * x_11_0_1 + WQtempx * x_11_0_2 + 2.000000 * ABCDtemp * x_4_0_2;
    QUICKDouble x_11_1_0 = Qtempx * x_11_0_0 + WQtempx * x_11_0_1 + 2.000000 * ABCDtemp * x_4_0_1;
    LOCSTORE(store, 11, 7, STOREDIM, STOREDIM) += Qtempx * x_11_1_0 + WQtempx * x_11_1_1 + CDtemp * (x_11_0_0 - ABcom * x_11_0_1) + 2.000000 * ABCDtemp * x_4_1_1;
    QUICKDouble x_6_3_1 = Qtempz * x_6_0_1 + WQtempz * x_6_0_2 + ABCDtemp * x_1_0_2;
    LOCSTORE(store, 14, 9, STOREDIM, STOREDIM) += Qtempz * x_14_3_0 + WQtempz * x_14_3_1 + CDtemp * (x_14_0_0 - ABcom * x_14_0_1) + 2.000000 * ABCDtemp * x_6_3_1;
    LOCSTORE(store, 13, 6, STOREDIM, STOREDIM) += Qtempx * x_13_3_0 + WQtempx * x_13_3_1 + 2.000000 * ABCDtemp * x_6_3_1;
    QUICKDouble x_5_3_1 = Qtempz * x_5_0_1 + WQtempz * x_5_0_2 + ABCDtemp * x_2_0_2;
    LOCSTORE(store, 16, 9, STOREDIM, STOREDIM) += Qtempz * x_16_3_0 + WQtempz * x_16_3_1 + CDtemp * (x_16_0_0 - ABcom * x_16_0_1) + 2.000000 * ABCDtemp * x_5_3_1;
    LOCSTORE(store, 15, 5, STOREDIM, STOREDIM) += Qtempy * x_15_3_0 + WQtempy * x_15_3_1 + 2.000000 * ABCDtemp * x_5_3_1;
    QUICKDouble x_4_3_1 = Qtempz * x_4_0_1 + WQtempz * x_4_0_2;
    LOCSTORE(store, 12, 5, STOREDIM, STOREDIM) += Qtempy * x_12_3_0 + WQtempy * x_12_3_1 + 2.000000 * ABCDtemp * x_4_3_1;
    LOCSTORE(store, 11, 6, STOREDIM, STOREDIM) += Qtempx * x_11_3_0 + WQtempx * x_11_3_1 + 2.000000 * ABCDtemp * x_4_3_1;
    QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
    QUICKDouble x_10_3_1 = Qtempz * x_10_0_1 + WQtempz * x_10_0_2 + ABCDtemp * x_4_0_2;
    QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
    QUICKDouble x_10_3_0 = Qtempz * x_10_0_0 + WQtempz * x_10_0_1 + ABCDtemp * x_4_0_1;
    LOCSTORE(store, 10, 5, STOREDIM, STOREDIM) += Qtempy * x_10_3_0 + WQtempy * x_10_3_1 + ABCDtemp * x_6_3_1;
    LOCSTORE(store, 10, 6, STOREDIM, STOREDIM) += Qtempx * x_10_3_0 + WQtempx * x_10_3_1 + ABCDtemp * x_5_3_1;
    LOCSTORE(store, 10, 9, STOREDIM, STOREDIM) += Qtempz * x_10_3_0 + WQtempz * x_10_3_1 + CDtemp * (x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_4_3_1;
    QUICKDouble x_5_2_1 = Qtempy * x_5_0_1 + WQtempy * x_5_0_2 + ABCDtemp * x_3_0_2;
    LOCSTORE(store, 15, 8, STOREDIM, STOREDIM) += Qtempy * x_15_2_0 + WQtempy * x_15_2_1 + CDtemp * (x_15_0_0 - ABcom * x_15_0_1) + 2.000000 * ABCDtemp * x_5_2_1;
    QUICKDouble x_6_2_1 = Qtempy * x_6_0_1 + WQtempy * x_6_0_2;
    LOCSTORE(store, 13, 4, STOREDIM, STOREDIM) += Qtempx * x_13_2_0 + WQtempx * x_13_2_1 + 2.000000 * ABCDtemp * x_6_2_1;
    QUICKDouble x_10_2_1 = Qtempy * x_10_0_1 + WQtempy * x_10_0_2 + ABCDtemp * x_6_0_2;
    QUICKDouble x_10_2_0 = Qtempy * x_10_0_0 + WQtempy * x_10_0_1 + ABCDtemp * x_6_0_1;
    LOCSTORE(store, 10, 4, STOREDIM, STOREDIM) += Qtempx * x_10_2_0 + WQtempx * x_10_2_1 + ABCDtemp * x_5_2_1;
    LOCSTORE(store, 10, 8, STOREDIM, STOREDIM) += Qtempy * x_10_2_0 + WQtempy * x_10_2_1 + CDtemp * (x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_6_2_1;
    // [FS|DS] integral - End 

}

