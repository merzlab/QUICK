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

    // [SS|GS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
    QUICKDouble VY_4 = VY(0, 0, 4);
    QUICKDouble x_0_3_3 = Qtempz * VY_3 + WQtempz * VY_4;
    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
    QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
    QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
    LOCSTORE(store, 0, 23, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_10_0 + WQtempx * x_0_10_1 + CDtemp * (x_0_5_0 - ABcom * x_0_5_1);
    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
    QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
    QUICKDouble x_0_2_3 = Qtempy * VY_3 + WQtempy * VY_4;
    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
    QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
    QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    LOCSTORE(store, 0, 28, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_11_0 + WQtempx * x_0_11_1 + 2.000000 * CDtemp * (x_0_4_0 - ABcom * x_0_4_1);
    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
    QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
    LOCSTORE(store, 0, 20, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_12_0 + WQtempx * x_0_12_1 + CDtemp * (x_0_8_0 - ABcom * x_0_8_1);
    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
    QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    LOCSTORE(store, 0, 26, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_13_0 + WQtempx * x_0_13_1 + 2.000000 * CDtemp * (x_0_6_0 - ABcom * x_0_6_1);
    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
    QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
    LOCSTORE(store, 0, 21, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_14_0 + WQtempx * x_0_14_1 + CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    LOCSTORE(store, 0, 30, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_15_0 + WQtempy * x_0_15_1 + 2.000000 * CDtemp * (x_0_5_0 - ABcom * x_0_5_1);
    LOCSTORE(store, 0, 24, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_15_0 + WQtempx * x_0_15_1;
    QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
    QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
    LOCSTORE(store, 0, 25, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_16_0 + WQtempx * x_0_16_1;
    LOCSTORE(store, 0, 22, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_16_0 + WQtempy * x_0_16_1 + CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
    QUICKDouble x_0_1_3 = Qtempx * VY_3 + WQtempx * VY_4;
    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3 + CDtemp * (VY_2 - ABcom * VY_3);
    QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
    QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2.000000 * CDtemp * (x_0_1_1 - ABcom * x_0_1_2);
    LOCSTORE(store, 0, 32, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_17_0 + WQtempx * x_0_17_1 + 3.000000 * CDtemp * (x_0_7_0 - ABcom * x_0_7_1);
    QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2.000000 * CDtemp * (x_0_2_1 - ABcom * x_0_2_2);
    LOCSTORE(store, 0, 33, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_18_0 + WQtempy * x_0_18_1 + 3.000000 * CDtemp * (x_0_8_0 - ABcom * x_0_8_1);
    LOCSTORE(store, 0, 29, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_18_0 + WQtempx * x_0_18_1;
    QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2.000000 * CDtemp * (x_0_3_1 - ABcom * x_0_3_2);
    LOCSTORE(store, 0, 34, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_0_19_0 + WQtempz * x_0_19_1 + 3.000000 * CDtemp * (x_0_9_0 - ABcom * x_0_9_1);
    LOCSTORE(store, 0, 31, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_19_0 + WQtempy * x_0_19_1;
    LOCSTORE(store, 0, 27, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_19_0 + WQtempx * x_0_19_1;
    // [SS|GS] integral - End 

}

