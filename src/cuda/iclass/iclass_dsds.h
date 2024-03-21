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

}

