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

}

