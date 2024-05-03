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

}

