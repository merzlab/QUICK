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

