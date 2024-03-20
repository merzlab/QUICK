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

    // [HS|SS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
    QUICKDouble VY_4 = VY(0, 0, 4);
    QUICKDouble x_2_0_3 = Ptempy * VY_3 + WPtempy * VY_4;
    QUICKDouble VY_5 = VY(0, 0, 5);
    QUICKDouble x_2_0_4 = Ptempy * VY_4 + WPtempy * VY_5;
    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
    QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
    QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
    QUICKDouble x_20_0_0 = Ptempx * x_12_0_0 + WPtempx * x_12_0_1 + ABtemp * (x_8_0_0 - CDcom * x_8_0_1);
    QUICKDouble x_20_0_1 = Ptempx * x_12_0_1 + WPtempx * x_12_0_2 + ABtemp * (x_8_0_1 - CDcom * x_8_0_2);
    LOCSTORE(store, 46, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_20_0_0 + WPtempx * x_20_0_1 + 2.000000 * ABtemp * (x_12_0_0 - CDcom * x_12_0_1);
    QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
    QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
    QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
    QUICKDouble x_3_0_3 = Ptempz * VY_3 + WPtempz * VY_4;
    QUICKDouble x_3_0_4 = Ptempz * VY_4 + WPtempz * VY_5;
    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
    QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
    QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
    QUICKDouble x_21_0_0 = Ptempx * x_14_0_0 + WPtempx * x_14_0_1 + ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_21_0_1 = Ptempx * x_14_0_1 + WPtempx * x_14_0_2 + ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    LOCSTORE(store, 44, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_21_0_0 + WPtempx * x_21_0_1 + 2.000000 * ABtemp * (x_14_0_0 - CDcom * x_14_0_1);
    QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
    QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
    QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
    QUICKDouble x_22_0_0 = Ptempy * x_16_0_0 + WPtempy * x_16_0_1 + ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_22_0_1 = Ptempy * x_16_0_1 + WPtempy * x_16_0_2 + ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    LOCSTORE(store, 42, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_22_0_0 + WPtempy * x_22_0_1 + 2.000000 * ABtemp * (x_16_0_0 - CDcom * x_16_0_1);
    LOCSTORE(store, 35, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_22_0_0 + WPtempx * x_22_0_1;
    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
    QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
    QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
    QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
    QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
    QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
    QUICKDouble x_23_0_0 = Ptempx * x_10_0_0 + WPtempx * x_10_0_1 + ABtemp * (x_5_0_0 - CDcom * x_5_0_1);
    QUICKDouble x_23_0_1 = Ptempx * x_10_0_1 + WPtempx * x_10_0_2 + ABtemp * (x_5_0_1 - CDcom * x_5_0_2);
    LOCSTORE(store, 38, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_23_0_0 + WPtempx * x_23_0_1 + 2.000000 * ABtemp * (x_10_0_0 - CDcom * x_10_0_1);
    QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_24_0_0 = Ptempx * x_15_0_0 + WPtempx * x_15_0_1;
    QUICKDouble x_24_0_1 = Ptempx * x_15_0_1 + WPtempx * x_15_0_2;
    LOCSTORE(store, 37, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_24_0_0 + WPtempx * x_24_0_1 + ABtemp * (x_15_0_0 - CDcom * x_15_0_1);
    QUICKDouble x_25_0_0 = Ptempx * x_16_0_0 + WPtempx * x_16_0_1;
    QUICKDouble x_25_0_1 = Ptempx * x_16_0_1 + WPtempx * x_16_0_2;
    LOCSTORE(store, 36, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_25_0_0 + WPtempx * x_25_0_1 + ABtemp * (x_16_0_0 - CDcom * x_16_0_1);
    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
    QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
    QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
    QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_26_0_0 = Ptempx * x_13_0_0 + WPtempx * x_13_0_1 + 2.000000 * ABtemp * (x_6_0_0 - CDcom * x_6_0_1);
    QUICKDouble x_26_0_1 = Ptempx * x_13_0_1 + WPtempx * x_13_0_2 + 2.000000 * ABtemp * (x_6_0_1 - CDcom * x_6_0_2);
    LOCSTORE(store, 50, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_26_0_0 + WPtempx * x_26_0_1 + 3.000000 * ABtemp * (x_13_0_0 - CDcom * x_13_0_1);
    QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 + 2.000000 * ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 + 2.000000 * ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_27_0_0 = Ptempx * x_19_0_0 + WPtempx * x_19_0_1;
    QUICKDouble x_27_0_1 = Ptempx * x_19_0_1 + WPtempx * x_19_0_2;
    LOCSTORE(store, 43, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_27_0_0 + WPtempx * x_27_0_1 + ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
    QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
    QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
    QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_28_0_0 = Ptempx * x_11_0_0 + WPtempx * x_11_0_1 + 2.000000 * ABtemp * (x_4_0_0 - CDcom * x_4_0_1);
    QUICKDouble x_28_0_1 = Ptempx * x_11_0_1 + WPtempx * x_11_0_2 + 2.000000 * ABtemp * (x_4_0_1 - CDcom * x_4_0_2);
    LOCSTORE(store, 52, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_28_0_0 + WPtempx * x_28_0_1 + 3.000000 * ABtemp * (x_11_0_0 - CDcom * x_11_0_1);
    QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 + 2.000000 * ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 + 2.000000 * ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_29_0_0 = Ptempx * x_18_0_0 + WPtempx * x_18_0_1;
    QUICKDouble x_29_0_1 = Ptempx * x_18_0_1 + WPtempx * x_18_0_2;
    LOCSTORE(store, 45, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_29_0_0 + WPtempx * x_29_0_1 + ABtemp * (x_18_0_0 - CDcom * x_18_0_1);
    QUICKDouble x_30_0_0 = Ptempy * x_15_0_0 + WPtempy * x_15_0_1 + 2.000000 * ABtemp * (x_5_0_0 - CDcom * x_5_0_1);
    QUICKDouble x_30_0_1 = Ptempy * x_15_0_1 + WPtempy * x_15_0_2 + 2.000000 * ABtemp * (x_5_0_1 - CDcom * x_5_0_2);
    LOCSTORE(store, 48, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_30_0_0 + WPtempy * x_30_0_1 + 3.000000 * ABtemp * (x_15_0_0 - CDcom * x_15_0_1);
    LOCSTORE(store, 39, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_30_0_0 + WPtempx * x_30_0_1;
    QUICKDouble x_31_0_0 = Ptempy * x_19_0_0 + WPtempy * x_19_0_1;
    QUICKDouble x_31_0_1 = Ptempy * x_19_0_1 + WPtempy * x_19_0_2;
    LOCSTORE(store, 41, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_31_0_0 + WPtempy * x_31_0_1 + ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    LOCSTORE(store, 40, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_31_0_0 + WPtempx * x_31_0_1;
    QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
    QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
    QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
    QUICKDouble x_1_0_3 = Ptempx * VY_3 + WPtempx * VY_4;
    QUICKDouble x_1_0_4 = Ptempx * VY_4 + WPtempx * VY_5;
    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
    QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 + 2.000000 * ABtemp * (x_1_0_1 - CDcom * x_1_0_2);
    QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 + 2.000000 * ABtemp * (x_1_0_2 - CDcom * x_1_0_3);
    QUICKDouble x_32_0_0 = Ptempx * x_17_0_0 + WPtempx * x_17_0_1 + 3.000000 * ABtemp * (x_7_0_0 - CDcom * x_7_0_1);
    QUICKDouble x_32_0_1 = Ptempx * x_17_0_1 + WPtempx * x_17_0_2 + 3.000000 * ABtemp * (x_7_0_1 - CDcom * x_7_0_2);
    LOCSTORE(store, 53, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_32_0_0 + WPtempx * x_32_0_1 + 4.000000 * ABtemp * (x_17_0_0 - CDcom * x_17_0_1);
    QUICKDouble x_33_0_0 = Ptempy * x_18_0_0 + WPtempy * x_18_0_1 + 3.000000 * ABtemp * (x_8_0_0 - CDcom * x_8_0_1);
    QUICKDouble x_33_0_1 = Ptempy * x_18_0_1 + WPtempy * x_18_0_2 + 3.000000 * ABtemp * (x_8_0_1 - CDcom * x_8_0_2);
    LOCSTORE(store, 54, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_33_0_0 + WPtempy * x_33_0_1 + 4.000000 * ABtemp * (x_18_0_0 - CDcom * x_18_0_1);
    LOCSTORE(store, 51, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_33_0_0 + WPtempx * x_33_0_1;
    QUICKDouble x_34_0_0 = Ptempz * x_19_0_0 + WPtempz * x_19_0_1 + 3.000000 * ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_34_0_1 = Ptempz * x_19_0_1 + WPtempz * x_19_0_2 + 3.000000 * ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    LOCSTORE(store, 55, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempz * x_34_0_0 + WPtempz * x_34_0_1 + 4.000000 * ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    LOCSTORE(store, 49, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempx * x_34_0_0 + WPtempx * x_34_0_1;
    LOCSTORE(store, 47, 0, STOREDIM, STOREDIM) STORE_OPERATOR Ptempy * x_34_0_0 + WPtempy * x_34_0_1;
    // [HS|SS] integral - End 

}

