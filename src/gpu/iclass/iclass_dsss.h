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

}

