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

    // [PS|PS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    LOCSTORE(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
    LOCSTORE(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
    LOCSTORE(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY_1;
    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
    LOCSTORE(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
    LOCSTORE(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY_1;
    LOCSTORE(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    LOCSTORE(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY_1;
    LOCSTORE(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
    LOCSTORE(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
    // [PS|PS] integral - End 

}

