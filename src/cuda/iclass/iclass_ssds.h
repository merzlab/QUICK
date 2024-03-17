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

    // [SS|DS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    LOCSTORE(store, 0, 7, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
    LOCSTORE(store, 0, 8, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
    LOCSTORE(store, 0, 4, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    LOCSTORE(store, 0, 9, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
    LOCSTORE(store, 0, 6, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
    LOCSTORE(store, 0, 5, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
    // [SS|DS] integral - End 

}

