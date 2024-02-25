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

    // [SS|FS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble x_0_2_0 = Qtempy * VY_0 + WQtempy * VY_1;
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_0_2_1 = Qtempy * VY_1 + WQtempy * VY_2;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_0_2_2 = Qtempy * VY_2 + WQtempy * VY_3;
    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
    LOCSTORE(store, 0, 11, STOREDIM, STOREDIM) += Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    QUICKDouble x_0_3_0 = Qtempz * VY_0 + WQtempz * VY_1;
    QUICKDouble x_0_3_1 = Qtempz * VY_1 + WQtempz * VY_2;
    QUICKDouble x_0_3_2 = Qtempz * VY_2 + WQtempz * VY_3;
    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
    LOCSTORE(store, 0, 15, STOREDIM, STOREDIM) += Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    LOCSTORE(store, 0, 10, STOREDIM, STOREDIM) += Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
    LOCSTORE(store, 0, 13, STOREDIM, STOREDIM) += Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    QUICKDouble x_0_1_0 = Qtempx * VY_0 + WQtempx * VY_1;
    QUICKDouble x_0_1_1 = Qtempx * VY_1 + WQtempx * VY_2;
    QUICKDouble x_0_1_2 = Qtempx * VY_2 + WQtempx * VY_3;
    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2 + CDtemp * (VY_1 - ABcom * VY_2);
    LOCSTORE(store, 0, 17, STOREDIM, STOREDIM) += Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2.000000 * CDtemp * (x_0_1_0 - ABcom * x_0_1_1);
    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2 + CDtemp * (VY_1 - ABcom * VY_2);
    LOCSTORE(store, 0, 18, STOREDIM, STOREDIM) += Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2.000000 * CDtemp * (x_0_2_0 - ABcom * x_0_2_1);
    LOCSTORE(store, 0, 12, STOREDIM, STOREDIM) += Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1 + CDtemp * (VY_0 - ABcom * VY_1);
    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2 + CDtemp * (VY_1 - ABcom * VY_2);
    LOCSTORE(store, 0, 19, STOREDIM, STOREDIM) += Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2.000000 * CDtemp * (x_0_3_0 - ABcom * x_0_3_1);
    LOCSTORE(store, 0, 16, STOREDIM, STOREDIM) += Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
    LOCSTORE(store, 0, 14, STOREDIM, STOREDIM) += Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
    // [SS|FS] integral - End 

}

