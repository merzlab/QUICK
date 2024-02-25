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

    // [SS|PS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    LOCSTORE(store, 0, 1, STOREDIM, STOREDIM) += Qtempx * VY_0 + WQtempx * VY_1;
    LOCSTORE(store, 0, 2, STOREDIM, STOREDIM) += Qtempy * VY_0 + WQtempy * VY_1;
    LOCSTORE(store, 0, 3, STOREDIM, STOREDIM) += Qtempz * VY_0 + WQtempz * VY_1;
    // [SS|PS] integral - End 

}

