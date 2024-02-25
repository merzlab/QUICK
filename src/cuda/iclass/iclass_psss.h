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

    // [PS|SS] integral - Start
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble VY_1 = VY(0, 0, 1);
    LOCSTORE(store, 1, 0, STOREDIM, STOREDIM) += Ptempx * VY_0 + WPtempx * VY_1;
    LOCSTORE(store, 2, 0, STOREDIM, STOREDIM) += Ptempy * VY_0 + WPtempy * VY_1;
    LOCSTORE(store, 3, 0, STOREDIM, STOREDIM) += Ptempz * VY_0 + WPtempz * VY_1;
    // [PS|SS] integral - End 

}

