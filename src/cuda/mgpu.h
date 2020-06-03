/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 04/29/2020                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains definitions relevent for mgpu.h and must  !
  ! be only included in mgpu.cu. Definitions required for all cuda      !
  ! source must be included in gpu.h.                                   !
  !---------------------------------------------------------------------!
*/

#ifndef QUICK_MGPU_H
#define QUICK_MGPU_H

   // device information
   int validDevCount = 0;      // Number of devices that can be used
   int* gpu_dev_id;            // Holds device IDs  

#endif
