/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 10/04/2022                            !
  !                                                                     ! 
  ! Copyright (C) 2021-2022 Merz lab                                    !
  ! Copyright (C) 2021-2022 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions for obtaining walltimes.        !
  !---------------------------------------------------------------------!
*/

#ifndef __TIME_H__
#define __TIME_H__

extern "C" {
void init_ref_time_();
void walltime_(double* t);
}

#endif

