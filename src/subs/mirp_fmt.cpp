/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 02/05/2021                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains wrapper functions required to use FmT     !
  ! function from MIRP library.                                         !
  !                                                                     ! 
  !---------------------------------------------------------------------!
*/
#ifdef MIRP
#include <mirp/kernels/boys.h>
#include <mirp/math.h>

#include <iostream>
#include "mirp_fmt.hpp"

// Fortran accessible fmt function from mirp library
void mirp_fmt_(int *m, double *t, double *fmt){

  double *F = new double[*m+1]();

  mirp_boys_exact(F, *m, *t);

  for(int i=0; i<=*m+1; ++i) fmt[i]=F[i];

}
#endif
