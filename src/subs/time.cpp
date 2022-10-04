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

#include <stdio.h>
#include <sys/time.h>
#include "time.hpp"

// declare structs to store Epoch times
static struct timeval refTime, endTime;

// sets reference epoch time
void init_ref_time_(){
    gettimeofday(&refTime, 0);
    return;
}

// calculate and return time elapsed since reference
void walltime_(double* t){

    gettimeofday(&endTime, 0);    
    long seconds = endTime.tv_sec - refTime.tv_sec;
    long microseconds = endTime.tv_usec - refTime.tv_usec;
    *t = seconds+microseconds*1e-6; 
    return;
}
