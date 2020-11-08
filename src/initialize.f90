!
!	initialize.f90
!	new_quick
!
!	Created by Yipu Miao on 5/09/2010.
!	Copyright 2011 University of Florida. All rights reserved.
!
! This file list initial step of Quick

!------------------------------------------------------
! Subroutine List:
!------------------------------------------------------
! initialize1(ierror)
! outputCopyright(ierror)
! allocateAtoms
! allocateAtoms_ECP
! allocatebasis
!------------------------------------------------------

subroutine initialize1(ierr)

   !  This subroutine is to initalize variables, set their default values
   !  most of them are called from interface "init". See modules' files to
   !  see detailed implementation

   use quick_molspec_module
   use quick_method_module
   use quick_timer_module
   implicit none

   ! Parameter list
   integer ierr    ! Error Flag

#ifdef MPIV
   !--------------------MPI/ALL NODES--------------------------------
   ! MPI Initializer
   call initialize_quick_mpi()
   !------------------- End MPI  -----------------------------------
#endif

   call init(quick_method)     !initialize quick_method namelist
   call init(quick_molspec)    !initialize quick_molspec namelist

   ierr=0
   call cpu_time(timer_begin%TTotal) !Trigger time counter

   return

end subroutine initialize1


subroutine outputCopyright(io,ierr)

   !  Output Copyright information

   implicit none

   ! parameter list
   integer ierr    ! Error Flag
   integer io

   write(io,'(A107)') "| *********************************************************************************************************"
   write(io,'(A107)') "| **                                                                                                     **"
   write(io,'(A107)') "| **           888888888888                                                                              **"
   write(io,'(A107)') "| **         8888888888888888                                                                            **"
   write(io,'(A107)') "| **      888888888888888888                                                                             **"
   write(io,'(A107)') "| **     888888888888888888                                                                              **"
   write(io,'(A107)') "| **    888888888888888                                                                                  **"
   write(io,'(A107)') "| **   88888888888888888888                               88888                       8888:              **"
   write(io,'(A107)') "| **   8888888888888888888888Z                            88888                       8888:              **"
   write(io,'(A107)') "| **   888888888888888888888888?                          88888                       8888:              **"
   write(io,'(A107)') "| **   8888888888888      8888888                                        888888       8888:              **"
   write(io,'(A107)') "| **    88888888888         888888     8888:     88888    88888        888888888I     8888:    888888    **"
   write(io,'(A107)') "| **    8888888888           88888:    8888:     88888    88888      $888888888888    8888:   888888     **"
   write(io,'(A107)') "| **    I8Z 88888             88888    8888:     88888    88888    .888888     8888   8888: 888888       **"
   write(io,'(A107)') "| **    .8Z 88888             88888    8888:     88888    88888    $88888             8888:88888         **"
   write(io,'(A107)') "| **     8I 88888      .=88. .88888    8888:     88888    88888    88888              8888888888         **"
   write(io,'(A107)') "| **    :8  88888      888888$8888$    8888:     88888    88888    8888O              88888888888        **"
   write(io,'(A107)') "| **   ,7   +88888.     8888888888.    8888:     88888    88888    888888             88888O888888       **"
   write(io,'(A107)') "| **         $888888:.   .8888888      88888....888888    88888     888888     8888   8888:  888888      **"
   write(io,'(A107)') "| **          I8888888888888888888     888888888888888    88888     O888888888888O    8888:   888888     **"
   write(io,'(A107)') "| **            O888888888888888888     88888888888888    88888       88888888888$    8888:    888888    **"
   write(io,'(A107)') "| **               8888888Z     888      .8888I  88888    88888         8888888       8888:     888888   **"
   write(io,'(A107)') "| **                                                                                                     **"
   write(io,'(A107)') "| **                                                                                                     **"
   write(io,'(A107)') "| **                                         Copyright (c) 2020                                          **"
   write(io,'(A107)') "| **                          Regents of the University of California San Diego                          **"
   write(io,'(A107)') "| **                                    & Michigan State University                                      **"
   write(io,'(A107)') "| **                                        All Rights Reserved.                                         **"
   write(io,'(A107)') "| **                                                                                                     **"
   write(io,'(A107)') "| **                   This software provided pursuant to a license agreement containing                 **"
   write(io,'(A107)') "| **                   restrictions on its disclosure, duplication, and use. This software               **"
   write(io,'(A107)') "| **                   contains confidential and proprietary information, and may not be                 **"
   write(io,'(A107)') "| **                   extracted or distributed, in whole or in part, for any purpose                    **"
   write(io,'(A107)') "| **                   whatsoever, without the express written permission of the authors.                **"
   write(io,'(A107)') "| **                   This notice, and the associated author list, must be attached to                  **"
   write(io,'(A107)') "| **                   all copies, or extracts, of this software. Any additional                         **"
   write(io,'(A107)') "| **                   restrictions set forth in the license agreement also apply to this                **"
   write(io,'(A107)') "| **                   software.                                                                         **"
   write(io,'(A107)') "| *********************************************************************************************************"
   write(io,'(A107)') "|                                                                                                          "
   write(io,'(A107)') "| Cite this work as:                                                                                       "
   write(io,'(A107)') "| Manathunga, M.; Mu, D.; Miao, Y.;He, X.; Ayers,K; Brothers, E.; Götz, A.; Merz,K. M. QUICK-20.03         "
   write(io,'(A107)') "| University of California San Diego, CA and Michigan State University, East Lansing, MI, 2020             "
   write(io,'(A107)') "|                                                                                                          "
   write(io,'(A107)') "| If you have any comments or queries, please send us an email for technical support:                      "
   write(io,'(A107)') "| quick.merzlab@gmail.com                                                                                  "
   write(io,'(A107)') "|                                                                                                          "

   ierr=0

   return

end subroutine outputCopyright
