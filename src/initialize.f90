#include "config.h"
!
!        initialize.f90
!        new_quick
!
!        Created by Yipu Miao on 5/09/2010.
!        Copyright 2011 University of Florida. All rights reserved.
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

#ifdef MPI
   !--------------------MPI/ALL NODES--------------------------------
   ! MPI Initializer
   call MPI_initialize()
   !------------------- End MPI  -----------------------------------
#endif

   call init(quick_method)     !initialize quick_method namelist
   call init(quick_molspec)    !initialize quick_molspec namelist

   ierr = 0
   call cpu_time(timer_begin%TTotal) !Trigger time counter

   return

end subroutine initialize1

subroutine outputCopyright(io, ierr)

   !  Output Copyright information

   implicit none

   ! parameter list
   integer ierr    ! Error Flag
   integer io

   write (io, *) " **************************************************************************"
   write (io, *) " **                            QUICK                                     **"
   write (io, *) " **                                                                      **"
   write (io, *) " **                        Copyright (c) 2011                            **"
   write (io, *) " **                Regents of the University of Florida                  **"
   write (io, *) " **                       All Rights Reserved.                           **"
   write (io, *) " **                                                                      **"
   write (io, *) " **  This software provided pursuant to a license agreement containing   **"
   write (io, *) " **  restrictions on its disclosure, duplication, and use. This software **"
   write (io, *) " **  contains confidential and proprietary information, and may not be   **"
   write (io, *) " **  extracted or distributed, in whole or in part, for any purpose      **"
   write (io, *) " **  whatsoever, without the express written permission of the authors.  **"
   write (io, *) " **  This notice, and the associated author list, must be attached to    **"
   write (io, *) " **  all copies, or extracts, of this software. Any additional           **"
   write (io, *) " **  restrictions set forth in the license agreement also apply to this  **"
   write (io, *) " **  software.                                                           **"
   write (io, *) " **************************************************************************"
   write (io, *)
   write (io, *) " Cite this work as:"
   write (io, *) " Miao, Y.: He, X.: Ayers,K: Brothers, E.: Merz,K. M. QUICK;"
   write (io, *) " University of Florida, Gainesville, FL, 2010"
   write (io, *)
   write (io, *) " If you have any comment or queries, please send email for technic support:"
   write (io, *) " quick@qtp.ufl.edu"
   write (io, *)

   ierr = 0

   return

end subroutine outputCopyright
