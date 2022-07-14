!---------------------------------------------------------------------!
! Created by Madu Manathunga on 07/14/2022                            !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

module quick_molden_module
    implicit none
    
contains

subroutine write_molden(iMoldenFile, moldenFileName, ierr)
    
    implicit none
    integer, intent(in) :: iMoldenFile
    character(len=*) ::  moldenFileName
    integer, intent(out) :: ierr

    ! open file
    call quick_open(iMoldenFile,moldenFileName,'U','F','R',.false.,ierr)
    CHECK_ERROR(ierr)

    

    ! close file
    call close(moldenFileName)
end subroutine write_molden




end module quick_molden_module
