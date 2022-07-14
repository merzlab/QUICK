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

subroutine write_coordinates(iMoldenFile, ierr)

    use quick_molspec_module, only: quick_molspec, xyz, natom
    use quick_constants_module, only :: symbol, BOHRS_TO_A
    implicit none
    integer, intent(in) :: iMoldenFile
    integer :: i, j

    ! write atomic labels and coordinates
    write(iMoldenFile, '("[Atoms] (Ang)")')
    do i=1,natom
        write(iMoldenFile,'("A2,4x,I5,4x,I3,4x,F10.4,4x,F10.4,4x,F10.4")') &
        symbol(quick_molspec%iattype(i)), i, quick_molspec%iattype(i), (xyz(j,i)*BOHRS_TO_A,j=1,3)
    enddo

end subroutine write_coordinates

subroutine initialize_molden(iMoldenFile, moldenFileName, ierr)
    
    use quick_molspec_module, only: quick_molspec, xyz, natom
    use quick_constants_module, only :: symbol
    implicit none
    integer, intent(in) :: iMoldenFile
    character(len=*) ::  moldenFileName
    integer, intent(out) :: ierr
    integer :: i, j

    ! open file
    call quick_open(iMoldenFile,moldenFileName,'U','F','R',.false.,ierr)
    CHECK_ERROR(ierr)

    write(iMoldenFile, '("[Molden Format]")')

end subroutine initialize_molden

subroutine finalize_molden(iMoldenFile, ierr)

    implicit none
    integer, intent(in) :: iMoldenFile
    integer, intent(out) :: ierr

    ! close file
    call close(iMoldenFile)

end subroutine finalize_molden

end module quick_molden_module
