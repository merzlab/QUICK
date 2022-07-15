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
    private

    public :: quick_molden
    public :: initializeExport, finalizeExport, exportCoordinates, exportBasis

    type quick_molden_type
        integer :: iMoldenFile
    end type quick_molden_type

    type (quick_molden_type),save:: quick_molden

    interface initializeExport
        module procedure initialize_molden
    end interface initializeExport

    interface finalizeExport
        module procedure finalize_molden
    end interface finalizeExport

    interface exportCoordinates
        module procedure write_coordinates
    end interface exportCoordinates

    interface exportBasis
        module procedure write_basis_info
    end interface exportBasis
contains

subroutine write_coordinates(self, ierr)

    use quick_molspec_module, only: quick_molspec, xyz, natom
    use quick_constants_module, only : symbol, BOHRS_TO_A
    implicit none
    type (quick_molden_type), intent(in) :: self
    integer, intent(out) :: ierr
    integer :: i, j

    ! write atomic labels and coordinates
    write(self%iMoldenFile, '("[Atoms] (Ang)")')
    do i=1,natom
        write(self%iMoldenFile,'(2x,A2,4x,I5,4x,I3,4x,F10.4,4x,F10.4,4x,F10.4)') &
        symbol(quick_molspec%iattype(i)), i, quick_molspec%iattype(i), (xyz(j,i)*BOHRS_TO_A,j=1,3)
    enddo

end subroutine write_coordinates

subroutine write_basis_info(self, ierr)

    use quick_basis_module, only: quick_basis, nshell, nbasis, aexp
    use quick_molspec_module, only: natom
    implicit none
    type (quick_molden_type), intent(in) :: self
    integer, intent(out) :: ierr
    integer :: iatom, ishell, ibas, iprim, nprim, ibasInit, ibasEnd

    ! write basis function information
    write(self%iMoldenFile, '("[GTO] (AU)")')
    do iatom=1, natom
        write(self%iMoldenFile, '(2x, I5)') iatom

        do ishell=1, nshell
            if(quick_basis%katom(ishell) .eq. iatom) then
                nprim = quick_basis%kprim(ishell)
                if(quick_basis%ktype(ishell) .eq. 1) then
                    write(self%iMoldenFile, '(2x, "s", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 3) then
                    write(self%iMoldenFile, '(2x, "p", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 4) then
                    write(self%iMoldenFile, '(2x, "sp", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 6) then
                    write(self%iMoldenFile, '(2x, "d", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 10) then
                    write(self%iMoldenFile, '(2x, "f", 4x, I2)') nprim
                endif

                ibasInit = quick_basis%kstart(ishell)
                
                if(ishell .eq. nshell) then
                    ibasEnd  = nbasis
                else
                    ibasEnd = quick_basis%kstart(ishell+1)-1
                endif
                
                do ibas=ibasInit, ibasEnd
                    do iprim=1, nprim
                        write(self%iMoldenFile, '(2x, E12.6, 2x, E12.6)') quick_basis%gcexpo(iprim, ibas), aexp(iprim, ibas)
                    enddo
                enddo
            endif
        enddo
    enddo

end subroutine write_basis_info

subroutine initialize_molden(self, ierr)
    
    use quick_files_module, only : iMoldenFile, moldenFileName

    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr
    integer :: i, j

    self%iMoldenFile = iMoldenFile

    ! open file
    call quick_open(self%iMoldenFile,moldenFileName,'U','F','R',.false.,ierr)

    write(self%iMoldenFile, '("[Molden Format]")')

end subroutine initialize_molden

subroutine finalize_molden(self, ierr)

    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr

    ! close file
    close(self%iMoldenFile)

end subroutine finalize_molden

end module quick_molden_module
