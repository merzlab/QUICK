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
    public :: initializeExport, finalizeExport, exportCoordinates, exportBasis, exportMO, &
              exportSCF, exportOPT

    type quick_molden_type
      integer :: iMoldenFile

      ! true if this a geometry optimization
      logical :: opt
      
      ! number of atoms in molecule
      integer :: natom

      ! atom symbols
      character(len=2), allocatable :: atom_symbol(:)

      ! number of scf iterations to converge
      integer, dimension(:), allocatable :: nscf_snapshots

      ! scf energy during each iteration
      double precision,dimension(:,:),allocatable :: e_snapshots

      ! geometry during optimization
      double precision,dimension(:,:,:),allocatable :: xyz_snapshots

      ! counter to keep track of number of snapshots
      integer :: iexport_snapshot
    
      ! temporary vector for reorganizing mo coefficients
      double precision, dimension(:), allocatable :: reord_mo_vec

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

    interface exportMO
        module procedure write_mo
    end interface exportMO

    interface exportSCF
        module procedure write_scf
    end interface exportSCF

    interface exportOPT
        module procedure write_opt
    end interface exportOpt

contains

subroutine write_coordinates(self, ierr)

    use quick_molspec_module, only: quick_molspec, xyz, natom
    use quick_constants_module, only : symbol, BOHRS_TO_A
    implicit none
    type (quick_molden_type), intent(in) :: self
    integer, intent(out) :: ierr
    integer :: i, j, k

    ! write atomic labels and coordinates
    write(self%iMoldenFile, '("[Atoms] (AU)")')
    do i=1,natom
        write(self%iMoldenFile,'(2x,A2,4x,I5,4x,I3)',advance='no') &
             symbol(quick_molspec%iattype(i)), i, quick_molspec%iattype(i)
        if (self%opt) then
           ! in case of geometry optimization write last stored geometry
           ! we need to do this because optimizers may return the geometry
           ! for the next step which may be stored in xyz
           k = self%iexport_snapshot - 1
           write(self%iMoldenFile, '(3(2x,F20.10))') (self%xyz_snapshots(j,i,k),j=1,3)
         else
           ! if it's a single point calculation we can use xyz
           ! we can't use xyz_snapshots because they have not been populated
           write(self%iMoldenFile, '(3(2x,F20.10))') (xyz(j,i),j=1,3)
        endif
    enddo

end subroutine write_coordinates

subroutine write_basis_info(self, ierr)

    use quick_basis_module, only: quick_basis, nshell, nbasis, ncontract
    use quick_molspec_module, only: natom
    implicit none
    type (quick_molden_type), intent(in) :: self
    integer, intent(out) :: ierr
    integer :: iatom, ishell, ibas, iprim, nprim, j, ishell_idx
    logical :: print_gto
    double precision :: val_gccoeff, xnorm

    ! write basis function information
    write(self%iMoldenFile, '("[GTO] (AU)")')
    do iatom=1, natom
        write(self%iMoldenFile, '(2x, I5)') iatom

        ! s basis functions
        do ishell=1, nshell
            if(quick_basis%katom(ishell) .eq. iatom) then
                nprim = quick_basis%kprim(ishell)
                if(quick_basis%ktype(ishell) .eq. 1) then
                    write(self%iMoldenFile, '(2x, "s", 4x, I2)') nprim
                    do iprim=1, nprim
                        ishell_idx=quick_basis%ksumtype(ishell)
                        write(self%iMoldenFile, '(2E20.10)') &
                        quick_basis%gcexpo(iprim,ishell_idx), quick_basis%unnorm_gccoeff(iprim,ishell_idx) 
                    enddo                    
                endif
            endif
        enddo

        ! s, p basis functions of sp shell
        do ishell=1, nshell
            if(quick_basis%katom(ishell) .eq. iatom) then
                nprim = quick_basis%kprim(ishell)
                if(quick_basis%ktype(ishell) .eq. 4) then
                    write(self%iMoldenFile, '(2x, "s", 4x, I2)') nprim
                    do iprim=1, nprim
                        ishell_idx=quick_basis%ksumtype(ishell)
                        write(self%iMoldenFile, '(2E20.10)') &
                        quick_basis%gcexpo(iprim,ishell_idx), quick_basis%unnorm_gccoeff(iprim,ishell_idx)
                    enddo
                    write(self%iMoldenFile, '(2x, "p", 4x, I2)') nprim
                    do iprim=1, nprim
                        ishell_idx=quick_basis%ksumtype(ishell)
                        write(self%iMoldenFile, '(2E20.10)') &
                        quick_basis%gcexpo(iprim,ishell_idx), (quick_basis%unnorm_gccoeff(iprim,ishell_idx+1))
                    enddo
                endif
            endif
        enddo
        
        ! p, d, and f basis functions
        do ishell=1, nshell
            if(quick_basis%katom(ishell) .eq. iatom) then
                nprim = quick_basis%kprim(ishell)
                print_gto=.false.
                if(quick_basis%ktype(ishell) .eq. 3) then
                    print_gto=.true.
                    write(self%iMoldenFile, '(2x, "p", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 6) then
                    print_gto=.true.
                    write(self%iMoldenFile, '(2x, "d", 4x, I2)') nprim
                elseif(quick_basis%ktype(ishell) .eq. 10) then
                    print_gto=.true.
                    write(self%iMoldenFile, '(2x, "f", 4x, I2)') nprim
                endif
                
                do iprim=1, nprim
                    if(print_gto) then
                        ishell_idx=quick_basis%ksumtype(ishell)
                        write(self%iMoldenFile, '(2E20.10)') &
                        quick_basis%gcexpo(iprim,ishell_idx), quick_basis%unnorm_gccoeff(iprim,ishell_idx)
                    endif
                enddo
            endif
        enddo

        write(self%iMoldenFile, '("")')
    enddo

end subroutine write_basis_info

subroutine write_mo(self, ierr)

    use quick_basis_module, only: quick_basis, nbasis
    use quick_calculated_module, only: quick_qm_struct
    use quick_scratch_module
    use quick_molspec_module, only: quick_molspec
    use quick_method_module, only: quick_method
    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr    
    integer :: i, j, k, neleca, nelecb
    character(len=5) :: lbl1
    double precision :: occnum, occval

    if(.not. allocated(self%reord_mo_vec)) allocate(self%reord_mo_vec(nbasis))

    write(self%iMoldenFile, '("[MO]")')

    if(.not.quick_method%unrst) then
        neleca = quick_molspec%nElec/2
        occval = 2.0d0
    else
        neleca = quick_molspec%nElec
        nelecb = quick_molspec%nElecb
        occval = 1.0d0
    endif

    do i=1, nbasis
        if(neleca .gt. 0 ) then
            occnum=occval
            neleca=neleca-1
        else
            occnum=0.0d0
        endif

        write(lbl1,'(I5)') i
        write(self%iMoldenFile, '(A11)') "  Sym= a"//trim(adjustl(lbl1))
        write(self%iMoldenFile, '("  Ene= ", E20.10)') quick_qm_struct%E(i)
        write(self%iMoldenFile, '(2x, "Spin= Alpha" )') 

        ! write orbital occupation numbers
        write(self%iMoldenFile, '("  Occup= ", F10.8)') occnum 

        ! reorder molecular orbital coefficients        
         call reorder_mo_coeffs(quick_qm_struct%co, quick_basis%KLMN, nbasis, i, self%reord_mo_vec, ierr)

        ! write molecular orbital coefficients  
        do j=1, nbasis
            write(self%iMoldenFile, '(I4,F15.10)') j, self%reord_mo_vec(j)
        enddo
    enddo

    if(quick_method%unrst) then
        do i=1, nbasis
            if(nelecb .gt. 0 ) then
                occnum=occval
                nelecb=nelecb-1
            else
                occnum=0.0d0
            endif
    
            write(lbl1,'(I5)') i
            write(self%iMoldenFile, '(A11)') "  Sym= b"//trim(adjustl(lbl1))
            write(self%iMoldenFile, '("  Ene= ",E20.10)') quick_qm_struct%Eb(i)
            write(self%iMoldenFile, '(2x, "Spin= Beta" )')
    
            ! write orbital occupation numbers
            write(self%iMoldenFile, '("  Occup= ", F10.8)') occnum

            ! reorder molecular orbital coefficients        
            call reorder_mo_coeffs(quick_qm_struct%cob, quick_basis%KLMN, nbasis, i, self%reord_mo_vec, ierr)
    
            ! write molecular orbital coefficients        
            do j=1, nbasis
                write(self%iMoldenFile, '(I4,F15.10)') j, self%reord_mo_vec(j)
            enddo
        enddo
    endif

    if(allocated(self%reord_mo_vec)) deallocate(self%reord_mo_vec)

end subroutine write_mo

subroutine write_scf(self, ierr)

    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr
    integer :: i, j
    character(len=9) :: label

    write(self%iMoldenFile, '("[SCFCONV]")')

    do i=1, self%iexport_snapshot-1
       if (i == 1) then
          label = "scf-first"
       else if (i == (self%iexport_snapshot - 1) ) then
          label = "scf-last"
       else
          label = "scf"
       end if
       write(self%iMoldenFile, '(2x, a, 2x, I3, 2x, "THROUGH", 2x, I3)') trim(label), 1, self%nscf_snapshots(i)
        do j=1, self%nscf_snapshots(i)
            write(self%iMoldenFile, '(2x, E16.10)') self%e_snapshots(j,i)
        enddo
    enddo

end subroutine write_scf

subroutine write_opt(self, ierr)

    use quick_constants_module, only : BOHRS_TO_A
    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr
    integer :: i, j, k
    character(len=8) :: lbl1
    character(len=2) :: lbl2

    write(self%iMoldenFile, '("[GEOCONV]")')

    write(self%iMoldenFile, '("energy")')
    do i=1, self%iexport_snapshot-1
        write(self%iMoldenFile, '(2x, E16.10)') self%e_snapshots(self%nscf_snapshots(i),i)
    enddo

    write(self%iMoldenFile, '("[GEOMETRIES] (XYZ)")')

    do k=1, self%iexport_snapshot-1
       write(self%iMoldenFile, '(2x, I5)') self%natom
       write(self%iMoldenFile, '("")')
        do i=1,self%natom
           write(self%iMoldenFile,'(A2, 2x, 3F14.6)') &
                self%atom_symbol(i), (self%xyz_snapshots(j,i,k)*BOHRS_TO_A,j=1,3)
        enddo    
    enddo

end subroutine write_opt

subroutine initialize_molden(self, ierr)
    
    use quick_files_module, only : iMoldenFile, moldenFileName
    use quick_method_module, only: quick_method
    use quick_molspec_module, only: natom, quick_molspec
    use quick_constants_module, only : symbol
    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr
    integer :: i, dimy

    self%iMoldenFile = iMoldenFile
    self%iexport_snapshot=1
    self%natom = natom
    dimy = 1
    if (quick_method%opt) then
       self%opt = .true.
       dimy = quick_method%iopt
    else
       self%opt = .false.
    end if

    ! allocate memory
    if(.not. allocated(self%atom_symbol)) allocate(self%atom_symbol(natom))
    if(.not. allocated(self%nscf_snapshots)) allocate(self%nscf_snapshots(quick_method%iscf))
    if(.not. allocated(self%e_snapshots)) allocate(self%e_snapshots(quick_method%iscf, dimy))
    if(.not. allocated(self%xyz_snapshots)) allocate(self%xyz_snapshots(3, natom, dimy))

    ! store atom symbols
    do i = 1, natom
       self%atom_symbol(i) = symbol(quick_molspec%iattype(i))
    end do

    ! open file
    call quick_open(self%iMoldenFile,moldenFileName,'U','F','R',.false.,ierr)

    write(self%iMoldenFile, '("[Molden Format]")')

end subroutine initialize_molden

subroutine finalize_molden(self, ierr)

    implicit none
    type (quick_molden_type), intent(inout) :: self
    integer, intent(out) :: ierr

    ! deallocate memory
    if(allocated(self%atom_symbol)) deallocate(self%atom_symbol)
    if(allocated(self%nscf_snapshots)) deallocate(self%nscf_snapshots)
    if(allocated(self%e_snapshots)) deallocate(self%e_snapshots)
    if(allocated(self%xyz_snapshots)) deallocate(self%xyz_snapshots)

    ! close file
    close(self%iMoldenFile)

end subroutine finalize_molden

subroutine reorder_mo_coeffs(co, KLMN, nbasis, i, reord_mo_vec, ierr)

    implicit none
    double precision, intent(in) :: co(:,:)
    integer, intent(in) :: KLMN(:,:)
    integer, intent (in) :: nbasis, i 
    double precision, intent(inout) :: reord_mo_vec(:)
    integer, intent(out) :: ierr
    integer :: j
   
    j=1
    do while (j <= nbasis)
        reord_mo_vec(j)=co(j,i)
        ! order d functions. xx is the first, followed by yy, zz, xy, xz, yz
        if(KLMN(1,j) .eq. 2) then
            reord_mo_vec(j+1)=co(j+2,i)
            reord_mo_vec(j+2)=co(j+5,i)
            reord_mo_vec(j+3)=co(j+1,i)
            reord_mo_vec(j+4)=co(j+3,i)
            reord_mo_vec(j+5)=co(j+4,i)
            j=j+5
        endif

        ! order f functions. xxx is the first, followed by yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
        if(KLMN(1,j) .eq. 3) then
            reord_mo_vec(j+1)=co(j+3,i) 
            reord_mo_vec(j+2)=co(j+9,i) 
            reord_mo_vec(j+3)=co(j+2,i) 
            reord_mo_vec(j+4)=co(j+1,i) 
            reord_mo_vec(j+5)=co(j+4,i)
            reord_mo_vec(j+6)=co(j+7,i)
            reord_mo_vec(j+7)=co(j+8,i)
            reord_mo_vec(j+8)=co(j+6,i)
            reord_mo_vec(j+9)=co(j+5,i)
            j=j+9
        endif

        j=j+1
    enddo 
    
end subroutine reorder_mo_coeffs

end module quick_molden_module
