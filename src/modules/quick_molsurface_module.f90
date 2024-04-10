#include "util.fh"
!---------------------------------------------------------------------!
! Created by Etienne Palos on   xx/xx/2024                            !
!                                                                     ! 
! Copyright (C) 2024-2025 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!-----------------------------------------------------------------------------------!
! This module contains subroutines related to generating molecular surfaces         !
! These molecular surfaces are used to evaluate ESP, EFIELD, and other properties.  ! 
!--------------------------------------------------------------------------------- -!

module quick_molsurface_module

   implicit none
   private

   public :: computevdWsurface, computeMSKsurfaces

!   ! vdW Surface := Van der Waals Surface, grid points are generated on the surface defined by R_vdW
   interface computevdWsurface
   module procedure generate_vdW_surface
 end interface computevdWsurface

!   ! MKS Surface := Van der Waals Surface at c*R_vdw, where c = {1.4, 1.6, 1.8, 2.0}
 interface computeMSKsurfaces
   module procedure generate_MKS_surfaces
 end interface computeMSKsurfaces

 contains

!  atomic_symbols = [
!     "H",                                                                                                                                    "He",
!     "Li",   "Be",                                                                                   "B",    "C",    "N",    "O",    "F",    "Ne",
!     "Na",   "Mg",                                                                                   "Al",   "Si",   "P",    "S",    "Cl",   "Ar",
!     "K",    "Ca",   "Sc",   "Ti",   "V",    "Cr",   "Mn",   "Fe",   "Co",   "Ni",   "Cu",   "Zn",   "Ga",   "Ge",   "As",   "Se",   "Br",   "Kr",
!     "Rb",   "Sr",   "Y",    "Zr",   "Nb",   "Mo",   "Tc",   "Ru",   "Rh",   "Pd",   "Ag",   "Cd",   "In",   "Sn",   "Sb",   "Te",   "I",    "Xe",
!     "Cs"
!  ]

!  # list of free atomic polarizabilities (bohr^3) in order of atomic number
! free_polarizabilities = [ # Source: 2018 Table of static dipole polarizabilities of the neutral elements in the periodic table (Peter Schwerdtfeger & Jeffrey K. Nagle)
!                           # DOI: 10.1080/00268976.2018.1535143
!                           # https://www.tandfonline.com/doi/pdf/10.1080/00268976.2018.1535143?needAccess=true
!      4.50711,                                                                                                              1.38375,
!     164.1125, 37.74,                                                                        20.5, 11.3,  7.4,   5.3, 3.74, 2.66110,
!        162.7,  71.2,                                                                        57.8, 37.3,   25,  19.4, 14.6,  11.083,
!        289.8, 160.8,    97,   100,    87,    83,    68,    62,    55,    49,  46.5, 38.67,    50,   40,   30,  28.9,   21,   16.78,
!        319.8, 197.2,   162,   112,    98,    87,    79,    72,    66, 26.14,    55,    46,    65,   53,   43,    38, 32.9,   27.32,
!        400

!        # list of van der Waals radii (vdw) (angstroms) in order of atomic number
!        vdw_radii = [ # Source: ptable.com for elements unless in list below.
!                      # Be, B, Al, Ca (Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3658832/)
!                # Rb, Cs (Source: doi:10.1021/jp8111556)
!                      # vdw_radius is -1.0, if a definite value cannot be found. 
!            1.2,                                                                                                                                    1.4,
!            1.82,   1.53,                                                                                   1.92,   1.70,   1.55,   1.52,   1.47,   1.54,
!            2.27,   1.73,                                                                                   1.84,   2.10,   1.80,   1.80,   1.75,   1.88,
!            2.75,   2.31,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   1.63,   1.40,   1.39,   1.87,   -1.0,   1.85,   1.90,   1.85,   2.02,
!            3.03,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   -1.0,   1.63,   1.72,   1.58,   1.93,   2.17,   -1.0,   2.06,   1.98,   2.16,
!            3.43
!        ]


   subroutine generate_vdW_surface
!   subroutine generate_vdW_surface(ierr)

!      use quick_gridpoints_module
!      use quick_files_module
!      use quick_exception_module

! #ifdef CEW 
!      use quick_cew_module, only : quick_cew
! #endif
  
!      implicit double precision(a-h,o-z)
  
!      logical :: present, MPIsaved, readSAD
!      double precision:: xyzsaved(3,natom)
!      character(len=80) :: keywd
!      character(len=20) :: tempstring
!      character(len=340) :: sadfile
!      integer natomsaved
!      type(quick_method_type) quick_method_save
!      type(quick_molspec_type) quick_molspec_save
!      integer, intent(inout) :: ierr
!      logical :: use_cew_save
  
!      ! first save some important value
!      quick_method_save=quick_method
!      quick_molspec_save=quick_molspec
!      ! quick_molspec_type has pointers which may lead to memory leaks
!      ! therefore, assign individual variable values
!   !   quick_molspec_save%imult = quick_molspec%imult
!   !   quick_molspec_save%nelec = quick_molspec%nelec
  
!      natomsaved=natom
!      xyzsaved=xyz
!      MPIsaved=bMPI
  
!      istart = 1
!      ifinal = 80
!      ibasisstart = 1
!      ibasisend = 80
  
!      ! Then give them new value
!      bMPI=.false.
!      quick_molspec%imult=0
!      quick_method%HF=.true.
!      quick_method%DFT=.false.
!      quick_method%UNRST=.true.
!      quick_method%ZMAT=.false.
!      quick_method%divcon=.false.
!      quick_method%nodirect=.false.
!      call allocate_mol_sad(quick_molspec%iatomtype)
  
  
!      if (master) then
!         call PrtAct(ioutfile,"Begin SAD initial guess")
!         !-------------------------------------------
!         ! First, find atom type and initialize
!         !-------------------------------------------
  
!         natom=1
!         do I=1,3
!            xyz(I,1) = 0.0d0
!         enddo
  
!         do iitemp=1,quick_molspec%iatomtype
!            write(ioutfile,'(" For Atom Kind = ",i4)') iitemp
  
!            ! if quick is called through api multiple times, this is necessary
!            if(wrtStep .gt. 1) then
!              call deallocate_calculated
!            endif
  
!            do i=1,90
!               if(symbol(i).eq.quick_molspec%atom_type_sym(iitemp))then
!                  quick_molspec%imult = spinmult(i)
!                  quick_molspec%chg(1)=i
!                  quick_molspec%iattype(1)=i
!                  write(ioutfile,'(" ELEMENT = ",a)') symbol(i)
!               endif
!            enddo
!            if (quick_molspec%imult /= 1) quick_method%UNRST= .TRUE.
!            quick_molspec%nelec = quick_molspec%iattype(1)
!            if ((quick_method%DFT .OR. quick_method%SEDFT).and.quick_method%isg.eq.1) &
!                  call gridformSG1()
!            call check_quick_method_and_molspec(ioutfile,quick_molspec,quick_method)
  
!            !-------------------------------------------
!            ! At this point we have the positions and identities of the atoms. We also
!            ! have the number of electrons. Now we must assign basis functions. This
!            ! is done in a subroutine.
!            !-------------------------------------------
!            nsenhai=1
!            call readbasis(nsenhai,0,0,0,0,ierr)
!            CHECK_ERROR(ierr)
           
!            atombasis(iitemp)=nbasis
!            write (ioutfile,'(" BASIS FUNCTIONS = ",I4)') nbasis
  
!            if(nbasis < 1) then
!                   call PrtErr(iOutFile,'Unable to find basis set information for this atom.')
!                   call PrtMsg(iOutFile,'Update the corresponding basis set file or use a different basis set.')
!                   call quick_exit(iOutFile,1)
!            endif
  
!            ! if quick is called through api multiple times, this is necessary
!            if(wrtStep .gt. 1) then
!              call dealloc(quick_qm_struct)
!            endif
  
!            quick_qm_struct%nbasis => nbasis
!            call alloc(quick_qm_struct)
!            call init(quick_qm_struct)
  
!            ! this following subroutine is as same as normal basis set normlization
!            call normalize_basis()
!            if (quick_method%ecp) call store_basis_to_ecp()
!            !if (quick_method%DFT .OR. quick_method%SEDFT) call get_sigrad
  
!            ! Initialize Density arrays. Create initial density matrix guess.
!            present = .false.
!            if (quick_method%readdmx) inquire (file=dmxfilename,exist=present)
!            if (present) then
!               return
!            else
!               ! Initial Guess
!               diagelement=dble(quick_molspec%nelec)/dble(nbasis)
!               diagelementb=dble(quick_molspec%nelecb)/dble(nbasis)+1.d-8
!               do I=1,nbasis
!                  quick_qm_struct%dense(I,I)=diagelement
!                  quick_qm_struct%denseb(I,I)=diagelementb
!               enddo
!            endif
  
!            ! AWG Check if SAD file is present when requesting readSAD
!            ! AWG If not present fall back to computing SAD guess
!            ! AWG note the whole structure of this routine should be improved
!            readSAD = quick_method%readSAD
!            !readSAD = .false.
!            if (readSAD) then
!               sadfile = trim(sadGuessDir) // '/' // &
!                              trim(quick_molspec%atom_type_sym(iitemp))
!               inquire (file=sadfile, exist=present)
!               if (.not. present) readSAD = .false.
!            end if
 
!            ! From SCF calculation to get initial density guess
!            if(readSAD) then
  
!               open(212,file=sadfile)  !Read from sadfile
!               do i=1,nbasis
!                  do j=1,nbasis
!                     read(212,*) ii,jj,temp
!                     atomdens(iitemp,ii,jj)=temp
!                  enddo
!               enddo
!               close(212)
  
!            else

! #ifdef CEW
!               use_cew_save = quick_cew%use_cew
!               quick_cew%use_cew = .false.
! #endif
              
!               ! Compute SAD guess
!               call sad_uscf(.true., ierr)
!               do i=1,nbasis
!                  do j=1,nbasis
!                     atomdens(iitemp,i,j)=quick_qm_struct%dense(i,j)+quick_qm_struct%denseb(i,j)
!                  enddo
!               enddo
! #ifdef CEW
!               quick_cew%use_cew = use_cew_save
! #endif              
!               ! write SAD guess if requested
!               if(quick_method%writeSAD) then
!                  sadfile = trim(quick_molspec%atom_type_sym(iitemp))
!                  open(213,file=sadfile)
!                  do i=1,nbasis
!                     do j=1,nbasis
!                        write(213,*) i,j,atomdens(iitemp,i,j)
!                     enddo
!                  enddo
!                  close(213)             
!               endif           
  
!            endif
  
!            call deallocate_calculated
!            call dealloc(quick_qm_struct)
!         enddo
!         call PrtAct(ioutfile,"Finish SAD initial guess")
!      endif
  
!      natom=natomsaved
!      xyz=xyzsaved
  
!      quick_method=quick_method_save
!      quick_molspec=quick_molspec_save
!   !   quick_molspec%imult = quick_molspec_save%imult
!   !   quick_molspec%nelec = quick_molspec_save%nelec
  
!      bMPI=MPIsaved
  
!      return
  
!   end subroutine generate_vdW_surface

!   !!! MKS SURFACES 

!   subroutine scale_vdW_surface

!     integer :: i

!     ! Scale the vdW surface by a factor of c
!     ! c = {1.4, 1.6, 1.8, 2.0}

   end subroutine scale_vdW_surface

   subroutine generate_MKS_surfaces

!         ! This subroutine generates the MKS surfaces for the molecule
!         ! The MKS surfaces are generated by scaling the vdW surface by a factor of c
!         ! where c = {1.4, 1.6, 1.8, 2.0}
!         ! The MKS surfaces are used to evaluate ESP, EFIELD, and other properties
    
!         integer :: i
    
!         ! Generate the vdW surface
!         call generate_vdW_surface
    
!         ! Scale the vdW surface by a factor of c
!         ! c = {1.4, 1.6, 1.8, 2.0}
!         do i = 1, 4
!             call scale_vdW_surface(i)
!         end do

!     should be implemented in the future
!       call generate_vdW_surface
!       call scale_vdW_surface(i)


   end subroutine generate_MKS_surfaces


end module quick_molsurface_module
