!---------------------------------------------------------------------!
! Updated by Madu Manathunga on 05/28/2020                            !
!                                                                     !
! Previous contributors: Yipu Miao, Xio He, Alessandro Genoni,        !
!                         Ken Ayers & Ed Brothers                     !
!                                                                     !
! Copyright (C) 2020-2021 Merz lab                                    !
! Copyright (C) 2020-2021 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

#include "util.fh"

! molecule specification Module
module quick_molspec_module

   implicit none

   type quick_molspec_type

      ! number of atoms
      integer,pointer :: natom

      ! number of electron and beta electron
      integer :: nElec = 0
      integer :: nElecb = 0

      ! number of external atoms
      integer :: nExtAtom = 0

      ! multiplicity
      integer :: imult = 1

      ! molecular charge
      integer :: molchg = 0

      ! number of non-hydrogen atom
      integer :: nNonHAtom = 0

      ! number of hydrogen atom
      integer :: nHAtom = 0

      ! number of atom types
      integer :: iAtomType = 0

      ! symbol for respective atom type
      character(len=2), dimension(1:10) :: atom_type_sym

      ! distantce to the nearest atom
      double precision, dimension(:), allocatable :: distnbor

      ! distance matrix
      double precision, dimension(:,:), allocatable :: AtomDistance

      ! coordinate of atoms and external atoms
      double precision, dimension(:,:), pointer :: xyz => null()
      double precision, dimension(:,:), allocatable :: extxyz

      ! which atom type id every atom crosponds to
      integer,dimension(:),allocatable :: iattype

      ! atom charge, external atom charge and atom mass
      double precision, dimension(:), allocatable ::chg,extchg,iatmass

      ! basis set number
      integer,pointer:: nbasis

   end type quick_molspec_type

   type (quick_molspec_type),save:: quick_molspec
   double precision, dimension(:,:), allocatable,target :: xyz
   integer,target :: natom

   ! interface lists

   interface print
      module procedure print_quick_molspec
   end interface print

   interface alloc
      module procedure allocate_quick_molspec
   end interface alloc

   interface realloc
      module procedure reallocate_quick_molspec
   end interface realloc   

   interface init
      module procedure init_quick_molspec
   end interface init

#ifdef MPIV
   interface broadcast
      module procedure broadcast_quick_molspec
   end interface broadcast
#endif

   interface read
      module procedure read_quick_molspec
   end interface read

   interface read2
      module procedure read_quick_molspec_2
   end interface read2

   interface check
     module procedure check_quick_molspec
   end interface check

   interface dealloc
      module procedure deallocate_quick_molspec
   end interface dealloc

   interface set
      module procedure set_quick_molspec
   end interface set

contains

   !-------------------
   ! allocate
   !-------------------
   subroutine allocate_quick_molspec(self,ierr)
      use quick_exception_module
      implicit none
      integer i,j
      integer, intent(inout) :: ierr

      type (quick_molspec_type) self

      if (.not. allocated(xyz)) allocate(xyz(3,natom))
!      allocate(self%xyz(3,natom))
      if (.not. allocated(self%distnbor))  allocate(self%distnbor(natom))
      if (.not. allocated(self%iattype)) allocate(self%iattype(natom))
      if (.not. allocated(self%iatmass)) allocate(self%iatmass(natom))
      if (.not. allocated(self%chg)) allocate(self%chg(natom))
      if (.not. allocated(self%AtomDistance)) allocate(self%AtomDistance(natom,natom))
      do i=1,natom
         self%distnbor(i)=0
         self%iattype(i)=0
         self%iatmass(i)=0d0
         self%chg(i)=0d0
         do j=1,3
            xyz(j,i)=0d0
         enddo
         do j=1,natom
            self%AtomDistance(i,j)=0d0
         enddo
      enddo

      if (self%nextatom.gt.0) then
         if (.not. allocated(self%extxyz)) allocate(self%extxyz(3,self%nextatom))
         if (.not. allocated(self%extchg)) allocate(self%extchg(self%nextatom))
         do i=1,self%nextatom
            do j=1,3
               self%extxyz(j,i)=0d0
            enddo
            self%extchg(i)=0d0
         enddo
      endif

   end subroutine allocate_quick_molspec

   !-----------------------------
   ! subroutine to realloate data
   !-----------------------------

   subroutine reallocate_quick_molspec(self,ierr)

     use quick_exception_module

     implicit none
     
     type (quick_molspec_type), intent(inout) :: self
     integer, intent(inout) :: ierr
     integer :: current_size

     if(self%nextatom .gt. 0) then
       if(allocated(self%extchg)) current_size= size(self%extchg)
       if(current_size /= self%nextatom) then
         deallocate(self%extchg, stat=ierr)
         deallocate(self%extxyz, stat=ierr)
         allocate(self%extchg(self%nextatom), stat=ierr)
         allocate(self%extxyz(3,self%nextatom), stat=ierr)
         self%extchg=0.0d0
         self%extxyz=0.0d0
       endif
     endif

   end subroutine reallocate_quick_molspec

   !-------------------
   ! set initial value
   !-------------------
   subroutine init_quick_molspec(self,ierr)
      use quick_exception_module
      implicit none

      type (quick_molspec_type) self
      integer, intent(inout) :: ierr

      self%natom => natom
      self%nElec = 0
      self%nElecb = 0
      self%nExtAtom = 0
      self%imult = 1
      self%molchg = 0
      self%nNonHAtom = 0
      self%nHAtom = 0
      self%iAtomType = 0


   end subroutine init_quick_molspec

   !-------------------
   ! deallocate
   !-------------------
   subroutine deallocate_quick_molspec(self,ierr)
      use quick_exception_module
      implicit none

      type (quick_molspec_type) self
      integer, intent(inout) :: ierr

      if (allocated(xyz)) deallocate(xyz)
      if (allocated(self%distnbor)) deallocate(self%distnbor)
!      deallocate(self%xyz)
      if (allocated(self%iattype)) deallocate(self%iattype)
      if (allocated(self%iatmass)) deallocate(self%iatmass)
      if (allocated(self%chg)) deallocate(self%chg)

      ! if exist external charge
      if (self%nextatom.gt.0) then
        if (allocated(self%extxyz)) deallocate(self%extxyz)
        if (allocated(self%extchg)) deallocate(self%extchg)
      endif

   end subroutine deallocate_quick_molspec

#ifdef MPIV
   !-------------------
   ! broadcast variable list
   !-------------------
   subroutine broadcast_quick_molspec(self,ierr)
      use quick_mpi_module
      use quick_exception_module
      use mpi

      implicit none
      type (quick_molspec_type) self
      integer natom2
      integer, intent(inout) :: ierr

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%natom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      natom2=natom**2
      call MPI_BCAST(self%nElec,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%nElecb,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%nextatom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%imult,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%molchg,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%nNonHAtom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%nHAtom,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%iAtomType,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%atom_type_sym,20,mpi_character,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%distnbor,natom,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%AtomDistance,natom*natom,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      !call MPI_BCAST(self%xyz,natom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      !call MPI_BCAST(self%nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      if (self%nextatom.gt.0) then
         call MPI_BCAST(self%extxyz,self%nextatom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%extchg,self%nextatom,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif
      call MPI_BCAST(self%iattype,natom,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%chg,natom,mpi_integer,0,MPI_COMM_WORLD,mpierror)
   end subroutine broadcast_quick_molspec
#endif

   !----------------------
   ! read molecular specification phase 1
   ! this subroutine is to read charge, multiplicity, and number
   ! and kind of atom.
   !----------------------
  subroutine read_quick_molspec(self,input,isTemplate, hasKeywd, apiKeywd,ierr)

    use quick_constants_module
    use quick_exception_module

    implicit none
    type (quick_molspec_type) :: self
    integer, intent(inout) :: ierr
    integer :: input,rdinml,i,j,k
    integer :: ierror
    integer :: iAtomType
    integer :: nextatom
    integer :: check_iso
    double precision :: temp,rdnml
    character(len=200) :: keywd
    logical :: is_extcharge = .false.
    logical :: is_blank
    logical, intent(in)   :: isTemplate
    logical, intent(in)   :: hasKeywd
    character(len=200), intent(in) :: apikeywd

    !---------------------
    ! PART I
    !---------------------
    ! The first line is Keyword

    if( .not. hasKeywd ) then
      rewind(input)
      read (input,'(A132)') keywd
    else
      keywd = apikeywd
    endif

    call upcase(keywd,200)

    ! Read Charge
    if (index(keywd,'CHARGE=') /= 0) self%molchg = rdinml(keywd,'CHARGE')

    ! read multipilicity
    if (index(keywd,'MULT=') /= 0) self%imult = rdinml(keywd,'MULT')

    ! determine if external charge exists
    if (index(keywd,'EXTCHARGES') /= 0) is_extcharge=.true.

    ! get the atom number, type and number of external charges

  if( .not. isTemplate) then

    call findBlock(input,1)

    ! first is to read atom and atom kind
    iAtomType = 1
    natom = 0
    nextatom = 0
    do
       read(input,'(A80)',end=111,err=111) keywd
       i=1;j=80
       call upcase(keywd,80)
       call rdword(keywd,i,j)
       if (is_blank(keywd,1,80)) exit

       check_iso = index(keywd(:j),'ISO=')
       if (check_iso /=0) j=check_iso-2 

       do k=0,71
          if (keywd(i:j) == symbol(k)) then
             natom=natom+1
             ! check if atom type has been shown before
             if (.not.(any(self%atom_type_sym(1:iatomtype).eq.symbol(k)))) then
                !write(*,*) "Assigning value to atom_type_sym:", k, symbol(k)
                self%atom_type_sym(iAtomType)=symbol(k)
                iAtomType=iAtomType+1
             endif
          endif
       enddo
    enddo

    111     continue

    ! read external charge part
    if (is_extcharge)  then
       rewind(input)
       call findBlock(input,2)
       do
          read(input,'(A80)',end=112,err=112) keywd
          if (is_blank(keywd,1,80)) exit
          nextatom=nextatom+1
       enddo
    endif

    112     continue

    iAtomType=iAtomType-1
    self%iAtomType = iAtomType
    self%nextatom = nextatom
  endif

  end subroutine read_quick_molspec

   !----------------
   ! read external charge
   !----------------
   subroutine read_quick_molspec_2(self,input,ierr)
      use quick_constants_module
      use quick_exception_module

      implicit none
      ! parameter
      type (quick_molspec_type) self
      integer input
      integer, intent(inout) :: ierr
      ! inner varibles
      integer i,j,k,istart,ifinal,check_iso
      integer ierror
      double precision temp
      character(len=200) keywd


      rewind(input)

      call findBlock(input,1)

      do i=1,natom

         istart=1
         ifinal=80

         read (input,'(A80)') keywd
         call upcase(keywd,80)
         call rdword(keywd,istart,ifinal)

         !-----------------------------
         ! First, find the atom type and atom mass
         !-----------------------------
         check_iso=index(keywd(:ifinal),'ISO=')
         if (check_iso/=0) then
            do k=1,SYMBOL_MAX
               if (keywd(istart:check_iso-2) == symbol(k)) self%iattype(i)=k
            enddo
            call rdnum(keywd(:ifinal-1),check_iso+4,temp,ierror)
            self%iatmass(i)=temp
         else
            do k=1,SYMBOL_MAX
               if (keywd(istart:ifinal) == symbol(k)) self%iattype(i)=k
            enddo
            self%iatmass(i)=emass(self%iattype(i))
         endif

         !-----------------------------
         ! Next, find the xyz coordinates of the atom and convert to bohr.
         !-----------------------------
         do k=1,3
            istart=ifinal+1
            ifinal=80
            call rdword(keywd,istart,ifinal)
            call rdnum(keywd,istart,temp,ierror)
            xyz(k,i) = temp*A_TO_BOHRS
         enddo

      enddo

      self%xyz => xyz

      if (self%nextatom.gt.0) call read_quick_molespec_extcharges(self,input,ierr)

   end subroutine read_quick_molspec_2



    subroutine read_quick_molespec_extcharges(self,input,ierr)
        use quick_constants_module
        use quick_exception_module

        implicit none

        ! parameter
        type (quick_molspec_type) self
        integer input
        integer, intent(inout) :: ierr
        ! inner varibles
        integer i,j,k,istart,ifinal
        integer nextatom,ierror
        double precision temp
        character(len=200) keywd

        rewind(input)
        call findBlock(input,2)
        nextatom=self%nextatom

        do i=1,nextatom
            istart = 1
            ifinal = 80

            read(input,'(A80)') keywd

            do j=1,3
                ifinal=80
                call rdword(keywd,istart,ifinal)
                call rdnum(keywd,istart,temp,ierror)
                self%extxyz(j,i) = temp*A_TO_BOHRS
                istart=ifinal+1
            enddo

            call rdword(keywd,istart,ifinal)
            call rdnum(keywd,istart,self%extchg(i),ierror)

        enddo

    end subroutine read_quick_molespec_extcharges

   ! check if molecular specifications are correct
   subroutine check_quick_molspec(self, ierr)

     use quick_exception_module
     implicit none

     type (quick_molspec_type), intent(in) :: self
     integer, intent(inout) :: ierr

     if (self%imult .ne. 1) ierr=11

   end subroutine check_quick_molspec

   !-------------------
   ! print varibles
   !-------------------
   subroutine print_quick_molspec(self,io,ierr)

      use quick_exception_module
      use quick_constants_module
      implicit none
       
      integer, intent(inout) :: ierr
      integer io,i,j
      type(quick_molspec_type) self
      if (io.ne.0) then
         write(io,'(/," =========== Molecule Input ==========")')
         write(io,'(" TOTAL MOLECULAR CHARGE  = ",I4,4x,"MULTIPLICITY                = ",I4)') self%molchg,self%imult
         write(io,'(" TOTAL ATOM NUMBER       = ",i4,4x,"NUMBER OF ATOM TYPES        = ",i4)') self%natom,self%iAtomType
         write(io,'(" NUMBER OF HYDROGEN ATOM = ",i4,4x,"NUMBER OF NON-HYDROGEN ATOM = ",i4)') self%nhatom,self%nNonHAtom

         if(self%nextatom.gt.0 )then
           write(io,'(" NUMBER OF EXTERNAL POINT CHARGES = ",i4)') self%nextatom
         endif

         if (self%nelecb.ne.0) then
            write (io,'(" NUMBER OF ALPHA ELECTRONS = ",I4)') self%nelec
            write (io,'(" NUMBER OF BETA ELECTRONS  = ",I4)') self%nelecb
         else
            write (io,'(" NUMBER OF ELECTRONS     = ",I4)') self%nelec
         endif

         write(io,*)
         write(io,'(" -- INPUT GEOMETRY -- :")')
         do I=1,self%natom
            Write (io,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
                  symbol(self%iattype(I)),(self%xyz(j,I)*BOHRS_TO_A,j=1,3)
         enddo

         if(self%nextatom.gt.0 )then
            write(io,*)
            write(io,'(" -- EXTERNAL POINT CHARGES: (X,Y,Z,Q) -- ")')
            do i=1,self%nextatom
               write(io,'(4x,3(F10.4,1x),3x,F7.4)') (self%extxyz(j,i)*BOHRS_TO_A,j=1,3),self%extchg(i)
            enddo
         endif

         ! if no. of atom is less than 30, then output them
         if (self%natom.le.30) then
            write(io,*)
            write(io,'(" -- DISTANCE MATRIX -- :")')
            call PriSym(io,self%natom,self%atomdistance,'f12.5')
         endif
      endif

      return

   end subroutine print_quick_molspec

   !-------------------
   ! set up some varibles
   ! this subroutine is to automatically generate other molspec information from
   ! read-in molespec.
   !-------------------

   subroutine set_quick_molspec(self,ierr)

      use quick_exception_module
      use quick_constants_module

      implicit none

      integer, intent(inout) :: ierr
      integer i,j,k
      type (quick_molspec_type) self

      self%nelec=-self%molchg
      ! get atom charge and molecule charge before everything
      do i=1,natom
         self%chg(i)=self%iattype(i)
         self%nelec=self%nelec+self%chg(i)
      enddo

      ! first set Distance Matrix
      do i=1,natom
         do j=i,natom
            self%atomdistance(i,j)=0d0
            do k=1,3
                self%atomdistance(i,j)=self%atomdistance(i,j)+(self%xyz(k,i)-self%xyz(k,j))**2
            enddo
            self%atomdistance(i,j)=dsqrt(self%atomdistance(i,j))
            self%atomdistance(j,i)=self%atomdistance(i,j)
         enddo
      enddo

      ! second set distnbor
      do i=1,natom
         self%distnbor(i)=1.D30
         do j=1,natom
            if (j.ne.i) then
               self%distnbor(i)=min(self%distnbor(i),self%atomdistance(i,j))
            endif
         enddo
      enddo


      ! get and return no. of hydrogen atom and non-hydogren
      j=0
      do i=1,natom
         if (self%iattype(I).eq.1) j=j+1
      enddo
      self%nHAtom=j
      self%nNonHAtom=natom-j

   end subroutine set_quick_molspec

end module quick_molspec_module
