
!
!	quick_basis_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

!   Basis Module, to store basis information
module quick_basis_module

   !------------------------------------------------------------------------
   !  ATTRIBUTES  : see list below
   !  SUBROUTINES : none
   !  FUNCTIONS   : none
   !  DESCRIPTION : A quick explanation :  aexp(i,j) is the alpha
   !                exponent of the ith primitive of the jth function, and d(i,j) is
   !                is multiplicative coefficient.  quick_basis%ncenter(i) is the center upon which
   !                the ith basis function is located.  ncontract(i) is the degree of
   !                contraction of the ith basis function, and itype(i) is the ith basis
   !                function's angular momentum.  The ifirst and ilast arrays denot the
   !                first and last basis function number on a given center, i.e. if center
   !                number 8 has basis functions 7-21, ifirst(8)=7 and ilast(8) = 21.
   !                If the calculation is closed shell, nelec is the total number of electrons,
   !                otherwise it is the number of alpha electrons and nelecb is the
   !                number of basis elctrons.  The xyz and ichrg arrays denote atomic
   !                position and charge, while iattype is the atomic number of the atom, i.e.
   !                6 for carbon, etc. natom and nbasis are the number of atoms and
   !                basis functions. ibasis denotes the basis set chosen for.
   !------------------------------------------------------------------------

   use quick_gaussian_class_module
   implicit none

   type quick_basis_type

        type (gaussian), dimension(:), pointer :: gauss_fnc => null()

        ! total shell number
        integer, pointer :: nshell

        ! total primitive guassian function number
        integer, pointer :: nprim

        ! shell kinds
        integer, pointer :: jshell

        ! basis kinds
        integer, pointer :: jbasis

        ! total basis number
        integer, pointer :: nbasis

        ! the first and last basis function for an atom
        integer, dimension(:),allocatable :: first_basis_function, last_basis_function

        ! the first and last shell basis function for an atom, only used in MP2 calculation
        integer, dimension(:),allocatable :: first_shell_basis_function, last_shell_basis_function

        ! central atom for a basis function
        integer, dimension(:),allocatable :: ncenter

        ! starting basis for a shell
        integer, allocatable, dimension (:) :: kstart

        ! centeral atom for a shell
        integer, allocatable, dimension (:) :: katom

        ! shell type: 1=s, 3=p, 4=sp, 6=d, f=10
        integer, allocatable, dimension (:) :: ktype

        ! primitive guassian function for a shell
        integer, allocatable, dimension (:) :: kprim

        ! shell number for a kind of atom
        integer, dimension(0:92)  :: kshell


        ! sum of ktype if shell<j
        integer, allocatable, dimension(:) :: Ksumtype
        integer, allocatable, dimension(:) :: Qnumber

        ! angular momenta quantum number for a shell
        ! s=0~0, p=1~1, sp=0~1, d=2~2, f=3~3
        integer, allocatable, dimension(:) :: Qstart
        integer, allocatable, dimension(:) :: Qfinal

        ! the minimum exponent for a shell
        double precision, allocatable,dimension(:) ::  gcexpomin

        ! Qsbasis(i,j) and Qfbasis(i,j) stands for starting and ending basis
        ! function for shell i with angular momenta j
        integer, allocatable, dimension(:,:) :: Qsbasis,Qfbasis

        ! normalized coeffecient
        double precision, allocatable, dimension(:,:) :: gccoeff

        ! unnormalized contraction coefficients
        double precision, allocatable, dimension(:,:) :: unnorm_gccoeff

        ! basis set factor
        double precision, allocatable, dimension(:) :: cons

        ! combined coeffecient for two indices
        double precision, allocatable, dimension(:,:,:,:) :: Xcoeff

        ! exponent
        double precision, allocatable, dimension(:,:) :: gcexpo

        integer, allocatable, dimension(:,:) :: KLMN

#if defined CUDA_MPIV || defined HIP_MPIV
        integer :: mpi_qshell                   ! Total number or sorted shells

        integer,allocatable :: mpi_qshelln(:)   ! qshell ranges calculated by each gpu
#endif

   end type quick_basis_type

   type(quick_basis_type) quick_basis

   ! to be sub into gauss type
   type(gaussian),dimension(:), allocatable :: gauss
   integer,dimension(:),allocatable :: ncontract                                        ! to be sub
   integer, dimension(:,:), allocatable :: itype                                        ! to be sub
   double precision, dimension(:,:), allocatable :: aexp,dcoeff                         ! to be sub


   integer,target :: nshell,nprim,jshell,jbasis
   integer,target :: nbasis
   integer :: maxcontract

   ! used for 2e integral indices
   integer :: IJKLtype,III,JJJ,KKK,LLL,IJtype,KLtype
   integer, parameter :: longLongInt = selected_int_kind (16)
   integer(kind=longLongInt) :: intIndex
   integer, parameter :: bufferSize = 150000
   integer:: bufferInt
   double precision, dimension(bufferSize) :: intBuffer
   integer, dimension(bufferSize) :: aBuffer
   integer, dimension(bufferSize) :: bBuffer

logical, parameter ::  incoreInt =.false.! .true. !.false.
integer, parameter :: incoreSize = 100000000
integer incoreIndex
#if defined CUDA || defined HIP
double precision, dimension(1) :: intIncore
integer, dimension(1) :: aIncore
integer, dimension(1) :: bIncore
#else
double precision, dimension(incoreSize) :: intIncore
integer, dimension(incoreSize) :: aIncore
integer, dimension(incoreSize) :: bIncore
#endif

   ! used for hrr and vrr
   double precision :: Y,dnmax
   double precision :: Yaa(3),Ybb(3),Ycc(3)  ! only used for opt


   ! this is for SAD initial guess
   integer,allocatable,dimension(:) :: atombasis    ! basis number for every atom
   double precision,allocatable,dimension(:,:,:) :: atomdens    ! density matrix for ceitain atom


   double precision, allocatable, dimension(:,:) :: Apri,Kpri
   double precision, allocatable, dimension(:,:,:) :: Ppri

   ! they are for Schwartz cutoff
   double precision, allocatable, dimension(:,:) :: Ycutoff,cutmatrix,cutprim
   double precision, allocatable, dimension(:,:,:,:) :: Yxiaoprim !Yxiaoprim only used at shwartz cutoff


   ! for MP2
   double precision, allocatable, dimension(:,:,:) :: orbmp2dcsub
   double precision, allocatable, dimension(:,:) :: orbmp2
   double precision, allocatable, dimension(:,:,:,:,:) :: orbmp2i331
   double precision, allocatable, dimension(:,:,:,:,:) :: orbmp2j331
   double precision, allocatable, dimension(:,:,:,:) :: orbmp2k331
   double precision, allocatable, dimension(:,:,:) :: orbmp2k331dcsub

   !
   double precision, allocatable, dimension(:,:,:) :: Yxiao,Yxiaotemp,attraxiao

    !only for opt
   double precision, allocatable, dimension(:,:,:,:) :: attraxiaoopt

   ! only for dft
   double precision, allocatable, dimension(:) :: iao,phixiao,dphidxxiao,dphidyxiao,dphidzxiao
   double precision, allocatable, dimension(:,:) :: iaox, iaoxx, iaoxxx

#ifdef MPIV
   ! MPI
   integer,allocatable:: mpi_jshelln(:)    ! shell no. calculated this node
   integer,allocatable:: mpi_jshell(:,:)   ! shell series no. calcluated in this node

   integer,allocatable:: mpi_nbasisn(:)    ! basis no. calculated this node
   integer,allocatable:: mpi_nbasis(:,:)   ! basis series no. calcluated in this node
#endif

    interface alloc
        module procedure allocate_quick_basis
        module procedure allocate_basis
        module procedure allocate_host_xc_basis
    end interface alloc

    interface dealloc
        module procedure deallocate_quick_basis
        module procedure deallocate_host_xc_basis
    end interface dealloc

    interface print
        module procedure print_quick_basis
    end interface print

contains

   !----------------
   ! Allocate quick basis
   !----------------
   subroutine allocate_quick_basis(self,natom_arg,nshell_arg,nbasis_arg)
        use quick_gaussian_class_module
        use quick_size_module
        use quick_mpi_module
        implicit none
        integer natom_arg,nshell_arg,nbasis_arg,i,j
        type(quick_basis_type) self

        if(.not. associated (self%gauss_fnc)) allocate(self%gauss_fnc(nbasis_arg))
        if(.not. allocated (self%ncenter)) allocate(self%ncenter(nbasis_arg))
        if(.not. allocated(self%first_basis_function)) allocate(self%first_basis_function(natom_arg))
        if(.not. allocated(self%last_basis_function)) allocate(self%last_basis_function(natom_arg))
        if(.not. allocated(self%first_shell_basis_function)) allocate(self%first_shell_basis_function(natom_arg))
        if(.not. allocated(self%last_shell_basis_function)) allocate(self%last_shell_basis_function(natom_arg))

        if(.not. allocated(self%kstart)) allocate(self%kstart(nshell_arg))
        if(.not. allocated(self%katom)) allocate(self%katom(nshell_arg))
        if(.not. allocated(self%ktype)) allocate(self%ktype(nshell_arg))
        if(.not. allocated(self%kprim)) allocate(self%kprim(nshell_arg))

        if(.not. allocated(self%Qnumber)) allocate(self%Qnumber(nshell_arg))
        if(.not. allocated(self%Qstart)) allocate(self%Qstart(nshell_arg))
        if(.not. allocated(self%Qfinal)) allocate(self%Qfinal(nshell_arg))
        if(.not. allocated(self%Ksumtype)) allocate(self%Ksumtype(nshell_arg+1))
        if(.not. allocated(self%gcexpomin)) allocate(self%gcexpomin(nshell_arg))

        if(.not. allocated(self%Qsbasis)) allocate(self%Qsbasis(nshell_arg,0:3))
        if(.not. allocated(self%Qfbasis)) allocate(self%Qfbasis(nshell_arg,0:3))

        do i = 1, nshell_arg
            do j = 0, 3
                self%Qfbasis(i,j) = 0
                self%Qfbasis(i,j) = 0
            enddo
        enddo

        if(.not. allocated(self%gcexpo)) allocate(self%gcexpo(MAXPRIM,nbasis_arg))
        if(.not. allocated(self%gccoeff)) allocate(self%gccoeff(MAXPRIM,nbasis_arg))
        if(.not. allocated(self%unnorm_gccoeff)) allocate(self%unnorm_gccoeff(MAXPRIM,nbasis_arg))
        if(.not. allocated(self%cons)) allocate(self%cons(nbasis_arg))
        do i = 1, MAXPRIM
            do j = 1, nbasis_arg
                self%gcexpo( i, j) = 0.0
                self%gccoeff( i, j) = 0.0
                self%unnorm_gccoeff( i, j) = 0.0
            enddo
        enddo

        do i = 1, nbasis_arg
            self%cons(i) = 0.0
        enddo

        if(.not. allocated(self%KLMN)) allocate(self%KLMN(3,nbasis_arg))
#if defined CUDA_MPIV || defined HIP_MPIV
        if(.not. allocated(self%mpi_qshelln)) allocate(self%mpi_qshelln(mpisize+1))
#endif
   end subroutine allocate_quick_basis


   !----------------
   ! deallocate quick basis
   !----------------
   subroutine deallocate_quick_basis(self)
        use quick_gaussian_class_module
        implicit none
        type (quick_basis_type) self

        nullify(self%gauss_fnc)

        if (allocated(self%ncenter)) deallocate(self%ncenter)
        if (allocated(self%first_basis_function)) deallocate(self%first_basis_function)
        if (allocated(self%last_basis_function)) deallocate(self%last_basis_function)
        if (allocated(self%last_shell_basis_function)) deallocate(self%last_shell_basis_function)
        if (allocated(self%first_shell_basis_function)) deallocate(self%first_shell_basis_function)
        if (allocated(self%kstart)) deallocate(self%kstart)
        if (allocated(self%katom)) deallocate(self%katom)
        if (allocated(self%ktype)) deallocate(self%ktype)
        if (allocated(self%kprim)) deallocate(self%kprim)
        if (allocated(self%Qnumber)) deallocate(self%Qnumber)
        if (allocated(self%Qstart)) deallocate(self%Qstart)
        if (allocated(self%Qfinal)) deallocate(self%Qfinal)
        if (allocated(self%Ksumtype)) deallocate(self%Ksumtype)
        if (allocated(self%gcexpomin)) deallocate(self%gcexpomin)
        if (allocated(self%Qsbasis)) deallocate(self%Qsbasis)
        if (allocated(self%Qfbasis)) deallocate(self%Qfbasis)
        if (allocated(self%cons)) deallocate(self%cons)
        if (allocated(self%gcexpo)) deallocate(self%gcexpo)
        if (allocated(self%gccoeff)) deallocate(self%gccoeff)
        if (allocated(self%unnorm_gccoeff)) deallocate(self%unnorm_gccoeff)
        if (allocated(self%KLMN)) deallocate(self%KLMN)

#if defined CUDA_MPIV || defined HIP_MPIV
        if (allocated(self%mpi_qshelln)) deallocate(self%mpi_qshelln)
#endif

        if(allocated(Apri))          deallocate(Apri)
        if(allocated(Kpri))          deallocate(Kpri)
        if(allocated(cutprim))       deallocate(cutprim)
        if(allocated(Ppri))          deallocate(Ppri)
        if(allocated(self%Xcoeff)) deallocate(self%Xcoeff)

   end subroutine deallocate_quick_basis


   subroutine allocate_basis(self)

      implicit none
      type (quick_basis_type) :: self

      if(.not. allocated(Apri)) allocate(Apri(jbasis,jbasis))
      if(.not. allocated(Kpri)) allocate(Kpri(jbasis,jbasis))
      if(.not. allocated(cutprim)) allocate(cutprim(jbasis,jbasis))
      if(.not. allocated(Ppri)) allocate(Ppri(3,jbasis,jbasis))
      if(.not. allocated(self%Xcoeff)) allocate(self%Xcoeff(jbasis,jbasis,0:3,0:3))
   end subroutine

   ! following arrays are only required for host version of xc, allocate and
   ! deallocate them separately. Note that we use quick_basis_arg and isDFT
   ! as dummy arguments to call this through the alloc interface
   subroutine allocate_host_xc_basis(self, isDFT)

     implicit none
     type(quick_basis_type) :: self
     logical, intent(in) :: isDFT

     if(isDFT) then
       if(.not. allocated(phiXiao)) allocate(phiXiao(nbasis))
       if(.not. allocated(dPhidXXiao)) allocate(dPhidXXiao(nbasis))
       if(.not. allocated(dPhidYXiao)) allocate(dPhidYXiao(nbasis))
       if(.not. allocated(dPhidZXiao)) allocate(dPhidZXiao(nbasis))
       if(.not. allocated(iao)) allocate(iao(nbasis))
       if(.not. allocated(iaox)) allocate(iaox(3,nbasis))
       if(.not. allocated(iaoxx)) allocate(iaoxx(6,nbasis))
       if(.not. allocated(iaoxxx)) allocate(iaoxxx(10,nbasis))
     endif

   end subroutine allocate_host_xc_basis

   ! deallocate dft specific data structures
   subroutine deallocate_host_xc_basis(self, isDFT)

     implicit none
     type(quick_basis_type) :: self
     logical, intent(in) :: isDFT

     if(isDFT)then
        if(allocated(phiXiao))    deallocate(phiXiao)
        if(allocated(dPhidXXiao)) deallocate(dPhidXXiao)
        if(allocated(dPhidYXiao)) deallocate(dPhidYXiao)
        if(allocated(dPhidZXiao)) deallocate(dPhidZXiao)
        if(allocated(iao))    deallocate(iao)
        if(allocated(iaox))    deallocate(iaox)
        if(allocated(iaoxx))    deallocate(iaoxx)
        if(allocated(iaoxxx))    deallocate(iaoxxx)
     end if

   end subroutine deallocate_host_xc_basis


   subroutine print_quick_basis(self,ioutfile)
        implicit none
        type(quick_basis_type) self
        integer iOutFile

        if (ioutfile.ne.0) then
            write (iOutFile,*)
            write (iOutFile,'("============== BASIS INFOS ==============")')
            write (iOutFile,'(" BASIS FUNCTIONS = ",I4)') nbasis
            write (iOutFile,'(" NSHELL = ",I4, " NPRIM  = ", I4)') nshell,nprim
            write (iOutFile,'(" JSHELL = ",I4, " JBASIS = " ,I4)') jshell,jbasis
            write (iOutFile,*)
        endif
   end subroutine print_quick_basis

   subroutine normalize_basis()
      use quick_constants_module
      implicit none

      integer jbas,jcon,ibas,icon1,icon2,l
      double precision dconew,nxponent,xponent,gamma,xnorm

      do Jbas=1,nbasis
         do Jcon=1,ncontract(jbas)
            dcoeff(Jcon,Jbas)=dcoeff(Jcon,Jbas) *xnorm(aexp(Jcon,Jbas), &
                  itype(1,Jbas),itype(2,Jbas),itype(3,Jbas))
         enddo
      enddo

      do Ibas=1,nbasis
         dconew = 0.d0
         nxponent=-itype(1,Ibas)-itype(2,Ibas)-itype(3,Ibas)
         xponent=-1.5d0+dble(nxponent)
         do Icon1 = 1,ncontract(Ibas)
            do Icon2 = 1,ncontract(Ibas)
               dconew = dconew + dcoeff(Icon1,Ibas)*dcoeff(Icon2,Ibas) &
                     *(aexp(Icon1,Ibas)+aexp(Icon2,Ibas))**xponent
            enddo
         enddo
         gamma=1.d0
         do L=1,itype(1,Ibas)
            gamma = gamma * (dble(itype(1,Ibas) - L) + .5d0)
         enddo
         do L=1,itype(2,Ibas)
            gamma = gamma * (dble(itype(2,Ibas) - L) + .5d0)
         enddo
         do L=1,itype(3,Ibas)
            gamma = gamma * (dble(itype(3,Ibas) - L) + .5d0)
         enddo
         dconew = (dconew*gamma*PITO3HALF)**(-.5d0)
         do Icon1 = 1,ncontract(Ibas)
            dcoeff(Icon1,Ibas) = dconew*dcoeff(Icon1,Ibas)
         enddo
      enddo

    end subroutine


end module quick_basis_module
