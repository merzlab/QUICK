!
!	quick_calculated_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

!  Calculated Module, storing calculated information
module quick_calculated_module

   !------------------------------------------------------------------------
   !  ATTRIBUTES  : to complete later
   !  SUBROUTINES : to complete later
   !  FUNCTIONS   : to complete later
   !  DESCRIPTION : For the above list, Smatrix is the overlap matrix, X is the
   !                orthogonalization matrix, O is the operator matrix, CO and COB are
   !                the molecular orbital coefficients, Vec is a mtrix to hold eigenvectors
   !                after a diagonalization, DENSE and DENSEB are the denisty matrices,
   !                V2 is a scratch space for diagonalization, IDEGEN is a matrix of
   !                orbital degeneracies, E and EB are the molecular orbital energies,
   !                and Eel, Ecore, Etot. alec and Belec are self explanatory. GRADIENT
   !                is the energy gradient with respect to atomic motion.
   !  AUTHOR      : Yipu Miao
   !------------------------------------------------------------------------

   implicit none

   ! quick_qm_struct stores quantum chemistry electron information
   ! the elements will be introduced following
   type quick_qm_struct_type

      ! Basis Set Number
      integer,pointer :: nbasis

      ! overlap matrix, will be calculated once, independent on
      ! orbital coefficent. Its dimension is nbasis*nbasis
      double precision,dimension(:,:), allocatable :: s

      ! orthogonalization matrix, the dimension is nbasis*nbasis,
      ! is independent on orbital coefficent
      double precision,dimension(:,:), allocatable :: x

      ! one electron operator, a matrix of size nbasis,nbasis
      double precision, allocatable, dimension(:,:)   :: oneElecO

      ! operator matrix, the dimension is nbasis*nbasis. For HF, it's Fock Matrix
      double precision,dimension(:,:), allocatable :: o, psf,fd1g0
      double precision,dimension(:,:), allocatable :: od, fd,fxd
      integer, dimension(:,:), allocatable :: iarray      
      double precision, dimension(:,:,:), allocatable :: fds1

      ! matrix for saving XC potential, required for incremental KS build
      double precision,dimension(:,:), allocatable :: oxc 

      ! Beta operator matrix, the dimension is nbasis*nbasis. For HF, it's Fock
      ! Matrix
      double precision,dimension(:,:), allocatable :: ob, ewdmx
      double precision,dimension(:,:), allocatable :: obd, fbd

      ! matrix for saving beta XC potential, required for incremental KS build
      double precision,dimension(:,:), allocatable :: obxc

      ! saved operator matrix
      double precision,dimension(:,:), allocatable :: oSave

      ! saved beta operator matrix
      double precision,dimension(:,:), allocatable :: obSave

      ! orbital coeffecient, the dimension is nbasis*nbasis. If it's
      ! unrestricted system, CO will represent alpha electron coeffecient
      double precision,dimension(:,:), allocatable :: co

      ! beta electron orbital coeffecient, the dimension is nbasis*nbasis.
      ! when it's RHF system, this term will be vanished
      double precision,dimension(:,:), allocatable :: cob

      ! matrix to hold eigenvectors after diagonalization,
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: vec

      ! Density matrix, when it's unrestricted system, it presents alpha density
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: dense,psdense
      double precision,dimension(:,:), allocatable :: dense1,d1con,wdens1
      ! when it's unrestricted system, it presents beta density
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseb

      ! saved density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseSave

      ! saved beta density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: densebSave

      ! saved density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseOld

      ! saved beta density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: densebOld

      ! A matrix of orbital degeneracies
      integer, dimension(:),allocatable :: iDegen

      ! Molecular orbital energy, and beta orbital energy
      double precision,dimension(:), allocatable   :: E, Eb

      ! gradient with respect to atomic motion, the dimension is 3natom
      double precision,dimension(:), allocatable   :: gradient

      ! gradient of the point charges, the dimension is 3 times nextatom
      double precision,dimension(:), allocatable   :: ptchg_gradient

      ! dispersion energy gradient
      double precision,dimension(:,:), allocatable   :: disp_gradient

      ! hessian matrix and CPHF matrices, the dimension is 3natom*3natom
      double precision,dimension(:,:), allocatable :: hessian,cphfa,cphfb

      ! one electron energy
      double precision :: E1e

      ! electron energy
      double precision :: EEl

      ! exchange correlation energy
      double precision :: Exc

      ! dispersion correction energy
      double precision :: Edisp

      ! core energy
      double precision :: ECore

      ! external point charge energy
      double precision :: ECharge

      ! total energy, only to HF level
      double precision :: ETot

      ! MP2 perturbation energy
      double precision :: EMP2

      ! alpha electron energy
      double precision :: aElec

      ! beta electron energy
      double precision :: bElec

      ! The following elements are for PB model
      ! vacant electron energy for PB model
      double precision :: EElVac


      ! solvent electron energy for PB model
      double precision :: EElSol


      ! vacant electron energy for PB model
      double precision :: EElPb

      ! solvent exp
      double precision :: gsolexp

      ! Mulliken charge and Lowdin charge
      double precision,dimension(:),allocatable :: Mulliken,Lowdin


   end type quick_qm_struct_type

   type (quick_qm_struct_type) quick_qm_struct

   !----------------------
   ! Interface
   !----------------------
   interface alloc
      module procedure allocate_quick_qm_struct
   end interface alloc

   interface dealloc
      module procedure deallocate_quick_qm_struct
   end interface dealloc

   interface realloc
      module procedure reallocate_quick_qm_struct
   end interface realloc

#ifdef MPIV
   interface broadcast
      module procedure broadcast_quick_qm_struct
   end interface broadcast
#endif
   interface read
      module procedure read_quick_qm_struct
   end interface read

   interface init
      module procedure init_quick_qm_struct
   end interface init

   interface dat
      module procedure dat_quick_qm_struct
   end interface dat

   !----------------------
   ! Inner subroutines
   !----------------------
contains

   !--------------
   ! subroutine to allocate variables
   !--------------
   subroutine allocate_quick_qm_struct(self)
      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      implicit none

      integer nbasis
      integer natom
      integer nelec
      integer idimA
      integer nelecb
      integer i,j,ij     
      integer ntri

      type (quick_qm_struct_type) self

      nbasis=self%nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb

      ! those matrices is necessary for all calculation or the basic of other calculation
      if(.not. allocated(self%s)) allocate(self%s(nbasis,nbasis))
      if(.not. allocated(self%x)) allocate(self%x(nbasis,nbasis))
      if(.not. allocated(self%oneElecO)) allocate(self%oneElecO(nbasis,nbasis))
      if(.not. allocated(self%o)) allocate(self%o(nbasis,nbasis))
      if(.not. allocated(self%oSave)) allocate(self%oSave(nbasis,nbasis))
      if(.not. allocated(self%co)) allocate(self%co(nbasis,nbasis))
      if(.not. allocated(self%vec)) allocate(self%vec(nbasis,nbasis))
      if(.not. allocated(self%dense)) allocate(self%dense(nbasis,nbasis))
      if(.not. allocated(self%denseSave)) allocate(self%denseSave(nbasis,nbasis))
      if(.not. allocated(self%denseOld)) allocate(self%denseOld(nbasis,nbasis))
      if(.not. allocated(self%E)) allocate(self%E(nbasis))
      if(.not. allocated(self%iDegen)) allocate(self%iDegen(nbasis))

      if(.not. allocated(self%Mulliken)) allocate(self%Mulliken(natom))
      if(.not. allocated(self%Lowdin)) allocate(self%Lowdin(natom))

      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         if(.not. allocated(self%gradient)) allocate(self%gradient(3*natom))

         if(quick_method%edisp) then
           if(.not. allocated(self%disp_gradient)) allocate(self%disp_gradient(3,natom))
         endif
      endif

      if (quick_method%freq) then
         if(.not. allocated(self%gradient)) allocate(self%gradient(3*natom))
         if(.not. allocated(self%hessian))allocate(self%hessian(3*natom,3*natom))
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         if(.not. allocated(self%hessian)) allocate(self%hessian(3*natom,3*natom))
         if(.not. allocated(self%iarray)) allocate(self%iarray(nbasis,nbasis))
         if(.not. allocated(self%ewdmx)) allocate(self%ewdmx(nbasis,nbasis))
         ij = 0
         do i=1,nbasis
             do j=1,i
                 ij=ij+1
                 self%iarray(i,j)=ij
                 self%iarray(j,i)=ij
             end do
         end do
         ntri =  (nbasis*(nbasis+1))/2
         if(.not. allocated(self%fd)) allocate(self%fd(3*natom,ntri))
         if(.not. allocated(self%psf)) allocate(self%psf(nbasis,nbasis))
         if(.not. allocated(self%fd1g0)) allocate(self%fd1g0(3*natom,ntri))
         if(.not. allocated(self%od)) allocate(self%od(3*natom,ntri))
         if(.not. allocated(self%dense1)) allocate(self%dense1(3*natom,ntri))
         if(.not. allocated(self%wdens1)) allocate(self%wdens1(3*natom,ntri))
         if(.not. allocated(self%d1con)) allocate(self%d1con(3*natom,ntri))
         if(.not. allocated(self%psdense)) allocate(self%psdense(nbasis,nbasis))
         if(.not. allocated(self%fds1)) allocate(self%fds1(nbasis,nbasis,3*natom))
         if(.not. allocated(self%fxd)) allocate(self%fxd(nbasis,nbasis))
         if (quick_method%unrst) then
            if(.not. allocated(self%obd)) allocate(self%obd(3*natom,ntri))
            if(.not. allocated(self%fbd)) allocate(self%fbd(3*natom,ntri))
            idimA = (nbasis-nelec)*nelec + (nbasis-nelecB)*nelecB
         else
            idimA = 2*(nbasis-(nelec/2))*(nelec/2)
         endif
         if(.not. allocated(self%CPHFA)) allocate(self%CPHFA(idimA,idimA))
         if(.not. allocated(self%CPHFB)) allocate(self%CPHFB(idimA,natom*3))
         if(.not. allocated(self%Eb)) allocate(self%Eb(nbasis))
         if(.not. allocated(self%cob)) allocate(self%cob(nbasis,nbasis))
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         if(.not. allocated(self%ob)) allocate(self%ob(nbasis,nbasis))
         if(.not. allocated(self%obSave)) allocate(self%obSave(nbasis,nbasis))
         if(.not. allocated(self%densebSave)) allocate(self%densebSave(nbasis,nbasis))
         if(.not. allocated(self%densebOld)) allocate(self%densebOld(nbasis,nbasis))
         if(.not. allocated(self%cob)) allocate(self%cob(nbasis,nbasis))
         if(.not. allocated(self%Eb)) allocate(self%Eb(nbasis))
      endif

      if (quick_method%unrst .or. quick_method%DFT) then
         if(.not. allocated(self%denseb)) allocate(self%denseb(nbasis,nbasis))
      endif

      ! one more thing, DFT
      if (quick_method%DFT) then
         if(.not. allocated(self%oxc)) allocate(self%oxc(nbasis,nbasis))
         if(quick_method%unrst .and. (.not. allocated(self%obxc))) allocate(self%obxc(nbasis,nbasis))
      endif

      if (quick_method%grad) then
         if (quick_molspec%nextatom .gt. 0) then
            if(.not. allocated(self%ptchg_gradient)) allocate(self%ptchg_gradient(3*quick_molspec%nextatom))
         endif
      endif

   end subroutine
   
   !-----------------------------
   ! subroutine to realloate data
   !-----------------------------

   subroutine reallocate_quick_qm_struct(self,ierr)

     use quick_exception_module
     use quick_molspec_module,only: quick_molspec

     implicit none

     type (quick_qm_struct_type), intent(inout) :: self
     integer, intent(inout) :: ierr
     integer :: current_size     

     if(quick_molspec%nextatom .gt. 0) then
       if(allocated(self%ptchg_gradient)) current_size= size(self%ptchg_gradient)
       if(current_size /= quick_molspec%nextatom*3) then
         deallocate(self%ptchg_gradient, stat=ierr)
         allocate(self%ptchg_gradient(3*quick_molspec%nextatom), stat=ierr)
         self%ptchg_gradient=0.0d0
       endif
     endif

   end subroutine reallocate_quick_qm_struct

   !--------------
   ! subroutine to write data to dat file
   !--------------

   subroutine dat_quick_qm_struct(self, idatafile)

      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      logical fail

      integer nbasis
      integer natom
      integer nelec
      integer idimA
      integer nelecb

      
      integer idatafile
      type (quick_qm_struct_type) self

      nbasis=self%nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb


      call wchk_int(idatafile, "nbasis", nbasis, fail)
      call wchk_int(idatafile, "natom",  natom,  fail)
      call wchk_int(idatafile, "nelec",  nelec,  fail)
      call wchk_int(idatafile, "nelecb", nelecb, fail)
      call wchk_darray(idatafile, "s",        nbasis, nbasis, 1, self%s,        fail)
      call wchk_darray(idatafile, "x",        nbasis, nbasis, 1, self%x,        fail)
      call wchk_darray(idatafile, "o",        nbasis, nbasis, 1, self%o,        fail)
      call wchk_darray(idatafile, "co",       nbasis, nbasis, 1, self%co,       fail)
      call wchk_darray(idatafile, "vec",      nbasis, nbasis, 1, self%vec,      fail)
      call wchk_darray(idatafile, "dense",    nbasis, nbasis, 1, self%dense,    fail)
      call wchk_darray(idatafile, "E",        nbasis, 1,      1, self%E,        fail)
      call wchk_darray(idatafile, "iDegen",   nbasis, 1,      1, self%iDegen,   fail)
      call wchk_darray(idatafile, "Mulliken", nbasis, 1,      1, self%Mulliken, fail)
      call wchk_darray(idatafile, "Lowdin",   nbasis, 1,      1, self%Lowdin,   fail)

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         call wchk_darray(idatafile, "cob", nbasis, nbasis, 1, self%cob, fail)
         call wchk_darray(idatafile, "Eb", nbasis, 1, 1, self%Eb, fail)
      endif

      if (quick_method%unrst .or. quick_method%DFT) then
         call wchk_darray(idatafile, "denseb", nbasis, 1, 1, self%denseb, fail)
      endif


   end subroutine dat_quick_qm_struct

   !--------------
   ! subroutine to deallocate variables
   !--------------
   subroutine deallocate_quick_qm_struct(self)
      use quick_method_module,only: quick_method
      implicit none
      integer io

      integer nbasis
      integer natom
      integer nelec
      integer idimA
      integer nelecb

      type (quick_qm_struct_type) self

      nullify(self%nbasis)
      ! those matrices is necessary for all calculation or the basic of other calculation
      if (allocated(self%s)) deallocate(self%s)
      if (allocated(self%x)) deallocate(self%x)
      if (allocated(self%oneElecO)) deallocate(self%oneElecO)
      if (allocated(self%o)) deallocate(self%o)
      if (allocated(self%oSave)) deallocate(self%oSave)
      if (allocated(self%co)) deallocate(self%co)
      if (allocated(self%vec)) deallocate(self%vec)
      if (allocated(self%dense)) deallocate(self%dense)
      if (allocated(self%denseSave)) deallocate(self%denseSave)
      if (allocated(self%denseOld)) deallocate(self%denseOld)
      if (allocated(self%E)) deallocate(self%E)
      if (allocated(self%iDegen)) deallocate(self%iDegen)

      if (allocated(self%Mulliken)) deallocate(self%Mulliken)
      if (allocated(self%Lowdin)) deallocate(self%Lowdin)

      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         if (allocated(self%gradient)) deallocate(self%gradient)
         if (quick_method%extCharges) then
            if (allocated(self%ptchg_gradient)) deallocate(self%ptchg_gradient)
         endif
         if(quick_method%edisp) then
            if (allocated(self%disp_gradient)) deallocate(self%disp_gradient)
         endif
      endif

      if (quick_method%freq) then
         if (allocated(self%gradient)) deallocate(self%gradient)
        if (allocated(self%hessian)) deallocate(self%hessian)
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         if (allocated(self%hessian)) deallocate(self%hessian)
         if (allocated(self%iarray)) deallocate(self%iarray)
         if (allocated(self%ewdmx)) deallocate(self%ewdmx)
         if (allocated(self%od)) deallocate(self%od)
         if (allocated(self%fd)) deallocate(self%fd)
         if (allocated(self%psf)) deallocate(self%psf)
         if (allocated(self%fd1g0)) deallocate(self%fd1g0)
         if (allocated(self%dense1)) deallocate(self%dense1)
         if (allocated(self%wdens1)) deallocate(self%wdens1)
         if (allocated(self%d1con)) deallocate(self%d1con)
         if (allocated(self%psdense)) deallocate(self%psdense)
         if (allocated(self%fds1)) deallocate(self%fds1)
         if (allocated(self%fxd)) deallocate(self%fxd)
         if (allocated(self%CPHFA)) deallocate(self%CPHFA)
         if (allocated(self%CPHFB)) deallocate(self%CPHFB)
         if (quick_method%unrst) then
            if (allocated(self%fbd)) deallocate(self%fbd)
            if (allocated(self%obd)) deallocate(self%obd)
         endif
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         if(allocated(self%ob)) deallocate(self%ob)
         if(allocated(self%obSave)) deallocate(self%obSave)
         if(allocated(self%densebSave)) deallocate(self%densebSave)
         if(allocated(self%densebOld)) deallocate(self%densebOld)
         if (allocated(self%cob)) deallocate(self%cob)
         if (allocated(self%Eb)) deallocate(self%Eb)
      endif

      if (quick_method%unrst .or. quick_method%DFT) then
         if (allocated(self%denseb)) deallocate(self%denseb)
      endif

      ! one more thing, DFT
      if (quick_method%DFT) then
         if (allocated(self%oxc)) deallocate(self%oxc)
         if(quick_method%unrst .and. allocated(self%obxc)) deallocate(self%obxc)
      endif

   end subroutine

#ifdef MPIV
   !-------------------
   ! broadcast variable list
   !-------------------
   subroutine broadcast_quick_qm_struct(self)
      use quick_mpi_module
      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      use mpi
      implicit none
      type (quick_qm_struct_type) self
      integer natom
      integer nbasis,nbasis2
      integer nelec,nelecb
      integer iDimA

      nbasis=self%nbasis
      nbasis2=nbasis*nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb

      call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%nbasis,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%s,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%x,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%oneElecO,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%o,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%oSave,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%co,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%vec,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%dense,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%denseSave,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%denseOld,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%iDegen,nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(self%Mulliken,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%Lowdin,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(self%EEl,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%Exc,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ECore,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ECharge,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ETot,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

      if (quick_method%PBSOL) then
         call MPI_BCAST(self%EElVac,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%EElSol,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%EElPb,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%gsolexp,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif

      if (quick_method%unrst) then
         call MPI_BCAST(self%cob,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%denseb,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%Eb,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%aElec,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%bElec,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif

      if (quick_method%grad) then
         call MPI_BCAST(self%gradient,3*natom,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif

      if (quick_method%MP2) then
         call MPI_BCAST(self%gradient,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif

      if (quick_method%analHess) then
         call MPI_BCAST(self%hessian,3*natom*3*natom,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         if (quick_method%unrst) then
            idimA = (nbasis-nelec)*nelec + (nbasis-nelecB)*nelecB
         else
            idimA = 2*(nbasis-(nelec/2))*(nelec/2)
         endif
         call MPI_BCAST(self%cphfa,idimA*idimA,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(self%cphfb,idimA*natom*3,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      endif

   end subroutine broadcast_quick_qm_struct
#endif

   subroutine init_quick_qm_struct(self)
      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      implicit none

      integer nbasis
      integer natom
      integer nelec
      integer idimA
      integer nelecb
      integer ntri

      type (quick_qm_struct_type) self

      nbasis=self%nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb

      call zeroMatrix(self%s,nbasis)
      call zeroMatrix(self%x,nbasis)
      call zeroMatrix(self%oneElecO,nbasis)
      call zeroMatrix(self%o,nbasis)
      call zeroMatrix(self%oSave,nbasis)
      call zeroMatrix(self%co,nbasis)
      call zeroMatrix(self%vec,nbasis)
      call zeroMatrix(self%dense,nbasis)
      call zeroMatrix(self%denseSave,nbasis)
      call zeroMatrix(self%denseOld,nbasis)
      call zeroVec(self%E,nbasis)
      call zeroiVec(self%iDegen,nbasis)
      call zeroVec(self%Mulliken,natom)
      call zeroVec(self%Lowdin,natom)

      call zeroMatrix(self%denseOld,nbasis)

      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         call zeroVec(self%gradient,3*natom)
      endif
      
      if (quick_method%freq) then
      call zeroVec(self%gradient,3*natom)
      call zeroMatrix(self%hessian,3*natom)
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         call zeroMatrix(self%hessian,3*natom)
         ntri =  (nbasis*(nbasis+1))/2
         self%od=0.0d0
         self%fd=0.0d0
         call zeroMatrix(self%psf,nbasis)
         call zeroMatrix(self%fxd,nbasis)
         call zeroMatrix2(self%fd1g0,3*natom,ntri)
         call zeroMatrix2(self%dense1,3*natom,ntri)
         call zeroMatrix2(self%wdens1,3*natom,ntri)
         call zeroMatrix2(self%d1con,3*natom,ntri)
         call zeroMatrix(self%psdense,nbasis)
         self%fds1=0.0d0
         if (quick_method%unrst) then
            self%fbd=0.0d0
            self%obd=0.0d0
            idimA = (nbasis-nelec)*nelec + (nbasis-nelecB)*nelecB
         else
            idimA = 2*(nbasis-(nelec/2))*(nelec/2)
         endif
         call zeroMatrix(self%CPHFA,idimA)
         call zeroMatrix2(self%CPHFB,idimA,natom*3)
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         call zeroMatrix(self%cob,nbasis)
         call zeroMatrix(self%denseb,nbasis)
         call zeroVec(self%Eb,nbasis)
      endif

   end subroutine


   subroutine read_quick_qm_struct(self,keywd)
      implicit none
      character(len=200) :: keyWD
      double precision :: rdnml
      type (quick_qm_struct_type) self

      call upcase(keyWD,200)

      ! Experimental Solvantion energy
      if (index(keywd,'GSOL=') /= 0) self%Gsolexp = rdnml(keywd,'GSOL')

   end subroutine read_quick_qm_struct

end module quick_calculated_module
