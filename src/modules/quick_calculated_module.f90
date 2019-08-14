










!
!	quick_calculated_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

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

      ! operator matrix, the dimension is nbasis*nbasis. For HF, it's Fock Matrix
      double precision,dimension(:,:), allocatable :: o

      ! saved operator matrix
      double precision,dimension(:,:), allocatable :: oSave

      ! saved dft operator matrix
      double precision,dimension(:,:), allocatable :: oSaveDFT

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
      double precision,dimension(:,:), allocatable :: dense

      ! when it's unrestricted system, it presents beta density
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseb

      ! saved density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseSave

      ! saved density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseOld
      ! Initial density matrix
      ! the dimension is nbasis*nbasis.
      double precision,dimension(:,:), allocatable :: denseInt

      ! A matrix of orbital degeneracies
      integer, dimension(:),allocatable :: iDegen

      ! Molecular orbital energy, and beta orbital energy
      double precision,dimension(:), allocatable   :: E, Eb

      ! gradient with respect to atomic motion, the dimension is 3natom
      double precision,dimension(:), allocatable   :: gradient

      ! hessian matrix and CPHF matrices, the dimension is 3natom*3natom
      double precision,dimension(:,:), allocatable :: hessian,cphfa,cphfb

      ! electron energy
      double precision :: EEl

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
   interface broadcast
      module procedure broadcast_quick_qm_struct
   end interface broadcast
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

      type (quick_qm_struct_type) self
      nbasis=self%nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb

      ! those matrices is necessary for all calculation or the basic of other calculation
      allocate(self%s(nbasis,nbasis))
      allocate(self%x(nbasis,nbasis))
      allocate(self%o(nbasis,nbasis))
      allocate(self%oSave(nbasis,nbasis))
      allocate(self%co(nbasis,nbasis))
      allocate(self%vec(nbasis,nbasis))
      allocate(self%dense(nbasis,nbasis))
      allocate(self%denseSave(nbasis,nbasis))
      allocate(self%denseOld(nbasis,nbasis))
      allocate(self%denseInt(nbasis,nbasis))
      allocate(self%E(nbasis))
      allocate(self%iDegen(nbasis))

      allocate(self%Mulliken(natom))
      allocate(self%Lowdin(natom))

      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         allocate(self%gradient(3*natom))
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         allocate(self%hessian(3*natom,3*natom))
         if (quick_method%unrst) then
            idimA = (nbasis-nelec)*nelec + (nbasis-nelecB)*nelecB
         else
            idimA = 2*(nbasis-(nelec/2))*(nelec/2)
         endif
         allocate(self%CPHFA(idimA,idimA))
         allocate(self%CPHFB(idimA,natom*3))
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         allocate(self%cob(nbasis,nbasis))
         allocate(self%Eb(nbasis))
      endif

      if (quick_method%unrst .or. quick_method%DFT) then
         allocate(self%denseb(nbasis,nbasis))
      endif

      ! one more thing, DFT
      if (quick_method%DFT) then
         allocate(self%oSaveDFT(nbasis,nbasis))
      endif

   end subroutine

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
      deallocate(self%s)
      deallocate(self%x)
      deallocate(self%o)
      deallocate(self%oSave)
      deallocate(self%co)
      deallocate(self%vec)
      deallocate(self%dense)
      deallocate(self%denseSave)
      deallocate(self%denseOld)
      deallocate(self%denseInt)
      deallocate(self%E)
      deallocate(self%iDegen)

      deallocate(self%Mulliken)
      deallocate(self%Lowdin)

      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         deallocate(self%gradient)
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         deallocate(self%hessian)
         deallocate(self%CPHFA)
         deallocate(self%CPHFB)
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         deallocate(self%cob)
         deallocate(self%Eb)
      endif

      if (quick_method%unrst .or. quick_method%DFT) then
         deallocate(self%denseb)
      endif

      ! one more thing, DFT
      if (quick_method%DFT) then
         deallocate(self%oSaveDFT)
      endif

   end subroutine

   !-------------------
   ! broadcast variable list
   !-------------------
   subroutine broadcast_quick_qm_struct(self)
      use quick_mpi_module
      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      implicit none
      include "mpif.h"
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
      call MPI_BCAST(self%o,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%oSave,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%co,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%vec,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%dense,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%denseSave,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%denseOld,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%denseInt,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%iDegen,nbasis,mpi_integer,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(self%Mulliken,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%Lowdin,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

      call MPI_BCAST(self%EEl,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ECore,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ECharge,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(self%ETot,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)


      if (quick_method%DFT)  call MPI_BCAST(self%oSaveDFT,nbasis2,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)

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

   subroutine init_quick_qm_struct(self)
      use quick_method_module,only: quick_method
      use quick_molspec_module,only: quick_molspec
      implicit none

      integer nbasis
      integer natom
      integer nelec
      integer idimA
      integer nelecb

      type (quick_qm_struct_type) self

      nbasis=self%nbasis
      natom=quick_molspec%natom
      nelec=quick_molspec%nelec
      nelecb=quick_molspec%nelecb

      call zeroMatrix(self%s,nbasis)
      call zeroMatrix(self%x,nbasis)
      call zeroMatrix(self%o,nbasis)
      call zeroMatrix(self%oSave,nbasis)
      call zeroMatrix(self%co,nbasis)
      call zeroMatrix(self%vec,nbasis)
      call zeroMatrix(self%dense,nbasis)
      call zeroMatrix(self%denseSave,nbasis)
      call zeroMatrix(self%denseOld,nbasis)
      call zeroMatrix(self%denseInt,nbasis)
      call zeroVec(self%E,nbasis)
      call zeroiVec(self%iDegen,nbasis)
      call zeroVec(self%Mulliken,natom)
      call zeroVec(self%Lowdin,natom)

      call zeroMatrix(self%denseOld,nbasis)


      ! if 1st order derivation, which is gradient calculation is requested
      if (quick_method%grad) then
         call zeroVec(self%gradient,3*natom)
      endif

      ! if 2nd order derivation, which is Hessian matrix calculation is requested
      if (quick_method%analHess) then
         call zeroMatrix(self%hessian,3*natom)
         if (quick_method%unrst) then
            idimA = (nbasis-nelec)*nelec + (nbasis-nelecB)*nelecB
         else
            idimA = 2*(nbasis-(nelec/2))*(nelec/2)
         endif
         call zeroMatrix(self%CPHFA,idimA)
         call zeroMatrix2(self%CPHFA,idimA,natom*3)
      endif

      ! if unrestricted, some more varibles is required to be allocated
      if (quick_method%unrst) then
         call zeroMatrix(self%cob,nbasis)
         call zeroMatrix(self%denseb,nbasis)
         call zeroVec(self%Eb,nbasis)
      endif

      ! one more thing, DFT
      if (quick_method%DFT) then
         call zeroMatrix(self%oSaveDFT,nbasis)
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
