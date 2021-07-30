module quick_dlfind_module
   use dlf_parameter_module, only: rk
   implicit none
   private
!   public :: dlfind_init
   public :: dlfind_init, dlfind_run, dlfind_final 
   real(rk),allocatable :: tmpcoords(:),tmpcoords2(:)  
   integer, allocatable :: spec(:)
!   double precision  :: mass(:)

contains
!   subroutine dlfind_quick_test

!      call dlf_coords_init

!   end subroutine dlfind_quick_test
   subroutine dlfind_init
!      use dlf_parameter_module, only: rk
!      use dlf_global, only: glob,pi,stdout,printl,printf
      use dlf_global, only: glob,pi,stdout,printl,printf,stderr
      use dlf_stat, only: stat
      use dlf_allocate, only: allocate,deallocate
      use dlf_store, only: store_initialise
      use dlf_constants, only: dlf_constants_init,dlf_constants_get
      use quick_constants_module, only: EMASS
      use quick_molspec_module, only: natom, quick_molspec
      use lbfgs_module

      implicit none
      integer              :: ivar,nat,nframe,nmass,nweight,nz,tsrel,iat, jat
      integer              :: ierr,nspec
      integer              :: tdlf_farm
      integer              :: n_po_scaling
      integer              :: coupled_states
      integer              :: micro_esp_fit
      logical                      :: needhessian ! do we need a Hessian?
      integer       :: nvarin ! number of variables to read in
      integer       :: nvarin2! number of variables to read in
!      integer       :: nspec  ! number of values in the integer
    
      nspec=3*quick_molspec%natom
      spec(:)=0
      call dlf_coords_init
      call dlf_default_init(nspec,spec)
    
      ivar=1
      tdlf_farm=1 ! set default value
      n_po_scaling=0 ! set default value
      coupled_states=1 ! set default value
      micro_esp_fit=0 ! set default value


      call dlf_default_set(3*quick_molspec%natom)

      if (.not. allocated(tmpcoords)) allocate(tmpcoords(3*quick_molspec%natom))
      if (.not. allocated(tmpcoords2)) allocate(tmpcoords2(quick_molspec%natom))      
      if (.not. allocated(spec)) allocate(spec(3*quick_molspec%natom))
!      if (.not. allocated(mass)) allocate(mass(quick_molspec%natom))

      do iat=1, quick_molspec%natom
         tmpcoords2(iat)=EMASS(quick_molspec%iattype(iat))
         print*, tmpcoords2(iat)
      enddo

      ! initialise (reset) all statistics counters
      stat%sene=0
      stat%pene=0
      call dlf_stat_reset

      ! initialise dlf_store
      call store_initialise

      nvarin=3*quick_molspec%natom
      nvarin2=quick_molspec%natom
      nmass=quick_molspec%natom
      nframe = 0
      nz= quick_molspec%natom
      nweight=0
      ! allocate storage
      call dlf_allocate_glob(nvarin,nvarin2,nspec,tmpcoords,tmpcoords2,spec,&
      nz,nframe,nmass,nweight,n_po_scaling)

      ! initialise coordinate transform, allocate memory for it
      call dlf_coords_init
      ! initialise search algorithm
      call dlf_formstep_init(needhessian)
      ! initialise line search
      call linesearch_init

   end subroutine dlfind_init

!#if 0 
   subroutine dlfind_run(coords,gradient)
      use quick_molspec_module, only: natom, quick_molspec 
!      USE dlf_parameter_module, only: rk
      use dlf_global, only: glob,stderr,stdout,printl,pi
      USE lbfgs_module

      implicit none

      real(rk)  ,intent(inout)    :: coords(3*quick_molspec%natom)
      real(rk)  ,intent(inout)    :: gradient(3*quick_molspec%natom)
      logical                      :: needhessian ! do we need a Hessian?
      integer       :: nvarin ! number of variables to read in
                                     !  3*nat
      integer       :: nvarin2! number of variables to read in
                        !  in the second array (coords2)
      integer       :: nspec  ! number of values in the integer
                        !  array spec
                                     ! a parallel run, 0 otherwise
      integer              :: ivar,nat,nframe,nmass,nweight,nz,iat, jat
      integer              :: n_po_scaling
      integer              :: nicore=0
      logical              :: massweight
 
      nvarin=3*quick_molspec%natom
      massweight=.False.
      n_po_scaling=0 ! set default value
print*,"Entered dlfind_run"
      !dimension conversion
      glob%xcoords = reshape(coords,(/3,nat/))
      glob%xgradient = reshape(gradient,(/3,nat/))
print*,glob%xgradient
    
      call dlf_cartesian_xtoi(quick_molspec%natom,nvarin,nicore,massweight,glob%xcoords,glob%xgradient,&
    glob%icoords,glob%igradient)
      
      CALL DLF_LBFGS_STEP(GLOB%ICOORDS,GLOB%IGRADIENT,GLOB%STEP)

      call dlf_cartesian_itox(quick_molspec%natom,nvarin,nicore,massweight,glob%icoords,glob%xcoords)

      do iat=1,natom
         do jat=1,3
            coords((jat-1)*3 + iat) = glob%xcoords(iat,jat)
         enddo
      enddo

   end subroutine dlfind_run

   subroutine dlfind_final
      implicit none

      if (allocated(tmpcoords)) deallocate(tmpcoords)
      if (allocated(tmpcoords2)) deallocate(tmpcoords2)
      if (allocated(spec)) deallocate(spec)
!      if (allocated(mass)) deallocate(mass)
      ! ====================================================================
      ! CLOSE DOWN
      ! ====================================================================

      ! shut down finally
      call dlf_deallocate_glob()

      ! delete memory for line search 
      call linesearch_destroy

      ! delete memory for search algorithm
      call dlf_formstep_destroy

      ! delete memory for internal coordinates
      call dlf_coords_destroy

   end subroutine dlfind_final
!#endif
end module quick_dlfind_module
