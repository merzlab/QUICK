!
!        Interface between CRYSTAL and DL-FIND
!
!
!
!
!   Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!   Tom Keal (thomas.keal@stfc.ac.uk)
!   Joanne Carr (joanne.carr@stfc.ac.uk)
!
!   This file is part of DL-FIND.
!
!   DL-FIND is free software: you can redistribute it and/or modify
!   it under the terms of the GNU Lesser General Public License as
!   published by the Free Software Foundation, either version 3 of the
!   License, or (at your option) any later version.
!
!   DL-FIND is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!   GNU Lesser General Public License for more details.
!
!   You should have received a copy of the GNU Lesser General Public
!   License along with DL-FIND.  If not, see
!   <http://www.gnu.org/licenses/>.
!
!     ..................................................................
      subroutine dl_find_crystal(natoms)
      use dlfind_module, only: dlf_nframe, dlf_nweight, dlf_nmass, dlf_nz, &
                    & dlf_ncons, dlf_nconn, dlf_coords, dlf_n_po_scaling
      use basato_module ! for xa
      implicit none
      integer, intent(in) :: natoms
      integer             :: nvar, nvar2, nspec, master

      ! ****************************************************************

      dlf_coords(:) = reshape(xa(:3,:natoms), (/ 3*natoms /))

      nvar = 3*natoms
      nvar2 = dlf_nframe*nvar + dlf_nweight + dlf_nmass + dlf_n_po_scaling
      nspec = natoms + dlf_nz + 5*dlf_ncons + 2*dlf_nconn
      master = 1 ! expedient way of dealing with this with crystal, as the 
                 ! master variable was introduced to help gamess cope with I/O

      call dl_find(nvar, nvar2, nspec, master)

      write(6,*)'DL-FIND COMPLETED'

      return
      end subroutine dl_find_crystal

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_params(nvar,nvar2,nspec,coords,coords2,spec,ierr, &
    tolerance,printl,maxcycle,maxene,tatoms,icoord, &
    iopt,iline,maxstep,scalestep,lbfgs_mem,nimage,nebk,dump,restart,&
    nz,ncons,nconn,update,maxupd,delta,soft,inithessian,carthessian,tsrel, &
    maxrot,tolrot,nframe,nmass,nweight,timestep,fric0,fricfac,fricp, &
    imultistate, state_i,state_j,pf_c1,pf_c2,gp_c3,gp_c4,ln_t1,ln_t2, &
    printf,tolerance_e,distort,massweight,minstep,maxdump,task,temperature, &
    po_pop_size,po_radius,po_contraction,po_tolerance_r,po_tolerance_g, &
    po_distribution,po_maxcycle,po_init_pop_size,po_reset,po_mutation_rate, &
    po_death_rate,po_scalefac,po_nsave,ntasks,tdlf_farm,n_po_scaling, &
    neb_climb_test,neb_freeze_test,nzero, coupled_states, qtsflag, &
    imicroiter, maxmicrocycle, micro_esp_fit )

  use dlf_parameter_module, only: rk
  use dlfind_module
  implicit none
  integer   ,intent(in)      :: nvar 
  integer   ,intent(in)      :: nvar2
  integer   ,intent(in)      :: nspec
  real(float), intent(inout) :: coords(nvar)  ! start coordinates
  real(float), intent(inout) :: coords2(max(nvar2,1)) ! a real array that can be 
                                           ! used depending on the calculation
                                           ! e.g. a second set of coordinates
  integer   ,intent(inout)   :: spec(nspec)  ! specifications like fragment or frozen
  integer   ,intent(out)     :: ierr
  real(float)  ,intent(inout)   :: tolerance
  real(float)  ,intent(inout)   :: tolerance_e
  integer   ,intent(inout)   :: printl
  integer   ,intent(inout)   :: maxcycle
  integer   ,intent(inout)   :: maxene
  integer   ,intent(inout)   :: tatoms
  integer   ,intent(inout)   :: icoord
  integer   ,intent(inout)   :: iopt
  integer   ,intent(inout)   :: iline
  real(float)  ,intent(inout)   :: maxstep
  real(float)  ,intent(inout)   :: scalestep
  integer   ,intent(inout)   :: lbfgs_mem
  integer   ,intent(inout)   :: nimage
  real(float)  ,intent(inout)   :: nebk
  integer   ,intent(inout)   :: dump
  integer   ,intent(inout)   :: restart
  integer   ,intent(inout)   :: nz
  integer   ,intent(inout)   :: ncons
  integer   ,intent(inout)   :: nconn
  integer   ,intent(inout)   :: update
  integer   ,intent(inout)   :: maxupd
  real(float)  ,intent(inout)   :: delta
  real(float)  ,intent(inout)   :: soft
  integer   ,intent(inout)   :: inithessian
  integer   ,intent(inout)   :: carthessian
  integer   ,intent(inout)   :: tsrel
  integer   ,intent(inout)   :: maxrot
  real(float)  ,intent(inout)   :: tolrot
  integer   ,intent(inout)   :: nframe
  integer   ,intent(inout)   :: nmass
  integer   ,intent(inout)   :: nweight
  real(float)  ,intent(inout)   :: timestep
  real(float)  ,intent(inout)   :: fric0
  real(float)  ,intent(inout)   :: fricfac
  real(float)  ,intent(inout)   :: fricp
  integer   ,intent(inout)   :: imultistate
  integer   ,intent(inout)   :: state_i
  integer   ,intent(inout)   :: state_j
  real(float)  ,intent(inout)   :: pf_c1  
  real(float)  ,intent(inout)   :: pf_c2  
  real(float)  ,intent(inout)   :: gp_c3  
  real(float)  ,intent(inout)   :: gp_c4
  real(float)  ,intent(inout)   :: ln_t1  
  real(float)  ,intent(inout)   :: ln_t2  
  integer   ,intent(inout)   :: printf
  real(float)  ,intent(inout)   :: distort
  integer   ,intent(inout)   :: massweight
  real(float)  ,intent(inout)   :: minstep
  integer   ,intent(inout)   :: maxdump
  integer   ,intent(inout)   :: task
  real(float)  ,intent(inout)   :: temperature
  integer   ,intent(inout)   :: po_pop_size
  real(float)  ,intent(inout)   :: po_radius
  real(float)  ,intent(inout)   :: po_contraction
  real(float)  ,intent(inout)   :: po_tolerance_r
  real(float)  ,intent(inout)   :: po_tolerance_g
  integer   ,intent(inout)   :: po_distribution
  integer   ,intent(inout)   :: po_maxcycle
  integer   ,intent(inout)   :: po_init_pop_size
  integer   ,intent(inout)   :: po_reset
  real(float)  ,intent(inout)   :: po_mutation_rate
  real(float)  ,intent(inout)   :: po_death_rate
  real(float)  ,intent(inout)   :: po_scalefac
  integer   ,intent(inout)   :: po_nsave
  integer   ,intent(inout)   :: ntasks
  integer   ,intent(inout)   :: tdlf_farm
  integer   ,intent(inout)   :: n_po_scaling
  real(float)  ,intent(inout)   :: neb_climb_test
  real(float)  ,intent(inout)   :: neb_freeze_test
  integer   ,intent(inout)   :: nzero
  integer   ,intent(inout)   :: coupled_states
  integer   ,intent(inout)   :: qtsflag
  integer   ,intent(inout)   :: imicroiter
  integer   ,intent(inout)   :: maxmicrocycle
  integer   ,intent(inout)   :: micro_esp_fit

  ! local variables
! **********************************************************************

! Strategy: crystal's INPOPT reads in the DLFIND section of the input deck
!           and stores the corresponding variables in dlfind_module, with a
!           dlf_ prefix.
!           When dl_find is called, dlf_default_init sets initial values to 
!           junk in most cases, to recognise user input.
!           Then dlf_get_params is called to fill arguments with input values.
!           Then dlf_default_set assigns default values to stuff not assigned
!           via input.
!
!           So, here in dlf_get_params, dummy arguments corresponding to 
!           variables that will never be under user control can simply not be 
!           assigned if the defaults are what we want.
!           Also need to give initial values to the dlf_ variables that match 
!           the default_init ones, to safely cover an absent input directive.
!           Inconsistent options will be dealt with in or just after the input 
!           reading routine.

! A note on units: the array section containing the masses has them in atomic 
! mass units.  Coordinates are in bohr (atomic units) and energies are returned 
! from crystal in hartree (atomic units), with corresponding gradients.
! The temperature should be input in kelvin.
! The coords in dlf_put_coords are in bohr, so that crystal can do its own usual 
! conversions.
! The routine write_xyz in dlf_util.f90 does a conversion to Angstrom before writing.

! Things hardwired
  ierr = 0 ! not used here for error-checking
  tatoms = 1 ! could possibly use tatoms = 0 to fudge input of coords in terms 
             ! of symmetry allowed directions.  Need to investigate further...
  tdlf_farm = 0

! Things from CRYSTAL's internal data structures
  coords(:nvar) = dlf_coords(:nvar)
  nz = dlf_nz
  nmass = dlf_nmass

! Things defined by reading the dl-find section of the CRYSTAL input deck
  coords2(:) = 0.0D0
  if (nvar2 > 0 .and. allocated(dlf_coords2)) coords2(:nvar2) = dlf_coords2(:nvar2)
  nweight = dlf_nweight
  maxcycle= dlf_maxcycle
  maxene= dlf_maxene
  spec(:) = dlf_spec(:)

  printl= dlf_printl
  printf= dlf_printf
  tolerance= dlf_tolerance
  tolerance_e= dlf_tolerance_e
  iopt= dlf_iopt
  iline= dlf_iline
  maxstep= dlf_maxstep
  scalestep= dlf_scalestep
  lbfgs_mem= dlf_lbfgs_mem
  update= dlf_update
  maxupd= dlf_maxupd
  delta= dlf_delta
  inithessian = dlf_inithessian 
  carthessian= dlf_carthessian
  minstep =  dlf_minstep
  maxdump= dlf_maxdump
  nimage= dlf_nimage
  icoord= dlf_icoord
  soft= dlf_soft
  nebk= dlf_nebk

! new options, not yet implemented in CRYSTAL
  neb_climb_test= -1.0d0
  neb_freeze_test= -1.0d0

  maxrot= dlf_maxrot
  tolrot= dlf_tolrot
  tsrel = dlf_tsrel
  massweight = dlf_massweight
  nframe = dlf_nframe
  distort = dlf_distort
  imultistate = dlf_imultistate
  state_i = dlf_state_i
  state_j = dlf_state_j
  pf_c1 = dlf_pf_c1
  pf_c2 = dlf_pf_c2
  gp_c3 = dlf_gp_c3
  gp_c4 = dlf_gp_c4
  ln_t1 = dlf_ln_t1
  ln_t2 = dlf_ln_t2

  ncons= dlf_ncons
  nconn= dlf_nconn
  dump= dlf_dump
  restart= dlf_restart

  timestep= dlf_timestep
  fric0= dlf_fric0
  fricfac= dlf_fricfac
  fricp= dlf_fricp

  task = dlf_task

  temperature = dlf_temperature

  po_pop_size = dlf_po_pop_size
  po_radius = dlf_po_radius
  po_contraction = dlf_po_contraction
  po_tolerance_r = dlf_po_tolerance_r
  po_tolerance_g = dlf_po_tolerance_g
  po_distribution = dlf_po_distribution
  po_maxcycle = dlf_po_maxcycle
  po_init_pop_size = dlf_po_init_pop_size
  po_reset = dlf_po_reset
  po_mutation_rate = dlf_po_mutation_rate
  po_death_rate = dlf_po_death_rate
  po_scalefac = dlf_po_scalefac
  po_nsave = dlf_po_nsave
  n_po_scaling = dlf_n_po_scaling

  ntasks = dlf_ntasks
  tdlf_farm = 0
  nzero=0 ! new option, not yet implemented in crystal
  coupled_states=1 ! new option, not yet implemented in crystal

end subroutine dlf_get_params

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_gradient(nvar,coords,energy,gradient,iimage,status)
  use dlf_parameter_module, only: rk
  use gradient_memory, only : forze
  use parinf_module, only: inf
!  use unit11
!  use unit95

  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(out)   :: status
  real(rk)                 :: dum_coords(nvar)
  logical                  :: err1, err2

  !external ftn

! **********************************************************************

! ??? how to deal with coords vs displacements in the array passed to ftn?
! Currently, ftn is coaxed into accepting coordinates rather than displacements...

  status = 0 ! status /= 0 means error
  err1 = .false.
  err2 = .false.
  ! the two reinits below are not needed for the I/O-to-disk main branch of crystal.
  ! If they are used with the DL branch, check that everything is updated that needs 
  ! to be before calling!  (e.g. coordinates)
  !call reinit95()
  !call reinit11()
  dum_coords(:) = coords(:)
  call ftn(dum_coords, energy, err1)

  if (inf(35) == 1) then ! flags SCF convergence failure: too many cycles
     if (inf(66) == 1) then
        gradient(:nvar) = -forze(:nvar)
     else
        status = 1
     end if
  else
     gradient(:nvar) = -forze(:nvar)
  end if
! Note that the call to crystal's totgra (libforce6.f), in which the array forze is allocated, 
! is skipped if inf(35) == 1 *unless* inf(66) /= 0 (see keyword POSTSCF in crystal's input.f90)

! A simple test has been used to check that this assignment of forze to gradient is 
! equivalent to calling gftn as below, for analytic gradients and symmdir .true.
! Just missing some print statements and the catch that will shut the program down if 
! symmetrised analytic gradients aren't available.
  !call gftn(uxst, gradient, err2, ftn)

  if (err1 .or. err2) status = 1 

end subroutine dlf_get_gradient

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_hessian(nvar,coords,hessian,status)
  !  get the hessian at a given geometry
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: hessian(nvar,nvar)
  integer   ,intent(out)   :: status

! **********************************************************************

! Dummy routine currently; numerical Hessian could be provided here if desired.

  call dlf_fail("dlf_get_hessian called but not coded currently")

end subroutine dlf_get_hessian

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_coords(nvar,mode,energy,coords,iam)
  use dlf_parameter_module
  use dlf_global, only: stdout
  use basato_module
  use parinf_module, only: lprint, iunit
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: mode
  integer   ,intent(in)    :: iam
  real(rk)  ,intent(in)    :: energy
  real(rk)  ,intent(in)    :: coords(nvar)

  integer                  :: i, j, ii, lpr
  logical                  :: od
  character(len=80)        :: fname
  character(len=60)        :: label
! **********************************************************************

! Note that what gets passed back to mainopt via OPTS(X) is nonsense 
! currently (actually it should still be a vector of zeros, unchanged 
! from the original call) as nothing has been coded yet to cover passing 
! back the optimized geometry.

! Only do this writing of files if I am the rank-zero processor
  if (iam /= 0) return

  if (mod(nvar,3) == 0) then
    !assume coords are atoms.  Here (for CRYSTAL) this means that we're not 
    ! imposing any symmetry restrictions.

    ! Need to make sure that xa(:3,:LIM016) in Includes/BASATO.INC is updated.
    ii = 0
    do i = 1, nvar/3
       do j = 1, 3
          ii = ii + 1
          xa(j,i) = coords(ii)
       end do
    end do

    if (mode==2) then
      !write(stdout,*)"Writing TS mode in xyz format to fort.33"
      !lpr=lprint(33)
      !lprint(33) = 1
      !call COOPRT ! to write xyz style coords to fort.33
      !            ! Note that this file is written to in append mode!
      !lprint(33)=lpr
      write(stdout,*)"writing TS mode in xyz format to tsmode.xyz" 
      open(unit=400,file="tsmode.xyz")
      write(400,*) nvar/3
      write(400,*)
      do i = 1, nvar/3
        write(400,'(3f12.7)') coords( (i-1)*3+1 : (i-1)*3+3 )
      end do
      close(400)
    else if (mode==3) then
      write(stdout,*)"Writing coordinates to TS.str"
      inquire(unit=iunit(34),opened=od)
      if (od) close(iunit(34))
      call EXTPRT("TS.str")
      write(stdout,*)"Writing current coordinates in xyz format to fort.33"
      lpr=lprint(33)
      lprint(33) = 1
      call COOPRT ! to write xyz style coords to fort.33
                  ! Note that this file is written to in append mode!
      lprint(33)=lpr
    else if (mode==4) then
      write(stdout,*)"Writing coordinates to minimum_+.str"
      inquire(unit=iunit(34),opened=od)
      if (od) close(iunit(34))
      call EXTPRT("minimum_+.str")
      write(stdout,*)"Writing current coordinates in xyz format to fort.33"
      lpr=lprint(33)
      lprint(33) = 1
      call COOPRT ! to write xyz style coords to fort.33
                  ! Note that this file is written to in append mode!
      lprint(33)=lpr
    else if (mode==5) then
      write(stdout,*)"Writing coordinates to minimum_-.str"
      inquire(unit=iunit(34),opened=od)
      if (od) close(iunit(34))
      call EXTPRT("minimum_-.str")
      write(stdout,*)"Writing current coordinates in xyz format to fort.33"
      lpr=lprint(33)
      lprint(33) = 1
      call COOPRT ! to write xyz style coords to fort.33
                  ! Note that this file is written to in append mode!
      lprint(33)=lpr
    else if (mode < 0) then
      write(label, *) abs(mode)
      fname = "coords_"//trim(adjustl(label))//".str"
      write(stdout,*)"Writing coordinates to ",trim(adjustl(fname))
      inquire(unit=iunit(34),opened=od)
      if (od) close(iunit(34))
      call EXTPRT(trim(adjustl(fname)))
      write(stdout,*)"Writing current coordinates in xyz format to fort.33"
      lpr=lprint(33)
      lprint(33) = 1
      call COOPRT ! to write xyz style coords to fort.33
                  ! Note that this file is written to in append mode!
      lprint(33)=lpr
    else
      write(stdout,*)"Writing coordinates to coords.str" 
      inquire(unit=iunit(34),opened=od)
      if (od) close(iunit(34))
      call EXTPRT("coords.str")
      write(stdout,*)"Writing current coordinates in xyz format to fort.33" 
      lpr=lprint(33)
      lprint(33) = 1
      call COOPRT ! to write xyz style coords to fort.33 
                  ! Note that this file is written to in append mode!
      lprint(33)=lpr
    end if
  else
    !print*,"Coords in put_coords: ",coords
  end if
end subroutine dlf_put_coords

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_error()
  implicit none
! **********************************************************************

! 1st argument 0 means stop on error, rather than print a warning or info
! Will also take care of the MPI_Abort()

  call ERRVRS(0,"dlf_error","Failure in DL-FIND")

end subroutine dlf_error

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_update()
  implicit none
! **********************************************************************
  ! only a dummy routine here.
end subroutine dlf_update


subroutine dlf_get_multistate_gradients(nvar,coords,energy,gradient,iimage,status)
  ! only a dummy routine up to now
  ! for conical intersection search
  use dlf_parameter_module
  implicit none
  integer   ,intent(in)    :: nvar
  integer   ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(in)    :: energy(2)
  real(rk)  ,intent(in)    :: gradient(nvar,2)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: status

  call dlf_fail("dlf_get_multistate_gradients called but not coded currently")

end subroutine dlf_get_multistate_gradients


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)

  implicit none

  integer, intent(in) :: dlf_nprocs ! total number of processors
  integer, intent(in) :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer, intent(in) :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument

! not needed as CRYSTAL will sort out the MPI stuff
  call dlf_fail("dlf_put_procinfo called in error")

end subroutine dlf_put_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_procinfo(dlf_nprocs, dlf_iam, dlf_global_comm)
  use Numbers
  use PARAL1_MODULE, only: global_nproc, global_iam, global_comm
  implicit none

  integer :: dlf_nprocs ! total number of processors
  integer :: dlf_iam ! my rank, from 0, in mpi_comm_world
  integer :: dlf_global_comm ! world-wide communicator
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

  dlf_nprocs = global_nproc
  dlf_iam = global_iam
  dlf_global_comm = global_comm

end subroutine dlf_get_procinfo


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_put_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)

  implicit none

  integer, intent(in) :: dlf_ntasks          ! number of taskfarms
  integer, intent(in) :: dlf_nprocs_per_task ! no of procs per farm
  integer, intent(in) :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer, intent(in) :: dlf_mytask          ! rank of my farm, from 0
  integer, intent(in) :: dlf_task_comm       ! communicator within each farm
  integer, intent(in) :: dlf_ax_tasks_comm   ! communicator involving the 
                                             ! i-th proc from each farm
! **********************************************************************

!!! variable in the calling program = corresponding dummy argument 

! not needed as CRYSTAL will sort out the MPI stuff
  call dlf_fail("dlf_put_taskfarm called in error")

end subroutine dlf_put_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_get_taskfarm(dlf_ntasks, dlf_nprocs_per_task, dlf_iam_in_task, &
                        dlf_mytask, dlf_task_comm, dlf_ax_tasks_comm)
  use Numbers
  use PARAL1_MODULE, only: ntasks, nproc, iam, mytask, task_comms, ax_tasks_comms
  implicit none

  integer :: dlf_ntasks          ! number of taskfarms
  integer :: dlf_nprocs_per_task ! no of procs per farm
  integer :: dlf_iam_in_task     ! my rank, from 0, in my farm
  integer :: dlf_mytask          ! rank of my farm, from 0
  integer :: dlf_task_comm       ! communicator within each farm
  integer :: dlf_ax_tasks_comm   ! communicator involving the
                                 ! i-th proc from each farm
! **********************************************************************

!!! dummy argument = corresponding variable in the calling program

! we should only be in this routine AFTER make_task_farm in libxMPI.f 
! has been called (with its replacement of the iam, nproc for MPI_COMM_WORLD 
! with values for the taskfarms)

  dlf_ntasks = ntasks
  dlf_nprocs_per_task = nproc
  dlf_iam_in_task = iam
  dlf_mytask = mytask 
  dlf_task_comm = task_comms
  dlf_ax_tasks_comm = ax_tasks_comms

end subroutine dlf_get_taskfarm


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_output(dum_stdout, dum_stderr)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,keep_alloutput
  implicit none
  integer :: dum_stdout
  integer :: dum_stderr
  integer :: ierr

! sort out output units; particularly important on multiple processors
 
! set unit numbers for main output and error messages
  if (dum_stdout >= 0) stdout = dum_stdout 
  if (dum_stderr >= 0) stderr = dum_stderr

  if (glob%nprocs > 1) then
     ! write some info on the parallelization
     write(stdout,'(1x,a,i10,a)')"I have rank ",glob%iam," in mpi_comm_world"
     write(stdout,'(1x,a,i10)')"Total number of processors = ",glob%nprocs
     if (keep_alloutput) then
        write(stdout,'(1x,a)')"Keeping output from all processors"
     else
        write(stdout,'(1x,a)')"Not keeping output from processors /= 0"
     end if
  end if

end subroutine dlf_output
