! **********************************************************************
! **                                                                  **
! **                      Parallel optimisation                       **
! **                                                                  **
! **********************************************************************
!!****h* DL-FIND/parallel_opt
!!
!! NAME
!! parallel_opt
!!
!! FUNCTION
!! Performs a parallel optimisation; currently either stochastic search 
!! or genetic algorithm.
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk),
!!  Joanne Carr (j.m.carr@dl.ac.uk)
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as
!!  published by the Free Software Foundation, either version 3 of the
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public
!!  License along with DL-FIND.  If not, see
!!  <http://www.gnu.org/licenses/>.
!!
!! SOURCE
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_parallel_opt
!!
!! FUNCTION
!! Set up, perform and close down after, the parallel optimisation
!! Note that two numbers are hardwired in the code at present -- 
!! small, mainly used as the tolerance on fractional energy differences to 
!! distinguish between structures; and the reduction of the radius by a 
!! factor of 4 after the initial population is set up in the genetic algorithm
!! (search for the comment: Scale the radius from its initial value).
!!
!! INPUTS
!!
!! glob%xcoords
!!
!! for SS -- 
!! glob%po_pop_size
!! glob%po_radius(:)
!! glob%po_contraction
!! glob%po_tolerance_r(:)
!! glob%po_tolerance_g
!! glob%po_distribution
!! glob%po_maxcycle
!! glob%po_scalefac
!!
!! for GA -- 
!! glob%po_pop_size
!! glob%po_radius(:)
!! glob%po_tolerance_g
!! glob%po_maxcycle
!! glob%po_init_pop_size
!! glob%po_reset
!! glob%po_mutation_rate
!! glob%po_death_rate
!! glob%po_nsave
!!
!! OUTPUTS
!!
!! tconv
!!
!! Local variables:
!!  * xcoords_best(:,:)
!!  * energy_best
!!  * for GA, if nsave/=0 -- energies_save(:)
!!                           nevals_save(:)
!!                           xcoords_save(:,:,:)
!!
!! Files, if converged -- best.xyz
!! best_active.xyz
!!
!! SYNOPSIS
subroutine dlf_parallel_opt(trestarted_report, tconv &
#ifdef GAMESS
    ,core&
#endif
    )
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl,printf
  use dlf_stat, only: stat
  use dlf_allocate, only: allocate,deallocate
  use dlf_checkpoint
  use dlf_sort_module
  implicit none
#ifdef GAMESS
  real(rk) :: core(*) ! GAMESS memory, not used in DL-FIND
#endif
  integer              :: status, iimage, i, j, k, seed_size, nmutations 
  integer              :: nresets, noffspring, lower_index, arr_size
  integer,dimension(1) :: position
  integer,dimension(8) :: time_data
  integer,allocatable  :: seed(:), nevals_save(:)
  logical              :: stochastic, genetic, trestarted_report, tconv, texit
  logical              :: dummy_logical, trestarted, tno_diversity
  real(rk)             :: random, energy_best
  real(rk)             :: mean_init_e, mean_init_e2, sigma_init_e
  real(rk),allocatable :: pop_energies(:), init_pop_energies(:)
  real(rk),allocatable :: icoords_best(:), igradient_best(:), xcoords_best(:,:)
  real(rk),allocatable :: energies_save(:), xcoords_save(:,:,:)
  real(rk),allocatable :: pop_icoords(:,:), init_pop_icoords(:,:)
  real(rk),allocatable :: pop_xgradient(:,:,:)

! may want to change this for different systems...
  real(rk), parameter  :: small=1.0D-12

! **********************************************************************

! Initialise variables
    stochastic = (glob%iopt == 51)
    genetic = (glob%iopt == 52)
    tconv = .false.
    texit = .false.
    seed_size = 1
    energy_best = huge(1.0D0)
    iimage = 1
!    stat%sene = 0
!    stat%ccycle = 0
    ! GA variables
    if (genetic) then
       nresets = 0
       tno_diversity = .false.

       if (glob%po_pop_size < 4) then
          write(stdout,'(1x,a)')"ERROR: working population size is too small"
          write(stdout,'(1x,a)')"Size must be >= 4 for genetic algorithm"
          call dlf_fail("Input population size too small")
       end if

       if (glob%po_init_pop_size < glob%po_pop_size) then
          write(stdout,'(1x,a)')"ERROR: initial population size is too small"
          write(stdout,'(1x,a)')"Size must be >= working population size"
          call dlf_fail("Input initial population size too small")
       end if

       nmutations = nint(glob%po_pop_size * glob%nivar * glob%po_mutation_rate)
       ! use nint above to allow nmutations = 0
       if (nmutations > glob%po_pop_size * glob%nivar) &
                                     &nmutations = glob%po_pop_size * glob%nivar
       noffspring = int(glob%po_pop_size * glob%po_death_rate) + 1
       ! int truncates,  so noffspring will never be zero
       ! check noffspring against the upper limit of glob%po_pop_size - 2
       ! (want to leave at least two individuals as possible parents)
       if (noffspring > glob%po_pop_size - 2) noffspring = glob%po_pop_size - 2
       if (mod(noffspring,2) == 1) then
          ! want to replace the two parents with their two offspring, so ensure 
          ! noffspring is even by increasing the value by 1 if it is odd
          noffspring = noffspring + 1
          if (noffspring > glob%po_pop_size - 2) noffspring = noffspring - 2
       end if

       write(stdout,'(1x,a,i10)')"In dlf_parallel_opt, noffspring = ",noffspring
       write(stdout,'(1x,a,i10)')"In dlf_parallel_opt, nmutations = ",nmutations
    end if
    
! Initialise random number generator on the zeroth processor only
! (no other processor should generate any random numbers...)
! (pgf95 is rather picky about the random number seed, hence the 
! complicated procedure below...)

    if (glob%iam == 0) then
       call random_seed(size = seed_size)
       call allocate(seed, seed_size)
       seed = 1
       call date_and_time(values = time_data)
       seed(1) = time_data(8)
       do i = 1, seed_size - 1
          seed(i + 1) = time_data(8 - mod(i,8)) * seed(1) * i
          ! use smallest units of time_data first, and make the repeats different
          if (seed(i + 1) == 0) seed(i + 1) = i
       end do
       seed = 11*seed ! so seed does not just contain small values
                      ! Also helps to ensure that not all the elements are even 
                      ! (which hinders performance for pgf95)
       call random_seed(put = seed)
       !!! for testing    call random_seed(get = seed)
       write(stdout,'(1x,a)')"Random number seed = "
       do i = 1, seed_size
          write(stdout,*) seed(i)
       end do
       write(stdout,'(1x,a)')"End of random number seed"
       call random_number(random)
       write(stdout,*)"random 1 ",random
       call random_number(random)
       write(stdout,*)"random 2 ",random
    
       call deallocate(seed)
    end if

! allocate storage
    call allocate(pop_icoords, glob%po_pop_size, glob%nivar)
    call allocate(pop_xgradient, glob%po_pop_size, 3, glob%nat)
    call allocate(pop_energies, glob%po_pop_size)
    call allocate(icoords_best, glob%nivar)
    call allocate(igradient_best, glob%nivar)
    call allocate(xcoords_best, 3, glob%nat)
    call allocate(glob%po_radius, glob%nivar)
    call allocate(glob%po_tolerance_r, glob%nivar)

! Set the sample and tolerance radii arrays 
! Note that on restart, the base values glob%po_radius_base and glob%po_tol_r_base
! are obtained from the restart file, but the contents of glob%po_radius_scaling(:) 
! must still be determined from the passed tmpcoords2 array in dlf_get_params. 
! glob%po_radius_scaling(:) can therefore be different from the original if desired.
    arr_size = SIZE(glob%po_radius_scaling, 1)
    if (arr_size == 1 .and. glob%nivar > 1) then
       glob%po_radius(:) = glob%po_radius_base
       glob%po_tolerance_r(:) = glob%po_tol_r_base
    elseif (arr_size == glob%nivar) then
       glob%po_radius(:) = glob%po_radius_base * glob%po_radius_scaling(:)
       glob%po_tolerance_r(:) = glob%po_tol_r_base * glob%po_radius_scaling(:)
    else
       call dlf_fail("Radii scaling factors incompatible with working coordinates")
    end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Setup of initial population or individual
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (stochastic) then

       lower_index = 1

       ! read checkpoint file OR sort out initial individual
       if (glob%restart == 1) then

          call clock_start("CHECKPOINT")
          call dlf_checkpoint_po_read
          call clock_stop("CHECKPOINT")
          if (.not.trestarted) call dlf_fail("Restart attempt failed")

       else

          ! evaluate initial point
          call clock_start("EANDG")
          call dlf_get_gradient(glob%nvar,glob%xcoords,energy_best, &
               glob%xgradient,iimage,&
#ifdef GAMESS
               core,&
#endif
               status)
          stat%sene = stat%sene + 1
          stat%pene = stat%pene + 1
          call clock_stop("EANDG")

          ! check of NaN in the energy (comparison of NaN with any number
          ! returns .false. , pgf90 does not understand isnan() )
          if ( abs(energy_best) > huge(1.D0) ) then
            status = 1
          else
            if (.not. abs(energy_best) < huge(1.D0) ) status = 1
          end if

          if (status/=0) then
             call dlf_report(trestarted_report)
             call dlf_fail("Energy evaluation failed")
          end if

          xcoords_best(:,:) = glob%xcoords(:,:)
   
          ! convert glob%xcoords and glob%xgradient from Cartesians to the 
          ! working (i-labelled) coordinates glob%icoords and glob%igradient
          call clock_start("COORDS")
          call dlf_coords_xtoi(dummy_logical,dummy_logical,iimage,0)
          call clock_stop("COORDS")

          icoords_best(:) = glob%icoords(:)
          igradient_best(:) = glob%igradient(:)

       end if 

    else if (genetic) then

       lower_index = 2
       call allocate(init_pop_icoords,glob%po_init_pop_size,glob%nivar)
       call allocate(init_pop_energies,glob%po_init_pop_size)

       if (glob%po_nsave > 0) then
          call allocate(xcoords_save,glob%po_nsave,3,glob%nat)
          call allocate(energies_save,glob%po_nsave)
          call allocate(nevals_save,glob%po_nsave)
          energies_save(:) = huge(1.0D0)
          nevals_save(:) = 0
          xcoords_save(:,:,:) = 0.0D0
       end if

       ! read checkpoint file OR sort out initial population
       if (glob%restart == 1) then

          call clock_start("CHECKPOINT")
          call dlf_checkpoint_po_read
          call clock_stop("CHECKPOINT")
          if (.not.trestarted) call dlf_fail("Restart attempt failed")

       else

          ! convert glob%xcoords from Cartesians to the
          ! working (i-labelled) coordinates glob%icoords
          ! The glob%[x,i]gradient arrays contain junk at this point
          glob%xgradient = 0.0D0 ! Safest to initialize though
          call clock_start("COORDS")
          call dlf_coords_xtoi(dummy_logical,dummy_logical,iimage,0)
          call clock_stop("COORDS")

          ! Make initial population...
          if (glob%iam == 0) then
             call clock_start("FORMSTEP")
             call dlf_genetic_initialpop(init_pop_icoords(:,:),&
                                & glob%po_init_pop_size, glob%icoords(:))
             call clock_stop("FORMSTEP")
          end if

          call dlf_global_real_bcast(init_pop_icoords(:,:), &
                                   & glob%po_init_pop_size*glob%nivar, 0)

          ! ...then work out the energy of each individual...
          init_pop_energies(:) = 0.0D0
          do j = 1, glob%po_init_pop_size
             k = mod(j,glob%ntasks)
             stat%sene = stat%sene + 1
             if (k == glob%mytask) then

                glob%icoords(:) = init_pop_icoords(j,:)
   
                ! convert glob%icoords (used as a temporary storage array) back
                ! to Cartesians in glob%xcoords, before call to dlf_get_gradient
   
                call clock_start("COORDS")
                call dlf_coords_itox(iimage)
                call clock_stop("COORDS")

                call clock_start("EANDG")
!!! using pop_xgradient array as a dummy in the call, because IN THE CURRENT 
!!! CODE it will not be used before it is filled with the proper values, and
!!! to save memory I don't want an unnecessary init_pop_xgradient array.
                call dlf_get_gradient(glob%nvar,glob%xcoords, &
                     init_pop_energies(j), pop_xgradient(1,:,:),iimage,&
#ifdef GAMESS
                     core,&
#endif
                     status)
                stat%pene = stat%pene + 1
                call clock_stop("EANDG")

                ! check of NaN in the energy (comparison of NaN with any number
                ! returns .false. , pgf90 does not understand isnan() )
                if ( abs(init_pop_energies(j)) > huge(1.D0) ) then
                  status = 1
                else
                  if (.not. abs(init_pop_energies(j)) < huge(1.D0) ) status = 1
                end if

                if (status/=0) then
                   init_pop_energies(j) = huge(1.0D0)
                   !pop_xgradient(j,:,:) = huge(1.0D0)
                end if

             endif
          enddo

          !!!call dlf_global_real_sum(init_pop_energies(:), glob%po_init_pop_size)
          !!!init_pop_energies(:) = init_pop_energies(:)/glob%nprocs_per_task
          call dlf_tasks_real_sum(init_pop_energies(:), glob%po_init_pop_size)

          ! ...then sort the population on energy...
          call dlf_sort(init_pop_icoords(:,:), init_pop_energies(:), &
                      & glob%po_init_pop_size, glob%nivar)

          ! Write information for the starting population
          mean_init_e = sum(init_pop_energies(:)) / dble(glob%po_init_pop_size)
          mean_init_e2 =sum(init_pop_energies(:)**2)/dble(glob%po_init_pop_size)
          sigma_init_e = sqrt(mean_init_e2 - mean_init_e**2)

          write(stdout,'(1x,a)')"Initial population:"
          write(stdout,'(1x,a,es16.9)')"Lowest energy = ",init_pop_energies(1)
          write(stdout,'(1x,a,es16.9)')"Mean energy = ",mean_init_e
          write(stdout,'(1x,a,es16.9)')"Standard deviation of the energy = ",&
                                       &sigma_init_e

          ! ...and put the glob%po_pop_size fittest (=lowest-energy) individuals
          ! into the working population, pop_icoords(:,:) and pop_energies(:)
          pop_icoords(:,:) = init_pop_icoords(:glob%po_pop_size,:)
          pop_energies(:) = init_pop_energies(:glob%po_pop_size)

       end if

       ! Scale the radius from its initial value, for use in setting up the 
       ! population, to something more appropriate for the mutations.

       glob%po_radius(:) = 0.25D0*glob%po_radius(:)

    else
    
       call dlf_fail("Parallel optimisation option requested that does not &
                     &yet exist")

    end if
   
    call flush(stdout)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! End setup of initial population or individual
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Main cycles loop
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    do i = 1, glob%po_maxcycle

       stat%ccycle = stat%ccycle + 1

! Generate the individuals in the population for this cycle and store them in 
! pop_icoords(:,:)
       if (glob%iam == 0) then
          call clock_start("FORMSTEP")
          if (stochastic) then
             call dlf_stoch_makepop
          else if (genetic) then
             call dlf_genetic_makepop
          end if
          call clock_stop("FORMSTEP")
       end if

       if (genetic) then
          call dlf_global_log_bcast(tno_diversity, 1, 0)
          if (tno_diversity) then
             write(stdout,'(1x,a)')"dlf_parallel_opt WARNING"
             write(stdout,'(1x,a)')"Insufficient genetic diversity in population"
             !exit ! don't necesarily need to exit here
             write(stdout,'(1x,a)')"Resetting population from known individual &
                                   &of lowest energy"
             write(stdout,'(1x,a)')"(Could also try a higher mutation rate)"
             if (glob%iam == 0) then
                call clock_start("FORMSTEP")
                call dlf_genetic_initialpop(pop_icoords(:,:), &
                                   & glob%po_pop_size, pop_icoords(1,:))
                ! Note, dlf_genetic_initialpop doesn't calculate energies
                call clock_stop("FORMSTEP")
             end if
             tno_diversity = .false.
          else
             if (nmutations > 0) then
                if (glob%iam == 0) call dlf_genetic_mutate
             end if
          end if
       end if

       ! This call can go here rather than before the if (genetic) block above 
       ! because the subroutine calls that require pop_icoords(:,:) to be known
       ! should only be made from the master processor (glob%iam ==0), which knows 
       ! the full pop_icoords after the call to dlf_genetic_makepop

      call dlf_global_real_bcast(pop_icoords(:,:),glob%po_pop_size*glob%nivar,0)

       ! This (re)initialization is necessary for the parallel implementation, 
       ! when MPI_Allreduce calls are made for the pop_energies(:) and 
       ! pop_xgradient(:,:,:) arrays.
       call dlf_init_engarrays

       ! calculate energy and gradient for each individual in the population
       ! (apart from the first individual in a GA run...)
       call dlf_get_engarrays

! Make the full pop_energies (and pop_xgradient) arrays known on all processors
       !!!call dlf_global_real_sum(pop_energies(:), glob%po_pop_size)
       !!!pop_energies(:) = pop_energies(:)/glob%nprocs_per_task
       !!!call dlf_global_real_sum(pop_xgradient(:,:,:), glob%po_pop_size*glob%nvar)
       !!!pop_xgradient(:,:,:) = pop_xgradient(:,:,:)/glob%nprocs_per_task
       call dlf_tasks_real_sum(pop_energies(:), glob%po_pop_size)
       call dlf_tasks_real_sum(pop_xgradient(:,:,:), glob%po_pop_size*glob%nvar)

! Prepare for next cycle; test for convergence or failure
       if (stochastic) then
          call dlf_stoch_endcycle
          if (texit) exit
       else if (genetic) then
          call dlf_sort(pop_icoords(:,:), pop_xgradient(:,:,:), &
                  & pop_energies(:), glob%po_pop_size, glob%nivar, 3, glob%nat)
          call dlf_genetic_endcycle
       end if

       if ( (mod(i,glob%po_reset) == 0) .and. genetic) then
          nresets = nresets + 1
          write(stdout,'(1x,a)')"Resetting population from known individual of &
                                &lowest energy"

          if (glob%iam == 0) then
             call clock_start("FORMSTEP")
             call dlf_genetic_initialpop(pop_icoords(:,:), &
                                & glob%po_pop_size, pop_icoords(1,:))
             call clock_stop("FORMSTEP")
          end if

          call dlf_global_real_bcast(pop_icoords(:,:),&
                                    &glob%po_pop_size*glob%nivar,0)
          call dlf_init_engarrays
          call dlf_get_engarrays
          call dlf_tasks_real_sum(pop_energies(:), glob%po_pop_size)
          call dlf_tasks_real_sum(pop_xgradient(:,:,:), &
                                 &glob%po_pop_size*glob%nvar)
          call dlf_sort(pop_icoords(:,:), pop_xgradient(:,:,:), &
                  & pop_energies(:), glob%po_pop_size, glob%nivar, 3, glob%nat)

       end if

       call flush(stdout)

    end do ! the cycles loop

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! End of main cycles loop
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (genetic .and. tconv .and. glob%po_nsave > 0) then
       write(stdout,'(1x,a)')"End of genetic algorithm"
       write(stdout,'(1x,a,i10,a)')"List of ",glob%po_nsave," lowest-energy minima"
       do i = 1, glob%po_nsave
          if (energies_save(i) == huge(1.0D0)) then
             write(stdout,'(1x,a)')"Warning: no more unique minima found"
             exit
          end if
          write(stdout,'(1x,i10)') i
          write(stdout,'(1x,a,es16.9)')"Energy = ",energies_save(i)
          write(stdout,'(1x,a,i10,a)')"Found after ",nevals_save(i),&
                                     &" energy evaluations"
          write(stdout,'(1x,a)')"Cartesian coordinates:"
          write(stdout,'(3(1x,es16.9))') xcoords_save(i,:,:)
          call dlf_put_coords(glob%nvar,-i,energies_save(i),xcoords_save(i,:,:)&
                             &,glob%iam)
       end do
    end if

    call flush(stdout)
    call dlf_po_destroy ! deallocate memory
    return

contains
!!****

! Internal procedures as they should not be called from any subroutine apart 
! from dlf_parallel_opt


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_init_engarrays
!!
!! FUNCTION
!! 
!! Initialize population energy and gradient arrays to zero before any
!! possible MPI_Allreduce calls are made
!!
!! SYNOPSIS
subroutine dlf_init_engarrays
!! SOURCE

    ! This (re)initialization is necessary for the parallel implementation, when
    ! MPI_Allreduce calls are made.
    if (stochastic) then
       pop_xgradient(:,:,:) = 0.0D0
       pop_energies(:) = 0.0D0
    else if (genetic) then
       ! the first member of the population is excluded from mutation and is
       ! thus unchanged in this generation - no need to recalculate
       if (glob%mytask == 0) then
          pop_xgradient(2:glob%po_pop_size,:,:) = 0.0D0
          pop_energies(2:glob%po_pop_size) = 0.0D0
       else
          pop_xgradient(:,:,:) = 0.0D0
          pop_energies(:) = 0.0D0
       end if
    end if

end subroutine dlf_init_engarrays
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_get_engarrays
!!
!! FUNCTION
!!
!! Calculate energy and gradient for each individual in the population
!! (apart from the first individual in a GA run...)
!! Parallelised, with each taskfarm essentially calculating e and g for 
!! the i-th, (i+ntasks)-th, (i+2*ntasks)-th, etc. individuals.
!!
!! INPUTS
!!
!! pop_icoords(:,:)
!! lower_index
!! glob%po_pop_size
!! glob%ntasks
!! glob%mytask
!! iimage
!! glob%nvar
!!
!! OUTPUTS
!!
!! pop_energies(:)
!! pop_xgradient(:,:,:)
!!
!! SYNOPSIS
subroutine dlf_get_engarrays
!! SOURCE

integer :: l, m, status

    do m = lower_index, glob%po_pop_size
       l = mod(m,glob%ntasks)
       stat%sene = stat%sene + 1
       if (l == glob%mytask) then

          glob%icoords(:) = pop_icoords(m,:)

          ! convert glob%icoords (used as a temporary storage array) back
          ! to Cartesians in glob%xcoords, before call to dlf_get_gradient

          call clock_start("COORDS")
          call dlf_coords_itox(iimage)
          call clock_stop("COORDS")

          call clock_start("EANDG")
          call dlf_get_gradient(glob%nvar,glob%xcoords,pop_energies(m), &
               pop_xgradient(m,:,:),iimage,&
#ifdef GAMESS
               core,&
#endif
               status)
          stat%pene = stat%pene + 1
          call clock_stop("EANDG")

          ! check of NaN in the energy (comparison of NaN with any number
          ! returns .false. , pgf90 does not understand isnan() )
          if ( abs(pop_energies(m)) > huge(1.D0) ) then
            status = 1
          else
            if (.not. abs(pop_energies(m)) < huge(1.D0) ) status = 1
          end if

          if (status/=0) then
             pop_energies(m) = huge(1.0D0)
             pop_xgradient(m,:,:) = huge(1.0D0)
          end if

       endif
    enddo

end subroutine dlf_get_engarrays
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_stoch_makepop
!!
!! FUNCTION
!!
!! Create the population for a stochastic search.
!! Only executed on the rank-zero processor, as it depends on random 
!! numbers.
!! Works wholly in "i" coordinates.
!!
!! INPUTS
!!
!! glob%po_pop_size
!! glob%nivar
!! glob%iam
!! glob%po_distribution
!! glob%po_radius(:)
!! glob%po_scalefac
!! igradient_best(:)
!!
!! OUTPUTS
!!
!! pop_icoords(:,:)
!!
!! SYNOPSIS
subroutine dlf_stoch_makepop
!! SOURCE

integer :: l,m

    if (glob%iam /= 0) return ! as only iam == 0 has seeded the random number 
                              ! generator.

! we are working wholly in "i" coordinates in this subroutine

    pop_icoords(:,:) = 0.0D0
    do l = 1, glob%po_pop_size
       do m = 1, glob%nivar
          call random_number(random)
          select case (glob%po_distribution)
          case (3) ! force bias
            random = random * glob%po_radius(m)
            random = random * glob%po_scalefac * abs(igradient_best(m))
            if (igradient_best(m) > 0.0D0) random = -1.0D0*random
          case (2) ! force_direction_bias
            random = random * glob%po_radius(m)
            if (igradient_best(m) > 0.0D0) random = -1.0D0*random
          case (1)
            ! uniform distribution over the full hypersphere in config.space
            random = (random * 2.0D0 * glob%po_radius(m)) - glob%po_radius(m)
          case default
             write(stderr,'(1x,a,i2,a)') "Parallel optimisation &
             &distribution type ",glob%po_distribution," not implemented"
             call dlf_fail("Bad option for parallel optimisation distribution &
                           &type")
          end select
          pop_icoords(l,m) = icoords_best(m) + random
       end do
    enddo
   
end subroutine dlf_stoch_makepop
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_genetic_makepop
!!
!! FUNCTION
!!
!! Create the population during the cycles of the genetic algorithm.
!! Only executed on the rank-zero processor, as it depends on random 
!! numbers.
!! Works wholly in "i" coordinates.
!!
!! INPUTS
!!
!! glob%iam
!! glob%nivar
!! glob%po_pop_size
!! noffspring
!! pop_energies(:)
!! pop_icoords(:,:)
!!
!! OUTPUTS
!!
!! tno_diversity
!! pop_icoords(:,:) 
!!
!! SYNOPSIS
subroutine dlf_genetic_makepop
!! SOURCE

integer               :: l, m, kk, cross_over
integer,dimension(2)  :: ip
real(rk)              :: sum_w, dummy, dummy2, delta, harvest, blending
real(rk), allocatable :: w_cost(:)

    if (glob%iam /= 0) return ! as only iam == 0 has seeded the random number 
                              ! generator.

! set cost(i.e. energy)-weighted probabilities of parenthood for the breeding 
! population, i.e. the glob%po_pop_size - noffspring lowest-energy individuals 

    call allocate(w_cost,glob%po_pop_size - noffspring)
    sum_w = 0.0D0

    do l = 1, glob%po_pop_size - noffspring
       w_cost(l) = pop_energies(l) - pop_energies(glob%po_pop_size - noffspring + 1)
    end do

    sum_w = sum(w_cost)

    ! test whether the (sorted) energies of the breeding section of the 
    ! population are all identical to at least a chosen precision.

    dummy = abs( (pop_energies(1) - pop_energies(glob%po_pop_size-noffspring)) &
               & / pop_energies(1) )

    ! Also test for the case where, of the breeding population plus the 
    ! next-highest-energy configuration, the lowest is unique and the rest are 
    ! identical.  This situation would mean that the lowest-energy individual 
    ! is the first parent, and no second parent can be selected in the scheme 
    ! in the offspring generation loop below.

    dummy2 = abs( (pop_energies(2) - pop_energies(glob%po_pop_size-noffspring+1)) &
               & / pop_energies(2) )

    ! if the population is lacking in genetic diversity then return

    if (sum_w == 0.0D0 .or. dummy < small .or. dummy2 < small) then
       tno_diversity = .true.
       call deallocate(w_cost)
       return
    end if

    tno_diversity = .false.

    ! Accumulate the w_cost's

    w_cost(:) = w_cost(:)/sum_w

    do l = 2, glob%po_pop_size - noffspring
       w_cost(l) = w_cost(l) + w_cost(l-1)
    end do

    ! in case there are rounding errors, set the final w_cost to 1.0

    w_cost(glob%po_pop_size - noffspring) = 1.0D0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! offspring generation loop
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    do l = 1, noffspring - 1, 2

       ! select parents: ip(1:2).  Makes sure that ip(1) /= ip(2)

       kk = 1
       do
          call random_number(harvest)
          if (harvest < w_cost(1)) then
              ip(kk) = 1
              kk = kk + 1
          else
              do m = 2, glob%po_pop_size - noffspring
                 if (harvest <= w_cost(m)) then
                    ip(kk) = m
                    kk = kk + 1
                    exit
                 end if
              end do
          end if
          if (kk == 3) then
              if (ip(1) == ip(2)) then
                 kk = 2 ! choose a different "father"
              else
                 exit ! we have two different parents
              end if
          end if
       end do

       ! choose one of the glob%nivar variables as the cross-over point

       call random_number(harvest)
       cross_over = int( harvest * glob%nivar ) + 1
       delta = pop_icoords(ip(1),cross_over) - pop_icoords(ip(2),cross_over)

       ! Check we have a viable cross over point given the parents

       if ( (abs( delta/pop_icoords(ip(1),cross_over) ) < 1.0D3*small) .and. &
           &(cross_over == 1 .or. cross_over == glob%nivar) ) then
          ! no point crossing as the offspring will be identical to the parents
 
          ! Shift the cross-over point one element to the right until we find a 
          ! point at which the parents differ.
          ! At present, if no such point exists then we just continue with the 
          ! value of cross_over from after the do loop, regardless. IMPROVE !!!
          do m = 1, glob%nivar - 1
             cross_over = cross_over + 1
             if (cross_over > glob%nivar) cross_over = 1
             delta = pop_icoords(ip(1),cross_over)-pop_icoords(ip(2),cross_over)
             if (abs( delta/pop_icoords(ip(1),cross_over) ) > 1.0D3*small) exit
          end do
       end if

       ! Choose blending factor

       call random_number(blending)

       ! Make offspring
 
       m = glob%po_pop_size - noffspring ! to save characters

       pop_icoords(m + l     , cross_over) = pop_icoords(ip(1), cross_over) &
                                                           & - blending * delta
       pop_icoords(m + l + 1 , cross_over) = pop_icoords(ip(2), cross_over) &
                                                           & + blending * delta
       if (cross_over == glob%nivar) then
          pop_icoords(m + l     , 1 : glob%nivar - 1) = &
                                       & pop_icoords(ip(1), 1 : glob%nivar - 1)
          pop_icoords(m + l + 1 , 1 : glob%nivar - 1) = &
                                       & pop_icoords(ip(2), 1 : glob%nivar - 1)
       else
           if (cross_over == 1) then
              pop_icoords(m + l     , 2 : glob%nivar) = &
                                           & pop_icoords(ip(2), 2 : glob%nivar)
              pop_icoords(m + l + 1 , 2 : glob%nivar) = &
                                           & pop_icoords(ip(1), 2 : glob%nivar)
           else
              pop_icoords(m + l     , 1 : cross_over - 1) = &
                                       & pop_icoords(ip(1), 1 : cross_over - 1)
              pop_icoords(m + l + 1 , 1 : cross_over - 1) = &
                                       & pop_icoords(ip(2), 1 : cross_over - 1)
 
              pop_icoords(m + l     , cross_over + 1 : glob%nivar) = &
                              & pop_icoords(ip(2), cross_over + 1 : glob%nivar)
              pop_icoords(m + l + 1 , cross_over + 1 : glob%nivar) = &
                              & pop_icoords(ip(1), cross_over + 1 : glob%nivar)
           end if
       end if
    end do

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! end of offspring generation loop
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    call deallocate(w_cost)

end subroutine dlf_genetic_makepop
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_genetic_initialpop
!!
!! FUNCTION
!!
!! Create the "initial" population for the genetic algorithm, including 
!! the population after a reset has been requested, from a starting 
!! structure in icoords form.
!! Only executed on the rank-zero processor, as it depends on random 
!! numbers.
!! 
!! INPUTS
!!
!! princeps(:)
!! n
!! glob%nivar
!! glob%po_radius(:)
!! glob%iam
!!
!! OUTPUTS
!!
!! dum_pop(:,:)
!!
!! SYNOPSIS
subroutine dlf_genetic_initialpop(dum_pop,n,princeps)
!! SOURCE

! princeps is the "first citizen" in icoords form, from which the rest of the 
! population will be seeded.
! n is the population size (can be either the initial or working population)
! On return, dum_pop contains the newly created population in icoords form

  integer  :: n, l, m
  real(rk) :: r
  real(rk) :: princeps(:), dum_pop(:,:)

  if (glob%iam /= 0) return ! as only iam == 0 has seeded the random number 
                            ! generator.

  dum_pop(1,:) = princeps(:)
  
  do l = 2, n ! n is the population size
     do m = 1, glob%nivar
        call random_number(r)
        r = (r * 2.0D0 * glob%po_radius(m)) - glob%po_radius(m)
        dum_pop(l,m) = dum_pop(1,m) + r
     end do
  end do

end subroutine dlf_genetic_initialpop
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_genetic_mutate
!!
!! FUNCTION
!!
!! Mutate the population in the genetic algorithm by perturbing i-coordinates.
!! Only executed on the rank-zero processor, as it depends on random 
!! numbers.
!!
!! INPUTS
!!
!! glob%iam
!! nmutations
!! glob%po_pop_size
!! glob%nivar
!! glob%po_radius(:)
!! pop_icoords(:,:)
!!
!! OUTPUTS
!!
!! pop_icoords(:,:)
!!
!! SYNOPSIS
subroutine dlf_genetic_mutate
!! SOURCE

integer  :: l, indiv, element
real(rk) :: harvest, shift, factor

  if (glob%iam /= 0) return ! as only iam == 0 has seeded the random number 
                            ! generator.

  do l = 1, nmutations
     call random_number(harvest)
     indiv = int( harvest*(glob%po_pop_size - 1) ) + 2
     ! the "+ 2" means indiv is never one (i.e. the lowest-energy individual)
     if (indiv > glob%po_pop_size) indiv = glob%po_pop_size

     call random_number(harvest)
     element = int( harvest*glob%nivar ) + 1
     if (element > glob%nivar) element = glob%nivar

     call random_number(harvest)
     factor = (glob%po_radius(element)*glob%po_radius(element))/2.0D0
     shift = -factor * log( sqrt( 3.1415D0*factor )*harvest )
     !!! why this form?
     if (abs(shift) > glob%po_radius(element)) shift = glob%po_radius(element)

     call random_number(harvest)
     if (harvest < 0.5D0) shift = -shift
     pop_icoords(indiv, element) = pop_icoords(indiv, element) + shift
  end do

end subroutine dlf_genetic_mutate
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_stoch_endcycle
!!
!! FUNCTION
!!
!! Book-keeping at the end of a stochastic search cycle, including 
!! tests for failure and convergence and checkpointing.
!!
!! INPUTS
!!
!! pop_energies(:)
!! pop_icoords(:,:)
!! pop_xgradient(:,:,:)
!! glob%po_tolerance_g
!! glob%po_tolerance_r(:)
!! glob%iam
!! glob%po_radius(:)
!! glob%po_contraction
!! glob%dump
!! printl
!! printf
!!
!! OUTPUTS
!!
!! energy_best
!! icoords_best(:)
!! xcoords_best(:,:)
!! igradient_best(:)
!! tconv
!! texit
!!
!! files, if converged -- best.xyz
!! best_active.xyz
!!
!! SYNOPSIS
subroutine dlf_stoch_endcycle
!! SOURCE
 
integer :: l, k

    if ( minval(pop_energies(:)) < energy_best ) then ! store new "best" arrays
       energy_best = minval(pop_energies(:))
       position(1:1) = minloc(pop_energies(:))
       icoords_best(:) = pop_icoords(position(1),:)

       ! convert icoords_best(:) to Cartesians and store in xcoords_best(3,:)
       glob%icoords(:) = icoords_best(:)
       call clock_start("COORDS")
       call dlf_coords_itox(iimage)
       call clock_stop("COORDS")
       xcoords_best(:,:) = glob%xcoords(:,:)

       ! convert pop_xgradient(position(1), :, :) from Cartesians and store in
       ! igradient_best(:)
       glob%xgradient(:,:) = pop_xgradient(position(1), :, :)
       call clock_start("COORDS")
       call dlf_coords_xtoi(dummy_logical,dummy_logical,iimage,0)
       call clock_stop("COORDS")
       igradient_best(:) = glob%igradient(:)
    endif

    write(stdout,'(1x,a,i10)')"CYCLE ",i 
    ! i is the counter in the cycles do loop of the calling subroutine

! Make sure the value of glob%po_tolerance_g is appropriate for the coord system
    if ( maxval(abs(igradient_best(:))) < glob%po_tolerance_g ) then
       if (printl > 0) then
          write(stdout,'(1x,a)')"****Stochastic search converged****"
          write(stdout,'(1x,a,es16.9)')"Lowest energy = ",energy_best
          if (printl >= 4) then
             write(stdout,'(1x,a)')"for Cartesian coordinates "
             write(stdout,'(3(1x,es16.9))') xcoords_best(:,:)
          end if
          write(stdout,'(1x,a,es16.9)')"Max component of absolute force = ",&
                                       &maxval(abs(igradient_best(:)))
          write(stdout,'(1x,a,i10)')"Number of energy evaluations required = ",&
                                    &1 + glob%po_pop_size*i
       end if

       if (printf>=2 .and. glob%iam == 0) then
          call clock_start("XYZ")
          open(unit=400,file="best.xyz")
          open(unit=401,file="best_active.xyz")
          call write_xyz(400,glob%nat,glob%znuc,xcoords_best)
          call write_xyz_active(401,glob%nat,glob%znuc,glob%spec,xcoords_best)
          close(400)
          close(401)
          call dlf_put_coords(glob%nvar,1,energy_best,xcoords_best,glob%iam)
          call clock_stop("XYZ")
       end if
       tconv = .true.
       texit = .true.
       return
    end if

    glob%po_radius(:) = glob%po_contraction * glob%po_radius(:)

! Stop if any component of the vector of radii falls below its tolerance.
    do l = 1, glob%nivar
       if (glob%po_radius(l) < glob%po_tolerance_r(l)) then
          if (printl > 0) then
             write(stdout,'(1x,a)')"Search restricted to less than minimum radius"
             write(stdout,'(1x,a)')"Sample radius, tolerance:"
             do k = 1, glob%nivar
                write(stdout,'(2(1x,es16.9))') glob%po_radius(k),glob%po_tolerance_r(k) 
             end do
             write(stdout,'(1x,a,es16.9)')"Lowest energy = ",energy_best
             if (printl >= 4) then
                write(stdout,'(1x,a)')"for Cartesian coordinates "
                write(stdout,'(3(1x,es16.9))') xcoords_best(:,:)
             end if
             write(stdout,'(1x,a,es16.9)')"Max component of absolute force = ", &
                                          &maxval(abs(igradient_best(:)))
             write(stdout,'(1x,a,i10)')"Number of energy evaluations required = ",&
                                       &1 + glob%po_pop_size*i
             write(stdout,'(1x,a)')"Stopping"
          end if
          texit = .true.
          return
       end if
    end do

    if (glob%dump > 0 .and. mod(i,glob%dump)==0) then
       call clock_start("CHECKPOINT")
       if (printl>=6) write(stdout,"('Writing restart information')")
       status = 1
       call dlf_checkpoint_write(status)
       call dlf_checkpoint_po_write
       call clock_stop("CHECKPOINT")
    end if

    write(stdout,'(1x,a,es16.9)')"Lowest energy = ",energy_best
    if (printl >= 4) then
       write(stdout,'(1x,a)')"for Cartesian coordinates "
       write(stdout,'(3(1x,es16.9))') xcoords_best(:,:)
    end if
    write(stdout,'(1x,a,es16.9)')"Max component of absolute force = ", &
                                &maxval(abs(igradient_best(:)))
    write(stdout,'(1x,a,i10)')"Energy calls = ",1 + glob%po_pop_size*i

    call flush(stdout)

end subroutine dlf_stoch_endcycle
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_genetic_endcycle
!!
!! FUNCTION
!!
!! Book-keeping at the end of a genetic algorithm cycle, including
!! checkpointing and a test for convergence.
!!
!! INPUTS
!!
!! pop_energies(:)
!! pop_icoords(:,:)
!! pop_xgradient(:,:,:)
!! glob%po_tolerance_g
!! glob%iam
!! glob%po_nsave
!! glob%dump
!! printl
!! printf
!!
!! OUTPUTS
!!
!! energy_best
!! icoords_best(:)
!! xcoords_best(:,:)
!! igradient_best(:)
!! tconv
!!
!! if nsave/=0 -- energies_save(:)
!! nevals_save(:)
!! xcoords_save(:,:,:)
!!
!! local variables -- mean_e
!! sigma_e
!!
!! files, if converged -- best.xyz
!! best_active.xyz
!!
!! SYNOPSIS
subroutine dlf_genetic_endcycle
!! SOURCE

integer  :: m, l
real(rk) :: mean_e, mean_e2, sigma_e

    energy_best = pop_energies(1)
    icoords_best(:) = pop_icoords(1,:)

    ! convert icoords_best(:) to Cartesians and store in xcoords_best(3,:)
    glob%icoords(:) = icoords_best(:)
    call clock_start("COORDS")
    call dlf_coords_itox(iimage)
    call clock_stop("COORDS")
    xcoords_best(:,:) = glob%xcoords(:,:)

    ! convert pop_xgradient(1, :, :) from Cartesians and store in
    ! igradient_best(:) !!! assuming we know the gradient already....
    glob%xgradient(:,:) = pop_xgradient(1, :, :)
    call clock_start("COORDS")
    call dlf_coords_xtoi(dummy_logical,dummy_logical,iimage,0)
    call clock_stop("COORDS")
    igradient_best(:) = glob%igradient(:)

    mean_e = sum(pop_energies(:)) / dble(glob%po_pop_size)
    mean_e2 = sum(pop_energies(:)**2) / dble(glob%po_pop_size)
    sigma_e = sqrt(mean_e2 - mean_e**2)

! Write details for this generation
    ! i is the counter in the cycles do loop of the calling subroutine
    write(stdout,'(1x,a,i10)')"CYCLE ",i
    write(stdout,'(1x,a,es16.9)')"Lowest energy = ",pop_energies(1)
    if (printl >= 4) then
       write(stdout,'(1x,a)')"for Cartesian coordinates "
       write(stdout,'(3(1x,es16.9))') xcoords_best(:,:)
    end if
    write(stdout,'(1x,a,es16.9)')"Max component of absolute force = ",&
                                 &maxval(abs(igradient_best(:)))
    write(stdout,'(1x,a,es16.9)')"Mean energy for population = ",mean_e
    write(stdout,'(1x,a,es16.9)')"Standard deviation of the energy = ",sigma_e
    write(stdout,'(1x,a,i10)')"Energy calls = ", &
                   &glob%po_init_pop_size + (glob%po_pop_size - 1)*(i + nresets)

! Make sure the value of glob%po_tolerance_g is appropriate for the coord system
    if ( (maxval(abs(igradient_best(:))) < glob%po_tolerance_g) .AND. &
         (energy_best < HUGE(1.0D0)) ) then
    ! Added a catch for energy_best == huge above to avoid false convergence in the (unlikely!) case the coord transformation
    ! for the gradient vastly changes the values from when the Cartesian gradient vector was set to huge in dlf_get_engarrays
       if (printl > 0) then
          write(stdout,'(1x,a)')"****Genetic algorithm converged to a minimum****"
          write(stdout,'(1x,a,es16.9)')"Lowest energy = ",energy_best
          if (printl >= 4) then
             write(stdout,'(1x,a)')"for Cartesian coordinates "
             write(stdout,'(3(1x,es16.9))') xcoords_best(:,:)
          end if
          write(stdout,'(1x,a,es16.9)')"Max component of absolute force = ", &
                                       &maxval(abs(igradient_best(:)))
          write(stdout,'(1x,a,i10)')"Number of energy evaluations required = ",&
                  &glob%po_init_pop_size + (glob%po_pop_size - 1)*(i + nresets)
          write(stdout,'(1x,a)')"GA continuing... (minimum may be only local)"
       end if

       if (printf>=2 .and. glob%iam == 0) then
          call clock_start("XYZ")
          open(unit=400,file="best.xyz",position="APPEND")
          open(unit=401,file="best_active.xyz",position="APPEND")
          call write_xyz(400,glob%nat,glob%znuc,xcoords_best)
          call write_xyz_active(401,glob%nat,glob%znuc,glob%spec,xcoords_best)
          close(400)
          close(401)
          call clock_stop("XYZ")
       end if

       if (glob%po_nsave > 0) then
          if (energy_best < energies_save(glob%po_nsave)) then
             do m = 1, glob%po_nsave
                !!!crudely check for multiple incidences of a particular minimum
                if ( abs( ( energy_best - energies_save(m) )/energy_best ) &
                   & < small ) exit
                if (energy_best < energies_save(m)) then
                   do l = glob%po_nsave-1, m, -1
                      energies_save(l+1) = energies_save(l)
                      xcoords_save(l+1,:,:) = xcoords_save(l,:,:)
                      nevals_save(l+1) = nevals_save(l)
                   end do
                   energies_save(m) = energy_best
                   xcoords_save(m,:,:) = xcoords_best(:,:)
                   nevals_save(m) = glob%po_init_pop_size + &
                                    &(glob%po_pop_size - 1)*(i + nresets)
                   exit
                end if
             end do
          end if
       end if 

       tconv = .true.
    end if

    if (glob%dump > 0 .and. mod(i,glob%dump)==0) then
       call clock_start("CHECKPOINT")
       if (printl>=6) write(stdout,"('Writing restart information')")
       status = 1
       call dlf_checkpoint_write(status)
       call dlf_checkpoint_po_write
       call clock_stop("CHECKPOINT")
    end if

    call flush(stdout)

end subroutine dlf_genetic_endcycle
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_po_destroy
!!
!! FUNCTION
!!
!! Deallocate population arrays
!!
!! SYNOPSIS
subroutine dlf_po_destroy
!! SOURCE

    call deallocate(pop_icoords)
    call deallocate(pop_xgradient)
    call deallocate(pop_energies)

    call deallocate(icoords_best)
    call deallocate(igradient_best)
    call deallocate(xcoords_best)

    if (genetic) then

       call deallocate(init_pop_icoords)
       call deallocate(init_pop_energies)

    end if

    call deallocate(glob%po_radius)
    call deallocate(glob%po_tolerance_r)

end subroutine dlf_po_destroy
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_checkpoint_po_read
!!
!! FUNCTION
!!
!! Reading of the parallel optimisation checkpoint file
!!
!! OUTPUTS
!!
!! trestarted
!!
!! SS -- energy_best
!! icoords_best(:)
!! xcoords_best(:,:)
!! igradient_best(:)
!!
!! GA -- pop_icoords(:,:)
!! pop_energies(:)
!! pop_xgradient(:,:,:)
!!
!! SYNOPSIS
subroutine dlf_checkpoint_po_read
!! SOURCE

logical               :: tchk, tallocated
real(rk), allocatable :: dummy(:,:,:)

  trestarted=.false.
    
  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_parallel_opt.chk",EXIST=tchk)
  if (.not.tchk) then
    write(stdout,10) "File dlf_parallel_opt.chk not found"
    return
  end if

  ! open the checkpoint file
  if (tchkform) then
    open(unit=100,file="dlf_parallel_opt.chk",form="formatted")
  else
    open(unit=100,file="dlf_parallel_opt.chk",form="unformatted")
  end if

  if (stochastic) then

    call read_separator(100,"Best arrays",tchk)
    if (.not. tchk) return
    if (tchkform) then
       read(100,*,end=201,err=200) energy_best, icoords_best(:), &
                                  &xcoords_best(:,:), igradient_best(:)
    else
      read(100,end=201,err=200) energy_best, icoords_best(:), &
                               &xcoords_best(:,:), igradient_best(:)
    end if

  else if (genetic) then

    call read_separator(100,"Population",tchk)
    if (.not. tchk) return
    if (tchkform) then
       read(100,*,end=201,err=200) pop_icoords(:,:), pop_energies(:)
       read(100,*,end=201,err=200) tallocated
       if (tallocated .and. allocated(pop_xgradient)) then
          read(100,*,end=201,err=200) pop_xgradient(:,:,:)
       else if (tallocated) then
          call allocate(dummy, glob%po_pop_size, 3, glob%nat)
          read(100,*,end=201,err=200) dummy(:,:,:)
          call deallocate(dummy)
       end if
    else
       read(100,end=201,err=200) pop_icoords(:,:), pop_energies(:)
       read(100,end=201,err=200) tallocated
       if (tallocated .and. allocated(pop_xgradient)) then
          read(100,end=201,err=200) pop_xgradient(:,:,:)
       else if (tallocated) then
          call allocate(dummy, glob%po_pop_size, 3, glob%nat)
          read(100,end=201,err=200) dummy(:,:,:)
          call deallocate(dummy)
       end if
    end if

  end if

  call read_separator(100,"END",tchk)
  if (.not.tchk) return

! successful reading
  close(100)
  trestarted=.true.
  return

  ! return on error
200 continue
  write(stdout,10) "Error reading parallel optimisation checkpoint file"
  return
201 continue
  write(stdout,10) "Error (EOF) reading parallel optimisation checkpoint file"
  return

10 format("Checkpoint reading WARNING: ",a)

end subroutine dlf_checkpoint_po_read
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* parallel_opt/dlf_checkpoint_po_write
!!
!! FUNCTION
!!
!! Writing of the parallel optimisation checkpoint file
!!
!! INPUTS
!!
!! trestarted
!!
!! SS -- energy_best
!! icoords_best(:)
!! xcoords_best(:,:)
!! igradient_best(:)
!!
!! GA -- pop_icoords(:,:)
!! pop_energies(:)
!! pop_xgradient(:,:,:)
!!
!! SYNOPSIS
subroutine dlf_checkpoint_po_write
!! SOURCE

! Only want one processor to do the writing; that processor must know         
! all the necessary information.
  if (glob%iam /= 0) return 

! Open the checkpoint file
  if (tchkform) then
    open(unit=100,file="dlf_parallel_opt.chk",form="formatted")
  else
    open(unit=100,file="dlf_parallel_opt.chk",form="unformatted")
  end if

  if (stochastic) then

    call write_separator(100,"Best arrays")
    if (tchkform) then
       write(100,*) energy_best, icoords_best(:), &
                   &xcoords_best(:,:), igradient_best(:)
    else
      write(100) energy_best, icoords_best(:), &
                &xcoords_best(:,:), igradient_best(:)
    end if

  else if (genetic) then

    call write_separator(100,"Population")
    if (tchkform) then
       write(100,*) pop_icoords(:,:), pop_energies(:)
       write(100,*) allocated(pop_xgradient)
       if (allocated(pop_xgradient)) write(100,*) pop_xgradient(:,:,:)
    else
       write(100) pop_icoords(:,:), pop_energies(:)
       write(100) allocated(pop_xgradient)
       if (allocated(pop_xgradient)) write(100) pop_xgradient(:,:,:)
    end if

  end if

  call write_separator(100,"END")
  close(100)

end subroutine dlf_checkpoint_po_write
!!****


end subroutine dlf_parallel_opt
!!****
