! **********************************************************************
! **               Line search or trust radius unit                   **
! **********************************************************************
!!****h* DL-FIND/scalestep
!!
!! NAME
!! scalestep
!!
!! FUNCTION
!! Line search or trust radius
!!
!! DATA
!! $Date$
!! $Rev$
!! $Author$
!! $URL$
!! $Id$
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk)
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
!!****
module dlf_scalestep_module
  use dlf_parameter_module, only: rk

  !types
  type trustradius_type
    real(rk) :: startrad
    real(rk) :: maxrad ! maximum trust radius, from glob%maxstep
    real(rk) :: radius
    real(rk) :: minrad
  end type trustradius_type

  ! variables
  ! tr - global trust radius
  ! trmic - microiterative trust radius
  type(trustradius_type),save :: tr, trmic

end module dlf_scalestep_module

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
module dlf_linesearch
  use dlf_parameter_module, only: rk
  implicit none
  ! module only to be used locally in scalestep.f90
  ! used for storage of line search data
  real(rk),allocatable,save :: oldgradient(:) ! glob%nivar
end module dlf_linesearch

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* scalestep/dlf_scalestep
!!
!! FUNCTION
!!
!! Do line search or a scaling of the step based on some trust radius.
!! the parameter glob%iline defines which algorithm to use
!!
!! SYNOPSIS
subroutine dlf_scalestep
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  use dlf_linesearch, only: oldgradient
  use dlf_scalestep_module, only: tr
  implicit none
  !
  real(rk)  :: svar
  integer   :: ivar(1)
  integer   :: geomCoords
! **********************************************************************
  if (glob%icoord >= 10 .and. glob%icoord <= 19) then
     ! In the case of Lagrange-Newton coordinates, the change in the two
     ! extra coordinates (Lagrange multipliers) can be significantly
     ! larger than that of the geometrical coordinates.
     ! These coordinates should therefore not be included in the step
     ! scaling calculations, as otherwise the optimisation will grind to
     ! a halt.
     geomCoords = glob%nivar - 2
  else
     geomCoords = glob%nivar 
  endif

  select case (glob%iline)

! ======================================================================
! Simple scaling
! ======================================================================
  case (0)

    if(glob%icoord<200 .or. glob%icoord>=300) then
      glob%step(:)= glob%scalestep * glob%step(:)
      svar=maxval(abs(glob%step(1:geomCoords)))
      
      if(svar > glob%maxstep) then
        svar= glob%maxstep/svar
        if(printl>=4) write(stdout,'("Scaling step back by ",f10.5)') svar
        glob%step(:)= glob%step(:) * svar
      end if
    else
      ! Dimer: scale only first half of step (dimer midpoint)
      ivar=shape(glob%step)
      ivar=ivar/2
      glob%step(1:ivar(1))= glob%scalestep * glob%step(1:ivar(1))
      svar=maxval(abs(glob%step(1:ivar(1))))
      if(svar > glob%maxstep) then
        svar= glob%maxstep/svar
        glob%step(1:ivar(1))= glob%step(1:ivar(1)) * svar
      end if
    end if

! ======================================================================
! Trust radius approaches
! ======================================================================
  case(1,2)

    svar=dsqrt(sum(glob%step(1:geomCoords)**2))
    if(printl >= 2) then
      write(*,"(' Predicted step length ',es10.4)") svar
      write(*,"(' Trust radius          ',es10.4)") tr%radius
    end if
    if(svar > tr%radius) then
      svar= tr%radius/svar
      glob%step(:)= glob%step(:) * svar
    end if
    oldgradient(:)= glob%igradient(:)

! ======================================================================
! Line search
! ======================================================================
  case(3)

    ! the step may be scaled to whatever is suitable
    svar=dsqrt(sum(glob%step(1:geomCoords)**2))
    svar= tr%startrad/svar
    glob%step(:)= glob%step(:) * svar


! ======================================================================
! Wrong optimisation type setting
! ======================================================================
  case default
    write(stderr,*) "Line search ",glob%iline,"not implemented"
    call dlf_fail("Line search error error")

  end select
end subroutine dlf_scalestep
!!****



! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* scalestep/dlf_scalestep_microiter
!!
!! FUNCTION
!!
!! Do a scaling of the microiterative step based on some trust radius.
!!
!! Trust radius only at the moment
!!
!! SYNOPSIS
subroutine dlf_scalestep_microiter(nvar, step)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,printl,stdout
  use dlf_scalestep_module, only: trmic
  implicit none
  !
  integer, intent(in) :: nvar
  real(rk), intent(inout) :: step(nvar)
  !
  real(rk)  :: svar
  integer   :: ivar(1)
! **********************************************************************

  svar=dsqrt(sum(step(1:nvar)**2.0d0))
  if(printl >= 2) then
     write(*,"(' Predicted step length ',es10.4)") svar
     write(*,"(' Trust radius          ',es10.4)") trmic%radius
  end if
  if(svar > trmic%radius) then
     svar= trmic%radius/svar
     step(:)= step(:) * svar
  end if

end subroutine dlf_scalestep_microiter
!!****



! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* scalestep/test_acceptance
!!
!! FUNCTION
!!
!! Scale the step based on the energy as acceptance criterion.
!! Used if glob%iline == 1.
!!
!! SYNOPSIS
subroutine test_acceptance
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_scalestep_module, only: tr
  use dlf_linesearch, only: oldgradient
  implicit none
  !
  real(rk)  :: svar,grad1,grad2
! **********************************************************************
  if (glob%iline.ne.1) &
    call dlf_fail("warning, test_acceptance should only be called for iline=1")

  if(.not.glob%toldenergy) then
    ! first step
    tr%radius=tr%startrad
    if(printl >=6) write(stdout,"(' Accepting step as it is the first')")
    glob%taccepted=.true.
    return
  end if

  if(glob%energy < glob%oldenergy) then
    ! accept step
    if(printl >=6) write(stdout,"(' Accepting step...')")
    if(printl >=6) write(stdout,"(' Energy ',es10.3,&
        &' old Energy ',es10.3)") glob%energy, glob%oldenergy

    ! include Wolfe conditions ... (two parameters are free to chose)
    ! oldgradient is set in dlf_scalestep
    grad1=dot_product(glob%step(:),oldgradient(:))
    if(glob%energy < glob%oldenergy + 1.D-4 * grad1) then
      !this was the Armijo condition
      grad2=dot_product(glob%step(:),glob%igradient(:))
      if(printl >=6) write(stdout,*) &
            "Armijo condition fulfilled"
      !if(grad2 >= grad1*0.6D0) then ! this is taken from the web. I think it is wrong
      if(abs(grad2) <= abs(grad1)*0.99D0) then
      !this was the curvature condition
        if(printl >=4) write(stdout,*) &
            "Wolfe conditions fulfilled, increasing trust radius"
        tr%radius= min(tr%radius * 2.D0,tr%maxrad)
      end if
    end if

    glob%taccepted=.true.

  else
    ! reject step

    if(printl >=2) write(stdout,"(' Rejecting step, energy ',es10.3,&
        &' higher')") glob%energy-glob%oldenergy
    if(printl >=6) write(stdout,"(' Energy ',es10.3,&
        &' old Energy ',es10.3)") glob%energy, glob%oldenergy

    svar=dsqrt(dot_product(glob%step(:),glob%step(:)))
    tr%radius= min(svar,tr%radius) * 0.5D0
    if(printl >=4) write(*,"(' Trust radius          ',es10.4)") tr%radius

    if (glob%imicroiter == 1) then
       call dlf_microiter_reset_macrostep
    else
       glob%step(:)= glob%step(:) * 0.5D0
       ! set back coordinates
       glob%icoords(:)= glob%icoords(:) - glob%step(:)
    end if

    glob%taccepted=.false.

    if(tr%radius < tr%minrad) then
      if(printl >=2) write(stdout,&
          "(' Step too small, restarting optimiser.')")
      call dlf_formstep_restart
      tr%radius=tr%maxrad
      glob%oldenergy=glob%energy
      glob%taccepted=.true.

      ! Removed reset of HDLCs. Doesn't work because step/itox
      ! called next before xtoi so itox will use icoords that are
      ! inconsistent with the new HDLC system.
      !
      !! reset hdlc coordinates if used
      !if(glob%icoord>=1 .and. glob%icoord<=4) then
      !  call dlf_hdlc_reset
      !  ! the arrays glob%spec,glob%znuc have to be changed when using 
      !  !  more instances of hdlc
      !  call dlf_hdlc_create(glob%nvar/3,glob%nicore,glob%spec,glob%micspec,&
      !      glob%znuc,1,glob%xcoords,glob%weight,glob%mass)
      !  ! recalculate iweight
      !  call dlf_hdlc_getweight(glob%nat,glob%nivar,glob%nicore,glob%micspec,&
      !      glob%weight,glob%iweight)
      !end if


    end if

  end if

  if(printl >=6) write(*,"(' Trust radius ',es10.3)") &
      tr%radius

end subroutine test_acceptance
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* scalestep/test_acceptance_microiter
!!
!! FUNCTION
!!
!! Scale the microiterative step based on the energy as acceptance criterion.
!!
!! TODO: Refactor - merge with standard routine?
!!
!! SYNOPSIS
subroutine test_acceptance_microiter(nvar, energy, igradient, &
     oldgradient, tmicoldenergy, oldenergy, step, icoords, taccepted, tswitch)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_scalestep_module, only: trmic
  implicit none
  !
  integer, intent(in) :: nvar
  real(rk), intent(in) :: energy
  real(rk), intent(in) :: igradient(nvar), oldgradient(nvar)
  logical, intent(in) :: tmicoldenergy
  real(rk), intent(inout) :: oldenergy
  real(rk), intent(inout) :: step(nvar), icoords(nvar)
  logical, intent(out) :: taccepted
  logical, intent(out) :: tswitch
  real(rk)  :: svar,grad1,grad2
! **********************************************************************
  tswitch = .false.

  if(.not. tmicoldenergy) then
    ! first step in this microiterative cycle
    if(printl >=6) write(stdout,"(' Accepting step as it is first in microiterative cycle')")
    taccepted=.true.
    return
  end if

  if(energy < oldenergy) then
    ! accept step
    if(printl >=6) write(stdout,"(' Accepting step...')")
    if(printl >=6) write(stdout,"(' Energy ',es10.3,&
        &' old Energy ',es10.3)") energy, oldenergy

    ! include Wolfe conditions ... (two parameters are free to chose)
    ! oldgradient is set in dlf_scalestep_microiter
    grad1=dot_product(step(:),oldgradient(:))
    if(energy < oldenergy + 1.D-4 * grad1) then
      !this was the Armijo condition
      grad2=dot_product(step(:),igradient(:))
      if(printl >=6) write(stdout,*) &
            "Armijo condition fulfilled"
      !if(grad2 >= grad1*0.6D0) then ! this is taken from the web. I think it is wrong
      if(abs(grad2) <= abs(grad1)*0.99D0) then
      !this was the curvature condition
        if(printl >=4) write(stdout,*) &
            "Wolfe conditions fulfilled, increasing trust radius"
        trmic%radius= min(trmic%radius * 2.D0,trmic%maxrad)
      end if
    end if

    taccepted=.true.

  else
    ! reject step

    if(printl >=2) write(stdout,"(' Rejecting step, energy ',es10.3,&
        &' higher')") energy - oldenergy
    if(printl >=6) write(stdout,"(' Energy ',es10.3,&
        &' old Energy ',es10.3)") energy, oldenergy

    svar=dsqrt(dot_product(step(:), step(:)))
    trmic%radius= min(svar,trmic%radius) * 0.5D0
    if(printl >=4) write(*,"(' Trust radius          ',es10.4)") trmic%radius


    step(:) = step(:) * 0.5D0

    ! set back coordinates
    icoords(:) = icoords(:) - step(:)

    taccepted=.false.

    if(trmic%radius < trmic%minrad) then
      if(printl >=2) write(stdout,&
          "(' Step too small, switching to macroiterations.')")
      trmic%radius = min(1.D-1,glob%maxstep)
      oldenergy = energy
      taccepted=.true.
      tswitch = .true.

      ! Probably shouldn't reset HDLC coordinates as we aren't
      ! restarting the optimiser
      !if(glob%icoord>=1 .and. glob%icoord<=4) then
        !if(printl >=2) write(stdout,"(' Resetting HDLC coordinates.')") 
        ! call dlf_hdlc_reset
        !! the arrays glob%spec,glob%znuc have to be changed when using 
        !!  more instances of hdlc
        !call dlf_hdlc_create(glob%nvar/3,glob%nicore,glob%spec,glob%micspec,glob%znuc,1, &
        !    glob%xcoords,glob%weight,glob%mass)
        !! recalculate iweight
        !call dlf_hdlc_getweight(glob%nat,glob%nivar,glob%nicore,glob%micspec,glob%weight,glob%iweight)
      !end if

    end if

  end if

  if(printl >=6) write(*,"(' Trust radius ',es10.3)") &
      trmic%radius

end subroutine test_acceptance_microiter
!!****


! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* scalestep/test_acceptance_g
!!
!! FUNCTION
!!
!! Scale the step based on the gradient projection on the step as 
!! acceptance criterion.
!! Used if glob%iline == 2.
!!
!! SYNOPSIS
subroutine test_acceptance_g
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_scalestep_module, only: tr
  use dlf_linesearch, only: oldgradient
  implicit none
  !
  real(rk)  :: svar,proj
  real(rk)  :: oldproj,astep
! **********************************************************************
  if (glob%iline.ne.2) call &
      dlf_fail("test_acceptance_g should only be called for iline=2")

  if(.not.glob%toldenergy) then
    ! first step
    tr%radius=tr%startrad
    if(printl >=6) write(stdout,"(' Accepting step as it is the first')")
    oldgradient(:)=glob%igradient(:)
    return
  end if

  ! calculate projection of the new gradient on the old search direction
  proj=-dot_product(glob%igradient(:),glob%step(:))
  oldproj=-dot_product(oldgradient(:),glob%step(:))
  ! scaling
  svar=oldproj/(oldproj-proj)

  if(proj > 0.D0 .or. svar>0.9D0 ) then 
    ! accept step
    if(printl >=6) write(stdout,"(' Accepting step...')")
    if(printl >=6) write(stdout,"(' Projection of gradient on step: ',es10.3,&
        &' Energy ',es10.3)") proj,glob%energy

    ! We have nothing like Wolfe conditions here, thus we scale the 
    ! trust radius based on the gradient projections (to a minimum of 
    ! 0.9 and a maximum of 2)
    if(svar>0.9D0) tr%radius= tr%radius * min(svar,2.D0)

    glob%taccepted=.true.

    oldgradient(:)=glob%igradient(:)

  else
    ! reject step

    if(printl >=2) write(stdout,"(' Rejecting step...')")
    if(printl >=6) write(stdout,"(' Projection of gradient on step: ',es10.3,&
        &' oldproj ',es10.3)") proj,oldproj

    if(abs(oldproj-proj)>1.D-10) then
      svar=oldproj/(oldproj-proj) * 0.9D0 ! this factor is empirical
    else
      ! this may be the case if the step is very small
      svar=1.D-1
    end if
    if(svar<=-0.8D0) svar=-0.8D0 ! allow negative scaling
    if(printl >=2) write(stdout,"('scaling:',es10.3)") svar

    ! TODO Implement this?
    if (glob%imicroiter == 1) then
       call dlf_fail("Microiterative opt with test_acceptance_g not yet implemented")
    end if

    ! set back coordinates
    glob%icoords(:)= glob%icoords(:) - glob%step(:) * (1.D0-svar)

    glob%step(:)= glob%step(:) * svar

    astep = dsqrt(dot_product(glob%step(:),glob%step(:)))
    tr%radius= min( astep * 1.5D0 , tr%radius * svar )
    
    glob%taccepted=.false.

    if(tr%radius < tr%minrad) then ! < this does not seem too good
      if(printl >=2) write(stdout,&
          "(' Step too small, restarting optimiser.')")
      call dlf_formstep_restart
      tr%radius=tr%maxrad
      oldgradient(:)=glob%igradient(:)
      glob%taccepted=.true.
    end if

  end if
  tr%radius= min(tr%radius,tr%maxrad)

  if(printl >=6) write(*,"(' Trust radius ',es10.3)") &
      tr%radius

end subroutine test_acceptance_g
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine linesearch_init
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_scalestep_module, only: tr, trmic
  use dlf_linesearch, only: oldgradient
  use dlf_allocate, only: allocate
  implicit none
! **********************************************************************
  ! Microiterative trust radius init
  ! TODO: Is there a more logical place for it to go?
  ! Note moved before iline test in case main trust radius 
  ! is constant.
  ! TODO: Should micro version be linked to iline?
  if (glob%imicroiter > 0) then
     trmic%startrad=min(1.D-1,glob%maxstep)
     trmic%maxrad=glob%maxstep
     trmic%minrad=1.D-7
     ! Initialise starting radius here - only once
     ! Following HDLCOpt, the trust radius from the last 
     ! microiterative cycle is retained for the next
     trmic%radius = trmic%startrad
  end if

  if (glob%iline/=1.and.glob%iline/=2.and.glob%iline/=3) return

  tr%startrad=min(1.D-1,glob%maxstep)
  tr%maxrad=glob%maxstep
  tr%minrad=1.D-7
  call allocate(oldgradient,glob%nivar)

end subroutine linesearch_init

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine linesearch_destroy
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob
  use dlf_linesearch, only: oldgradient
  use dlf_allocate, only: deallocate
  implicit none
  integer :: fail
! **********************************************************************
  if (glob%iline/=1.and.glob%iline/=2.and.glob%iline/=3) return
  if (allocated(oldgradient)) call deallocate(oldgradient)
end subroutine linesearch_destroy

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine linesearch
  ! do line search, glob%accepted is false as long as we are doing line 
  ! search and true if a new direction should be calculated
  ! THIS ROUTINE DOES NOT WORK WELL AT THE MOMENT
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stderr,stdout,printl
  use dlf_scalestep_module, only: tr
  use dlf_linesearch, only: oldgradient
  implicit none
  !
  real(rk)  :: svar,proj,newalpha,oldproj
  real(rk),save :: oldalpha,alpha
! **********************************************************************
  if (glob%imicroiter == 1) then
     call dlf_fail("Microiterative opt with linesearch not yet implemented")
  end if

  if (glob%iline.ne.3) &
    call dlf_fail("linesearch should only be called for iline=3")

  if(.not.glob%toldenergy) then
    ! ==================================================================
    ! first step
    ! ==================================================================

    tr%radius=tr%startrad
    if(printl >=6) write(stdout,"(' Accepting step as it is the first')")
    alpha=1.D0!tr%radius
    oldalpha=0.D0
    oldgradient=glob%igradient(:)
    return
  end if

  ! calculate projection of the new gradient on the old search direction
  proj=-dot_product(glob%igradient(:),glob%step(:))
  oldproj=-dot_product(oldgradient(:),glob%step(:))

  if(abs(proj) < 1.D-8 ) then ! hardcoded line search convergence criterion
    ! ==================================================================
    ! accept step
    ! ==================================================================
    if(printl >=6) write(stdout,"(' Line search finished')")
    if(printl >=6) write(stdout,"(' Projection of gradient on step: ',es10.3,&
        &' Energy ',es10.3)") proj,glob%energy

    glob%step(:)=glob%step(:)*alpha

    glob%taccepted=.true.

! debug print
print*,"qq Current accepted"
    write(*,'("qq Position ",2es15.5)') glob%icoords(:)
    write(*,'("qq Gradient ",2es15.5)') glob%igradient(:)
    write(*,'("qq Step     ",2es15.5)') glob%step(:)

    ! start values for the following line search
    alpha=1.D0
    oldalpha=0.D0

  else
    ! ==================================================================
    ! Start or continue line search
    ! ==================================================================

    if(printl >=6) write(*,"(' Projection of gradient on step: ',es10.3,&
        &' oldproj ',es10.3)") proj,oldproj

 write(*,"('qq Projection of gradient on step: ',es10.3,&
          &' energy ',es10.3)") proj,glob%energy

    write(*,'("qq Position ",2es15.5)') glob%icoords(:)
    write(*,'("qq Gradient ",2es15.5)') glob%igradient(:)
    write(*,'("qq Step     ",2es15.5)') glob%step(:)

    if(printl >=2) write(stdout,"(' Line search continuing')")
    if(printl >=6) write(stdout,"(' Projection of gradient on step: ',es10.3,&
        &' oldproj ',es10.3)") proj,oldproj

 write(*,"('qq Projection of gradient on step: ',es10.3,&
          &' energy ',es10.3)") proj,glob%energy

      svar=oldproj/(oldproj-proj) 
!!$    ! do not change the step size too drastically
!!$    svar=min(svar,2.D0)
!!$    svar=max(svar,0.5D0)
      newalpha=oldalpha+svar*(alpha-oldalpha)
      if(newalpha/alpha < 0.D0 ) then
        ! we predict a line minimum in the inverse of the search direction
        if(proj>0.d0) then
          newalpha=1.4D0*alpha
        end if
      else
        if(newalpha/alpha > 1.2D0 ) newalpha=1.2D0*alpha
        if(newalpha/alpha < 0.6D0 ) newalpha=0.6D0*alpha
      end if

      write(*,'("newal, alpha, frac",3f10.5)') newalpha,alpha,newalpha/alpha

      ! set back coordinates
      glob%icoords(:)= glob%icoords(:) + glob%step(:) * (newalpha-alpha)

      ! store alphas for next step
      oldalpha=alpha
      alpha=newalpha

    glob%taccepted=.false.

    write(*,'("qq Position ",2es15.5)') glob%icoords(:)

  end if

  oldgradient=glob%igradient(:)

  if(printl >=6) write(*,"(' Trust radius ',es10.3)") &
      tr%radius
  if(printl >=6) write(*,"(' Current scaling ',2es10.3)") &
      alpha,proj
  
end subroutine linesearch

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_linesearch_write
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob

  use dlf_linesearch, only: oldgradient
  use dlf_scalestep_module, only: tr

  use dlf_checkpoint, only: tchkform,write_separator
  implicit none
  integer :: fail
! **********************************************************************
  if (glob%iline/=1.and.glob%iline/=2.and.glob%iline/=3) return
  if(tchkform) then
    open(unit=100,file="dlf_linesearch.chk",form="formatted")
    call write_separator(100,"Linesearch-Arrays")
    write(100,*) oldgradient,tr
    call write_separator(100,"END")
    close(100)
  else
    open(unit=100,file="dlf_linesearch.chk",form="unformatted")
    call write_separator(100,"Linesearch-Arrays")
    write(100) oldgradient,tr
    call write_separator(100,"END")
    close(100)
  end if
end subroutine dlf_checkpoint_linesearch_write

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine dlf_checkpoint_linesearch_read(tok)
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout

  use dlf_linesearch, only: oldgradient
  use dlf_scalestep_module, only: tr

  use dlf_checkpoint, only: tchkform,read_separator
  implicit none
  logical,intent(out) :: tok
  logical             :: tchk
! **********************************************************************
  tok=.true.
  if (glob%iline/=1.and.glob%iline/=2.and.glob%iline/=3) return
  tok=.false.

  ! check if checkpoint file exists
  INQUIRE(FILE="dlf_linesearch.chk",EXIST=tchk)
  if(.not.tchk) then
    write(stdout,10) "File dlf_linesearch.chk not found"
    return
  end if

  if(tchkform) then
    open(unit=100,file="dlf_linesearch.chk",form="formatted")
  else
    open(unit=100,file="dlf_linesearch.chk",form="unformatted")
  end if

  call read_separator(100,"Linesearch-Arrays",tchk)
  if(.not.tchk) return 

  if(tchkform) then
    read(100,*,end=201,err=200) oldgradient,tr
  else
    read(100,end=201,err=200) oldgradient,tr
  end if
  call read_separator(100,"END",tchk)
  if(.not.tchk) return 

  close(100)
  tok=.true.
  return

  ! return on error
200 continue
  close(100)
  write(stdout,10) "Error reading CG checkpoint file"
  return
201 continue
  close(100)
  write(stdout,10) "Error (EOF) reading CG checkpoint file"
  return
10 format("Checkpoint reading WARNING: ",a)
end subroutine dlf_checkpoint_linesearch_read


