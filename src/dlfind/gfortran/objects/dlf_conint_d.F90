! **********************************************************************
! **                                                                  **
! **            Conical intersection optimisation routines            **
! **                                                                  **
! **                          Tom Keal, 2007                          **
! **                                                                  **
! **********************************************************************
!!****h* DL-FIND/conint
!!
!! NAME
!! conint
!!
!! FUNCTION
!! Routines for conical intersection optimisations
!!
!! COMMENTS
!! Three optimisation algorithms are implemented: penalty function (pf),
!! gradient projection (gp), and Lagrange-Newton (ln).
!!
!! A detailed comparison of these algorithms can be found in:
!!
!! Thomas W. Keal, Axel Koslowski, Walter Thiel, "Comparison of 
!! algorithms for conical intersection optimisation using semiempirical
!! methods", Theo. Chem. Acc. (2007). doi: 10.1007/s00214-007-0331-5
!!
!! The nomenclature used here is the same as in the above paper.
!! Equation numbers refer to this paper.
!!
!! DATA
!!  $Date$
!!  $Rev$
!!  $Author$
!!  $URL$
!!  $Id$
!!
!! COPYRIGHT
!!
!!  Copyright 2007 Tom Keal (thomas.keal@stfc.ac.uk)
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
module dlf_conint
  use dlf_parameter_module, only: rk
  type conint_type
     ! Gradients for the Lagrange-Newton method
     ! Gradient average (g_i + g_j) / 2
     real(rk), allocatable :: xGradMean(:,:)
     real(rk), allocatable :: iGradMean(:)
     real(rk), allocatable :: iGradMeanOld(:)
     ! Gradient difference in cartesians and internals
     real(rk), allocatable :: xGradDiff(:,:)
     real(rk), allocatable :: iGradDiff(:)
     real(rk), allocatable :: iGradDiffOld(:)
     ! Gradient of the interstate coupling in internals
     real(rk), allocatable :: iCoupling(:) 
     real(rk), allocatable :: iCouplingOld(:)
     ! Flag for activation of extrapolatable functions
     logical :: extrap
     ! Flag to indicate if old gradients were orthogonalised
     logical :: oldIsOrthog
  end type conint_type
  type(conint_type), save :: conint
end module dlf_conint
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_conint_check_consistency
!!
!! FUNCTION
!!
!! Check that no incompatible sets of input options have been specified
!! for a multistate calculation.
!!
!! The allowed options vary significantly between multistate methods.
!!
!! For the penalty function (imultistate=1) and gradient projection 
!! methods (imultistate=2), the optimisation algorithm and coordinate
!! system is essentially independent of the method. However, there is 
!! no well-defined objective function for the gradient projection 
!! method, so no routines that assume that an objective function exists
!! can be used (e.g. an energy-based trust radius).
!!
!! For the Lagrange-Newton method (imultistate=3), the Lagrange-Newton
!! optimiser must be used (iopt=40) with a special coordinate system
!! (icoord=10:19).
!!
!! INPUTS
!!
!! glob%imultistate
!! glob%icoord
!! glob%iopt
!! glob%iline 
!! glob%inithessian
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_conint_check_consistency
!! SOURCE
  use dlf_global, only: glob, stderr

  select case (glob%imultistate)
  case (1)
     ! Penalty function checks
     if (glob%icoord > 9) call dlf_fail("A standard coordinate system &
          &must be used with the penalty function method")
     if (glob%iopt == 30 .or. glob%iopt == 40) call dlf_fail("The &
          &selected optimiser is not compatible with the penalty & 
          &function method")
     if (glob%nzero /= 0) call dlf_fail("Soft mode skipping with &
          &nzero is not compatible with the penalty function method")
     if (glob%imicroiter > 0) call dlf_fail("Microiterative &
          &conical intersection search is not yet implemented")
  case (2)
     ! Gradient projection checks
     if (glob%icoord > 9) call dlf_fail("A standard coordinate system &
          &must be used with the gradient projection method")
     if (glob%iopt == 30 .or. glob%iopt == 40) call dlf_fail("The &
          &selected optimiser is not compatible with the gradient &
          &projection method")
     if (glob%iline == 1) call dlf_fail("Energy-based trust radius &
          &not possible with the gradient projection method")
     if (glob%nzero /= 0) call dlf_fail("Soft mode skipping with &
          &nzero is not compatible with the gradient projection method")
     if (glob%imicroiter > 0) call dlf_fail("Microiterative &
          &conical intersection search is not yet implemented")
  case (3)
     ! Lagrange-Newton checks
     if (glob%iopt /= 40) call dlf_fail("The Lagrange-Newton optimiser &
          &must be used with the Lagrange-Newton method")
     if (glob%icoord < 10 .or. glob%icoord > 19) call dlf_fail("A &
          &Lagrange-Newton coordinate system must be used with the &
          &Lagrange-Newton method")
     if (glob%inithessian == 0) call dlf_fail("The Hessian for the &
          &Lagrange-Newton method cannot be calculated externally")
     if (glob%nzero /= 0) call dlf_fail("Soft mode skipping with &
          &nzero is not supported with the Lagrange-Newton method")   
     if (glob%imicroiter > 0) call dlf_fail("Microiterative &
          &conical intersection search is not yet implemented")  
  case default
    write(stderr, '(a,i4,a)') &
         "Multistate calculation", glob%imultistate, "not implemented"
    call dlf_fail("Multistate calculation option error")
  end select

end subroutine dlf_conint_check_consistency
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_make_conint_gradient
!!
!! FUNCTION
!!
!! Call the appropriate routine for constructing the optimiser 
!! gradient from the individual state gradients.
!!
!! INPUTS
!!
!! glob%imultistate
!!
!! OUTPUTS
!! 
!! none
!!
!! SYNOPSIS
subroutine dlf_make_conint_gradient
!! SOURCE
  use dlf_global, only: glob, stderr

  select case (glob%imultistate) 
  case (1)
     call dlf_make_pf_gradient
  case (2)
     call dlf_make_gp_gradient
  case (3)
     call dlf_make_ln_gradient_pretrans
  case default
    write(stderr, '(a,i4,a)') &
         "Multistate calculation", glob%imultistate, "not implemented"
    call dlf_fail("Multistate calculation option error")
  end select

end subroutine dlf_make_conint_gradient
!!****
!!
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_make_pf_gradient
!!
!! FUNCTION
!!
!! Construct penalty function gradient from individual state gradients.
!!
!! INPUTS
!!
!! glob%msenergy
!! glob%msgradient
!! glob%pf_c1
!! glob%pf_c2
!!
!! OUTPUTS
!! 
!! glob%energy
!! glob%xgradient
!!
!! SYNOPSIS
subroutine dlf_make_pf_gradient
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout, printl
  implicit none
  real(rk) :: wdiff, gscale
  integer :: ivar
  real(rk) :: c1h, c2h

  if (printl >= 6) write(stdout, '(a)') &
       "Calculating penalty function gradient"

! Convert c1 from (kcal/mol)^(-1) to Eh^(-1)
  c1h = glob%pf_c1 * 627.511525d0
! Convert c2 from kcal/mol to Eh
  c2h = glob%pf_c2 / 627.511525d0

! Calculate objective function for the penalty function method (Eq. 4)
! Stored in glob%energy for use by the optimiser.
  wdiff = 1.0d0 + &
       ((glob%msenergy(2) - glob%msenergy(1)) ** 2.0d0) / &
       (c2h ** 2.0d0)

  glob%energy = 0.5d0 * (glob%msenergy(1) + glob%msenergy(2)) + &
       c1h * (c2h ** 2.0d0) * dlog(wdiff)

! Calculate the gradient of the objective function (gradient of Eq. 4)
! Stored in glob%xgradient for use by the optimiser (after 
! transformation to the desired coordinate system).
  gscale = 2.0d0 * c1h * & 
       (glob%msenergy(2) - glob%msenergy(1)) / wdiff

  glob%xgradient = (0.5d0 + gscale) * glob%msgradient(:,:,2) + &
                   (0.5d0 - gscale) * glob%msgradient(:,:,1)

  if (printl >=6) then
     write(stdout, '(a,/,f20.10)') "PF energy:", glob%energy
     write(stdout, '(a)') "PF gradient:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') glob%xgradient(:, ivar)
     end do
  end if

end subroutine dlf_make_pf_gradient
!!****
!!
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_make_gp_gradient
!!
!! FUNCTION
!!
!! Construct gradient projection gradient from individual state
!! gradients.
!!
!! INPUTS
!!
!! glob%msenergy 
!! glob%msgradient
!! glob%gp_c3
!! glob%gp_c4
!!
!! OUTPUTS
!! 
!! glob%energy
!! glob%xgradient
!!
!! SYNOPSIS
subroutine dlf_make_gp_gradient
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout, printl
  implicit none
  real(rk), external :: ddot
  real(rk) :: gradDiff(3, glob%nat)
  real(rk) :: normGradDiff, normCoupling
  real(rk) :: projectedCoupling
  real(rk) :: tildeCoupling(3, glob%nat) ! orthonormalised coupling
  real(rk) :: f1(3, glob%nat)
  real(rk) :: f2(3, glob%nat)
  real(rk) :: tiny = 1.0d-10
  integer  :: ivar
  integer  :: i3nat

  if (printl >= 6) write(stdout, '(a)') &
       "Calculating gradient projection gradient"

  i3nat = 3 * glob%nat

! There is no 'energy' associated with the gradient projection algorithm
  glob%energy = 0.0d0

! Calculate gradient projection gradient
! Important note: for consistency the gradient difference and energy 
! difference are defined here as they are defined in the paper (Eqs. 2 
! and 5), and not as in the original MNDO implentation where the signs 
! of both were reversed.
! The two implementations are mathematically equivalent.

! Gradient difference is defined as g(I) - g(J)
! This is to be consistent with the paper (Eq. 2) 
! Note it is not consistent with the original MNDO implementation,
! where the sign is reversed
  gradDiff = glob%msgradient(:,:,1) - glob%msgradient(:,:,2)

! Normalise the gradient difference (but avoid division by zero).
  normGradDiff = sqrt(ddot(i3nat, gradDiff, 1, gradDiff, 1))
  if (normGradDiff > tiny) gradDiff = (1.D0 / normGradDiff) * gradDiff

! Now we can calculate f1 (Eq. 5)
! Note again the sign of the energy difference is consistent with the 
! paper but opposite to that in the MNDO implementation.
  f1 = 2.0d0 * (glob%msenergy(1) - glob%msenergy(2)) * gradDiff

! Compute projection of coupling vector along gradient difference vector
  projectedCoupling = ddot(i3nat, glob%mscoupling, 1, gradDiff, 1)

! Orthogonalise coupling vector: h(IJ) - <g(IJ)|h(IJ)> g(IJ)
  tildeCoupling = glob%mscoupling - (projectedCoupling * gradDiff)

! Normalise
  normCoupling = sqrt(ddot(i3nat, tildeCoupling, 1, tildeCoupling, 1))
  if (normCoupling > tiny) &
       tildeCoupling = (1.D0 / normCoupling) * tildeCoupling

! The gradient difference and coupling vectors are now orthonormal, 
! so we can calculate the projected gradient
!
! The projection matrix is defined as (Eq. 6):
! P = 1 - [g(IJ) * g(IJ,transposed)] - [h(IJ) * h(IJ,transposed)]
!
! And the projected gradient (Eq. 7):
! f2 = P * g(J)
!    = g(J) - [g(IJ) * g(IJ,transposed) * g(J)] 
!           - [h(IJ) * h(IJ,transposed) * g(J)] 
! 
! The last multiplication in the second and third terms is actually a 
! dot product.
  f2 = glob%msgradient(:,:,2) &
       - (gradDiff * ddot(i3nat, gradDiff, 1, &
                          glob%msgradient(1,1,2), 1)) &
       - (tildeCoupling * ddot(i3nat, tildeCoupling, 1, &
                               glob%msgradient(1,1,2), 1))

! Now form the gradient for the optimiser (Eq. 8)
! g = c3*[c4*f1 + (1-c4)*f2]
  glob%xgradient = glob%gp_c3 * ((glob%gp_c4 * f1) + &
                                 ((1.D0 - glob%gp_c4) * f2))

print*, "check 5"

! Warning (when comparing the MNDO and DL-FIND implementations)...
! The orthonormalisation procedure when carried out in ChemShell units
! leads to fairly significant variation compared to when carried out 
! in MNDO units. This is a numerical issue rather than an implementation
! difference.

  if (printl >= 6) then
     write(stdout, '(a,/,f20.10)') "GP energy:", glob%energy
     write(stdout, '(a)') "Normalised gradient difference:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') gradDiff(:, ivar)
     end do
     write(stdout, '(a)') "Orthonormal interstate coupling:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') tildeCoupling(:, ivar)
     end do
     write(stdout, '(a)') "GP gradient:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') glob%xgradient(:, ivar)
     end do
  end if

end subroutine dlf_make_gp_gradient
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_ln_allocate
!!
!! FUNCTION
!!
!! Allocates conint arrays and initialises extra coordinates required
!! for the Lagrange-Newton method
!!
!! INPUTS
!!
!! glob%nat
!! glob%nivar
!!
!! OUTPUTS
!!
!! glob%icoords(glob%nivar - 1:glob%nivar)
!! conint%extrap
!! conint%oldIsOrthog
!! 
!! SYNOPSIS
subroutine dlf_ln_allocate
!! SOURCE
  use dlf_global, only: glob
  use dlf_conint, only: conint
  use dlf_allocate, only: allocate

  call allocate(conint%xGradMean, 3, glob%nat)
  call allocate(conint%iGradMean, glob%nivar - 2)
  call allocate(conint%iGradMeanOld, glob%nivar - 2)
  call allocate(conint%xGradDiff, 3, glob%nat)
  call allocate(conint%iGradDiff, glob%nivar - 2)
  call allocate(conint%iGradDiffOld, glob%nivar - 2)
  call allocate(conint%iCoupling, glob%nivar - 2)
  call allocate(conint%iCouplingOld, glob%nivar - 2)

  glob%icoords(glob%nivar - 1:glob%nivar) = 0.0d0

  conint%extrap = .false.
  conint%oldIsOrthog = .false.

end subroutine dlf_ln_allocate
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_make_ln_gradient_pretrans
!!
!! FUNCTION
!!
!! Builds the unchanging part of the Lagrange-Newton gradient,
!! i.e. (gi + gj)/2. This is transformed by the usual xtoi procedure.
!!
!! The other parts of the gradient, gij and hij, are transformed 
!! separately and added after coordinate transformation.
!!
!! Rationale: 
!! * The internal forms of gij and hij are directly required to form the
!!   constraint part of the Hessian
!! * The gradient used for updating the non-constraint part of the
!!   Hessian only uses (gi + gj)/2 until it reaches the seam, when
!!   the information from g and h is used as well
!!
!! We also calculate xGradDiff here (cartesian gij) for use in 
!! dlf_coords_xtoi.
!!
!! INPUTS
!!
!! glob%msgradient
!!
!! OUTPUTS
!! 
!! glob%energy
!! conint%xGradMean
!! conint%xGradDiff
!!
!! SYNOPSIS
subroutine dlf_make_ln_gradient_pretrans
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout, printl
  use dlf_conint, only: conint
  implicit none
  integer  :: ivar

  if (printl >= 6) write(stdout, '(a)') &
       "Calculating base Lagrange-Newton gradients"

! There is no 'energy' associated with the Lagrange-Newton algorithm
  glob%energy = 0.0d0

! The unchanging part of the Lagrange-Newton gradient: (gi + gj) / 2
  conint%xGradMean = (glob%msgradient(:,:,1) + glob%msgradient(:,:,2)) &
       / 2.0d0

! Gradient difference vector (gij = gi - gj) in Cartesians
  conint%xGradDiff = glob%msgradient(:,:,1) - glob%msgradient(:,:,2)

  if (printl >= 6) then
     write(stdout, '(a,/,f20.10)') "LN energy:", glob%energy
     write(stdout, '(a)') "Mean of state gradients:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') conint%xGradMean(:, ivar)
     end do
     write(stdout, '(a)') "Difference of state gradients:"
     do ivar = 1, glob%nat
        write(stdout,'(3f20.10)') conint%xGradDiff(:, ivar)
     end do
  end if

end subroutine dlf_make_ln_gradient_pretrans
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_ln_xtoi
!!
!! FUNCTION
!!
!! Convert the base LN gradient (gi+gj)/2, the gradient difference,
!! and the interstate coupling gradient from Cartesians to internals.
!!
!! INPUTS
!!
!! glob%nvar
!! glob%nivar
!! glob%xcoords
!! conint%xGradMean
!! conint%xGradDiff
!! conint%mscoupling 
!!
!! OUTPUTS
!!
!! glob%icoords(1:glob%nivar - 2)
!! conint%iGradMean
!! conint%iGradDiff
!! conint%iCoupling
!!
!! SYNOPSIS
subroutine dlf_ln_xtoi
!! SOURCE
  use dlf_global, only: glob, stdout, printl
  use dlf_conint, only: conint
  implicit none
  integer :: ivar

  call dlf_direct_xtoi(glob%nvar, glob%nivar - 2, glob%nicore, glob%xcoords, &
       conint%xGradMean, glob%icoords(1:glob%nivar - 2), &
       conint%iGradMean)
  ! The following is a hack to transform the gradient
  ! difference vector and interstate coupling vector from cartesians
  ! to internals.
  ! Unfortunately coordinate and gradient transformation routines are 
  ! not separate, so we transform the same coordinates three times...
  call dlf_direct_xtoi(glob%nvar, glob%nivar - 2, glob%nicore, glob%xcoords, &
       conint%xGradDiff, glob%icoords(1:glob%nivar - 2), &
       conint%iGradDiff)
  call dlf_direct_xtoi(glob%nvar, glob%nivar - 2, glob%nicore, glob%xcoords, &
       glob%mscoupling, glob%icoords(1:glob%nivar - 2), &
       conint%iCoupling)

  if (printl >= 6) then
     write(stdout, '(a)') "Mean of state gradients in internals:"
     do ivar = 1, glob%nivar - 2
        write(stdout,'(f20.10)') conint%iGradMean(ivar)
     end do     
     write(stdout, '(a)') "Difference of state gradients in internals:"
     do ivar = 1, glob%nivar - 2
        write(stdout,'(f20.10)') conint%iGradDiff(ivar)
     end do     
     write(stdout, '(a)') "Interstate coupling gradient in internals:"
     do ivar = 1, glob%nivar - 2
        write(stdout,'(f20.10)') conint%iCoupling(ivar)
     end do     
  end if

end subroutine dlf_ln_xtoi
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_make_ln_gradient_posttrans
!!
!! FUNCTION
!!
!! Build the full Lagrange-Newton gradient in internal coordinates.
!!
!! INPUTS
!!
!! glob%nivar
!! glob%msenergy
!! conint%extrap
!! conint%iGradDiff
!! conint%iGradDiffOld
!! conint%iCoupling
!! conint%iCouplingOld
!! conint%iGradMean
!! glob%icoords(glob%nivar-1:glob%nivar)
!!
!! OUTPUTS
!! 
!! conint%extrap
!! conint%iGradDiff
!! conint%iCoupling
!! glob%igradient
!!
!! SYNOPSIS
subroutine dlf_make_ln_gradient_posttrans
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob, stdout, printl
  use dlf_conint, only: conint 
  implicit none
  real(rk), external :: ddot
  real(rk) :: extrapThresholdOn 
  real(rk) :: extrapThresholdOff
  real(rk) :: deltaE
  real(rk) :: gDotG, hDotH, hDotG, tan4Theta
  real(rk) ::  twoTheta, cos2Theta, sin2Theta
  real(rk) :: o1(glob%nivar - 2), o2(glob%nivar - 2)
  real(rk) :: nrmO1, nrmO2, nrmGOld, nrmHOld
  real(rk) :: spO1GOld, spO2GOld, spO1HOld, spO2HOld
  real(rk) :: cO1GOld, cO2GOld, cO1HOld, cO2HOld
  logical :: transposed
  real(rk) :: cGGold, cHHold
  real(rk) :: signG, signH
  real(rk) :: orthoTest
  integer  :: ivar
  integer  :: lnnivar

  if (printl >= 6) write(stdout, '(a)') &
       "Calculating full Lagrange-Newton gradient"

  ! Number of coordinates (not including Lagrange multipliers)
  lnnivar = glob%nivar - 2

  ! Thresholds for switching the orthogonalisation procedure on/off
  ! Convert user-defined values from kcal/mol to Eh
  extrapThresholdOn = glob%ln_t1 / 627.511525d0
  extrapThresholdOff = glob%ln_t2 / 627.511525d0

  ! Energy difference between states
  deltaE = glob%msenergy(1) - glob%msenergy(2)

  ! Determine if the energy gap is small enough to switch on 
  ! extrapolatable functions
  if (.not. conint%extrap .and. abs(deltaE) < extrapThresholdOn) then
     conint%extrap = .true.
     if (printl >= 4) then
        write(stdout, '(a)') "Conical intersection seam found: &
          &using extrapolatable functions."
        write(stdout, '(a, f20.10)') "Energy gap between states:", &
             abs(deltaE)
        write(stdout, '(a, f20.10)') "Below threshold of:       ", &
             extrapThresholdOn
     end if
  end if
  ! Determine if the energy gap is too large to continue using
  ! extrapolatable functions
  if (conint%extrap .and. abs(deltaE) > extrapThresholdOff) then
     conint%extrap = .false.
     if (printl >= 4) then
        write(stdout, '(a)') "Conical intersection seam lost: &
             &extrapolatable functions off."
        write(stdout, '(a, f20.10)') "Energy gap between states:", &
             abs(deltaE)
        write(stdout, '(a, f20.10)') "Above threshold of:       ", &
             extrapThresholdOff
     end if
  end if

  ! Start of orthogonalisation routine
  if (conint%extrap) then

     if (printl >= 6) write(stdout, '(a)') &
          "Orthogonalising gradient difference and coupling vectors"

     ! Halve the gradient difference vector
     ! and the gradient difference vector from the previous iteration
     conint%iGradDiff = conint%iGradDiff / 2.0d0
     conint%iGradDiffOld = conint%iGradDiffOld / 2.0d0

     ! Calculate the orthogonalisation angle (Eq. 18)
     ! Note that g here is actually 1/2g
     ! and there is a minus sign missing from Eq. 18
     ! (and the same equation in the earlier Yarkony papers)
     gDotG = ddot(lnnivar, conint%iGradDiff, 1, conint%iGradDiff, 1)
     hDotH = ddot(lnnivar, conint%iCoupling, 1, conint%iCoupling, 1)
     hDotG = ddot(lnnivar, conint%iCoupling, 1, conint%iGradDiff, 1)
     tan4Theta = -(2.0d0 * hDotG) / (hDotH - gDotG)
     twoTheta = atan(tan4Theta) / 2.0d0
     cos2Theta = cos(twoTheta)
     sin2Theta = sin(twoTheta)

     ! Orthogonalise the vectors (Eq. 17)
     o1 = cos2Theta * conint%iGradDiff + sin2Theta * conint%iCoupling
     o2 = -sin2Theta * conint%iGradDiff + cos2Theta * conint%iCoupling

     ! Test for transpositions and sign changes by comparing angles
     nrmO1 = sqrt(ddot(lnnivar, o1, 1, o1, 1))
     nrmO2 = sqrt(ddot(lnnivar, o2, 1, o2, 1))
     nrmGOld = sqrt(ddot(lnnivar, conint%iGradDiffOld, 1, &
          conint%iGradDiffOld, 1))
     nrmHOld = sqrt(ddot(lnnivar, conint%iCouplingOld, 1, &
          conint%iCouplingOld, 1))

     spO1GOld = ddot(lnnivar, o1, 1, conint%iGradDiffOld, 1)
     spO2GOld = ddot(lnnivar, o2, 1, conint%iGradDiffOld, 1) 
     spO1HOld = ddot(lnnivar, o1, 1, conint%iCouplingOld, 1)
     spO2HOld = ddot(lnnivar, o2, 1, conint%iCouplingOld, 1)

     ! cos(angle)
     cO1GOld = spO1GOld / (nrmO1 * nrmGOld)
     cO2GOld = spO2GOld / (nrmO2 * nrmGOld)
     cO1HOld = spO1HOld / (nrmO1 * nrmHOld)
     cO2HOld = spO2HOld / (nrmO2 * nrmHOld)

     ! First, check for transpositions
     if (abs(cO2GOld) > abs(cO1GOld)) then
        ! Transposition detected
        transposed = .true.
        cGGOld = cO2GOld
        cHHOld = cO1HOld
     else
        ! No transposition detected
        transposed = .false.
        cGGOld = cO1GOld
        cHHOld = cO2HOld
     endif

     ! Now check for sign changes in the transposition-corrected
     ! vectors
     if (cGGOld < 0.0d0) then
        ! Sign change in the gradient difference vector
        signG = -1.0d0
     else
        ! No sign change in the gradient difference vector
        signG = 1.0d0
     endif
     if (cHHOld < 0.0d0) then
        ! Sign change in the interstate coupling gradient
        signH = -1.0d0
     else 
        ! No sign change in the interstate coupling gradient
        signH = 1.0d0
     endif
     
     ! Copy the orthogonalised, transposition-corrected, 
     ! sign change-corrected vectors back, overwriting the original
     ! vectors (which are no longer needed)
     if (.not. transposed) then
        conint%iGradDiff = signG * o1
        conint%iCoupling = signH * o2
     else
        conint%iGradDiff = signG * o2
        conint%iCoupling = signH * o1
     endif

     ! Double the gradient difference vector again
     conint%iGradDiff = conint%iGradDiff * 2.0d0
     conint%iGradDiffOld = conint%iGradDiffOld * 2.0d0

     orthoTest = ddot(lnnivar, conint%iGradDiff, 1, conint%iCoupling, 1)

     if (printl >= 6) then
        write(stdout, '(a)') "Orthogonalised gradient difference:"
        do ivar = 1, glob%nivar - 2
           write(stdout,'(f20.10)') conint%iGradDiff(ivar)
        end do
        write(stdout, '(a)') "Orthogonalised interstate coupling:"
        do ivar = 1, glob%nivar - 2
           write(stdout,'(f20.10)') conint%iCoupling(ivar)
        end do
        write(stdout, '(a, f20.10)') &
             "Check of orthogonality: dot product: ", orthoTest
     end if


  endif ! End of orthogonalisation routine

  ! Construct the gradient of the Lagrangian (Eq. 16)
  ! del(Lij) = (gi + gj)/2 + xi1.gij + xi2.hij  
  ! The first term is already calculated from make_ln_gradient_pretrans
  ! The Lagrange multipliers xi1 and xi2 are the last two array
  ! elements of glob%icoords.
  glob%igradient(1:lnnivar) = conint%iGradMean(1:lnnivar) + &
       glob%icoords(lnnivar + 1) * conint%iGradDiff(1:lnnivar) + &
       glob%icoords(lnnivar + 2) * conint%iCoupling(1:lnnivar)

  ! Calculate the final two gradients (corresponding to the 
  ! constraints).
  ! If orthogonalisation is not switched on, the final two gradients
  ! are exactly as in Eq. 13.
  ! If orthogonalisation is switched on, the orthogonalisation procedure
  ! should also be applied to the final two gradients:
  !  - Halve the first gradient (energy difference), like we halved g.
  !  - Apply the rotation matrix from Eq. 17, i.e.
  !
  !  |  cos(2th)  sin(2th) | | (1/2)dE |  =  |  (1/2).cos(2th).dE |
  !  | -sin(2th)  cos(2th) | |    0    |     | -(1/2).sin(2th).dE | 
  !
  !  - If g and h were transposed, or there were sign changes, apply the
  !    same corrections to these gradients.
  !  - Double the first gradient again.

  ! Note: this treatment of the residual energy is an improvement on 
  ! the treatment described in the MNDO implementation paper, in which 
  ! transpositions and sign changes were taken into account, but the 
  ! orthogonalisation matrix was not applied. In practice, the two 
  ! approaches rarely differ in terms of convergence cycles (as the 
  ! original treatment is usually a good approximation to the current
  ! treatment).
  !
  ! In principle, the threshold for switching on orthogonalisation is 
  ! no longer required with the new treatment, as the constraint 
  ! equations with orthogonalised g and h now hold at any point, not 
  ! just points on the seam.
  ! However, the only difference this would make in practice is to 
  ! include orthogonalised g and h information into the Hessian at all 
  ! times, and whether this is advantageous or not is a moot point.
  ! Advantages of removing thresholds:
  !  - The algorithm is considerably simplified.
  !  - No disruption on crossing the threshold (fixes the problem cases
  !    s-transoid butadiene and penta-3,5-dieniminium from the MNDO 
  !    implementation paper).
  !  - Use of the full Hessian may force the optimiser to find the seam
  !    more quickly.
  ! Disadvantages of removing thresholds:
  !  - Use of the full Hessian may force the optimiser to find the seam
  !    too quickly! (i.e. a high energy seam). This happens in the case
  !    of diazomethane, greatly increasing the number of iterations 
  !    required to find the true MECP.
  !  - In most cases, removing the threshold slightly increases the 
  !    number of iterations required to reach convergence. 
  ! The best solution may be to make the thresholds user-definable.
  if (.not. conint%extrap) then
     glob%igradient(lnnivar + 1) = deltaE
     glob%igradient(lnnivar + 2) = 0.0d0
  else if (transposed) then
     glob%igradient(lnnivar + 1) = -1.0d0 * sin2Theta * deltaE * signG
     glob%igradient(lnnivar + 2) = 0.5d0 * cos2Theta * deltaE * signH
  else
     glob%igradient(lnnivar + 1) = cos2Theta * deltaE * signG
     glob%igradient(lnnivar + 2) = -0.5d0 * sin2Theta * deltaE * signH
  endif

  if (printl >= 6) then
     write(stdout, '(a)') "LN gradient:"
     do ivar = 1, glob%nivar
        write(stdout,'(f20.10)') glob%igradient(ivar)
     end do
  end if

end subroutine dlf_make_ln_gradient_posttrans
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_conint_make_ln_hess
!!
!! FUNCTION
!!
!! Make the Lagrange-Newton Hessian
!! Called from dlf_makehessian and analogous to it.
!!
!! In the Lagrange-Newton method the top left (nivar-2)*(nivar-2) 
!! Hessian is treated in a standard way. In order for this to work 
!! all the Hessian update routines called must be independent of glob.
!!
!! The rest of the Hessian contains constraint gradients corresponding
!! to the gradient difference and interstate coupling gradient vectors.
!!
!! INPUTS
!!
!! glob%nivar
!! conint%iGradMean
!! conint%iGradMeanOld
!! conint%extrap
!! conint%oldIsOrthog
!! glob%icoords
!! conint%iGradDiff
!! conint%iGradDiffOld
!! conint%iCoupling
!! conint%iCouplingOld
!!
!! OUTPUTS
!! 
!! glob%havehessian
!! glob%ihessian
!! tconv
!!
!! SYNOPSIS
subroutine dlf_conint_make_ln_hess(trerun_energy,tconv)
!! SOURCE
  use dlf_parameter_module, only: rk
  use dlf_global, only: glob,stdout,printl
  use dlf_stat, only: stat
  use dlf_conint, only: conint 
  use dlf_hessian, only: oldc,fd_hess_running
  implicit none
  logical, intent(inout) :: trerun_energy
  logical, intent(inout) :: tconv
  integer                :: status
  logical,save           :: fracrecalc ! behaves as in makehessian,
                                       ! possibly not relevant here
  integer                :: iimage ! not used here
  integer :: ivar
  integer :: lnnivar
  ! Gradients for Hessian update
  real(rk) :: updGradient(glob%nivar - 2)
  real(rk) :: updGradientOld(glob%nivar - 2)
  logical  :: was_updated
! **********************************************************************
  ! lnnivar is the dimension of the optimisation problem without the
  ! constraints. All Hessian construction, updating etc. should only
  ! apply to the lnnivar * lnnivar part of the Hessian
  lnnivar = glob%nivar - 2

  ! Gradients for Hessian update (not used for any other purpose)
  updGradient = conint%iGradMean
  updGradientOld = conint%iGradMeanOld
  if (conint%extrap .and. conint%oldIsOrthog) then
     ! If both the old and the new gradient difference/couplings
     ! have been orthogonalised, they are slowly varying and can be 
     ! included in the Hessian update.
     ! Note that the Lagrange multipliers of the current iteration are
     ! always used.
     updGradient = updGradient + &
          glob%icoords(lnnivar + 1) * conint%iGradDiff + &
          glob%icoords(lnnivar + 2) * conint%iCoupling
     updGradientOld = updGradientOld + &
          glob%icoords(lnnivar + 1) * conint%iGradDiffOld + &
          glob%icoords(lnnivar + 2) * conint%iCouplingOld
  endif
  call dlf_hessian_update(lnnivar, glob%icoords(1:lnnivar), & 
       oldc(1:lnnivar), updGradient, updGradientOld, &
       glob%ihessian(1:lnnivar, 1:lnnivar), glob%havehessian, &
       fracrecalc,was_updated)
  
  if(glob%havehessian) then
       ! Fill in the last two rows/columns of the hessian
       ! with gradient difference/coupling after a Hessian update
       glob%ihessian(1:lnnivar, lnnivar + 1) = &
            conint%iGradDiff(1:lnnivar)
       glob%ihessian(lnnivar + 1, 1:lnnivar) = &
            conint%iGradDiff(1:lnnivar)
       glob%ihessian(1:lnnivar, lnnivar + 2) = &
            conint%iCoupling(1:lnnivar)
       glob%ihessian(lnnivar + 2, 1:lnnivar) = &
            conint%iCoupling(1:lnnivar)
       ! The last 2 x 2 matrix of zeroes
       glob%ihessian(lnnivar + 1:lnnivar + 2, lnnivar + 1:lnnivar + 2) &
            = 0.0d0
       if (printl >= 6) then
          write(stdout, '(a)') "Hessian update gradient 1:"
          do ivar = 1, lnnivar
             write(stdout,'(f20.10)') updGradient(ivar)
          end do
          write(stdout, '(a)') "Hessian update gradient 2:"
          do ivar = 1, lnnivar
             write(stdout,'(f20.10)') updGradientOld(ivar)
          end do
          write(stdout, '(a)') "Lagrange-Newton Hessian:"
          do ivar = 1, glob%nivar
             write(stdout,'(12f10.5)') glob%ihessian(ivar,:)
          end do
       end if

  else
    ! We don't have a Hessian yet

    if(.not. fd_hess_running) then
      ! test for convergence before calculating the Hessian
      call convergence_test(stat%ccycle,.false.,tconv)
      if(tconv) return
      if (glob%inithessian == 4) then
         ! Initial Hessian is the identity matrix
         glob%ihessian = 0.0d0
         do ivar = 1, lnnivar
            glob%ihessian(ivar, ivar) = 1.0d0
         end do
         glob%havehessian = .true.
      end if
      if (glob%inithessian == 0) then
         call dlf_fail("External Hessian is incompatible with &
              &Lagrange-Newton")
      end if
    end if !(.not. fd_hess_running)

    if(.not.glob%havehessian) then
       ! For finite difference Hessians, it is not safe to use 
       ! extrapolatable functions (as the seam may be lost during 
       ! the finite difference procedure), so we call with iGradMean
       ! instead of updGradient
      if (glob%inithessian == 3) then
         ! Simple diagonal Hessian a la MNDO
         call dlf_diaghessian(lnnivar, glob%energy, &
              glob%icoords(1:lnnivar), conint%iGradMean, &
              glob%ihessian(1:lnnivar, 1:lnnivar), glob%havehessian)
      else
         ! Finite Difference Hessian calculation in internal coordinates
         call dlf_fdhessian(lnnivar, fracrecalc, glob%energy, &
              glob%icoords(1:lnnivar), conint%iGradMean, &
              glob%ihessian(1:lnnivar, 1:lnnivar), glob%havehessian)
      end if
      ! check if FD-Hessian calculation currently running
      trerun_energy=(fd_hess_running) 
      if(trerun_energy) then
        call clock_start("COORDS")
        call dlf_coords_itox(iimage)
        call clock_stop("COORDS")
      end if
    end if !(.not.glob%havehessian)

    if(glob%havehessian) then
       ! Fill in the last two rows/columns of the hessian
       ! with gradient difference/coupling
       glob%ihessian(1:lnnivar, lnnivar + 1) = &
            conint%iGradDiff(1:lnnivar)
       glob%ihessian(lnnivar + 1, 1:lnnivar) = &
            conint%iGradDiff(1:lnnivar)
       glob%ihessian(1:lnnivar, lnnivar + 2) = &
            conint%iCoupling(1:lnnivar)
       glob%ihessian(lnnivar + 2, 1:lnnivar) = &
            conint%iCoupling(1:lnnivar)
       ! The last 2 x 2 matrix of zeroes
       glob%ihessian(lnnivar + 1:lnnivar + 2, lnnivar + 1:lnnivar + 2) &
            = 0.0d0
       if (printl >= 6) then
          write(stdout, '(a)') "Lagrange-Newton Hessian:"
          do ivar = 1, glob%nivar
             write(stdout,'(12f10.5)') glob%ihessian(ivar,:)
          end do
       end if
    endif !(glob%havehessian)

  end if ! the first (glob%havehessian)

end subroutine dlf_conint_make_ln_hess
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_ln_savegrads
!!
!! FUNCTION
!!
!! Save the Lagrange-Newton gradients for the next step
!! Called from dlf_formstep
!!
!! INPUTS
!!
!! conint%iGradMean
!! conint%iGradDiff
!! conint%iCoupling
!! conint%extrap
!!
!! OUTPUTS
!! 
!! conint%iGradMeanOld
!! conint%iGradDiffOld
!! conint%iCouplingOld
!! conint%oldIsOrthog
!!
!! SYNOPSIS
subroutine dlf_ln_savegrads
!! SOURCE
  use dlf_conint, only: conint 
  implicit none

  conint%iGradMeanOld = conint%iGradMean
  conint%iGradDiffOld = conint%iGradDiff
  conint%iCouplingOld = conint%iCoupling

  if (conint%extrap) then
     conint%oldIsOrthog = .true.
  else
     conint%oldIsOrthog = .false.
  endif

end subroutine dlf_ln_savegrads
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_checkpoint_conint_write
!!
!! FUNCTION
!!
!! Write checkpoint file for the conint module (only necessary for the
!! Lagrange-Newton method).
!!
!! INPUTS
!!
!! conint%...
!!
!! OUPUTS
!!
!! file: dlf_conint.chk
!!
!! SYNOPSIS
subroutine dlf_checkpoint_conint_write
!! SOURCE
   use dlf_global, only: glob
   use dlf_conint, only: conint
   use dlf_checkpoint, only: tchkform, write_separator
   implicit none

   select case (glob%imultistate)
   case (0)
      ! Not a multistate calculation
   case (1, 2)
      ! No checkpoint required
   case (3)
      if (tchkform) then
         open(unit=100, file="dlf_conint.chk", form="formatted")
      else
         open(unit=100, file="dlf_conint.chk", form="unformatted")
      end if
      call write_separator(100, "LN data")
      if (tchkform) then
         write(100,*) conint%xGradMean, conint%iGradMean, &
              conint%iGradMeanOld, conint%xGradDiff, &
              conint%iGradDiffOld, conint%iCoupling, &
              conint%iCouplingOld, conint%extrap, conint%oldIsOrthog
      else
         write(100) conint%xGradMean, conint%iGradMean, &
              conint%iGradMeanOld, conint%xGradDiff, &
              conint%iGradDiffOld, conint%iCoupling, &
              conint%iCouplingOld, conint%extrap, conint%oldIsOrthog
      end if
      call write_separator(100, "END LN data")
      close(100)
   end select

end subroutine dlf_checkpoint_conint_write
!!****

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!!****f* conint/dlf_checkpoint_conint_read
!!
!! FUNCTION
!!
!! Read checkpoint file for the conint module (only necessary for the
!! Lagrange-Newton method).
!!
!! INPUTS
!!
!! file: dlf_conint.chk
!!
!! OUPUTS
!!
!! conint%...
!!
!! SYNOPSIS
subroutine dlf_checkpoint_conint_read(tok)
!! SOURCE
   use dlf_global, only: glob, stdout
   use dlf_conint, only: conint
   use dlf_checkpoint, only: tchkform, read_separator
   implicit none
   logical, intent(out) :: tok
   logical :: tchk

   tok = .false.

   select case (glob%imultistate)
   case (0)
      ! Not a multistate calculation
      tok = .true.
   case (1, 2)
      ! No checkpoint required
      tok = .true.
   case (3)
      ! check if checkpoint file exists
      inquire(file="dlf_conint.chk", exist=tchk)
      if(.not.tchk) then
         write(stdout,10) "File dlf_conint.chk not found"
         return
      end if
      if (tchkform) then
         open(unit=100, file="dlf_conint.chk", form="formatted")
      else
         open(unit=100, file="dlf_conint.chk", form="unformatted")
      end if
      call read_separator(100, "LN data", tchk)
      if (.not. tchk) return
      if (tchkform) then
         read(100,*,end=201,err=200) conint%xGradMean, conint%iGradMean, &
              conint%iGradMeanOld, conint%xGradDiff, &
              conint%iGradDiffOld, conint%iCoupling, &
              conint%iCouplingOld, conint%extrap, conint%oldIsOrthog
      else
         read(100,end=201,err=200) conint%xGradMean, conint%iGradMean, &
              conint%iGradMeanOld, conint%xGradDiff, &
              conint%iGradDiffOld, conint%iCoupling, &
              conint%iCouplingOld, conint%extrap, conint%oldIsOrthog
      end if
      call read_separator(100, "END LN data", tchk)
      if (.not. tchk) return
      close(100)
      tok = .true.
   end select
   return

   ! return on error
200 continue
   write(stdout,10) "Error reading conint checkpoint file"
   return
201 continue
   write(stdout,10) "Error (EOF) reading conint checkpoint file"
   return

10 format("Checkpoint reading WARNING: ",a)

end subroutine dlf_checkpoint_conint_read
!!****
