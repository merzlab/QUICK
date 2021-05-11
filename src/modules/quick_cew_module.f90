module quick_cew_module

  use, intrinsic :: iso_c_binding
  implicit none

  public :: quick_cew_type
  public :: quick_cew
  public :: new_quick_cew
  public :: quick_cew_prescf
  public :: quick_cew_grad

  
  private

  
  type quick_cew_type
     logical :: use_cew = .false.
     double precision :: beta = 0.d0
     double precision :: zeta = 0.d0
     double precision, allocatable :: qmq(:)
     double precision :: E = 0.d0
  end type quick_cew_type


  type(quick_cew_type),save :: quick_cew
  double precision,parameter :: sqrt_pi = 1.7724538509055159d0



  subroutine cew_getrecip( pt, pot ) bind(c,name="cew_getpotatpt_")
    use, intrinsic :: iso_c_binding
    implicit none
    real(kind=c_double),intent(in) :: pt
    real(kind=c_double),intent(out) :: pot
  end subroutine cew_getrecip

  subroutine cew_getgrdgrad( pt, pot ) bind(c,name="cew_getgrdatpt_")
    use, intrinsic :: iso_c_binding
    implicit none
    real(kind=c_double),intent(in) :: pt
    real(kind=c_double),intent(out) :: pot
  end subroutine cew_getgrdgrad

  
  subroutine cew_accdens( pt, pot ) bind(c,name="cew_accdensatpt_")
    use, intrinsic :: iso_c_binding
    implicit none
    real(kind=c_double),intent(in) :: pt
    real(kind=c_double),intent(out) :: pot
  end subroutine cew_accdens


  
contains

  
  subroutine new_quick_cew_type(self,beta,nqm,qmq)
    
    implicit none
    type(quick_cew_type), intent(inout) :: self
    double precision,intent(in) :: beta
    integer,intent(in) :: nqm
    double precision,intent(in) :: qmq(nqm)
    

    self%use_cew = .true.
    self%beta = beta
    self%zeta = beta*beta
    if ( allocated(self%qmq) ) then
       deallocate(self%qmq)
    end if
    allocate(self%qmq(nqm))
    self%qmq = qmq
    
  end subroutine new_quick_cew_type


  
  subroutine new_quick_cew(beta,nqm,qmq)

    implicit none
    double precision,intent(in) :: beta
    integer,intent(in) :: nqm
    double precision,intent(in) :: qmq(nqm)

    call new_quick_cew_type( quick_cew )

  end subroutine new_quick_cew

  
  
  subroutine quick_cew_prescf()
    
    use quick_api_module, only : quick_api
    use quick_long_range_module, only: computeLongRange    
    implicit none

    double precision :: E
    double precision :: pot
    double precision :: qa,qb
    double precision :: rvec(3),r
    double precision :: ew_self
    integer :: a,b
    
    ! CEw: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.6b00198
    ! See Eq (83).
    !
    ! The 1st term [Eq (84)] is the periodic MM-MM interaction.
    ! Quick is not responsible for Eq (84).
    !
    ! The 2nd term is the standard QM energy (NN, EN, EE interactions)
    ! when there are no external point charges.
    ! Quick is responsible for this term, but no modifications
    ! to the code are necessary.
    !
    ! The 3rd term is the interaction of the QM region with
    ! the nearby MM point charges. Quick is responsible for this
    ! term, but no modifications to the code are necessary.
    !
    ! The 4th term is the QM interaction with the reciprocal
    ! space potential caused by the MM and reference QM charges.
    ! The interaction of the potential with the electrons is
    ! handled via numerical quadrature and inserted into the
    ! core Hamiltonian.  The interaction with the nuceli
    ! simply require one to lookup the value of the potential
    ! at the nuclear positions and multiply them by the nuclear
    ! charges.
    !
    ! The 5th term removes the short-range real-space interactions
    ! between the nuceli and the Ewald Gaussians. The Ewald
    ! Gaussians involved in this term must correspond to the
    ! surrounding MM charges appearing in the 3rd term (and the
    ! reference QM charges).  The 5th term also remove the
    ! short-range real-space interactions between the electron
    ! density and the nearby Ewald Gaussians.
    ! Quick is responsible for this term, and it will be
    ! included in this routine.
    !
    ! The 6th term is the interaction of the reference QM
    ! charge interactions with the reciprocal space potential
    ! with itself. Quick is not responsible for this term.
    !
    ! The last term is the real space interaction between
    ! the reference QM charges with their corresponding
    ! Ewald Gaussians.  Quick is responsible for this term
    ! and it shall be included below. In principle, the
    ! external CEw library could do this, but that's not
    ! the way the library was originally written (for no
    ! good reason).
    !

    
    !
    ! erf(beta*r)/r as r -> 0
    !
    ew_self = 2.d0 * quick_cew%beta / sqrt_pi

    
    E = 0.d0

    ! 
    ! Remove the interaction of the nuclei with the Ewald Gaussians
    ! located at the external point positions
    !
    ! E -= \sum_{a \in QM} \sum_{b \in MM} Za*Qb*erf(\beta r)/r   
    !
    ! This is the first term in Eq. (92).
    ! [which is a component of the 1st term in Eq (89)]
    !
    do a=1, quick_api%natom
       qa = quick_api%atomic_numbers(a)
       do b=1, quick_api%nxt_ptchg
          qb = quick_api%ptchg_crd(4,b)
          rvec = quick_api%coords(1:3,a)-quick_api%ptchg_crd(1:3,b)
          r = sqrt(rvec(1)*rvec(1)+rvec(2)*rvec(2)+rvec(3)*rvec(3))
          E = E - qa*qb*erf( quick_cew%beta * r ) / r
       end do
    end do


    ! 
    ! Remove the interaction of the nuclei with the Ewald Gaussians
    ! located at the nuclei positions
    !
    ! E -= \sum_{a \in QM} \sum_{b \in QM} Za*Qb*erf(\beta r)/r
    !      where Qb is the MM-like charge of the QM atom
    !
    ! This is the other component within the 1st term of Eq. (89).
    !
    ! We will also compute Eq. (86) at the same time [the last term
    ! in Eq (83)].  By itself, it reads:
    ! 
    ! E += \sum_{a \in QM} \sum_{b \in QM} (1/2)*Qa*Qb*erf(\beta r)/r
    !
    ! but we can combine these 2 equations and evaluate:
    !
    ! E -= \sum_{a \in QM} \sum_{b \in QM} (Za-Qa/2)*Qb*erf(\beta r)/r
    !
    do a=1, quick_api%natom
       qa = quick_api%atomic_numbers(a) - 0.5d0 * quick_cew%qmq(a)
       do b=1, quick_api%natom
          qb = quick_cew%qmq(b)
          if ( a /= b ) then
             rvec = quick_api%coords(1:3,a)-quick_api%coords(1:3,b)
             r = sqrt(rvec(1)*rvec(1)+rvec(2)*rvec(2)+rvec(3)*rvec(3))
             E = E - qa*qb*erf( quick_cew%beta * r ) / r
          else
             E = E - qa*qb*ew_self
          end if
       end do
    end do




    !
    ! Remove the real-space interaction of the electron density
    ! with the Ewald Gaussians. This is the contribution of
    ! Eq (90) to Eq. (83).
    !
    !
    ! TODO *****************************************************
    !
    !
    ! do a=1, quick_api%natom
    !   qa = quick_api%atomic_numbers(a)
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|a0)
    !        H(i,j) += qa * (quick_cew%zeta/PI)**1.5d0 * (ij|a0)
    !        where (r|a0) = exp( - quick_cew%zeta * |r-Ra|^2 )

      do a=1, quick_api%natom
        qa = quick_api%atomic_numbers(a)
        call computeLongRange(quick_api%coords(1:3,a), quick_cew%zeta, qa)
      enddo

    !
    ! do a=1, quick_api%nxt_ptchg
    !   qa = quick_api%ptchg_crd(4,a)
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|a0)
    !        H(i,j) += qa * (quick_cew%zeta/PI)**1.5d0 * (ij|a0)
    !        where (r|a0) = exp( - quick_cew%zeta * |r-Ra|^2 )
    !

      do a=1, quick_api%nxt_ptchg
        qa = quick_api%ptchg_crd(4,a)
        call computeLongRange(quick_api%ptchg_crd(1:3,a), quick_cew%zeta, qa)
      enddo

    ! Notice that we are ADDING (not subtracting) this term to
    ! the core Hamiltonian, because the density matrix is a
    ! "number density", whereas the electron charge density
    ! has a negative sign. The negative sign due to the charge
    ! density and the negative sign due to the removal will
    ! cancel each other, and thus we add.  In other words,
    ! Eq (90) has minus signs, but the whole term has a minus
    ! sign in Eq (83), and those signs cancel.
    !



    
    !
    ! Interact the nuceli with the reciprocal space potential
    ! This is the 1st term in Eq (87)
    !
    do a=1, quick_api%natom
       qa = quick_api%atomic_numbers(a)
       pot = 0.d0
       call cew_getrecip( quick_api%coords(1,a), pot )
       E = E + qa * pot
    end do

    
    !
    ! Interact the electron density with the reciprocal
    ! space potential.  That is,
    !
    ! H(i,j) -= \int Vrecip(r) * phi_i(r) * phi_j(r) d3r
    !    or, equivalently,
    ! H(i,j) += \int (-Vrecip(r)) * phi_i(r) * phi_j(r) d3r
    !
    ! The minus sign occurs because the density matrix
    ! is a number density, not a charge density.
    ! In practice, we will lookup the value of Vrecip(r)
    ! and negate it.
    !

    call quick_cew_prescf_quad()
    
    
  end subroutine quick_cew_prescf



  subroutine quick_cew_prescf_quad()
   use allmod
   use xc_f90_types_m
   use xc_f90_lib_m
   implicit none

#ifdef MPIV
   include "mpif.h"
#endif

   !integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, I, J
   !common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   double precision, dimension(1) :: libxc_rho
   double precision, dimension(1) :: libxc_sigma
   double precision, dimension(1) :: libxc_exc
   double precision, dimension(1) :: libxc_vrhoa
   double precision, dimension(1) :: libxc_vsigmaa
   integer :: iatm, ibas, ibin, icount, ifunc, igp, jbas, jcount, ierror 
   double precision :: density, densityb, densitysum, dfdgaa, dfdgaa2, dfdgab, &
   dfdgab2, dfdr, dfdr2, dphi2dx, dphi2dy, dphi2dz, dphidx, dphidy, dphidz, &
   gax, gay, gaz, gbx, gby, gbz, gridx, gridy, gridz, phi, phi2, quicktest, &
   sigma, sswt, temp, tempgx, tempgy, tempgz, tsttmp_exc, tsttmp_vrhoa, &
   tsttmp_vsigmaa, weight, xdot, ydot, zdot, xiaodot, zkec, Ex, Ec, Eelxc

#ifdef MPIV
   integer :: i, ii, irad_end, irad_init, jj
#endif


   double precision :: Vrecip, cew_pt(3)
   

#if defined MPIV && !defined CUDA_MPIV
      if(bMPI) then
         irad_init = quick_dft_grid%igridptll(mpirank+1)
         irad_end = quick_dft_grid%igridptul(mpirank+1)
      else
         irad_init = 1
         irad_end = quick_dft_grid%nbins
      endif
   do Ibin=irad_init, irad_end
   
#else
    do Ibin=1, quick_dft_grid%nbins
#endif

        Igp=quick_dft_grid%bin_counter(Ibin)+1

        do while(Igp < quick_dft_grid%bin_counter(Ibin+1)+1)

           gridx=quick_dft_grid%gridxb(Igp)
           gridy=quick_dft_grid%gridyb(Igp)
           gridz=quick_dft_grid%gridzb(Igp)

           sswt=quick_dft_grid%gridb_sswt(Igp)
           weight=quick_dft_grid%gridb_weight(Igp)
           Iatm=quick_dft_grid%gridb_atm(Igp)

            if (weight < quick_method%DMCutoff ) then
               continue
            else

               icount=quick_dft_grid%basf_counter(Ibin)+1
               do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
               Ibas=quick_dft_grid%basf(icount)+1
                  call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                  dphidz,Ibas)
                  phixiao(Ibas)=phi
                  dphidxxiao(Ibas)=dphidx
                  dphidyxiao(Ibas)=dphidy
                  dphidzxiao(Ibas)=dphidz

                  icount=icount+1
               enddo

!  Next, evaluate the densities at the grid point and the gradient
!  at that grid point.

               !call denspt_new_imp(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
               !gbx,gby,gbz,Ibin)

               !if (density < quick_method%DMCutoff ) then
               !   continue
               !else

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
               cew_pt(1) = gridx
               cew_pt(2) = gridy
               cew_pt(3) = gridz
               Vrecip = 0.d0
               call cew_getrecip( cew_pt(1), Vrecip )
               dfdr = -Vrecip
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               
!  This allows the calculation of the derivative of the functional with regard to the 
!  density (dfdr), with regard to the alpha-alpha density invariant (df/dgaa), and the
!  alpha-beta density invariant.

                  densitysum=2.0d0*density
                  sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)

!  Now loop over basis functions and compute the addition to the matrix element.
!                  do Ibas=1,nbasis
                  icount=quick_dft_grid%basf_counter(Ibin)+1
                  do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                  Ibas=quick_dft_grid%basf(icount)+1

                     phi=phixiao(Ibas)
                     dphidx=dphidxxiao(Ibas)
                     dphidy=dphidyxiao(Ibas)
                     dphidz=dphidzxiao(Ibas)
                     quicktest = DABS(dphidx+dphidy+dphidz+phi)

                     if (quicktest < quick_method%DMCutoff ) then
                        continue
                     else
                        jcount=icount
                        do while(jcount<quick_dft_grid%basf_counter(Ibin+1)+1)
                        Jbas = quick_dft_grid%basf(jcount)+1
                        !do Jbas=Ibas,nbasis
                           phi2=phixiao(Jbas)
                           !dphi2dx=dphidxxiao(Jbas)
                           !dphi2dy=dphidyxiao(Jbas)
                           !dphi2dz=dphidzxiao(Jbas)
                           temp = phi*phi2
                           !tempgx = phi*dphi2dx + phi2*dphidx
                           !tempgy = phi*dphi2dy + phi2*dphidy
                           !tempgz = phi*dphi2dz + phi2*dphidz
                           quick_qm_struct%o(Jbas,Ibas) = &
                                & quick_qm_struct%o(Jbas,Ibas) &
                                & + temp*dfdr*weight
                           jcount=jcount+1
                        enddo
                     endif
                     icount=icount+1
                  enddo
               endif
            endif
         !enddo

         Igp=Igp+1
      enddo
   enddo


   return

  end subroutine quick_cew_prescf_quad
  






  subroutine quick_cew_grad()
    
    use quick_api_module, only : quick_api
    
    implicit none

    !
    ! erf(beta*r)/r as r -> 0
    !
    ew_self = 2.d0 * quick_cew%beta / sqrt_pi


    
    do a=1, quick_api%natom
       qa = quick_cew%qmq(a)
       do b=1, quick_api%natom
          if ( a /= b ) then
             qb = quick_api%atomic_numbers(b) - 0.5d0 * quick_cew%qmq(b)
             rvec = quick_api%coords(1:3,a)-quick_api%coords(1:3,b)
             r2 = rvec(1)*rvec(1)+rvec(2)*rvec(2)+rvec(3)*rvec(3)
             r  = sqrt(r2)
             oor2 = 1.d0 / r2
             oor3 = oor2 / r
             !
             ! dedr =>  - qa*qb * (1/r) * d/dr * erf(beta*r)/r
             !
             dedr = -qa*qb*( ew_self * exp(-quick_cew%zeta*r2) * oor2 &
                  & - erf(quick_cew%beta*r) * oor3 )
             !
             rvec = rvec * dedr
             quick_api%gradient(1,a) = quick_api%gradient(1,a) + rvec(1)
             quick_api%gradient(2,a) = quick_api%gradient(2,a) + rvec(2)
             quick_api%gradient(3,a) = quick_api%gradient(3,a) + rvec(3)
             quick_api%gradient(1,b) = quick_api%gradient(1,b) - rvec(1)
             quick_api%gradient(2,b) = quick_api%gradient(2,b) - rvec(2)
             quick_api%gradient(3,b) = quick_api%gradient(3,b) - rvec(3)
         end if
       end do
    end do

    
    do a=1, quick_api%natom
       qa = quick_api%atomic_numbers(a)
       rvec = 0.d0
       call cew_getrecipgrad( quick_api%coords(1,a), rvec(1) )
       gradient(1,a) = gradient(1,a) + qa * rvec(1)
       gradient(2,a) = gradient(2,a) + qa * rvec(2)
       gradient(3,a) = gradient(3,a) + qa * rvec(3)
    end do

    
    do a=1, quick_api%nxt_ptchg
       qa = quick_api%ptchg_crd(4,a)
       do b=1, quick_api%natom
          if ( a /= b ) then
             qb = quick_api%atomic_numbers(b)
             rvec = quick_api%ptchg_crd(1:3,a)-quick_api%coords(1:3,b)
             r2 = rvec(1)*rvec(1)+rvec(2)*rvec(2)+rvec(3)*rvec(3)
             r  = sqrt(r2)
             oor2 = 1.d0 / r2
             oor3 = oor2 / r
             !
             ! dedr =>  - qa*qb * (1/r) * d/dr * erf(beta*r)/r
             !
             dedr = -qa*qb*( ew_self * exp(-quick_cew%zeta*r2) * oor2 &
                  & - erf(quick_cew%beta*r) * oor3 )
             !
             rvec = rvec * dedr
             quick_api%ptchg_grad(1,a) = quick_api%ptchg_grad(1,a) + rvec(1)
             quick_api%ptchg_grad(2,a) = quick_api%ptchg_grad(2,a) + rvec(2)
             quick_api%ptchg_grad(3,a) = quick_api%ptchg_grad(3,a) + rvec(3)
             quick_api%gradient(1,b) = quick_api%gradient(1,b) - rvec(1)
             quick_api%gradient(2,b) = quick_api%gradient(2,b) - rvec(2)
             quick_api%gradient(3,b) = quick_api%gradient(3,b) - rvec(3)
         end if
       end do
    end do


    !
    ! TODO *****************************************************
    !
    !
    ! do c=1, quick_api%natom
    !   qc = quick_api%atomic_numbers(a) * (quick_cew%zeta/PI)**1.5d0
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|c0)
    !        ga(1) = ga(1) + D(i,j) * qc * d/dXa (ij|c0)
    !        ga(2) = ga(2) + D(i,j) * qc * d/dYa (ij|c0)
    !        ga(3) = ga(3) + D(i,j) * qc * d/dZa (ij|c0)
    !        gb(1) = gb(1) + D(i,j) * qc * d/dXb (ij|c0)
    !        gb(2) = gb(2) + D(i,j) * qc * d/dYb (ij|c0)
    !        gb(3) = gb(3) + D(i,j) * qc * d/dZb (ij|c0)
    !        gc(1) = gc(1) + D(i,j) * qc * d/dXc (ij|c0)
    !        gc(2) = gc(2) + D(i,j) * qc * d/dYc (ij|c0)
    !        gc(3) = gc(3) + D(i,j) * qc * d/dZc (ij|c0)
    !
    ! do c=1, quick_api%nxt_ptchg
    !   qc = quick_api%ptchg_crd(4,a) * (quick_cew%zeta/PI)**1.5d0
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|c0)
    !        ga(1) = ga(1) + D(i,j) * qc * d/dXa (ij|c0)
    !        ga(2) = ga(2) + D(i,j) * qc * d/dYa (ij|c0)
    !        ga(3) = ga(3) + D(i,j) * qc * d/dZa (ij|c0)
    !        gb(1) = gb(1) + D(i,j) * qc * d/dXb (ij|c0)
    !        gb(2) = gb(2) + D(i,j) * qc * d/dYb (ij|c0)
    !        gb(3) = gb(3) + D(i,j) * qc * d/dZb (ij|c0)
    !        gc(1) = gc(1) + D(i,j) * qc * d/dXc (ij|c0)
    !        gc(2) = gc(2) + D(i,j) * qc * d/dYc (ij|c0)
    !        gc(3) = gc(3) + D(i,j) * qc * d/dZc (ij|c0)


    call quick_cew_grad_quad()

    
  end subroutine quick_cew_grad






  subroutine quick_cew_grad_quad()


   use allmod
   use xc_f90_types_m
   use xc_f90_lib_m
   implicit double precision(a-h,o-z)

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   double precision, dimension(1) :: libxc_rho
   double precision, dimension(1) :: libxc_sigma
   double precision, dimension(1) :: libxc_exc
   double precision, dimension(1) :: libxc_vrhoa
   double precision, dimension(1) :: libxc_vsigmaa
   
   double precision :: Vrecip, cew_pt(3), chargedens, cew_grd(3)

#ifdef MPIV
   include "mpif.h"
#endif

#if defined MPIV && !defined CUDA_MPIV
      if(bMPI) then
         irad_init = quick_dft_grid%igridptll(mpirank+1)
         irad_end = quick_dft_grid%igridptul(mpirank+1)
      else
         irad_init = 1
         irad_end = quick_dft_grid%nbins
      endif
      do Ibin=irad_init, irad_end
#else
      do Ibin=1, quick_dft_grid%nbins
#endif

!  Calculate the weight of the grid point in the SSW scheme.  If
!  the grid point has a zero weight, we can skip it.

!    do Ibin=1, quick_dft_grid%nbins
        Igp=quick_dft_grid%bin_counter(Ibin)+1

        do while(Igp < quick_dft_grid%bin_counter(Ibin+1)+1)

           gridx=quick_dft_grid%gridxb(Igp)
           gridy=quick_dft_grid%gridyb(Igp)
           gridz=quick_dft_grid%gridzb(Igp)

           sswt=quick_dft_grid%gridb_sswt(Igp)
           weight=quick_dft_grid%gridb_weight(Igp)
           Iatm=quick_dft_grid%gridb_atm(Igp)

            if (weight < quick_method%DMCutoff ) then
               continue
            else

               icount=quick_dft_grid%basf_counter(Ibin)+1
               do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                  Ibas=quick_dft_grid%basf(icount)+1

                  call pteval_new_imp(gridx,gridy,gridz,phi,dphidx,dphidy, &
                  dphidz,Ibas,icount)

                  phixiao(Ibas)=phi
                  dphidxxiao(Ibas)=dphidx
                  dphidyxiao(Ibas)=dphidy
                  dphidzxiao(Ibas)=dphidz

                  icount=icount+1
               enddo

               

!  evaluate the densities at the grid point and the gradient at that grid point            
               call denspt_new_imp(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
               gbx,gby,gbz,Ibin)

               if (density < quick_method%DMCutoff ) then
                  continue

               else
!  This allows the calculation of the derivative of the functional
!  with regard to the density (dfdr), with regard to the alpha-alpha
!  density invariant (df/dgaa), and the alpha-beta density invariant.

                  densitysum=2.0d0*density
                  sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)

                  libxc_rho(1)=densitysum
                  libxc_sigma(1)=sigma

                  tsttmp_exc=0.0d0
                  tsttmp_vrhoa=0.0d0
                  tsttmp_vsigmaa=0.0d0

                  ! call becke_E(density, densityb, gax, gay, gaz, gbx, gby,gbz, Ex)
                  ! call lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz,Ec)
                  
                  ! zkec=Ex+Ec
                  
                  ! call becke(density, gax, gay, gaz, gbx, gby, gbz, dfdr, dfdgaa, dfdgab)
                  ! call lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, dfdr2, dfdgaa2, dfdgab2)
                  
                  ! dfdr = dfdr + dfdr2
                  ! dfdgaa = dfdgaa + dfdgaa2
                  ! dfdgab = dfdgab + dfdgab2
                  
                  ! xdot = 2.d0*dfdgaa*gax + dfdgab*gbx
                  ! ydot = 2.d0*dfdgaa*gay + dfdgab*gby
                  ! zdot = 2.d0*dfdgaa*gaz + dfdgab*gbz

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  
                  Vrecip = 0.d0
                  call cew_getrecip( cew_pt(1), Vrecip )
                  zkec = -densitysum*Vrecip
                  dfdr = -Vrecip

                  chargedens = - weight * densitysum
                  cew_pt(1) = gridx
                  cew_pt(2) = gridy
                  cew_pt(3) = gridz
                  cew_grd = 0.d0
                  call cew_accdens(cew_pt(1),chargedens,cew_grd(1))
                  quick_api%gradient(1,Iatm) = quick_api%gradient(1,Iatm) + cew_grd(1)
                  quick_api%gradient(2,Iatm) = quick_api%gradient(2,Iatm) + cew_grd(2)
                  quick_api%gradient(3,Iatm) = quick_api%gradient(3,Iatm) + cew_grd(3)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! Now loop over basis functions and compute the addition to the matrix
! element.
                  icount=quick_dft_grid%basf_counter(Ibin)+1
                  do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                     Ibas=quick_dft_grid%basf(icount)+1

                     phi=phixiao(Ibas)
                     dphidx=dphidxxiao(Ibas)
                     dphidy=dphidyxiao(Ibas)
                     dphidz=dphidzxiao(Ibas)

                     !call pteval_new_imp(gridx,gridy,gridz,phi,dphidx,dphidy, &
                     !dphidz,Ibas,icount)


                     quicktest = DABS(dphidx+dphidy+dphidz+phi)
                     
                     if (quicktest < quick_method%DMCutoff ) then
                        continue
                     else
                        call pt2der(gridx,gridy,gridz,dxdx,dxdy,dxdz, &
                        dydy,dydz,dzdz,Ibas,icount)

                        Ibasstart=(quick_basis%ncenter(Ibas)-1)*3

                        jcount=quick_dft_grid%basf_counter(Ibin)+1
                        do while(jcount<quick_dft_grid%basf_counter(Ibin+1)+1)
                           Jbas = quick_dft_grid%basf(jcount)+1 

                           phi2=phixiao(Jbas)
                           dphi2dx=dphidxxiao(Jbas)
                           dphi2dy=dphidyxiao(Jbas)
                           dphi2dz=dphidzxiao(Jbas)

                           !call pteval_new_imp(gridx,gridy,gridz,phi2,dphi2dx,dphi2dy, &
                           !dphi2dz,Jbas,jcount)

                           quick_qm_struct%gradient(Ibasstart+1) =quick_qm_struct%gradient(Ibasstart+1) - &
                           2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*&
                           (dfdr*dphidx*phi2 &
                           + xdot*(dxdx*phi2+dphidx*dphi2dx) &
                           + ydot*(dxdy*phi2+dphidx*dphi2dy) &
                           + zdot*(dxdz*phi2+dphidx*dphi2dz))
                           quick_qm_struct%gradient(Ibasstart+2)= quick_qm_struct%gradient(Ibasstart+2) - &
                           2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*&
                           (dfdr*dphidy*phi2 &
                           + xdot*(dxdy*phi2+dphidy*dphi2dx) &
                           + ydot*(dydy*phi2+dphidy*dphi2dy) &
                           + zdot*(dydz*phi2+dphidy*dphi2dz))
                           quick_qm_struct%gradient(Ibasstart+3)= quick_qm_struct%gradient(Ibasstart+3) - &
                           2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*&
                           (dfdr*dphidz*phi2 &
                           + xdot*(dxdz*phi2+dphidz*dphi2dx) &
                           + ydot*(dydz*phi2+dphidz*dphi2dy) &
                           + zdot*(dzdz*phi2+dphidz*dphi2dz))
                           jcount=jcount+1
                        enddo
                     endif

                  icount=icount+1
                  enddo

!  We are now completely done with the derivative of the exchange correlation energy with nuclear displacement
!  at this point. Now we need to do the quadrature weight derivatives. At this point in the loop, we know that
!  the density and the weight are not zero. Now check to see fi the weight is one. If it isn't, we need to
!  actually calculate the energy and the derivatives of the quadrature at this point. Due to the volume of code,
!  this is done in sswder. Note that if a new weighting scheme is ever added, this needs
!  to be modified with a second subprogram.
                  if (sswt == 1.d0) then
                     continue
                  else
                     call sswder(gridx,gridy,gridz,zkec,weight/sswt,Iatm)
                  endif
               endif
            endif
!         enddo

      Igp=Igp+1
      enddo
   enddo


   return


  end subroutine quick_cew_grad_quad




  
end module quick_cew_module
