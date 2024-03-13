#ifdef CEW
module quick_cew_module

  use, intrinsic :: iso_c_binding
  implicit none

  public :: quick_cew_type
  public :: quick_cew
  public :: new_quick_cew
  public :: quick_cew_prescf
  public :: quick_cew_grad
  public :: print

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  public :: upload
  public :: delete
  public :: cew_accdens
#endif
  
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

  double precision,parameter :: AMBERELE = 18.2223d0
  double precision,parameter :: BOHRS_TO_A = 0.529177249d0
  double precision,parameter :: AU_PER_AMBER_CHARGE = 1.d0 / amberele
  double precision,parameter :: AU_PER_AMBER_ANGSTROM = 1.d0 / bohrs_to_a
  double precision,parameter :: AU_PER_AMBER_KCAL_PER_MOL = &
       & AU_PER_AMBER_CHARGE *  AU_PER_AMBER_CHARGE / AU_PER_AMBER_ANGSTROM
  double precision,parameter :: AU_PER_AMBER_FORCE = &
       & AU_PER_AMBER_KCAL_PER_MOL / AU_PER_AMBER_ANGSTROM

  
  interface
     subroutine cew_getrecip( pt, pot ) bind(c,name="cew_getpotatpt_")
       use, intrinsic :: iso_c_binding
       implicit none
       real(kind=c_double),intent(in) :: pt
       real(kind=c_double),intent(out) :: pot
     end subroutine cew_getrecip
     
     subroutine cew_getgrdatpt( pt, grd ) bind(c,name="cew_getgrdatpt_")
       use, intrinsic :: iso_c_binding
       implicit none
       real(kind=c_double),intent(in) :: pt
       real(kind=c_double),intent(out) :: grd
     end subroutine cew_getgrdatpt
     
     
     subroutine cew_accdens( pt, dens, grd ) bind(c,name="cew_accdensatpt_")
       use, intrinsic :: iso_c_binding
       implicit none
       real(kind=c_double),intent(in) :: pt
       real(kind=c_double),intent(in) :: dens
       real(kind=c_double),intent(out) :: grd
     end subroutine cew_accdens
  end interface

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  ! MM: interface for GPU uploading of cew info
  interface upload
    module procedure upload_cew
  end interface upload

  interface delete
    module procedure delete_cew_vrecip
  end interface
#endif  

  interface print
    module procedure print_cew
  end interface print
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

    call new_quick_cew_type( quick_cew,beta,nqm,qmq )

  end subroutine new_quick_cew

  
  
  subroutine quick_cew_prescf()
    
    !use quick_api_module, only : quick_api
    use quick_molspec_module, only: quick_molspec
    use quick_lri_module, only : computeLRI
    use quick_calculated_module, only : quick_qm_struct
    use quick_basis_module
    use quick_method_module, only: quick_method
#ifdef MPIV
    use quick_mpi_module
    use mpi
#endif    

    implicit none

    double precision :: E
    double precision :: pot
    double precision :: qa,qb
    double precision :: rvec(3),r
    double precision :: ew_self
    integer :: a,b,c,ierr
    double precision, dimension(:), allocatable :: chgs
#ifdef MPIV
    integer :: atominit, atomlast, extatominit, extatomlast
    double precision :: Esum
    Esum = 0.0d0

    atominit = natomll(mpirank+1)
    atomlast = natomul(mpirank+1)
    extatominit = nextatomll(mpirank+1)
    extatomlast = nextatomul(mpirank+1)
#endif

    ierr=0

    !double precision :: Emm, Eqm, Epot
    
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

    !Emm = 0.d0

#ifdef MPIV
    do b=extatominit, extatomlast
#else
    do b=1, quick_molspec%nextatom
#endif
       qb = quick_molspec%extchg(b)
       do a=1, quick_molspec%natom
          qa = quick_molspec%chg(a)
          rvec = quick_molspec%xyz(1:3,a)-quick_molspec%extxyz(1:3,b)
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
    !Eqm = 0.d0

#ifdef MPIV
    do a=atominit, atomlast
#else
    do a=1, quick_molspec%natom
#endif
       qa = quick_molspec%chg(a) - 0.5d0 * quick_cew%qmq(a)
       do b=1, quick_molspec%natom
          qb = quick_cew%qmq(b)
          if ( a /= b ) then
             rvec = quick_molspec%xyz(1:3,a)-quick_molspec%xyz(1:3,b)
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
    ! do a=1, quick_molspec%natom
    !   qa = quick_molspec%chg(a)
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|a0)
    !        H(i,j) += qa * (quick_cew%zeta/PI)**1.5d0 * (ij|a0)
    !        where (r|a0) = exp( - quick_cew%zeta * |r-Ra|^2 )


#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

    if(.not. allocated(chgs)) allocate(chgs(quick_molspec%natom+quick_molspec%nextatom))

    do c=1, (quick_molspec%natom + quick_molspec%nextatom)
      if ( c <= quick_molspec%natom ) then
        chgs(c)=quick_cew%qmq(c)
      else
        chgs(c)=quick_molspec%extchg(c-quick_molspec%natom)
      endif
    enddo

    call gpu_upload_lri(quick_cew%zeta, chgs, ierr)

    call gpu_upload_cew_vrecip(ierr)

    call gpu_get_lri(quick_qm_struct%o)

    if ( .not. (quick_method%grad .or. quick_method%opt) ) then 
      call gpu_delete_lri(ierr)
      call delete_cew_vrecip(quick_cew, ierr)
    endif

    if(allocated(chgs)) deallocate(chgs)

#else

    do a=1, quick_molspec%natom
       qa = quick_cew%qmq(a)
       call computeLRI(quick_molspec%xyz(1:3,a), quick_cew%zeta, qa)
    enddo

    !
    ! do a=1, quick_api%nxt_ptchg
    !   qa = quick_molspec%extchg(a)
    !   foreach AO i
    !      foreach AO j
    !        Compute (ij|a0)
    !        H(i,j) += qa * (quick_cew%zeta/PI)**1.5d0 * (ij|a0)
    !        where (r|a0) = exp( - quick_cew%zeta * |r-Ra|^2 )
    !

    do a=1, quick_molspec%nextatom
       qa = quick_molspec%extchg(a)
       call computeLRI(quick_molspec%extxyz(1:3,a), quick_cew%zeta, qa)
    enddo
#endif

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
    !Epot = 0.d0
#ifdef MPIV
    do a=atominit, atomlast
#else
    do a=1, quick_molspec%natom
#endif
       qa = quick_molspec%chg(a)
       pot = 0.d0
       call cew_getrecip( quick_molspec%xyz(1,a), pot )
       E = E + qa * pot
    end do
    !write(6,*)"Ecore",(quick_qm_struct%ECore + E)

#ifdef MPIV
    call MPI_REDUCE(E,Esum,1,mpi_double_precision,mpi_sum,0, MPI_COMM_WORLD, mpierror)
    E=Esum
#endif

    quick_qm_struct%ECore = quick_qm_struct%ECore + E
    
    
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

#if !defined (CUDA) && !defined (CUDA_MPIV) && !defined (HIP) && !defined (HIP_MPIV)
    call quick_cew_prescf_quad()
#endif    
    
  end subroutine quick_cew_prescf



  subroutine quick_cew_prescf_quad()
   use allmod
   use xc_f90_types_m
   use xc_f90_lib_m
   use quick_gridpoints_module, only : quick_dft_grid
   use quick_molspec_module, only : quick_molspec
#ifdef MPIV
    use mpi
#endif
   implicit none

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


   double precision :: Vrecip, cew_pt(3),localsswt
   

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
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

!            call getssw(gridx,gridy,gridz,Iatm,natom,quick_molspec%xyz,localsswt)
!            weight = weight * localsswt/sswt
            
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
               !endif
               !enddo

            Igp=Igp+1
         enddo
      end do

   return

  end subroutine quick_cew_prescf_quad
  






  subroutine quick_cew_grad()
    
    !use quick_api_module, only : quick_api
    use quick_molspec_module, only : quick_molspec
    use quick_calculated_module, only : quick_qm_struct
    use quick_lri_grad_module, only: computeLRIGrad
    !use quick_lri_grad_module, only: computeLRINumGrad
    use quick_gridpoints_module, only : quick_dft_grid
    use quick_method_module, only: quick_method
#ifdef MPIV
    use quick_mpi_module
#endif    

    implicit none
    integer :: a,b,c,k,oa,ob,oc,ierr
    double precision :: c_coord(3), rvec(3)
    double precision :: r,r2,oor2,oor3,qa,qb,qc
    double precision :: dedr
    double precision :: ew_self
    double precision :: ga(3), gb(3), gc(3)

#ifdef MPIV
    integer :: atominit, atomlast, extatominit, extatomlast

    atominit = natomll(mpirank+1)
    atomlast = natomul(mpirank+1)
    extatominit = nextatomll(mpirank+1)
    extatomlast = nextatomul(mpirank+1)
#endif

    ierr=0

    !
    ! erf(beta*r)/r as r -> 0
    !
    ew_self = 2.d0 * quick_cew%beta / sqrt_pi


#ifdef MPIV
    do a=atominit, atomlast
#else
    do a=1, quick_molspec%natom
#endif
       qa = quick_cew%qmq(a)
       do b=1, quick_molspec%natom
          if ( a /= b ) then
             qb = quick_molspec%chg(b) - 0.5d0 * quick_cew%qmq(b)
             rvec = quick_molspec%xyz(1:3,a)-quick_molspec%xyz(1:3,b)
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
             oa = 3*(a-1)
             ob = 3*(b-1)
             do k=1,3
                quick_qm_struct%gradient(oa+k) = quick_qm_struct%gradient(oa+k) + rvec(k)
                quick_qm_struct%gradient(ob+k) = quick_qm_struct%gradient(ob+k) - rvec(k)
             end do
         end if
       end do
    end do

    
#ifdef MPIV
    do a=atominit, atomlast
#else    
    do a=1, quick_molspec%natom
#endif
       qa = quick_molspec%chg(a)
       rvec = 0.d0
       call cew_getgrdatpt( quick_molspec%xyz(1,a), rvec(1) )
       ! write(6,'(a,i4,7f15.10)')"planewavegrd ",a-1,qa, &
       !      & quick_molspec%xyz(1,a),quick_molspec%xyz(2,a),quick_molspec%xyz(3,a), &
       !      & rvec(1),rvec(2),rvec(3)
       oa = 3*(a-1)
       do k=1,3
          quick_qm_struct%gradient(oa+k) = quick_qm_struct%gradient(oa+k) + qa * rvec(k)
       end do
    end do

    
#ifdef MPIV
    do a=extatominit, extatomlast
#else    
    do a=1, quick_molspec%nextatom
#endif
       qa = quick_molspec%extchg(a)
       do b=1, quick_molspec%natom
          qb = quick_molspec%chg(b)
          rvec = quick_molspec%extxyz(1:3,a)-quick_molspec%xyz(1:3,b)
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
          oa = 3*(a-1)
          ob = 3*(b-1)
          do k=1,3
             quick_qm_struct%ptchg_gradient(oa+k) = quick_qm_struct%ptchg_gradient(oa+k) + rvec(k)
             quick_qm_struct%gradient(ob+k) = quick_qm_struct%gradient(ob+k) - rvec(k)
          end do
       end do
    end do

    !
    ! TODO *****************************************************
    !
    !
    ! do c=1, quick_molspec%natom
    !   qc = quick_molspec%chg(a) * (quick_cew%zeta/PI)**1.5d0
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
    !   qc = quick_molspec%extchg(a) * (quick_cew%zeta/PI)**1.5d0
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

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV

    call gpu_get_lri_grad(quick_qm_struct%gradient,quick_qm_struct%ptchg_gradient)

    call gpu_delete_lri(ierr)

    if(quick_method%HF) then
      call gpu_reupload_dft_grid()
      call gpu_getcew_grad_quad(quick_qm_struct%gradient)
      call delete_cew_vrecip(quick_cew, ierr)
    endif

#else
    do c=1, (quick_molspec%natom + quick_molspec%nextatom)
       if ( c <= quick_molspec%natom ) then
          c_coord=quick_molspec%xyz(1:3,c)
          qc=quick_cew%qmq(c)
       else
          c_coord=quick_molspec%extxyz(1:3,c-quick_molspec%natom)
          qc=quick_molspec%extchg(c-quick_molspec%natom)   
       end if
       call computeLRIGrad(c_coord,quick_cew%zeta,qc,c)
    enddo
#endif

#if !defined (CUDA) && !defined (CUDA_MPIV) && !defined (HIP) && !defined (HIP_MPIV)
    call quick_cew_grad_quad()
#endif
    
  end subroutine quick_cew_grad





  ! subroutine getssw(gridx,gridy,gridz,Iparent,natom,xyz,p)
  !   implicit none
  !   double precision,intent(in) :: gridx,gridy,gridz
  !   integer,intent(in) :: Iparent,natom
  !   double precision,intent(in) :: xyz(3,natom)
  !   double precision,intent(out) :: p
    
  !   double precision,parameter :: a = 0.64
  !   integer :: iat,jat
  !   double precision :: uw(natom)
  !   double precision :: mu,muoa,g,s,z
  !   double precision :: rxg(4,natom)
  !   double precision :: rigv(3),rig,rig2
  !   double precision :: rjgv(3),rjg,rjg2
  !   double precision :: rijv(3),rij,rij2
  !   double precision :: sumw
   

  !   uw = 0.d0
    
  !   do iat=1,natom
  !      rigv(1) = xyz(1,iat)-gridx
  !      rigv(2) = xyz(2,iat)-gridy
  !      rigv(3) = xyz(3,iat)-gridz
  !      rig2 = rigv(1)*rigv(1)+rigv(2)*rigv(2)+rigv(3)*rigv(3)
  !      rig = sqrt(rig2)
  !      rxg(1:3,iat) = rigv(1:3)
  !      rxg(4,iat) = rig
  !   end do


  !   ! Calculate wi(rg)
  !   do iat=1,natom
  !      uw(iat) = 1.d0
       
  !      rigv(1:3) = rxg(1:3,iat)
  !      rig = rxg(4,iat)

  !      ! wi(rg) = \prod_{j /= i} s(mu_{ij})
  !      do jat=1,natom
  !         if ( jat == iat ) then
  !            cycle
  !         end if
  !         rjgv(1:3) = rxg(1:3,jat)
  !         rjg = rxg(4,jat)
          
  !         rijv(1) = xyz(1,iat)-xyz(1,jat)
  !         rijv(2) = xyz(2,iat)-xyz(2,jat)
  !         rijv(3) = xyz(3,iat)-xyz(3,jat)
  !         rij2 = rijv(1)*rijv(1)+rijv(2)*rijv(2)+rijv(3)*rijv(3)
  !         rij = sqrt(rij2)

  !         mu = (rig-rjg)/rij

          
  !         if ( mu <= -a ) then
  !            g = -1.d0
  !         else if ( mu >= a ) then
  !            g = 1.d0
  !         else
  !            muoa = mu/a
  !            z = (35.d0*muoa - 35.d0*muoa**3 &
  !                 & + 21.d0*muoa**5 - 5.d0*muoa**7)/16.d0
  !            g = z
  !         end if
  !         !if ( iat == 1 .and. jat == 3 ) write(6,'(es20.10)')mu
          
  !         s = 0.50d0 * (1.d0-g)
  !         !if ( iat == 1 .and. jat == 3 ) write(6,'(2es20.10)')mu,s

  !         uw(iat) = uw(iat) * s
  !      end do
  !   end do

    
  !   sumw = 0.d0
  !   do iat=1,natom
  !      sumw = sumw + uw(iat)
  !   end do
  !   p = uw(Iparent) / sumw

  !   !write(6,'(es20.10)')sumw

    
  ! end subroutine getssw



  ! subroutine getsswnumder(gridx,gridy,gridz,Iparent,natom,xyz,dp)
  !   implicit none
  !   double precision,intent(in) :: gridx,gridy,gridz
  !   integer,intent(in) :: Iparent,natom
  !   double precision,intent(in) :: xyz(3,natom)
  !   double precision,intent(out) :: dp(3,natom)
  !   double precision,parameter :: delta = 2.5d-5
  !   double precision :: phi,plo
  !   double precision :: tmpxyz(3,natom)
  !   double precision :: tx,ty,tz
  !   integer :: iat,k

  !   tmpxyz=xyz
  !   tx = gridx - xyz(1,Iparent)
  !   ty = gridy - xyz(2,Iparent)
  !   tz = gridz - xyz(3,Iparent)

    
  !   do iat=1,natom
  !      do k=1,3
          
  !         tmpxyz(k,iat) = tmpxyz(k,iat) + delta
          
  !         call getssw( tx+tmpxyz(1,Iparent), &
  !              & ty+tmpxyz(2,Iparent), &
  !              & tz+tmpxyz(3,Iparent), &
  !              & Iparent,natom,tmpxyz,phi)

          
  !         tmpxyz(k,iat) = tmpxyz(k,iat) - 2.d0*delta
          
  !         call getssw( tx+tmpxyz(1,Iparent), &
  !              & ty+tmpxyz(2,Iparent), &
  !              & tz+tmpxyz(3,Iparent), &
  !              & Iparent,natom,tmpxyz,plo)

  !         tmpxyz(k,iat) = tmpxyz(k,iat) + delta

  !         dp(k,iat) = (phi-plo)/(2.d0*delta)
  !      end do
  !   end do
    
  ! end subroutine getsswnumder




  




  subroutine quick_cew_grad_quad()


   use allmod
   use xc_f90_types_m
   use xc_f90_lib_m
   use quick_gridpoints_module, only : quick_dft_grid
   !use quick_api_module, only : quick_api
   use quick_calculated_module, only : quick_qm_struct
   use quick_molspec_module, only : quick_molspec
#ifdef MPIV
   use mpi
#endif
   
   implicit double precision(a-h,o-z)

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   double precision, dimension(1) :: libxc_rho
   double precision, dimension(1) :: libxc_sigma
   double precision, dimension(1) :: libxc_exc
   double precision, dimension(1) :: libxc_vrhoa
   double precision, dimension(1) :: libxc_vsigmaa
   
   double precision :: Vrecip, cew_pt(3), chargedens, cew_grd(3)
   integer :: Iatm,Ibas,Ibasstart,Ibin,icount,Igp,Jbas,jcount
   integer :: k,oi,i
   !double precision,allocatable :: spcder(:)
   double precision :: grdx,grdy,grdz
   double precision :: sumg(3),localsswt
   double precision :: dp(3,natom)
   double precision :: ap(3,natom)

   !allocate( spcder(3*natom) )
   !spcder = 0.d0
   
#ifdef MPIV
   integer :: irad_init, irad_end
#endif

#if defined MPIV && !defined CUDA_MPIV && !defined HIP_MPIV
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
                       & dphidz,Ibas,icount)

                  phixiao(Ibas)=phi
                  dphidxxiao(Ibas)=dphidx
                  dphidyxiao(Ibas)=dphidy
                  dphidzxiao(Ibas)=dphidz

                  icount=icount+1
               enddo

               
               !call getssw(gridx,gridy,gridz,Iatm,natom,quick_molspec%xyz,localsswt)
               !weight = weight * localsswt/sswt
            

               !  evaluate the densities at the grid point and the gradient at that grid point

               
               
               call denspt_new_imp(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                    & gbx,gby,gbz,Ibin)
               

               !if (density < quick_method%DMCutoff ) then
               !   continue
            !else
!  This allows the calculation of the derivative of the functional
!  with regard to the density (dfdr), with regard to the alpha-alpha
!  density invariant (df/dgaa), and the alpha-beta density invariant.



                  
                  densitysum=2.0d0*density
                  sigma=4.0d0*(gax*gax+gay*gay+gaz*gaz)

                  libxc_rho(1)=densitysum
                  libxc_sigma(1)=sigma

                  !tsttmp_exc=0.0d0
                  !tsttmp_vrhoa=0.0d0
                  !tsttmp_vsigmaa=0.0d0

                  ! call becke_E(density, densityb, gax, gay, gaz, gbx, gby,gbz, Ex)
                  ! call lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz,Ec)
                  
                  ! zkec=Ex+Ec
                  
                  ! call becke(density, gax, gay, gaz, gbx, gby, gbz, dfdr, dfdgaa, dfdgab)
                  ! call lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, dfdr2, dfdgaa2, dfdgab2)
                  
                  ! dfdr = dfdr + dfdr2
                  ! dfdgaa = dfdgaa + dfdgaa2
                  ! dfdgab = dfdgab + dfdgab2
                  
                  !xdot = 2.d0*dfdgaa*gax + dfdgab*gbx
                  !ydot = 2.d0*dfdgaa*gay + dfdgab*gby
                  !zdot = 2.d0*dfdgaa*gaz + dfdgab*gbz
                  xdot = 0.d0
                  ydot = 0.d0
                  zdot = 0.d0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  
                  Vrecip = 0.d0
                  zkec=0.d0
                  dfdr=0.d0
                  cew_pt(1) = gridx
                  cew_pt(2) = gridy
                  cew_pt(3) = gridz
                  call cew_getrecip( cew_pt(1), Vrecip )
                  zkec = -densitysum*Vrecip
                  dfdr = -Vrecip

                  chargedens = -weight * densitysum
                  cew_pt(1) = gridx
                  cew_pt(2) = gridy
                  cew_pt(3) = gridz
                  cew_grd = 0.d0
                  call cew_accdens(cew_pt(1),chargedens,cew_grd(1))
                  oi = 3*(Iatm-1)
                  do k=1,3
                     quick_qm_struct%gradient(oi+k) = quick_qm_struct%gradient(oi+k) + cew_grd(k)
                  end do

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! Now loop over basis functions and compute the addition to the matrix
                  ! element.
                  sumg = 0.d0
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
                        !call pt2der(gridx,gridy,gridz,dxdx,dxdy,dxdz, &
                        !     & dydy,dydz,dzdz,Ibas,icount)

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

                           grdx= -2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*(dfdr*dphidx*phi2)
                           grdy= -2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*(dfdr*dphidy*phi2)
                           grdz= -2.d0*quick_qm_struct%dense(Ibas,Jbas)*weight*(dfdr*dphidz*phi2)


                           !
                           ! This is the d\rho/dX contribution from line 1 of Eq (39)
                           !
                           quick_qm_struct%gradient(Ibasstart+1) = &
                                & quick_qm_struct%gradient(Ibasstart+1) + grdx
                           quick_qm_struct%gradient(Ibasstart+2) = &
                                & quick_qm_struct%gradient(Ibasstart+2) + grdy
                           quick_qm_struct%gradient(Ibasstart+3) = &
                                & quick_qm_struct%gradient(Ibasstart+3) + grdz

                           sumg(1) = sumg(1) + grdx
                           sumg(2) = sumg(2) + grdy
                           sumg(3) = sumg(3) + grdz
                           
                           jcount=jcount+1
                        enddo
                     endif

                  icount=icount+1
               enddo

               !
               ! This is the d\rho/dx * dx/dX contribution from line 1 of Eq (39)
               !
               
               oi = 3*(Iatm-1)
               do k=1,3
                 quick_qm_struct%gradient(oi+k) = quick_qm_struct%gradient(oi+k) - sumg(k)
               end do
                  
!  We are now completely done with the derivative of the exchange correlation energy with nuclear displacement
!  at this point. Now we need to do the quadrature weight derivatives. At this point in the loop, we know that
               !  the density and the weight are not zero.
               
                  if (sswt == 1.d0) then
                     continue
                  else
                     ! The sswder routine is not giving me the proper weight gradients
                     !call sswder(gridx,gridy,gridz,zkec,weight/sswt,Iatm)

                     
                     !call getsswnumder(gridx,gridy,gridz,Iatm,natom,xyz(1:3,1:natom),dp)
                     call getsswanader(gridx,gridy,gridz,Iatm,natom,xyz(1:3,1:natom),ap)

                     !DO i=1,natom
                     !   write(6,'(2I4,9ES13.4)')i,Iatm,dp(:,i),ap(:,i),dp(:,i)-ap(:,i)
                     !END DO
                     
                     sumg(1) = weight / sswt
                     !write(6,'(3es20.10)')weight,localsswt,sumg(1)
                     DO i=1,natom
                        oi = 3*(i-1)
                        do k=1,3
                           quick_qm_struct%gradient(oi+k)=quick_qm_struct%gradient(oi+k) &
                                & + ap(k,i)*zkec*sumg(1)
                        end do
                     END DO

                     
                  endif
               endif
            !endif
!         enddo

      Igp=Igp+1
   enddo
end do
   !deallocate( spcder )
   
   return


  end subroutine quick_cew_grad_quad

#if defined CUDA || defined CUDA_MPIV || defined HIP || defined HIP_MPIV
  ! MM: upload cew info onto GPU
  subroutine upload_cew(self, ierr)

    implicit none
    type(quick_cew_type), intent(in) :: self
    integer, intent(out) :: ierr

    ierr=0
    call gpu_set_cew(self%use_cew)

  end subroutine upload_cew

  subroutine delete_cew_vrecip(self, ierr)

    implicit none
    type(quick_cew_type), intent(in) :: self ! dummy argument to access through interface
    integer, intent(inout) :: ierr

    ierr=0
  
    call gpu_delete_cew_vrecip(ierr)

  end subroutine delete_cew_vrecip

#endif

  subroutine print_cew(self, iOutfile, ierr)
  
    implicit none
    type(quick_cew_type), intent(in) :: self
    integer, intent(out) :: iOutfile
    integer, intent(out) :: ierr

    if(self%use_cew) then 
      write(iOutfile,'("| CEw = ON")') 
    else
      write(iOutfile,'("| CEw = OFF")')
    endif

    ierr=0

  end subroutine print_cew

end module quick_cew_module
#endif
