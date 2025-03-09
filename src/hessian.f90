#include "util.fh"
! Ed Brothers. October 22,2002
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine calchessian(failed)
  use allmod
  implicit double precision(a-h,o-z)
  logical failed
  
  failed=.false.

  ! This subroutine calculates the second derivative of energy with respect
  ! to nuclear displacement.  It then uses the analytical Hessian to
  ! optimize geometry if this is an optimization job, and finally calculates
  ! the frequency.  Note that if this is an optimization job it should have
  ! already passed though the LBFGS optimizer before getting here, and thus
  ! requires only refinement.
  
  call PrtAct(ioutfile,"Begin Hessian calculation")

  ! First print out a warning.
  if ( .not. quick_method%opt) &
       write (ioutfile,'(/" WARNING !! FREQUENCIES ONLY VALID AT ", &
       & "OPTIMIZED GEOMETRY!!!")')
       
  ! Now calculate the Hessian.
  if (quick_method%analhess) then !.and.quick_method%HF) then
    ! Analytical Hessian Matrix, but is broken now
    write (ioutfile,'(/"ANALYTICAL HESSIAN CURRENTLY BROKEN.")')
    call HFHessian
  else
    ! Numerical Hessian Matrix
    call fdhessian(failed)
  endif
  
  ! Output Hessian Matrix
  write (ioutfile,'(/"HESSIAN MATRIX ")')
  call PriHessian(ioutfile,3*natom,quick_qm_struct%hessian,'f12.6')
  
  call PrtAct(ioutfile,"Finish Hessian calculation")
end subroutine calchessian


! Ed Brothers. January 21, 2003.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine fdhessian(failed)
  use allmod
  use quick_grad_cshell_module, only: cshell_gradient
  use quick_grad_oshell_module, only: oshell_gradient
  use quick_exception_module, only: RaiseException
  implicit double precision(a-h,o-z)

  character(len=1) cartsym(3)
  logical :: failed
  integer :: ierr ! temporarily adding ierr here, but error propagation must be
                  ! fixed soon

  ierr=0
  cartsym(1) = 'X'
  cartsym(2) = 'Y'
  cartsym(3) = 'Z'


  ! Finite difference hessian:  When you just can't solve the CPSCF.

  ! Now take a step in each of the cartesian directions, perform an
  ! scf cycle, calculate the gradient, and add that into the Hessian.
  ! Store everything (additional scf output) in the .cphf file. Note that this
  ! is a central difference finite difference.

  stepsize = 0.02D0

  do Iatom = 1,natom
     do Idirection = 1,3
        xyz(Idirection,Iatom) = xyz(Idirection,Iatom) + stepsize
        SAFE_CALL(getEnergy(.false.,ierr))

        if (quick_method%UNRST) then
            SAFE_CALL(oshell_gradient(ierr))
        else
            SAFE_CALL(cshell_gradient(ierr))
        endif

        Idest = (Iatom-1)*3 + Idirection
        do Iadd = 1,natom*3
           quick_qm_struct%hessian(Iadd,Idest) = quick_qm_struct%gradient(Iadd)
        enddo

        xyz(Idirection,Iatom) = xyz(Idirection,Iatom)-2.d0*stepsize
        SAFE_CALL(getEnergy(.false.,ierr))
        if (quick_method%UNRST) then
            SAFE_CALL(oshell_gradient(ierr))
        else
            SAFE_CALL(cshell_gradient(ierr))
        endif
        Idest = (Iatom-1)*3 + Idirection
        do Iadd = 1,natom*3
           quick_qm_struct%hessian(Iadd,Idest) =(quick_qm_struct%hessian(Iadd,Idest) - quick_qm_struct%gradient(Iadd)) &
                /(2.d0*stepsize)
        enddo

        ! Finally, return the coordinates to the original location.

        xyz(Idirection,Iatom) = xyz(Idirection,Iatom) + stepsize
     enddo
  enddo

end subroutine fdhessian




! Ed Brothers. October 22, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine HFHessian
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic
  use quick_cutoff_module, only: cshell_density_cutoff, cshell_dnscreen
  use quick_eri_fock1_cshell_module
 
  implicit double precision(a-h,o-z)
  ! dimension W(2*(maxbasis/2)**2,2*(maxbasis/2)**2),
  dimension itype2(3,2),ielecfld(3)
  allocatable W(:,:)
  character(len=1) cartsym(3)
    double precision, dimension(:), allocatable :: B0,BU
  double precision g_table(200),a,b
!  integer i,j,k,ii,jj,kk,g_count
  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, ntri
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

  cartsym(1) = 'X'
  cartsym(2) = 'Y'
  cartsym(3) = 'Z'

  ! The purpose of this subroutine is to calculate the 2nd derivative of
  ! the HF energy with respect to nuclear displacement.  The results
  ! of this are stored in Hessian, which is organized by atom and then
  ! by direction of displacement, i.e. element 1 is the gradient of the
  ! x diplacement of atom 1, element 5 is the y displacement of atom 2,
  ! and Hessian(1,5) is the d2E/dX(1)dY(2).

  ! Please note that there is only one set of code for restricted and
  ! unrestricted HF.

  ! Please also note the Hessian is symmetric.

  do I=1,3
     ielecfld(I)=0
  enddo
  do Iatm=1,natom*3
     do Jatm=1,natom*3
        quick_qm_struct%hessian(Jatm,Iatm)=0.d0
     enddo
  enddo


  !---------------------------------------------------------------------
  !  1) The second derivative of the nuclear repulsion.
  !---------------------------------------------------------------------

  call get_nuclear_repulsion_hessian

  !---------------------------------------------------------------------
  !  2) The second derivative of one electron term
  !---------------------------------------------------------------------

  call get_oneen_hessian

  !---------------------------------------------------------------------
  !  3) The second derivative of the electron repulsion term
  !---------------------------------------------------------------------

  call get_eri_hessian

  !---------------------------------------------------------------------
  !  4) If DFT, the second derivative of exchahnge correlation  term
  !---------------------------------------------------------------------

!  call get_xc_hessian

  !---------------------------------------------------------------------
  !  5) The CPHF part
  !---------------------------------------------------------------------

     call form_D1W1

  !---------------------------------------------------------------------
  !  5) FINAL HESSIAN CONTRIBUTION
  !---------------------------------------------------------------------

     call hess_total

end subroutine HFHessian


subroutine get_nuclear_repulsion_hessian
  use allmod
  implicit double precision(a-h,o-z)
  double precision g_table(200),a,b
  integer i,j,k,ii,jj,kk,g_count

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Note that the hessian is the sum of a series of term, each explained
  ! when we get to it.
  ! 1)  The 2nd derivative of the nuclear repulsion.

  ! We saw in the gradient code that:
  ! dVnn/dXA = ZA (Sum over B) ZB*(XB-XA) RAB^-3

  ! Using the same math, we can see that:

  ! dVnn/dXAdXA = ZA ZB ((3 (XA-XB)^2 RAB^-5) -  RAB^-3)
  ! dVnn/dXAdXB = ZA ZB ((-3 (XA-XB)^2 RAB^-5) +  RAB^-3)
  ! dVnn/dXAdYA = ZA ZB (3 (XA-XB)(YA-YB) RAB^-5)
  ! dVnn/dXAdYB = ZA ZB (-3 (XA-XB)(YA-YB) RAB^-5)

  ! An interesting fact is that d2Vnn/dXAdYB=-d2Vnn/dXAdYA.  Another
  ! intesting fact is d2Vnn/dXAdXA=d2Vnn/dXBdXB.  We use this
  ! in the next loop.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  do Iatm=1,natom
     do Jatm=Iatm+1,natom
        Istart = (Iatm-1)*3
        Jstart = (Jatm-1)*3
        XAminXB = xyz(1,Iatm)-xyz(1,Jatm)
        YAminYB = xyz(2,Iatm)-xyz(2,Jatm)
        ZAminZB = xyz(3,Iatm)-xyz(3,Jatm)
        temp = (XAminXB**2.d0 + YAminYB**2.d0 + ZAminZB**2.d0)
        RAB3 = temp**(-1.5d0)
        RAB5 = temp**(-2.5d0)
        ZA = quick_molspec%chg(Iatm)
        ZB = quick_molspec%chg(Jatm)
        RAB3 = ZA*ZB*RAB3
        RAB5 = ZA*ZB*RAB5

        temp = (3.d0*RAB5*XAminXB**2.d0-RAB3)
        quick_qm_struct%hessian(Istart+1,Istart+1) = quick_qm_struct%hessian(Istart+1,Istart+1)+temp
        quick_qm_struct%hessian(Jstart+1,Jstart+1) = quick_qm_struct%hessian(Jstart+1,Jstart+1)+temp
        quick_qm_struct%hessian(Jstart+1,Istart+1) = -temp

        temp = (3.d0*RAB5*YAminYB**2.d0-RAB3)
        quick_qm_struct%hessian(Istart+2,Istart+2) = quick_qm_struct%hessian(Istart+2,Istart+2)+temp
        quick_qm_struct%hessian(Jstart+2,Jstart+2) = quick_qm_struct%hessian(Jstart+2,Jstart+2)+temp
        quick_qm_struct%hessian(Jstart+2,Istart+2) = -temp

        temp = (3.d0*RAB5*ZAminZB**2.d0-RAB3)
        quick_qm_struct%hessian(Istart+3,Istart+3) = quick_qm_struct%hessian(Istart+3,Istart+3)+temp
        quick_qm_struct%hessian(Jstart+3,Jstart+3) = quick_qm_struct%hessian(Jstart+3,Jstart+3)+temp
        quick_qm_struct%hessian(Jstart+3,Istart+3) = -temp

        temp = (3.d0*RAB5*XAminXB*YAminYB)
        quick_qm_struct%hessian(Istart+2,Istart+1) = quick_qm_struct%hessian(Istart+2,Istart+1)+temp
        quick_qm_struct%hessian(Jstart+2,Jstart+1) = quick_qm_struct%hessian(Jstart+2,Jstart+1)+temp
        quick_qm_struct%hessian(Jstart+1,Istart+2) = -temp
        quick_qm_struct%hessian(Jstart+2,Istart+1) = -temp

        temp = (3.d0*RAB5*XAminXB*ZAminZB)
        quick_qm_struct%hessian(Istart+3,Istart+1) = quick_qm_struct%hessian(Istart+3,Istart+1)+temp
        quick_qm_struct%hessian(Jstart+3,Jstart+1) = quick_qm_struct%hessian(Jstart+3,Jstart+1)+temp
        quick_qm_struct%hessian(Jstart+1,Istart+3) = -temp
        quick_qm_struct%hessian(Jstart+3,Istart+1) = -temp

        temp = (3.d0*RAB5*YAminYB*ZAminZB)
        quick_qm_struct%hessian(Istart+3,Istart+2) = quick_qm_struct%hessian(Istart+3,Istart+2)+temp
        quick_qm_struct%hessian(Jstart+3,Jstart+2) = quick_qm_struct%hessian(Jstart+3,Jstart+2)+temp
        quick_qm_struct%hessian(Jstart+2,Istart+3) = -temp
        quick_qm_struct%hessian(Jstart+3,Istart+2) = -temp
     enddo
  enddo

  write(ioutfile,*)
  write(ioutfile,*)'The 2nd derivative of the nuclear repulsion'
  do Iatm=1,natom*3
     write(ioutfile,'(9(F7.4,7X))')(quick_qm_struct%hessian(Jatm,Iatm),Jatm=1,natom*3)
  enddo

  return

end subroutine get_nuclear_repulsion_hessian


subroutine get_oneen_hessian
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic, attrashellfock1
  use quick_cutoff_module, only: cshell_density_cutoff
  use quick_eri_cshell_module, only: getCshellEri

  implicit double precision(a-h,o-z)
  dimension itype2(3,2),ielecfld(3)
  double precision g_table(200),a,b
  integer i,j,k,ii,jj,kk,g_count
  logical :: ijcon

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! 2)  The negative of the energy weighted density matrix element i j
  ! with the second derivative of the ij overlap.

  ! 3)  The second derivative of the 1 electron kinetic energy term ij
  ! times the density matrix element ij.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  call get_ke_ovp_hessian

  write(ioutfile,*)
  write(ioutfile,*)'The 2nd derivative of the Overlap integral and KE'
  do Iatm=1,natom*3
     write(ioutfile,'(9(F7.4,7X))')(quick_qm_struct%hessian(Jatm,Iatm),Jatm=1,natom*3)
  enddo

     do Ibas=1,nbasis
        ISTART = (quick_basis%ncenter(Ibas)-1) *3
           do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis

              JSTART = (quick_basis%ncenter(Jbas)-1) *3

              DENSEJI = quick_qm_struct%dense(Jbas,Ibas)

  !  We have selecte two basis functions, now loop over angular momentum.
              do Imomentum=1,3

  !  do the Ibas derivatives first. In order to prevent code duplication,
  !  this has been implemented in a seperate subroutine. 
                 ijcon = .true.
                 call get_ijbas_fockderiv(Imomentum, Ibas, Jbas, Ibas, ISTART, ijcon, DENSEJI)

  !  do the Jbas derivatives.
                 ijcon = .false.
                 call get_ijbas_fockderiv(Imomentum, Ibas, Jbas, Jbas, JSTART, ijcon, DENSEJI)

              enddo
           enddo
     enddo

     ntri =  (nbasis*(nbasis+1))/2
 
     write(ioutfile,*)
     write(ioutfile,*)"  Derivative Fock after 1st-order Kinetic  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ",(I+2)/3
        do J = 1, ntri
!           do K= 1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%fd(I,J), &
           quick_qm_struct%fd(I+1,J),quick_qm_struct%fd(I+2,J)
!           enddo
        enddo
     enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! 4)  The second derivative of the 1 electron nuclear attraction term ij
  ! ij times the density matrix element ij.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  call get_attractshell_hessian

  write(ioutfile,*)
  write(ioutfile,*)'The 2nd derivative of the 1 electron nuclear attraction term'
  do Iatm=1,natom*3
     write(ioutfile,'(9(F7.4,7X))')(quick_qm_struct%hessian(Jatm,Iatm),Jatm=1,natom*3)
  enddo

     do IIsh = 1, jshell
        do JJsh = IIsh, jshell
           call attrashellfock1(IIsh,JJsh)
        enddo
     enddo

     write(ioutfile,*)
     write(ioutfile,*)"  Derivative Fock after 1st-order Ele-Nuc  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ", (I+2)/3
        do J = 1, ntri
!           do K=1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%fd(I,J), &
           quick_qm_struct%fd(I+1,J),quick_qm_struct%fd(I+2,J)
!           enddo
        enddo
     enddo

     write(ioutfile,*)
     write(ioutfile,*)"  1st-order Overlap  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ", (I+2)/3
        do J = 1, ntri
!           do K= 1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%od(I,J), &
           quick_qm_struct%od(I+1,J),quick_qm_struct%od(I+2,J)
!           enddo
        enddo
     enddo

  return

end subroutine get_oneen_hessian


subroutine get_xc_hessian
    use allmod
    use quick_method_module, only : quick_method
    use quick_gridpoints_module
    use xc_f90_types_m
    use xc_f90_lib_m
    implicit none

    integer :: iatm, ibas, ibin, icount, ifunc, igp, jbas, jcount, &
    ibasstart,irad_init,irad_end, ierror, imomentum
    double precision :: density, densityb, densitysum, dfdgaa, &
    df2dgaa2, dfdgab,dfdgbb, df2dgab2, dfdr, dfdrb, df2dr2, dphi2dx, &
    df2drdgaa,dphi2dy, dphi2dz, dphidx, dphidy, dphidz,gax, gay, gaz, gbx, &
    gby, gbz, gaa, gab, gbb, gridx, gridy, gridz, phi, phi2, &
    quicktest,sigma, sswt, temp, tempgx, tempgy, tempgz, &
    tsttmp_exc, tsttmp_vrhoa, tsttmp_vsigmaa, weight, xdot, &
    ydot, zdot, xiaodot, zkec, Ex, Ec, Eelxc,excpp, &
    xdotb, ydotb, zdotb, dxdx, dxdy, dxdz, dydy, dydz, dzdz

    double precision, dimension(3) :: phix
    double precision, dimension(6) :: phixx
    double precision, dimension(10) :: phixxx

    double precision, dimension(2) :: libxc_rho
    double precision, dimension(3) :: libxc_sigma
    double precision, dimension(1) :: libxc_exc
    double precision, dimension(2) :: libxc_vrho
    double precision, dimension(3) :: libxc_vsigma
    double precision, dimension(3) :: libxc_v2rho2
    double precision, dimension(6) :: libxc_v2rhosigma
    double precision, dimension(6) :: libxc_v2sigma2
    type(xc_f90_pointer_t), dimension(quick_method%nof_functionals) ::xc_func
    type(xc_f90_pointer_t), dimension(quick_method%nof_functionals) ::xc_info
    double precision :: tgrd(3), tsum(3)
    integer :: i,k,oi
    double precision,allocatable :: dp(:,:)
    double precision :: gwt(3,natom)
    double precision :: hwt(3,natom,3,natom)
    double precision :: VM, ValM, ValX
    double precision,allocatable :: DA(:), DM(:)
    double precision :: hess(3,natom,3,natom),fda(3,natom,nbasis*(nbasis+1)/2)
    double precision :: gden(3)
    double precision :: thrsh,wght

     allocate(DA(NBASIS*(NBASIS+1)/2)) 
     allocate(DM(NBASIS))

     fda=0.0d0
     hess=0.0d0
print*,'inside xc_hessian'

!     call formMaxDen(nbasis,quick_qm_struct%dense,DA,DM)

     if(quick_method%uselibxc) then
  !  Initiate the libxc functionals
        do ifunc=1, quick_method%nof_functionals
              call xc_f90_func_init(xc_func(ifunc), &
              xc_info(ifunc),quick_method%functional_id(ifunc),XC_UNPOLARIZED)
        enddo
     endif

        do Ibin=1, quick_dft_grid%nbins
print*,'Ibin:',Ibin
  !  Calculate the weight of the grid point in the SSW scheme.  If
  !  the grid point has a zero weight, we can skip it.

          Igp=quick_dft_grid%bin_counter(Ibin)+1

          do while(Igp < quick_dft_grid%bin_counter(Ibin+1)+1)
print*,'Igp:',Igp

             gridx=quick_dft_grid%gridxb(Igp)
             gridy=quick_dft_grid%gridyb(Igp)
             gridz=quick_dft_grid%gridzb(Igp)

             sswt=quick_dft_grid%gridb_sswt(Igp)
             weight=quick_dft_grid%gridb_weight(Igp)
             Iatm=quick_dft_grid%gridb_atm(Igp)

              if (weight < quick_method%DMCutoff ) then
                 continue
              else

  ! Form AO and derivative values at the grid point

                 ValM = 0.0d0
                 icount=quick_dft_grid%basf_counter(Ibin)+1
                 do while (icount < quick_dft_grid%basf_counter(Ibin+1)+1)
                    Ibas=quick_dft_grid%basf(icount)+1

!                    call pt3dr(gridx,gridy,gridz,phi,phix(1:3),&
!                    phixx(1:6),phixxx(1:10),Ibas,icount)

                    iao(Ibas)=phi
                    iaox(:,Ibas)=phix
                    iaoxx(:,Ibas)=phixx
                    iaoxxx(:,Ibas)=phixxx

                    ValX = MAX(Abs(IAO(I)),Abs(IAOX(1,I)),Abs(IAOX(2,I)), &
                           Abs(IAOX(3,I)),Abs(IAOXX(1,I)),Abs(IAOXX(2,I)), &
                           Abs(IAOXX(3,I)),Abs(IAOXX(4,I)),Abs(IAOXX(5,I)), &
                           Abs(IAOXX(6,I)))

                    icount=icount+1
                    If(ValX.GT.ValM) ValM = ValX
                 enddo
                 VM=ValM

  !  evaluate the densities at the grid point and the gradient at that grid
  !  point 

                 call denspt_cshell(gridx,gridy,gridz,density,densityb,&
                 gax,gay,gaz,gbx,gby,gbz,Ibin)

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

                    excpp=0.0d0
                    dfdr=0.0d0
                    dfdrb=0.0d0

                    dfdgaa=0.0d0
                    dfdgab=0.0d0
                    dfdgbb=0.0d0

                    if(quick_method%uselibxc) then


                       do ifunc=1, quick_method%nof_functionals
                          select case(xc_f90_info_family(xc_info(ifunc)))
                             case(XC_FAMILY_LDA)
                                call xc_f90_lda_exc(xc_func(ifunc),1,&
                                libxc_rho(1),libxc_exc(1))
                                call xc_f90_lda_vxc_fxc(xc_func(ifunc),1,&
                                libxc_rho(1),libxc_vrho(1),libxc_v2rho2(1))
                                libxc_vsigma(1) = 0.0d0
                                libxc_vsigma(2) = 0.0d0
                                libxc_vsigma(3) = 0.0d0
                                libxc_v2sigma2(1) = 0.0d0
                                libxc_v2sigma2(2) = 0.0d0
                                libxc_v2sigma2(3) = 0.0d0
                                libxc_v2sigma2(4) = 0.0d0
                                libxc_v2sigma2(5) = 0.0d0
                                libxc_v2sigma2(6) = 0.0d0
                                libxc_v2rhosigma(1) = 0.0d0
                                libxc_v2rhosigma(2) = 0.0d0
                                libxc_v2rhosigma(3) = 0.0d0
                                libxc_v2rhosigma(4) = 0.0d0
                                libxc_v2rhosigma(5) = 0.0d0
                                libxc_v2rhosigma(6) = 0.0d0
                             case(XC_FAMILY_GGA, XC_FAMILY_HYB_GGA)
                                call xc_f90_gga_exc(xc_func(ifunc),1,&
                                libxc_rho(1),libxc_sigma(1),libxc_exc(1))
                                call xc_f90_gga_vxc_fxc(xc_func(ifunc),1,&
                                libxc_rho(1), libxc_sigma(1),&
                                libxc_vrho(1),libxc_vsigma(1),libxc_v2rho2(1),& 
                                libxc_v2rhosigma(1),libxc_v2sigma2(1))
                          end select

                        excpp=excpp+libxc_exc(1)
                        dfdr=dfdr+libxc_vrho(1)
                        dfdgaa=dfdgaa+libxc_vsigma(1)
                        df2dr2=df2dr2+libxc_v2rho2(1)
                        df2drdgaa=df2drdgaa+libxc_v2rhosigma(1)
                        df2dgaa2=df2dgaa2+libxc_v2sigma2(1)

                       enddo

                       zkec=(density+densityb)*excpp

                       xdot = 4.0d0*dfdgaa*gax
                       ydot = 4.0d0*dfdgaa*gay
                       zdot = 4.0d0*dfdgaa*gaz

!                       call ssw2der(gridx,gridy,gridz,iatm,natom,xyz(1:3,1:natom),zkec,gwt,hwt)     
!                       gden(1) = gax
!                       gden(2) = gay
!                       gden(3) = gaz
!                       thrsh = quick_method%maxIntegralCutoff
!                       wght = weight/sswt
!                       call formFdHess(nbasis,natom,thrsh,DA,DM,gden,wght, &
!                                libxc_vrho(1),libxc_v2rho2(1),libxc_v2rho2(2), &
!                                libxc_vsigma(1),libxc_vsigma(2), &
!                                libxc_v2rhosigma(1),libxc_v2rhosigma(3),libxc_v2rhosigma(2), &
!                                libxc_v2sigma2(1),libxc_v2sigma2(3), &
!                                libxc_v2sigma2(2),libxc_v2sigma2(4), &
!                                Ibin,iao,iaox,iaoxx,iaoxxx,VM,iatm,gwt,hwt,fda,hess)

                    endif
                 endif
              endif
        Igp=Igp+1
        enddo
     enddo

     if(quick_method%uselibxc) then
  !  Uninitilize libxc functionals
        do ifunc=1, quick_method%nof_functionals
           call xc_f90_func_end(xc_func(ifunc))
        enddo
     endif


end subroutine get_xc_hessian

subroutine hess_total
  use allmod
  implicit real*8 (a-h,o-z)
!---------------------------------------------------------------------
!  calculates two contributions (last two) to the final hessian
!  tr(d1*f1) - tr(w1*s1)
!
!  makes the full hessian matrix out of its upper triangle
!---------------------------------------------------------------------

  ntri=nbasis*(nbasis+1)/2

!  do IAT=1,NATOMS
!     do J=1,3
!        do K=1,ntri
!           quick_qm_struct%fd(3*(IAT-1)+J,K)=quick_qm_struct%fd(3*(IAT-1)+J,K) &
!                               +quick_qm_struct%fd1g0(3*(IAT-1)+J,K)
!        enddo 
!     enddo
!  enddo

  call calc_d1f1_w1s1(natom,ntri,nbasis,quick_qm_struct%dense1, &
                     quick_qm_struct%fd,quick_qm_struct%wdens1, &
                     quick_qm_struct%od,quick_qm_struct%hessian)

  call hess_full(quick_qm_struct%hessian,natom)

end subroutine hess_total

subroutine calc_d1f1_w1s1(natoms,ntri, ncf, &
                         den1,fock1,wen1,over1,hess)
      implicit real*8 (a-h,o-z)
!---------------------------------------------------------------------
!  calculates two contributions (last two) to the final hessian
!  tr(d1*f1) - tr(w1*s1)
!---------------------------------------------------------------------
! Input :
!
! natoms  - number of atoms
! ntri    - ncf*(ncf+1)/2
! ncf     - basis set dimension
!  den1() - Ist-order density matrix
! fock1() - Ist-order fock matrix
!  wen1() _ Ist-order weighted density
! over1() - Ist-order overlap matrix
!
! Input/Output  :
!
! hess()  - final hessian
!---------------------------------------------------------------------
      dimension den1(3*natoms,ntri),fock1(3*natoms,ntri)
      dimension wen1(3*natoms,ntri),over1(3*natoms,ntri)
      dimension hess(3*natoms,3*natoms)
!-----------------------------------------------------------------
! Atom=Btom
!
      do iat=1,natoms
         do ixyz=1,3
            call spur(den1(3*(IAT-1)+ixyz,:),fock1(3*(IAT-1)+ixyz,:),ncf,df1)
            call spur(wen1(3*(IAT-1)+ixyz,:),over1(3*(IAT-1)+ixyz,:),ncf,ws1)
            dewe=df1-ws1
            hess(3*(IAT-1)+ixyz,3*(IAT-1)+ixyz)=hess(3*(IAT-1)+ixyz,3*(IAT-1)+ixyz)+dewe
            do jxyz=ixyz+1,3
                call spur(den1(3*(IAT-1)+ixyz,:),fock1(3*(IAT-1)+jxyz,:),ncf,df1)
                call spur(den1(3*(IAT-1)+jxyz,:),fock1(3*(IAT-1)+ixyz,:),ncf,fd1)
                call spur(wen1(3*(IAT-1)+ixyz,:),over1(3*(IAT-1)+jxyz,:),ncf,ws1)
                call spur(wen1(3*(IAT-1)+jxyz,:),over1(3*(IAT-1)+ixyz,:),ncf,sw1)
                df=df1+fd1
                ws=ws1+sw1
                dewe=df-ws
                dewe= dewe*0.5d0
                hess(3*(IAT-1)+jxyz,3*(IAT-1)+ixyz)=hess(3*(IAT-1)+jxyz,3*(IAT-1)+ixyz)+dewe
            enddo
         enddo
      enddo
!
! different atoms :
!
      do iat=1,natoms
         do jat=iat+1,natoms
            do ixyz=1,3
               do jxyz=1,3
                  call spur(den1(3*(IAT-1)+ixyz,:),fock1(3*(JAT-1)+jxyz,:),ncf,df1)
                  call spur(den1(3*(JAT-1)+jxyz,:),fock1(3*(IAT-1)+ixyz,:),ncf,fd1)
                  call spur(wen1(3*(IAT-1)+ixyz,:),over1(3*(JAT-1)+jxyz,:),ncf,ws1)
                  call spur(wen1(3*(JAT-1)+jxyz,:),over1(3*(IAT-1)+ixyz,:),ncf,sw1)
                  df=df1+fd1
                  ws=ws1+sw1
                  dewe=df-ws
                  dewe= dewe*0.5d0
                  hess(3*(JAT-1)+jxyz,3*(IAT-1)+ixyz)=hess(3*(JAT-1)+jxyz,3*(IAT-1)+ixyz)+dewe
               enddo
            enddo
         enddo
      enddo

end subroutine calc_d1f1_w1s1

!======================================================================
!
      subroutine hess_full(hess,na)
      implicit real*8 (a-h,o-z)
      dimension hess(3*na,3*na)
!
      do i=1,na
         do ixyz=1,3
            do j=i,na
               if(j.eq.i) then
                  do jxyz=ixyz,3
                     hess(3*(i-1)+ixyz,3*(j-1)+jxyz)=hess(3*(j-1)+jxyz,3*(i-1)+ixyz)
                  enddo
               else
                  do jxyz=1,3
                     hess(3*(i-1)+ixyz,3*(j-1)+jxyz)=hess(3*(j-1)+jxyz,3*(i-1)+ixyz)
                  enddo
               endif
            enddo
         enddo
      enddo
!
      end
!======================================================================
SUBROUTINE MATDIAG(A,N,VM,V,D,IErr)
IMPLICIT DOUBLE PRECISION (A-H,O-Z)
INTEGER i4err

!  Diagonalizes a Real Symmetric matrix A by Householder
!  reduction to Tridiagonal form
!
!  ARGUMENTS
!
!  A     -  input matrix
!           on exit contains eigenvectors
!  N     -  dimension of A
!  VM    -  scratch space for eigenvector ordering (N*N)
!  V     -  scratch space (N)
!  D     -  on exit contains eigenvalues

DIMENSION A(N,N),VM(N,N),V(N),D(N),S(10)

If(N.GT.3) Then
  call dsyev('V','U',N,A,N,D,VM,N*N,i4err)
Else
  call dsyev('V','U',N,A,N,D,S,10,i4err)
EndIf

IERR=int(i4err)
RETURN
END SUBROUTINE
!=============================================================================
      subroutine spur (A,B,n,s)
!  This subroutine calculates the trace of the product of two symmetrical
!  matrices, stored in triangular form (upper triangle column-wise)
!  Arguments
!  INTENT(IN)
!  A, B = two symmatrical matrices, stored as n*(n+1)/2 long arrays
!         indexing: (ij)=i*(i-1)/2+j where i>=j
!  n    = dimension
!  INTENT(OUT)
!  s = Trace(A*B)
!  Spur is German for Trace
!
      implicit real*8 (a-h,o-z)
      parameter(two=2.0d0)
      dimension A(*), B(*)
      ntri=n*(n+1)/2
      s=ddot(ntri,A,1,B,1)*two
      ii=0
      do i=1,n
        ii=ii+i
        s=s-A(ii)*B(ii)
      end do
      end
!==============================================================

subroutine get_ke_ovp_hessian
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic
  implicit double precision(a-h,o-z)
  dimension itype2(3,2),ielecfld(3)
  logical :: ijcon
  double precision g_table(200),a,b
  integer i,j,k,ii,jj,kk,g_count

  ! First we need to form the energy weighted density matrix.

  ! The energy weighted denisty matrix fora a restricted calculation is:
  ! Q(i,j) =2*(Sum over alpha electrons a)  E(a) C(I,a) C(J,a)
  ! Where C is the alpha or beta molecular orbital coefficients, and
  ! E is the alpha or beta molecular orbital energies.
  ! We'll store this in HOLD as we don't really need it except breifly.

#ifdef OSHELL
  do I=1,nbasis
    do J=1,nbasis
      HOLDJI = 0.d0
      ! Alpha part
      do K=1,quick_molspec%nelec
        HOLDJI = HOLDJI + (quick_qm_struct%E(K)*quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))
      enddo
      ! Beta part
      do K=1,quick_molspec%nelecb
        HOLDJI = HOLDJI + (quick_qm_struct%EB(K)*quick_qm_struct%cob(J,K)*quick_qm_struct%cob(I,K))
      enddo
      quick_scratch%hold(J,I) = HOLDJI
    enddo
  enddo

#else

  do I=1,nbasis
    do J=1,nbasis
      HOLDJI = 0.d0
      do K=1,quick_molspec%nelec/2
        HOLDJI = HOLDJI + (quick_qm_struct%E(K)*quick_qm_struct%co(J,K)*quick_qm_struct%co(I,K))
      enddo
      quick_scratch%hold(J,I) = 2.d0*HOLDJI
    enddo
  enddo
#endif

  ! Before we begin this, a quick note on the second derivative of Gaussian
  ! orbitals.  If there is only one center, the second derivative is zero.
  ! (As with the first derivative.)  If there are two or more centers, there
  ! are three possibilities.

  ! d/dXA d/dXA ((x-XA)^i (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! = 4a^2((x-XA)^(i+2) (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! - 2a(2i+1)((x-XA)^(i) (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! + (i-1)i((x-XA)^(i-2) (y-YA)^j (z-ZA)^k e^(-ar^2))

  ! d/dXA d/dYA ((x-XA)^i (y-YA)^j (z-ZA)^k e^(-ar^2))
  ! = 4a^2 ((x-XA)^(i+1)(y-YA)^(j+1)(z-ZA)^k e^(-ar^2))
  ! - 2ai ((x-XA)^(i-1)(y-YA)^(j+1)(z-ZA)^k e^(-ar^2))
  ! - 2aj ((x-XA)^(i+1)(y-YA)^(j-1)(z-ZA)^k e^(-ar^2))
  ! -  ij ((x-XA)^(i-1)(y-YA)^(j-1)(z-ZA)^k e^(-ar^2))

  ! d/dXA d/dXB gtfonA gtfonB = dgtfonA/dXA gtfonB + gtfonA dgtfonB/dXB

  ! Note the final case is explained in the gradient code, as it is just a
  ! sum of first derivatives.

  do Ibas=1,nbasis
    ISTART = (quick_basis%ncenter(Ibas)-1) *3
    do Jbas=quick_basis%last_basis_function(quick_basis%ncenter(IBAS))+1,nbasis
      JSTART = (quick_basis%ncenter(Jbas)-1) *3
        DENSEJI = quick_qm_struct%dense(Jbas,Ibas)
        if(quick_method%unrst) DENSEJI = DENSEJI+quick_qm_struct%denseb(Jbas,Ibas)

  !  We have selected our two basis functions, now loop over angular momentum.
      do Imomentum=1,3
  
  !  First,calculate the d^2/dXA^2 type terms.
  !  Do the Ibas derivatives first. In order to prevent code duplication,
  !  this has been implemented in a seperate subroutine. 
        ijcon = .true.
        call get_ijbas_deriv_hessian(Imomentum, Ibas, Jbas, Ibas, ISTART, ijcon, DENSEJI)

  !  Do the Jbas derivatives.
        ijcon = .false.
        call get_ijbas_deriv_hessian(Imomentum, Ibas, Jbas, Jbas, JSTART, ijcon, DENSEJI)

  ! Now we are going to do derivatives of the d2/dXAdYA type.  Note
  ! that we are still in the IMOMENTUM loop.
        do Imomentum2=Imomentum+1,3
  !  Do the Ibas derivatives first.
          ijcon = .true.
          call get_ijbas2_deriv_hessian(Imomentum, Imomentum2, Ibas, Jbas, Ibas, ISTART, ijcon, DENSEJI) 

  !  Do the Jbas derivatives.
          ijcon = .false.
          call get_ijbas2_deriv_hessian(Imomentum, Imomentum2, Ibas, Jbas, Jbas, JSTART, ijcon, DENSEJI)
        enddo

  ! The last part is the d^2/dXAdYB portion.  Note that we are still
  ! inside the IMOMENTUM loop.
        do Jmomentum=1,3
          call get_bas_deriv_hessian(Imomentum, Jmomentum, Ibas, Jbas, ISTART, JSTART, DENSEJI)
        enddo

  ! Quick note:  The three loops closed here are Imomentum,Jbas, and Ibas.
      enddo
    enddo
  enddo

  return

end subroutine get_ke_ovp_hessian


subroutine get_ijbas_deriv_hessian(Imomentum, Ibas, Jbas, mbas, mstart, ijcon, DENSEJI)
  use allmod
  use quick_overlap_module, only: gpt, opf, overlap
  use quick_oei_module, only: ekinetic
  implicit double precision(a-h,o-z)
  logical :: ijcon
  double precision g_table(200), a,b
  integer i,j,k,ii,jj,kk,g_count

  d2SI = 0.d0
  d2KEI = 0.d0

  Ax = xyz(1,quick_basis%ncenter(Jbas))
  Bx = xyz(1,quick_basis%ncenter(Ibas))
  Ay = xyz(2,quick_basis%ncenter(Jbas))
  By = xyz(2,quick_basis%ncenter(Ibas))
  Az = xyz(3,quick_basis%ncenter(Jbas))
  Bz = xyz(3,quick_basis%ncenter(Ibas))

  itype(Imomentum,mbas) = itype(Imomentum,mbas)+2
  i = itype(1,Jbas)
  j = itype(2,Jbas)
  k = itype(3,Jbas)
  ii = itype(1,Ibas)
  jj = itype(2,Ibas)
  kk = itype(3,Ibas)
  g_count = i+ii+j+jj+k+kk+2

  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        b = aexp(Icon,Ibas)
        a = aexp(Jcon,Jbas)
        if(ijcon) then
            mcon = Icon
        else
            mcon = Jcon
        endif

        call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

        d2SI = d2SI + 4.d0*aexp(mcon,mbas)*aexp(mcon,mbas) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)               

        d2KEI = d2KEI + 4.d0*aexp(mcon,mbas)*aexp(mcon,mbas) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
     enddo
  enddo
  itype(Imomentum,mbas) = itype(Imomentum,mbas)-2

  i = itype(1,Jbas)
  j = itype(2,Jbas)
  k = itype(3,Jbas)
  ii = itype(1,Ibas)
  jj = itype(2,Ibas)
  kk = itype(3,Ibas)
  g_count = i+ii+j+jj+k+kk+2

  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        b = aexp(Icon,Ibas)
        a = aexp(Jcon,Jbas)
        if(ijcon) then
            mcon = Icon
        else
            mcon = Jcon
        endif

        call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

        d2SI = d2SI - 2.d0*aexp(mcon,mbas) &
             *(1.d0+2.d0*dble(itype(Imomentum,mbas))) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
        d2KEI = d2KEI - 2.d0*aexp(mcon,mbas) &
             *(1.d0+2.d0*dble(itype(Imomentum,mbas))) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
     enddo
  enddo

  if (itype(Imomentum,mbas) >= 2) then
     const = dble(itype(Imomentum,mbas)) &
          *dble(itype(Imomentum,mbas)-1)
     itype(Imomentum,mbas) = itype(Imomentum,mbas)-2
     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2
     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SI = d2SI + const* &
                dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEI = d2KEI + const* &
                dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
             *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
        enddo
     enddo
     itype(Imomentum,mbas) = itype(Imomentum,mbas)+2
  endif
  quick_qm_struct%hessian(mstart+Imomentum,mstart+Imomentum) = &
       quick_qm_struct%hessian(mstart+Imomentum,mstart+Imomentum) &
       -d2SI*quick_scratch%hold(Jbas,Ibas)*2.d0 &
       +d2KeI*DENSEJI*2.d0

  return

end subroutine get_ijbas_deriv_hessian

subroutine get_ijbas2_deriv_hessian(Imomentum, Imomentum2, Ibas, Jbas, mbas, mstart, ijcon, DENSEJI)
  use allmod
  use quick_overlap_module, only: gpt, opf, overlap
  use quick_oei_module, only: ekinetic
  implicit double precision(a-h,o-z)
  logical :: ijcon
  double precision g_table(200), a,b
  integer i,j,k,ii,jj,kk,g_count

  d2SI = 0.d0
  d2KEI = 0.d0

  Ax = xyz(1,quick_basis%ncenter(Jbas))
  Bx = xyz(1,quick_basis%ncenter(Ibas))
  Ay = xyz(2,quick_basis%ncenter(Jbas))
  By = xyz(2,quick_basis%ncenter(Ibas))
  Az = xyz(3,quick_basis%ncenter(Jbas))
  Bz = xyz(3,quick_basis%ncenter(Ibas))

  itype(Imomentum,mbas) = itype(Imomentum,mbas)+1
  itype(Imomentum2,mbas) = itype(Imomentum2,mbas)+1
  i = itype(1,Jbas)
  j = itype(2,Jbas)
  k = itype(3,Jbas)
  ii = itype(1,Ibas)
  jj = itype(2,Ibas)
  kk = itype(3,Ibas)
  g_count = i+ii+j+jj+k+kk+2

  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        b = aexp(Icon,Ibas)
        a = aexp(Jcon,Jbas)
        if(ijcon) then
            mcon = Icon
        else
            mcon = Jcon
        endif

        call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

        d2SI = d2SI + 4.d0*aexp(mcon,mbas)*aexp(mcon,mbas) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
        d2KEI = d2KEI + 4.d0*aexp(mcon,mbas)*aexp(mcon,mbas) &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
     enddo
  enddo
  itype(Imomentum,mbas) = itype(Imomentum,mbas)-1
  itype(Imomentum2,mbas) = itype(Imomentum2,mbas)-1

  if (itype(Imomentum,mbas) /= 0) then
     const = dble(itype(Imomentum,mbas))
     itype(Imomentum,mbas) = itype(Imomentum,mbas)-1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)+1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)
           if(ijcon) then
               mcon = Icon
           else
               mcon = Jcon
           endif

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SI = d2SI - 2.d0*aexp(mcon,mbas)*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
           *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEI = d2KEI - 2.d0*aexp(mcon,mbas)*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,mbas) = itype(Imomentum,mbas)+1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)-1
  endif

  if (itype(Imomentum2,mbas) /= 0) then
     const = dble(itype(Imomentum2,mbas))
     itype(Imomentum,mbas) = itype(Imomentum,mbas)+1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)-1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)
           if(ijcon) then
               mcon = Icon
           else
               mcon = Jcon
           endif

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SI = d2SI - 2.d0*aexp(mcon,mbas)*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEI = d2KEI - 2.d0*aexp(mcon,mbas)*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,mbas) = itype(Imomentum,mbas)-1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)+1
  endif

  if (itype(Imomentum2,mbas) /= 0 .and. &
       itype(Imomentum,mbas) /= 0) then
     const = dble(itype(Imomentum2,mbas))* &
          dble(itype(Imomentum,mbas))
     itype(Imomentum,mbas) = itype(Imomentum,mbas)-1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)-1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SI = d2SI +const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEI = d2KEI + const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,mbas) = itype(Imomentum,mbas)+1
     itype(Imomentum2,mbas) = itype(Imomentum2,mbas)+1
  endif

  quick_qm_struct%hessian(mstart+Imomentum2,mstart+Imomentum) = &
       quick_qm_struct%hessian(mstart+Imomentum2,mstart+Imomentum) &
       -d2SI*quick_scratch%hold(Jbas,Ibas)*2.d0 &
       +d2KeI*DENSEJI*2.d0

  return

end subroutine get_ijbas2_deriv_hessian

subroutine get_bas_deriv_hessian(Imomentum, Jmomentum, Ibas, Jbas, ISTART, JSTART, DENSEJI)
  use allmod
  use quick_overlap_module, only: gpt, opf, overlap
  use quick_oei_module, only: ekinetic
  implicit double precision(a-h,o-z)
  logical :: ijcon
  double precision g_table(200), a,b
  integer i,j,k,ii,jj,kk,g_count

  d2SIJ = 0.d0
  d2KEIJ = 0.d0

  Ax = xyz(1,quick_basis%ncenter(Jbas))
  Bx = xyz(1,quick_basis%ncenter(Ibas))
  Ay = xyz(2,quick_basis%ncenter(Jbas))
  By = xyz(2,quick_basis%ncenter(Ibas))
  Az = xyz(3,quick_basis%ncenter(Jbas))
  Bz = xyz(3,quick_basis%ncenter(Ibas))

  itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
  itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)+1

  i = itype(1,Jbas)
  j = itype(2,Jbas)
  k = itype(3,Jbas)
  ii = itype(1,Ibas)
  jj = itype(2,Ibas)
  kk = itype(3,Ibas)
  g_count = i+ii+j+jj+k+kk+2

  do Icon=1,ncontract(Ibas)
     do Jcon=1,ncontract(Jbas)
        b = aexp(Icon,Ibas)
        a = aexp(Jcon,Jbas)

        call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

        d2SIJ = d2SIJ + 4.d0*a*b &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        d2KEIJ = d2KEIJ + 4.d0*a*b &
             *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

     enddo
  enddo
  itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
  itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)-1

  if (itype(Jmomentum,Jbas) /= 0) then
     const = dble(itype(Jmomentum,Jbas))
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)-1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)

           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SIJ = d2SIJ - 2.d0*b*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
           *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEIJ = d2KEIJ - 2.d0*b*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)+1
  endif

  if (itype(Imomentum,Ibas) /= 0) then
     const = dble(itype(Imomentum,Ibas))
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)+1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)

           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

           d2SIJ = d2SIJ - 2.d0*a*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEIJ = d2KEIJ - 2.d0*a*const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)-1
  endif

  if (itype(Imomentum,Ibas) /= 0 .and. &
       itype(Jmomentum,Jbas) /= 0) then
     const = dble(itype(Imomentum,Ibas))* &
          dble(itype(Jmomentum,Jbas))
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)-1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)-1

     i = itype(1,Jbas)
     j = itype(2,Jbas)
     k = itype(3,Jbas)
     ii = itype(1,Ibas)
     jj = itype(2,Ibas)
     kk = itype(3,Ibas)
     g_count = i+ii+j+jj+k+kk+2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           b = aexp(Icon,Ibas)
           a = aexp(Jcon,Jbas)

           call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

          d2SIJ = d2SIJ +const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
           *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

           d2KEIJ = d2KEIJ +const &
                *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
          *ekinetic(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)

        enddo
     enddo
     itype(Imomentum,Ibas) = itype(Imomentum,Ibas)+1
     itype(Jmomentum,Jbas) = itype(Jmomentum,Jbas)+1
  endif

  ! Now we add the contribution to the Hessian array.

  quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) = &
       quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) &
       -d2SIJ*quick_scratch%hold(Jbas,Ibas)*2.d0 &
       +d2KeIJ*DENSEJI*2.d0

  return

end subroutine get_bas_deriv_hessian

subroutine get_attractshell_hessian
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic
  implicit double precision(a-h,o-z)
  dimension itype2(3,2),ielecfld(3)
  double precision g_table(200), a,b
  integer i,j,k,ii,jj,kk,g_count

  do I=1,3
     ielecfld(I)=0
  enddo

  do Ibas=1,nbasis
     iA=quick_basis%ncenter(Ibas)
     ISTART = (iA-1)*3
     do I=1,3
        itype2(I,1) = itype(I,Ibas)
     enddo

     do Jbas=Ibas,nbasis
        iB = quick_basis%ncenter(Jbas)
        JSTART = (iB-1)*3
        do I=1,3
           itype2(I,2) = itype(I,Jbas)
        enddo

        do iC = 1,natom
           iCSTART = (iC-1)*3

           ! As before, if all terms are on the same atom, they move with the
           ! atom and the derivative is zero.

           if (iA == iC .and. iB == iC) then
              continue
           else
              DENSEJI=quick_qm_struct%dense(Jbas,Ibas)
              if (quick_method%unrst) DENSEJI = DENSEJI+quick_qm_struct%denseb(Jbas,Ibas)
              if (Ibas /= Jbas) DENSEJI=2.d0*DENSEJI
              chgC = quick_molspec%chg(iC)

              ! First, take the second derivative of the center the electron is being
              ! attracted to.  These are the d2/dCXdCY and d2/dCX2 derivatives.


              do ICmomentum=1,3
                 ielecfld(ICmomentum) =  ielecfld(ICmomentum)+1
                 do JCmomentum=ICmomentum,3
                    ielecfld(JCmomentum) =  ielecfld(JCmomentum)+1
                    d2NAC=0.d0
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2NAC= d2NAC + dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype(1,Jbas),itype(2,Jbas),itype(3,Jbas), &
                               itype(1,Ibas),itype(2,Ibas),itype(3,Ibas), &
                               ielecfld(1),ielecfld(2),ielecfld(3), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    quick_qm_struct%hessian(ICstart+JCmomentum,ICstart+ICmomentum) = &
                         quick_qm_struct%hessian(ICstart+JCmomentum,ICstart+ICmomentum)+ &
                         DENSEJI*d2NAC
                    ielecfld(JCmomentum) =  ielecfld(JCmomentum)-1
                 enddo
 
                 ! Now calculate the derivatives of the type d2/dCXdIbasCenterX. This is
                 ! basically moving the attracting center and one of the centers that a
                 ! basis function is on.
 
                 ! This is where we begin using the itype2 array.  If Ibas=Jbas and
                 ! we adjust the angular momentum of Ibas in itype, we also inadvertently
                 ! adjust the angular momentum of Jbas.  This leads to large errors.  Thus
                 ! we use a dummy array.
 
                 ! Note we are still in the ICmomentum loop.  First, loop over Ibas
                 ! momentums.
 
                 do IImomentum = 1,3
                    d2NAC=0.d0
                    itype2(IImomentum,1) = itype2(IImomentum,1)+1
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2NAC= d2NAC + 2.d0*aexp(Icon,Ibas)* &
                               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               ielecfld(1),ielecfld(2),ielecfld(3), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(IImomentum,1) = itype2(IImomentum,1)-1
 
                    if (itype2(IImomentum,1) /= 0) then
                       const = dble(itype2(IImomentum,1))
                       itype2(IImomentum,1) = itype2(IImomentum,1)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2NAC= d2NAC -const* &
                                  dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  ielecfld(1),ielecfld(2),ielecfld(3), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(IImomentum,1) = itype2(IImomentum,1)+1
                    endif
 
                    if (iA == iC .and. ICmomentum == IImomentum) &
                         d2NAC=d2NAC*2.d0
                    if (ICstart+ICmomentum >= Istart+IImomentum) then
                       quick_qm_struct%hessian(ICstart+ICmomentum,Istart+IImomentum) = &
                            quick_qm_struct%hessian(ICstart+ICmomentum,Istart+IImomentum)+ &
                            DENSEJI*d2NAC
                    else
                       quick_qm_struct%hessian(Istart+IImomentum,ICstart+ICmomentum) = &
                            quick_qm_struct%hessian(Istart+IImomentum,ICstart+ICmomentum)+ &
                            DENSEJI*d2NAC
                    endif
                 enddo
 
                 ! Now loop over Jbas momentums.
 
                 do JJmomentum = 1,3
                    d2NAC=0.d0
                    itype2(JJmomentum,2) = itype2(JJmomentum,2)+1
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2NAC= d2NAC + 2.d0*aexp(Jcon,Jbas)* &
                               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               ielecfld(1),ielecfld(2),ielecfld(3), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(JJmomentum,2) = itype2(JJmomentum,2)-1
 
                    if (itype2(JJmomentum,2) /= 0) then
                       const = dble(itype2(JJmomentum,2))
                       itype2(JJmomentum,2) = itype2(JJmomentum,2)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2NAC= d2NAC -const* &
                                  dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *electricfld(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  ielecfld(1),ielecfld(2),ielecfld(3), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(JJmomentum,2) = itype2(JJmomentum,2)+1
                    endif
 
                    if (iB == iC .and. ICmomentum == JJmomentum) &
                         d2NAC=d2NAC*2.d0
                    if (ICstart+ICmomentum >= Jstart+JJmomentum) then
                       quick_qm_struct%hessian(ICstart+ICmomentum,Jstart+JJmomentum) = &
                            quick_qm_struct%hessian(ICstart+ICmomentum,Jstart+JJmomentum)+ &
                            DENSEJI*d2NAC
                    else
                       quick_qm_struct%hessian(Jstart+JJmomentum,ICstart+ICmomentum) = &
                            quick_qm_struct%hessian(Jstart+JJmomentum,ICstart+ICmomentum)+ &
                            DENSEJI*d2NAC
                    endif
 
                 enddo
                 ielecfld(ICmomentum) =  ielecfld(ICmomentum)-1
              enddo
 
              ! Please note we have exited all inner loops at this point and are only
              ! inside the Ibas,Jbas,IC loop here.
 
              ! At this point we have found all of the elements of the hessian that
              ! involve the attractive center.  Now we perturb the atoms on which the
              ! basis functions lie.  This is exactly analogous to what was done with
              ! the two center integrals above in 2) and 3) with one exception.(d2/dXAdYB)
              ! First,calculate the d^2/dXA^2 type terms.
 
              do Imomentum=1,3
                 d2AI = 0.d0
                 d2AJ =0.d0
 
                 ! do the Ibas derivatives first.

                 itype2(Imomentum,1) = itype2(Imomentum,1)+2
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       d2AI = d2AI + 4.d0*aexp(Icon,Ibas)*aexp(Icon,Ibas) &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype2(1,2),itype2(2,2),itype2(3,2), &
                            itype2(1,1),itype2(2,1),itype2(3,1), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                    enddo
                 enddo
                 itype2(Imomentum,1) = itype2(Imomentum,1)-2

                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       d2AI = d2AI - 2.d0*aexp(Icon,Ibas) &
                            *(1.d0+2.d0*dble(itype2(Imomentum,1))) &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype2(1,2),itype2(2,2),itype2(3,2), &
                            itype2(1,1),itype2(2,1),itype2(3,1), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                    enddo
                 enddo


                 if (itype2(Imomentum,1) >= 2) then
                    const = dble(itype2(Imomentum,1)) &
                         *dble(itype2(Imomentum,1)-1)
                    itype2(Imomentum,1) = itype2(Imomentum,1)-2
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2AI = d2AI + const* &
                               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(Imomentum,1) = itype2(Imomentum,1)+2
                 endif
                 quick_qm_struct%hessian(ISTART+Imomentum,ISTART+Imomentum) = &
                      quick_qm_struct%hessian(ISTART+Imomentum,ISTART+Imomentum) &
                      +d2AI*DENSEJI

                 ! Now do the Jbas derivatives.

                 itype2(Imomentum,2) = itype2(Imomentum,2)+2
                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       d2AJ = d2AJ + 4.d0*aexp(Jcon,Jbas)*aexp(Jcon,Jbas) &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype2(1,2),itype2(2,2),itype2(3,2), &
                            itype2(1,1),itype2(2,1),itype2(3,1), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                    enddo
                 enddo
                 itype2(Imomentum,2) = itype2(Imomentum,2)-2

                 do Icon=1,ncontract(Ibas)
                    do Jcon=1,ncontract(Jbas)
                       d2AJ = d2AJ - 2.d0*aexp(Jcon,Jbas) &
                            *(1.d0+2.d0*dble(itype2(Imomentum,2))) &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                            itype2(1,2),itype2(2,2),itype2(3,2), &
                            itype2(1,1),itype2(2,1),itype2(3,1), &
                            xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                            xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                            xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                    enddo
                 enddo


                 if (itype2(Imomentum,2) >= 2) then
                    const = dble(itype2(Imomentum,2)) &
                         *dble(itype2(Imomentum,2)-1)
                    itype2(Imomentum,2) = itype2(Imomentum,2)-2
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2AJ = d2AJ + const* &
                               dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(Imomentum,2) = itype2(Imomentum,2)+2
                 endif
                 quick_qm_struct%hessian(JSTART+Imomentum,JSTART+Imomentum) = &
                      quick_qm_struct%hessian(JSTART+Imomentum,JSTART+Imomentum) &
                      +d2AJ*DENSEJI


                 ! Now we are going to do derivatives of the d2/dXAdYA type. Note that
                 ! we are still in the IMOMENTUM loop.

                 do Imomentum2=Imomentum+1,3
                    d2AI = 0.d0
                    d2AJ = 0.d0

                    ! do the Ibas derivatives first.

                   itype2(Imomentum,1) = itype2(Imomentum,1)+1
                    itype2(Imomentum2,1) = itype2(Imomentum2,1)+1
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2AI = d2AI + 4.d0*aexp(Icon,Ibas)*aexp(Icon,Ibas) &
                               *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(Imomentum,1) = itype2(Imomentum,1)-1
                    itype2(Imomentum2,1) = itype2(Imomentum2,1)-1

                    if (itype2(Imomentum,1) /= 0) then
                       const = dble(itype2(Imomentum,1))
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)+1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AI = d2AI - 2.d0*aexp(Icon,Ibas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)-1
                    endif

                    if (itype2(Imomentum2,1) /= 0) then
                       const = dble(itype2(Imomentum2,1))
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AI = d2AI - 2.d0*aexp(Icon,Ibas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)+1
                    endif

                    if (itype2(Imomentum2,1) /= 0 .and. &
                         itype2(Imomentum,1) /= 0) then
                       const = dble(itype2(Imomentum2,1))* &
                            dble(itype2(Imomentum,1))
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AI = d2AI +const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Imomentum2,1) = itype2(Imomentum2,1)+1
                    endif

                    ! Now do the Jbas derivatives.

                    itype2(Imomentum,2) = itype2(Imomentum,2)+1
                    itype2(Imomentum2,2) = itype2(Imomentum2,2)+1
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2AJ = d2AJ + 4.d0*aexp(Jcon,Jbas)*aexp(Jcon,Jbas) &
                               *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(Imomentum,2) = itype2(Imomentum,2)-1
                    itype2(Imomentum2,2) = itype2(Imomentum2,2)-1

                    if (itype2(Imomentum,2) /= 0) then
                       const = dble(itype2(Imomentum,2))
                       itype2(Imomentum,2) = itype2(Imomentum,2)-1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)+1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AJ = d2AJ - 2.d0*aexp(Jcon,Jbas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,2) = itype2(Imomentum,2)+1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)-1
                    endif

                    if (itype2(Imomentum2,2) /= 0) then
                       const = dble(itype2(Imomentum2,2))
                       itype2(Imomentum,2) = itype2(Imomentum,2)+1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AJ = d2AJ - 2.d0*aexp(Jcon,Jbas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,2) = itype2(Imomentum,2)-1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)+1
                    endif

                    if (itype2(Imomentum2,2) /= 0 .and. &
                         itype2(Imomentum,2) /= 0) then
                       const = dble(itype2(Imomentum2,2))* &
                            dble(itype2(Imomentum,2))
                       itype2(Imomentum,2) = itype2(Imomentum,2)-1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AJ = d2AJ + const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,2) = itype2(Imomentum,2)+1
                       itype2(Imomentum2,2) = itype2(Imomentum2,2)+1
                    endif

                    ! Now add the contributions to the Hessian Array.

                    quick_qm_struct%hessian(ISTART+Imomentum2,ISTART+Imomentum) = &
                         quick_qm_struct%hessian(ISTART+Imomentum2,ISTART+Imomentum) &
                         +d2AI*DENSEJI
                    quick_qm_struct%hessian(JSTART+Imomentum2,JSTART+Imomentum) = &
                         quick_qm_struct%hessian(JSTART+Imomentum2,JSTART+Imomentum) &
                         +d2AJ*DENSEJI
                 enddo

                 ! Close the Imomentum2 loop.

                 ! The last part is the d^2/dXAdYB portion.  Note that we are still
                 ! inside the IMOMENTUM loop.



                 do Jmomentum=1,3
                    d2AIJ = 0.d0

                    itype2(Imomentum,1) = itype2(Imomentum,1)+1
                    itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
                    do Icon=1,ncontract(Ibas)
                       do Jcon=1,ncontract(Jbas)
                          d2AIJ = d2AIJ + 4.d0*aexp(Icon,Ibas)*aexp(Jcon,Jbas) &
                               *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                               *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                               itype2(1,2),itype2(2,2),itype2(3,2), &
                               itype2(1,1),itype2(2,1),itype2(3,1), &
                               xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                               xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                               xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                       enddo
                    enddo
                    itype2(Imomentum,1) = itype2(Imomentum,1)-1
                    itype2(Jmomentum,2) = itype2(Jmomentum,2)-1

                    if (itype2(Jmomentum,2) /= 0) then
                       const = dble(itype2(Jmomentum,2))
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AIJ = d2AIJ - 2.d0*aexp(Icon,Ibas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
                    endif

                    if (itype2(Imomentum,1) /= 0) then
                       const = dble(itype2(Imomentum,1))
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AIJ = d2AIJ - 2.d0*aexp(Jcon,Jbas)*const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
                    endif

                    if (itype2(Imomentum,1) /= 0 .and. &
                         itype2(Jmomentum,2) /= 0) then
                       const = dble(itype2(Imomentum,1))* &
                            dble(itype2(Jmomentum,2))
                       itype2(Imomentum,1) = itype2(Imomentum,1)-1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
                       do Icon=1,ncontract(Ibas)
                          do Jcon=1,ncontract(Jbas)
                             d2AIJ = d2AIJ +const &
                                  *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                                  *attraction(aexp(Jcon,Jbas),aexp(Icon,Ibas), &
                                  itype2(1,2),itype2(2,2),itype2(3,2), &
                                  itype2(1,1),itype2(2,1),itype2(3,1), &
                                  xyz(1,iB),xyz(2,iB),xyz(3,iB), &
                                  xyz(1,iA),xyz(2,iA),xyz(3,iA), &
                                  xyz(1,iC),xyz(2,iC),xyz(3,iC), chgC)
                          enddo
                       enddo
                       itype2(Imomentum,1) = itype2(Imomentum,1)+1
                       itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
                    endif

                    ! Now we add the contribution to the Hessian array.

                    if (iA /= iB) then
                       quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) = &
                            quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) &
                            +d2AIJ*DENSEJI
                    else
                       if (Imomentum == Jmomentum) then
                          quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) = &
                               quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) &
                               +2.d0*d2AIJ*DENSEJI
                       ELSEIF (Jmomentum > Imomentum) then
                          quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) = &
                               quick_qm_struct%hessian(JSTART+Jmomentum,ISTART+Imomentum) &
                               +d2AIJ*DENSEJI
                       else
                          quick_qm_struct%hessian(ISTART+Imomentum,JSTART+Jmomentum) = &
                               quick_qm_struct%hessian(ISTART+Imomentum,JSTART+Jmomentum) &
                               +d2AIJ*DENSEJI
                       endif
                    endif

                    ! Now close the Jmomentum and Imomentum loops.

                 enddo
              enddo

           endif
        enddo
     enddo
  enddo

  return

end subroutine get_attractshell_hessian

subroutine get_eri_hessian
  use allmod
  use quick_overlap_module, only: gpt, overlap
  use quick_oei_module, only: ekinetic
  use quick_cutoff_module, only: cshell_density_cutoff, cshell_dnscreen
  use quick_eri_fock1_cshell_module

  implicit double precision(a-h,o-z)
  dimension itype2(3,2),ielecfld(3)
  double precision g_table(200), a,b
  integer i,j,k,g_count
  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2, ntri
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

  ! 5)  The 2nd derivative of the 4center 2e- terms with respect to X times
  ! the coefficient found in the energy. (i.e. the multiplicative
  ! constants from the density matrix that arise as these are both
  ! the exchange and correlation integrals.)


  do I=1,nbasis
     ! Set some variables to reduce access time for some of the more
     ! used quantities.

     DENSEII=quick_qm_struct%dense(I,I)
     if (quick_method%unrst) DENSEII=DENSEII+quick_qm_struct%denseb(I,I)

     ! Neglect all the (ii|ii) integrals, as they move with the core.

     do J=I+1,nbasis
        ! Set some variables to reduce access time for some of the more
        ! used quantities. (AGAIN)

        DENSEJI=quick_qm_struct%dense(J,I)
        DENSEJJ=quick_qm_struct%dense(J,J)
        if (quick_method%unrst) then
           DENSEJI=DENSEJI+quick_qm_struct%denseb(J,I)
           DENSEJJ=DENSEJJ+quick_qm_struct%denseb(J,J)
        endif

        ! Find  all the (ii|jj) integrals.

        constant = (DENSEII*DENSEJJ-.5d0*DENSEJI*DENSEJI)

        call hess2elec(I,I,J,J,constant)

        ! Find  all the (ij|jj) integrals.

        constant =  DENSEJJ*DENSEJI
        call hess2elec(I,J,J,J,constant)


        ! Find  all the (ii|ij) integrals.
        constant= DENSEJI*DENSEII
        call hess2elec(I,I,I,J,constant)

        ! Find all the (ij|ij) integrals
        constant =(1.5d0*DENSEJI*DENSEJI-0.50d0*DENSEJJ*DENSEII)
        call hess2elec(I,J,I,J,constant)

        do K=J+1,nbasis
           ! Set some variables to reduce access time for some of the more
           ! used quantities. (AGAIN)

           DENSEKI=quick_qm_struct%dense(K,I)
           DENSEKJ=quick_qm_struct%dense(K,J)
           DENSEKK=quick_qm_struct%dense(K,K)
           if (quick_method%unrst) then
              DENSEKI=DENSEKI+quick_qm_struct%denseb(K,I)
              DENSEKJ=DENSEKJ+quick_qm_struct%denseb(K,J)
              DENSEKK=DENSEKK+quick_qm_struct%denseb(K,K)
           endif

           ! Find all the (ij|ik) integrals where j>i,k>j

           constant = (3.0d0*DENSEJI*DENSEKI-DENSEKJ*DENSEII)
           call hess2elec(I,J,I,K,constant)

           ! Find all the (ij|kk) integrals where j>i, k>j.

           constant=(2.d0*DENSEJI*DENSEKK-DENSEKI*DENSEKJ)
           call hess2elec(I,J,K,K,constant)

           ! Find all the (ik|jj) integrals where j>i, k>j.

           constant= (2.d0*DENSEKI*DENSEJJ-DENSEKJ*DENSEJI)
           call hess2elec(I,K,J,J,constant)

           ! Find all the (ii|jk) integrals where j>i, k>j.

           constant = (2.d0*DENSEKJ*DENSEII-DENSEJI*DENSEKI)
           call hess2elec(I,I,J,K,constant)
        enddo

        do K=I+1,nbasis-1
           DENSEKI=quick_qm_struct%dense(K,I)
           DENSEKJ=quick_qm_struct%dense(K,J)
           DENSEKK=quick_qm_struct%dense(K,K)
           if (quick_method%unrst) then
              DENSEKI=DENSEKI+quick_qm_struct%denseb(K,I)
              DENSEKJ=DENSEKJ+quick_qm_struct%denseb(K,J)
              DENSEKK=DENSEKK+quick_qm_struct%denseb(K,K)
           endif

           do L=K+1,nbasis
              DENSELJ=quick_qm_struct%dense(L,J)
              DENSELI=quick_qm_struct%dense(L,I)
              DENSELK=quick_qm_struct%dense(L,K)
              if (quick_method%unrst) then
                 DENSELJ=DENSELJ+quick_qm_struct%denseb(L,J)
                 DENSELI=DENSELI+quick_qm_struct%denseb(L,I)
                 DENSELK=DENSELK+quick_qm_struct%denseb(L,K)
              endif

              ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
              ! can be equal.

              constant = (4.d0*DENSEJI*DENSELK-DENSEKI*DENSELJ &
                   -DENSELI*DENSEKJ)
              call hess2elec(I,J,K,L,constant)

           enddo
        enddo
     enddo
  enddo

  write(ioutfile,*)
  write(ioutfile,*)'The 2nd derivative of the 4center 2e- terms'
  do Iatm=1,natom*3
     write(ioutfile,'(9(F7.4,7X))')(quick_qm_struct%hessian(Jatm,Iatm),Jatm=1,natom*3)
  enddo

    call cshell_density_cutoff

     do II=1,jshell
        do JJ=II,jshell
        Testtmp=Ycutoff(II,JJ)
           do KK=II,jshell
              do LL=KK,jshell
                 if(quick_basis%katom(II).eq.quick_basis%katom(JJ).and.quick_basis%katom(II).eq. &
                 quick_basis%katom(KK).and.quick_basis%katom(II).eq.quick_basis%katom(LL))then
                    continue
                 else
                    testCutoff = TESTtmp*Ycutoff(KK,LL)
                    if(testCutoff.gt.quick_method%gradCutoff)then
                       DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
                       cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
                       cutoffTest=testCutoff*DNmax
                       if(cutoffTest.gt.quick_method%gradCutoff)then
                          call eri_fock1_cshell
                       endif
                    endif
                 endif
              enddo
           enddo
        enddo
     enddo

         ntri =  (nbasis*(nbasis+1))/2

     write(ioutfile,*)
     write(ioutfile,*)"  Derivative Fock after G(D0,g1)  "
     write(ioutfile,*)"     X             Y             Z "
     do I = 1, natom*3, 3
        write(ioutfile,*)" Atom no : ", (I+2)/3
        do J = 1, ntri
!           do K= 1, nbasis
           write(ioutfile,'(i3,2X,3(F9.6,7X))')J,quick_qm_struct%fd(I,J), &
           quick_qm_struct%fd(I+1,J),quick_qm_struct%fd(I+2,J)
!           enddo
        enddo
     enddo

  return

end subroutine get_eri_hessian

! Ed Brothers. November 5, 2002.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine hess2elec(Ibas,Jbas,IIbas,JJbas,coeff)
  use allmod
  implicit double precision(a-h,o-z)

  dimension itype2(3,4)
  logical :: same
  same = .false.

  ! The purpose of this subroutine is to calculate the second derivative of
  ! the 2-electron 4-center integrals and add them into the total hessian.
  ! This requires the use of a multiplicitave constant (coeff) which arises
  ! by the way the integral enters the total energy. For example, (ij|ij) is
  ! a repulsion integral and an exchange integral, thus it enters the energy
  ! with a coefficient of (1.5d0*DENSEJI*DENSEJI-0.50d0*DENSEJJ*DENSEII).


  ! First, find the centers the basis functions are located on.  If all the
  ! functions are on the same center, return as this is a zero result.

  iA = quick_basis%ncenter(Ibas)
  iB = quick_basis%ncenter(Jbas)
  iC = quick_basis%ncenter(IIbas)
  iD = quick_basis%ncenter(JJbas)

  same = iA.eq.iB
  same = same .and. iB.eq.iC
  same = same .and. iC.eq.iD

  if (same) return

  iAstart = (iA-1)*3
  iBstart = (iB-1)*3
  iCstart = (iC-1)*3
  iDstart = (iD-1)*3

  ! The itype2 array was added because if Ibas=Jbas, the code raises two
  ! angular momentums instead of one.

  do Imomentum=1,3
     itype2(Imomentum,1) = itype(Imomentum,Ibas)
     itype2(Imomentum,2) = itype(Imomentum,Jbas)
     itype2(Imomentum,3) = itype(Imomentum,IIbas)
     itype2(Imomentum,4) = itype(Imomentum,JJbas)
  enddo

  ! The first thing to calculate is the d2/dXA elements of the hessian.

  do Imomentum=1,3
     D2I = 0.d0
     D2J = 0.d0
     D2II = 0.d0
     D2JJ = 0.d0

     ! Ibas' center first.

     itype2(Imomentum,1) = itype2(Imomentum,1)+2
     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2I = d2I + 4.d0*aexp(Icon,Ibas)*aexp(Icon,Ibas) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo
     itype2(Imomentum,1) = itype2(Imomentum,1)-2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2I = d2I - 2.d0*aexp(Icon,Ibas) &
                      *(1.d0+2.d0*dble(itype2(Imomentum,1))) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo


     if (itype2(Imomentum,1) >= 2) then
        const = dble(itype2(Imomentum,1)) &
             *dble(itype2(Imomentum,1)-1)
        itype2(Imomentum,1) = itype2(Imomentum,1)-2
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2I = d2I + const* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,1) = itype2(Imomentum,1)+2
     endif
     quick_qm_struct%hessian(iASTART+Imomentum,iASTART+Imomentum) = &
          quick_qm_struct%hessian(iASTART+Imomentum,iASTART+Imomentum) &
          +d2I*coeff

     ! Jbas' center.

     itype2(Imomentum,2) = itype2(Imomentum,2)+2
     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2J = d2J + 4.d0*aexp(Jcon,Jbas)*aexp(Jcon,Jbas) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo
     itype2(Imomentum,2) = itype2(Imomentum,2)-2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2J = d2J - 2.d0*aexp(Jcon,Jbas) &
                      *(1.d0+2.d0*dble(itype2(Imomentum,2))) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo


     if (itype2(Imomentum,2) >= 2) then
        const = dble(itype2(Imomentum,2)) &
             *dble(itype2(Imomentum,2)-1)
        itype2(Imomentum,2) = itype2(Imomentum,2)-2
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2J = d2J + const* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,2) = itype2(Imomentum,2)+2
     endif
     quick_qm_struct%hessian(iBSTART+Imomentum,iBSTART+Imomentum) = &
          quick_qm_struct%hessian(iBSTART+Imomentum,iBSTART+Imomentum) &
          +d2J*coeff
     ! IIbas' center.

     itype2(Imomentum,3) = itype2(Imomentum,3)+2
     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2II = d2II + 4.d0*aexp(IIcon,IIbas)*aexp(IIcon,IIbas) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo
     itype2(Imomentum,3) = itype2(Imomentum,3)-2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2II = d2II - 2.d0*aexp(IIcon,IIbas) &
                      *(1.d0+2.d0*dble(itype2(Imomentum,3))) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo


     if (itype2(Imomentum,3) >= 2) then
        const = dble(itype2(Imomentum,3)) &
             *dble(itype2(Imomentum,3)-1)
        itype2(Imomentum,3) = itype2(Imomentum,3)-2
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2II = d2II + const* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,3) = itype2(Imomentum,3)+2
     endif
     quick_qm_struct%hessian(iCSTART+Imomentum,iCSTART+Imomentum) = &
          quick_qm_struct%hessian(iCSTART+Imomentum,iCSTART+Imomentum) &
          +d2II*coeff

     ! JJbas' center.

     itype2(Imomentum,4) = itype2(Imomentum,4)+2
     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2JJ = d2JJ + 4.d0*aexp(JJcon,JJbas)*aexp(JJcon,JJbas) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo
     itype2(Imomentum,4) = itype2(Imomentum,4)-2

     do Icon=1,ncontract(Ibas)
        do Jcon=1,ncontract(Jbas)
           do IIcon=1,ncontract(IIbas)
              do JJcon=1,ncontract(JJbas)
                 d2JJ = d2JJ - 2.d0*aexp(JJcon,JJbas) &
                      *(1.d0+2.d0*dble(itype2(Imomentum,4))) &
                      *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                      *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                      *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                      aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                      itype2(1,1), itype2(2,1), itype2(3,1), &
                      itype2(1,2), itype2(2,2), itype2(3,2), &
                      itype2(1,3), itype2(2,3), itype2(3,3), &
                      itype2(1,4), itype2(2,4), itype2(3,4), &
                      xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                      xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                      xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                      xyz(1,iD), xyz(2,iD),xyz(3,iD))
              enddo
           enddo
        enddo
     enddo


     if (itype2(Imomentum,4) >= 2) then
        const = dble(itype2(Imomentum,4)) &
             *dble(itype2(Imomentum,4)-1)
        itype2(Imomentum,4) = itype2(Imomentum,4)-2
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2JJ = d2JJ + const* &
                         dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,4) = itype2(Imomentum,4)+2
     endif
     quick_qm_struct%hessian(iDSTART+Imomentum,iDSTART+Imomentum) = &
          quick_qm_struct%hessian(iDSTART+Imomentum,iDSTART+Imomentum) &
          +d2JJ*coeff


     ! Now do the d^2E/dXAdXB type integrals.

     do Jmomentum=Imomentum+1,3
        d2I=0.d0
        d2J=0.d0
        d2II=0.d0
        d2JJ=0.d0

        ! Start  with Ibas

        itype2(Imomentum,1) = itype2(Imomentum,1)+1
        itype2(Jmomentum,1) = itype2(Jmomentum,1)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2I = d2I + 4.d0*aexp(Icon,Ibas)*aexp(Icon,Ibas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,1) = itype2(Imomentum,1)-1
        itype2(Jmomentum,1) = itype2(Jmomentum,1)-1

        if (itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2I = d2I - 2.d0*aexp(Icon,Ibas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)-1
        endif

        if (itype2(Jmomentum,1) /= 0) then
           const=dble(itype2(Jmomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2I = d2I - 2.d0*aexp(Icon,Ibas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)+1
        endif

        if (itype2(Jmomentum,1) /= 0 .and. itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Jmomentum,1)*itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2I = d2I +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,1) = itype2(Jmomentum,1)+1
        endif

        quick_qm_struct%hessian(iASTART+Jmomentum,iASTART+Imomentum) = &
             quick_qm_struct%hessian(iASTART+Jmomentum,iASTART+Imomentum) &
             +d2I*coeff

        ! Now do Jbas.

        itype2(Imomentum,2) = itype2(Imomentum,2)+1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2J = d2J + 4.d0*aexp(Jcon,Jbas)*aexp(Jcon,Jbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,2) = itype2(Imomentum,2)-1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)-1

        if (itype2(Imomentum,2) /= 0) then
           const=dble(itype2(Imomentum,2))
           itype2(Imomentum,2) = itype2(Imomentum,2)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2J = d2J - 2.d0*aexp(Jcon,Jbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,2) = itype2(Imomentum,2)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
        endif

        if (itype2(Jmomentum,2) /= 0) then
           const=dble(itype2(Jmomentum,2))
           itype2(Imomentum,2) = itype2(Imomentum,2)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2J = d2J - 2.d0*aexp(Jcon,Jbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,2) = itype2(Imomentum,2)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif

        if (itype2(Jmomentum,2) /= 0 .and. itype2(Imomentum,2) /= 0) then
           const=dble(itype2(Jmomentum,2)*itype2(Imomentum,2))
           itype2(Imomentum,2) = itype2(Imomentum,2)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2J = d2J +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,2) = itype2(Imomentum,2)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif

        quick_qm_struct%hessian(iBSTART+Jmomentum,iBSTART+Imomentum) = &
             quick_qm_struct%hessian(iBSTART+Jmomentum,iBSTART+Imomentum) &
             +d2J*coeff


        ! do IIbas

        itype2(Imomentum,3) = itype2(Imomentum,3)+1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2II = d2II + 4.d0*aexp(IIcon,IIbas)*aexp(IIcon,IIbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,3) = itype2(Imomentum,3)-1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)-1

        if (itype2(Imomentum,3) /= 0) then
           const=dble(itype2(Imomentum,3))
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2II = d2II - 2.d0*aexp(IIcon,IIbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
        endif

        if (itype2(Jmomentum,3) /= 0) then
           const=dble(itype2(Jmomentum,3))
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2II = d2II - 2.d0*aexp(IIcon,IIbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif

        if (itype2(Jmomentum,3) /= 0 .and. itype2(Imomentum,3) /= 0) then
           const=dble(itype2(Jmomentum,3)*itype2(Imomentum,3))
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2II = d2II +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif

        quick_qm_struct%hessian(iCSTART+Jmomentum,iCSTART+Imomentum) = &
             quick_qm_struct%hessian(iCSTART+Jmomentum,iCSTART+Imomentum) &
             +d2II*coeff

        ! Now do JJbas.

        itype2(Imomentum,4) = itype2(Imomentum,4)+1
        itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2JJ = d2JJ + 4.d0*aexp(JJcon,JJbas)*aexp(JJcon,JJbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,4) = itype2(Imomentum,4)-1
        itype2(Jmomentum,4) = itype2(Jmomentum,4)-1

        if (itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2JJ = d2JJ - 2.d0*aexp(JJcon,JJbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
        endif

        if (itype2(Jmomentum,4) /= 0) then
           const=dble(itype2(Jmomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2JJ= d2JJ - 2.d0*aexp(JJcon,JJbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        endif

        if (itype2(Jmomentum,4) /= 0 .and. itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Jmomentum,4)*itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2JJ = d2JJ +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        endif

        quick_qm_struct%hessian(iDSTART+Jmomentum,iDSTART+Imomentum) = &
             quick_qm_struct%hessian(iDSTART+Jmomentum,iDSTART+Imomentum) &
             +d2JJ*coeff

     enddo

     ! We have just closed the Jmomentum loop.
     ! Now we need to calculate the d2E/dX1dY2 type terms.  This requires 6
     ! different combinations.

     do Jmomentum=1,3

        ! do the Ibas with Jbas first.

        d2XY = 0.d0

        itype2(Imomentum,1) = itype2(Imomentum,1)+1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(Icon,Ibas)*aexp(Jcon,Jbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,1) = itype2(Imomentum,1)-1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)-1

        if (itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Jcon,Jbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
        endif

        if (itype2(Jmomentum,2) /= 0) then
           const=dble(itype2(Jmomentum,2))
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Icon,Ibas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif

        if (itype2(Jmomentum,2) /= 0 .and. itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Jmomentum,2)*itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif
        if (iA /= iB) then
           if (iB > iA) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

        ! Ibas with IIbas.

        d2XY = 0.d0

        itype2(Imomentum,1) = itype2(Imomentum,1)+1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(Icon,Ibas)*aexp(IIcon,IIbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,1) = itype2(Imomentum,1)-1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)-1

        if (itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(IIcon,IIbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
        endif

        if (itype2(Jmomentum,3) /= 0) then
           const=dble(itype2(Jmomentum,3))
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Icon,Ibas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif

        if (itype2(Jmomentum,3) /= 0 .and. itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Jmomentum,3)*itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif
        if (iA /= iC) then
           if (iC > iA) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iCSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iCSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iCSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iCSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

        ! Ibas with JJbas.

        d2XY = 0.d0

        itype2(Imomentum,1) = itype2(Imomentum,1)+1
        itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(Icon,Ibas)*aexp(JJcon,JJbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,1) = itype2(Imomentum,1)-1
        itype2(Jmomentum,4) = itype2(Jmomentum,4)-1

        if (itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(JJcon,JJbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
        endif

        if (itype2(Jmomentum,4) /= 0) then
           const=dble(itype2(Jmomentum,4))
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Icon,Ibas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        endif

        if (itype2(Jmomentum,4) /= 0 .and. itype2(Imomentum,1) /= 0) then
           const=dble(itype2(Jmomentum,4)*itype2(Imomentum,1))
           itype2(Imomentum,1) = itype2(Imomentum,1)-1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,1) = itype2(Imomentum,1)+1
           itype2(Jmomentum,4) = itype2(Jmomentum,4)+1
        endif
        if (iA /= iD) then
           if (iD > iA) then
              quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iDSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iDSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) = &
                   quick_qm_struct%hessian(iDSTART+Jmomentum,iASTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iASTART+Imomentum,iDSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iASTART+Imomentum,iDSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

        ! IIbas with Jbas.

        d2XY = 0.d0

        itype2(Imomentum,3) = itype2(Imomentum,3)+1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(IIcon,IIbas)*aexp(Jcon,Jbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,3) = itype2(Imomentum,3)-1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)-1

        if (itype2(Imomentum,3) /= 0) then
           const=dble(itype2(Imomentum,3))
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Jcon,Jbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
        endif

        if (itype2(Jmomentum,2) /= 0) then
           const=dble(itype2(Jmomentum,2))
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(IIcon,IIbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif

        if (itype2(Jmomentum,2) /= 0 .and. itype2(Imomentum,3) /= 0) then
           const=dble(itype2(Jmomentum,2)*itype2(Imomentum,3))
           itype2(Imomentum,3) = itype2(Imomentum,3)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,3) = itype2(Imomentum,3)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif
        if (iC /= iB) then
           if (iB > iC) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iCSTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iCSTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iCSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iCSTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iCSTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

        ! JJbas with Jbas.

        d2XY = 0.d0

        itype2(Imomentum,4) = itype2(Imomentum,4)+1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(JJcon,JJbas)*aexp(Jcon,Jbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,4) = itype2(Imomentum,4)-1
        itype2(Jmomentum,2) = itype2(Jmomentum,2)-1

        if (itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(Jcon,Jbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
        endif

        if (itype2(Jmomentum,2) /= 0) then
           const=dble(itype2(Jmomentum,2))
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(JJcon,JJbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif

        if (itype2(Jmomentum,2) /= 0 .and. itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Jmomentum,2)*itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,2) = itype2(Jmomentum,2)+1
        endif
        if (iD /= iB) then
           if (iB > iD) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iDSTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iDSTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iBSTART+Jmomentum,iDSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iDSTART+Imomentum,iBSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iDSTART+Imomentum,iBSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

        ! JJbas with IIbas.

        d2XY = 0.d0

        itype2(Imomentum,4) = itype2(Imomentum,4)+1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        do Icon=1,ncontract(Ibas)
           do Jcon=1,ncontract(Jbas)
              do IIcon=1,ncontract(IIbas)
                 do JJcon=1,ncontract(JJbas)
                    d2XY = d2XY + 4.d0*aexp(JJcon,JJbas)*aexp(IIcon,IIbas) &
                         *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                         *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                         *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                         aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                         itype2(1,1), itype2(2,1), itype2(3,1), &
                         itype2(1,2), itype2(2,2), itype2(3,2), &
                         itype2(1,3), itype2(2,3), itype2(3,3), &
                         itype2(1,4), itype2(2,4), itype2(3,4), &
                         xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                         xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                         xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                         xyz(1,iD), xyz(2,iD),xyz(3,iD))
                 enddo
              enddo
           enddo
        enddo
        itype2(Imomentum,4) = itype2(Imomentum,4)-1
        itype2(Jmomentum,3) = itype2(Jmomentum,3)-1

        if (itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(IIcon,IIbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
        endif

        if (itype2(Jmomentum,3) /= 0) then
           const=dble(itype2(Jmomentum,3))
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY - 2.d0*aexp(JJcon,JJbas)*const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif

        if (itype2(Jmomentum,3) /= 0 .and. itype2(Imomentum,4) /= 0) then
           const=dble(itype2(Jmomentum,3)*itype2(Imomentum,4))
           itype2(Imomentum,4) = itype2(Imomentum,4)-1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)-1
           do Icon=1,ncontract(Ibas)
              do Jcon=1,ncontract(Jbas)
                 do IIcon=1,ncontract(IIbas)
                    do JJcon=1,ncontract(JJbas)
                       d2XY = d2XY +const &
                            *dcoeff(Jcon,Jbas)*dcoeff(Icon,Ibas) &
                            *dcoeff(JJcon,JJbas)*dcoeff(IIcon,IIbas) &
                            *repulsion_prim(aexp(Icon,Ibas),aexp(Jcon,Jbas), &
                            aexp(IIcon,IIbas),aexp(JJcon,JJbas), &
                            itype2(1,1), itype2(2,1), itype2(3,1), &
                            itype2(1,2), itype2(2,2), itype2(3,2), &
                            itype2(1,3), itype2(2,3), itype2(3,3), &
                            itype2(1,4), itype2(2,4), itype2(3,4), &
                            xyz(1,iA), xyz(2,iA),xyz(3,iA), &
                            xyz(1,iB), xyz(2,iB),xyz(3,iB), &
                            xyz(1,iC), xyz(2,iC),xyz(3,iC), &
                            xyz(1,iD), xyz(2,iD),xyz(3,iD))
                    enddo
                 enddo
              enddo
           enddo
           itype2(Imomentum,4) = itype2(Imomentum,4)+1
           itype2(Jmomentum,3) = itype2(Jmomentum,3)+1
        endif
        if (iD /= iC) then
           if (iC > iD) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iDSTART+Imomentum,iCSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iDSTART+Imomentum,iCSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        else
           if (Imomentum == Jmomentum) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) &
                   +2.d0*d2XY*coeff
           ELSEIF (Jmomentum > Imomentum) then
              quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) = &
                   quick_qm_struct%hessian(iCSTART+Jmomentum,iDSTART+Imomentum) &
                   +d2XY*coeff
           else
              quick_qm_struct%hessian(iDSTART+Imomentum,iCSTART+Jmomentum) = &
                   quick_qm_struct%hessian(iDSTART+Imomentum,iCSTART+Jmomentum) &
                   +d2XY*coeff
           endif
        endif

     enddo
  enddo
  return
end subroutine hess2elec


subroutine attrashellhess(IIsh,JJsh)
  use allmod
  use quick_overlap_module, only: opf, overlap
  implicit double precision(a-h,o-z)
  dimension aux(0:20)
  double precision AA(3),BB(3),CC(3),PP(3)
  common /xiaoattra/attra,aux,AA,BB,CC,PP,g

  double precision RA(3),RB(3),RP(3), valopf, g_table(200)

  Ax=xyz(1,quick_basis%katom(IIsh))
  Ay=xyz(2,quick_basis%katom(IIsh))
  Az=xyz(3,quick_basis%katom(IIsh))

  Bx=xyz(1,quick_basis%katom(JJsh))
  By=xyz(2,quick_basis%katom(JJsh))
  Bz=xyz(3,quick_basis%katom(JJsh))

  ! The purpose of this subroutine is to calculate the nuclear attraction
  ! of an electron  distributed between gtfs with orbital exponents a
  ! and b on A and B with angular momentums defined by i,j,k (a's x, y
  ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
  ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
  ! arises from the recusion relationship.

  ! The this is taken from the recursive relation found in Obara and Saika,
  ! J. Chem. Phys. 84 (7) 1986, 3963.

  ! The first step is generating all the necessary auxillary integrals.
  ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
  ! The values of m range from 0 to i+j+k+ii+jj+kk.

  NII2=quick_basis%Qfinal(IIsh)
  NJJ2=quick_basis%Qfinal(JJsh)
  Maxm=NII2+NJJ2+1+1

  do ips=1,quick_basis%kprim(IIsh)
     a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
     do jps=1,quick_basis%kprim(JJsh)
        b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))

        valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
        quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)

        if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then

          g = a+b
          Px = (a*Ax + b*Bx)/g
          Py = (a*Ay + b*By)/g
          Pz = (a*Az + b*Bz)/g
          g_table = g**(-1.5)

          constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
                * 2.d0 * sqrt(g/Pi)

          do iatom=1,natom+quick_molspec%nextatom
             if(quick_basis%katom(IIsh).eq.iatom.and.quick_basis%katom(JJsh).eq.iatom)then
                 continue
              else
                if(iatom<=natom)then
                 Cx=xyz(1,iatom)
                 Cy=xyz(2,iatom)
                 Cz=xyz(3,iatom)
                 Z=-1.0d0*quick_molspec%chg(iatom)
                else
                 Cx=quick_molspec%extxyz(1,iatom-natom)
                 Cy=quick_molspec%extxyz(2,iatom-natom)
                 Cz=quick_molspec%extxyz(3,iatom-natom)
                 Z=-1.0d0*quick_molspec%extchg(iatom-natom)
                endif

                PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2

                U = g* PCsquare
                !    Maxm = i+j+k+ii+jj+kk
                call FmT(Maxm,U,aux)
                do L = 0,maxm
                   aux(L) = aux(L)*constant*Z
                   attraxiao(1,1,L)=aux(L)
                enddo

                do L = 0,maxm-1
                   attraxiaoopt(1,1,1,L)=2.0d0*g*(Px-Cx)*aux(L+1)
                   attraxiaoopt(2,1,1,L)=2.0d0*g*(Py-Cy)*aux(L+1)
                   attraxiaoopt(3,1,1,L)=2.0d0*g*(Pz-Cz)*aux(L+1)
                enddo

                ! At this point all the auxillary integrals have been
                ! calculated.
                ! It is now time to decompase the attraction integral to it's
                ! auxillary integrals through the recursion scheme.  To do
                ! this we use
                ! a recursive function.

                !    attraction =
                !    attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &

                      !    Cx,Cy,Cz,Px,Py,Pz,g)
                NIJ1=10*NII2+NJJ2

!                call nuclearattrahess(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &

!                      Cx,Cy,Cz,Px,Py,Pz,iatom)

             endif

          enddo
        endif
     enddo
  enddo

  ! Xiao HE remember to multiply Z   01/12/2008
  !    attraction = attraction*(-1.d0)* Z
  return

end subroutine attrashellhess

double precision function electricfld(a,b,i,j,k,ii,jj,kk, &
     idx,idy,idz,Ax,Ay,Az, &
     Bx,By,Bz,Cx,Cy,Cz,Z)
  use quick_constants_module
  use quick_overlap_module, only: overlap
  implicit double precision(a-h,o-z)
  dimension aux(0:20)
  double precision :: g_table(200)

  ! Variables needed later:
  !    pi=3.1415926535897932385

  g = a+b
  Px = (a*Ax + b*Bx)/g
  Py = (a*Ay + b*By)/g
  Pz = (a*Az + b*Bz)/g
  g_table = g**(-1.5)

  PCsquare = (Px-Cx)**2.d0 + (Py -Cy)**2.d0 + (Pz -Cz)**2.d0

  ! The purpose of this subroutine is to calculate the derivative of the
  ! nuclear attraction integral with respect to the nuclear displacement
  ! of the atom the electronic distribution is attracted to:

  ! d/dXC (Integral over all space) Phi(mu) Phi(nu) 1/rC

  ! The notation is the same used throughout: gtfs with orbital exponents a
  ! and b on A and B with angular momentums defined by i,j,k (a's x, y
  ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
  ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
  ! arises from the recusion relationship. New to this are the idx, idy, and
  ! idz terms which denote derivatives in the x y and z direction for the
  ! C atom.

  ! The this is taken from the recursive relation found in Obara and Saika,
  ! J. Chem. Phys. 84 (7) 1986, 3963.

  ! The first step is generating all the necessary auxillary integrals.
  ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
  ! The values of m range from 0 to i+j+k+ii+jj+kk+2. This is exactly the
  ! same as in the attraction code, and is necessary as we will be calling
  ! that code eventually.

  U = g* PCsquare
  Maxm = i+j+k+ii+jj+kk+2
  call FmT(Maxm,U,aux)
  constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
       * 2.d0 * (g/Pi)**0.5d0
  do L = 0,maxm
     aux(L) = aux(L)*constant
  enddo

  ! At this point all the auxillary integrals have been calculated.
  ! It is now time to decompase the attraction integral to it's
  ! auxillary integrals through the recursion scheme.  To do this we use
  ! a recursive function.

  electricfld = -1.d0*Z*elctfldrecurse(i,j,k,ii,jj,kk,idx,idy,idz, &
       0,aux,Ax,Ay,Az,Bx,By,Bz, &
       Cx,Cy,Cz,Px,Py,Pz,g)

  return
end function electricfld


double precision recursive function elctfldrecurse(i,j,k,ii,jj,kk, &
     idx,idy,idz,m,aux,Ax,Ay,Az,Bx,By,Bz, &
     Cx,Cy,Cz,Px,Py,Pz,g) &
     result(elctfldrec)
  implicit double precision(a-h,o-z)
  dimension iexponents(6),center(12),aux(0:20)

  ! The this is taken from the recursive relation found in Obara and Saika,
  ! J. Chem. Phys. 84 (7) 1986, 3963.

  ! Check to see if the integral has become just the nuclear attraction
  ! integral.

  if (idx+idy+idz == 0) then
     elctfldrec = attrecurse(i,j,k,ii,jj,kk,m,aux,Ax,Ay, &
          Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g)

     ! If it hasn't, check to see if it has become a simple integral over s
     ! functions.

  ELSEIF (i+j+k+ii+jj+kk == 0) then
     if (idx == 2) then
        elctfldrec = -2.d0*g*attrecurse(i,j,k,ii,jj,kk, &
             m+1,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g) &
             +4.d0*g*g*(Px-Cx)*(Px-Cx)*attrecurse(i,j,k,ii,jj,kk, &
             m+2,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idy == 2) then
        elctfldrec = -2.d0*g*attrecurse(i,j,k,ii,jj,kk, &
             m+1,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g) &
             +4.d0*g*g*(Py-Cy)*(Py-Cy)*attrecurse(i,j,k,ii,jj,kk, &
             m+2,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idz == 2) then
        elctfldrec = -2.d0*g*attrecurse(i,j,k,ii,jj,kk, &
             m+1,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g) &
             +4.d0*g*g*(Pz-Cz)*(Pz-Cz)*attrecurse(i,j,k,ii,jj,kk, &
             m+2,aux,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idx == 1 .and. idy == 1) then
        elctfldrec = 4.d0*g*g*(Px-Cx)*(Py-Cy)* &
             attrecurse(i,j,k,ii,jj,kk,m+2,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idx == 1 .and. idz == 1) then
        elctfldrec = 4.d0*g*g*(Px-Cx)*(Pz-Cz)* &
             attrecurse(i,j,k,ii,jj,kk,m+2,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idy == 1 .and. idz == 1) then
        elctfldrec = 4.d0*g*g*(Py-Cy)*(Pz-Cz)* &
             attrecurse(i,j,k,ii,jj,kk,m+2,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idx == 1) then
        elctfldrec = 2.d0*g*(Px-Cx)* &
             attrecurse(i,j,k,ii,jj,kk,m+1,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idy == 1) then
        elctfldrec = 2.d0*g*(Py-Cy)* &
             attrecurse(i,j,k,ii,jj,kk,m+1,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     ELSEIF (idz == 1) then
        elctfldrec = 2.d0*g*(Pz-Cz)* &
             attrecurse(i,j,k,ii,jj,kk,m+1,aux,Ax,Ay,Az,Bx,By,Bz, &
             Cx,Cy,Cz,Px,Py,Pz,g)
     endif

     ! Otherwise, use the recusion relation from Obara and Saika.  The first
     ! step is to find the lowest nonzero angular momentum exponent.  This is
     ! because the more exponents equal zero the fewer terms need to be
     ! calculated, and each recursive loop reduces the angular momentum
     ! exponents. This therefore reorders the atoms and sets the exponent
     ! to be reduced.

  else
     iexponents(1) = i
     iexponents(2) = j
     iexponents(3) = k
     iexponents(4) = ii
     iexponents(5) = jj
     iexponents(6) = kk
     center(7) = Cx
     center(8) = Cy
     center(9) = Cz
     center(10)= Px
     center(11)= Py
     center(12)= Pz
     ilownum=300
     ilowex=300
     do L=1,6
        if (iexponents(L) < ilowex .and. iexponents(L) /= 0) then
           ilowex=iexponents(L)
           ilownum=L
        endif
     enddo
     if (ilownum <= 3) then
        center(1)=Ax
        center(2)=Ay
        center(3)=Az
        center(4)=Bx
        center(5)=By
        center(6)=Bz
     else
        center(4)=Ax
        center(5)=Ay
        center(6)=Az
        center(1)=Bx
        center(2)=By
        center(3)=Bz
        iexponents(4) = i
        iexponents(5) = j
        iexponents(6) = k
        iexponents(1) = ii
        iexponents(2) = jj
        iexponents(3) = kk
        ilownum = ilownum - 3
     endif

     ! The first step is lowering the orbital exponent by one.

     iexponents(ilownum) = iexponents(ilownum)-1

     ! At this point, calculate the first two terms of the recusion
     ! relation.

     elctfldrec = 0.d0
     PA = center(9+ilownum)-center(ilownum)
     if (PA /= 0) elctfldrec  = elctfldrec  + PA * &
          elctfldrecurse(iexponents(1),iexponents(2), &
          iexponents(3),iexponents(4), &
          iexponents(5),iexponents(6), &
          idx,idy,idz,m,aux, &
          center(1),center(2),center(3), &
          center(4),center(5),center(6), &
          center(7),center(8),center(9), &
          center(10),center(11),center(12),g)

     PC = center(9+ilownum)-center(6+ilownum)
     if (PC /= 0) elctfldrec  = elctfldrec  - PC * &
          elctfldrecurse(iexponents(1),iexponents(2), &
          iexponents(3),iexponents(4), &
          iexponents(5),iexponents(6), &
          idx,idy,idz,m+1,aux, &
          center(1),center(2),center(3), &
          center(4),center(5),center(6), &
          center(7),center(8),center(9), &
          center(10),center(11),center(12),g)

     ! The next two terms only arise is the angual momentum of the dimension
     ! of A that has already been lowered is not zero.  In other words, if a
     ! (px|1/rc|px) was passed to this subroutine, we are now considering
     ! (s|1/rc|px), and the following term does not arise, as the x expoent
     ! on A is zero.

     if (iexponents(ilownum) /= 0) then
        coeff = dble(iexponents(ilownum))/(2.d0*g)
        iexponents(ilownum) = iexponents(ilownum)-1
        elctfldrec  = elctfldrec  + coeff*( &
             elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy,idz,m,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g) &
             -elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy,idz,m+1,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g) &
             )
        iexponents(ilownum) = iexponents(ilownum)+1
     endif

     ! The next two terms only arise is the angual momentum of the dimension
     ! of A that has already been lowered is not zero in B.  If a
     ! (px|1/rc|px) was passed to this subroutine, we are now considering
     ! (s|1/rc|px), and the following term does arise, as the x exponent on
     ! B is 1.

     if (iexponents(ilownum+3) /= 0) then
        coeff = dble(iexponents(ilownum+3))/(2.d0*g)
        iexponents(ilownum+3) = iexponents(ilownum+3)-1
        elctfldrec = elctfldrec + coeff*( &
             elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy,idz,m,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g) &
             -elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy,idz,m+1,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g) &
             )
        iexponents(ilownum+3) = iexponents(ilownum+3)+1
     endif

     ! Finally there is a lowering of the derivative term, which only occurs
     ! if the angular momentum being lowered in this step corresponds to the
     ! derivative direction.

     if (ilownum == 1 .and. idx > 0) then
        elctfldrec = elctfldrec + dble(idx)* &
             elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx-1,idy,idz,m+1,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g)
     ELSEIF (ilownum == 2 .and. idy > 0) then
        elctfldrec = elctfldrec + dble(idy)* &
             elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy-1,idz,m+1,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g)
     ELSEIF (ilownum == 3 .and. idz > 0) then
        elctfldrec = elctfldrec + dble(idz)* &
             elctfldrecurse(iexponents(1),iexponents(2), &
             iexponents(3),iexponents(4), &
             iexponents(5),iexponents(6), &
             idx,idy,idz-1,m+1,aux, &
             center(1),center(2),center(3), &
             center(4),center(5),center(6), &
             center(7),center(8),center(9), &
             center(10),center(11),center(12),g)
     endif
  endif


  return
end function elctfldrecurse




! S. Dixon's diagonalization code from DivCon.
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    SUBROUTINE hessDIAG(NDIM,A,NEVEC1,TOLERA,V,EVAL1,IDEGEN1,EVEC1, &
    IERROR)

! DRIVER ROUTINE FOR DIAGONALIZATION OF THE REAL, SYMMETRIC,
! MATRIX A.

! VARIABLES REQUIRED:
! ------------------

! NDIM = ORDER OF THE MATRIX A (I.E., A IS NDIM BY NDIM);

! A = REAL SYMMETRIC MATRIX TO BE DIAGONALIZED.  ONLY THE LOWER
! HALF OF A NEED BE FILLED WHEN CALLING.  A IS DESTROYED BY
! THIS ROUTINE.

! NEVEC1 = THE NUMBER OF EIGENVECTORS REQUIRED;

! TOLERA = TOLERANCE FACTOR USED IN THE QR ITERATION TO DETERMINE
! WHEN OFF-DIAGONAL ENTRIES ARE ESSENTIALLY ZERO.  (DEFAULT
! IS 1.0D-8).

! V = 3 BY NDIM WORKSPACE.


! VARIABLES RETURNED:
! ------------------

! EVAL1 = EIGENVALUES OF A (SORTED IN INCREASING ALGEBRAIC VALUE);

! IDEGEN1 = DEGENERACIES (NUMBER OF TIMES REPEATED) FOR EIGENVALUES;

! EVEC1 = EIGENVECTORS OF A (IN COLUMNS OF EVEC1);

! ERROR CODES:  IERROR=0 - SUCCESSFUL CALCULATION.
! IERROR=1 - NO CONVERGENCE IN QR ITERATION.


! PROGRAMMED BY S. L. DIXON, OCT., 1991.


    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION A(NDIM,*),V(3,*),EVAL1(*),IDEGEN1(*),EVEC1(NDIM,*)
    DIMENSION A(natom*3,natom*3),V(3,natom*3),EVAL1(natom*3), &
    IDEGEN1(natom*3),EVEC1(natom*3,natom*3)

! FLAG FOR WHETHER OR NOT TO COMPUTE EIGENVALUES:


    if(NDIM == 1)then
        EVAL1(1) = A(1,1)
        IDEGEN1(1) = 1
        EVEC1(1,1) = 1.0D0
        RETURN
    endif

! TRIDIAGONALIZE THE MATRIX A.  THIS WILL OVERWRITE THE DIAGONAL
! AND SUBDIAGONAL OF A WITH THE TRIDIAGONALIZED VERSION.  THE
! HOUSEHOLDER VECTORS ARE RETURNED IN THE ROWS ABOVE THE DIAGONAL,
! AND THE BETAHS ARE RETURNED BELOW THE SUBDIAGONAL.

    CALL hessTRIDI(NDIM,V,A)

! COMPUTE NORM OF TRIDIAGONAL MATRIX FROM THE "LARGEST" COLUMN.

    ANORM = ABS(A(1,1)) + ABS(A(2,1))
    if(NDIM > 2)then
        do 20 I=2,NDIM-1
            AICOL = ABS(A(I-1,I)) + ABS(A(I,I)) + ABS(A(I+1,I))
            ANORM = MAX(ANORM,AICOL)
        20 enddo
    endif
    ANCOL = ABS(A(NDIM-1,NDIM)) + ABS(A(NDIM,NDIM))
    ANORM = MAX(ANORM,ANCOL)

! GET EIGENVALUES AND DEGENERACIES OF THE TRIDIAGONAL MATRIX A.
! if THE CALLING ROUTINE HAS NOT SUPPLIED A TOLERANCE FACTOR FOR
! OFF-DIAGONAL ENTRIES IN THE QR ITERATION, A DEFAULT OF 1.0D-8
! WILL BE USED.

    TOLTMP = TOLERA
    if(TOLTMP <= 0.0D0) TOLTMP = 1.0D-8
    CALL hessEIGVAL(NDIM,A,V,TOLTMP,ANORM,EVAL1,IERROR)
    if(IERROR /= 0) RETURN

! DETERMINE DEGENERACIES OF EIGENVALUES.

    CALL hessDEGEN(NDIM,EVAL1,TOLTMP,ANORM,IDEGEN1)

! GET EIGENVECTORS OF TRIDIAGONALIZED VERSION OF A.

    if(NEVEC1 <= 0) RETURN
    CALL hessEIGVEC(NDIM,NEVEC1,A,V,TOLTMP,ANORM,EVAL1,IDEGEN1,EVEC1)

! PREMULTIPLY EVEC1 BY THE HOUSEHOLDER MATRIX USED TO TRIDIAGONALIZE
! A.  THIS TRANSFORMS EIGENVECTORS OF THE TRIDIAGONAL MATRIX TO
! THOSE OF THE ORIGINAL MATRIX A.  SEE SUBROUTINE TRIDI FOR
! STORAGE OF HOUSEHOLDER TRANSFORMATION.

    if(NDIM > 2)then
        do 30 K=1,NDIM-2
            V(1,K) = A(K+2,K)
        30 enddo
    
    ! SWAP STORAGE SO THAT THE EXPENSIVE TRIPLE LOOP BELOW DOESN'T
    ! HAVE TO JUMP ACROSS COLUMNS TO GET ENTRIES OF A.
    
        do 50 I=2,NDIM
            do 40 J=1,I-1
                A(I,J) = A(J,I)
            40 enddo
        50 enddo
        do 160 J=1,NEVEC1
            do 140 M=2,NDIM-1
                K = NDIM - M
                BETAH = V(1,K)
                if(ABS(BETAH) < 1.0D-50) CONTINUE !GO TO 140
                SUM = 0.0D0
                do 100 I=K+1,NDIM
                    SUM = SUM + A(I,K)*EVEC1(I,J)
                100 enddo
                BSUM = BETAH*SUM
                do 120 I=K+1,NDIM
                    EVEC1(I,J) = EVEC1(I,J) - A(I,K)*BSUM
                120 enddo
            140 enddo
        160 enddo
    endif
    RETURN
    end SUBROUTINE hessDIAG



    SUBROUTINE hessTRIDI(NDIM,V,A)

! TRIDIAGONALIZES A REAL, SYMMETRIC MATRIX A BY THE METHOD OF
! HOUSEHOLDER (J. H. WILKINSON, THE COMPUTER JOURNAL, VOL. 3,
! P. 23 (1960)).  NDIM IS THE ORDER OF A.  THE DIAGONAL AND
! SUBDIAGONAL OF A ARE OVERWRITTEN WITH THE TRIDIAGONALIZED
! VERSION OF A.  THE VECTORS USED IN EACH HOUSEHOLDER
! TRANSFORMATION ARE STORED ABOVE THE DIAGONAL IN THE FIRST
! NDIM-2 ROWS OF A.  THE BETAHS ARE RETURNED BELOW THE SUBDIAGONAL
! OF A.  V IS A WORKSPACE ARRAY.

! PROGRAMMED BY S. L. DIXON, OCT., 1991.


    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION A(NDIM,*),V(3,*)
    DIMENSION A(natom*3,natom*3),V(3,natom*3)

! THRESH WILL BE USED AS A THRESHOLD TO DETERMINE if A VALUE SHOULD
! BE CONSIDERED TO BE ZERO.  THIS CAN BE CHANGED BY THE USER.

    THRESH = 1.0D-50

! if A IS 2 BY 2 OR SMALLER, then IT IS ALREADY TRIDIAGONAL -- NO
! NEED TO CONTINUE.

    if(NDIM <= 2) GO TO 1000
    do 500 K=1,NDIM-2
    
    ! DETERMINE THE VECTOR V USED IN THE HOUSEHOLDER TRANSFORMATION P.
    ! FOR EACH VALUE OF K THE HOUSEHOLDER MATRIX P IS DEFINED AS:
    
    ! P = I - BETAH*V*V'
    
    
    ! CONSTRUCT A HOUSEHOLDER TRANSFORMATION ONLY if THERE IS A NONZERO
    ! OFF-DIAGONAL ELEMENT BELOW A(K,K).
    
        ALPHA2 = 0.0D0
        do 60 I=K+1,NDIM
            V(1,I) = A(I,K)
            ALPHA2 = ALPHA2 + V(1,I)**2
        60 enddo
        APTEMP = ALPHA2 - V(1,K+1)**2
        ALPHA = DSQRT(ALPHA2)
        if(ALPHA >= THRESH)then
            BETAH = 1.0D0/(ALPHA*(ALPHA + ABS(V(1,K+1))))
            SGN = SIGN(1.0D0,V(1,K+1))
            V(1,K+1) = V(1,K+1) + SGN*ALPHA
        
        ! NOW OVERWRITE A WITH P'*A*P.  THE ENTRIES BELOW THE SUBDIAGONAL
        ! IN THE KTH COLUMN ARE ZEROED BY THE PREMULTIPLICATION BY P'.
        ! THESE ENTRIES WILL BE LEFT ALONE TO SAVE TIME.
        
            AKV = APTEMP + A(K+1,K)*V(1,K+1)
            S = BETAH*AKV
            A(K+1,K) = A(K+1,K) - S*V(1,K+1)
        
        ! NOW THE SUBMATRIX CONSISTING OF ROWS K+1,NDIM AND COLUMNS K+1,NDIM
        ! MUST BE OVERWRITTEN WITH THE TRANSFORMATION.
        
            DOT12 = 0.0D0
            BHALF = BETAH*0.5D0
            do 220 I=K+1,NDIM
                SUM = 0.0D0
                do 100 J=K+1,I
                    SUM = SUM + A(I,J)*V(1,J)
                100 enddo
                if(I < NDIM)then
                    do 180 J=I+1,NDIM
                    
                    ! AN UPPER TRIANGULAR ENTRY OF A WILL BE REQUIRED.  MUST USE
                    ! THE SYMMETRIC ENTRY IN THE LOWER TRIANGULAR PART OF A.
                    
                        SUM = SUM + A(J,I)*V(1,J)
                    180 enddo
                endif
                V(2,I) = BETAH*SUM
                DOT12 = DOT12 + V(1,I)*V(2,I)
            220 enddo
            BH12 = BHALF*DOT12
            do 300 I=K+1,NDIM
                V(2,I) = V(2,I) - BH12*V(1,I)
            300 enddo
            do 350 J=K+1,NDIM
                do 310 I=J,NDIM
                    A(I,J) = A(I,J) - V(1,I)*V(2,J) - V(2,I)*V(1,J)
                310 enddo
            
            ! STORE V(1,J) ABOVE THE DIAGONAL IN ROW K OF A
            
                A(K,J) = V(1,J)
            350 enddo
        
        ! STORE BETAH BELOW THE SUBDIAGONAL OF A.
        
            A(K+2,K) = BETAH
        else
        
        ! NO HOUSEHOLDER TRANSFORMATION IS NECESSARY BECAUSE THE OFF-
        ! DIAGONALS ARE ALL ESSENTIALLY ZERO.
        
            A(K+2,K) = 0.0D0
            do 460 J=K+1,NDIM
                A(K,J) = 0.0D0
            460 enddo
        endif
    500 enddo
    1000 RETURN
    end SUBROUTINE hessTRIDI



    SUBROUTINE hessEIGVAL(NDIM,A,BETAH,TOLERA,ANORM,EVAL1,IERROR)

! QR ROUTINE FOR THE DETERMINATION OF ALL THE EIGENVALUES
! OF THE NDIM BY NDIM SYMMETRIC, TRIDIAGONAL MATRIX A.

! INPUT:

! NDIM   = SIZE OF MATRIX A.
! A      = NDIM BY NDIM SYMMETRIC TRIDIAGONAL MATRIX.
! BETAH   = 3 BY NDIM WORKSPACE.
! TOLERA = SMALL NUMBER USED TO DETERMINE WHEN OFF-DIAGONAL
! ELEMENTS ARE ESSENTIALLY ZERO.
! ANORM  = ABSOLUTE COLUMN NORM OF TRIDIAGONAL MATRIX A.


! RETURNED:

! EVAL1   = EIGENVALUES OF A IN ASCENDING ORDER.
! IERROR = 1 if QR ITERATION DID NOT CONVERGE; 0 OTHERWISE.

! PROGRAMMED BY S. L. DIXON.


    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION A(NDIM,*),BETAH(3,*),EVAL1(*)
    DIMENSION A(natom*3,natom*3),BETAH(3,natom*3),EVAL1(natom*3)
    IERROR = 0
! ITMAX = 20
    ITMAX = 200

! TOLERANCE FOR OFF-DIAGONAL ELEMENTS:

    EPSLON = TOLERA*ANORM

! COPY DIAGONAL ELEMENTS OF A TO EVAL1, AND SUBDIAGONAL ELEMENTS
! TO BETAH.

    EVAL1(1) = A(1,1)
    BETAH(1,1) = A(2,1)
    if(NDIM > 2)then
        do 50 I=2,NDIM-1
            EVAL1(I) = A(I,I)
            BETAH(1,I) = A(I+1,I)
        50 enddo
    endif
    EVAL1(NDIM) = A(NDIM,NDIM)

! EACH QR ITERATION WILL OPERATE ON THE UNREDUCED TRIDIAGONAL
! SUBMATRIX WITH UPPER LEFT ELEMENT (L,L) AND LOWER RIGHT ELEMENT
! (N,N).

    L = 1
    N = NDIM
    ITER = 0

! FIND THE SMALLEST UNREDUCED SUBMATRIX WITH LOWER RIGHT CORNER AT
! (N,N).  I.E., SEARCH UPWARD FOR A BETAH THAT IS ZERO.

    80 KUPPER = N-L
    do 100 K=1,KUPPER
        I = N-K
        if(ABS(BETAH(1,I)) <= EPSLON)then
            L = I+1
            GO TO 150
        endif
    100 enddo

! if WE GET TO THE NEXT STATEMENT, then THERE ARE NO ZERO OFF-DIAGONALS
! FOR THE SUBMATRIX WITH UPPER LEFT A(L,L) AND LOWER RIGHT A(N,N).
! WE CAN STILL GET EIGENVALUES if THE MATRIX IS 2 BY 2 OR 1 BY 1.
! OTHERWISE, do ANOTHER QR ITERATION PROVIDED ITMAX CYCLES HAVE
! NOT OCCURRED.

    if(L == N .OR. L == N-1)then
        GO TO 150
    else
        if(ITER == ITMAX)then
            IERROR = 1
            GO TO 1000
        else
            GO TO 200
        endif
    endif

! if WE GET TO 150 then A(L,L-1) IS ZERO AND THE UNREDUCED SUBMATRIX
! HAS UPPER LEFT AT A(L,L) AND LOWER RIGHT AT A(N,N).  WE CAN
! EXTRACT ONE EIGENVALUE if THIS MATRIX IS 1 BY 1 AND 2 EIGENVALUES
! if IT IS 2 BY 2.

    150 if(L == N)then
    
    ! IT'S A 1 BY 1 AND EVAL1(N) IS AN EIGENVALUE.  if L=2 OR 1 WE ARE
    ! DONE.  OTHERWISE, UPDATE N, RESET L AND ITER, AND REPEAT THE
    ! SEARCH.
    
        if(L <= 2)then
            GO TO 500
        else
            N = L-1
            L = 1
            ITER = 0
            GO TO 80
        endif
    ELSEIF(L == N-1)then
    
    ! THE UNREDUCED SUBMATRIX IS A 2 BY 2.  OVERWRITE EVAL1(N-1)
    ! AND EVAL1(N) WITH THE EIGENVALUES OF THE LOWER RIGHT 2 BY 2.
    
        BTERM = EVAL1(N-1) + EVAL1(N)
        ROOT1 = BTERM*0.5D0
        ROOT2 = ROOT1
        DISCR = BTERM**2 - 4.0D0*(EVAL1(N-1)*EVAL1(N)-BETAH(1,N-1)**2)
        if(DISCR > 0.0D0)then
            D = DSQRT(DISCR)*0.5D0
            ROOT1 = ROOT1 - D
            ROOT2 = ROOT2 + D
        endif
        EVAL1(N-1) = ROOT1
        EVAL1(N) = ROOT2
    
    ! SEE if WE ARE DONE.  if NOT, RESET N, L, AND ITER AND LOOK
    ! FOR NEXT UNREDUCED SUBMATRIX.
    
        if(L <= 2)then
            GO TO 500
        else
            N = L-1
            L = 1
            ITER = 0
            GO TO 80
        endif
    else
    
    ! AN EIGENVALUE WAS FOUND AND THE NEW UNREDUCED MATRIX LIMITS
    ! N AND L ARE SET.  do A QR ITERATION ON NEW MATRIX.
    
        ITER = 0
        GO TO 200
    endif

! QR ITERATION BEGINS HERE.

    200 ITER = ITER + 1

! USE EIGENVALUES OF THE LOWER RIGHT 2 BY 2 TO COMPUTE SHIFT.  SHIFT
! BY THE EIGENVALUE CLOSEST TO EVAL1(N).

    D = (EVAL1(N-1) - EVAL1(N))*0.5D0
    SIGND = 1.0D0
    if(D < 0.0D0) SIGND = -1.0D0
    SHIFT = EVAL1(N) + D - SIGND*DSQRT(D*D + BETAH(1,N-1)**2)
    P = EVAL1(L) - SHIFT
    R = BETAH(1,L)
    T = EVAL1(L)
    W = BETAH(1,L)

! OVERWRITE A WITH Q'*A*Q.

    do 250 K=L,N-1
        D = DSQRT(P*P + R*R)
        C = P/D
        S = R/D
        if(K /= L) BETAH(1,K-1) = D
        CC = C*C
        SS = 1.0D0 - CC
        CS = C*S
        CSW = 2.0D0*CS*W
        AK1 = EVAL1(K+1)
        EVAL1(K) = CC*T + CSW + SS*AK1
        P = (CC - SS)*W + CS*(AK1 - T)
        T = SS*T - CSW + CC*AK1
        R = S*BETAH(1,K+1)
        W = C*BETAH(1,K+1)
    250 enddo
    BETAH(1,N-1) = P
    EVAL1(N) = T

! GO BACK AND SEE if L AND N NEED TO BE UPDATED.

    GO TO 80

! SORT EIGENVALUES IN ASCENDING ALGEBRAIC ORDER.

    500 do 600 I=2,NDIM
        JMAX = NDIM-I+1
        ISORT = 0
        do 550 J=1,JMAX
            if(EVAL1(J) > EVAL1(J+1))then
                ETEMP = EVAL1(J)
                EVAL1(J) = EVAL1(J+1)
                EVAL1(J+1) = ETEMP
                ISORT = 1
            endif
        550 enddo
        if(ISORT == 0) GO TO 1000
    600 enddo
    1000 RETURN
    end SUBROUTINE hessEIGVAL



    SUBROUTINE hessDEGEN(NDIM,EVAL1,TOLERA,ANORM,IDEGEN1)

! DETERMINES DEGENERACIES OF THE EIGENVALUES.

! INPUT:

! NDIM   = SIZE OF MATRIX BEING DIAGONALIZED.
! EVAL1   = SORTED EIGENVALUES (INCREASING VALUE).
! TOLERA = SAME TOLERANCE USED TO DETERMINE EIGENVALUES.
! ANORM  = ABSOLUTE COLUMN NORM OF TRIDIAGONAL MATRIX.


! RETURNED:

! IDEGEN1 = DEGENERACIES OF EIGENVALUES.

    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION EVAL1(*),IDEGEN1(*)
    DIMENSION EVAL1(natom*3),IDEGEN1(natom*3)

! DETERMINE DEGENERACIES OF EIGENVALUES.  ADJACENT EIGENVALUES
! WILL BE CONSIDERED TO BE DEGENERATE WHEN THEY DIFFER BY LESS
! THAN DTOLER.

    DTOLER = MAX(ANORM*DSQRT(TOLERA),1.0D-8)
    NSAME = 1
    do 200 I=2,NDIM
        DIFF = ABS(EVAL1(I-1) - EVAL1(I))
        if(DIFF <= DTOLER)then
        
        ! EIGENVALUES I-1 AND I ARE DEGENERATE.
        
            NSAME = NSAME + 1
            if(I == NDIM)then
            
            ! WE'VE COME TO THE LAST REQUESTED EIGENVALUE, AND IT'S TIME
            ! TO ASSIGN DEGENERACIES FOR THE BLOCK ENDING WITH THE ITH
            ! EIGENVALUE.
            
                do 100 J=I-NSAME+1,I
                    IDEGEN1(J) = NSAME
                100 enddo
            endif
        
        ! GO TO THE NEXT EIGENVALUE (if THERE ARE ANY LEFT) AND SEE if
        ! IT'S DEGENERATE WITH THE NSAME EIGENVALUES WE'VE ALREADY
        ! FOUND.
        
            GO TO 200
        else
        
        ! EITHER EIGENVALUE I-1 IS NONDEGENERATE OR IT'S THE LAST
        ! EIGENVALUE IN A DEGENERATE BLOCK.  CORRESPONDINGLY, ASSIGN THE
        ! PROPER DEGENERACY TO I-1 OR TO EACH EIGENVALUE IN THE BLOCK.
        
            do 150 J=I-NSAME,I-1
                IDEGEN1(J) = NSAME
            150 enddo
            NSAME = 1
        
        ! if I=NDIM then IT MUST BE THE CASE THAT THIS LAST EIGENVALUE
        ! IS NONDEGENERATE.
        
            if(I == NDIM) IDEGEN1(I) = 1
        endif
    200 enddo
    RETURN
    end SUBROUTINE hessDEGEN



    SUBROUTINE hessEIGVEC(NDIM,NEVEC1,A,AWORK,TOLERA,ANORM,EVAL1,IDEGEN1, &
    EVEC1)

! INVERSE ITERATION ROUTINE FOR EIGENVECTOR DETERMINATION.
! CALCULATES THE EIGENVECTORS OF AN NDIM BY NDIM SYMMETRIC,
! TRIDIAGONAL MATRIX A.

! INPUT:

! NDIM   = SIZE OF MATRIX A.
! NEVEC1  = NUMBER OF EIGENVECTORS REQUIRED.
! A      = NDIM BY NDIM TRIDIAGONAL MATRIX.
! TOLERA = SAME TOLERANCE USED TO DETERMINE EIGENVALUES.
! ANORM  = ABSOLUTE COLUMN NORM OF TRIDIAGONAL MATRIX A.
! EVAL1   = SORTED EIGENVALUES OF A.
! IDEGEN1 = DEGENERACIES OF EIGENVALUES.


! RETURNED:

! EVEC1   = EIGENVECTORS OF TRIDIAGONAL MATRIX (IN COLUMNS).

! PROGRAMMED BY S. L. DIXON, OCT., 1991.


    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION A(NDIM,*),AWORK(3,*),EVAL1(*),IDEGEN1(*),EVEC1(NDIM,*)
    DIMENSION A(natom*3,natom*3),AWORK(3,natom*3),EVAL1(natom*3), &
    IDEGEN1(natom*3),EVEC1(natom*3,natom*3)
    LOGICAL :: ORTH
    IRAND = 13876532

! COMPUTE THRESHOLD EPSLON WHICH WILL BE USED if THE INVERSE ITERATION
! MATRIX IS SINGULAR.

    EPSLON = ANORM*TOLERA

! WHEN DEGENERACIES OCCUR, THERE ARE RARE INSTANCES WHEN THE
! DEGENERATE BLOCK OF EIGENVECTORS ARE NOT LINEARLY INDEPENDENT.
! IN THESE CASES, AN ADDITIONAL PASS THROUGH THE INVERSE ITERATION
! (WITH A NEW SET OF RANDOM NUMBERS) IS CARRIED OUT, I.E., CONTROL
! PASSES TO STATEMENT 40.  IVECT WILL KEEP TRACK OF THE CURRENT
! STARTING EIGENVECTOR WHEN ADDITIONAL PASSES ARE NECESSARY.

    IVECT = 1
    NFAIL = 0
    NPRTRB = 0

! do ONE ITERATION FOR EACH EIGENVECTOR.

    40 ORTH = .TRUE.
    NDEGEN = 0
    MSTART = IVECT
    do 380 M=MSTART,NEVEC1
        if(IDEGEN1(M) > 1) NDEGEN = NDEGEN + 1
        Z = EVAL1(M)
    
    ! if THE INVERSE ITERATION HAS FAILED TWICE DUE TO NON-ORTHOGONALITY
    ! OF DEGENERATE EIGENVECTORS, PERTURB THE EIGENVALUE BY A SMALL
    ! AMOUNT.
    
        if(NFAIL >= 2)then
            NPRTRB = NPRTRB + 1
            Z = Z + 0.001D0*DBLE(NPRTRB)*TOLERA
        endif
    
    ! STORE THE TRIDIAGONAL ENTRIES OF THE INVERSE ITERATION MATRIX IN
    ! THE 3 BY NDIM WORKSPACE AWORK.
    
        AWORK(1,1) = 0.0D0
        AWORK(2,1) = A(1,1) - Z
        AWORK(3,1) = A(2,1)
        if(NDIM > 2)then
            do 80 I=2,NDIM-1
                AWORK(1,I) = A(I,I-1)
                AWORK(2,I) = A(I,I) - Z
                AWORK(3,I) = A(I+1,I)
            80 enddo
        endif
        AWORK(1,NDIM) = A(NDIM,NDIM-1)
        AWORK(2,NDIM) = A(NDIM,NDIM) - Z
        AWORK(3,NDIM) = 0.0D0
    
    ! ASSIGN INVERSE ITERATION VECTOR FROM RANDOM NUMBERS.
    
        do 120 I=1,NDIM
            CALL RANDOM(IRAND,RNDOM)
            RNDOM = 2.0D0*(RNDOM - 0.5D0)
            EVEC1(I,M) = RNDOM*TOLERA
        120 enddo
    
    ! CARRY OUT FORWARD GAUSSIAN ELIMINATION WITH ROW PIVOTING
    ! ON THE INVERSE ITERATION MATRIX.
    
        do 160 K=1,NDIM-1
            ADIAG = ABS(AWORK(2,K))
            ASUB = ABS(AWORK(1,K+1))
            if(ADIAG >= ASUB)then
            
            ! USE PIVOTAL ELEMENT FROM ROW K.
            
                if(AWORK(2,K) == 0.0D0) AWORK(2,K) = EPSLON
                T = AWORK(1,K+1)/AWORK(2,K)
                AWORK(1,K+1) = 0.0D0
                AWORK(2,K+1) = AWORK(2,K+1) - T*AWORK(3,K)
            
            ! LEFT-JUSTIFY EQUATION K SO THAT DIAGONAL ENTRY IS STORED
            ! IN AWORK(1,K).
            
                AWORK(1,K) = AWORK(2,K)
                AWORK(2,K) = AWORK(3,K)
                AWORK(3,K) = 0.0D0
            
            ! OPERATE ON VECTOR AS WELL.
            
                EVEC1(K+1,M) = EVEC1(K+1,M) - T*EVEC1(K,M)
            else
            
            ! USE PIVOTAL ELEMENT FROM ROW K+1 AND SWAP ROWS K AND K+1.
            
                if(AWORK(1,K+1) == 0.0D0) AWORK(1,K+1) = EPSLON
                T = AWORK(2,K)/AWORK(1,K+1)
                ATEMP = AWORK(3,K) - T*AWORK(2,K+1)
                AWORK(1,K) = AWORK(1,K+1)
                AWORK(2,K) = AWORK(2,K+1)
                AWORK(3,K) = AWORK(3,K+1)
                AWORK(1,K+1) = 0.0D0
                AWORK(2,K+1) = ATEMP
                AWORK(3,K+1) = -T*AWORK(3,K+1)
            
            ! OPERATE ON VECTOR AND SWAP ENTRIES.
            
                ETEMP = EVEC1(K+1,M)
                EVEC1(K+1,M) = EVEC1(K,M) - ETEMP*T
                EVEC1(K,M) = ETEMP
            endif
        160 enddo
    
    ! FORWARD ELIMINATION COMPLETE.  BACK SUBSTITUTE TO GET SOLUTION.
    ! OVERWRITE COLUMN M OF EVEC1 WITH SOLUTION.
    
        if(AWORK(2,NDIM) == 0.0D0) AWORK(2,NDIM) = EPSLON
        EVEC1(NDIM,M) = EVEC1(NDIM,M)/AWORK(2,NDIM)
        ETEMP = EVEC1(NDIM-1,M) - AWORK(2,NDIM-1)*EVEC1(NDIM,M)
        EVEC1(NDIM-1,M) = ETEMP/AWORK(1,NDIM-1)
        ENORM = EVEC1(NDIM,M)**2 + EVEC1(NDIM-1,M)**2
        if(NDIM > 2)then
        
        ! CAUTION: PROBLEM LOOP FOR SOME IBM RS/6000 COMPILERS.  VALUE
        ! OF K CAN GET LOST WHEN OPTIMIZE FLAG IS USED.
        
            do 200 L=1,NDIM-2
                K = NDIM-L-1
                ETEMP = EVEC1(K,M) - AWORK(2,K)*EVEC1(K+1,M) &
                - AWORK(3,K)*EVEC1(K+2,M)
                EVEC1(K,M) = ETEMP/AWORK(1,K)
                ENORM = ENORM + EVEC1(K,M)**2
            200 enddo
        endif
        EINV = 1.0D0/DSQRT(ENORM)
    
    ! NORMALIZE EIGENVECTOR.
    
        do 240 I=1,NDIM
            EVEC1(I,M) = EVEC1(I,M)*EINV
        240 enddo
    
    ! if WE HAVE COME TO THE END OF A DEGENERATE BLOCK OF EIGENVECTORS,
    ! ORTHOGONALIZE THE BLOCK.
    
        if(NDEGEN > 1)then
            if(NDEGEN == IDEGEN1(M) .OR. M == NEVEC1)then
                JSTART = M-NDEGEN+1
                CALL hessORTHOG(NDIM,NDEGEN,JSTART,EVEC1,ORTH)
                if(ORTH)then
                    NFAIL = 0
                    NPRTRB = 0
                
                ! THE DEGENERATE VECTORS WERE LINEARLY INDEPENDENT AND WERE
                ! SUCCESSFULLY ORTHOGONALIZED.
                
                    IVECT = IVECT + NDEGEN
                    NDEGEN = 0
                else
                
                ! THE BLOCK IS APPARENTLY NOT LINEARLY INDEPENDENT.  GO BACK
                ! AND REPEAT THE INVERSE ITERATION FOR THESE VECTORS.  AFTER
                ! AN INDEPENDENT SET HAS BEEN FOUND, ANY ADDITIONAL EIGENVECTORS
                ! WILL BE DETERMINED.
                
                    NFAIL = NFAIL + 1
                    GO TO 40
                endif
            endif
        endif
    
    ! THE CURRENT EIGENVECTOR SHOULD BE OKAY if IT IS NONDEGENERATE.
    
        if(IDEGEN1(M) == 1) IVECT = IVECT + 1
    380 enddo
    RETURN
    end SUBROUTINE hessEIGVEC



    SUBROUTINE hessORTHOG(NDIM,NVECT,JSTART,VECT,ORTH)

! CONSTRUCTS A SET OF ORTHONORMAL VECTORS FROM THE NVECT LINEARLY
! INDEPENDENT, NORMALIZED VECTORS IN THE ARRAY VECT.  THE VECTORS
! SHOULD BE STORED COLUMNWISE, STARTING IN COLUMN JSTART.  VECT IS
! OVERWRITTEN WITH THE ORTHONORMAL SET.  ALL VECTORS ARE NDIM BY 1.
! ORTH IS RETURNED WITH A VALUE OF .TRUE. if THE SET WAS LINEARLY
! INDEPENDENT AND .FALSE. OTHERWISE.

! PROGRAMMED BY S. L. DIXON.


    use allmod
    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
! DIMENSION VECT(NDIM,*)
    DIMENSION VECT(natom*3,natom*3)
    LOGICAL :: ORTH

    ORTH = .TRUE.
    ORTEST = 1.0D-8

! BEGIN ORTHOGONALIZATION.

    JSTOP = JSTART + NVECT - 1
    do 120 J=JSTART,JSTOP
        if(J > JSTART)then
        
        ! SUBTRACT OFF COMPONENTS OF PREVIOUSLY DETERMINED ORTHOGONAL
        ! VECTORS FROM THE VECTOR IN COLUMN J.
        
            do 60 JPREV=JSTART,J-1
                DOT = 0.0D0
                do 20 I=1,NDIM
                    DOT = DOT + VECT(I,JPREV)*VECT(I,J)
                20 enddo
                do 40 I=1,NDIM
                    VECT(I,J) = VECT(I,J) - DOT*VECT(I,JPREV)
                40 enddo
            60 enddo
        endif
    
    ! NORMALIZE COLUMN J.
    
        VJNORM = 0.0D0
        do 80 I=1,NDIM
            VJNORM = VJNORM + VECT(I,J)**2
        80 enddo
        VJNORM = DSQRT(VJNORM)
    
    ! if THE NORM OF THIS VECTOR IS TOO SMALL then THE VECTORS ARE
    ! NOT LINEARLY INDEPENDENT.
    
        if(VJNORM < ORTEST)then
            ORTH = .FALSE.
            GO TO 1000
        endif
        do 100 I=1,NDIM
            VECT(I,J) = VECT(I,J)/VJNORM
        100 enddo
    120 enddo
    1000 RETURN
    end SUBROUTINE hessORTHOG


subroutine PriHessian(io,n,mat,fm) ! format: f(x.y) x>7 sugg 12.5,12.7,14.9
  implicit none
  integer j,jj,n,io,n5,nf,x,y,ini,ifi,k,i,iatom,imom
  double precision mat(n,n)
  character fm*(*),ch,fm2*10
  character*40 fmt1,fmt2,fmt3,fmt4
  character(len=1) cartsym(3)
  character(len=5) name(n)
  cartsym(1) = 'X'
  cartsym(2) = 'Y'
  cartsym(3) = 'Z'

  iatom=1
  imom=0
  do i=1,n
    imom=imom+1
    if (imom.eq.4) then
        imom=1
        iatom=iatom+1
    endif
    write(name(i),'(i4,a1)') iatom,cartsym(imom)
  enddo
    
  
  n5=n/5
  nf=mod(n,5)
  fm2=fm
  ch=fm2(1:1)
  k=index(fm2,'.')
  read(fm2(2:k-1),*) x
  read(fm2(k+1:10),*) y

  write(fmt1,101) ch,x,y
  write(fmt2,102) nf,ch,x,y
101 format('(a5,5',a1,i2,'.',i2,')')
102 format('(a5,',i2,a1,i2,'.',i2,')')
  write(fmt3,103) x-7
  write(fmt4,104) nf
103 format('(3x,5(',i2,'x,i7))')
104 format('(1x,',i2,'(7x,a5))')

  do jj=1,n5
     ini=1+(jj-1)*5
     write(io,'(8x,5(a5,7x))') (name(j),j=ini,jj*5)
     do k=1+(jj-1)*5,n
        ifi=min(jj*5,k)
        write(io,fmt1) name(k),(mat(k,j),j=ini,ifi)
     enddo
     !         if (jj.ne.n5.or.nf.ne.0) write(io,*)
  enddo

  if (nf.ne.0) then
     ini=n-nf+1
     write(io,fmt4)(name(j),j=ini,n)
     do k=ini,n
        write(io,fmt2) name(k),(mat(k,j),j=ini,k)
     enddo
  endif
  call flush(io)

end subroutine PriHessian
