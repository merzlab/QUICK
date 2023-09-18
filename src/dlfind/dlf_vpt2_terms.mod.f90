! Module with the actual vibrational perturbation theory (VPT2)
! formula implementations. Furthermore, everything related to 
! Fermi resonance classification and treatment is contained here.

module dlf_vpt2_terms
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none
real(rk), parameter :: ft1_tol_crit_abs_def=0.1_rk
real(rk), parameter :: ft2_tol_crit_abs_def=0.1_rk
real(rk), parameter :: ft1_tol_crit_isaacson_def=1.e5_rk
real(rk), parameter :: ft2_tol_crit_isaacson_def=1.e5_rk
real(rk), parameter :: ft1_tol_crit_martin_def=1.e14_rk
real(rk), parameter :: ft2_tol_crit_martin_def=1.e14_rk
real(rk),save :: hdcpt2_alpha
real(rk),save :: hdcpt2_beta
real(rk),save :: ft1_tol_crit_abs=ft1_tol_crit_abs_def
real(rk),save :: ft2_tol_crit_abs=ft2_tol_crit_abs_def
real(rk),save :: ft1_tol_abs
real(rk),save :: ft2_tol_abs
real(rk),save :: ft1_tol_abs_hard
real(rk),save :: ft2_tol_abs_hard
real(rk),save :: ft1_tol_crit_isaacson=ft1_tol_crit_isaacson_def
real(rk),save :: ft2_tol_crit_isaacson=ft2_tol_crit_isaacson_def
real(rk),save :: ft1_tol_isaacson
real(rk),save :: ft2_tol_isaacson
real(rk),save :: ft1_tol_crit_martin=ft1_tol_crit_martin_def
real(rk),save :: ft2_tol_crit_martin=ft2_tol_crit_martin_def
real(rk),save :: ft1_tol_martin
real(rk),save :: ft2_tol_martin
!!real(rk), parameter :: resonance_threshold=200._rk, resonance_threshold_crit=0.1_rk
!!real(rk), parameter :: hdcpt2_alpha_def=1.0_rk
!!real(rk), parameter :: hdcpt2_beta_def =5.0e5_rk
!!real(rk), parameter :: ft1_tol_crit_abs_def=0.1_rk
!!real(rk), parameter :: ft2_tol_crit_abs_def=0.1_rk
!!real(rk), parameter :: ft1_tol_abs_def=100._rk
!!real(rk), parameter :: ft2_tol_abs_def=100._rk
!!real(rk), parameter :: ft1_tol_abs_hard_def=200._rk
!!real(rk), parameter :: ft2_tol_abs_hard_def=200._rk
!!real(rk), parameter :: ft1_tol_crit_isaacson_def=1.e5_rk
!!real(rk), parameter :: ft2_tol_crit_isaacson_def=1.e5_rk
!!real(rk), parameter :: ft1_tol_isaacson_def=0.4_rk
!!real(rk), parameter :: ft2_tol_isaacson_def=0.2_rk
!!real(rk), parameter :: ft1_tol_crit_martin_def=1.e14_rk
!!real(rk), parameter :: ft2_tol_crit_martin_def=1.e14_rk
!!real(rk), parameter :: ft1_tol_martin_def=1._rk
!!real(rk), parameter :: ft2_tol_martin_def=1._rk
!type fermi_param
!  real(rk) :: tol_crit
!  real(rk) :: tol
!  real(rk) :: tol_E_hard
!end type fermi_param
!real(rk), save :: hdcpt2_alpha=hdcpt2_alpha_def
!real(rk), save :: hdcpt2_beta =hdcpt2_beta_def

contains

! ****************************************
! ****************************************
! ****************************************

subroutine vpt2_init_fermi_res_params(vpt2_res_tol_deltae,vpt2_res_tol_deltae_hard, &
                   & vpt2_res_tol_martin,vpt2_res_tol_isaacson,vpt2_hdcpt2_alpha,vpt2_hdcpt2_beta)
use dlf_vpt2_utility, only: error_print
implicit none
real(rk), intent(in) :: vpt2_res_tol_deltae
real(rk), intent(in) :: vpt2_res_tol_deltae_hard
real(rk), intent(in) :: vpt2_res_tol_martin
real(rk), intent(in) :: vpt2_res_tol_isaacson
real(rk), intent(in) :: vpt2_hdcpt2_alpha
real(rk), intent(in) :: vpt2_hdcpt2_beta

ft1_tol_abs        = vpt2_res_tol_deltae
ft2_tol_abs        = vpt2_res_tol_deltae
ft1_tol_abs_hard   = vpt2_res_tol_deltae_hard
ft2_tol_abs_hard   = vpt2_res_tol_deltae_hard
ft1_tol_isaacson   = vpt2_res_tol_isaacson
ft2_tol_isaacson   = vpt2_res_tol_isaacson
ft1_tol_martin     = vpt2_res_tol_martin
ft2_tol_martin     = vpt2_res_tol_martin
hdcpt2_alpha       = vpt2_hdcpt2_alpha
hdcpt2_beta        = vpt2_hdcpt2_beta

end subroutine

! ****************************************
! ****************************************
! ****************************************

! Detect Fermi type I resonances
!
!! Fermi resonance criteria: 
! * Most primitive version: consider absolute energy differences (nu_i - 2 nu_j) for Fermi type I 
!                           and (nu_i - nu_j - nu_k) for Fermi type II resonances
! * Isaacson criterion:     consider ratio between energy difference and corresponding cubic force
!                           constant, relations obtained from 
!                            (i)   Isaacson and Zhang, Theor Chim Acta (1988) 74:493-511.
!                            (ii)  Isaacson and Hung, J. Chem. Phys. 108 101, 3928 (1994); doi: 10.1063/1.467511
!                            (iii) Isaacson, J. Chem. Phys. 108, 9978 (1998); doi: 10.1063/1.476496.
!                             A threshold value of 0.2 is proposed for both |k_ijj/deltaE| and |k_ijk/deltaE|
!                             Their cubic force constants k assume a different notation convention!
!                             (Nielsen nomenclature)
!                             k_ijk = phi_ijk
!                             k_iij = 1/2 * phi_iij
!                             k_iii = 1/6 * phi_iii
!                              => we use criteria |phi_ijj/deltaE| > 0.4 (doubled)
!                                 and |phi_ijk/deltaE| > 0.2 (unchanged)
! * Martin criterion from Jan M. L. Martin, Timothy J. Lee, Peter R. Taylor, and Jean-Pierre François, 
!                     Journal of Chemical Physics 103, 2589 (1995); doi: 10.1063/1.469681
!                     see Appendix A
!                     - Consider difference E difference between variational and perturbative treatment
!                     - Fermi Type 1: phi_ijj^4/(256*deltaE^3)
!                     - Fermi Type 2: phi_ijk^4/(64 *deltaE^3)
!                     No concrete value is proposed in their paper; Barone and co-workers use values between
!                     1 and 10 cm^-1, see 
!                     - Barone, J. Chem. Phys. 122, 014108 (2005); doi: 10.1063/1.1824881. and
!                     - Bloino, Biczysko and Barone, J. Phys. Chem. A 2015, 119, 11862−11874.

subroutine detect_fermi_type1(nu1,nu2,cubic,mode,resonance_found)
use dlf_vpt2_utility, only: error_print
implicit none
real(rk), intent(in) :: nu1,nu2,cubic
character(len=*), intent(in) :: mode
logical, intent(out) :: resonance_found

real(rk) :: deltaE,ratio

resonance_found=.false.
deltaE=2*nu2-nu1

select case(mode)
  case ('deltaE_absolute')
    if     (abs(deltaE).le.ft1_tol_crit_abs) then
      call error_print('detect_fermi_type1: Critical Fermi type I resonancce detected!')
    elseif (abs(deltaE).le.ft1_tol_abs) then
      resonance_found=.true.
      return
    endif
  case('martin')
    ratio=abs(cubic**4/(256._rk*deltaE**3))
    if     (ratio.ge.ft1_tol_crit_martin) then
      call error_print('detect_fermi_type1: Critical Fermi type I resonancce detected!')
    elseif (ratio.ge.ft1_tol_martin) then
      if (abs(deltaE)<=ft1_tol_abs_hard) then
        resonance_found=.true.
      endif
      return
    endif
  case ('isaacson')
    ratio=abs(cubic/deltaE)
    if     (ratio.ge.ft1_tol_crit_isaacson) then
      call error_print('detect_fermi_type1: Critical Fermi type I resonancce detected!')
    elseif (ratio.ge.ft1_tol_isaacson) then
      if (abs(deltaE)<=ft1_tol_abs_hard) then
        resonance_found=.true.
      endif
      return
    endif
  case default
    call error_print('detect_fermi_type1: called with invalid mode: '//mode)
end select 

end subroutine detect_fermi_type1

! ****************************************
! ****************************************
! ****************************************

! Detect Fermi type II resonances. See the comments for the 
! type I resonance detection routine above.

subroutine detect_fermi_type2(nu1,nu2,nu3,cubic,mode,resonance_found)
use dlf_vpt2_utility, only: error_print
implicit none
real(rk), intent(in) :: nu1,nu2,nu3,cubic
character(len=*), intent(in) :: mode
logical, intent(out) :: resonance_found

real(rk) :: deltaE,ratio

resonance_found=.false.
deltaE=nu1-nu2-nu3

select case(mode)
  case ('deltaE_absolute')
    if     (abs(deltaE).le.ft2_tol_crit_abs) then
      call error_print('detect_fermi_type2: Critical Fermi type II resonancce detected!')
    elseif (abs(deltaE).le.ft2_tol_abs) then
      resonance_found=.true.
      return
    endif
  case('martin')
    ratio=abs(cubic**4/(64._rk*deltaE**3))
    if     (ratio.ge.ft2_tol_crit_martin) then
      call error_print('detect_fermi_type2: Critical Fermi type II resonancce detected!')
    elseif (ratio.ge.ft2_tol_martin) then
      if (abs(deltaE)<=ft2_tol_abs_hard) then
        resonance_found=.true.
      endif
      return
    endif
  case ('isaacson')
    ratio=abs(cubic/deltaE)
    if     (ratio.ge.ft2_tol_crit_isaacson) then
      call error_print('detect_fermi_type2: Critical Fermi type II resonancce detected!')
    elseif (ratio.ge.ft2_tol_isaacson) then
      if (abs(deltaE)<=ft2_tol_abs_hard) then
        resonance_found=.true.
      endif
      return
    endif
  case default
    call error_print('detect_fermi_type2: called with invalid mode: '//mode)
end select 

end subroutine detect_fermi_type2

! ****************************************
! ****************************************
! ****************************************

! Caclulate the Coriolis coupling constants zeta

subroutine get_coriolis_coupling_constants(nvrt,neff,normal_mode_vectors,coriolis)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nvrt,neff
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(3,neff,neff), intent(out) :: coriolis

integer  :: i,j,m,nat
real(rk), dimension(3) :: lim, ljm, cp

nat=nint(real(nvrt)/3.)
coriolis=0._rk
do i=1,neff
  do j=1,i-1
    do m=1,nat
      lim=normal_mode_vectors(3*(m-1)+1:3*(m-1)+3,i)
      ljm=normal_mode_vectors(3*(m-1)+1:3*(m-1)+3,j)
      cp=dlf_cross_product(lim,ljm)
      coriolis(:,i,j)=coriolis(:,i,j)+cp(:)
    enddo
  enddo
enddo

! Symmetric fill
do i=1,neff
  do j=i+1,neff
    coriolis(:,i,j)=-coriolis(:,j,i)
  enddo
enddo

end subroutine get_coriolis_coupling_constants

! ****************************************
! ****************************************
! ****************************************

! Different, but equivalent way to obtain the zetas.
! This was just implemented for consistency checks.

subroutine get_coriolis_coupling_constants_alternative(nvrt,neff,normal_mode_vectors,coriolis)
use dlf_linalg_interface_mod
use dlf_vpt2_utility, only: matrix_output
implicit none
integer, intent(in) :: nvrt,neff
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(3,neff,neff), intent(out) :: coriolis

integer  :: m,i1,i2,nat
real(rk), dimension(nvrt,nvrt) :: Mx,My,Mz
real(rk), dimension(3,3) :: Mxat,Myat,Mzat

nat=nint(real(nvrt)/3.)

!call matrix_output(normal_mode_vectors,stdout,'ES20.12','Normal modes')
!call matrix_output(dlf_matmul_simp(transpose(normal_mode_vectors),normal_mode_vectors),stdout,'ES20.12','Ortho test')
!read(*,*)

Mx=0._rk
My=0._rk
Mz=0._rk

Mxat=0._rk
Myat=0._rk
Mzat=0._rk

coriolis=0._rk

Mxat(2,3)=1._rk
Mxat(3,2)=-1._rk

Myat(1,3)=-1._rk
Myat(3,1)=1._rk

Mzat(1,2)=1._rk
Mzat(2,1)=-1._rk

do m=1,nat
  i1=3*(m-1)+1
  i2=3*(m-1)+3
  Mx(i1:i2,i1:i2)=Mxat(1:3,1:3)
  My(i1:i2,i1:i2)=Myat(1:3,1:3)
  Mz(i1:i2,i1:i2)=Mzat(1:3,1:3)
enddo

!call test_M_matrix_properties(nvrt,Mx,My,Mz)

coriolis(1,:,:)=dlf_matrix_ortho_trans(normal_mode_vectors,Mx,0)
coriolis(2,:,:)=dlf_matrix_ortho_trans(normal_mode_vectors,My,0)
coriolis(3,:,:)=dlf_matrix_ortho_trans(normal_mode_vectors,Mz,0)

end subroutine get_coriolis_coupling_constants_alternative

! ****************************************
! ****************************************
! ****************************************

! Yet another way to compute the Coriolis zetas. This was just implemented for 
! consistency checks

subroutine get_coriolis_coupling_constants_alternative2(nvrt,neff,normal_mode_vectors,coriolis)
implicit none
integer, intent(in) :: nvrt,neff
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(3,neff,neff), intent(out) :: coriolis

integer  :: i,j,k,m,k1,k2,nat
real(rk) :: L1,L2,L3,L4

nat=nint(real(nvrt)/3.)
coriolis=0._rk
do i=1,neff
  do j=1,i-1
    do k=1,3
      k1=mod(k+1,3)
      k2=mod(k+2,3)
      if(k1.eq.0) k1=3
      if(k2.eq.0) k2=3
      do m=1,nat
        L1=normal_mode_vectors(3*(m-1)+k1,i)
        L2=normal_mode_vectors(3*(m-1)+k2,j)
        L3=normal_mode_vectors(3*(m-1)+k2,i)
        L4=normal_mode_vectors(3*(m-1)+k1,j)
        coriolis(k,i,j)=coriolis(k,i,j)+L1*L2-L3*L4
      enddo
    enddo
  enddo
enddo

! Symmetric fill
do i=1,neff
  do j=i+1,neff
    coriolis(:,i,j)=-coriolis(:,j,i)
  enddo
enddo

end subroutine get_coriolis_coupling_constants_alternative2

! ****************************************
! ****************************************
! ****************************************

! Coriolis coupling contribution for the X matrix elements coupling the transition mode 
! (with imaginary frequency) to the bound modes. (in the case of saddle points!) 

subroutine coriolis_contribution_xiF(neff,nreal,coriolis_partial,A,B,C,nu_harm_real,nu_im,xiF_cor)
implicit none
integer, intent(in) :: neff,nreal
real(rk), intent(in) :: A,B,C
real(rk), dimension(3,neff), intent(in) :: coriolis_partial
real(rk), dimension(nreal), intent(in) :: nu_harm_real
real(rk), intent(in) :: nu_im
real(rk), dimension(neff), intent(out)  :: xiF_cor

integer  :: i

xiF_cor=0._rk

do i=1,nreal
  xiF_cor(i)=A*coriolis_partial(1,i)**2+B*coriolis_partial(2,i)**2+C*coriolis_partial(3,i)**2
  !xiF_cor(i)=xiF_cor(i)*(nu_harm_real(i)/nu_im - nu_im/nu_harm_real(i))
  xiF_cor(i)=xiF_cor(i)*(nu_harm_real(i)/nu_im - nu_im/nu_harm_real(i))
enddo

return
end subroutine coriolis_contribution_xiF

! ****************************************
! ****************************************
! ****************************************

! Self-explaining

subroutine coriolis_contribution_xmatrix(neff,coriolis,A,B,C,nu_harm,xcor)
implicit none
integer, intent(in) :: neff
real(rk), intent(in) :: A,B,C
real(rk), dimension(3,neff,neff), intent(in) :: coriolis
real(rk), dimension(neff), intent(in) :: nu_harm
real(rk), dimension(neff,neff), intent(out)  :: xcor

integer  :: i,j

xcor=0._rk

do i=1,neff
  do j=1,i-1
    xcor(i,j)=A*coriolis(1,i,j)**2+B*coriolis(2,i,j)**2+C*coriolis(3,i,j)**2
    xcor(i,j)=xcor(i,j)*(nu_harm(i)/nu_harm(j)+nu_harm(j)/nu_harm(i))
  enddo
enddo

return
end subroutine coriolis_contribution_xmatrix

! ****************************************
! ****************************************
! ****************************************

! Coriolis contribution to the constant anharmonic term E_0.

subroutine coriolis_contribution_E0(neff,coriolis,A,B,C,nu_harm,E0cor)
implicit none
integer, intent(in) :: neff
real(rk), intent(in) :: A,B,C
real(rk), dimension(3,neff,neff), intent(in) :: coriolis
real(rk), dimension(neff), intent(in) :: nu_harm
real(rk), intent(out)  :: E0cor

integer  :: i,j,k
real(rk) :: sumtmp
real(rk), dimension(3) :: bvec

bvec(1)=A
bvec(2)=B
bvec(3)=C
E0cor=0._rk

if (neff==1) return

do k=1,3
  sumtmp=0._rk
  do i=1,neff
    do j=1,i-1
      sumtmp=sumtmp+(coriolis(k,i,j))**2
    enddo
  enddo
  E0cor=E0cor+bvec(k)*(1._rk+2._rk*sumtmp)
enddo
E0cor=E0cor/(-4._rk)

return
end subroutine coriolis_contribution_E0

! ****************************************
! ****************************************
! ****************************************
!
! VPT2 X matrix and E_0 driver routine. Handles different cases of 
! resonance treatments
!
! Get anharmonic constants X_ij and constant anharmonic term E0 from (modified) VPT2
! 
! Input: - reduced cubic (f_ijk) and semi-diagonal quartic force constants (f_iijk), in cm^-1, 
!          with respect to dimensionless normal coordinates
!        - harmonic vibrational wavenumbers (cm^-1)
! Output: X_ij and E0 in cm^-1
! Expressions obtained from 
! (1) Ian M. Mills, "Vibration-Rotation Structure in Asymmetric- and Symmetric-Top Molecules" in 
!     "Molecular Spectroscopy: Modern Research", Eds: K. Narahari Rao and C. Weldon Mathews, 
!     Academic Press, New York, 1972, p.115--140.
! (2) K.M.Kuhler, D.G. Truhlar, A.D.Isaacson, J. Chem. Phys. 104 (1996), 4664--4671.
!
! * Fermi resonance handling: 
!        Depends on input variable resonance_mode:
!          - 'none':    use simple VPT2, ignoring any resonances (might be very unsafe)
!          - 'depert':  deperturbed VPT2, removing all resonant terms (no variational treatment is done)
!          - 'dcpt2':   degeneracy corrected PT2, where resonant terms are replaced by a higher-order correct
!                       non-singular term
!          - 'hdcpt2':  Hybrid degeneracy corrected PT2, which uses an interpolation between simple VPT2 and
!                       DCPT2
! * Saddle point handling implemented via separate subroutine vpt2_saddle_point
! *    - imaginary frequency must be given as the last entry of nu_harm, but as a real, positive number!
! *    - Of course, the cubic and quartic force constants, as well as normal mode eigenvectors have to be in the 
! *      corresponding sort order as well
! *    - Reduced cubic and quartic force constants are given as if ALL frequencies were real
! *      i.e. handling of the imaginary frequency is done within the formulas for X_ij, X_ii and E0
! *      this avoids handling complex reduced force constants
! *    - On output, xFF contains the diagonal anharmonic constant of the transition mode (a real number)
! *      and xiF is a vector containing the imaginary off-diagonal anharmonic constants involving the transition mode
! *      The xiF are given as real numbers and have to be multiplied by (-i), following the convention in 
! *      Miller et al., Chem. Phys. Lett. 172 (1990), 62--68.
! * X matrix convention: only elements with j .le. i are non-zero
! * Term energies: E(v)/hc = E0/hc + sum_i (nu_i*(v_i+1/2)) + sum_i { sum_{j<=i} [ X_ij (v_i+1/2)*(v_j+1/2)] }

subroutine vpt2_driver(nvrt,neff,nu_harm,normal_mode_vectors,cubic,quartic, & 
                      & A,B,C,E0,X,zpve_harm,zpve_anh,resonance_mode, dep_crit, &
                      & saddle_point,xFF,xiF)
use dlf_vpt2_utility, only: matrix_output, error_print
use dlf_global, only: stdout
implicit none
integer, intent(in) :: nvrt,neff
real(rk), dimension(neff), intent(in) :: nu_harm
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
real(rk), intent(in)  :: A,B,C
real(rk), intent(out) :: E0,zpve_harm,zpve_anh
real(rk), dimension(neff,neff), intent(out) :: X
logical, intent(in) :: saddle_point
real(rk), intent(out), optional :: xFF
real(rk), intent(out), dimension(neff), optional :: xiF
character(len=*), intent(in) :: dep_crit
character(len=*), intent(in) :: resonance_mode

integer  :: i,j,k,m,nfinp1,nfinp2
integer  :: nreal,rescount
real(rk) :: E0coriolis,zpve_corio_via_x,zpve_corio_total,zpve_sp,E0sp
character(10) :: kchar
logical, dimension(neff,neff) :: ftype1
logical, dimension(neff,neff,neff) :: ftype2
real(rk), dimension(3,neff,neff) :: coriolis
real(rk), dimension(neff,neff) :: Xcoriolis,Xsp
real(rk), dimension(neff) :: xiF_cor
logical :: res_found

X=0._rk
ftype1=.false.
ftype2=.false.

if (saddle_point) then
  nreal=neff-1
  if (.not. present(xFF)) call error_print('vpt2_generate_x_mat_deperturbation: argument xFF is missing')
  if (.not. present(xiF)) call error_print('vpt2_generate_x_mat_deperturbation: argument xiF is missing')
else
  nreal=neff
endif

if     (dep_crit=='all') then
  ftype1(:,:)=.true.
  ftype2(:,:,:)=.true.
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  write(stdout,'(A)') '&&&&  Special deperturbation mode: Treating all cubic terms as resonant! &&&&'
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  write(stdout,'(A)') ''
elseif (dep_crit=='none') then
  ftype1(:,:)=.false.
  ftype2(:,:,:)=.false.
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  write(stdout,'(A)') '&&&&  Special deperturbation mode: Treating no cubic terms as resonant!  &&&&'
  write(stdout,'(A)') '&&&&    This should be equivalent to convenctional, noncorrected VPT2.   &&&&'
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  write(stdout,'(A)') ''
elseif (dep_crit=='manual_input') then
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  write(stdout,'(A)') '&&&&  Reading Fermi resonance definitions from user-supplied input file  &&&&'
  write(stdout,'(A)') '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
  open (9080, file='fermi.inp')
  read (9080,*) nfinp1, nfinp2
  do m=1,nfinp1
    read(9080,*) i,j
    ftype1(i,j)=.true.
    ftype2(i,j,j)=.true.
  enddo
  do m=1,nfinp2
    read(9080,*) i,j,k
    ftype2(i,j,k)=.true.
    ftype2(i,k,j)=.true.
  enddo
  close(9080)
elseif (dep_crit=='deltaE_absolute' .or. dep_crit=='isaacson' .or. dep_crit=='martin') then
  ! Check for Fermi resonances
  do i=1,nreal
    do j=1,nreal
      if (i.eq.j) cycle
      ! Fermi type 1
      call detect_fermi_type1(nu_harm(i),nu_harm(j),cubic(i,j,j),dep_crit,res_found)
      ftype1(i,j)=res_found
      ftype2(i,j,j)=res_found
      ! Fermi type 2
      do k=1,j-1
        if (i.eq.k) cycle
        call detect_fermi_type2(nu_harm(i),nu_harm(j),nu_harm(k),cubic(i,j,k),dep_crit,res_found)
        ftype2(i,j,k)=res_found
        ftype2(i,k,j)=res_found
      enddo
    enddo
  enddo
else
  call error_print('vpt2_generate_x_mat_deperturbation: Invalid value for argument dep_crit was passed')
endif

! Output all combinations flagged as resonances
if (dep_crit /= 'all' .and. dep_crit /= 'none') then
  write(stdout,'(A)') ''
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%% Fermi resonances to be deperturbed %%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') ''
  write(stdout,'(2A)') 'Detection criterion: ', dep_crit
  if     (dep_crit=='deltaE_absolute') then
    write(stdout,'(A,2F10.2)') 'Tolerances (type I, type II) : ', ft1_tol_abs, ft2_tol_abs
  elseif (dep_crit=='isaacson') then
    write(stdout,'(A,2F10.2)') 'Tolerances (type I, type II)            : ', ft1_tol_isaacson, ft2_tol_isaacson
    write(stdout,'(A,2F10.2)') 'Hard limits on deltaE (type I, type II) : ', ft1_tol_abs_hard, ft2_tol_abs_hard
  elseif (dep_crit=='martin') then
    write(stdout,'(A,2F10.2)') 'Tolerances (type I, type II) :            ', ft1_tol_martin, ft2_tol_martin
    write(stdout,'(A,2F10.2)') 'Hard limits on deltaE (type I, type II) : ', ft1_tol_abs_hard, ft2_tol_abs_hard
  endif
  write(stdout,'(A)') ''
  ! Fermi type 1
  rescount=0
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%     Fermi type I resonances         %%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') ''
  do i=1,nreal
    do j=1,nreal
      if (i.eq.j) cycle
      if (ftype1(i,j)) then
        rescount=rescount+1
        if (rescount == 1) write (stdout,'(2A5,2A13,A16,2A13,A17,A29)') 'i', 'j', 'nu(i)', 'nu(j)', '2nu(j)-nu(i)', &
                  &      'phi_ijj','|phi/dE|','|phi^4/256dE³|','(all in cm^-1 or dim.less)'
        write(stdout,'(2I5,2F13.2,F16.2,2F13.2,F17.2)') i, j, nu_harm(i), nu_harm(j), 2*nu_harm(j)-nu_harm(i), cubic(i,j,j), &
                 &      abs(cubic(i,j,j)/(2*nu_harm(j)-nu_harm(i))), abs(cubic(i,j,j)**4/(256._rk*(2*nu_harm(j)-nu_harm(i))**3))
      endif
    enddo
  enddo
  if (rescount == 0) write(stdout,'(A)') '     No Fermi type I resonances found!     '
  write(stdout,'(A)') ''
  ! Fermi type 2
  rescount=0
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%     Fermi type II resonances        %%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') ''
  do i=1,nreal
    do j=1,nreal
      if (i.eq.j) cycle
      do k=1,j-1
        if (i.eq.k) cycle
        if (ftype2(i,j,k)) then
          rescount=rescount+1
          if (rescount == 1) write (stdout,'(3A5,3A13,A19,2A13,A17,A29)') 'i', 'j','k','nu(i)','nu(j)','nu(k)', &
             &     'nu(i)-nu(j)-nu(k)','phi_ijk','|phi/dE|','|phi^4/64dE³|','(all in cm^-1 or dim.less)'
          write(stdout,'(3I5,3F13.2,F19.2,2F13.2,F17.2)') i, j, k, nu_harm(i), nu_harm(j), nu_harm(k), &
                          & nu_harm(i)-nu_harm(j)-nu_harm(k), cubic(i,j,k), &
                          & abs(cubic(i,j,k)/(nu_harm(i)-nu_harm(j)-nu_harm(k))), &
                          & abs(cubic(i,j,k)**4/(64._rk*(nu_harm(i)-nu_harm(j)-nu_harm(k))**3))
        endif
      enddo
    enddo
  enddo
  if (rescount == 0) write(stdout,'(A)') '     No Fermi type II resonances found!     '
  write(stdout,'(A)') ''
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') '>>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% <<<'
  write(stdout,'(A)') ''
endif

! Get anharmonic X matrix and constant E0 term, depending on resonance-handling mode

if     (resonance_mode=='none') then
  call vpt2_get_X_E0_simple(nvrt,neff,nreal,nu_harm,&
                 & cubic,quartic,E0,X)
elseif (resonance_mode=='depert') then
  call vpt2_get_X_E0_deperturbation(nvrt,neff,nreal,nu_harm,&
                         & cubic,quartic,ftype1,ftype2,E0,X)
elseif (resonance_mode=='dcpt2') then
  call vpt2_get_X_E0_hdcpt2(nvrt,neff,nreal,nu_harm, &
                & cubic,quartic,E0,X,.false.)
elseif (resonance_mode=='hdcpt2') then
  call vpt2_get_X_E0_hdcpt2(nvrt,neff,nreal,nu_harm, &
                & cubic,quartic,E0,X,.true., &
                & hdcpt2_alpha,hdcpt2_beta)
else
  call error_print('vpt2_driver: invalid resonance_mode!')
endif

coriolis=0._rk

call get_coriolis_coupling_constants_alternative2(nvrt,neff,normal_mode_vectors,coriolis)
do k=1,3
  write(kchar,'(I10)') k
  call matrix_output(coriolis(k,:,:),stdout,'ES20.12','Coriolis coupling matrix, k = '//trim(kchar))
enddo

call vpt2_get_zpve_direct_analyze_terms(nreal,nvrt,nu_harm(1:nreal),normal_mode_vectors(:,1:nreal), &
              & cubic(1:nreal,1:nreal,1:nreal),quartic(1:nreal,1:nreal,1:nreal),A,B,C, &
              & coriolis(1:3,1:nreal,1:nreal),E0,X(1:nreal,1:nreal),zpve_anh,zpve_harm)

Xcoriolis=0._rk
call coriolis_contribution_xmatrix(nreal,coriolis(:,1:nreal,1:nreal),A,B,C,nu_harm(1:nreal),Xcoriolis(1:nreal,1:nreal))
call coriolis_contribution_E0(nreal,coriolis(:,1:nreal,1:nreal),A,B,C,nu_harm(1:nreal),E0coriolis)

write(stdout,*) ''
write(stdout,*) 'Coriolis contribution to E0: (cm-1)', E0coriolis
write(stdout,*) ''

zpve_corio_via_x=sum(sum(Xcoriolis(1:nreal,1:nreal),dim=2),dim=1)/4._rk

write(stdout,*) 'Coriolis correction to ZPVE via X (cm-1): ', zpve_corio_via_x
write(stdout,*) ''

zpve_corio_total=E0coriolis+zpve_corio_via_x

write(stdout,*) '**********************************'
write(stdout,*) ''
write(stdout,*) 'Total Coriolis correction to ZPVE (cm-1): ', zpve_corio_total
write(stdout,*) ''

!call matrix_output(X,stdout,'F20.12','X matrix WITHOUT Coriolis contribution (cm-1)')
call matrix_output(X,stdout,'D14.6','X matrix WITHOUT Coriolis contribution (cm-1)')

!call matrix_output(Xcoriolis,stdout,'F20.12','Coriolis contribution to X (cm-1)')
call matrix_output(Xcoriolis,stdout,'D14.6','Coriolis contribution to X (cm-1)')

E0 = E0+E0coriolis
X  = X + Xcoriolis

call vpt2_get_zpve_direct(nreal,nvrt,nu_harm(1:nreal),normal_mode_vectors(:,1:nreal),cubic(1:nreal,1:nreal,1:nreal), &
                        & quartic(1:nreal,1:nreal,1:nreal),A,B,C,coriolis(1:3,1:nreal,1:nreal),zpve_anh,zpve_harm)

if (saddle_point) then
  write(stdout,*) ''
  write(stdout,*) '************  ZPVE, direct, no saddle point contributions  ************'
  write(stdout,*) ''
  write(stdout,*) 'ZPVE, harmonic (cm^-1) :   ', zpve_harm
  write(stdout,*) 'ZPVE, anharmonic (cm^-1) : ', zpve_anh
  write(stdout,*) '------------------------------------------------'
  write(stdout,*) 'ZPVE anh. corr. (cm^-1) :  ', zpve_anh-zpve_harm
  write(stdout,*) ''
  write(stdout,*) '***********************************************************************'
  call vpt2_saddle_point(neff,nreal,nu_harm(1:nreal),nu_harm(neff),cubic,quartic,E0sp,Xsp,zpve_sp,xFF,xiF)
  call coriolis_contribution_xiF(neff,nreal,coriolis(:,neff,:),A,B,C,nu_harm(1:nreal),nu_harm(neff),xiF_cor)
  
  zpve_anh=zpve_anh+zpve_sp
  E0=E0+E0sp
  X=X+Xsp
  
  write(stdout,*) ''
  write(stdout,*) ' Coriolis contribution to xiF:'
  write(stdout,*) ''
  write(stdout,'(A4,3(1X,A20))') 'i', 'xiF(i)', 'xiF_cor(i)', 'xiF(i)+xiF_cor(i)'
  write(stdout,'(A)') '------------------------------------------------------'
  do i=1,nreal
    write(stdout,'(I4,3(1X,F20.12))') i, xiF(i), xiF_cor(i), xiF(i)+xiF_cor(i)
  enddo
  write(stdout,*) ''
  xiF=xiF+xiF_cor
else
  write(stdout,*) ''
  write(stdout,*) '**********************  ZPVE, direct   ********************************'
  write(stdout,*) ''
  write(stdout,*) 'ZPVE, harmonic (cm^-1) :   ', zpve_harm
  write(stdout,*) 'ZPVE, anharmonic (cm^-1) : ', zpve_anh
  write(stdout,*) '------------------------------------------------'
  write(stdout,*) 'ZPVE anh. corr. (cm^-1) :  ', zpve_anh-zpve_harm
  write(stdout,*) ''
  write(stdout,*) '***********************************************************************'
endif

return
end subroutine vpt2_driver

! ****************************************
! ****************************************
! ****************************************

! Implementation of the straightforward, 
! non-corrected, resonance-ignoring VPT2 formulas

subroutine vpt2_get_X_E0_simple(nvrt,neff,nreal,nu_harm,cubic,quartic, & 
                                   & E0,X)
  implicit none
  integer, intent(in) :: nvrt,neff,nreal
  real(rk), dimension(neff), intent(in) :: nu_harm
  real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
  real(rk), intent(out) :: E0
  real(rk), dimension(neff,neff), intent(out) :: X

  integer  :: i,j,k
  real(rk) :: sumtmp,sumtmp2,sumtmp3,sumtmp4
  real(rk), dimension(neff,neff,neff) :: D
  
  X=0._rk
  E0=0._rk
  
  ! Build D matrix
  D=0._rk
  do i=1,nreal
    do j=1,nreal
      do k=1,nreal
        D(i,j,k) =    ( nu_harm(i)+nu_harm(j)+nu_harm(k)) * ( nu_harm(i)-nu_harm(j)-nu_harm(k)) & 
        &           * (-nu_harm(i)+nu_harm(j)-nu_harm(k)) * (-nu_harm(i)-nu_harm(j)+nu_harm(k))
      enddo
    enddo
  enddo
  
  ! Diagonal X elements
  do i=1,nreal
    sumtmp=0._rk
    do j=1,nreal
      sumtmp=sumtmp + cubic(i,i,j)**2 * (8._rk*nu_harm(i)**2 - 3._rk*nu_harm(j)**2)/ & 
                     & (nu_harm(j)*(4._rk*nu_harm(i)**2 - nu_harm(j)**2))
    enddo
    X(i,i)=quartic(i,i,i) - sumtmp
    X(i,i)=X(i,i)/16._rk
  enddo
  
  ! Off-diagonal elements
  do i=1,nreal
    do j=1,i-1
      sumtmp=0._rk
      do k=1,nreal
        sumtmp=sumtmp+cubic(i,i,k)*cubic(k,j,j)/nu_harm(k)
      enddo
      sumtmp2=0._rk
      do k=1,nreal
        sumtmp2=sumtmp2 + cubic(i,j,k)**2 * nu_harm(k) * (nu_harm(k)**2 - nu_harm(i)**2 - nu_harm(j)**2)/D(i,j,k)
      enddo
      X(i,j)=quartic(i,j,j) - sumtmp - 2._rk * sumtmp2
      X(i,j)=X(i,j)/4._rk
    enddo
  enddo
  
  ! Constant anharmonic term
  sumtmp=0._rk
  do i=1,nreal
    sumtmp=sumtmp+quartic(i,i,i)
  enddo
  
  sumtmp2=0._rk
  do i=1,nreal
    sumtmp2 = sumtmp2 + cubic(i,i,i)**2/nu_harm(i)
  enddo
  
  sumtmp3=0._rk
  do i=1,nreal
    do j=1,nreal
      if (j.eq.i) cycle
      sumtmp3=sumtmp3+nu_harm(i)*cubic(i,j,j)**2/(4._rk*nu_harm(j)**2 - nu_harm(i)**2)
    enddo
  enddo
  
  sumtmp4=0._rk
  do i=1,nreal
    do j=i+1,nreal
      do k=j+1,nreal
        sumtmp4=sumtmp4 + nu_harm(i)*nu_harm(j)*nu_harm(k)*cubic(i,j,k)**2/D(i,j,k)
      enddo
    enddo
  enddo
  
  E0 = 1._rk/64._rk*sumtmp - 7._rk/576._rk * sumtmp2 + 3._rk/64._rk * sumtmp3 - sumtmp4/4._rk

return
end subroutine vpt2_get_X_E0_simple

! ****************************************
! ****************************************
! ****************************************

! Implementation of deperturbed VPT2 formulas
! (see documentation)

subroutine vpt2_get_X_E0_deperturbation(nvrt,neff,nreal,nu_harm,cubic,quartic, & 
                                   & ftype1,ftype2,E0,X)
  implicit none
  integer, intent(in) :: nvrt,neff,nreal
  real(rk), dimension(neff), intent(in) :: nu_harm
  real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
  logical, dimension(neff,neff), intent(in)      :: ftype1
  logical, dimension(neff,neff,neff), intent(in) :: ftype2
  real(rk), intent(out) :: E0
  real(rk), dimension(neff,neff), intent(out) :: X

  integer  :: i,j,k
  real(rk) :: sumtmp,sumtmp2,sumtmp3,sumtmp4
  real(rk), dimension(neff,neff,neff) :: Dpinv, Dppinv
  
  X=0._rk
  E0=0._rk
  
  ! Build 1/D' matrix (with [1/D']ijk=nu_i*nu_j*nu_k/Dijk)
  Dpinv=0._rk
  do i=1,nreal
    do j=1,nreal
      do k=1,nreal
        Dpinv(i,j,k)=1._rk/(nu_harm(i)+nu_harm(j)+nu_harm(k))
        if (.not. ftype2(i,j,k)) Dpinv(i,j,k)=Dpinv(i,j,k)+1._rk/( nu_harm(i)-nu_harm(j)-nu_harm(k))
        if (.not. ftype2(j,i,k)) Dpinv(i,j,k)=Dpinv(i,j,k)+1._rk/(-nu_harm(i)+nu_harm(j)-nu_harm(k))
        if (.not. ftype2(k,i,j)) Dpinv(i,j,k)=Dpinv(i,j,k)+1._rk/(-nu_harm(i)-nu_harm(j)+nu_harm(k))
      enddo
    enddo
  enddo
  Dpinv=Dpinv/8._rk
  
  ! Build 1/D'' matrix (with [1/D'']ijk=nu_k*(nu_k²-nu_i²-nu_j²)/Dijk)
  Dppinv=0._rk
  do i=1,nreal
    do j=1,nreal
      do k=1,nreal
        Dppinv(i,j,k)=1._rk/(nu_harm(i)+nu_harm(j)+nu_harm(k))
        if (.not. ftype2(i,j,k)) Dppinv(i,j,k)=Dppinv(i,j,k)-1._rk/( nu_harm(i)-nu_harm(j)-nu_harm(k))
        if (.not. ftype2(j,i,k)) Dppinv(i,j,k)=Dppinv(i,j,k)-1._rk/(-nu_harm(i)+nu_harm(j)-nu_harm(k))
        if (.not. ftype2(k,i,j)) Dppinv(i,j,k)=Dppinv(i,j,k)+1._rk/(-nu_harm(i)-nu_harm(j)+nu_harm(k))
      enddo
    enddo
  enddo
  Dppinv=Dppinv/4._rk
  
  ! Diagonal X elements
  do i=1,nreal
    sumtmp=0._rk
    do j=1,nreal
      if (ftype1(j,i)) then
        sumtmp=sumtmp + cubic(i,i,j)**2 * (2._rk/nu_harm(j)+0.5_rk/(2._rk*nu_harm(i)+nu_harm(j)))
      else
        sumtmp=sumtmp + cubic(i,i,j)**2 * (2._rk/nu_harm(j)+0.5_rk/(2._rk*nu_harm(i)+nu_harm(j)) &
               & -0.5_rk/(2._rk*nu_harm(i)-nu_harm(j)))
      endif
    enddo
    X(i,i)=quartic(i,i,i) - sumtmp
    X(i,i)=X(i,i)/16._rk
  enddo
  
  ! Off-diagonal elements
  do i=1,nreal
    do j=1,i-1
      sumtmp=0._rk
      do k=1,nreal
        sumtmp=sumtmp+cubic(i,i,k)*cubic(k,j,j)/nu_harm(k)
      enddo
      sumtmp2=0._rk
      do k=1,nreal
        sumtmp2=sumtmp2 + cubic(i,j,k)**2 * Dppinv(i,j,k)
      enddo
      X(i,j)=quartic(i,j,j) - sumtmp - 2._rk * sumtmp2
      X(i,j)=X(i,j)/4._rk
    enddo
  enddo
  
  ! Constant anharmonic term
  sumtmp=0._rk
  do i=1,nreal
    sumtmp=sumtmp+quartic(i,i,i)
  enddo
  
  sumtmp2=0._rk
  do i=1,nreal
    sumtmp2 = sumtmp2 + cubic(i,i,i)**2/nu_harm(i)
  enddo
  
  sumtmp3=0._rk
  do i=1,nreal
    do j=1,nreal
      if (j.eq.i) cycle
      if (ftype1(i,j)) then
        sumtmp3=sumtmp3+cubic(i,j,j)**2*(-0.5_rk/(2._rk*nu_harm(j)+nu_harm(i)))
      else
        sumtmp3=sumtmp3+cubic(i,j,j)**2*(0.5_rk/(2._rk*nu_harm(j)-nu_harm(i))-0.5_rk/(2._rk*nu_harm(j)+nu_harm(i)))
      endif
    enddo
  enddo
  
  sumtmp4=0._rk
  do i=1,nreal
    do j=i+1,nreal
      do k=j+1,nreal
        sumtmp4=sumtmp4 + Dpinv(i,j,k)*cubic(i,j,k)**2
      enddo
    enddo
  enddo
  
  E0 = 1._rk/64._rk*sumtmp - 7._rk/576._rk * sumtmp2 + 3._rk/64._rk * sumtmp3 - sumtmp4/4._rk

return
end subroutine vpt2_get_X_E0_deperturbation

! ****************************************
! ****************************************
! ****************************************

! Implementation of hybrid and conventional degeneracy-correction second-order
! perturbation theory (DCPT2 and HDCPT2). Consult documentation for more information 
! and references.

subroutine vpt2_get_X_E0_hdcpt2(nvrt,neff,nreal,nu_harm,cubic,quartic,E0,X,hybrid,alpha,beta)
  use dlf_global, only: stdout
  use dlf_vpt2_utility, only: error_print
  implicit none
  integer, intent(in) :: nvrt,neff,nreal
  real(rk), dimension(neff), intent(in) :: nu_harm
  real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
  real(rk), intent(out) :: E0
  real(rk), dimension(neff,neff), intent(out) :: X
  logical, intent(in)  :: hybrid
  real(rk), intent(in), optional :: alpha,beta

  integer  :: i,j,k,m
  real(rk) :: sumtmp,sumtmp2,sumtmp3,sumtmp4,sumtmp3_alt,sumtmp4_alt
  real(rk) :: sgn,sgn1,sgn2,diff,eps,k2
  
  real(rk), dimension(neff,neff,neff) :: t3a_com,t4a_com
  real(rk), dimension(neff,neff) :: t1_dc,t2_dc,t5_dc
  real(rk), dimension(neff,neff,neff) :: t3_dc,t3b_dc,t3c_dc,t3d_dc
  real(rk), dimension(neff,neff,neff) :: t4_dc,t4b_dc,t4c_dc,t4d_dc
  real(rk), dimension(neff,neff) :: t1_con,t2_con,t5_con
  real(rk), dimension(neff,neff,neff) :: t3_con,t3b_con,t3c_con,t3d_con
  real(rk), dimension(neff,neff,neff) :: t4_con,t4b_con,t4c_con,t4d_con
  real(rk), dimension(neff,neff) :: t1_eff,t2_eff,t5_eff
  real(rk), dimension(neff,neff,neff) :: t3_eff,t4_eff
  real(rk), dimension(neff,neff) :: lamt1,lamt2,lamt5
  real(rk), dimension(neff,neff,neff) :: lamt3b, lamt3c, lamt3d
  real(rk), dimension(neff,neff,neff) :: lamt4b, lamt4c, lamt4d
  
  X=0._rk
  E0=0._rk
  
  if (hybrid) then
    if (.not. present(alpha)) call error_print('vpt2_get_X_E0_hdcpt2: alpha missing for HDCPT2')
    if (.not. present(beta))  call error_print('vpt2_get_X_E0_hdcpt2: beta missing for HDCPT2')
    lamt1(:,:)=1._rk
    lamt2(:,:)=1._rk
    lamt3b(:,:,:)=1._rk
    lamt3c(:,:,:)=1._rk
    lamt3d(:,:,:)=1._rk
    lamt4b(:,:,:)=1._rk
    lamt4c(:,:,:)=1._rk
    lamt4d(:,:,:)=1._rk
    lamt5(:,:)=1._rk
  endif
  
  ! t1(i,j) terms (Kuhler1996 eqs. 14-18)
  t1_dc=0._rk
  do i=1,neff
    do j=1,neff
      if (j.eq.i) cycle
      diff=2._rk*nu_harm(j)-nu_harm(i)
      sgn=sign(1._rk,diff)
      eps=abs(diff)/2._rk
      k2=3._rk/64._rk*nu_harm(i)*cubic(i,j,j)**2/(2._rk*nu_harm(j)+nu_harm(i))
      t1_dc(i,j)=sgn*(sqrt(eps*eps+k2)-eps)
      if (hybrid) lamt1(i,j)=hdcpt2_lambda( k2, eps, alpha, beta )
    enddo
  enddo
  
  ! t2(i,m) terms (Kuhler1996 eqs. 19-24)
  t2_dc=0._rk
  do i=1,neff
    do m=1,neff
      if (m==i) cycle
      diff=2._rk*nu_harm(i)-nu_harm(m)
      eps=abs(diff)/2._rk
      k2=abs(cubic(i,i,m)**2*(8._rk*nu_harm(i)**2 - 3._rk*nu_harm(m)**2))/(16._rk*nu_harm(m)*(2._rk*nu_harm(i)+nu_harm(m)))
      sgn1=sign(1._rk,-8._rk*nu_harm(i)**2 + 3._rk*nu_harm(m)**2)
      sgn2=sign(1._rk,diff)
      sgn=sgn1*sgn2
      t2_dc(i,m)=sgn*(sqrt(eps*eps+k2)-eps)
      if (hybrid) lamt2(i,m)=hdcpt2_lambda( k2, eps, alpha, beta )
    enddo
  enddo
  
  ! t3(i,j,k) terms (Kuhler1996, eqs. 25-30)
  ! t3a: first term in eq. 27 (resonance-free)
  ! t3b: eq. 28
  ! t3c: eq. 29
  ! t3d: eq. 30
  t3a_com=0._rk
  t3b_dc=0._rk
  t3c_dc=0._rk
  t3d_dc=0._rk
  do i=1,neff
    do j=1,neff
      do k=1,neff
        ! t3a, non-resonant, equivalent in DCPT2 and conventional VPT2
        t3a_com(i,j,k)=-cubic(i,j,k)**2/(32._rk*(nu_harm(i)+nu_harm(j)+nu_harm(k)))
        ! t3b
        diff=nu_harm(i)-nu_harm(j)-nu_harm(k)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,k)**2/32._rk
        sgn=sign(1._rk,-diff)
        t3b_dc(i,j,k)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt3b(i,j,k)=hdcpt2_lambda( k2, eps, alpha, beta )
        ! t3c
        diff=-nu_harm(i)+nu_harm(j)-nu_harm(k)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,k)**2/32._rk
        sgn=sign(1._rk,-diff)
        t3c_dc(i,j,k)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt3c(i,j,k)=hdcpt2_lambda( k2, eps, alpha, beta )
        ! t3d
        diff=-nu_harm(i)-nu_harm(j)+nu_harm(k)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,k)**2/32._rk
        sgn=sign(1._rk,-diff)
        t3d_dc(i,j,k)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt3d(i,j,k)=hdcpt2_lambda( k2, eps, alpha, beta )
      enddo
    enddo
  enddo
  t3_dc=t3a_com+t3b_dc+t3c_dc+t3d_dc
  
  ! t4(i,j,m) terms (Kuhler1996, eqs. 32-37)
  ! t4a: first term in eq. 32 (resonance-free)
  ! t4b: eq. 33
  ! t4c: eq. 34
  ! t4d: eq. 35
  t4a_com=0._rk
  t4b_dc=0._rk
  t4c_dc=0._rk
  t4d_dc=0._rk
  do i=1,neff
    do j=1,neff
      do m=1,neff
        ! t4a, non-resonant, equivalent in DCPT2 and conventional VPT2
        t4a_com(i,j,m)=-cubic(i,j,m)**2/(8._rk*(nu_harm(i)+nu_harm(j)+nu_harm(m)))
        ! t4b
        diff=nu_harm(i)-nu_harm(j)-nu_harm(m)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,m)**2/8._rk
        sgn=sign(1._rk,diff)
        t4b_dc(i,j,m)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt4b(i,j,m)=hdcpt2_lambda( k2, eps, alpha, beta )
        ! t4c
        diff=-nu_harm(i)+nu_harm(j)-nu_harm(m)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,m)**2/8._rk
        sgn=sign(1._rk,diff)
        t4c_dc(i,j,m)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt4c(i,j,m)=hdcpt2_lambda( k2, eps, alpha, beta )
        ! t4d
        diff=-nu_harm(i)-nu_harm(j)+nu_harm(m)
        eps=abs(diff)/2._rk
        k2=cubic(i,j,m)**2/8._rk
        sgn=sign(1._rk,-diff)
        t4d_dc(i,j,m)=sgn*(sqrt(eps*eps+k2)-eps)
        if (hybrid) lamt4d(i,j,m)=hdcpt2_lambda( k2, eps, alpha, beta )
      enddo
    enddo
  enddo
  t4_dc=t4a_com+t4b_dc+t4c_dc+t4d_dc
  
  ! t5(i,j) terms (Kuhler1996, eqs. 36,37)
  t5_dc=0._rk
  do i=1,neff
    do j=1,neff
      diff=2._rk*nu_harm(i)-nu_harm(j)
      sgn=sign(1._rk,-diff)
      !write(stdout,'(A,2I4,F20.12)') 'i,j,sgn: ',i,j,sgn
      eps=abs(diff)/2._rk
      k2=nu_harm(i)*cubic(i,i,j)**2/(2._rk*(2._rk*nu_harm(i)+nu_harm(j)))
      t5_dc(i,j)=sgn*(sqrt(eps*eps+k2)-eps)
      if (hybrid) lamt5(i,j)=hdcpt2_lambda( k2, eps, alpha, beta )
    enddo
  enddo
  
  ! Calculate conventional VPT2 terms for HDCPT2 mode
  
  if (hybrid) then
    ! t1(i,j) terms
    t1_con=0._rk
    do i=1,neff
      do j=1,neff
        if (j.eq.i) cycle
        t1_con(i,j)=(3._rk*nu_harm(i)*cubic(i,j,j)**2)/(64._rk*(4*nu_harm(j)**2-nu_harm(i)**2))
      enddo
    enddo
    
    ! t2(i,m) terms
    t2_con=0._rk
    do i=1,neff
      do m=1,neff
        if ( m == i ) cycle
        t2_con(i,m)=-cubic(i,i,m)**2*(8._rk*nu_harm(i)**2 -3._rk*nu_harm(m)**2)/ &
                 & (16._rk*nu_harm(m)*(4._rk*nu_harm(i)**2 -nu_harm(m)**2))
      enddo
    enddo
    
    ! t3(i,j,k) terms (Kuhler1996, eqs. 25-30)
    t3b_con=0._rk
    t3c_con=0._rk
    t3d_con=0._rk
    do i=1,neff
      do j=1,neff
        do k=1,neff
          !t3co(i,j,k)=-nu_harm(i)*nu_harm(j)*nu_harm(k)*cubic(i,j,k)**2 / &
          !             & (4._rk* (nu_harm(i)+nu_harm(j)+nu_harm(k)) * (nu_harm(i)-nu_harm(j)-nu_harm(k)) & 
          !             & *(-nu_harm(i)+nu_harm(j)-nu_harm(k))*(-nu_harm(i)-nu_harm(j)+nu_harm(k)))
          t3b_con(i,j,k)=-cubic(i,j,k)**2/(32._rk*( nu_harm(i)-nu_harm(j)-nu_harm(k)))
          t3c_con(i,j,k)=-cubic(i,j,k)**2/(32._rk*(-nu_harm(i)+nu_harm(j)-nu_harm(k)))
          t3d_con(i,j,k)=-cubic(i,j,k)**2/(32._rk*(-nu_harm(i)-nu_harm(j)+nu_harm(k)))
        enddo
      enddo
    enddo
    t3_con=t3a_com+t3b_con+t3c_con+t3d_con
    
    ! t4(i,j,m) terms (Kuhler1996, eqs. 32-37)
    t4b_con=0._rk
    t4c_con=0._rk
    t4d_con=0._rk
    do i=1,neff
      do j=1,neff
        do m=1,neff
          !t4co(i,j,m)=(-cubic(i,j,m)**2*nu_harm(m)*(-nu_harm(i)**2-nu_harm(j)**2+nu_harm(m)**2))/ &
          !& (2._rk*(nu_harm(i)+nu_harm(j)+nu_harm(m))*(nu_harm(i)-nu_harm(j)-nu_harm(m))* &
          !& (-nu_harm(i)+nu_harm(j)-nu_harm(m))*(-nu_harm(i)-nu_harm(j)+nu_harm(m)))
          t4b_con(i,j,m)=+cubic(i,j,m)**2/(8._rk*( nu_harm(i)-nu_harm(j)-nu_harm(m)))
          t4c_con(i,j,m)=+cubic(i,j,m)**2/(8._rk*(-nu_harm(i)+nu_harm(j)-nu_harm(m)))
          t4d_con(i,j,m)=-cubic(i,j,m)**2/(8._rk*(-nu_harm(i)-nu_harm(j)+nu_harm(m)))
        enddo
      enddo
    enddo
    t4_con=t4a_com+t4b_con+t4c_con+t4d_con
    
    ! t5(i,j) terms (Kuhler1996, eqs. 36,37)
    t5_con=0._rk
    do i=1,neff
      do j=1,neff
        t5_con(i,j)=(-cubic(i,i,j)**2*nu_harm(i))/(2._rk*(4._rk*nu_harm(i)**2-nu_harm(j)**2))
      enddo
    enddo
  endif
  
  if (hybrid) then
    t1_eff(:,:)=lamt1(:,:)*t1_con(:,:) + (1._rk-lamt1(:,:)) *t1_dc(:,:)
    t2_eff(:,:)=lamt2(:,:)*t2_con(:,:) + (1._rk-lamt2(:,:)) *t2_dc(:,:)
    t5_eff(:,:)=lamt5(:,:)*t5_con(:,:) + (1._rk-lamt5(:,:)) *t5_dc(:,:)
    t3_eff(:,:,:)= t3a_com(:,:,:) &
                 & + lamt3b(:,:,:)*t3b_con(:,:,:) + (1._rk-lamt3b(:,:,:)) *t3b_dc(:,:,:) &
                 & + lamt3c(:,:,:)*t3c_con(:,:,:) + (1._rk-lamt3c(:,:,:)) *t3c_dc(:,:,:) &
                 & + lamt3d(:,:,:)*t3d_con(:,:,:) + (1._rk-lamt3d(:,:,:)) *t3d_dc(:,:,:)
    t4_eff(:,:,:)= t4a_com(:,:,:) &
                 & + lamt4b(:,:,:)*t4b_con(:,:,:) + (1._rk-lamt4b(:,:,:)) *t4b_dc(:,:,:) &
                 & + lamt4c(:,:,:)*t4c_con(:,:,:) + (1._rk-lamt4c(:,:,:)) *t4c_dc(:,:,:) &
                 & + lamt4d(:,:,:)*t4d_con(:,:,:) + (1._rk-lamt4d(:,:,:)) *t4d_dc(:,:,:)
  else
    t1_eff(:,:)  =t1_dc(:,:)
    t2_eff(:,:)  =t2_dc(:,:)
    t3_eff(:,:,:)=t3_dc(:,:,:)
    t4_eff(:,:,:)=t4_dc(:,:,:)
    t5_eff(:,:)  =t5_dc(:,:)
  endif
  
     if (hybrid) then
       !write(stdout,*) 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
       !write(stdout,*) 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
       !
       !call matrix_output(lamt1,stdout,'F20.12','HDCPT2, lambda1')
       !read(*,*)
       !call matrix_output(lamt2,stdout,'F20.12','HDCPT2, lambda2')
       !read(*,*)
       !
       !do k=1,neff
       !  write(kchar,'(I10)') k
       !  call matrix_output(lamt3b(:,:,k),stdout,'F20.12','HDCPT2, lambda3b, k= '//kchar)
       !  call matrix_output(lamt3c(:,:,k),stdout,'F20.12','HDCPT2, lambda3c, k= '//kchar)
       !  call matrix_output(lamt3d(:,:,k),stdout,'F20.12','HDCPT2, lambda3d, k= '//kchar)
       !  read(*,*)
       !enddo
       !
       !do k=1,neff
       !  write(kchar,'(I10)') k
       !  call matrix_output(lamt4b(:,:,k),stdout,'F20.12','HDCPT2, lambda4b, k= '//kchar)
       !  call matrix_output(lamt4c(:,:,k),stdout,'F20.12','HDCPT2, lambda4c, k= '//kchar)
       !  call matrix_output(lamt4d(:,:,k),stdout,'F20.12','HDCPT2, lambda4d, k= '//kchar)
       !  read(*,*)
       !enddo
       !
       !call matrix_output(lamt5,stdout,'F20.12','HDCPT2, lambda5')
       !
       !write(stdout,*) 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
       !write(stdout,*) 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
       !read(*,*)
       continue
     endif
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
     !
     !call matrix_output(t1_dc,stdout,'F20.12','t1, DCPT2')
     !call matrix_output(t1_con,stdout,'F20.12','t1, conventional')
     !call matrix_output(t1_dc-t1_con,stdout,'F20.12','t1, diff (DCPT2-conventional)')
     !read(*,*)
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
     !
     !call matrix_output(t2_dc,stdout,'F20.12','t2, DCPT2')
     !call matrix_output(t2_con,stdout,'F20.12','t2, conventional')
     !call matrix_output(t2_dc-t2_con,stdout,'F20.12','t2, diff (DCPT2-conventional)')
     !read(*,*)
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
     !
     !do k=1,neff
     !  write(kchar,'(I10)') k
     !  call matrix_output(t3_dc(:,:,k),stdout,'F20.12','t3, DCPT2, k= '//kchar)
     !  call matrix_output(t3_con(:,:,k),stdout,'F20.12','t3, conventional, k= '//kchar)
     !  call matrix_output(t3_dc(:,:,k)-t3_con(:,:,k),stdout,'F20.12','t3, diff (DCPT2--conventional), k= '//kchar)
     !enddo
     !read(*,*)
     !
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
     !
     !do k=1,neff
     !  write(kchar,'(I10)') k
     !  call matrix_output(t4_dc(:,:,k),stdout,'F20.12','t4, DCPT2, k= '//kchar)
     !  call matrix_output(t4_con(:,:,k),stdout,'F20.12','t4, conventional, k= '//kchar)
     !  call matrix_output(t4_dc(:,:,k)-t4_con(:,:,k),stdout,'F20.12','t4, diff (DCPT2--conventional), k= '//kchar)
     !enddo
     !read(*,*)
     !
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
     !
     !call matrix_output(t5_dc,stdout,'F20.12','t5, DCPT2')
     !call matrix_output(t5_con,stdout,'F20.12','t5, conventional')
     !call matrix_output(t5_dc-t5_con,stdout,'F20.12','t5, diff (DCPT2-conventional)')
     !read(*,*)
     !
     !write(stdout,*) '*****************************************'
     !write(stdout,*) '*****************************************'
  
  ! Diagonal X elements
  do i=1,neff
    sumtmp=0._rk
    do j=1,neff
      if (j.eq.i) cycle
      sumtmp=sumtmp + t2_eff(i,j)
    enddo
    ! j=i term:
    sumtmp=sumtmp - 5._rk*cubic(i,i,i)**2/(48._rk*nu_harm(i))
    X(i,i)=quartic(i,i,i)/16._rk + sumtmp
  enddo
  
  ! Off-diagonal elements
  do i=1,neff
    do j=1,i-1
      sumtmp=0._rk
      do k=1,neff
        sumtmp=sumtmp+cubic(i,i,k)*cubic(k,j,j)/nu_harm(k)
      enddo
      sumtmp2=0._rk
      do k=1,neff
        if (k.eq.i .or. k.eq.j) cycle
        sumtmp2=sumtmp2+t4_eff(i,j,k)
      enddo
      ! k=i term
      sumtmp2=sumtmp2+t5_eff(i,j)
      ! k=j term
      sumtmp2=sumtmp2+t5_eff(j,i)
      X(i,j)=quartic(i,j,j)/4._rk - sumtmp/4._rk + sumtmp2
    enddo
  enddo
  
  ! Constant anharmonic term
  sumtmp=0._rk
  do i=1,neff
    sumtmp=sumtmp+quartic(i,i,i)
  enddo
  
  sumtmp2=0._rk
  do i=1,neff
    sumtmp2 = sumtmp2 + cubic(i,i,i)**2/nu_harm(i)
  enddo
  
  sumtmp3=0._rk
  do i=1,neff
    do j=1,neff
      if (j.eq.i) cycle
      sumtmp3=sumtmp3+t1_eff(i,j)
    enddo
  enddo
  
  sumtmp4=0._rk
  do i=1,neff
    do j=i+1,neff
      do k=j+1,neff
        sumtmp4=sumtmp4+t3_eff(i,j,k)
      enddo
    enddo
  enddo

  sumtmp3_alt=0._rk
  do i=1,neff
    do j=1,neff
      if (j.eq.i) cycle
      sumtmp3_alt=sumtmp3_alt+t1_con(i,j)
    enddo
  enddo
  
  sumtmp4_alt=0._rk
  do i=1,neff
    do j=i+1,neff
      do k=j+1,neff
        sumtmp4_alt=sumtmp4_alt+t3_con(i,j,k)
      enddo
    enddo
  enddo
  
  E0 = 1._rk/64._rk*sumtmp - 7._rk/576._rk * sumtmp2 + sumtmp3 + sumtmp4
  
  write(stdout,*) '***********************************'
  write(stdout,*) 'Sum (E0):       ', E0
  write(stdout,*) 'Sum (E0,conv.): ', 1._rk/64._rk*sumtmp - 7._rk/576._rk * sumtmp2 + sumtmp3_alt + sumtmp4_alt
  write(stdout,*) ''

return
end subroutine vpt2_get_X_E0_hdcpt2

! ****************************************
! ****************************************
! ****************************************

! Get the lambdas (interpolation weights) for 
! hybrid degeneracy-corrected PT2

pure function hdcpt2_lambda(k2, eps, alpha, beta)  result(y)
  implicit none
  real(rk), intent(in) :: k2, eps, alpha, beta
  real(rk) :: y
  
  y = sqrt(k2*eps*eps) - beta
  y = alpha * y
  y = (tanh(y)+1._rk)/2._rk
  
  return

end function hdcpt2_lambda

! ****************************************
! ****************************************
! ****************************************
!
! Saddle point contributions to the X matrix and to E0
!
! The matrix Xsp contains the correction to the X matrix of the bound modes
! by taking into account the non-separability of the reaction coordinate.
! That means the 0th-order X matrix for which this correction should be applied
! is the one obtained from a VPT2 treatment of all bound modes, completely ignoring
! the reaction coordinate. 
!
! Likewise, E0sp is the correction to the constant anharmonic term, arising from 
! bound mode <-> reaction coordinate coupling.
! zpve_sp is the corresponding zero-point energy correction. It contains both E0sp and
! Xsp contributions. It should be applied to the anharmonic ZPVE of the saddle point that
! was obtained by again completely ignoring the reaction coordinate (treating the SP
! as a molecule with 3N-7 degrees of freedom), similar to the case of Xsp and E0sp.
!
! xFF and xiF are the anharmonic constants of the reaction coordinate (xFF, a real number),
! and the ones describing the coupling between bound modes and reaction coordinate (xiF, these
! are imaginary, but stored as xiF/(-i) in a real array). These quantities are only required
! for a semi-classical TST calculation.
!
! Expressions obtained from
! (1) Miller et al., Chem. Phys. Lett. 172 (1990), 62--68.
! (2) Hernandez et al., J. Chem. Phys. 99 (1993), 950--962.
! (3) Hernandez and Miller, Chem. Phys. Lett. 214 (1993), 129--136.

subroutine vpt2_saddle_point(neff,nreal,nu_harm_real,nu_im,cubic,quartic,E0sp,Xsp,zpve_sp,xFF,xiF)
use dlf_global, only: stdout,printl
use dlf_vpt2_utility, only: matrix_output, vector_output
implicit none
integer, intent(in) :: neff,nreal
real(rk), dimension(nreal), intent(in) :: nu_harm_real
real(rk), intent(in) :: nu_im
real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
real(rk), intent(out) :: E0sp,zpve_sp
real(rk), dimension(neff,neff), intent(out) :: Xsp
real(rk), intent(out) :: xFF
real(rk), intent(out), dimension(neff) :: xiF

integer  :: F,i,j,m
real(rk) :: sumtmp,sumtmp2,zpve_sp_explicit,zpve_sp_explicit_Xcont

F=neff

E0sp=0._rk
zpve_sp=0._rk
Xsp=0._rk
xFF=0._rk
xiF=0._rk

! SP contribution to diagonal X_ii elements (i!=F)

do i=1,nreal
  Xsp(i,i)=cubic(i,i,F)**2/16._rk*(8._rk*nu_harm_real(i)**2 + 3._rk*nu_im**2)/(nu_im*(4._rk*nu_harm_real(i)**2 + nu_im**2))
enddo

! SP contribution to off-diagonal X_ij elements (i!=F, j!=F)

do i=1,nreal
  do j=1,i-1
    Xsp(i,j)=cubic(i,i,F)*cubic(j,j,F)/(4._rk*nu_im)
    Xsp(i,j)=Xsp(i,j) + &
      & (cubic(i,j,F)**2 * nu_im *(nu_im**2+nu_harm_real(i)**2+nu_harm_real(j)**2)) / &
      & (2._rk*(nu_im**2 + (nu_harm_real(i)+nu_harm_real(j))**2) * (nu_im**2 + (nu_harm_real(i)-nu_harm_real(j))**2))
  enddo
enddo

! Diagonal X element of transition mode

sumtmp=0._rk
do i=1,nreal
  sumtmp=sumtmp+cubic(F,F,i)**2*(8._rk*nu_im**2 + 3._rk*nu_harm_real(i)**2)/(nu_harm_real(i)*(4._rk*nu_im**2 + nu_harm_real(i)**2))
enddo
xFF=-quartic(F,F,F)/16._rk -5._rk/48._rk*cubic(F,F,F)**2/nu_im + sumtmp/16._rk

! Off-diagonal X elements of transition mode, divided by (-i)

do i=1,nreal
  sumtmp=0._rk
  sumtmp2=0._rk
  do m=1,nreal
    sumtmp=sumtmp+cubic(i,i,m)*cubic(F,F,m)/nu_harm_real(m)
  enddo
  do m=1,nreal
    sumtmp2=sumtmp2+cubic(i,m,F)**2*nu_harm_real(m)*(nu_harm_real(i)**2-nu_harm_real(m)**2-nu_im**2) / &
                    & ((nu_im**2+(nu_harm_real(i)+nu_harm_real(m))**2)*(nu_im**2+(nu_harm_real(i)-nu_harm_real(m))**2))
  enddo
  xiF(i)=quartic(i,F,F)/4._rk - sumtmp/4._rk + cubic(i,i,F)*cubic(F,F,F)/(4._rk*nu_im) &
             & + sumtmp2/2._rk + nu_im*cubic(i,F,F)**2/(2._rk*(nu_harm_real(i)**2+4._rk*nu_im**2))
enddo

! SP Contribution to E0

sumtmp=0._rk
sumtmp2=0._rk
do i=1,nreal
  sumtmp=sumtmp+nu_harm_real(i)*cubic(i,F,F)**2/(4._rk*nu_im**2+nu_harm_real(i)**2)
  sumtmp=sumtmp+nu_im          *cubic(i,i,F)**2/(nu_im**2+4._rk*nu_harm_real(i)**2)
enddo
do i=1,nreal
  do j=i+1,nreal
    sumtmp2=sumtmp2+cubic(i,j,F)**2*nu_harm_real(i)*nu_harm_real(j)*nu_im / &
           & ((nu_im**2+(nu_harm_real(i)+nu_harm_real(j))**2)*(nu_im**2+(nu_harm_real(i)-nu_harm_real(j))**2))
  enddo
enddo
E0sp= -quartic(F,F,F)/64._rk -7._rk/576._rk*cubic(F,F,F)**2/nu_im + 3._rk/64._rk*sumtmp - sumtmp2/4._rk

! SP contribution to ZPVE
sumtmp=0._rk
sumtmp2=0._rk
do i=1,nreal
  sumtmp=sumtmp + nu_harm_real(i)*cubic(i,F,F)**2/(4._rk*nu_im**2+nu_harm_real(i)**2)
  !sumtmp=sumtmp+cubic(i,i,F)**2*(4._rk*nu_harm_real(i)**2+3._rk*nu_im**2)/(32._rk*nu_im*(nu_im**2+4._rk*nu_harm_real(i)**2))
enddo
do i=1,nreal
  do j=1,nreal
    sumtmp2=sumtmp2 + cubic(i,i,F)*cubic(j,j,F)/nu_im
    sumtmp2=sumtmp2 + 2._rk*nu_im*cubic(i,j,F)**2/(nu_im**2+(nu_harm_real(i)+nu_harm_real(j))**2)
  enddo
enddo

zpve_sp= -quartic(F,F,F)/64._rk -7._rk/576._rk*cubic(F,F,F)**2/nu_im + &
           & 3._rk/64._rk*sumtmp + 1._rk/32._rk*sumtmp2

zpve_sp_explicit_Xcont=sum(sum(Xsp(1:nreal,1:nreal),dim=2),dim=1)/4._rk
zpve_sp_explicit=E0sp+zpve_sp_explicit_Xcont

write(stdout,*) '.................................'
call vector_output(nu_harm_real,stdout,'F20.12','Real vib. frequencies')
write(stdout,*) 'Imaginary frequency: ', nu_im
write(stdout,*) '.................................'

if(printl>=6) then
  write(stdout,'(A)') 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  write(stdout,'(A)') 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  write(stdout,'(A)') ' '
  write(stdout,'(A)') ' --------------------------------'
  write(stdout,'(A)') ' vpt2_saddle_point debug output '
  write(stdout,'(A)') ' --------------------------------'
  write(stdout,'(A)') ' '
  call matrix_output(Xsp(1:nreal,1:nreal),stdout,'F20.12','SP contribution to X matrix of bound modes')
  write(stdout,'(A)') ' '
  write (stdout,'(A,F20.12)') 'Diagonal anharmonic constant of transition mode x_FF (cm-1): ', xFF
  write(stdout,'(A)') ' '
  write (stdout,'(A,F20.12)') 'Imaginary off-diagonal anharmonic coupling constants of transition mode x_iF/(-i) (cm-1): '
  write(stdout,'(A)') ' '
  do i=1,nreal
    write(stdout,'(I4,1X,F20.12)') i, xiF(i)
  enddo
  write(stdout,'(A)') ' '
  write(stdout,'(A)') '**************************'
  write(stdout,'(A)') 'E0/ZPVE contributions: '
  write(stdout,'(A)') '**************************'
  write(stdout,'(A)') ' '
  write (stdout,'(A,F20.12)') 'SP contr. to ZPVE, direct (cm-1):             ', zpve_sp
  write (stdout,'(A,F20.12)') 'SP contr. to ZPVE, explicit summation (cm-1): ', zpve_sp_explicit
  write (stdout,'(A,F20.12)') 'SP contr. to E0 (cm-1):                       ', E0sp
  write (stdout,'(A,F20.12)') 'SP contr. to ZPVE via X matrix (cm-1):        ', zpve_sp_explicit_Xcont
  write(stdout,'(A)') ' '
  write(stdout,'(A)') 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  write(stdout,'(A)') 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
end if
end subroutine vpt2_saddle_point

! ****************************************
! ****************************************
! ****************************************

! Calculate ZPVE from E0 (constant anharmonic term) and 1/4*sum_i,j<i (X_ij)

subroutine vpt2_get_zpve(neff,nu_harm,E0,X,zpve_anh,zpve_harm,saddle_point)
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), intent(in) :: E0
real(rk), intent(in), dimension(neff,neff) :: X
real(rk), intent(out) :: zpve_anh, zpve_harm
logical, intent(in), optional :: saddle_point

integer :: nreal

if (present(saddle_point)) then
  if (saddle_point) then
    nreal=neff-1
  else
    nreal=neff
  endif
else
  nreal=neff
endif

zpve_harm=sum(nu_harm(1:nreal))/2._rk
zpve_anh=zpve_harm + E0 + sum(sum(X(1:nreal,1:nreal),dim=2),dim=1)/4._rk

return
end subroutine vpt2_get_zpve

! ****************************************
! ****************************************
! ****************************************

! Calculate ZPVE directly using expression from
! Michael S. Schuurman, Wesley D. Allen, Paul von Ragué Schleyer, and Henry F. Schaefer
! The Journal of Chemical Physics 122, 104302 (2005); doi: 10.1063/1.1853377
!
! Direct, resonance-free calculation of the anharmonic zero-point vibrational energy
!  (as opposed to 'indirect' calculation via constant term E0 and X matrix elements,
!   which can suffer from resonance contributions. It can be shown that the resonant terms
!   cancel exactly for the ZPVE)
! 
! Corresponds to E0 + 1/4*sum_i,j<i (X_ij), substituting all the VPT2 expressions and then simplifying.
! The resulting expression contains no resonance terms, and is therefore universally applicable.
! Coriolis corrections are included as well

subroutine vpt2_get_zpve_direct(neff,nvrt,nu_harm,normal_mode_vectors,cubic,quartic,A,B,C,coriolis,zpve_anh,zpve_harm)
use dlf_global, only: stdout
implicit none
integer, intent(in) :: neff,nvrt
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
real(rk), intent(in)  :: A,B,C
real(rk), dimension(3,neff,neff), intent(in) :: coriolis
real(rk), intent(out) :: zpve_anh, zpve_harm

real(rk) :: anh,sumtmp,innersumtmp
real(rk), dimension(3) :: rc
integer :: i,j,k

rc(1)=A
rc(2)=B
rc(3)=C

zpve_harm=sum(nu_harm)/2._rk

sumtmp=0._rk
do i=1,neff
  do j=1,neff
    do k=1,neff
      sumtmp=sumtmp -cubic(i,i,k)*cubic(j,j,k)/(32._rk*nu_harm(k))
      sumtmp=sumtmp -cubic(i,j,k)*cubic(i,j,k)/(48._rk*(nu_harm(i)+nu_harm(j)+nu_harm(k)))
    enddo
    sumtmp=sumtmp+quartic(i,j,j)/32._rk
  enddo
enddo

anh=sumtmp

if (neff==1) then
  sumtmp=0._rk
else
  sumtmp=0._rk
  do k=1,3
    innersumtmp=0._rk
    do i=1,neff
      do j=1,i-1
        !innersumtmp=innersumtmp+coriolis(k,i,j)**2*(rc(k)*(nu_harm(i)+nu_harm(j))-(nu_harm(i)-nu_harm(j))**2)/(nu_harm(i)*nu_harm(j))
        innersumtmp=innersumtmp+coriolis(k,i,j)**2*(-(nu_harm(i)-nu_harm(j))**2)/(nu_harm(i)*nu_harm(j))
      enddo
    enddo
    sumtmp=sumtmp+rc(k)*(1._rk+innersumtmp)
  enddo
  sumtmp=-sumtmp/4._rk
endif

write(stdout,*) 'ZPVE, direct, Coriolis contribution: ', sumtmp

anh=anh+sumtmp

zpve_anh=zpve_harm+anh

return
end subroutine vpt2_get_zpve_direct

! ****************************************
! ****************************************
! ****************************************

! Debug/test routine to inspect individual terms from the routine vpt2_get_zpve_direct
!

subroutine vpt2_get_zpve_direct_analyze_terms(neff,nvrt,nu_harm,normal_mode_vectors,cubic,quartic, &
                                          &   A,B,C,coriolis,E0,X,zpve_anh,zpve_harm)
use dlf_global, only: stdout,printl
implicit none
integer, intent(in) :: neff,nvrt
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), dimension(nvrt,neff),  intent(in)  :: normal_mode_vectors
real(rk), dimension(neff,neff,neff), intent(in) :: cubic,quartic
real(rk), intent(in)  :: A,B,C
real(rk), dimension(3,neff,neff), intent(in) :: coriolis
real(rk), intent(in) :: E0
real(rk), intent(in), dimension(neff,neff) :: X
real(rk), intent(out) :: zpve_anh, zpve_harm

real(rk) :: anh,sumtmp,innersumtmp,ft1,ft2,fquartic
real(rk) :: ft1ex,ft2ex
real(rk) :: sumtmp2,sumtmp3,sumtmp4,Zx
real(rk), dimension(3) :: rc
real(rk), dimension(neff,neff,neff) :: D
integer :: i,j,k

rc(1)=A
rc(2)=B
rc(3)=C

zpve_harm=sum(nu_harm)/2._rk

sumtmp=0._rk
do i=1,neff
  do j=1,neff
    do k=1,neff
      sumtmp=sumtmp -cubic(i,i,k)*cubic(j,j,k)/(32._rk*nu_harm(k))
      sumtmp=sumtmp -cubic(i,j,k)*cubic(i,j,k)/(48._rk*(nu_harm(i)+nu_harm(j)+nu_harm(k)))
    enddo
    sumtmp=sumtmp+quartic(i,j,j)/32._rk
  enddo
enddo

anh=sumtmp

ft1=0._rk
ft2=0._rk
fquartic=0._rk

sumtmp=0._rk
do i=1,neff
  do j=1,neff
    sumtmp=sumtmp+quartic(i,j,j)/32._rk
  enddo
enddo
fquartic=sumtmp

sumtmp=0._rk
sumtmp2=0._rk
sumtmp3=0._rk

do i=1,neff
  sumtmp=sumtmp+cubic(i,i,i)**2/nu_harm(i)
enddo

do i=1,neff
  do j=1,neff
    if (j.eq.i) cycle
    sumtmp2=sumtmp2 + nu_harm(i)*cubic(i,j,j)**2/(4._rk*nu_harm(j)**2-nu_harm(i)**2)
  enddo
enddo

do i=1,neff
  do j=1,neff
    do k=1,neff
      sumtmp3=sumtmp3 + cubic(i,i,k)*cubic(j,j,k)/nu_harm(k)
    enddo
  enddo
enddo

ft1=-1._rk/144._rk*sumtmp + 1._rk/16._rk*sumtmp2 - 1._rk/32._rk * sumtmp3

sumtmp=0._rk
sumtmp2=0._rk
sumtmp3=0._rk

do i=1,neff
  sumtmp=sumtmp+cubic(i,i,i)**2/nu_harm(i)
enddo

do i=1,neff
  do j=1,neff
    if (j.eq.i) cycle
    sumtmp2=sumtmp2 + cubic(i,i,j)**2/(2._rk*nu_harm(i)+nu_harm(j)) * 3._rk/48._rk
    sumtmp2=sumtmp2 + cubic(i,i,j)**2*nu_harm(i)/(4._rk*nu_harm(i)**2-nu_harm(j)**2) * (-1._rk/8._rk)
  enddo
enddo

do i=1,neff
  do j=1,neff
    do k=1,neff
      sumtmp3=sumtmp3 + cubic(i,j,k)**2/(nu_harm(i)+nu_harm(j)+nu_harm(k))
    enddo
  enddo
enddo

ft2=1._rk/144._rk*sumtmp + sumtmp2 - 1._rk/48._rk * sumtmp3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

sumtmp=0._rk
sumtmp2=0._rk
sumtmp3=0._rk
sumtmp4=0._rk

do i=1,neff
  sumtmp=sumtmp + cubic(i,i,i)**2/nu_harm(i)
enddo

do i=1,neff
  do j=1,neff
    if (i.eq.j) cycle
    sumtmp2=sumtmp2+nu_harm(i)*cubic(i,j,j)**2/(4._rk*nu_harm(j)**2-nu_harm(i)**2)
  enddo
enddo

do i=1,neff
  do j=1,i-1
    do k=1,neff
      sumtmp3=sumtmp3+cubic(i,i,k)*cubic(k,j,j)/nu_harm(k)
    enddo
  enddo
enddo

do i=1,neff
  do k=1,neff
    sumtmp4=sumtmp4+cubic(i,i,k)**2*(8._rk*nu_harm(i)**2-3._rk*nu_harm(k)**2)/(nu_harm(k)*(4._rk*nu_harm(i)**2-nu_harm(k)**2))
  enddo
enddo

ft1ex= - 7._rk/576._rk*sumtmp + 3._rk/64._rk*sumtmp2 - 1._rk/16._rk * sumtmp3 - 1._rk/64._rk * sumtmp4

do i=1,neff
  do j=1,neff
    do k=1,neff
      D(i,j,k)=(-nu_harm(i)**2+(nu_harm(j)-nu_harm(k))**2)*(-nu_harm(i)**2+(nu_harm(j)+nu_harm(k))**2)
    enddo
  enddo
enddo

sumtmp=0._rk
sumtmp2=0._rk

do i=1,neff
  do j=i+1,neff
    do k=j+1,neff
      sumtmp=sumtmp+cubic(i,j,k)**2*nu_harm(i)*nu_harm(j)*nu_harm(k)/D(i,j,k)
    enddo
  enddo
enddo

do i=1,neff
  do j=1,i-1
    do k=1,neff
      sumtmp2=sumtmp2+cubic(i,j,k)**2*nu_harm(k)*(nu_harm(k)**2-nu_harm(i)**2-nu_harm(j)**2)/D(i,j,k)
    enddo
  enddo
enddo

ft2ex= -sumtmp/4._rk - sumtmp2/8._rk

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Zx=0._rk
do i=1,neff
  do j=1,i
    Zx=Zx+X(i,j)/4._rk
  enddo
enddo

if(printl>=6) then
  write(stdout,'(A)') ''
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') ''
  write(stdout,'(A)') ' ZPVE debug analysis'
  write(stdout,'(A)') ''
  write(stdout,*)     ' FT1ex ', ft1ex
  write(stdout,*)     ' FT1 = ', ft1
  write(stdout,*)     ' FT2ex ', ft2ex
  write(stdout,*)     ' FT2 = ', ft2
  write(stdout,*)     ' FQ  = ', fquartic
  write(stdout,'(A)') ' -------------------------------'
  write(stdout,*)     ' sum = ', ft1+ft2+fquartic
  write(stdout,'(A)') ''
  write(stdout,*)     ' anh = ', anh
  write(stdout,'(A)') ''
  write(stdout,*)     ' E0   = ', E0
  write(stdout,*)     ' Zx   = ', Zx
  write(stdout,*)     ' E0+Zx= ', E0+Zx
  write(stdout,'(A)') ''
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
  write(stdout,'(A)') ''
end if

!read(*,*)

! Coriolis (Zkinetic)
if (neff==1) then
  sumtmp=0._rk
else
  sumtmp=0._rk
  do k=1,3
    innersumtmp=0._rk
    do i=1,neff
      do j=1,i-1
        !innersumtmp=innersumtmp+coriolis(k,i,j)**2*(rc(k)*(nu_harm(i)+nu_harm(j))-(nu_harm(i)-nu_harm(j))**2)/(nu_harm(i)*nu_harm(j))
        innersumtmp=innersumtmp+coriolis(k,i,j)**2*(-(nu_harm(i)-nu_harm(j))**2)/(nu_harm(i)*nu_harm(j))
      enddo
    enddo
    sumtmp=sumtmp+rc(k)*(1._rk+innersumtmp)
  enddo
  sumtmp=-sumtmp/4._rk
endif

write(stdout,*) 'ZPVE, direct, Coriolis contribution: ', sumtmp

anh=anh+sumtmp

zpve_anh=zpve_harm+anh

return
end subroutine vpt2_get_zpve_direct_analyze_terms

! ****************************************
! ****************************************
! ****************************************

! Get anharmonic fundamental frequencies 
! (Energy term differences between states with quantum number
! vectors n = ( 0 0 ... 1 ... 0 ) and vibrational ground state 
! [ n = ( 0 0 ... 0 ... 0 ) ]
! These may be compared to spectroscopically determined frequencies

subroutine vpt2_fundamentals(neff,nu_harm,X,fundamentals,saddle_point)
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), intent(in), dimension(neff,neff) :: X
real(rk), intent(out), dimension(neff) :: fundamentals
logical, intent(in), optional :: saddle_point

real(rk), dimension(neff,neff) :: Xfilled
integer :: i,j
integer :: nreal

if (present(saddle_point)) then
  if (saddle_point) then
    nreal=neff-1
  else
    nreal=neff
  endif
else
  nreal=neff
endif

Xfilled=X
do i=1,neff
  do j=i+1,neff
    Xfilled(i,j)=Xfilled(j,i)
  enddo
enddo

fundamentals(:)=0._rk
do i=1,nreal
  fundamentals(i)=nu_harm(i)+1.5_rk*Xfilled(i,i)+0.5_rk*sum(Xfilled(i,:))
enddo

return
end subroutine vpt2_fundamentals


! ****************************************
! ****************************************
! ****************************************

! VPT2 term energy for arbitrary quantum number combination.
! E = E_0 + Sum_{i=1}^{3N-6} [ ny_i * (n_i + 1/2) ] + Sum_{i=1}^{3N-6} Sum_{j=1}^{i} [ X_ij * (n_i + 1/2)*(n_j + 1/2) ]

subroutine vpt2_term_energy(neff,nu_harm,E0,qn,X,ener,ener_rel_to_gs,saddle_point)
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), intent(in) :: E0
real(rk), intent(in), dimension(neff,neff) :: X
integer, intent(in), dimension(neff) :: qn
real(rk), intent(out) :: ener, ener_rel_to_gs
logical, intent(in), optional :: saddle_point

integer :: i,j,nreal
real(rk) :: sumtmp,zpve_anh,zpve_harm

if (present(saddle_point)) then
  if (saddle_point) then
    nreal=neff-1
  else
    nreal=neff
  endif
else
  nreal=neff
endif

sumtmp=0._rk
do i=1,nreal
  sumtmp=sumtmp+(real(qn(i))+0.5_rk)*nu_harm(i)
enddo

ener=E0+sumtmp

sumtmp=0._rk

do i=1,nreal
  do j=1,i
    sumtmp=sumtmp+(real(qn(i))+0.5_rk)*(real(qn(j))+0.5_rk)*X(i,j)
  enddo
enddo

ener=ener+sumtmp

call vpt2_get_zpve(nreal,nu_harm(1:nreal),E0,X(1:nreal,1:nreal),zpve_anh,zpve_harm)
ener_rel_to_gs=ener-zpve_anh

return
end subroutine vpt2_term_energy

! ****************************************
! ****************************************
! ****************************************

! Simplified version of vpt2_term_energy
! * Energy is relative to vibrational ground state, not PES minimum.
! * Function can be vectorized

pure function E_vpt2(neff,nu_harm,qn,X) result(ener_rel_to_gs)
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff) :: nu_harm
real(rk), intent(in), dimension(neff,neff) :: X
integer, intent(in), dimension(neff) :: qn
real(rk) :: ener_rel_to_gs

integer :: i,j
real(rk) :: sumtmp

sumtmp=0._rk
do i=1,neff
  sumtmp=sumtmp+real(qn(i))*nu_harm(i)
enddo

ener_rel_to_gs=sumtmp

sumtmp=0._rk

do i=1,neff
  do j=1,i
    sumtmp=sumtmp+real(qn(i))*real(qn(j))*X(i,j)
  enddo
enddo

ener_rel_to_gs=ener_rel_to_gs+sumtmp

return
end function E_vpt2

! ****************************************
! ****************************************
! ****************************************

! Alternative version of E_vpt2 which is able 
! to handle cases (approximately), where an increase 
! in quantum number leads to a decrease in energy (a typical
! unphysical VPT2 artifact at high energies). Controlled via 
! SPT ("simple perturbation theory") mask 

pure function E_vpt2_hybrid(neff,nu_harm,nu_funda_vpt2,spt_mask,qn,X) result(ener_rel_to_gs)
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff) :: nu_harm, nu_funda_vpt2
logical, intent(in), dimension(neff)  :: spt_mask
real(rk), intent(in), dimension(neff,neff) :: X
integer, intent(in), dimension(neff) :: qn
real(rk) :: ener_rel_to_gs

integer :: i,j
real(rk) :: sumtmp
logical, dimension(neff) :: vpt2_mask

vpt2_mask(:)=.not. spt_mask(:)

!sumtmp=0._rk
!do i=1,neff
!  sumtmp=sumtmp+real(qn(i))*nu_harm(i)
!enddo
sumtmp=sum(real(qn(:))*nu_harm(:),mask=vpt2_mask)
sumtmp=sumtmp+sum(real(qn(:))*nu_funda_vpt2(:),mask=spt_mask)

ener_rel_to_gs=sumtmp

sumtmp=0._rk

do i=1,neff
  do j=1,i
    if (spt_mask(i) .or. spt_mask(j)) cycle
    sumtmp=sumtmp+real(qn(i))*real(qn(j))*X(i,j)
  enddo
enddo

ener_rel_to_gs=ener_rel_to_gs+sumtmp

return
end function E_vpt2_hybrid

! ****************************************
! ****************************************
! ****************************************

end module dlf_vpt2_terms


