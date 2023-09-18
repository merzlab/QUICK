! Contains routines for harmonic normal coordinate analysis

module dlf_vpt2_freq
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none

contains

!***************************************************
!***************************************************

! Do frequency analysis, given the coordinates and Hessian 
! of a stationary point (minimum or 1st-order saddle point).
! Input data is assumed to be in atomic units. Frequencies 
! will be returned in cm^-1

! Expect Hessian and coordinates in order xyzxyz...
! Normal mode vectors are output in normal_mode_vectors in mass-weighted cartesians
! Additional vector cart_disp_vectors contains mass-unweighted normal modes

subroutine get_frequencies_mw_ts_or_min(nat,nzero,coo1d,atsymb,mv,hess,freqs, &
                    & eigenvalues_au,normal_mode_vectors,nimag,proj,dryrun)
use dlf_vpt2_project, only: proj_trans_rot_dir_mw, unscramble_transrot_eigenvectors_mw
use dlf_sort_module, only: dlf_sort_shell_ind
use dlf_vpt2_utility, only: error_print
use dlf_linalg_interface_mod
use dlf_constants
use dlf_global, only: stdout
implicit none
integer, intent(in)  :: nat,nzero
integer, intent(out) :: nimag
real(rk), dimension(3*nat),intent(in) :: coo1d
character(2), intent(in), dimension(nat) :: atsymb
real(rk), intent(in), dimension(nat) :: mv
real(rk), intent(in), dimension(3*nat,3*nat)    :: hess
real(rk), intent(out), dimension(3*nat,3*nat-nzero) :: normal_mode_vectors
real(rk), intent(out), dimension(3*nat-nzero) :: freqs,eigenvalues_au
logical, intent(in), optional :: proj,dryrun

integer :: i,ix,iy,iz,j,zcounter,pcounter,ncounter,jtarget
real(rk) :: sgn,zerthresh
real(rk), dimension(3*nat,3*nat-nzero) :: cart_disp_vectors
real(rk), dimension(3,nat) :: coo
real(rk), dimension(3*nat) :: eigval,zervec,abseval,eigval_sorted,freqs_all
real(rk), dimension(3*nat) :: eigenvalues_au_all
real(rk), dimension(3*nat,3*nat) :: hessmwpr, eigvec, eigvec_sorted, eigvec_tmp
real(rk), dimension(3*nat,3*nat) :: hessmw
integer, dimension(3*nat) :: sortind
logical :: proj_int,dryrun_int
real(rk) :: amu2au
real(rk) :: au2cmi

call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)
call dlf_constants_get('AMU',amu2au)

proj_int=.false.
if (present(proj)) then
  proj_int=proj
endif
dryrun_int=.false.
if (present(dryrun)) then
  dryrun_int=dryrun
endif

zervec=0._rk

do i=1,nat
  coo(1,i)=coo1d(3*(i-1)+1)
  coo(2,i)=coo1d(3*(i-1)+2)
  coo(3,i)=coo1d(3*(i-1)+3)
enddo

call mass_weight_hess(nat,3*nat,hess,mv,amu2au,hessmw)
if (proj_int) then
  if (nzero==5) then
    call proj_trans_rot_dir_mw(nat,3*nat,mv,hessmw,coo1d,zervec,hessmwpr,linear_in=.true.)
  else
    call proj_trans_rot_dir_mw(nat,3*nat,mv,hessmw,coo1d,zervec,hessmwpr)
  endif
else
  hessmwpr=hessmw
endif

if (.not. dryrun_int) then
  write(stdout,'(A)') '*******************************************************************'
  write(stdout,'(A)') '****************  Frequency Analysis (Cartesian)  *****************'
  write(stdout,'(A)') '*******************************************************************'
  if (proj_int) then
    write(stdout,'(A)') '******** Rotations and translations  projected from Hessian *******'
    write(stdout,'(A)') '*******************************************************************'
  endif
endif

call hessian_eigenvalues(3*nat,hessmwpr,eigval,eigvec)

abseval=abs(eigval)

! Make sort index for abs. eigenvalues

call dlf_sort_shell_ind(abseval,sortind)

zerthresh=0._rk
do i=1,nzero
  zerthresh=max(zerthresh,abseval(sortind(i)))
enddo

! Make sort index for signed eigenvalues

call dlf_sort_shell_ind(eigval,sortind)

! Sort eigenvalues, eigenvectors, so that first 5/6
! correspond to rotations/translations

eigval_sorted=0._rk
eigvec_sorted=0._rk
freqs_all=0._rk
freqs=0._rk
eigenvalues_au_all=0._rk
eigenvalues_au=0._rk
normal_mode_vectors=0._rk
cart_disp_vectors=0._rk
nimag=0
zcounter=0
ncounter=0
pcounter=0
do i=1,3*nat
  j=sortind(i)
  if (abs(eigval(j)).le.zerthresh) then
    zcounter=zcounter+1
    jtarget=zcounter
    sgn=sign(1._rk,eigval(j))
    if (zcounter .gt. nzero) call error_print('get_frequencies_mw_ts_or_min: wrong no. of zero eigenvalues')
  else
    if(eigval(j).lt.0._rk) then
      sgn=-1._rk
      ncounter=ncounter+1
      jtarget=3*nat - (ncounter-1)
    else
      sgn=1._rk
      pcounter=pcounter+1
      jtarget=nzero+pcounter
    endif
    if (ncounter+pcounter .gt. 3*nat-nzero) call error_print('get_frequencies_mw_ts_or_min: wrong no. of non-zero eigenvalues')
  endif
  eigval_sorted(jtarget)=eigval(j)
  freqs_all(jtarget)=sgn*sqrt(abs(eigval(j))/amu2au)*au2cmi
  eigenvalues_au_all(jtarget)=eigval(j)/amu2au
  eigvec_sorted(1:3*nat,jtarget)=eigvec(1:3*nat,j)
enddo

nimag=ncounter
freqs(1:3*nat-nzero)=freqs_all(nzero+1:3*nat)
eigenvalues_au(1:3*nat-nzero)=eigenvalues_au_all(nzero+1:3*nat)
if (nzero==5) then
  call unscramble_transrot_eigenvectors_mw(nat,3*nat,mv,eigvec_sorted,coo1d,eigvec_tmp,linear_in=.true.)
else
  call unscramble_transrot_eigenvectors_mw(nat,3*nat,mv,eigvec_sorted,coo1d,eigvec_tmp)
endif
eigvec_sorted=eigvec_tmp
normal_mode_vectors(1:3*nat,1:3*nat-nzero)=eigvec_sorted(1:3*nat,nzero+1:3*nat)

if (.not. dryrun_int) then
  write(stdout,*) ''
  write(stdout,'(A)') '--------------------------'
  write(stdout,'(A)') '  Frequencies'
  write(stdout,'(A)') '--------------------------'
  write(stdout,*) ''
  open(999,file='statpt_freq.molden')
  write(999,'(A)') '[Molden Format]'
  write(999,'(A)') '[FREQ]'
  write(stdout,'(A10,A15)') 'Index', 'Freq (cm^-1)'
  do i=1,3*nat
    write(stdout,'(i10,f15.5)')i,freqs_all(i)
    write(999,*) freqs_all(i)
  enddo
endif

! un-mass-weight the normal modes
do i=1,nat
  ix=3*(i-1)+1
  iy=3*(i-1)+2
  iz=3*(i-1)+3
  eigvec_sorted(ix,:)=eigvec_sorted(ix,:)*sqrt(amu2au/mv(i))
  eigvec_sorted(iy,:)=eigvec_sorted(iy,:)*sqrt(amu2au/mv(i))
  eigvec_sorted(iz,:)=eigvec_sorted(iz,:)*sqrt(amu2au/mv(i))
  cart_disp_vectors(ix,:)=normal_mode_vectors(ix,:)*sqrt(amu2au/mv(i))
  cart_disp_vectors(iy,:)=normal_mode_vectors(iy,:)*sqrt(amu2au/mv(i))
  cart_disp_vectors(iz,:)=normal_mode_vectors(iz,:)*sqrt(amu2au/mv(i))
enddo

! renormalize the eigenvectors column-wise
do j=1,3*nat
  eigvec_sorted(:,j)=eigvec_sorted(:,j)/dlf_vector_norm(eigvec_sorted(:,j))
enddo

if (.not. dryrun_int) then
  write(999,'(A)') '[FR-COORD]'
  do i=1,nat
    write(999,'(a,3f22.15)')atsymb(i),coo(1,i),coo(2,i),coo(3,i)
  enddo
  write(999,'(A)') '[FR-NORM-COORD]'
  do i=1,3*nat
    write(999,*) 'vibration', i
    write(999,'(3ES17.6)') (eigvec_sorted(j,i),j=1,3*nat)
  enddo
  close(999)
endif

return
end subroutine get_frequencies_mw_ts_or_min

!***************************************************
!***************************************************

! Just print frequencies, for the case of reading a previous checkpoint file

subroutine print_frequencies(nfreqs,freqs)
use dlf_global, only: stdout
implicit none
integer, intent(in)  :: nfreqs
real(rk), intent(in), dimension(nfreqs) :: freqs

integer :: i

write(stdout,'(A)') '*******************************************************************'
write(stdout,'(A)') '***********  Summary of previous frequency analysis ***************'
write(stdout,'(A)') '*******************************************************************'

write(stdout,'(A10,A15)') 'Index', 'Freq (cm^-1)'
do i=1,nfreqs
  write(stdout,'(i10,f15.5)') i, freqs(i)
enddo

return
end subroutine print_frequencies

!***************************************************
!***************************************************

! Get reduced mass corresponding to a certain normal mode q

function reduced_mass(nat,N,q,mv)
use dlf_vpt2_utility, only: checkeq
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: nat, N
real(rk), dimension(N), intent(in) :: q
real(rk), dimension(nat), intent(in) :: mv
real(rk) :: reduced_mass

real(rk), dimension(N) :: qnorm
integer :: i,atind

call checkeq(3*nat,N,'reduced_mass')

qnorm=q/dlf_vector_norm(q)
reduced_mass=0._rk

do i=1,N
  atind=mod(i,nat)
  if (atind.eq.0) atind=nat
  reduced_mass=reduced_mass+mv(atind)*qnorm(i)**2
enddo

return
end function reduced_mass

!***************************************************
!***************************************************

! Wrapping routine for the LAPACK/BLAS eigenvalue routines. 
! Note: instead of direct call to linear algebra routines, 
! the DL-FIND wrapper matrix_diagonalise is used.

subroutine hessian_eigenvalues(N,H,lambda,evec)
use dlf_vpt2_utility, only: error_print, matrix_output, vector_output
use dlf_linalg_interface_mod, only: dlf_vector_norm
use dlf_sort_module, only: dlf_sort_shell_ind
implicit none
integer, intent(in) :: N
real(rk), dimension(:,:), intent(in), contiguous :: H
real(rk), dimension(:), intent(out), contiguous  :: lambda
real(rk), dimension(:,:), intent(out), contiguous  :: evec
interface
  subroutine dlf_matrix_diagonalise(N,a,evals,evecs)
    use dlf_parameter_module, only: rk
    implicit none
    integer  ,intent(in)    :: N
    real(rk) ,intent(in)    :: A(N,N)
    real(rk) ,intent(out)   :: evals(N)
    real(rk) ,intent(out)   :: evecs(N,N)
  end subroutine dlf_matrix_diagonalise
end interface

integer :: i,lwork,lpk_info
real(rk), dimension(N) :: lambda_raw,lambda_imag
real(rk), dimension(1,N) :: vl_dummy
real(rk), dimension(N,N) :: vr
real(rk), dimension(:), allocatable :: work
integer, dimension(N) :: sortind

if (size(lambda).ne.N) call error_print('Size of lambda doesnt match N in hessian_eigenvalues.')
if (size(H,1).ne.N .or. size(H,2).ne.N) call error_print('Size of H doesnt match N in hessian_eigenvalues.')
if (size(evec,1).ne.N .or. size(evec,2).ne.N) call error_print('Size of evec doesnt match N in hessian_eigenvalues.')

!call matrix_output(H,6,'F20.12','hessian to diagonalize...')

call dlf_matrix_diagonalise(N,H,lambda,evec)

!if (.true.) then
!  allocate(work(1))
!  call dgeev('N','V',N,H,N,lambda_raw,lambda_imag,vl_dummy,1,vr,N,work,-1,lpk_info)
!  lwork=int(work(1))
!  deallocate(work)
!  allocate(work(lwork))
!  call dgeev('N','V',N,H,N,lambda_raw,lambda_imag,vl_dummy,1,vr,N,work,lwork,lpk_info)
!  deallocate(work)
!  write(*,*) "hessian_eigenvalues, LAPACK status: ",lpk_info
!  call dlf_sort_shell_ind(lambda_raw,sortind)
!  do i=1,N
!    lambda(i)=lambda_raw(sortind(i))
!    evec(:,i)=vr(:,sortind(i))
!  enddo
!  call vector_output(lambda_imag,6,'F20.12','imaginary components of eigenvalues')
!  call matrix_output(evec,6,'F20.12','eigenvectors')
!endif

do i=1,N
  evec(:,i)=evec(:,i)/dlf_vector_norm(evec(:,i))
enddo

return

end subroutine hessian_eigenvalues

!***************************************************
!***************************************************

! Convert Hessian from regular to mass-weighted Cartesians

subroutine mass_weight_hess(natom,Ndim,Hin,mv,my,Hout)
use dlf_vpt2_utility, only: error_print
use dlf_linalg_interface_mod
implicit none

integer, intent(in) :: natom,Ndim
real(rk), dimension(:,:), intent(in) :: Hin
real(rk), dimension(:), intent(in) :: mv
real(rk), intent(in) :: my
real(rk), dimension(:,:), intent(out) :: Hout

integer :: i,i1,i2,i3
real(rk) :: tmp
real(rk),dimension(size(Hin,dim=1),size(Hin,dim=2)) :: invsqrtmv

if (size(mv).ne.natom) call error_print('Size of mv doesnt match nat in mass_weight_hess')
if (size(Hin,1).ne.Ndim  .or. size(Hin,2).ne.Ndim)  call error_print('Size of Hin doesnt match Ndim in mass_weight_hess.')
if (size(Hout,1).ne.Ndim .or. size(Hout,2).ne.Ndim) call error_print('Size of Hout doesnt match Ndim in mass_weight_hess.')
if (Ndim.ne.3*natom) call error_print('Ndim != 3*nat in mass_weight_hess')

invsqrtmv=0._rk
do i=1,natom
  tmp=1._rk/sqrt(mv(i))
  i1=3*(i-1)+1
  i2=3*(i-1)+2
  i3=3*(i-1)+3
  invsqrtmv(i1,i1)=tmp
  invsqrtmv(i2,i2)=tmp
  invsqrtmv(i3,i3)=tmp
enddo

Hout=dlf_matrix_ortho_trans(invsqrtmv,Hin,0)
Hout=Hout*my

return
end subroutine mass_weight_hess

!***************************************************
!***************************************************

end module dlf_vpt2_freq




