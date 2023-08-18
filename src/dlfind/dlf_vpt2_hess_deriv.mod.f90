! Contains routines for calculating numerical 2nd, 3rd and 4th 
! derivatives of the potential energy surface.

!!#define TEST_RANDOM_HESSIAN_ORDER
!!#define TEST_FIXED_HESSIAN_ORDER
!!#define TEST_MODIFIED_HESSIAN_ORDER
!!#define VPT2_HESS_DEBUG_OUT
!!#define VPT2_GRAD_DEBUG_OUT
!!#define CHECK_ACTUAL_ARRAY_SHAPES

module dlf_vpt2_hess_deriv
use dlf_parameter_module
use dlf_global, only: glob, stdout
implicit none

integer, save :: nbas_save
real(rk), allocatable, dimension(:), save   :: coefficients_save
real(rk), allocatable, dimension(:,:), save :: sqrtmass_save
real(rk), allocatable, dimension(:,:), save :: normal_modes_nonmw_save
real(rk), allocatable, dimension(:,:), save :: normal_modes_save
real(rk), allocatable, dimension(:,:), save :: hessian_save
real(rk), allocatable, dimension(:,:,:), save :: cubic_save, quartic_sd_save

contains

! ****************************************
! ****************************************
! ****************************************

! Convert between different ordering of indices for
! Hessian displacements

function modify_displ_order(k,nvareff) result(knu)
implicit none
integer, intent(in) :: k, nvareff
integer :: knu

integer :: indx

if (k<0) then
  indx=nvareff+k+1
  knu=int((indx+1)/2)
  if (mod(indx,2)==0) knu=-knu
  return
elseif (k>0) then
  indx=nvareff+k
  knu=int((indx+1)/2)
  if (mod(indx,2)==0) knu=-knu
  return
else
  knu=0
  return
endif

return
end function modify_displ_order

! ****************************************
! ****************************************
! ****************************************

! Initialize master punch file, for the case that no usable information from previous 
! runs has been found by the read_punch_files routine.

subroutine write_master_punch_file_header(nat,nvar,nvareff,atsym,coord_ref,timestamp)
  use dlf_vpt2_utility, only: f_rename, vector_output, matrix_output
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
  implicit none
  integer, intent(in)  :: nat,nvar,nvareff
  character(len=2), intent(in), dimension(nat) :: atsym
  real(rk), intent(in), dimension(nvar) :: coord_ref
  character(len=23), intent(out) :: timestamp
  
  integer, dimension(8) :: time_values
  logical :: punch_exists
  character(len=500) :: fn_punch,fn_punch_rename,tag
  character(len=23)  :: timestamp_old
  integer :: i,ios
  
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(coord_ref) ], &
     &  'dim. mismatch, write_master_punch_file_header')
  call checkeq( [nat, size(atsym) ], &
     &  'dim. mismatch, write_master_punch_file_header')
  write(stdout,*) 'write_master_punch_file_header: array shape check successful.'
#endif
  
  call date_and_time(values=time_values)
  write(timestamp,'(I4.4,5(A1,I2.2),A1,I3.3)') time_values(1),'-',time_values(2),'-', & 
                             & time_values(3),'_',time_values(5),'.',time_values(6), & 
                             & '.',time_values(7),'_',time_values(8)
  
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'
  inquire(file=trim(adjustl(fn_punch)),exist=punch_exists)
  
  ! File exists, but it does not belong to the current structure => rename existing file
  if (punch_exists) then
    fn_punch_rename=''
    open(1159,file=trim(adjustl(fn_punch)),action='read')
    read(1159,'(A)',iostat=ios) tag
    if (trim(adjustl(tag))=='$TIMESTAMP_MASTER') then
      timestamp_old=''
      read(1159,'(A)',iostat=ios) timestamp_old
      if (len_trim(timestamp_old)==0 .or. ios/=0) then
        write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_corrupt_foundat_',timestamp,'.dat'
      else
        write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_ts_',timestamp_old,'.dat'
      endif
    else
      write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_corrupt_foundat_',timestamp,'.dat'
    endif
    close(1159)
    call f_rename(trim(adjustl(fn_punch)),trim(adjustl(fn_punch_rename)))
  endif
  
  open(1161,file=trim(adjustl(fn_punch)),action='write')
  write(1161,'(A)') '$TIMESTAMP_MASTER'
  write(1161,'(A)') timestamp
  write(1161,'(A)') '$NAT'
  write(1161,'(I0)') nat
  write(1161,'(A)') '$NVAR'
  write(1161,'(I0)') nvar
  write(1161,'(A)') '$NVAREFF'
  write(1161,'(I0)') nvareff
  write(1161,'(A)') '$ATLIST'
  do i=1,nat
    write(1161,'(A2)') atsym(i)
  enddo
  write(1161,'(A)') '$COORD'
  do i=1,nvar
    write(1161,'(ES20.12)') coord_ref(i)
  enddo
  close(1161)
  
  return
end subroutine write_master_punch_file_header

! ****************************************
! ****************************************
! ****************************************

! Central Hessian, harmonic frequencies, Hessian eigenvalues and
! normal coordinates are appended to master punch file

subroutine write_hessian_to_master_punch_file(nvar,nvareff, & 
                                & normal_modes,freqs,evals,hess0,timestamp)
  use dlf_vpt2_utility, only: vector_output, matrix_output
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
  implicit none
  integer, intent(in)  :: nvar,nvareff
  real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes
  real(rk), intent(in), dimension(nvareff) :: freqs, evals
  real(rk), intent(in), dimension(nvar,nvar) :: hess0
  character(len=23), intent(in) :: timestamp
  
  logical :: punch_exists
  character(len=500) :: fn_punch,tag
  integer :: i,ios
  
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(normal_modes,dim=1) , &
     &            size(hess0,dim=1),size(hess0,dim=2) ], &
     &  'dim. mismatch, write_hessian_to_master_punch_file')
  call checkeq( [nvareff, size(normal_modes,dim=2), &
     &           size(freqs), size(evals) ], &
     &  'dim. mismatch, write_hessian_to_master_punch_file')
  write(stdout,*) 'write_hessian_to_master_punch_file: array shape check successful.'
#endif
  
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'

  open(1161,file=trim(adjustl(fn_punch)),action='write',position='append')
  write(1161,'(A)') '$NORMAL_MODES'
  call matrix_output(normal_modes,1161,'ES24.16','__BLANK__')
  write(1161,'(A)') '$FREQS'
  call vector_output(freqs,1161,'ES24.16','__BLANK__')
  write(1161,'(A)') '$EIGENVALUES'
  call vector_output(evals,1161,'ES24.16','__BLANK__')
  write(1161,'(A)') '$HESSIAN_CART'
  write(1161,'(I0,1X,I0,1X,ES15.6)') 0, 0, 0._rk
  call matrix_output(hess0,1161,'ES24.16','__BLANK__')
  close(1161)
  
  return
end subroutine write_hessian_to_master_punch_file

! ****************************************
! ****************************************
! ****************************************

! Initialize slave punch file, for the case that no usable information from previous 
! runs has been found by the read_punch_files routine

subroutine write_slave_punch_file_header(timestamp)
  use dlf_vpt2_utility, only: f_rename
  implicit none
  character(len=23), intent(in) :: timestamp
  
  logical :: punch_exists
  character(len=500) :: fn_punch,fn_punch_rename,tag
  character(len=23)  :: timestamp_old
  integer :: ios
  
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'
  inquire(file=trim(adjustl(fn_punch)),exist=punch_exists)
  
  write(*,*) 'fn_punch: '//trim(adjustl(fn_punch))
  write(*,*) 'Exists?: ', punch_exists
  
  ! File exists, but it does not belong to the current structure => rename existing file
  if (punch_exists) then
    fn_punch_rename=''
    open(1159,file=trim(adjustl(fn_punch)),action='read')
    read(1159,'(A)',iostat=ios) tag
    if (trim(adjustl(tag))=='$TIMESTAMP_SLAVE') then
      timestamp_old=''
      read(1159,'(A)',iostat=ios) timestamp_old
      if (len_trim(timestamp_old)==0 .or. ios/=0) then
        write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_corrupt_foundat_',timestamp,'.dat'
      else
        write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_ts_',timestamp_old,'.dat'
      endif
    else
      write(fn_punch_rename,'(A,I6.6,3A)') 'dlf_vpt2_restart_proc_',glob%iam,'_bkp_corrupt_foundat_',timestamp,'.dat'
    endif
    close(1159)
    call f_rename(trim(adjustl(fn_punch)),trim(adjustl(fn_punch_rename)))
  endif
  
  open(1161,file=trim(adjustl(fn_punch)),action='write')
  write(1161,'(A)') '$TIMESTAMP_SLAVE'
  write(1161,'(A)') timestamp
  close(1161)
  write(*,*) 'File created: '//trim(adjustl(fn_punch))
  
  return
end subroutine write_slave_punch_file_header

! ****************************************
! ****************************************
! ****************************************

! For MPI runs. Collect all punch files that were found in the 
! individual rank working directories to a central directory, 
! namely the working directory of the master rank (=0).

subroutine gather_punch_files_in_rank0_dir(nvar)
  use dlf_vpt2_utility, only: f_rename, f_remove, &
      & dlf_global_char_bcast_rank0, dlf_global_log_gather_rank0, &
      & dlf_global_log_bcast_rank0,dlf_global_int_gather_rank0, &
      & dlf_global_char_gather_rank1, dlf_mpi_send, dlf_mpi_recv, &
      & error_print
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer, intent(in) :: nvar
  
  character(len=max(1000,24*nvar+100)) :: line_buffer
  integer, parameter :: max_files_per_proc=100
  integer, dimension(8) :: time_values
  character(len=23) :: timestamp, timestamp_old
  logical :: punch_exists,stamp_file_exists, ex_sig
  logical, dimension(:), allocatable :: sfe_vec
  character(len=500) :: fn_punch, fn_punch_rename, tag
  character(len=500), dimension(max_files_per_proc) :: punch_list_proc
  character(len=500), allocatable, dimension(:,:) :: punch_matrix
  integer :: i,j,nlines,curr_slave,itag,nfiles_proc,ios
  integer, allocatable, dimension(:) :: nfiles_vec
  logical, allocatable, dimension(:) :: slaveterm
  integer, parameter :: my_tag_initiate        = 1010
  integer, parameter :: my_tag_nlines          = 2020
  integer, parameter :: my_tag_string_transfer = 3030
  
  if (glob%nprocs==1) return
  
  ex_sig=.false.
  call allocate(sfe_vec,glob%nprocs)
  sfe_vec(:)=.false.
  
  if (glob%iam==0) then
    call date_and_time(values=time_values)
    write(timestamp,'(I4.4,5(A1,I2.2),A1,I3.3)') time_values(1),'-',time_values(2),'-', & 
                               & time_values(3),'_',time_values(5),'.',time_values(6), & 
                               & '.',time_values(7),'_',time_values(8)
    inquire(file=timestamp,exist=stamp_file_exists)
    
    if (stamp_file_exists) then
      call f_rename(timestamp,timestamp//'.bkp')
    endif
    
    open(1746,file=timestamp,status='new',action='write')
    write(1746,'(A)') ' '
    close(1746)
    
    stamp_file_exists=.true.
    
  endif
  
  call dlf_global_char_bcast_rank0(timestamp,0)
  
  if (glob%iam/=0) then
    inquire(file=timestamp,exist=stamp_file_exists)
  endif
  
  call dlf_global_log_gather_rank0(stamp_file_exists,sfe_vec,glob%nprocs,0)
  
  if (glob%iam==0) then
    if (all(sfe_vec)) then
      ex_sig=.true.
    endif
    call f_remove(timestamp)
  endif
  call deallocate(sfe_vec)
  
  call dlf_global_log_bcast_rank0(ex_sig,0)
  
  if (ex_sig) return
  
  nfiles_proc=0
  punch_list_proc(:)=''
  do i=0,max_files_per_proc-1
    fn_punch=''
    write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',i,'.dat'
    inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
    if (punch_exists) then
      nfiles_proc=nfiles_proc+1
      punch_list_proc(nfiles_proc)=fn_punch
    endif
  enddo
  
  allocate(nfiles_vec(0:glob%nprocs-1))
  allocate(punch_matrix(max_files_per_proc,0:glob%nprocs-1))
  allocate(slaveterm(1:glob%nprocs-1))
  call dlf_global_int_gather_rank0(nfiles_proc,nfiles_vec,glob%nprocs,0)
  call dlf_global_char_gather_rank1(punch_list_proc,max_files_per_proc,punch_matrix,glob%nprocs,0)
  
  ! Take care of any duplicate files on master that might otherwise be overwritten
  
  if (glob%iam==0) then
    do j=1,glob%nprocs-1
      do i=1,nfiles_vec(j)
        fn_punch=punch_matrix(i,j)
        inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
        if (punch_exists) then
          fn_punch_rename=''
          timestamp_old=''
          open(1159,file=trim(adjustl(fn_punch)),action='read')
          read(1159,'(A)',iostat=ios) tag
          read(1159,'(A)',iostat=ios) timestamp_old
          if (ios == 0 .and. (trim(adjustl(tag))=='$TIMESTAMP_MASTER' .or. &
                            & trim(adjustl(tag))=='$TIMESTAMP_SLAVE') .and. & 
                            & len_trim(timestamp_old)/=0 ) then
            write(fn_punch_rename,'(4A)') trim(adjustl(fn_punch)),'_bkp_', &
                        & timestamp_old,'.dat'
          else
            write(fn_punch_rename,'(4A)') trim(adjustl(fn_punch)), &
                        & '_bkp_corrupt_foundat_', timestamp,'.dat'
          endif
          close(1159)
          call f_rename(trim(adjustl(fn_punch)),trim(adjustl(fn_punch_rename)))
        endif
      enddo
    enddo
  endif
  
  ! Master loop, receiving all file data from the slave processes
  if (glob%iam==0) then
    slaveterm(:)=.false.
    do while (.true.)
      ! exit loop when all slaves have already finished sending data
      if (all(slaveterm)) exit
      ! Receive punch file name from arbitrary slave
      curr_slave=-1
      itag=my_tag_initiate
      call dlf_mpi_recv(fn_punch,curr_slave,itag)
      if (trim(adjustl(fn_punch))=='I_AM_DONE') then
        slaveterm(curr_slave)=.true.
        cycle
      else
        itag=my_tag_nlines
        call dlf_mpi_recv(nlines,curr_slave,itag)
        open(7658,file=trim(adjustl(fn_punch)),action='write')
        itag=my_tag_string_transfer
        do i=1,nlines
          call dlf_mpi_recv(line_buffer,curr_slave,itag)
          write(7658,'(A)') trim(line_buffer)
        enddo
        close(7658)
      endif
    enddo
  else
  ! Slave loop, sending file data
    do j=1,nfiles_proc
      fn_punch=punch_list_proc(j)
      call dlf_mpi_send(fn_punch,0,my_tag_initiate)
      nlines = 0 
      open (7653, file=trim(adjustl(fn_punch)),action='read')
      do while (.true.)
        read(7653,*,iostat=ios)
        if (ios/=0) exit
        nlines = nlines + 1
      enddo
      rewind(7653)
      call dlf_mpi_send(nlines,0,my_tag_nlines)
      do i=1,nlines
        read(7653,'(A)') line_buffer
        call dlf_mpi_send(line_buffer,0,my_tag_string_transfer)
      enddo
      close(7653)
    enddo
    fn_punch='I_AM_DONE'
    call dlf_mpi_send(fn_punch,0,my_tag_initiate)
  endif
  
  deallocate(nfiles_vec)
  deallocate(slaveterm)
  deallocate(punch_matrix)
  
  if (glob%iam/=0) then
    do j=1,nfiles_proc
      call f_remove(trim(adjustl(punch_list_proc(j))))
    enddo
  endif
  
  return

end subroutine gather_punch_files_in_rank0_dir

! ****************************************
! ****************************************
! ****************************************

! Read punch files and return the data that were found in the appropriate 
! variables.

subroutine read_punch_files ( nat_ref, nvar_ref, nvareff_ref, nhess_max, ngrad_max, &
      & ngrad4hess_max, atsym_ref, coord_ref, displacement_map, displacement_map_grad, & 
      & displacement_map_grad4hess, nhessdone, hessdone, ngraddone, graddone, & 
      & ngraddone4hess, graddone4hess, normal_modes, freqs, evals, hessians_cart, &
      & grad_cart, grad_cart_4hess, fake_nm, delQ_ref, delQ4hess_ref, timestamp )
use dlf_vpt2_utility, only: f_rename
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in)  :: nat_ref,nvar_ref,nvareff_ref,nhess_max,ngrad_max, ngrad4hess_max
character(len=2), dimension(nat_ref), intent(in) :: atsym_ref
real(rk), intent(in), dimension(nvar_ref) :: coord_ref
integer, dimension(-nvareff_ref:+nvareff_ref), intent(in) :: displacement_map
integer, dimension(nvareff_ref,nvareff_ref,-2:2,-2:2), intent(in) :: displacement_map_grad
integer, dimension(nvareff_ref,nvareff_ref,-1:1,-1:1), intent(in) :: displacement_map_grad4hess
integer, intent(out) :: nhessdone,ngraddone,ngraddone4hess
logical, intent(out), dimension(nhess_max) :: hessdone
logical, intent(out), dimension(ngrad_max) :: graddone
logical, intent(out), dimension(ngrad4hess_max) :: graddone4hess
real(rk), intent(out), dimension(nvar_ref,nvareff_ref) :: normal_modes, fake_nm
real(rk), intent(out), dimension(nvareff_ref) :: freqs,evals
real(rk), dimension(nvar_ref,nvar_ref,nhess_max), intent(out) :: hessians_cart
real(rk), dimension(nvar_ref,ngrad_max), intent(out) :: grad_cart
real(rk), dimension(nvar_ref,ngrad4hess_max), intent(out) :: grad_cart_4hess
real(rk), intent(in) :: delQ_ref, delQ4hess_ref
character(len=23), intent(out) :: timestamp

integer, parameter :: nprocs_read_max=500
integer, parameter :: mark_as_error  =-999999
real(rk), parameter :: coordcomparetol=1.e-5_rk
integer :: i,il,j,k,m,imaster,absk,signk
integer :: io,ip,so,sp
integer :: i_hess_index, i_grad_index, i_punch_max
logical :: punch_exists
character(len=500) :: fn_punch,tag
integer :: nat_punch, nvar_punch, nvareff_punch
character(len=2) :: atsym_punch
real(rk) :: coord_val_punch, delQ, delQ2, dQcomp
integer :: error, ios
character(len=1)  :: chdum,disp_type
character(len=23) :: timestamp_master
integer, allocatable, dimension(:) :: punchtype_vec
character(len=23), allocatable, dimension(:)  :: timestamps_vec
character(len=500), allocatable, dimension(:) :: fn_punch_list
logical, allocatable, dimension(:) :: valid_punch
logical :: for_hess, found_hess0, found_nm, found_fake_nm, found_eval, found_freq
logical :: refined, found_nm_refined, found_eval_refined, found_freq_refined

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nat_ref, size(atsym_ref) ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [nvar_ref, &
             &   size(coord_ref), &
             &   size(normal_modes, dim=1) , &
             &   size(fake_nm, dim=1) , &
             &   size(hessians_cart,dim=1),  &
             &   size(hessians_cart,dim=2), &
             &   size(grad_cart,dim=1), &
             &   size(grad_cart_4hess,dim=1) &
             &   ], &
             &  'dim. mismatch, read_punch_files')
  call checkeq( [nvareff_ref, &
             &   size(freqs), & 
             &   size(evals), & 
             &   size(normal_modes, dim=2), &
             &   size(fake_nm, dim=2), &
             &   size(displacement_map_grad,dim=1), &
             &   size(displacement_map_grad,dim=2) , &
             &   size(displacement_map_grad4hess,dim=1), &
             &   size(displacement_map_grad4hess,dim=2) ], &
             &  'dim. mismatch, read_punch_files')
  call checkeq( [nhess_max, size(hessdone), size(hessians_cart, dim=3)  ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [ngrad_max, size(graddone), size(grad_cart, dim=2)  ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [ngrad4hess_max, size(graddone4hess), size(grad_cart_4hess, dim=2)  ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [2*nvareff_ref+1, size(displacement_map)  ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [3, size(displacement_map_grad4hess,dim=3), &
             &      size(displacement_map_grad4hess,dim=4) &
             &          ], &
     &  'dim. mismatch, read_punch_files')
  call checkeq( [5, size(displacement_map_grad,dim=3), &
             &      size(displacement_map_grad,dim=4) &
             &          ], &
     &  'dim. mismatch, read_punch_files')
  write(stdout,*) 'read_punch_files: array shape check successful.'
#endif

hessians_cart(:,:,:)=0._rk
normal_modes(:,:)=0._rk
fake_nm(:,:)=0._rk
grad_cart(:,:)=0._rk
grad_cart_4hess(:,:)=0._rk
nhessdone=0
ngraddone=0
ngraddone4hess=0
hessdone(:)=.false.
graddone(:)=.false.
graddone4hess(:)=.false.
error=0
timestamp=''
found_hess0=       .false.
found_nm=          .false.
found_fake_nm=     .false.
found_eval=        .false.
found_freq=        .false.
found_nm_refined=  .false.
found_eval_refined=.false.
found_freq_refined=.false.

allocate(valid_punch(0:nprocs_read_max))
valid_punch(:)=.false.

i_punch_max=-1
do i=0,nprocs_read_max
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',i,'.dat'
  inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
  if (punch_exists) then
    valid_punch(i)=.true.
    i_punch_max=i
  endif
enddo

if (i_punch_max==-1) then
  write(stdout,'(A)') 'read_punch_files: No punch files found!'
  deallocate(valid_punch)
  return
endif

allocate(punchtype_vec(0:i_punch_max))
allocate(timestamps_vec(0:i_punch_max))
allocate(fn_punch_list(0:i_punch_max))

punchtype_vec(:)=mark_as_error
timestamps_vec(:)='ERR'
fn_punch_list(:)=''

imaster=-1
do i=0,i_punch_max
  if (.not. valid_punch(i)) cycle
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',i,'.dat'
  fn_punch_list(i)=fn_punch
  open(4477,file=trim(adjustl(fn_punch)), action='read')
  read(4477,'(A)',iostat=ios) tag
  if (tag=='$TIMESTAMP_MASTER') then
    punchtype_vec(i)=1
    if (imaster<0) imaster=i
    read(4477,'(A)',iostat=ios) timestamps_vec(i)
    close(4477)
  elseif (tag=='$TIMESTAMP_SLAVE') then
    punchtype_vec(i)=0
    read(4477,'(A)',iostat=ios) timestamps_vec(i)
    close(4477)
  else
    close(4477)
    cycle
  endif
enddo

if (imaster<0 .or. count(punchtype_vec(:)==1)==0) then
  write(stdout,'(A)') 'read_punch_files: No master punch file found!'
  deallocate(valid_punch)
  deallocate(fn_punch_list)
  deallocate(timestamps_vec)
  deallocate(punchtype_vec)
  return
endif

timestamp_master=timestamps_vec(imaster)

deallocate(valid_punch)
allocate(valid_punch(0:i_punch_max))
valid_punch(:)=.false.
do i=0,i_punch_max
  if (.not. (punchtype_vec(i)==0 .or. punchtype_vec(i)==1)) cycle
  if (punchtype_vec(i)==1 .and. i/=imaster) cycle
  if (timestamps_vec(i)==timestamp_master) valid_punch(i)=.true.
enddo

fn_punch=fn_punch_list(imaster)

outerblock: do i=1,1
  open(4477,file=trim(adjustl(fn_punch)), action='read')
  read(4477,'(A)',iostat=ios) chdum
  read(4477,'(A)',iostat=ios) chdum
  tag=''
  read(4477,'(A)',iostat=ios) tag
  if (ios/=0 .or. trim(adjustl(tag))/='$NAT') then
    error=1
    exit outerblock
  endif
  read(4477,*,iostat=ios) nat_punch
  if (ios/=0 .or. nat_punch/=nat_ref) then
    error=2
    exit outerblock
  endif
  tag=''
  read(4477,'(A)',iostat=ios) tag
  if (ios/=0 .or. trim(adjustl(tag))/='$NVAR') then
    error=3
    exit outerblock
  endif
  read(4477,*,iostat=ios) nvar_punch
  if (ios/=0 .or. nvar_punch/=nvar_ref) then
    error=4
    exit outerblock
  endif
  tag=''
  read(4477,'(A)',iostat=ios) tag
  if (ios/=0 .or. trim(adjustl(tag))/='$NVAREFF') then
    error=5
    exit outerblock
  endif
  read(4477,*,iostat=ios) nvareff_punch
  if (ios/=0 .or. nvareff_punch/=nvareff_ref) then
    error=6
    exit outerblock
  endif
  tag=''
  read(4477,'(A)',iostat=ios) tag
  if (ios/=0 .or. trim(adjustl(tag))/='$ATLIST') then
    error=7
    exit outerblock
  endif
  do j=1,nat_ref
    read(4477,'(A2)',iostat=ios) atsym_punch
    if (ios/=0 .or. trim(adjustl(atsym_punch))/=trim(adjustl(atsym_ref(j)))) then
      error=800+j
      exit outerblock
    endif
  enddo
  tag=''
  read(4477,'(A)',iostat=ios) tag
  if (ios/=0 .or. trim(adjustl(tag))/='$COORD') then
    error=9
    exit outerblock
  endif
  do j=1,nvar_ref
    read(4477,*,iostat=ios) coord_val_punch
    if (ios/=0 .or. abs(coord_val_punch-coord_ref(j))>coordcomparetol) then
      error=1000+j
      exit outerblock
    endif
  enddo
  !!tag=''
  !!read(4477,'(A)',iostat=ios) tag
  !!if (ios/=0 .or. (trim(adjustl(tag))/='$NORMAL_MODES' & 
  !!         & .and. trim(adjustl(tag))/='$FAKE_NM')) then
  !!  error=11
  !!  exit outerblock
  !!endif
  !!if (trim(adjustl(tag))=='$NORMAL_MODES') then
  !!  do j=1,nvar_ref
  !!    read(4477,*,iostat=ios) (normal_modes(j,k),k=1,nvareff_ref)
  !!    if (ios/=0) then
  !!      error=1100+j
  !!      exit outerblock
  !!    endif
  !!  enddo
  !!  tag=''
  !!  read(4477,'(A)',iostat=ios) tag
  !!  if (ios/=0 .or. trim(adjustl(tag))/='$FREQS') then
  !!    error=12
  !!    exit outerblock
  !!  endif
  !!  do j=1,nvareff_ref
  !!    read(4477,*,iostat=ios) freqs(j)
  !!    if (ios/=0) then
  !!      error=1200+j
  !!      exit outerblock
  !!    endif
  !!  enddo
  !!  tag=''
  !!  read(4477,'(A)',iostat=ios) tag
  !!  if (ios/=0 .or. trim(adjustl(tag))/='$EIGENVALUES') then
  !!    error=13
  !!    exit outerblock
  !!  endif
  !!  do j=1,nvareff_ref
  !!    read(4477,*,iostat=ios) evals(j)
  !!    if (ios/=0) then
  !!      error=1300+j
  !!      exit outerblock
  !!    endif
  !!  enddo
  !!elseif (trim(adjustl(tag))=='$FAKE_NM')
  !!  do j=1,nvar_ref
  !!    read(4477,*,iostat=ios) (fake_nm(j,k),k=1,nvareff_ref)
  !!    if (ios/=0) then
  !!      error=2000+j
  !!      exit outerblock
  !!    endif
  !!  enddo
  !!endif
end do outerblock

if (error/=0) then
  close(4477)
  write(stdout,'(A,I0)') 'read_punch_files: Mismatch while reading master punchfile. Error code: ', error
  deallocate(valid_punch)
  deallocate(fn_punch_list)
  deallocate(timestamps_vec)
  deallocate(punchtype_vec)
  return
endif

punchfileloop: do il=-1,i_punch_max
  if (il==imaster) cycle punchfileloop
  if (il==-1) then
    i=imaster
  else
    i=il
    if (.not.valid_punch(i)) cycle punchfileloop
    fn_punch=fn_punch_list(i)
    open(4477,file=trim(adjustl(fn_punch)), action='read')
    read(4477,'(A)',iostat=ios) chdum
    read(4477,'(A)',iostat=ios) chdum
  endif
  ios=0
  hessgradloop: do while (ios==0)
    read(4477,'(A)',iostat=ios) tag
    if     (ios==0 .and. trim(adjustl(tag))=='$HESSIAN_CART') then
      dQcomp=delQ_ref
      k=nvareff_ref+1
      read(4477,*,iostat=ios) absk,signk,delQ
      k=absk*signk
      if (ios/=0 .or. abs(k)>nvareff_ref) then
        exit hessgradloop
      endif
      if (abs(delQ)>coordcomparetol .and. abs(delQ-dQcomp)>coordcomparetol) then
        write(stdout,'(A,I0,A,ES12.4,A,ES12.4)') 'Skip reading Hessian from ' // &
           &             'punch file due to deltaQ mismatch, k=', &
           & k, ', desired: ', dQcomp, ', actual: ', delQ
        do j=1,nvareff_ref
          read(4477,'(A)',iostat=ios) chdum
          if (ios/=0) exit hessgradloop
        enddo
        cycle hessgradloop
      endif
      i_hess_index = displacement_map(k)
      do j=1,nvar_ref
        read(4477,*,iostat=ios) (hessians_cart(j,m,i_hess_index),m=1,nvar_ref)
        if (ios/=0) then
          hessians_cart(:,:,i_hess_index)=0._rk
          exit hessgradloop
        endif
      enddo
      hessdone(i_hess_index)=.true.
      if (k==0) found_hess0=.true.
    elseif (ios==0 .and. (trim(adjustl(tag))=='$GRADIENT_CART' .or. &
                       &  trim(adjustl(tag))=='$GRADIENT_CART_4HESS' )) then
      if (trim(adjustl(tag))=='$GRADIENT_CART_4HESS') then
        for_hess=.true.
        dQcomp=delQ4hess_ref
      else
        for_hess=.false.
        dQcomp=delQ_ref
      endif
      read(4477,'(A1)',iostat=ios) disp_type
      if (ios/=0 .or. (disp_type/='S' .and. disp_type/='D')) then
        exit hessgradloop
      endif
      if (disp_type=='D' .and. for_hess) exit hessgradloop
      io=nvareff_ref+1
      ip=nvareff_ref+1
      so=-10
      sp=-10
      delQ =-10._rk
      delQ2=-10._rk
      if     (disp_type=='S') then
        delQ2=dQcomp
        ip=1
        sp=0
        read(4477,*,iostat=ios) io
        read(4477,*,iostat=ios) so
        read(4477,*,iostat=ios) delQ
      elseif (disp_type=='D') then
        read(4477,*,iostat=ios) io, ip
        read(4477,*,iostat=ios) so, sp
        read(4477,*,iostat=ios) delQ, delQ2
      endif
      if ( ios/=0 .or. &
         & io>nvareff_ref .or. &
         & ip>nvareff_ref .or. &
         & io<1 .or. &
         & ip<1 .or. &
         & abs(so)>2 .or. &
         & abs(sp)>2 ) exit hessgradloop
      if (abs(delQ-dQcomp) >coordcomparetol .or. &
        & abs(delQ2-dQcomp)>coordcomparetol) then
        write(stdout,'(A,I0,A,ES12.4,A,ES12.4,1X,ES12.4)') 'Skip reading ' // &
           &         'gradient from punch file due to deltaQ mismatch, k=', &
           & k, ', desired (2x): ', dQcomp, ', actual: ', delQ, delQ2
        do j=1,nvar_ref
          read(4477,'(A)',iostat=ios) chdum
          if (ios/=0) exit hessgradloop
        enddo
        cycle hessgradloop
      endif
      if (for_hess) then
        i_grad_index = displacement_map_grad4hess(io,io,so,0)
      else
        if     (disp_type=='S') then
          i_grad_index = displacement_map_grad(io,io,so,0)
        elseif (disp_type=='D') then
          i_grad_index = displacement_map_grad(io,ip,so,sp)
        endif
      endif
      do j=1,nvar_ref
        if (for_hess) then
          read(4477,*,iostat=ios) grad_cart_4hess(j,i_grad_index)
        else
          read(4477,*,iostat=ios) grad_cart(j,i_grad_index)
        endif
        if (ios/=0) then
          if (for_hess) then
            grad_cart_4hess(:,i_grad_index)=0._rk
          else
            grad_cart(:,i_grad_index)=0._rk
          endif
          exit hessgradloop
        endif
      enddo
      graddone(i_grad_index)=.true.
    elseif (ios==0 .and. (trim(adjustl(tag))=='$NORMAL_MODES' &
                   & .or. trim(adjustl(tag))=='$NORMAL_MODES_REFINED' )) then
      if (trim(adjustl(tag))=='$NORMAL_MODES_REFINED') then
        refined=.true.
      else
        refined=.false.
      endif
      if (found_nm_refined .and. .not. refined) then
        do j=1,nvar_ref
          read(4477,'(A)',iostat=ios) chdum
          if (ios/=0)  exit hessgradloop
        enddo
      else
        do j=1,nvar_ref
          read(4477,*,iostat=ios) (normal_modes(j,k),k=1,nvareff_ref)
          if (ios/=0)  exit hessgradloop
        enddo
      endif
      found_nm=.true.
      if (refined) found_nm_refined=.true.
    elseif (ios==0 .and. trim(adjustl(tag))=='$FAKE_NM') then
      do j=1,nvar_ref
        read(4477,*,iostat=ios) (fake_nm(j,k),k=1,nvareff_ref)
        if (ios/=0) exit hessgradloop
      enddo
      found_fake_nm=.true.
    elseif (ios==0 .and. ( trim(adjustl(tag))=='$FREQS' &
                    & .or. trim(adjustl(tag))=='$FREQS_REFINED' )) then
      if (trim(adjustl(tag))=='$FREQS_REFINED') then
        refined=.true.
      else
        refined=.false.
      endif
      if (found_freq_refined .and. .not. refined) then
        do j=1,nvareff_ref
          read(4477,'(A)',iostat=ios) chdum
          if (ios/=0)  exit hessgradloop
        enddo
      else
        do j=1,nvareff_ref
          read(4477,*,iostat=ios) freqs(j)
          if (ios/=0) exit hessgradloop
        enddo
      endif
      found_freq=.true.
      if (refined) found_freq_refined=.true.
    elseif (ios==0 .and. ( trim(adjustl(tag))=='$EIGENVALUES' &
                    & .or. trim(adjustl(tag))=='$EIGENVALUES_REFINED' )) then
      if (trim(adjustl(tag))=='$EIGENVALUES_REFINED') then
        refined=.true.
      else
        refined=.false.
      endif
      if (found_eval_refined .and. .not. refined) then
        do j=1,nvareff_ref
          read(4477,'(A)',iostat=ios) chdum
          if (ios/=0)  exit hessgradloop
        enddo
      else
        do j=1,nvareff_ref
          read(4477,*,iostat=ios) evals(j)
          if (ios/=0) exit hessgradloop
        enddo
      endif
      found_eval=.true.
      if (refined) found_eval_refined=.true.
    else
      exit hessgradloop
    endif
  enddo hessgradloop
  close(4477)
end do punchfileloop

deallocate(timestamps_vec)

nhessdone=count(hessdone)
ngraddone=count(graddone)
ngraddone4hess=count(graddone4hess)
if ( & 
     & (.not. found_eval) .or. &
     & (.not. found_freq) .or. &
     & (.not. found_nm)   ) then
  nhessdone=0
  ngraddone=0
endif
if ( .not. found_fake_nm ) ngraddone4hess=0

timestamp=timestamp_master

write(stdout,'(A)') '....................................................................'
write(stdout,'(A,I0,A)') 'read_punch_files: ', nhessdone, ' Hessians from previous calculation(s) found.'
write(stdout,'(A,I0,A)') 'read_punch_files: ', ngraddone, " gradients for cubic/quartic fc's from previous calculation(s) found."
write(stdout,'(A,I0,A)') 'read_punch_files: ', ngraddone4hess, " gradients for Hessian from previous calculation(s) found."
write(stdout,'(A)') '....................................................................'

!call punch_file_collapse(1,count(valid_punch),500,nvareff_ref, &
!         &  pack(fn_punch_list,valid_punch),maxloc(pack(punchtype_vec,valid_punch),dim=1))

!write(stdout,*) 'imaster: ', imaster
!if (imaster /= 0) then
!  call f_rename(trim(adjustl(fn_punch_list(imaster))),'dlf_vpt2_restart_proc_000000.dat')
!endif

deallocate(valid_punch)
deallocate(fn_punch_list)
deallocate(punchtype_vec)

return
end subroutine read_punch_files

! ****************************************
! ****************************************
! ****************************************

! Combine punch files from dlf_vpt2_restart_proc_xxxxxx.dat to 
! dlf_vpt2_restart_proc_yyyyyy.dat in a single checkpoint file
! (xxxxxx = lbnd, yyyyyy=ubnd)

subroutine punch_file_collapse_range(lbnd,ubnd,charlength,nvareff,imaster)
  use dlf_vpt2_utility, only: f_remove
  implicit none
  integer, intent(in) :: lbnd,ubnd,nvareff,imaster,charlength
  
  character(len=charlength), dimension(lbnd:ubnd) :: filelist
  integer :: i
  
  if (lbnd==ubnd) return
  filelist(:)=''
    
  do i=lbnd,ubnd
    write(filelist(i),'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',i,'.dat'
  enddo
  
  call punch_file_collapse(lbnd,ubnd,charlength,nvareff,filelist,imaster)
  
  return
end subroutine punch_file_collapse_range

! ****************************************
! ****************************************
! ****************************************

! Combine punch files given in the filelist into the master punch file

subroutine punch_file_collapse(lbnd,ubnd,charlength,nvareff,filelist,imaster)
  use dlf_vpt2_utility, only: f_remove
  implicit none
  integer, intent(in) :: lbnd,ubnd,nvareff,imaster,charlength
  character(len=charlength), dimension(lbnd:ubnd), intent(in) :: filelist
  
  character(len=1) :: chdum
  character(len=max(2000,nvareff*24+10)) :: readline
  integer :: i,ios
  logical :: file_exists
  
  if (size(filelist)==1) return
  
  open(7465,file=trim(adjustl(filelist(imaster))),action='write',status='old',position='append')
  
  do i=lbnd,ubnd
    if (i==imaster) cycle
    inquire(file=trim(adjustl(filelist(i))),exist=file_exists)
    if (.not. file_exists) cycle
    open(7467,file=trim(adjustl(filelist(i))),action='read')
    ios=0
    read(7467,'(A)',iostat=ios) chdum
    read(7467,'(A)',iostat=ios) chdum
    do while (ios==0)
      readline=''
      read(7467,'(A)',iostat=ios) readline
      if (ios==0) then
        write(7465,'(A)') trim(readline)
      endif
    end do
    close(7467)
  enddo
  
  close(7465)
  
  do i=lbnd,ubnd
    if (i==imaster) cycle
    call f_remove(trim(adjustl(filelist(i))))
  enddo
  
  return
end subroutine punch_file_collapse

! ****************************************
! ****************************************
! ****************************************

! Calculate finite-difference Hessian from gradients of displaced geometries
! Displacedments are done along 'fake normal modes'. These are just the normal modes 
! that would arise from a unit matrix Hessian. The advantage compared to Cartesian 
! displacements is that there are six coordinates less for which displaced gradients
! have to be calculated. In principle this could also be used to exploit symmetry  
! very easily, as the fake normal modes are automatically symmetry-adapted. (not
! implemented)

subroutine get_hessian_finite_difference(grad_routine,nat,nvar,nvareff,ngrad4hess, &
               & graddone4hess,grad4hess,joblist_grad4hess,displacement_map_grad4hess, &
               & fake_nm,coord0,mv,mv3,hessian_cart,deltaQ_uniform)
use dlf_vpt2_utility, only: dlf_gl_bcast, matrix_output, error_print
use dlf_linalg_interface_mod
use dlf_vpt2_freq, only: get_frequencies_mw_ts_or_min
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_GRAD_DEBUG_OUT
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine grad_routine
end interface

integer, intent(in) :: nat,nvar,nvareff,ngrad4hess
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), intent(in), dimension(nat) :: mv
logical, intent(inout), dimension(ngrad4hess) :: graddone4hess
real(rk), intent(inout), dimension(nvar,ngrad4hess) :: grad4hess
integer, intent(in), dimension(nvareff,nvareff,-1:1,-1:1) :: displacement_map_grad4hess
integer, intent(in), dimension(ngrad4hess,4) :: joblist_grad4hess
real(rk), intent(inout), dimension(nvar,nvareff) :: fake_nm
real(rk), intent(out), dimension(nvar,nvar)   :: hessian_cart
real(rk), intent(in) :: deltaQ_uniform

real(rk), dimension(nvar,nvar) :: dummy_hess
real(rk), dimension(nvar) :: fake_eval, fake_freq
real(rk), dimension(nvar,nvareff) :: fake_nm_non_mw
real(rk), dimension(nvar,nvar) :: invsqrtmass
integer :: nimag_dummy, ngraddone, k
character(len=2), dimension(nat) :: atsym_dummy
logical :: punch_exists
character(len=500) :: fn_punch

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nat, size(mv) ], &
   &  'dim. mismatch, get_hessian_finite_difference')
call checkeq( [ nvar, &
            & size(coord0), &
            & size(mv3), &
            & size(fake_nm, dim=1), &
            & size(grad4hess,dim=1), &
            & size(hessian_cart,dim=1), &
            & size(hessian_cart,dim=2) &
            & ], &
   &  'dim. mismatch, get_hessian_finite_difference')
call checkeq( [nvareff, &
            & size(displacement_map_grad4hess, dim=1), &
            & size(displacement_map_grad4hess, dim=2), &
            & size(fake_nm, dim=2) &
            & ], &
   &  'dim. mismatch, get_hessian_finite_difference')
call checkeq( [ ngrad4hess, & 
              & size(graddone4hess), &
              & size(joblist_grad4hess,dim=1), &
              & size(grad4hess,dim=2)  ], &
   &  'dim. mismatch, get_hessian_finite_difference')
call checkeq( [ 3, & 
              & size(displacement_map_grad4hess, dim=3), &
              & size(displacement_map_grad4hess, dim=4)  &
              &  ], &
   &  'dim. mismatch, get_hessian_finite_difference')
call checkeq( [ 4, size(joblist_grad4hess,dim=2)  ], &
   &  'dim. mismatch, get_hessian_finite_difference')
write(stdout,*) 'get_hessian_finite_difference: array shape check successful.'
#endif

ngraddone=count(graddone4hess)
if (ngraddone<1) then
  if (glob%iam==0) then
    atsym_dummy(:)='  '
    dummy_hess = dlf_unit_mat(nvar)
    call get_frequencies_mw_ts_or_min(nat,nvar-nvareff,coord0,atsym_dummy,mv,dummy_hess,fake_freq, &
              & fake_eval,fake_nm,nimag_dummy,proj=.true.,dryrun=.true.)
    fn_punch=''
    write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'
    inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
    if (.not. punch_exists) then
      call error_print('get_hessian_finite_difference: punch file missing!')
    endif
    open(1361, file=trim(adjustl(fn_punch)), status="old", position="append", action="write")
    write(1361,'(A)') '$FAKE_NM'
    call matrix_output(fake_nm,1361,'ES24.16','__BLANK__')
    close(1361)
  endif
endif

call dlf_gl_bcast(fake_nm,nvar,nvareff,0)

call calculate_displaced_gradients_mpi(grad_routine,nvar,nvareff,ngrad4hess, &
               & coord0,fake_nm,mv3, joblist_grad4hess,graddone4hess,grad4hess, &
               & deltaQ_uniform,hessmode_in=.true.)

invsqrtmass(:,:)=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
fake_nm_non_mw=dlf_matmul_simp(invsqrtmass,fake_nm)

call hessian_from_gradient_list(nvar,nvareff,ngrad4hess, &
                  & grad4hess,mv3,fake_nm,fake_nm_non_mw, &
                  & displacement_map_grad4hess,hessian_cart,deltaQ_uniform)

return
end subroutine get_hessian_finite_difference

! ****************************************
! ****************************************
! ****************************************

! Get Hessian from precalculated list of displaced gradients.
! This allows to logically separate the gradient calculations from the 
! finite-difference Hessian calculation. (needed e.g. for restart
! capabilities, helpful for parallelization). 
! This is meant to be combined with calculate_displaced_gradients_mpi

subroutine hessian_from_gradient_list(nvar,nvareff,ngrad4hess, &
                  & grad4hess,mv3,fake_nm,fake_nm_non_mw, &
                  & displacement_map_grad4hess,hessian_cart,deltaQ_uniform)
use dlf_vpt2_utility, only: error_print, matrix_output
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad4hess
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad4hess), intent(in) :: grad4hess
real(rk), dimension(nvar,nvareff), intent(in) :: fake_nm,fake_nm_non_mw
integer, dimension(nvareff,nvareff,-1:1,-1:1), intent(in) :: displacement_map_grad4hess
real(rk), intent(in) :: deltaQ_uniform
real(rk), intent(out),dimension(nvar,nvar) :: hessian_cart

integer :: io,m,inu,jnu
integer, dimension(nvareff,nvareff) :: ndata
real(rk), dimension(nvareff,nvareff) :: hessian_fnm
real(rk), dimension(nvar)      :: fd_cart
real(rk), dimension(nvareff)   :: fd_fnm
real(rk), dimension(nvar,nvar) :: sqrtmass

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [ nvar, &
            & size(fake_nm, dim=1), &
            & size(fake_nm_non_mw, dim=1), &
            & size(grad4hess,dim=1), &
            & size(hessian_cart,dim=1), &
            & size(hessian_cart,dim=2), &
            & size(mv3) &
            & ], &
   &  'dim. mismatch, hessian_from_gradient_list')
call checkeq( [nvareff, &
            & size(displacement_map_grad4hess, dim=1), &
            & size(displacement_map_grad4hess, dim=2), &
            & size(fake_nm, dim=2), &
            & size(fake_nm_non_mw, dim=2) &
            & ], &
   &  'dim. mismatch, hessian_from_gradient_list')
call checkeq( [ ngrad4hess, & 
            & size(grad4hess,dim=2)  ], &
   &  'dim. mismatch, hessian_from_gradient_list')
call checkeq( [ 3, & 
            & size(displacement_map_grad4hess, dim=3), &
            & size(displacement_map_grad4hess, dim=4)  &
            &  ], &
   &  'dim. mismatch, hessian_from_gradient_list')
write(stdout,*) 'hessian_from_gradient_list: array shape check successful.'
#endif

hessian_cart(:,:)=0._rk
hessian_fnm(:,:)=0._rk
ndata(:,:)=0

! Iterate over fake normal modes
do io=1,nvareff
  fd_cart(:)=grad4hess(:,displacement_map_grad4hess(io,io, 1,0)) & 
             & - grad4hess(:,displacement_map_grad4hess(io,io, -1,0))
  fd_fnm (:)=dlf_matmul_simp(transpose(fake_nm_non_mw),fd_cart)
  fd_fnm (:)=fd_fnm(:)/(2*deltaQ_uniform)
  do m=1,nvareff
    inu=max(io,m)
    jnu=min(io,m)
    hessian_fnm(inu,jnu)=hessian_fnm(inu,jnu)+fd_fnm(m)
    ndata(inu,jnu)=ndata(inu,jnu)+1
  enddo
enddo

! compensate overcounting due to redundancies
ndata(:,:)=max(ndata(:,:),1)
hessian_fnm(:,:)=hessian_fnm(:,:)/real(ndata(:,:))

open(1414,file='dbg1.dat')
call matrix_output(hessian_fnm,1414,'ES20.12','hessian_fnm')
close(1414)

! symmetric fill
do inu=1,nvareff
  do jnu=inu+1,nvareff
    hessian_fnm(inu,jnu)=hessian_fnm(jnu,inu)
  enddo
enddo

! transform Hessian in fake normal modes to Cartesian Hessian
hessian_cart= dlf_matrix_ortho_trans(fake_nm,hessian_fnm,1)

sqrtmass(:,:)=0._rk
do m=1,nvar
  sqrtmass(m,m)=sqrt(mv3(m))
enddo

hessian_cart= dlf_matmul_simp(sqrtmass,hessian_cart)
hessian_cart= dlf_matmul_simp(hessian_cart,sqrtmass)

return
end subroutine hessian_from_gradient_list

! ****************************************
! ****************************************
! ****************************************

! 3rd/4th derivatives from displaced gradients. 
! The calculation is repeated for two step sizes
! differing by a factor lambda_fac (e.g. 2), and the 
! resulting force constants for the two deltaQ's are 
! extrapolated to zero step size.

subroutine derivatives_from_gradients_extrapolate( & 
               & grad_routine,nvar,nvareff,ngrad, &
               & coord0,normal_modes,normal_modes_non_mw, mv3, hess_eval, &
               & joblist,displacement_map_grad,cubic_fc,quartic_fc,deltaQ_uniform)
use dlf_vpt2_utility, only: error_print, dlf_global_real_bcast, &
            & dlf_global_int_bcast, dlf_global_int_scatter_rank0, &
            & dlf_global_int_scatter_rank1, vector_output
use dlf_linalg_interface_mod
use dlf_allocate
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_GRAD_DEBUG_OUT
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine grad_routine
end interface
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), dimension(nvar,nvareff), intent(in) :: normal_modes
real(rk), dimension(nvar,nvareff), intent(in) :: normal_modes_non_mw
real(rk), dimension(nvareff), intent(in) :: hess_eval
integer, dimension(ngrad,4),intent(in) :: joblist
integer, dimension(nvareff,nvareff,-1:1,-1:1), intent(in) :: displacement_map_grad
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
real(rk), intent(in) :: deltaQ_uniform

real(rk), parameter :: lambda_fac=2._rk
logical, dimension(ngrad) :: grad_done
real(rk), dimension(nvar,ngrad)  :: grad_cart_01, grad_cart_02
real(rk) :: lambda1, lambda2
real(rk), dimension(nvareff,nvareff,nvareff) :: cubic_fc_01, cubic_fc_02
real(rk), dimension(nvareff,nvareff,nvareff) :: quartic_fc_01, quartic_fc_02

write(stdout,*) 'Entering derivatives_from_gradients_extrapolate...'

lambda1=deltaQ_uniform
lambda2=lambda1*lambda_fac

grad_cart_01(:,:)=0._rk
grad_cart_02(:,:)=0._rk
cubic_fc_01(:,:,:)=0._rk
cubic_fc_02(:,:,:)=0._rk
quartic_fc_01(:,:,:)=0._rk
quartic_fc_02(:,:,:)=0._rk

write(stdout,*) 'Calling calculate_displaced_gradients_mpi (1/2)...'
grad_done(:)=.false.
call calculate_displaced_gradients_mpi(grad_routine,nvar,nvareff,ngrad, &
               & coord0,normal_modes,mv3, &
               & joblist,grad_done,grad_cart_01,lambda1)

write(stdout,*) 'Calling derivatives_from_gradient_list (1/2)...'
call derivatives_from_gradient_list(nvar,nvareff,ngrad, &
                  & grad_cart_01,hess_eval,normal_modes_non_mw, &
                  & displacement_map_grad,cubic_fc_01,quartic_fc_01,lambda1)

write(stdout,*) 'Calling calculate_displaced_gradients_mpi (2/2)...'
grad_done(:)=.false.
call calculate_displaced_gradients_mpi(grad_routine,nvar,nvareff,ngrad, &
               & coord0,normal_modes,mv3, &
               & joblist,grad_done,grad_cart_02,lambda2)

write(stdout,*) 'Calling derivatives_from_gradient_list (2/2)...'
call derivatives_from_gradient_list(nvar,nvareff,ngrad, &
                  & grad_cart_02,hess_eval,normal_modes_non_mw, &
                  & displacement_map_grad,cubic_fc_02,quartic_fc_02,lambda2)

cubic_fc(:,:,:)  =(lambda1**2*cubic_fc_02(:,:,:)  -lambda2**2*cubic_fc_01(:,:,:))  /(lambda1**2-lambda2**2)
quartic_fc(:,:,:)=(lambda1**2*quartic_fc_02(:,:,:)-lambda2**2*quartic_fc_01(:,:,:))/(lambda1**2-lambda2**2)

write(stdout,*) 'Returning from derivatives_from_gradients_extrapolate...'

return
end subroutine derivatives_from_gradients_extrapolate

! ****************************************
! ****************************************
! ****************************************

! Given a job list that specifies the displacements to be covered,
! this routine computes gradients of geometries that are displaced 
! along one or two normal coordinates. This routine is MPI-parallelized:
! The total joblist is distributed among the MPI ranks, and every 
! processes works through its individual work package.

subroutine calculate_displaced_gradients_mpi(grad_routine,nvar,nvareff,ngrad, &
               & coord0,normal_modes,mv3, &
               & joblist,grad_done,grad_cart,deltaQ_uniform,silent_in,hessmode_in)
use dlf_vpt2_utility, only: error_print, dlf_global_real_bcast, &
            & dlf_global_int_bcast, dlf_global_int_scatter_rank0, &
            & dlf_global_int_scatter_rank1, vector_output
use dlf_linalg_interface_mod
use dlf_allocate
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_GRAD_DEBUG_OUT
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
  subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine grad_routine
end interface
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes
integer, dimension(ngrad,4),intent(in) :: joblist
logical, dimension(ngrad), intent(inout) :: grad_done
real(rk), dimension(nvar,ngrad), intent(inout)    :: grad_cart
real(rk), intent(in) :: deltaQ_uniform
logical, intent(in),optional :: silent_in, hessmode_in

real(rk), dimension(nvar,nvar) :: invsqrtmass
real(rk), dimension(nvar) :: grad
real(rk), dimension(nvar,nvareff) :: normal_modes_non_mw
real(rk), dimension(nvar) :: coordtmp
real(rk) :: dQo,dQp
integer,dimension(ngrad) :: proc_assignment
integer,dimension(ngrad) :: jlist_this_proc
integer,dimension(ngrad) :: jlist

integer, dimension(:),allocatable :: jlist_todo, ntodo_vec
integer, dimension(:),allocatable :: jpmin_vec,jpmax_vec
integer, dimension(:,:),allocatable :: jlist_matrix
integer  :: istat,gradcount,nproc,io,ip,so,sp
real(rk) :: energy
integer, parameter :: iimage=1,kiter=-1
integer :: k,j,m, ires
integer  :: ndone, ntodo, ntodo_this_process
character(len=500) :: fn_punch
logical :: punch_exists
character(len=300) :: gradtag

logical  :: silent, hessmode

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, size(coord0), size(mv3), size(normal_modes, dim=1), &
            &  size(grad_cart,dim=1) ], &
   &  'dim. mismatch, calculate_displaced_gradients_mpi')
call checkeq( [nvareff, size(normal_modes, dim=2)], &
            & 'dim. mismatch, calculate_displaced_gradient_mpi')
call checkeq( [ ngrad, & 
              & size(grad_done), &
              & size(joblist,dim=1), &
              & size(grad_cart,dim=2)  ], &
   &  'dim. mismatch, calculate_displaced_gradients_mpi')
call checkeq( [ 4, size(joblist,dim=2)  ], &
   &  'dim. mismatch, calculate_displaced_gradients_mpi')
write(stdout,*) 'calculate_displaced_gradients_mpi: array shape check successful.'
#endif

fn_punch=''
write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'

inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
if (.not. punch_exists) then
  call error_print('calculate_displaced_gradients_mpi: punch file missing!')
endif

open(3499, file=trim(adjustl(fn_punch)), status="old", position="append", action="write")

nproc=glob%nprocs

silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif
hessmode=.false.
if (present(hessmode_in)) then
  hessmode=hessmode_in
endif

!Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

! Find out what has been already done, and what's left to do and distribute job ranges along processes

ndone=count(grad_done)
ntodo=ngrad-ndone

if (glob%iam==0) then
  do j=1,ngrad
    jlist(j)=j
  enddo
  allocate(jlist_todo(ntodo))
  allocate(ntodo_vec(0:nproc-1))
  allocate(jpmin_vec(0:nproc-1))
  allocate(jpmax_vec(0:nproc-1))
  allocate(jlist_matrix(ngrad,0:nproc-1))
  jlist_todo = pack(jlist,.not.grad_done)
  jlist_matrix(:,:)=0
  proc_assignment(:)=-1
  if (nproc==1) then
    ntodo_vec(0)=ntodo
    jlist_matrix(1:ntodo,0)=jlist_todo(1:ntodo)
    do k=1,ntodo
      proc_assignment(jlist_todo(k))=0
    enddo
  else
    jpmax_vec(:)=int(ntodo/nproc)
    ires=mod(ntodo,nproc)
    if (ires>0) jpmax_vec(0:ires-1)=jpmax_vec(0:ires-1)+1
    ntodo_vec(:)=jpmax_vec(:)
    jpmin_vec(0)=1
    do m=1,nproc-1
      if (jpmax_vec(m)==0) then
        jpmin_vec(m)=-1
        jpmax_vec(m)=-2
      else
        jpmin_vec(m)=jpmax_vec(m-1)+1
        jpmax_vec(m)=jpmin_vec(m)+jpmax_vec(m)-1
      endif
    enddo
    do m=0,nproc-1
      if (jpmin_vec(m)<0) then
        !ntodo_vec(m)=0
        continue
      else
        !ntodo_vec(m)=jpmax_vec(m)-jpmin_vec(m)+1
        jlist_matrix(1:ntodo_vec(m),m)=jlist_todo(jpmin_vec(m):jpmax_vec(m))
        do k=1,ntodo_vec(m)
          proc_assignment(jlist_matrix(k,m))=m
        enddo
      endif
    enddo
  endif
else
  allocate(ntodo_vec(0))
  allocate(jlist_matrix(0,0))
endif

call dlf_global_int_bcast(proc_assignment,ngrad,0)
call dlf_global_int_scatter_rank0(ntodo_vec,ntodo_this_process,nproc,0)
call dlf_global_int_scatter_rank1(jlist_matrix,ngrad,jlist_this_proc,nproc,0)

deallocate(ntodo_vec)
deallocate(jlist_matrix)

if (glob%iam==0) then
  deallocate(jlist_todo)
  deallocate(jpmin_vec)
  deallocate(jpmax_vec)
endif

if (hessmode) then
  if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  if(.not.silent) write(stdout,'(A)') '~~~~~~Single-Numerical~Differentiation~of~Analytical~Gradients~for~Hessian~~~~~~'
  if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
else
  if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  if(.not.silent) write(stdout,'(A)') '~Double-Numerical~Differentiation~of~Analytical~Gradients~for~Cubic/Quartic~FCs~'
  if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
endif
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of Gradient calls:              ', ngrad
if(.not.silent) write(stdout,'(A,I0)') 'Already done in previous run(s):             ', ndone
if(.not.silent) write(stdout,'(A,I0)') 'Total number of gradient calls in this run:  ', ntodo
if(.not.silent) write(stdout,'(A,I0)') 'Number of calls for this process:            ', ntodo_this_process
if(.not.silent) write(stdout,'(A)') ''

gradcount=0

! Iterate over process-specific job list

do k=1,ntodo_this_process
  j=jlist_this_proc(k)
  io=joblist(j,1)
  ip=joblist(j,2)
  so=joblist(j,3)
  sp=joblist(j,4)
  dQo=so*deltaQ_uniform
  dQp=sp*deltaQ_uniform
  coordtmp=coord0
  coordtmp=coordtmp + dQo * normal_modes_non_mw(:,io)
  if (io/=ip) then 
    coordtmp=coordtmp + dQp * normal_modes_non_mw(:,ip)
  endif
  gradcount=gradcount+1
  gradtag=''
  if (io==ip) then
    write(gradtag,'(A,I0,A,SP,I2)') 'sing_', io, '_', so
  else
    write(gradtag,'(A,I0,A,I0,A,SP,I2,A,I2)') 'doub_', io, '_', ip, '_', so, '_', sp
  endif
  if(.not.silent) then 
    write(stdout,'(3A,I0,A,I0,A)') 'Computing gradient for displaced coordinates, ', &
                       & trim(adjustl(gradtag)), &
                       & ', progress for this proc.: (',gradcount,'/',ntodo_this_process,')'
  endif
#ifdef VPT2_GRAD_DEBUG_OUT
  call grad_routine(nvar,coordtmp,energy,grad,iimage,kiter,istat,trim(adjustl(gradtag)))
#else
  call grad_routine(nvar,coordtmp,energy,grad,iimage,kiter,istat)
#endif
  grad_cart(:,j) = grad(:)
  ! write to punch file
  if (hessmode) then
    write(3499,'(A)') '$GRADIENT_CART_4HESS'
  else
    write(3499,'(A)') '$GRADIENT_CART'
  endif
  if (io==ip) then 
    write(3499,'(A)') 'S'
    write(3499,'(I0)') io
    write(3499,'(I0)') so
    write(3499,'(ES15.6)') deltaQ_uniform
  else
    write(3499,'(A)') 'D'
    write(3499,'(I0,1X,I0)') io, ip
    write(3499,'(I0,1X,I0)') so, sp
    write(3499,'(ES15.6,1X,ES15.6)') deltaQ_uniform, deltaQ_uniform
  endif
  call vector_output(grad_cart(:,j),3499,'ES24.16','__BLANK__')
enddo

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

! Broadcast all intent(out) stuff
do j=1,ngrad
  if (proc_assignment(j)<0) then
    cycle
  endif
  call dlf_global_real_bcast(grad_cart(:,j),nvar,proc_assignment(j))
enddo

grad_done(:)=.true.

close(3499)

return
end subroutine calculate_displaced_gradients_mpi

! ****************************************
! ****************************************
! ****************************************

! Calculate 3rd and 4th derivatives from finite-difference gradients
! using two sources: displacements along the actual normal modes, 
! plus the displacements along fake normal modes that were initially 
! generated for determining the Hessian from finite differences.
! The addition of those fake normal mode displacements seems to 
! deteriorate the quality of the calculated force constants, 
! so that this idea should probably be abandoned.

subroutine deriv_from_grad_list_least_sqrs_joblist_hybrid_inp(nvar, &
                  & nvareff,ngrad1,ngrad2,mv3,grad_cart1,grad_cart2, & 
                  & fake_normal_modes,normal_modes,normal_modes_non_mw, &
                  & eigenvalues,freq,hess_cart,joblist1,joblist2, &
                  & cubic_fc,quartic_fc,dQ1,dQ2,refine_freq)
use dlf_vpt2_utility, only: error_print, matrix_output, vector_output, checkeq
use dlf_linalg_interface_mod
use dlf_vpt2_freq, only: hessian_eigenvalues
use dlf_allocate, only: allocate, deallocate
use dlf_constants
use dlf_sort_module, only: dlf_sort_shell_ind
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad1,ngrad2
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad1), intent(in) :: grad_cart1
real(rk), dimension(nvar,ngrad2), intent(in) :: grad_cart2
real(rk), dimension(nvar,nvareff), intent(inout) :: fake_normal_modes, &
                                     &  normal_modes,normal_modes_non_mw
real(rk), dimension(nvareff), intent(inout) :: freq,eigenvalues
real(rk), dimension(nvar,nvar), intent(inout) :: hess_cart
integer, dimension(ngrad1,4),intent(in) :: joblist1
integer, dimension(ngrad2,4),intent(in) :: joblist2
real(rk), intent(in) :: dQ1,dQ2
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
logical, intent(in) :: refine_freq

real(rk), dimension(nvareff,ngrad1) :: qnc1
real(rk), dimension(nvareff,ngrad2) :: qnc2
real(rk), dimension(nvareff,ngrad1+ngrad2) :: qnc_all
real(rk), dimension(nvar,ngrad1+ngrad2) :: grad_all

real(rk), dimension(nvar,ngrad2) :: disp_test_fnm
real(rk), dimension(nvar,ngrad2) :: disp_test_rnm

integer :: i,j,io,so

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, &
              &  size(grad_cart1,dim=1),&
              &  size(grad_cart2,dim=1),&
              &  size(mv3), &
              &  size(hess_cart,dim=1), &
              &  size(hess_cart,dim=2), &
              &  size(fake_normal_modes,dim=1), &
              &  size(normal_modes,dim=1), &
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, deriv_from_grad_list_least_sqrs_joblist_hybrid_inp')
call checkeq( [nvareff, &
   &     size(freq), &
   &     size(eigenvalues), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(fake_normal_modes,dim=2) , &
   &     size(normal_modes,dim=2) , &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, deriv_from_grad_list_least_sqrs_joblist_hybrid_inp')
call checkeq( [ngrad1, size(grad_cart1,dim=2), size(joblist1,dim=1) ], &
   &  'dim. mismatch, deriv_from_grad_list_least_sqrs_joblist_hybrid_inp')
call checkeq( [ngrad2, size(grad_cart2,dim=2), size(joblist2,dim=1) ], &
   &  'dim. mismatch, deriv_from_grad_list_least_sqrs_joblist_hybrid_inp')
call checkeq( [4, size(joblist1,dim=2), size(joblist2,dim=2) ], &
   &  'dim. mismatch, deriv_from_grad_list_least_sqrs_joblist_hybrid_inp')
write(stdout,*) 'deriv_from_grad_list_least_sqrs_joblist_hybrid_inp: array shape check successful.'
#endif

call joblist_to_qnc(nvareff,ngrad1,joblist1,dQ1,qnc1)
call joblist_fakenm_to_qnc(nvar,nvareff,ngrad2,joblist2,normal_modes, &
                             &  fake_normal_modes,dQ2,qnc2)

!call matrix_output(qnc2,6,'ES13.5','qnc2')
!read(*,*)

disp_test_fnm(:,:)=0._rk
disp_test_rnm(:,:)=0._rk

do i=1,ngrad2
  do j=1,nvareff
    disp_test_rnm(:,i)=disp_test_rnm(:,i)+qnc2(j,i)*normal_modes(:,j)
  enddo
  io=joblist2(i,1)
  so=joblist2(i,3)
  disp_test_fnm(:,i)=so*dQ2*fake_normal_modes(:,io)
enddo

call matrix_output(disp_test_fnm(:,:)-disp_test_rnm(:,:),6,'F20.12','disp_test')

qnc_all(1:nvareff,1:ngrad1)              =qnc1(1:nvareff,1:ngrad1)
qnc_all(1:nvareff,ngrad1+1:ngrad1+ngrad2)=qnc2(1:nvareff,1:ngrad2)

grad_all(1:nvar,1:ngrad1)              =grad_cart1(1:nvar,1:ngrad1)
grad_all(1:nvar,ngrad1+1:ngrad1+ngrad2)=grad_cart2(1:nvar,1:ngrad2)

call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad1+ngrad2,mv3, &
                & grad_all,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                & hess_cart,qnc_all,cubic_fc,quartic_fc,refine_freq,.false.)

return
end subroutine deriv_from_grad_list_least_sqrs_joblist_hybrid_inp

! ****************************************
! ****************************************
! ****************************************

! Compare 3rd/4th  derivatives obtained from finite-difference gradients 
! using two different approaches: 
! 1. Conventional finite-difference formula
! 2. Linear least-squares fit to the displaced gradients
!    using 1-, 2-, and 3-mode basis functions up to 4th polynomial order

subroutine compare_fd_formula_and_llsf(grad_routine,hess_routine, &
                  & cubic_quartic_routine,ndisp,nvar, &
                  & nvareff,ngrad1,ngrad2,coord0,mv3,grad_cart1,grad_cart2, & 
                  & fake_normal_modes,normal_modes,normal_modes_non_mw, &
                  & eigenvalues,freq,hess_cart,joblist1,joblist2, &
                  & displacement_map1,cubic_fc,quartic_fc,dQ1,dQ2)
use dlf_vpt2_utility, only: vector_output
use dlf_linalg_interface_mod
!use dlf_vpt2_freq, only: hessian_eigenvalues
use dlf_allocate, only: allocate, deallocate
!use dlf_constants
!use dlf_sort_module, only: dlf_sort_shell_ind
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
  interface
#ifdef VPT2_GRAD_DEBUG_OUT
    subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
    subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
#endif
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: energy
      real(rk)  ,intent(out)   :: gradient(nvar)
      integer   ,intent(in)    :: iimage
      integer   ,intent(in)    :: kiter
      integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
      character(len=*), intent(in),optional :: calctag
#endif
    end subroutine grad_routine
  end interface
  interface
#ifdef VPT2_HESS_DEBUG_OUT
    subroutine hess_routine(nvar,coords,hessian,status,calctag)
#else
    subroutine hess_routine(nvar,coords,hessian,status)
#endif
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: hessian(nvar,nvar)
      integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
      character(len=*), intent(in),optional :: calctag
#endif
    end subroutine hess_routine
  end interface
  interface
    subroutine cubic_quartic_routine(nvar,coords,cubic,quartic,status)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: cubic(nvar,nvar,nvar)
      real(rk)  ,intent(out)   :: quartic(nvar,nvar,nvar,nvar)
      integer   ,intent(out)   :: status
    end subroutine cubic_quartic_routine
  end interface
integer, intent(in) :: ndisp,nvar,nvareff,ngrad1,ngrad2
real(rk), dimension(nvar),intent(in) :: coord0
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad1), intent(in) :: grad_cart1
real(rk), dimension(nvar,ngrad2), intent(in) :: grad_cart2
real(rk), dimension(nvar,nvareff), intent(inout) :: fake_normal_modes, &
                                     &  normal_modes,normal_modes_non_mw
real(rk), dimension(nvareff), intent(inout) :: freq,eigenvalues
real(rk), dimension(nvar,nvar), intent(inout) :: hess_cart
integer, dimension(ngrad1,4),intent(in) :: joblist1
integer, dimension(ngrad2,4),intent(in) :: joblist2
integer, dimension(nvareff,nvareff,-ndisp:+ndisp,-ndisp:+ndisp),intent(in) :: displacement_map1
real(rk), intent(in) :: dQ1,dQ2
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc

integer, parameter :: nrandom=300
integer, parameter :: iimage=1,kiter=-1
integer :: istat
real(rk) :: energy
real(rk) :: dQrand
integer :: nbas,i,j
real(rk), dimension(nvar) :: coordtmp
real(rk), dimension(nvar,ngrad1+ngrad2) :: grad_cart_all
real(rk), dimension(nvar,ngrad1+ngrad2+nrandom) :: grad_cart_all_pl_random
real(rk), dimension(nvar,ngrad1/2+ngrad2) :: grad_cart_1_sm_pl_2
real(rk), dimension(nvar,nrandom) :: grad_cart_random
real(rk), dimension(nvareff) :: grad_nc_ref
real(rk), dimension(nvareff,nvareff) :: hessian_nc
real(rk), dimension(nvareff,ngrad1) :: grad_qn_1
real(rk), dimension(nvareff,ngrad2) :: grad_qn_2
real(rk), dimension(nvareff,ngrad1+ngrad2) :: grad_qn_all
real(rk), dimension(nvareff,ngrad1+ngrad2+nrandom) :: grad_qn_all_pl_random
real(rk), dimension(nvareff,nrandom) :: grad_qn_random
real(rk), dimension(nvareff,ngrad1) :: qnc_1
real(rk), dimension(nvareff,ngrad2) :: qnc_2
real(rk), dimension(nvareff,ngrad1+ngrad2) :: qnc_all
real(rk), dimension(nvareff,ngrad1+ngrad2+nrandom) :: qnc_all_pl_random
real(rk), dimension(nvareff,ngrad1/2+ngrad2) :: qnc_1_sm_pl_2
real(rk), dimension(nvareff,nrandom) :: qnc_random
real(rk), dimension(:,:,:), allocatable :: z_vectors_1, z_vectors_2, z_vectors_all, z_vectors_random
real(rk), dimension(:,:,:), allocatable :: z_vectors_all_pl_random
real(rk), dimension(:,:,:), allocatable :: z_vectors_1_sm, z_vectors_1_lg
real(rk), dimension(:), allocatable :: coefficients_fd, coefficients_llsf, coefficients_fd_reduced_sm, coefficients_fd_reduced_lg
real(rk), dimension(:), allocatable :: coefficients_exact
real(rk) :: chisq_fd_set_1, chisq_fd_set_2, chisq_fd_set_all, chisq_fd_random_set, &
          & chisq_fd_all_pl_random, chisq_fd_set_1_sm, chisq_fd_set_1_lg
real(rk) :: chisq_ex_set_1, chisq_ex_set_2, chisq_ex_set_all, chisq_ex_random_set, &
          & chisq_ex_all_pl_random, chisq_ex_set_1_sm, chisq_ex_set_1_lg
real(rk) :: chisq_llsf_set_1, chisq_llsf_set_2, chisq_llsf_set_all, chisq_llsf_random_set, & 
          & chisq_llsf_all_pl_random, chisq_llsf_set_1_sm, chisq_llsf_set_1_lg
integer :: io,ip,m,k

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, &
              &  size(grad_cart1,dim=1),&
              &  size(grad_cart2,dim=1),&
              &  size(mv3), &
              &  size(hess_cart,dim=1), &
              &  size(hess_cart,dim=2), &
              &  size(fake_normal_modes,dim=1), &
              &  size(normal_modes,dim=1), &
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
call checkeq( [nvareff, &
   &     size(freq), &
   &     size(eigenvalues), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(fake_normal_modes,dim=2) , &
   &     size(normal_modes,dim=2) , &
   &     size(displacement_map1,dim=1) , &
   &     size(displacement_map1,dim=2) , &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
call checkeq( [ngrad1, size(grad_cart1,dim=2), size(joblist1,dim=1) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
call checkeq( [ngrad2, size(grad_cart2,dim=2), size(joblist2,dim=1) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
call checkeq( [4, size(joblist1,dim=2), size(joblist2,dim=2) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
call checkeq( [2*ndisp+1, &
   &           size(displacement_map1,dim=3), &
   &           size(displacement_map1,dim=4) ], &
   &  'dim. mismatch, compare_fd_formula_and_llsf')
write(stdout,*) 'compare_fd_formula_and_llsf: array shape check successful.'
#endif

open(3344,file='fit_coefficients.dat')

cubic_fc(:,:,:)=0._rk
quartic_fc(:,:,:)=0._rk

nbas=nvareff*(nvareff+1)*(nvareff+2)
nbas=2*nbas/3
nbas=nbas -nvareff*(nvareff-1)

!!do io=1,nvareff
!!  do ip=1,nvareff
!!    write(*,'(25(I0,1X))') ((displacement_map1(io,ip,m,k), k=-2,2),m=-2,2)
!!  enddo
!!enddo
!!
!!read(*,*)

call allocate(coefficients_fd,nbas)
call allocate(coefficients_fd_reduced_sm,nbas)
call allocate(coefficients_fd_reduced_lg,nbas)
call allocate(coefficients_llsf,nbas)
call allocate(coefficients_exact,nbas)

call force_constants_to_fit_coefficients_exact(grad_routine,hess_routine, &
              &  cubic_quartic_routine,nvar,nvareff,nbas,coord0,mv3, &
              &  normal_modes,normal_modes_non_mw,coefficients_exact)

if (ndisp==1) then
  call derivatives_from_gradient_list(nvar,nvareff,ngrad1,grad_cart1,&
                & eigenvalues, normal_modes_non_mw, &
                & displacement_map1,cubic_fc,quartic_fc,dQ1,grad_nc_ref)
  hessian_nc(:,:)=0._rk
  do i=1,nvareff
    hessian_nc(i,i)=eigenvalues(i)
  enddo
  call force_constants_to_fit_coefficients(nvareff,nbas,grad_nc_ref,hessian_nc,cubic_fc,quartic_fc,coefficients_fd)
elseif (ndisp==2) then
  call derivatives_from_gradient_list(nvar,nvareff,ngrad1/2,grad_cart1(:,1:ngrad1/2),&
                & eigenvalues, normal_modes_non_mw, &
                & displacement_map1(1:nvareff,1:nvareff,-1:1,-1:1),cubic_fc,quartic_fc,dQ1,grad_nc_ref)
  hessian_nc(:,:)=0._rk
  do i=1,nvareff
    hessian_nc(i,i)=eigenvalues(i)
  enddo
  call force_constants_to_fit_coefficients(nvareff,nbas,grad_nc_ref,hessian_nc,cubic_fc,quartic_fc,coefficients_fd_reduced_sm)
  call derivatives_from_gradient_list(nvar,nvareff,ngrad1/2,grad_cart1(:,ngrad1/2+1:ngrad1),&
                & eigenvalues, normal_modes_non_mw, &
                & displacement_map1(1:nvareff,1:nvareff,-2:2:2,-2:2:2)-ngrad1/2,cubic_fc,quartic_fc,2*dQ1,grad_nc_ref)
  hessian_nc(:,:)=0._rk
  do i=1,nvareff
    hessian_nc(i,i)=eigenvalues(i)
  enddo
  call force_constants_to_fit_coefficients(nvareff,nbas,grad_nc_ref,hessian_nc,cubic_fc,quartic_fc,coefficients_fd_reduced_lg)
  call derivatives_from_gradient_list_dual_step(nvar,nvareff,ngrad1,mv3, &
            & grad_cart1,normal_modes,normal_modes_non_mw,eigenvalues, &
            & freq, hess_cart, displacement_map1,cubic_fc,quartic_fc,dQ1,&
            & .true.,hessian_nc,grad_nc_ref)
  call force_constants_to_fit_coefficients(nvareff,nbas,grad_nc_ref,hessian_nc,cubic_fc,quartic_fc,coefficients_fd)
else
  return
endif

call vector_output(coefficients_exact,3344,'ES13.5','coefficients_exact')
call vector_output(coefficients_fd,3344,'ES13.5','coefficients_fd')
call vector_output(coefficients_fd-coefficients_exact,3344,'ES13.5','difference, FD - exact')
if (ndisp==2) then
  call vector_output(coefficients_fd_reduced_sm,3344,'ES13.5','coefficients_fd_reduced_sm')
  call vector_output(coefficients_fd_reduced_lg,3344,'ES13.5','coefficients_fd_reduced_lg')
endif

! Create random set
dQrand=dQ2
call random_seed()
do j=1,nrandom
  do i=1,nvareff
    call random_number(qnc_random(i,j))
  enddo
enddo
qnc_random(:,:)=qnc_random(:,:)-0.5_rk
do j=1,nrandom
  qnc_random(:,j)=dQrand*qnc_random(:,j)/sqrt(dot_product(qnc_random(:,j),qnc_random(:,j)))
enddo

do j=1,nrandom
  coordtmp(:)=coord0(:)
  do i=1,nvareff
    coordtmp=coordtmp + qnc_random(i,j) * normal_modes_non_mw(:,i)
  enddo
#ifdef VPT2_GRAD_DEBUG_OUT
  call grad_routine(nvar,coordtmp,energy,grad_cart_random(:,j),iimage,kiter,istat,trim(adjustl(gradtag)))
#else
  call grad_routine(nvar,coordtmp,energy,grad_cart_random(:,j),iimage,kiter,istat)
#endif
  grad_qn_random(:,j)=dlf_matmul_simp(transpose(normal_modes_non_mw),grad_cart_random(:,j))
enddo
! end create random set

do i=1,ngrad1
  grad_qn_1(:,i)=dlf_matmul_simp(transpose(normal_modes_non_mw),grad_cart1(:,i))
enddo
do i=1,ngrad2
  grad_qn_2(:,i)=dlf_matmul_simp(transpose(normal_modes_non_mw),grad_cart2(:,i))
enddo

grad_qn_all(1:nvareff,1:ngrad1)              =grad_qn_1(1:nvareff,1:ngrad1)
grad_qn_all(1:nvareff,ngrad1+1:ngrad1+ngrad2)=grad_qn_2(1:nvareff,1:ngrad2)

grad_qn_all_pl_random(1:nvareff,1:ngrad1+ngrad2)=grad_qn_all(1:nvareff,1:ngrad1+ngrad2)
grad_qn_all_pl_random(1:nvareff,ngrad1+ngrad2+1:ngrad1+ngrad2+nrandom)=grad_qn_random(1:nvareff,1:nrandom)

grad_cart_all(1:nvar,1:ngrad1)              =grad_cart1(1:nvar,1:ngrad1)
grad_cart_all(1:nvar,ngrad1+1:ngrad1+ngrad2)=grad_cart2(1:nvar,1:ngrad2)

grad_cart_all_pl_random(1:nvar,1:ngrad1+ngrad2)=grad_cart_all(1:nvar,1:ngrad1+ngrad2)
grad_cart_all_pl_random(1:nvar,ngrad1+ngrad2+1:ngrad1+ngrad2+nrandom)=grad_cart_random(1:nvar,1:nrandom)

call joblist_to_qnc(nvareff,ngrad1,joblist1,dQ1,qnc_1)
call joblist_fakenm_to_qnc(nvar,nvareff,ngrad2,joblist2,normal_modes, &
                             &  fake_normal_modes,dQ2,qnc_2)

qnc_all(1:nvareff,1:ngrad1)              =qnc_1(1:nvareff,1:ngrad1)
qnc_all(1:nvareff,ngrad1+1:ngrad1+ngrad2)=qnc_2(1:nvareff,1:ngrad2)

qnc_all_pl_random(1:nvareff,1:ngrad1+ngrad2)=qnc_all(1:nvareff,1:ngrad1+ngrad2)
qnc_all_pl_random(1:nvareff,ngrad1+ngrad2+1:ngrad1+ngrad2+nrandom)=qnc_random(1:nvareff,1:nrandom)

call allocate(z_vectors_1,nvareff,ngrad1,nbas)
call allocate(z_vectors_2,nvareff,ngrad2,nbas)
call allocate(z_vectors_all,nvareff,ngrad1+ngrad2,nbas)
call allocate(z_vectors_random,nvareff,nrandom,nbas)
call allocate(z_vectors_all_pl_random,nvareff,ngrad1+ngrad2+nrandom,nbas)
if (ndisp==2) then
  call allocate(z_vectors_1_sm,nvareff,ngrad1/2,nbas)
  call allocate(z_vectors_1_lg,nvareff,ngrad1/2,nbas)
endif

call get_z_vectors(nvareff,ngrad1,nbas,qnc_1,z_vectors_1)
call get_z_vectors(nvareff,ngrad2,nbas,qnc_2,z_vectors_2)
call get_z_vectors(nvareff,ngrad1+ngrad2,nbas,qnc_all,z_vectors_all)
call get_z_vectors(nvareff,nrandom,nbas,qnc_random,z_vectors_random)
call get_z_vectors(nvareff,ngrad1+ngrad2+nrandom,nbas,qnc_all_pl_random,z_vectors_all_pl_random)
if (ndisp==2) then
  call get_z_vectors(nvareff,ngrad1/2,nbas,qnc_1(:,1:ngrad1/2),z_vectors_1_sm)
  call get_z_vectors(nvareff,ngrad1/2,nbas,qnc_1(:,ngrad1/2+1:ngrad1),z_vectors_1_lg)
endif

chisq_ex_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_exact,z_vectors_1,grad_qn_1)
chisq_ex_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_exact,z_vectors_2,grad_qn_2)
chisq_ex_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_exact,z_vectors_all,grad_qn_all)
chisq_ex_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_exact,z_vectors_random,grad_qn_random)
chisq_ex_all_pl_random=chi_square_fit &
      & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_exact,z_vectors_all_pl_random,grad_qn_all_pl_random)
if (ndisp==2) then
  chisq_ex_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_exact,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_ex_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_exact,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
endif

write(*,*) 'Chi**2/ngrad, using coefficients from analytical 3rd/4th'
write(*,*) 'chisq_ex_set_1/ngrad1            = ', chisq_ex_set_1/real(ngrad1)
write(*,*) 'chisq_ex_set_2/ngrad2            = ', chisq_ex_set_2/real(ngrad2)
write(*,*) 'chisq_ex_set_all/(ngrad1+ngrad2) = ', chisq_ex_set_all/real(ngrad1+ngrad2)
write(*,*) 'chisq_ex_random_set/(nrandom)    = ', chisq_ex_random_set/real(nrandom)
write(*,*) 'chisq_ex_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_ex_all_pl_random/real(ngrad1+ngrad2+nrandom)
if (ndisp==2) then
  write(*,*) 'chisq_ex_set_1_sm/(ngrad1/2)     = ', 2*chisq_ex_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_ex_set_1_lg/(ngrad1/2)     = ', 2*chisq_ex_set_1_lg/real(ngrad1)
endif

if (ndisp==2) then
  chisq_fd_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_fd_reduced_sm,z_vectors_1,grad_qn_1)
  chisq_fd_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_fd_reduced_sm,z_vectors_2,grad_qn_2)
  chisq_fd_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_fd_reduced_sm,z_vectors_all,grad_qn_all)
  chisq_fd_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_fd_reduced_sm,z_vectors_random,grad_qn_random)
  chisq_fd_all_pl_random=chi_square_fit &
     & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_fd_reduced_sm,z_vectors_all_pl_random,grad_qn_all_pl_random)
  chisq_fd_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd_reduced_sm,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_fd_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd_reduced_sm,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  
  write(*,*) 'Chi**2/ngrad, using coefficients from finite-difference formula (reduced, single-step-size set[sm])'
  write(*,*) 'chisq_fd_set_1/ngrad1            = ', chisq_fd_set_1/real(ngrad1)
  write(*,*) 'chisq_fd_set_2/ngrad2            = ', chisq_fd_set_2/real(ngrad2)
  write(*,*) 'chisq_fd_set_all/(ngrad1+ngrad2) = ', chisq_fd_set_all/real(ngrad1+ngrad2)
  write(*,*) 'chisq_fd_random_set/(nrandom)    = ', chisq_fd_random_set/real(nrandom)
  write(*,*) 'chisq_fd_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_fd_all_pl_random/real(ngrad1+ngrad2+nrandom)
  write(*,*) 'chisq_fd_set_1_sm/(ngrad1/2)     = ', 2*chisq_fd_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_fd_set_1_lg/(ngrad1/2)     = ', 2*chisq_fd_set_1_lg/real(ngrad1)
  
  chisq_fd_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_fd_reduced_lg,z_vectors_1,grad_qn_1)
  chisq_fd_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_fd_reduced_lg,z_vectors_2,grad_qn_2)
  chisq_fd_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_fd_reduced_lg,z_vectors_all,grad_qn_all)
  chisq_fd_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_fd_reduced_lg,z_vectors_random,grad_qn_random)
  chisq_fd_all_pl_random=chi_square_fit &
   & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_fd_reduced_lg,z_vectors_all_pl_random,grad_qn_all_pl_random)
  chisq_fd_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd_reduced_lg,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_fd_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd_reduced_lg,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  
  write(*,*) 'Chi**2/ngrad, using coefficients from finite-difference formula (reduced, single-step-size set[lg])'
  write(*,*) 'chisq_fd_set_1/ngrad1            = ', chisq_fd_set_1/real(ngrad1)
  write(*,*) 'chisq_fd_set_2/ngrad2            = ', chisq_fd_set_2/real(ngrad2)
  write(*,*) 'chisq_fd_set_all/(ngrad1+ngrad2) = ', chisq_fd_set_all/real(ngrad1+ngrad2)
  write(*,*) 'chisq_fd_random_set/(nrandom)    = ', chisq_fd_random_set/real(nrandom)
  write(*,*) 'chisq_fd_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_fd_all_pl_random/real(ngrad1+ngrad2+nrandom)
  write(*,*) 'chisq_fd_set_1_sm/(ngrad1/2)     = ', 2*chisq_fd_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_fd_set_1_lg/(ngrad1/2)     = ', 2*chisq_fd_set_1_lg/real(ngrad1)
endif

chisq_fd_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_fd,z_vectors_1,grad_qn_1)
chisq_fd_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_fd,z_vectors_2,grad_qn_2)
chisq_fd_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_fd,z_vectors_all,grad_qn_all)
chisq_fd_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_fd,z_vectors_random,grad_qn_random)
chisq_fd_all_pl_random=chi_square_fit &
   & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_fd,z_vectors_all_pl_random,grad_qn_all_pl_random)

write(*,*) 'Chi**2/ngrad, using coefficients from finite-difference formula'
write(*,*) 'chisq_fd_set_1/ngrad1            = ', chisq_fd_set_1/real(ngrad1)
write(*,*) 'chisq_fd_set_2/ngrad2            = ', chisq_fd_set_2/real(ngrad2)
write(*,*) 'chisq_fd_set_all/(ngrad1+ngrad2) = ', chisq_fd_set_all/real(ngrad1+ngrad2)
write(*,*) 'chisq_fd_random_set/(nrandom)    = ', chisq_fd_random_set/real(nrandom)
write(*,*) 'chisq_fd_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_fd_all_pl_random/real(ngrad1+ngrad2+nrandom)

if (ndisp==2) then
  chisq_fd_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_fd_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_fd,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  write(*,*) 'chisq_fd_set_1_sm/(ngrad1/2)     = ', 2*chisq_fd_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_fd_set_1_lg/(ngrad1/2)     = ', 2*chisq_fd_set_1_lg/real(ngrad1)
endif

!read(*,*)

call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad1,mv3, &
                  & grad_cart1,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,qnc_1,cubic_fc,quartic_fc, .false., .false.,coefficients_llsf)
                  
call vector_output(coefficients_llsf,3344,'ES13.5','coefficients_llsf, set 1 only')

chisq_llsf_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_llsf,z_vectors_1,grad_qn_1)
chisq_llsf_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_llsf,z_vectors_2,grad_qn_2)
chisq_llsf_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_llsf,z_vectors_all,grad_qn_all)
chisq_llsf_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_llsf,z_vectors_random,grad_qn_random)
chisq_llsf_all_pl_random=chi_square_fit & 
  & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_llsf,z_vectors_all_pl_random,grad_qn_all_pl_random)

write(*,*) 'Chi**2/ngrad, using coefficients from LLSF (set 1 only)'
write(*,*) 'chisq_llsf_set_1/ngrad1            = ', chisq_llsf_set_1/real(ngrad1)
write(*,*) 'chisq_llsf_set_2/ngrad2            = ', chisq_llsf_set_2/real(ngrad2)
write(*,*) 'chisq_llsf_set_all/(ngrad1+ngrad2) = ', chisq_llsf_set_all/real(ngrad1+ngrad2)
write(*,*) 'chisq_llsf_random_set/(nrandom)    = ', chisq_llsf_random_set/real(nrandom)
write(*,*) 'chisq_llsf_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_llsf_all_pl_random/real(ngrad1+ngrad2+nrandom)

if (ndisp==2) then
  chisq_llsf_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_llsf_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  write(*,*) 'chisq_llsf_set_1_sm/(ngrad1/2)     = ', 2*chisq_llsf_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_llsf_set_1_lg/(ngrad1/2)     = ', 2*chisq_llsf_set_1_lg/real(ngrad1)
endif

!read(*,*)

call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad1+ngrad2+nrandom,mv3, &
                  & grad_cart_all_pl_random,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,qnc_all_pl_random,cubic_fc,quartic_fc,.false.,.false.,coefficients_llsf)
                  
call vector_output(coefficients_llsf,3344,'ES13.5','coefficients_llsf, set 1 + 2 + random')

chisq_llsf_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_llsf,z_vectors_1,grad_qn_1)
chisq_llsf_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_llsf,z_vectors_2,grad_qn_2)
chisq_llsf_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_llsf,z_vectors_all,grad_qn_all)
chisq_llsf_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_llsf,z_vectors_random,grad_qn_random)
chisq_llsf_all_pl_random=chi_square_fit &
  & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_llsf,z_vectors_all_pl_random,grad_qn_all_pl_random)

write(*,*) 'Chi**2/ngrad, using coefficients from LLSF (set 1 + 2 + random)'
write(*,*) 'chisq_llsf_set_1/ngrad1            = ', chisq_llsf_set_1/real(ngrad1)
write(*,*) 'chisq_llsf_set_2/ngrad2            = ', chisq_llsf_set_2/real(ngrad2)
write(*,*) 'chisq_llsf_set_all/(ngrad1+ngrad2) = ', chisq_llsf_set_all/real(ngrad1+ngrad2)
write(*,*) 'chisq_llsf_random_set/(nrandom)    = ', chisq_llsf_random_set/real(nrandom)
write(*,*) 'chisq_llsf_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_llsf_all_pl_random/real(ngrad1+ngrad2+nrandom)

if (ndisp==2) then
  chisq_llsf_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_llsf_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  write(*,*) 'chisq_llsf_set_1_sm/(ngrad1/2)     = ', 2*chisq_llsf_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_llsf_set_1_lg/(ngrad1/2)     = ', 2*chisq_llsf_set_1_lg/real(ngrad1)
endif

if (ndisp==2) then
  grad_cart_1_sm_pl_2(:,1:ngrad1/2)=grad_cart1(:,1:ngrad1/2)
  grad_cart_1_sm_pl_2(:,ngrad1/2+1:ngrad1/2+ngrad2)=grad_cart2(:,1:ngrad2)
  qnc_1_sm_pl_2(:,1:ngrad1/2)=qnc_1(:,1:ngrad1/2)
  qnc_1_sm_pl_2(:,ngrad1/2+1:ngrad1/2+ngrad2)=qnc_2(:,1:ngrad2)
  call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad1/2+ngrad2,mv3, &
                    & grad_cart_1_sm_pl_2,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                    & hess_cart,qnc_1_sm_pl_2,cubic_fc,quartic_fc,.false.,.false.,coefficients_llsf)
                    
  call vector_output(coefficients_llsf,3344,'ES13.5','coefficients_llsf, half set 1 + set 2')
  
  chisq_llsf_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_llsf,z_vectors_1,grad_qn_1)
  chisq_llsf_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_llsf,z_vectors_2,grad_qn_2)
  chisq_llsf_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_llsf,z_vectors_all,grad_qn_all)
  chisq_llsf_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_llsf,z_vectors_random,grad_qn_random)
  chisq_llsf_all_pl_random=chi_square_fit & 
  & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_llsf,z_vectors_all_pl_random,grad_qn_all_pl_random)
  
  write(*,*) 'Chi**2/ngrad, using coefficients from LLSF (half set 1 + set 2)'
  write(*,*) 'chisq_llsf_set_1/ngrad1            = ', chisq_llsf_set_1/real(ngrad1)
  write(*,*) 'chisq_llsf_set_2/ngrad2            = ', chisq_llsf_set_2/real(ngrad2)
  write(*,*) 'chisq_llsf_set_all/(ngrad1+ngrad2) = ', chisq_llsf_set_all/real(ngrad1+ngrad2)
  write(*,*) 'chisq_llsf_random_set/(nrandom)    = ', chisq_llsf_random_set/real(nrandom)
  write(*,*) 'chisq_llsf_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_llsf_all_pl_random/real(ngrad1+ngrad2+nrandom)
  
  if (ndisp==2) then
    chisq_llsf_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
    chisq_llsf_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
    write(*,*) 'chisq_llsf_set_1_sm/(ngrad1/2)     = ', 2*chisq_llsf_set_1_sm/real(ngrad1)
    write(*,*) 'chisq_llsf_set_1_lg/(ngrad1/2)     = ', 2*chisq_llsf_set_1_lg/real(ngrad1)
  endif
endif

!read(*,*)

call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad1+ngrad2,mv3, &
                  & grad_cart_all,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,qnc_all,cubic_fc,quartic_fc,.false.,.false.,coefficients_llsf)
                  
call vector_output(coefficients_llsf,3344,'ES13.5','coefficients_llsf, set 1 + set 2')

chisq_llsf_set_1  =chi_square_fit(nvareff,ngrad1,nbas,coefficients_llsf,z_vectors_1,grad_qn_1)
chisq_llsf_set_2  =chi_square_fit(nvareff,ngrad2,nbas,coefficients_llsf,z_vectors_2,grad_qn_2)
chisq_llsf_set_all=chi_square_fit(nvareff,ngrad1+ngrad2,nbas,coefficients_llsf,z_vectors_all,grad_qn_all)
chisq_llsf_random_set=chi_square_fit(nvareff,nrandom,nbas,coefficients_llsf,z_vectors_random,grad_qn_random)
chisq_llsf_all_pl_random=chi_square_fit &
  & (nvareff,ngrad1+ngrad2+nrandom,nbas,coefficients_llsf,z_vectors_all_pl_random,grad_qn_all_pl_random)

write(*,*) 'Chi**2/ngrad, using coefficients from LLSF (set 1 + 2)'
write(*,*) 'chisq_llsf_set_1/ngrad1            = ', chisq_llsf_set_1/real(ngrad1)
write(*,*) 'chisq_llsf_set_2/ngrad2            = ', chisq_llsf_set_2/real(ngrad2)
write(*,*) 'chisq_llsf_set_all/(ngrad1+ngrad2) = ', chisq_llsf_set_all/real(ngrad1+ngrad2)
write(*,*) 'chisq_llsf_random_set/(nrandom)    = ', chisq_llsf_random_set/real(nrandom)
write(*,*) 'chisq_llsf_ALL/(ngrad1+ngrad2+nrnd)= ', chisq_llsf_all_pl_random/real(ngrad1+ngrad2+nrandom)

if (ndisp==2) then
  chisq_llsf_set_1_sm=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_sm,grad_qn_1(:,1:ngrad1/2))
  chisq_llsf_set_1_lg=chi_square_fit(nvareff,ngrad1/2,nbas,coefficients_llsf,z_vectors_1_lg,grad_qn_1(:,ngrad1/2+1:ngrad1))
  write(*,*) 'chisq_llsf_set_1_sm/(ngrad1/2)     = ', 2*chisq_llsf_set_1_sm/real(ngrad1)
  write(*,*) 'chisq_llsf_set_1_lg/(ngrad1/2)     = ', 2*chisq_llsf_set_1_lg/real(ngrad1)
endif

!read(*,*)

cubic_fc(:,:,:)=0._rk
quartic_fc(:,:,:)=0._rk
if (ndisp==1) then
  call derivatives_from_gradient_list(nvar,nvareff,ngrad1,grad_cart1,&
                & eigenvalues, normal_modes_non_mw, &
                & displacement_map1,cubic_fc,quartic_fc,dQ1,grad_nc_ref)
elseif (ndisp==2) then
  call derivatives_from_gradient_list_dual_step(nvar,nvareff,ngrad1,mv3, &
            & grad_cart1,normal_modes,normal_modes_non_mw,eigenvalues, &
            & freq, hess_cart, displacement_map1,cubic_fc,quartic_fc,dQ1,&
            & .false.,hessian_nc,grad_nc_ref)
endif

close(3344)

return

end subroutine compare_fd_formula_and_llsf

! ****************************************
! ****************************************
! ****************************************

! Convert cubic and quartic force constants to 
! coefficients for the basis functions of the linear least-squares fit

subroutine force_constants_to_fit_coefficients(nvareff,nbas,g0,hess,cubic,quartic,coeff)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq, matrix_output,vector_output
#endif
  implicit none
  integer, intent(in) :: nvareff, nbas
  real(rk), dimension(nvareff), intent(in) :: g0
  real(rk), dimension(nvareff,nvareff), intent(in) :: hess
  real(rk), dimension(nvareff,nvareff,nvareff), intent(in) :: cubic, quartic
  real(rk), dimension(nbas), intent(out) :: coeff
  
  integer :: i, io, ip, is, inu, jnu, knu
  integer, dimension(3) :: ot

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nbas, &
                &  size(coeff) ], &
     &  'dim. mismatch, force_constants_to_fit_coefficients')
  call checkeq( [nvareff, &
     &     size(g0), &
     &     size(hess,dim=1), &
     &     size(hess,dim=2), &
     &     size(cubic,dim=1), size(cubic,dim=2), &
     &     size(cubic,dim=3), size(quartic,dim=1), &
     &     size(quartic,dim=2), size(quartic,dim=3) &
     &     ], &
     &  'dim. mismatch, force_constants_to_fit_coefficients')
  write(stdout,*) 'force_constants_to_fit_coefficients: array shape check successful.'
#endif
  
  !!call vector_output(g0,6,'F20.12','g0')
  !!call matrix_output(hess,6,'F20.12','hess')
  !!call matrix_output(cubic(1,:,:),6,'F20.12','cubic1')
  !!call matrix_output(quartic(1,:,:),6,'F20.12','quartic1')
  
  i=0
  ! Type 1, 1D linear => gradient at reference geometry (ideally ~zero)
  do io=1,nvareff
    i=i+1
    coeff(i)=g0(io)
  enddo
  ! Type 2, 1D quadratic => H_oo
  do io=1,nvareff
    i=i+1
    coeff(i)=hess(io,io)
  enddo
  ! Type 3, 1D cubic => t_ooo
  do io=1,nvareff
    i=i+1
    coeff(i)=cubic(io,io,io)
  enddo
  ! Type 4, 1D quartic => u_oooo
  do io=1,nvareff
    i=i+1
    coeff(i)=quartic(io,io,io)
  enddo
  ! Type 5, 2D quadratic => H_op (ideally close to zero)
  do io=1,nvareff
    do ip=1,io-1
      i=i+1
      coeff(i)=hess(io,ip)
    enddo
  enddo
  ! Type 6, 2D cubic => t_oop
  do io=1,nvareff
    do ip=1,nvareff
      if (io==ip) cycle
      i=i+1
      ot=ot3(io,io,ip)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      coeff(i)=0.5_rk*cubic(inu,jnu,knu)
    enddo
  enddo
  ! Type 7, 2D quartic (A) => u_ooop
  do io=1,nvareff
    do ip=1,nvareff
      if (io==ip) cycle
      i=i+1
      ot=ot4(io,io,ip)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      coeff(i)=(1._rk/6._rk)*quartic(inu,jnu,knu)
    enddo
  enddo
  ! Type 8, 2D quartic (B) => u_oopp
  do io=1,nvareff
    do ip=1,io-1
      i=i+1
      ot=ot4(io,ip,ip)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      coeff(i)=0.25_rk*quartic(inu,jnu,knu)
    enddo
  enddo
  ! Type 9, 3D cubic => t_ops
  do io=1,nvareff
    do ip=1,io-1
      do is=1,ip-1
        i=i+1
        ot=ot3(io,ip,is)
        inu=ot(1)
        jnu=ot(2)
        knu=ot(3)
        coeff(i)=cubic(inu,jnu,knu)
      enddo
    enddo
  enddo
  ! Type 10, 3D quartic => u_oops
  do io=1,nvareff
    do ip=1,nvareff
      if (ip==io) cycle
      do is=1,ip-1
        if (is==io) cycle
        i=i+1
        ot=ot4(io,ip,is)
        inu=ot(1)
        jnu=ot(2)
        knu=ot(3)
        coeff(i)=0.5_rk*quartic(inu,jnu,knu)
      enddo
    enddo
  enddo
  
  return

end subroutine force_constants_to_fit_coefficients

! ****************************************
! ****************************************
! ****************************************

! For comparison purposes only
! Convert cubic and quartic force constants provided by the analytic routine
! specified by "cubic_quartic_routine" to fit coefficients of 
! the linear least-squares fit

subroutine force_constants_to_fit_coefficients_exact(grad_routine,hess_routine, &
                  &  cubic_quartic_routine,nvar,nvareff,nbas,coords,mv3, &
                  &  normal_modes,normal_modes_non_mw,coefficients)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq, matrix_output,vector_output
#endif
  use dlf_linalg_interface_mod
  implicit none
  interface
#ifdef VPT2_GRAD_DEBUG_OUT
    subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
    subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
#endif
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: energy
      real(rk)  ,intent(out)   :: gradient(nvar)
      integer   ,intent(in)    :: iimage
      integer   ,intent(in)    :: kiter
      integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
      character(len=*), intent(in),optional :: calctag
#endif
    end subroutine grad_routine
  end interface
  interface
#ifdef VPT2_HESS_DEBUG_OUT
    subroutine hess_routine(nvar,coords,hessian,status,calctag)
#else
    subroutine hess_routine(nvar,coords,hessian,status)
#endif
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: hessian(nvar,nvar)
      integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
      character(len=*), intent(in),optional :: calctag
#endif
    end subroutine hess_routine
  end interface
  interface
    subroutine cubic_quartic_routine(nvar,coords,cubic,quartic,status)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: cubic(nvar,nvar,nvar)
      real(rk)  ,intent(out)   :: quartic(nvar,nvar,nvar,nvar)
      integer   ,intent(out)   :: status
    end subroutine cubic_quartic_routine
  end interface
  integer, intent(in) :: nvar,nvareff, nbas
  real(rk), dimension(nvar), intent(in)  :: coords, mv3
  real(rk), dimension(nvar,nvareff), intent(in)  :: normal_modes,normal_modes_non_mw
  real(rk), dimension(nbas), intent(out) :: coefficients
  
  integer, parameter :: iimage=1,kiter=-1
  integer :: istat
  real(rk) :: energy
  real(rk), dimension(nvar) :: grad0_cart
  real(rk), dimension(nvareff) :: grad0_qn
  real(rk), dimension(nvar,nvar) :: hessian_cart
  real(rk), dimension(nvareff,nvareff) :: hessian_qn
  real(rk), dimension(nvareff,nvareff,nvareff) :: cubic_fc_nm, quartic_fc_nm

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nbas, size(coefficients) ], &
     &  'dim. mismatch, force_constants_to_fit_coefficients_exact')
  call checkeq( [nvar, &
     &     size(coords), &
     &     size(mv3), &
     &     size(normal_modes,dim=1), &
     &     size(normal_modes_non_mw,dim=1) &
     &     ], &
     &  'dim. mismatch, force_constants_to_fit_coefficients_exact')
  call checkeq( [nvareff, &
     &     size(normal_modes,dim=2), &
     &     size(normal_modes_non_mw,dim=2) &
     &     ], &
     &  'dim. mismatch, force_constants_to_fit_coefficients_exact')
  write(stdout,*) 'force_constants_to_fit_coefficients_exact: array shape check successful.'
#endif
  
  call grad_routine(nvar,coords,energy,grad0_cart,iimage,kiter,istat)
  call hess_routine(nvar,coords,hessian_cart,istat)
  call get_cubic_quartic_nm_via_analytical_routine(cubic_quartic_routine,nvar,nvareff, &
                                    &  coords,mv3,normal_modes,cubic_fc_nm,quartic_fc_nm)
  
  grad0_qn = dlf_matmul_simp(transpose(normal_modes_non_mw),grad0_cart)
  hessian_qn=dlf_matrix_ortho_trans(normal_modes_non_mw,hessian_cart,0)
  
  call force_constants_to_fit_coefficients(nvareff,nbas,grad0_qn,hessian_qn,cubic_fc_nm,quartic_fc_nm,coefficients)
  return

end subroutine force_constants_to_fit_coefficients_exact

! ****************************************
! ****************************************
! ****************************************

! Get 3rd/4th derivatives from precalculated list of displaced gradients. 
! A linear least-squares fit using up to quartic one-, two-, and three-mode basis functions
! is used to determine the force constants. It is assumed that the gradients have 
! been obtained using a displacement joblist with one- and two-mode displacements.

subroutine derivatives_from_gradient_list_least_squares_joblist_input(nvar,nvareff,ngrad,mv3, &
                  & grad_cart,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,joblist,cubic_fc,quartic_fc,deltaQ_uniform,refine_freq)
use dlf_vpt2_utility, only: error_print, matrix_output, vector_output, checkeq
use dlf_linalg_interface_mod
use dlf_vpt2_freq, only: hessian_eigenvalues
use dlf_allocate, only: allocate, deallocate
use dlf_constants
use dlf_sort_module, only: dlf_sort_shell_ind
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad), intent(in) :: grad_cart
real(rk), dimension(nvar,nvareff), intent(inout) :: normal_modes,normal_modes_non_mw
real(rk), dimension(nvareff), intent(inout) :: freq,eigenvalues
real(rk), dimension(nvar,nvar), intent(inout) :: hess_cart
integer, dimension(ngrad,4),intent(in) :: joblist
real(rk), intent(in) :: deltaQ_uniform
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
logical, intent(in) :: refine_freq

real(rk), dimension(nvareff,ngrad) :: qnc

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, size(grad_cart,dim=1),&
              &  size(mv3), &
              &  size(hess_cart,dim=1), &
              &  size(hess_cart,dim=2), &
              &  size(normal_modes,dim=1), &
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_joblist_input')
call checkeq( [nvareff, &
   &     size(freq), &
   &     size(eigenvalues), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(normal_modes,dim=2) , &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_joblist_input')
call checkeq( [ngrad, size(grad_cart,dim=2), size(joblist,dim=1) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_joblist_input')
call checkeq( [4, size(joblist,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_joblist_input')
write(stdout,*) 'derivatives_from_gradient_list_least_squares_joblist_input: array shape check successful.'
#endif

call joblist_to_qnc(nvareff,ngrad,joblist,deltaQ_uniform,qnc)

call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad,mv3, &
                & grad_cart,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                & hess_cart,qnc,cubic_fc,quartic_fc,refine_freq,.false.)

return
end subroutine derivatives_from_gradient_list_least_squares_joblist_input

! ****************************************
! ****************************************
! ****************************************

! Get 3rd/4th derivatives from precalculated list of displaced gradients. 
! A linear least-squares fit using up to quartic one-, two-, and three-mode basis functions
! is used to determine the force constants. Gradients from arbitrarily displaced geometries 
! can be input, by providing the normal coordinate elongations (qnc) for each gradient.

recursive subroutine derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad,mv3, &
                  & grad_cart,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,qnc,cubic_fc,quartic_fc,refine_freq,redo_fit,coefficients_out)
use dlf_vpt2_utility, only: error_print, matrix_output, vector_output, vector_out_multi, checkeq
use dlf_linalg_interface_mod
use dlf_vpt2_freq, only: hessian_eigenvalues
use dlf_allocate, only: allocate, deallocate
use dlf_constants
use dlf_sort_module, only: dlf_sort_shell_ind
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad), intent(in) :: grad_cart
real(rk), dimension(nvar,nvareff), intent(inout) :: normal_modes,normal_modes_non_mw
real(rk), dimension(nvareff), intent(inout) :: freq,eigenvalues
real(rk), dimension(nvar,nvar), intent(inout) :: hess_cart
real(rk), dimension(nvareff,ngrad), intent(in) :: qnc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
logical, intent(in) :: refine_freq, redo_fit
real(rk), dimension(:), optional, intent(out) :: coefficients_out

logical, parameter :: rotate=.false.   !, redo_fit=.true.
real(rk) :: maxabs, au2cmi, amu2au, det_dummy, chisq
integer  :: io,ip,is,isav,i,j
integer  :: inu,jnu,knu
integer  :: m,nbas,maxindex
real(rk), dimension(nvareff) :: eigval
real(rk), dimension(nvareff,nvareff) :: hessian_nc,eigvec
!integer, dimension(:), allocatable :: bf_type
!integer, dimension(:,:), allocatable :: qn_indices
real(rk), dimension(:,:), allocatable :: alpha, alpha_eigvecs, alpha_inv_eigvals, alpha_inverted
real(rk), dimension(:), allocatable   :: beta, alpha_eigvals, coefficients, variances
real(rk), dimension(:,:,:), allocatable :: z_vectors
real(rk), dimension(nvareff,ngrad) :: grad_qn
real(rk), dimension(nvareff) :: g0, freq_corrected
integer, dimension(nvareff) :: sortind
real(rk), dimension(nvareff,nvareff) :: nm1
real(rk), dimension(nvar,nvareff) :: nm_corr, old_normal_modes
real(rk), dimension(nvar,nvar) :: sqrtmass
real(rk), dimension(nvareff,nvareff,nvareff) :: cubic_fc_rot
real(rk), dimension(nvareff,nvareff,nvareff) :: quartic_fc_rot
real(rk), dimension(nvar) :: vector_of_ones
integer, dimension(3) :: ot
real(rk), dimension(nvareff,ngrad) :: qnc_new_basis
 
character(len=500) :: fn_punch

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, size(grad_cart,dim=1),&
              &  size(mv3), &
              &  size(hess_cart,dim=1), &
              &  size(hess_cart,dim=2), &
              &  size(normal_modes,dim=1), &
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_qnc_input')
call checkeq( [nvareff, &
   &     size(freq), &
   &     size(eigenvalues), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(normal_modes,dim=2) , &
   &     size(qnc,dim=1) , &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_qnc_input')
call checkeq( [ngrad, size(grad_cart,dim=2), size(qnc,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_least_squares_qnc_input')
write(stdout,*) 'derivatives_from_gradient_list_least_squares_qnc_input: array shape check successful.'
#endif

g0(:)=0._rk
hessian_nc(:,:)=0._rk
cubic_fc(:,:,:)=0._rk
quartic_fc(:,:,:)=0._rk

nbas=nvareff*(nvareff+1)*(nvareff+2)
nbas=2*nbas/3
nbas=nbas -nvareff*(nvareff-1)

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
if (present(coefficients_out)) then
  call checkeq( [nbas, size(coefficients_out) ],&
     &  'dim. mismatch, derivatives_from_gradient_list_least_squares_qnc_input')
endif
#endif

!call allocate(bf_type,nbas)
!call allocate(qn_indices,3,nbas)
call allocate(z_vectors,nvareff,ngrad,nbas)

call get_z_vectors(nvareff,ngrad,nbas,qnc,z_vectors)

do i=1,ngrad
  grad_qn(:,i)=dlf_matmul_simp(transpose(normal_modes_non_mw),grad_cart(:,i))
enddo

call allocate(alpha,nbas,nbas)
call allocate(beta,nbas)

do i=1,nbas
  do j=1,i
    alpha(i,j)=sum(z_vectors(:,:,i)*z_vectors(:,:,j))
  enddo
enddo

do i=1,nbas
  do j=i+1,nbas
    alpha(i,j)=alpha(j,i)
  enddo
enddo

do i=1,nbas
  beta(i)=sum(z_vectors(:,:,i)*grad_qn(:,:))
enddo

call allocate(alpha_eigvals,nbas)
call allocate(alpha_eigvecs,nbas,nbas)
call dlf_matrix_diagonalise(nbas,alpha,alpha_eigvals,alpha_eigvecs)
!call vector_output(alpha_eigvals,6,'ES13.5','alpha_eigvals')
!read(*,*)
!call matrix_output(alpha_eigvecs,6,'ES13.5','alpha_eigvecs')
!read(*,*)

call allocate(alpha_inv_eigvals,nbas,nbas)
call allocate(alpha_inverted,nbas,nbas)

alpha_inv_eigvals(:,:)=0._rk
do i=1,nbas
  if (alpha_eigvals(i)==0._rk) then
    alpha_inv_eigvals(i,i)=0._rk
!  if (abs(alpha_eigvals(i))<1.e-2_rk) then
!    alpha_inv_eigvals(i,i)=0._rk
  else
    alpha_inv_eigvals(i,i)=1._rk/alpha_eigvals(i)
  endif
enddo

call allocate(coefficients,nbas)
!call allocate(variances,nbas)

alpha_inverted=dlf_matrix_ortho_trans(alpha_eigvecs,alpha_inv_eigvals,1)
!alpha_inverted=alpha
!call dlf_matrix_invert(nbas,.false.,alpha_inverted,det_dummy)
coefficients=dlf_matmul_simp(alpha_inverted,beta)
if (present(coefficients_out)) then
  coefficients_out(:)=coefficients(:)
endif

!do i=1,nbas
!  !if (coefficients(i)==0._rk) then
!  !  variances(i)=-sqrt(alpha_inverted(i,i))
!  !else
!  !  variances(i)=sqrt(alpha_inverted(i,i))/abs(coefficients(i))
!  !endif
!  variances(i)=sqrt(alpha_inverted(i,i))*1.e-5_rk
!enddo

!call vector_out_multi(reshape(variances,[size(variances),1]),reshape([bf_type,pack(transpose(qn_indices),.true.)],[nbas,4]),6,'ES13.5','est. sigma')

!call vector_output(coefficients,6,'ES13.5','coefficients')
!read(*,*)

!chisq=chi_square_fit(nvareff,ngrad,nbas,coefficients,z_vectors,grad_qn)
!write(*,*) 'chisq=', chisq
!read(*,*)

! Extract force constants from the determined fit coefficients

i=0
! Type 1, 1D linear => gradient at reference geometry (ideally ~zero)
do io=1,nvareff
  i=i+1
  g0(io)=coefficients(i)
enddo
! Type 2, 1D quadratic => H_oo
do io=1,nvareff
  i=i+1
  hessian_nc(io,io)=coefficients(i)
enddo
! Type 3, 1D cubic => t_ooo
do io=1,nvareff
  i=i+1
  cubic_fc(io,io,io)=coefficients(i)
enddo
! Type 4, 1D quartic => u_oooo
do io=1,nvareff
  i=i+1
  quartic_fc(io,io,io)=coefficients(i)
enddo
! Type 5, 2D quadratic => H_op (ideally close to zero)
do io=1,nvareff
  do ip=1,io-1
    i=i+1
    hessian_nc(io,ip)=coefficients(i)
  enddo
enddo
! Type 6, 2D cubic => t_oop
do io=1,nvareff
  do ip=1,nvareff
    if (io==ip) cycle
    i=i+1
    ot=ot3(io,io,ip)
    inu=ot(1)
    jnu=ot(2)
    knu=ot(3)
    cubic_fc(inu,jnu,knu)=2*coefficients(i)
  enddo
enddo
! Type 7, 2D quartic (A) => u_ooop
do io=1,nvareff
  do ip=1,nvareff
    if (io==ip) cycle
    i=i+1
    ot=ot4(io,io,ip)
    inu=ot(1)
    jnu=ot(2)
    knu=ot(3)
    quartic_fc(inu,jnu,knu)=6*coefficients(i)
  enddo
enddo
! Type 8, 2D quartic (B) => u_oopp
do io=1,nvareff
  do ip=1,io-1
    i=i+1
    ot=ot4(io,ip,ip)
    inu=ot(1)
    jnu=ot(2)
    knu=ot(3)
    quartic_fc(inu,jnu,knu)=4*coefficients(i)
  enddo
enddo
! Type 9, 3D cubic => t_ops
do io=1,nvareff
  do ip=1,io-1
    do is=1,ip-1
      i=i+1
      ot=ot3(io,ip,is)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=coefficients(i)
    enddo
  enddo
enddo
! Type 10, 3D quartic => u_oops
do io=1,nvareff
  do ip=1,nvareff
    if (ip==io) cycle
    do is=1,ip-1
      if (is==io) cycle
      i=i+1
      ot=ot4(io,ip,is)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=2*coefficients(i)
    enddo
  enddo
enddo

call symmetric_fill_hessian(nvareff,hessian_nc,hessian_nc)

open(1115,file='dbg.out',action='write')
call vector_output(g0,1115,'ES20.12','gradient at reference geometry (in normal coordinates)')
call matrix_output(hessian_nc,1115,'ES20.12','Hessian at reference geometry (in normal coordinates)')

call hessian_eigenvalues(nvareff,hessian_nc,eigval,eigvec)
do m=1,nvareff
  maxindex=maxloc(abs(eigvec(:,m)),dim=1)
  maxabs  =eigvec(maxindex,m)
  if (maxabs < 0) eigvec(:,m)=-eigvec(:,m)
enddo
call matrix_output(eigvec,1115,'ES20.12','Hessian eigenvectors')
call vector_output(eigval,1115,'ES20.12','Hessian eigenvalues')
call vector_output(coefficients,1115,'ES13.5','coefficients') 

close(1115)

if (refine_freq) then
  old_normal_modes(:,:)=normal_modes(:,:)
  call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)
  call dlf_constants_get('AMU',amu2au)
  call dlf_sort_shell_ind(eigval,sortind)
  do i=1,nvareff
    j=sortind(i)
    eigenvalues(i)=eigval(j)
    freq_corrected(i)=sign(1._rk,eigenvalues(i))*sqrt(abs(eigenvalues(i)))*au2cmi
    nm1(:,i)=eigvec(:,j)
  enddo
  call vector_output(freq_corrected,1115,'F20.3','Frequencies')
  freq(:)=freq_corrected(:)
  hess_cart=dlf_matrix_ortho_trans(normal_modes,hessian_nc,1)
  sqrtmass(:,:)=0._rk
  do m=1,nvar
    sqrtmass(m,m)=sqrt(mv3(m))
  enddo
  hess_cart=dlf_matrix_ortho_trans(sqrtmass,hess_cart,0)
  nm_corr=dlf_matmul_simp(normal_modes,nm1)
  normal_modes(:,:)=nm_corr(:,:)
  nm_corr=dlf_matmul_simp(normal_modes_non_mw,nm1)
  normal_modes_non_mw(:,:)=nm_corr(:,:)
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'
  open(3198, file=trim(adjustl(fn_punch)), status="old", position="append", action="write")
  write(3198,'(A)') '$NORMAL_MODES_REFINED'
  call matrix_output(normal_modes,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$FREQS_REFINED'
  call vector_output(freq,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$EIGENVALUES_REFINED'
  call vector_output(eigenvalues,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$HESSIAN_CART_REFINED'
  write(3198,'(I0,1X,I0,1X,ES15.6)') 0, 0, 0._rk
  call matrix_output(hess_cart,3198,'ES24.16','__BLANK__')
  close(3198)

  if (redo_fit) then
    call qnc_old_nm_to_new_nm(nvar,nvareff,ngrad,old_normal_modes, &
                             &  normal_modes,qnc,qnc_new_basis)
    call derivatives_from_gradient_list_least_squares_qnc_input(nvar,nvareff,ngrad,mv3, &
                  & grad_cart,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,qnc_new_basis,cubic_fc,quartic_fc,.true.,.false.,coefficients)
    if (present(coefficients_out)) then
      coefficients_out(:)=coefficients(:)
    endif
  endif
  !if (rotate) then
  !  call symmetric_fill_cubic(nvareff,cubic_fc,cubic_fc)
  !  call symmetric_fill_quartic_semidiag(nvareff,quartic_fc,quartic_fc)
  !  vector_of_ones(:)=1._rk
  !  call convert_cubic_cart_to_normal_coord(nvareff,0,eigvec,vector_of_ones,cubic_fc,cubic_fc_rot)
  !  cubic_fc(:,:,:)=cubic_fc_rot(:,:,:)
  !  call rotate_quartic_force_constants_semidiag(nvareff,eigvec,quartic_fc,quartic_fc_rot)
  !  quartic_fc(:,:,:)=quartic_fc_rot(:,:,:)
  !endif

endif

call deallocate(z_vectors)
call deallocate(alpha)
call deallocate(beta)
call deallocate(alpha_eigvals)
call deallocate(alpha_eigvecs)
call deallocate(alpha_inv_eigvals)
call deallocate(alpha_inverted)
call deallocate(coefficients)

return
end subroutine derivatives_from_gradient_list_least_squares_qnc_input

! ****************************************
! ****************************************
! ****************************************

! Convert a displacement job list to normal coordinate elongations, 
! so they can be input into the more general 
! derivatives_from_gradient_list_least_squares_qnc_input routine

subroutine joblist_to_qnc(nvareff,ngrad,joblist,dQ,qnc)
use dlf_vpt2_utility, only: checkeq
implicit none
integer, intent(in) :: nvareff, ngrad
integer, dimension(ngrad,4),intent(in) :: joblist
real(rk), intent(in) :: dQ
real(rk), dimension(nvareff,ngrad), intent(out)  :: qnc

integer :: i,io,ip,so,sp

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [ngrad, &
   &     size(joblist,dim=1), &
   &     size(qnc,dim=2) ], &
   &  'dim. mismatch, joblist_to_qnc')
call checkeq( [nvareff, &
   &     size(qnc,dim=1) ], &
   &  'dim. mismatch, joblist_to_qnc')
call checkeq( [4, &
   &     size(joblist,dim=2) ], &
   &  'dim. mismatch, joblist_to_qnc')
write(stdout,*) 'joblist_to_qnc: array shape check successful.'
#endif

qnc(:,:)=0._rk

do i=1,ngrad
  io=joblist(i,1)
  ip=joblist(i,2)
  so=joblist(i,3)
  sp=joblist(i,4)
  if (io==ip) then
    qnc(io,i)=so
  else
    qnc(io,i)=so
    qnc(ip,i)=sp
  endif
enddo

qnc(:,:)=qnc(:,:)*dQ

end subroutine joblist_to_qnc

! ****************************************
! ****************************************
! ****************************************

! Convert normal mode elongations from an 'old' set of normal coordinates
! (e.g. fake normal coordinates, see finite-difference Hessian routine)
! to a new, updated set of normal coordinates

subroutine qnc_old_nm_to_new_nm(nvar,nvareff,ngrad,old_normal_modes, &
                             &  new_normal_modes,qnc_old,qnc_new)
use dlf_linalg_interface_mod
use dlf_vpt2_utility, only: checkeq
implicit none
integer, intent(in) :: nvar, nvareff, ngrad
real(rk), dimension(nvar,nvareff), intent(in) :: old_normal_modes, new_normal_modes
real(rk), dimension(nvareff,ngrad), intent(in)   :: qnc_old
real(rk), dimension(nvareff,ngrad), intent(out)  :: qnc_new

real(rk), dimension(nvareff,nvareff) :: transfo

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [ngrad, &
   &     size(qnc_old,dim=2), &
   &     size(qnc_new,dim=2) ], &
   &  'dim. mismatch, qnc_old_nm_to_new_nm')
call checkeq( [nvar, &
   &     size(old_normal_modes,dim=1), &
   &     size(new_normal_modes,dim=1) ], &
   &  'dim. mismatch, qnc_old_nm_to_new_nm')
call checkeq( [nvareff, &
   &     size(qnc_old,dim=1), &
   &     size(qnc_new,dim=1), &
   &     size(old_normal_modes,dim=2), &
   &     size(new_normal_modes,dim=2) ], &
   &  'dim. mismatch, qnc_old_nm_to_new_nm' )
write(stdout,*) 'qnc_old_nm_to_new_nm: array shape check successful.'
#endif

transfo=dlf_matmul_simp(transpose(new_normal_modes),old_normal_modes)
qnc_new=dlf_matmul_simp(transfo,qnc_old)

return

end subroutine qnc_old_nm_to_new_nm

! ****************************************
! ****************************************
! ****************************************

! Convert  job list of displacements in fake normal 
! coordinates to normal mode elongations in new set of
! normal coordinates

subroutine joblist_fakenm_to_qnc(nvar,nvareff,ngrad,joblist,normal_modes, &
                             &  fake_normal_modes,dQ,qnc)
use dlf_linalg_interface_mod
use dlf_vpt2_utility, only: checkeq
implicit none
integer, intent(in) :: nvar,nvareff, ngrad
integer, dimension(ngrad,4),intent(in) :: joblist
real(rk), dimension(nvar,nvareff), intent(in) :: normal_modes, fake_normal_modes
real(rk), intent(in) :: dQ
real(rk), dimension(nvareff,ngrad), intent(out)  :: qnc

integer :: i,io,ip,so,sp
real(rk), dimension(nvareff,ngrad) :: qnc_fnm
real(rk), dimension(nvareff,nvareff) :: transfo

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [ngrad, &
   &     size(joblist,dim=1), &
   &     size(qnc,dim=2) ], &
   &  'dim. mismatch, joblist_fakenm_to_qnc')
call checkeq( [nvar, &
   &     size(normal_modes,dim=1), &
   &     size(fake_normal_modes,dim=1) ], &
   &  'dim. mismatch, joblist_fakenm_to_qnc')
call checkeq( [nvareff, &
   &     size(qnc,dim=1), &
   &     size(normal_modes,dim=2), &
   &     size(fake_normal_modes,dim=2) ], &
   &  'dim. mismatch, joblist_fakenm_to_qnc')
call checkeq( [4, &
   &     size(joblist,dim=2) ], &
   &  'dim. mismatch, joblist_fakenm_to_qnc')
write(stdout,*) 'joblist_fakenm_to_qnc: array shape check successful.'
#endif

qnc_fnm(:,:)=0._rk

do i=1,ngrad
  io=joblist(i,1)
  ip=joblist(i,2)
  so=joblist(i,3)
  sp=joblist(i,4)
  if (io==ip) then
    qnc_fnm(io,i)=so
  else
    qnc_fnm(io,i)=so
    qnc_fnm(ip,i)=sp
  endif
enddo

qnc_fnm(:,:)=qnc_fnm(:,:)*dQ

transfo=dlf_matmul_simp(transpose(normal_modes),fake_normal_modes)
qnc=dlf_matmul_simp(transfo,qnc_fnm)

end subroutine joblist_fakenm_to_qnc

! ****************************************
! ****************************************
! ****************************************

! Calculate gradients (z vectors in this nomenclature) of the linear least-squares fit 
! basis functions for normal mode elongations qnc

subroutine get_z_vectors(nvareff,ngrad,nbas,qnc,z_vectors)
implicit none
integer, intent(in) :: nvareff,ngrad,nbas
real(rk), dimension(nvareff,ngrad), intent(in) :: qnc
real(rk), dimension(nvareff,ngrad,nbas), intent(out) :: z_vectors

integer :: i, io, ip, is, isav
integer, dimension(nbas)   :: bf_type
integer, dimension(3,nbas) :: qn_indices

bf_type(:)=0
qn_indices(:,:)=0
z_vectors(:,:,:)=0._rk

i=0
isav=0

! Type 1, 1D linear
do io=1,nvareff
  i=i+1
  bf_type(i)=1
  qn_indices(1,i)=io
enddo
write(*,'(A,X,I0,X,I0)') 'N (expected & check)', nvareff, i-isav
isav=i
! Type 2, 1D quadratic
do io=1,nvareff
  i=i+1
  bf_type(i)=2
  qn_indices(1,i)=io
enddo
write(*,'(A,X,I0,X,I0)') 'N (expected & check)', nvareff, i-isav
isav=i
! Type 3, 1D cubic
do io=1,nvareff
  i=i+1
  bf_type(i)=3
  qn_indices(1,i)=io
enddo
write(*,'(A,X,I0,X,I0)') 'N (expected & check)', nvareff, i-isav
isav=i
! Type 4, 1D quartic
do io=1,nvareff
  i=i+1
  bf_type(i)=4
  qn_indices(1,i)=io
enddo
write(*,'(A,X,I0,X,I0)') 'N (expected & check)', nvareff, i-isav
isav=i
! Type 5, 2D quadratic
do io=1,nvareff
  do ip=1,io-1
    i=i+1
    bf_type(i)=5
    qn_indices(1,i)=io
    qn_indices(2,i)=ip
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1)/2 (expected & check)', (nvareff*(nvareff-1))/2, i-isav
isav=i
! Type 6, 2D cubic
do io=1,nvareff
  do ip=1,nvareff
    if (io==ip) cycle
    i=i+1
    bf_type(i)=6
    qn_indices(1,i)=io
    qn_indices(2,i)=ip
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1) (expected & check)', nvareff*(nvareff-1), i-isav
isav=i
! Type 7, 2D quartic (A)
do io=1,nvareff
  do ip=1,nvareff
    if (io==ip) cycle
    i=i+1
    bf_type(i)=7
    qn_indices(1,i)=io
    qn_indices(2,i)=ip
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1) (expected & check)', nvareff*(nvareff-1), i-isav
isav=i
! Type 8, 2D quartic (B)
do io=1,nvareff
  do ip=1,io-1
    i=i+1
    bf_type(i)=8
    qn_indices(1,i)=io
    qn_indices(2,i)=ip
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1)/2 (expected & check)', (nvareff*(nvareff-1))/2, i-isav
isav=i
! Type 9, 3D cubic
do io=1,nvareff
  do ip=1,io-1
    do is=1,ip-1
      i=i+1
      bf_type(i)=9
      qn_indices(1,i)=io
      qn_indices(2,i)=ip
      qn_indices(3,i)=is
    enddo
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1)*(N-2)/6 (expected & check)', (nvareff*(nvareff-1)*(nvareff-2))/6, i-isav
isav=i
! Type 10, 3D quartic
do io=1,nvareff
  do ip=1,nvareff
    if (ip==io) cycle
    do is=1,ip-1
      if (is==io) cycle
      i=i+1
      bf_type(i)=10
      qn_indices(1,i)=io
      qn_indices(2,i)=ip
      qn_indices(3,i)=is
    enddo
  enddo
enddo
write(*,'(A,X,I0,X,I0)') 'N*(N-1)*(N-2)/2 (expected & check)', (nvareff*(nvareff-1)*(nvareff-2))/2, i-isav
write(*,'(A,X,I0,X,I0)') 'nbas (expected & check)', nbas, i
!read(*,*)
!call vector_output(real(bf_type,kind=rk),6,'F10.0','bf_type')
!read(*,*)
!call matrix_output(real(transpose(qn_indices),kind=rk),6,'F10.0','qn_indices')
!read(*,*)

do i=1,nbas
  call basis_func_evaluation(bf_type(i),qn_indices(1:3,i),nvareff,ngrad,qnc,z_vectors(1:nvareff,1:ngrad,i))
enddo
return

end subroutine get_z_vectors

! ****************************************
! ****************************************
! ****************************************

! Evaluate gradient of a single basis function (z vector), 
! but for all fitting points (i.e. for all qnc) 

subroutine basis_func_evaluation(bf_type,qn_ind,nvareff,npoints,qnc,zvecs)
use dlf_vpt2_utility, only: error_print, checkeq
implicit none
integer, intent(in) :: bf_type, nvareff, npoints
integer, dimension(3), intent(in) :: qn_ind
real(rk), dimension(nvareff,npoints), intent(in)  :: qnc
real(rk), dimension(nvareff,npoints), intent(out) :: zvecs

integer :: o,p,s

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [3, size(qn_ind) ], &
   &  'dim. mismatch, basis_func_evaluation')
call checkeq( [nvareff, &
   &     size(qnc,dim=1), &
   &     size(zvecs,dim=1) ], &
   &  'dim. mismatch, basis_func_evaluation')
call checkeq( [npoints, &
   &     size(qnc,dim=2), &
   &     size(zvecs,dim=2) ], &
   &  'dim. mismatch, basis_func_evaluation')
!write(stdout,*) 'basis_func_evaluation: array shape check successful.'
#endif

zvecs(:,:) = 0._rk

select case (bf_type)
  case (1)
    o=qn_ind(1)
    zvecs(o,1:npoints)=1._rk
  case (2)
    o=qn_ind(1)
    zvecs(o,1:npoints)=qnc(o,1:npoints)
  case (3)
    o=qn_ind(1)
    zvecs(o,1:npoints)=(qnc(o,1:npoints)*qnc(o,1:npoints))/2._rk
  case (4)
    o=qn_ind(1)
    zvecs(o,1:npoints)=(qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(o,1:npoints))/6._rk
  case (5)
    o=qn_ind(1)
    p=qn_ind(2)
    zvecs(o,1:npoints)=qnc(p,1:npoints)
    zvecs(p,1:npoints)=qnc(o,1:npoints)
  case (6)
    o=qn_ind(1)
    p=qn_ind(2)
    zvecs(o,1:npoints)=2*qnc(o,1:npoints)*qnc(p,1:npoints)
    zvecs(p,1:npoints)=qnc(o,1:npoints)*qnc(o,1:npoints)
  case (7)
    o=qn_ind(1)
    p=qn_ind(2)
    zvecs(o,1:npoints)=3*qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(p,1:npoints)
    zvecs(p,1:npoints)=qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(o,1:npoints)
  case (8)
    o=qn_ind(1)
    p=qn_ind(2)
    zvecs(o,1:npoints)=2*qnc(o,1:npoints)*qnc(p,1:npoints)*qnc(p,1:npoints)
    zvecs(p,1:npoints)=2*qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(p,1:npoints)
  case (9)
    o=qn_ind(1)
    p=qn_ind(2)
    s=qn_ind(3)
    zvecs(o,1:npoints)=qnc(p,1:npoints)*qnc(s,1:npoints)
    zvecs(p,1:npoints)=qnc(o,1:npoints)*qnc(s,1:npoints)
    zvecs(s,1:npoints)=qnc(o,1:npoints)*qnc(p,1:npoints)
  case (10)
    o=qn_ind(1)
    p=qn_ind(2)
    s=qn_ind(3)
    zvecs(o,1:npoints)=2*qnc(o,1:npoints)*qnc(p,1:npoints)*qnc(s,1:npoints)
    zvecs(p,1:npoints)=qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(s,1:npoints)
    zvecs(s,1:npoints)=qnc(o,1:npoints)*qnc(o,1:npoints)*qnc(p,1:npoints)
  case default
    call error_print('basis_func_evaluation: called with invalid bf_type')
end select

end subroutine basis_func_evaluation

! ****************************************
! ****************************************
! ****************************************

! Get error metric (Chi**2) for the linear least-squares fit

function chi_square_fit(nvareff,npoints,nbas,coefficients,z_vectors,grad_qn) result(X2)
use dlf_vpt2_utility, only: error_print, checkeq, matrix_output
implicit none
integer, intent(in) :: nvareff, nbas, npoints
real(rk), dimension(nbas), intent(in) :: coefficients
real(rk), dimension(nvareff,npoints,nbas), intent(in) :: z_vectors
real(rk), dimension(nvareff,npoints), intent(in) :: grad_qn
real(rk) :: X2

integer :: i
real(rk), dimension(nvareff,npoints) :: grad_qn_modeled
real(rk), dimension(nvareff,npoints) :: diffsq

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nbas, &
   &     size(coefficients), &
   &     size(z_vectors,dim=3) ], &
   &  'dim. mismatch, chi_square_fit')
call checkeq( [nvareff, &
   &     size(grad_qn,dim=1), &
   &     size(z_vectors,dim=1) ], &
   &  'dim. mismatch, chi_square_fit')
call checkeq( [npoints, &
   &     size(grad_qn,dim=2) ], &
   &  'dim. mismatch, chi_square_fit')
write(stdout,*) 'chi_square_fit: array shape check successful.'
#endif

grad_qn_modeled = 0._rk

do i=1,nbas
  grad_qn_modeled(:,:)=grad_qn_modeled(:,:)+coefficients(i)*z_vectors(:,:,i)
enddo

diffsq(:,:) = grad_qn_modeled(:,:) - grad_qn(:,:)
diffsq(:,:) = diffsq(:,:)*diffsq(:,:)

X2 = sum(diffsq)

!call matrix_output(grad_qn,6,'F20.12','grad_qn')
!call matrix_output(grad_qn_modeled,6,'F20.12','grad_qn_modeled')
!call matrix_output(grad_qn_modeled-grad_qn,6,'F20.12','difference, modeled - actual')
!read(*,*)

return
end function chi_square_fit

! ****************************************
! ****************************************
! ****************************************

! Get 3rd/4th derivatives from finite difference of previously calculated
! table of displaced gradients. Dual step means that two different step sizes 
! deltaQ are used (fixed factor of 2 between those). It means twice the computational
! effort, but due to the redundancy, numerical stability is greatly enhanced, 
! and sensitivity of VPT2 parameters on deltaQ is clearly reduced.
! This routine is similar in spirit to the derivatives_from_gradients_extrapolate 
! routine, but is mathematically more rationalized, because it does not 
! rely on any assumptions for the convergence behaviour of the force constants, 
! but is based solely on truncated Taylor expansions of the PES.
! This is now the default for vpt2_grad_only=.true.
! This routine is meant to be combined with calculate_displaced_gradients_mpi.

subroutine derivatives_from_gradient_list_dual_step(nvar,nvareff,ngrad,mv3, &
                  & grad_cart,normal_modes,normal_modes_non_mw,eigenvalues,freq, &
                  & hess_cart,displacement_map_grad,cubic_fc,quartic_fc,deltaQ_uniform, &
                  & refine_freq,hessian_nc_out,grad_nc_ref_out)
use dlf_vpt2_utility, only: error_print, matrix_output, vector_output, vector_out_multi
use dlf_linalg_interface_mod
use dlf_vpt2_freq, only: hessian_eigenvalues
use dlf_constants
use dlf_sort_module, only: dlf_sort_shell_ind
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), dimension(nvar), intent(in) :: mv3
real(rk), dimension(nvar,ngrad), intent(in) :: grad_cart
real(rk), dimension(nvar,nvareff), intent(inout) :: normal_modes,normal_modes_non_mw
real(rk), dimension(nvareff), intent(inout) :: freq,eigenvalues
real(rk), dimension(nvar,nvar), intent(inout) :: hess_cart
integer, dimension(nvareff,nvareff,-2:2,-2:2), intent(in) :: displacement_map_grad
real(rk), intent(in) :: deltaQ_uniform
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
logical, intent(in) :: refine_freq
real(rk), dimension(nvareff,nvareff), intent(out), optional :: hessian_nc_out
real(rk), dimension(nvareff),intent(out), optional :: grad_nc_ref_out

logical, parameter :: rotate=.false.
integer, dimension(3) :: ot
integer  :: inu,jnu,knu,io,ip,m,i,j,k
real(rk), dimension(nvar) :: A, B, C, D, E, F, G, H
real(rk), dimension(nvar) :: AA,BB,CC,DD,EE,FF,GG,HH
real(rk), dimension(nvareff) :: top,too,tpp,uooo,uoop,uopp,Ho,Hp,g0
real(rk), dimension(nvar) :: t_op,t_oo,t_pp,u_ooo,u_oop,u_opp,H_o,H_p,g_0
real(rk) :: dQ, dQ2, dQ3, maxabs, au2cmi, amu2au
integer, dimension(nvareff,nvareff,nvareff) :: ncubic
integer, dimension(nvareff,nvareff,nvareff) :: nquartic
integer, dimension(nvareff,nvareff)  :: nhess
integer :: ngrad_count, maxindex
real(rk), dimension(nvareff,nvareff) :: hessian_nc,eigvec
real(rk), dimension(nvareff) :: eigval, vector_of_ones, freq_corrected
real(rk), dimension(nvareff) :: grad_nc_ref
real(rk), dimension(nvareff,nvareff,nvareff) :: cubic_fc_rotated, quartic_fc_rotated
real(rk), dimension(nvareff,nvareff,nvareff,nvareff) :: quartic_tmp1, quartic_tmp2
integer, dimension(nvareff) :: sortind
real(rk), dimension(nvareff,nvareff) :: nm1
real(rk), dimension(nvar,nvareff) :: nm_corr, normal_modes_nonmw_old, normal_modes_old
real(rk), dimension(nvar,nvar) :: sqrtmass
real(rk), dimension(:,:), allocatable :: dummy_arr
integer :: size_disp
character(len=500) :: fn_punch

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, size(grad_cart,dim=1),&
              &  size(mv3), &
              &  size(hess_cart,dim=1), &
              &  size(hess_cart,dim=2), &
              &  size(normal_modes,dim=1), &
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_dual_step')
call checkeq( [nvareff, &
   &     size(freq), &
   &     size(eigenvalues), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(displacement_map_grad,dim=1), size(displacement_map_grad,dim=2), &
   &     size(normal_modes,dim=2) , &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_dual_step')
call checkeq( [ngrad, size(grad_cart,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list_dual_step')
call checkeq( [5, size(displacement_map_grad,dim=3), &
   &              size(displacement_map_grad,dim=4) ],  &
   &  'dim. mismatch, derivatives_from_gradient_list_dual_step')
write(stdout,*) 'derivatives_from_gradient_list_dual_step: array shape check successful.'
#endif

hessian_nc(:,:)=0._rk
grad_nc_ref(:)=0._rk
cubic_fc(:,:,:)=0._rk
quartic_fc(:,:,:)=0._rk
nhess(:,:)=0
ngrad_count=0
ncubic(:,:,:)=0
nquartic(:,:,:)=0
dQ=deltaQ_uniform
dQ2=dQ*dQ
dQ3=dQ2*dQ

!!call vector_output(mv3,6,'F20.12','mv3')
!!call matrix_output(normal_modes,6,'F20.12','normal_modes')
!!call matrix_output(normal_modes_non_mw,6,'F20.12','normal_modes_non_mw')
!!call vector_output(eigenvalues,6,'F20.12','eigenvalues')
!!call vector_output(freq,6,'F20.12','freq')
!!call matrix_output(hess_cart,6,'F20.12','hess_cart')
!!call matrix_output(grad_cart,6,'F20.12','grad_cart')
!!
!!write(*,*) 'nvar:    ', nvar
!!write(*,*) 'nvareff: ', nvareff
!!write(*,*) 'ngrad:   ', ngrad
!!write(*,*) 'dQ:      ', deltaQ_uniform
!!
!!size_disp=size(displacement_map_grad,dim=1)*size(displacement_map_grad,dim=2)*size(displacement_map_grad,dim=3) &
!!                   & *size(displacement_map_grad,dim=4)
!!!allocate(dummy_arr(size_disp,1))
!!!dummy_arr(:,:)=0._rk
!!!call vector_out_multi(dummy_arr,reshape(pack(displacement_map_grad,.true.),[size_disp,1]),6,'X','displ_map')
!!!deallocate(dummy_arr)
!!do io=1,nvareff
!!  do ip=1,nvareff
!!    write(*,'(25(I0,1X))') ((displacement_map_grad(io,ip,m,k), k=-2,2),m=-2,2)
!!  enddo
!!enddo
!!
!!read(*,*)

! Iterate over normal mode combinations
do io=1,nvareff
    E(:) =grad_cart(:,displacement_map_grad(io,io, 1,0))
    F(:) =grad_cart(:,displacement_map_grad(io,io,-1,0))
    EE(:)=grad_cart(:,displacement_map_grad(io,io, 2,0))
    FF(:)=grad_cart(:,displacement_map_grad(io,io,-2,0))
    u_ooo(:)= ( -2*E  +2*F  +1*EE  -1*FF )/(dQ3*2)
    H_o(:)  = ( +8*E  -8*F  -1*EE  +1*FF )/(dQ*12)
    uooo(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_ooo)
    Ho(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),H_o)
  do m=1,nvareff
    !uooo
    ot=ot4(io,io,m)
    inu=ot(1)
    jnu=ot(2)
    knu=ot(3)
    quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uooo(m)
    nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
    !Ho
    inu=max(io,m)
    jnu=min(io,m)
    hessian_nc(inu,jnu)=hessian_nc(inu,jnu)+Ho(m)
    nhess(inu,jnu)=nhess(inu,jnu)+1
  enddo
  do ip=1,io-1
    A(:)=grad_cart(:,displacement_map_grad(io,ip,+1,+1))
    B(:)=grad_cart(:,displacement_map_grad(io,ip,+1,-1))
    C(:)=grad_cart(:,displacement_map_grad(io,ip,-1,+1))
    D(:)=grad_cart(:,displacement_map_grad(io,ip,-1,-1))
    E(:)=grad_cart(:,displacement_map_grad(io,io, 1,0))
    F(:)=grad_cart(:,displacement_map_grad(io,io,-1,0))
    G(:)=grad_cart(:,displacement_map_grad(ip,ip, 1,0))
    H(:)=grad_cart(:,displacement_map_grad(ip,ip,-1,0))
    AA(:)=grad_cart(:,displacement_map_grad(io,ip,+2,+2))
    BB(:)=grad_cart(:,displacement_map_grad(io,ip,+2,-2))
    CC(:)=grad_cart(:,displacement_map_grad(io,ip,-2,+2))
    DD(:)=grad_cart(:,displacement_map_grad(io,ip,-2,-2))
    EE(:)=grad_cart(:,displacement_map_grad(io,io, 2,0))
    FF(:)=grad_cart(:,displacement_map_grad(io,io,-2,0))
    GG(:)=grad_cart(:,displacement_map_grad(ip,ip, 2,0))
    HH(:)=grad_cart(:,displacement_map_grad(ip,ip,-2,0))
    t_op(:) =( +16*A  -16*B  -16*C  +16*D &
          &     +0*E   +0*F   +0*G   +0*H &
          &    -1*AA  +1*BB  +1*CC  -1*DD &
          &    +0*EE  +0*FF  +0*GG  +0*HH )/(dQ2*48)
    t_oo(:) =( +16*A  +16*B  +16*C  +16*D &
          &     +0*E   +0*F  -32*G  -32*H &
          &    -1*AA  -1*BB  -1*CC  -1*DD &
          &    +0*EE  +0*FF  +2*GG  +2*HH )/(dQ2*24)
    t_pp(:) =( +16*A  +16*B  +16*C  +16*D &
          &    -32*E  -32*F   +0*G   +0*H &
          &    -1*AA  -1*BB  -1*CC  -1*DD &
          &    +2*EE  +2*FF  +0*GG  +0*HH )/(dQ2*24)
    u_oop(:)=( +32*A  -32*B  +32*C  -32*D &
          &     +0*E   +0*F  -64*G  +64*H &
          &    -1*AA  +1*BB  -1*CC  +1*DD &
          &    +0*EE  +0*FF  +2*GG  -2*HH )/(dQ3*48)
    u_opp(:)=( +32*A  +32*B  -32*C  -32*D &
          &    -64*E  +64*F   +0*G   +0*H &
          &    -1*AA  -1*BB  +1*CC  +1*DD &
          &    +2*EE  -2*FF  +0*GG  +0*HH )/(dQ3*48)
    g_0(:)  =( -16*A  -16*B  -16*C  -16*D &
          &    +32*E  +32*F  +32*G  +32*H &
          &    +1*AA  +1*BB  +1*CC  +1*DD &
          &    -2*EE  -2*FF  -2*GG  -2*HH )/real(60)
    top(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_op)
    too(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_oo)
    tpp(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_pp)
    uoop(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_oop)
    uopp(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_opp)
    g0(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),g_0)
    do m=1,nvareff
      !topm
      ot=ot3(io,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+top(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !toom
      ot=ot3(io,io,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+too(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !tppm
      ot=ot3(ip,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+tpp(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !uoopm
      ot=ot4(io,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uoop(m)
      nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
      !uoppm
      ot=ot4(ip,io,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uopp(m)
      nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
      !g0
      ngrad_count=ngrad_count+1
      grad_nc_ref(m)=grad_nc_ref(m)+g0(m)
    enddo
  enddo
enddo

nhess(:,:)=max(nhess(:,:),1)
ncubic(:,:,:)=max(ncubic(:,:,:),1)
nquartic(:,:,:)=max(nquartic(:,:,:),1)

grad_nc_ref(:)=grad_nc_ref(:)/real(ngrad_count)
hessian_nc(:,:)  =hessian_nc(:,:)/real(nhess(:,:))
cubic_fc(:,:,:)  =cubic_fc(:,:,:)/real(ncubic(:,:,:))
quartic_fc(:,:,:)=quartic_fc(:,:,:)/real(nquartic(:,:,:))

call symmetric_fill_hessian(nvareff,hessian_nc,hessian_nc)

open(1115,file='dbg.out',action='write')
call vector_output(grad_nc_ref,1115,'ES20.12','gradient at reference geometry (in normal coordinates)')
call matrix_output(hessian_nc,1115,'ES20.12','Hessian at reference geometry (in normal coordinates)')
call symmetric_fill_cubic(nvareff,cubic_fc,cubic_fc)
call symmetric_fill_quartic_semidiag(nvareff,quartic_fc,quartic_fc)

call hessian_eigenvalues(nvareff,hessian_nc,eigval,eigvec)
do m=1,nvareff
  maxindex=maxloc(abs(eigvec(:,m)),dim=1)
  maxabs  =eigvec(maxindex,m)
  if (maxabs < 0) eigvec(:,m)=-eigvec(:,m)
enddo
call matrix_output(eigvec,1115,'ES20.12','Hessian eigenvectors')
call vector_output(eigval,1115,'ES20.12','Hessian eigenvalues')

if (refine_freq) then
  normal_modes_nonmw_old=normal_modes_non_mw
  normal_modes_old=normal_modes
  call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)
  call dlf_constants_get('AMU',amu2au)
  call dlf_sort_shell_ind(eigval,sortind)
  do i=1,nvareff
    j=sortind(i)
    eigenvalues(i)=eigval(j)
    freq_corrected(i)=sign(1._rk,eigenvalues(i))*sqrt(abs(eigenvalues(i)))*au2cmi
    nm1(:,i)=eigvec(:,j)
  enddo
  call vector_output(freq_corrected,1115,'F20.3','Frequencies')
  freq(:)=freq_corrected(:)
  hess_cart=dlf_matrix_ortho_trans(normal_modes,hessian_nc,1)
  sqrtmass(:,:)=0._rk
  do m=1,nvar
    sqrtmass(m,m)=sqrt(mv3(m))
  enddo
  hess_cart=dlf_matrix_ortho_trans(sqrtmass,hess_cart,0)
  nm_corr=dlf_matmul_simp(normal_modes,nm1)
  normal_modes(:,:)=nm_corr(:,:)
  nm_corr=dlf_matmul_simp(normal_modes_non_mw,nm1)
  normal_modes_non_mw(:,:)=nm_corr(:,:)
  fn_punch=''
  write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'
  open(3198, file=trim(adjustl(fn_punch)), status="old", position="append", action="write")
  write(3198,'(A)') '$NORMAL_MODES_REFINED'
  call matrix_output(normal_modes,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$FREQS_REFINED'
  call vector_output(freq,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$EIGENVALUES_REFINED'
  call vector_output(eigenvalues,3198,'ES24.16','__BLANK__')
  write(3198,'(A)') '$HESSIAN_CART_REFINED'
  write(3198,'(I0,1X,I0,1X,ES15.6)') 0, 0, 0._rk
  call matrix_output(hess_cart,3198,'ES24.16','__BLANK__')
  close(3198)
  if (rotate) then
    call fd_calc_rotate(nvar,nvareff,ngrad,mv3,normal_modes_nonmw_old, &
                        & normal_modes_old,normal_modes_non_mw,normal_modes, & 
                        & eigenvalues,freq,hess_cart, &
                        & hessian_nc,cubic_fc,quartic_fc,cubic_fc_rotated, &
                        & quartic_fc_rotated)
    cubic_fc(:,:,:)  =cubic_fc_rotated(:,:,:)
    quartic_fc(:,:,:)=quartic_fc_rotated(:,:,:)
  endif
endif

close(1115)

if (present(hessian_nc_out)) then
  hessian_nc_out(:,:)=hessian_nc(:,:)
endif

if (present(grad_nc_ref_out)) then
  grad_nc_ref_out(:)=grad_nc_ref(:)
endif

!if (rotate) then
!
!  vector_of_ones(:)=1._rk
!  call convert_cubic_cart_to_normal_coord(nvareff,0,eigvec,vector_of_ones,cubic_fc,cubic_fc_rotated)
!  cubic_fc(:,:,:)=cubic_fc_rotated(:,:,:)
!  
!  do i=1,nvareff
!    do j=1,nvareff
!      do k=1,nvareff
!        do m=1,nvareff
!          if (i/=j .and. i/=k .and. i/=m .and. &
!              & j/=k .and. j/=m .and. & 
!              & k/=m ) then
!            quartic_tmp1(i,j,k,m)=0._rk
!          else
!            ot=ot4gen(i,j,k,m)
!            inu=ot(1)
!            jnu=ot(2)
!            knu=ot(3)
!            quartic_tmp1(i,j,k,m)=quartic_fc(inu,jnu,knu)
!          endif
!        enddo
!      enddo
!    enddo
!  enddo
!  
!  call convert_quartic_cart_to_normal_coord(nvareff,0,eigvec,vector_of_ones,quartic_tmp1,quartic_tmp2)
!  
!  do m=1,nvareff
!    quartic_fc(m,:,:)=quartic_tmp2(m,m,:,:)
!  enddo
!
!endif

!if (rotate) then
!  call symmetric_fill_cubic(nvareff,cubic_fc,cubic_fc)
!  call symmetric_fill_quartic_semidiag(nvareff,quartic_fc,quartic_fc)
!  vector_of_ones(:)=1._rk
!  call convert_cubic_cart_to_normal_coord(nvareff,0,eigvec,vector_of_ones,cubic_fc,cubic_fc_rotated)
!  cubic_fc(:,:,:)=cubic_fc_rotated(:,:,:)
!  call rotate_quartic_force_constants_semidiag(nvareff,eigvec,quartic_fc,quartic_fc_rotated)
!  quartic_fc(:,:,:)=quartic_fc_rotated(:,:,:)
!endif

return
end subroutine derivatives_from_gradient_list_dual_step

! ****************************************
! ****************************************
! ****************************************

! Rotate cubic/quartic force constants from old to new set of normal coordinates
! The initial idea was to improve 3rd/4th derivatives by correcting for 
! deficiencies in the initial set of normal coordinates used for the finite-difference
! Hessian. However, rotation of the force constant to the improved set of normal 
! coordinates does not really seem to improve the VPT2 results.

subroutine fd_calc_rotate(nvar,nvareff,ngrad,mv3,normal_modes_nonmw_old, &
                        & normal_modes_old,normal_modes_nonmw_new,normal_modes_new, & 
                        & eigenvalues,freq,hess_cart, &
                        & hessian_nm_old,cubic_old,quartic_old,cubic_new, &
                        & quartic_new)
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer, intent(in)  :: nvar,nvareff,ngrad
  real(rk), intent(in), dimension(nvar) :: mv3
  real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes_nonmw_old
  real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes_old
  real(rk), intent(inout), dimension(nvar,nvareff) :: normal_modes_nonmw_new
  real(rk), intent(inout), dimension(nvar,nvareff) :: normal_modes_new
  real(rk), intent(in), dimension(nvareff) :: eigenvalues, freq
  real(rk), intent(in), dimension(nvar,nvar) :: hess_cart
  real(rk), intent(in), dimension(nvareff,nvareff) :: hessian_nm_old
  real(rk), intent(in), dimension(nvareff,nvareff,nvareff) :: cubic_old, quartic_old
  real(rk), intent(out), dimension(nvareff,nvareff,nvareff) :: cubic_new, quartic_new
  
  integer :: i,j,k,io,ip,so,sp,ngrad_single
  real(rk), parameter :: dQloc=10._rk
  integer, dimension(nvareff,nvareff,-2:2,-2:2) :: displacement_map_grad_loc
  integer, dimension(4*nvareff**2,4) :: joblist_grad_loc
  real(rk), dimension(nvar) :: coords_dummy
  real(rk), dimension(nvar,ngrad) :: grad_cart_model
  logical, dimension(ngrad) :: graddone_loc
  real(rk), dimension(nvareff) :: eigenvalues_loc, freq_loc
  real(rk), dimension(nvar,nvar) :: hess_cart_loc
  real(rk), dimension(nvareff) :: zervec
  
  grad_cart_model(:,:)=0._rk
  eigenvalues_loc(:)=eigenvalues(:)
  freq_loc(:)=freq(:)
  hess_cart_loc(:,:)=hess_cart(:,:)
  
  ngrad_single=ngrad/2
  i=0
  do j=1,nvareff
    i=i+1
    displacement_map_grad_loc(j,j,1,0)=i
    joblist_grad_loc(i,1)=j
    joblist_grad_loc(i,2)=j
    joblist_grad_loc(i,3)=1
    joblist_grad_loc(i,4)=0
    i=i+1
    displacement_map_grad_loc(j,j,-1,0)=i
    joblist_grad_loc(i,1)=j
    joblist_grad_loc(i,2)=j
    joblist_grad_loc(i,3)=-1
    joblist_grad_loc(i,4)=0
  enddo
  do j=1,nvareff
    do k=1,j-1
      i=i+1
      displacement_map_grad_loc(j,k,1,1)=i
      joblist_grad_loc(i,1)=j
      joblist_grad_loc(i,2)=k
      joblist_grad_loc(i,3)=1
      joblist_grad_loc(i,4)=1
      i=i+1
      displacement_map_grad_loc(j,k,1,-1)=i
      joblist_grad_loc(i,1)=j
      joblist_grad_loc(i,2)=k
      joblist_grad_loc(i,3)=1
      joblist_grad_loc(i,4)=-1
      i=i+1
      displacement_map_grad_loc(j,k,-1,1)=i
      joblist_grad_loc(i,1)=j
      joblist_grad_loc(i,2)=k
      joblist_grad_loc(i,3)=-1
      joblist_grad_loc(i,4)=1
      i=i+1
      displacement_map_grad_loc(j,k,-1,-1)=i
      joblist_grad_loc(i,1)=j
      joblist_grad_loc(i,2)=k
      joblist_grad_loc(i,3)=-1
      joblist_grad_loc(i,4)=-1
    enddo
  enddo
  joblist_grad_loc(ngrad_single+1:ngrad,:)  =joblist_grad_loc(1:ngrad_single,:)
  joblist_grad_loc(ngrad_single+1:ngrad,3:4)=2*joblist_grad_loc(ngrad_single+1:ngrad,3:4)
  do j=ngrad_single+1,ngrad
    io=joblist_grad_loc(j,1)
    ip=joblist_grad_loc(j,2)
    so=joblist_grad_loc(j,3)
    sp=joblist_grad_loc(j,4)
    displacement_map_grad_loc(io,ip,so,sp)=j
  enddo
  
  graddone_loc(:)=.false.
  
  nbas_save=nvareff*(nvareff+1)*(nvareff+2)
  nbas_save=2*nbas_save/3
  nbas_save=nbas_save -nvareff*(nvareff-1)
  
  if (allocated(sqrtmass_save))           call deallocate(sqrtmass_save)
  if (allocated(normal_modes_save))       call deallocate(normal_modes_save)
  if (allocated(normal_modes_nonmw_save)) call deallocate(normal_modes_nonmw_save)
  !if (allocated(hessian_save))            call deallocate(hessian_save)
  !if (allocated(cubic_save))              call deallocate(cubic_save)
  !if (allocated(quartic_sd_save))         call deallocate(quartic_sd_save)
  if (allocated(coefficients_save))       call deallocate(coefficients_save)
  
  call allocate(sqrtmass_save,nvar,nvar)
  call allocate(normal_modes_save,nvar,nvareff)
  call allocate(normal_modes_nonmw_save,nvar,nvareff)
  !call allocate(hessian_save,nvareff,nvareff)
  !call allocate(cubic_save,nvareff,nvareff,nvareff)
  !call allocate(quartic_sd_save,nvareff,nvareff,nvareff)
  call allocate(coefficients_save,nbas_save)
  
  zervec(:)=0._rk
  call force_constants_to_fit_coefficients(nvareff,nbas_save,zervec,hessian_nm_old,cubic_old,quartic_old,coefficients_save)
  
  normal_modes_save(:,:)      =normal_modes_old(:,:)
  normal_modes_nonmw_save(:,:)=normal_modes_nonmw_old(:,:)
  !hessian_save(:,:)           =hessian_nm_old(:,:)
  !cubic_save(:,:,:)           =cubic_old(:,:,:)
  !quartic_sd_save(:,:,:)      =quartic_old(:,:,:)
  
  sqrtmass_save(:,:)=0._rk
  do i=1,nvar
    sqrtmass_save(i,i)=sqrt(mv3(i))
  enddo
  
  coords_dummy(:)=0._rk
  
  call calculate_displaced_gradients_mpi(get_gradient_from_force_constants,nvar,nvareff,ngrad, &
               & coords_dummy,normal_modes_new,mv3, &
               & joblist_grad_loc,graddone_loc,grad_cart_model,dQloc)
  
  call derivatives_from_gradient_list_dual_step(nvar,nvareff,ngrad,mv3, &
                  & grad_cart_model,normal_modes_new,normal_modes_nonmw_new,eigenvalues_loc,freq_loc, &
                  & hess_cart_loc,displacement_map_grad_loc,cubic_new,quartic_new,dQloc, &
                  & .false.)
                  
  call deallocate(sqrtmass_save)
  call deallocate(normal_modes_save)
  call deallocate(normal_modes_nonmw_save)
  !call deallocate(hessian_save)
  !call deallocate(cubic_save)
  !call deallocate(quartic_sd_save)
  call deallocate(coefficients_save)
  
  return
end subroutine fd_calc_rotate

#ifdef VPT2_GRAD_DEBUG_OUT
subroutine get_gradient_from_force_constants & 
      & (nvar,coords,energy,gradient,iimage,kiter,status,calctag)
#else
subroutine get_gradient_from_force_constants & 
      & (nvar,coords,energy,gradient,iimage,kiter,status)
#endif
  use dlf_vpt2_utility, only: error_print, vector_output, matrix_output
  use dlf_allocate, only: allocate, deallocate
  implicit none
  integer   ,intent(in)    :: nvar
  real(rk)  ,intent(in)    :: coords(nvar)
  real(rk)  ,intent(out)   :: energy
  real(rk)  ,intent(out)   :: gradient(nvar)
  integer   ,intent(in)    :: iimage
  integer   ,intent(in)    :: kiter
  integer   ,intent(out)   :: status
#ifdef VPT2_GRAD_DEBUG_OUT
  character(len=*), intent(in),optional :: calctag
#endif

  integer, parameter :: nzero=6
  integer :: i,j,k,l,inu,jnu,knu,nvareff
  real(rk), dimension(nvar-nzero)   :: coords_nm, grad_nm
  real(rk), dimension(nvar,nvar-nzero) :: trafo
  real(rk), dimension(:,:,:), allocatable :: z_vectors
  
  energy=0._rk
  status=0
  nvareff=nvar-nzero
  
  if ( &
    & (allocated(sqrtmass_save) .and. &
     & allocated(normal_modes_save) .and. &
     & allocated(normal_modes_nonmw_save) .and. &
     & allocated(hessian_save) .and. &
     & allocated(cubic_save) .and. &
     & allocated(quartic_sd_save)) .or. &
    & (allocated(coefficients_save) .and. &
     & allocated(sqrtmass_save) .and. &
     & allocated(normal_modes_save) .and. &
     & allocated(normal_modes_nonmw_save))  &
     & ) then
    continue
  else
    call error_print('get_gradient_from_force_constants: necessary arrays not allocated.')
  endif
  
  trafo=matmul(sqrtmass_save,normal_modes_save)
  ! Cartesian displacements to normal modes
  
  coords_nm=matmul(transpose(trafo),coords)
  !call vector_output(coords_nm,6,'ES13.5','coords_nm')
  
  grad_nm(:)=0._rk
  
  call allocate(z_vectors,nvareff,1,nbas_save)
  call get_z_vectors(nvareff,1,nbas_save,reshape(coords_nm,[size(coords_nm),1]),z_vectors)
  !call vector_output(coefficients_save,6,'ES13.5','coefficients')
  !call matrix_output(z_vectors(:,1,:),6,'ES13.5','zvec')
  do i=1,nbas_save
    grad_nm(:)=grad_nm(:)+coefficients_save(i)*z_vectors(:,1,i)
  enddo
  call deallocate(z_vectors)
  
  !! 2nd order contributions
  !
  !grad_nm(:)=grad_nm(:)+matmul(hessian_save,coords_nm)
  !
  !! 3rd order contributions (requires symmetrically filled cubic fc's)
  !
  !do i=1,nvareff
  !  grad_nm(i)=grad_nm(i)+0.5_rk*dot_product(coords_nm(:),matmul(cubic_save(:,:,i),coords_nm(:)))
  !enddo
  !
  !! 4th order contributions (requires symmetrically filled quartic fc's)
  !do i=1,nvareff
  !  grad_nm(i)=grad_nm(i)+(coords_nm(i)**3*quartic_sd_save(i,i,i))/6._rk
  !  do j=1,nvareff
  !    if (j==i) cycle
  !    grad_nm(i)=grad_nm(i)+(coords_nm(i)**2*coords_nm(j)*quartic_sd_save(i,i,j))/2._rk
  !  enddo
  !  do j=1,nvareff
  !    if (j==i) cycle
  !    do k=1,nvareff
  !      if (k==i) cycle
  !      grad_nm(i)=grad_nm(i)+(coords_nm(i)*coords_nm(j)*coords_nm(k)*quartic_sd_save(i,j,k))/2._rk
  !    enddo
  !  enddo
  !  do j=1,nvareff
  !    if (j==i) cycle
  !    do k=1,nvareff
  !      if (k==i) cycle
  !      do l=1,nvareff
  !        if (l==i) cycle
  !        if (j==k) then
  !          inu=j
  !          jnu=l
  !          knu=i
  !        elseif (j==l) then
  !          inu=j
  !          jnu=k
  !          knu=i
  !        elseif (k==l) then
  !          inu=k
  !          jnu=j
  !          knu=i
  !        else
  !          cycle
  !        endif
  !        grad_nm(i)=grad_nm(i)+(coords_nm(j)*coords_nm(k)*coords_nm(l)*quartic_sd_save(inu,jnu,knu))/6._rk
  !      enddo
  !    enddo
  !  enddo
  !enddo
  
  ! Convert to Cartesian gradient
  
  !call vector_output(grad_nm,6,'ES20.12','gradient, normald modes')
  gradient=matmul(trafo,grad_nm)
  !call vector_output(gradient,6,'ES20.12','gradient, Cartesian')
  !read(*,*)
  
  return
end subroutine get_gradient_from_force_constants

! ****************************************
! ****************************************
! ****************************************

! Get 3rd/4th derivatives from finite difference of previously calculated
! table of displaced gradients. A single step size deltaQ is used, in contrast 
! to the *_dual_step version. This means half the computational burden, but also
! less redundancy in the data and therefore rather large errors in the computed 
! force constants.
! This routine is meant to be combined with calculate_displaced_gradients_mpi.

subroutine derivatives_from_gradient_list(nvar,nvareff,ngrad, &
                  & grad_cart,hess_eval,normal_modes_non_mw, &
                  & displacement_map_grad,cubic_fc,quartic_fc,deltaQ_uniform, &
                  & grad_nc_ref_out)
use dlf_vpt2_utility, only: error_print, vector_output
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,ngrad
real(rk), dimension(nvar,ngrad), intent(in) :: grad_cart
real(rk), dimension(nvareff), intent(in) :: hess_eval
real(rk), dimension(nvar,nvareff), intent(in) :: normal_modes_non_mw
integer, dimension(nvareff,nvareff,-1:1,-1:1), intent(in) :: displacement_map_grad
real(rk), intent(in) :: deltaQ_uniform
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc
real(rk), intent(out), dimension(nvareff), optional :: grad_nc_ref_out

integer, dimension(3) :: ot
integer  :: inu,jnu,knu,io,ip,m,ngrad_count
real(rk), dimension(nvar) :: A,B,C,D,E,F,G,H
real(rk), dimension(nvareff) :: top,too,tpp,uooo,uoop,uopp
real(rk), dimension(nvar) :: t_op,t_oo,t_pp,u_ooo,u_oop,u_opp
real(rk), dimension(nvareff) :: grad_nc_ref, g0
real(rk), dimension(nvar) :: g_0
real(rk) :: dQ, dQ2, dQ3
integer, dimension(nvareff,nvareff,nvareff) :: ncubic
integer, dimension(nvareff,nvareff,nvareff) :: nquartic

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, size(grad_cart,dim=1),&
              &  size(normal_modes_non_mw,dim=1) ], &
   &  'dim. mismatch, derivatives_from_gradient_list')
call checkeq( [nvareff, &
   &     size(hess_eval), &
   &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
   &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
   &     size(quartic_fc,dim=2), size(quartic_fc,dim=3),&
   &     size(displacement_map_grad,dim=1), size(displacement_map_grad,dim=2), &
   &     size(normal_modes_non_mw,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list')
call checkeq( [ngrad, size(grad_cart,dim=2) ], &
   &  'dim. mismatch, derivatives_from_gradient_list')
call checkeq( [3, size(displacement_map_grad,dim=3), &
   &              size(displacement_map_grad,dim=4) ],  &
   &  'dim. mismatch, derivatives_from_gradient_list')
write(stdout,*) 'derivatives_from_gradient_list: array shape check successful.'
#endif

grad_nc_ref(:)=0._rk
cubic_fc(:,:,:)=0._rk
quartic_fc(:,:,:)=0._rk
ncubic(:,:,:)=0
nquartic(:,:,:)=0
ngrad_count=0
dQ=deltaQ_uniform
dQ2=dQ*dQ
dQ3=dQ2*dQ

! Iterate over normal mode combinations
do io=1,nvareff
  E(:)=grad_cart(:,displacement_map_grad(io,io, 1,0))
  F(:)=grad_cart(:,displacement_map_grad(io,io,-1,0))
  !!write(*,*) 'io  = ', io
  !!write(*,*) 'dQ  = ', dQ
  !!write(*,*) 'dQ³ = ', dQ3
  !!call vector_output(E(:),6,'ES18.10','grad_cart(+1)')
  !!call vector_output(F(:),6,'ES18.10','grad_cart(-1)')
  !!call vector_output(dlf_matmul_simp(transpose(normal_modes_non_mw),E),6,'ES18.10','grad_qn(+1)')
  !!call vector_output(dlf_matmul_simp(transpose(normal_modes_non_mw),F),6,'ES18.10','grad_qn(-1)')
  !!write(*,*) 'H_ii', hess_eval(io)
  !!read(*,*)
  u_ooo(:)=(E(:)-F(:))/real(2)
  uooo(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_ooo)
  uooo(io)=uooo(io) - hess_eval(io)*dQ
  uooo(:)=real(6)/dQ3*uooo(:)
  do m=1,nvareff
    ot=ot4(io,io,m)
    inu=ot(1)
    jnu=ot(2)
    knu=ot(3)
    quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uooo(m)
    nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
  enddo
  do ip=1,io-1
    A(:)=grad_cart(:,displacement_map_grad(io,ip,+1,+1))
    B(:)=grad_cart(:,displacement_map_grad(io,ip,+1,-1))
    C(:)=grad_cart(:,displacement_map_grad(io,ip,-1,+1))
    D(:)=grad_cart(:,displacement_map_grad(io,ip,-1,-1))
    E(:)=grad_cart(:,displacement_map_grad(io,io, 1,0))
    F(:)=grad_cart(:,displacement_map_grad(io,io,-1,0))
    G(:)=grad_cart(:,displacement_map_grad(ip,ip, 1,0))
    H(:)=grad_cart(:,displacement_map_grad(ip,ip,-1,0))
    t_op(:)= (A(:)-B(:)-C(:)+D(:))/(real(4)*dQ2)
    t_oo(:)= (A(:)+B(:)+C(:)+D(:)-2*G(:)-2*H(:))/(real(2)*dQ2)
    t_pp(:)= (A(:)+B(:)+C(:)+D(:)-2*E(:)-2*F(:))/(real(2)*dQ2)
    u_oop(:)=(0.5_rk*(A(:)-B(:)+C(:)-D(:)) + H(:)-G(:))/dQ3
    u_opp(:)=(0.5_rk*(A(:)+B(:)-C(:)-D(:)) + F(:)-E(:))/dQ3
    g_0(:)=-0.25_rk*(A(:)+B(:)+C(:)+D(:))+0.5_rk*(E(:)+F(:)+G(:)+H(:))
    top(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_op)
    too(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_oo)
    tpp(:) =dlf_matmul_simp(transpose(normal_modes_non_mw),t_pp)
    uoop(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_oop)
    uopp(:)=dlf_matmul_simp(transpose(normal_modes_non_mw),u_opp)
    g0(:)  =dlf_matmul_simp(transpose(normal_modes_non_mw),g_0)
    do m=1,nvareff
      !topm
      ot=ot3(io,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+top(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !toom
      ot=ot3(io,io,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+too(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !tppm
      ot=ot3(ip,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+tpp(m)
      ncubic(inu,jnu,knu)=ncubic(inu,jnu,knu)+1
      !uoopm
      ot=ot4(io,ip,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uoop(m)
      nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
      !uoppm
      ot=ot4(ip,io,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+uopp(m)
      nquartic(inu,jnu,knu)=nquartic(inu,jnu,knu)+1
      !g0
      ngrad_count=ngrad_count+1
      grad_nc_ref(m)=grad_nc_ref(m)+g0(m)
    enddo
  enddo
enddo

ncubic(:,:,:)=max(ncubic(:,:,:),1)
nquartic(:,:,:)=max(nquartic(:,:,:),1)

cubic_fc(:,:,:)  =cubic_fc(:,:,:)/real(ncubic(:,:,:))
quartic_fc(:,:,:)=quartic_fc(:,:,:)/real(nquartic(:,:,:))
grad_nc_ref(:)=grad_nc_ref(:)/real(ngrad_count)

if (present(grad_nc_ref_out)) then
  grad_nc_ref_out(:)=grad_nc_ref(:)
endif

return
end subroutine derivatives_from_gradient_list

! ****************************************
! ****************************************
! ****************************************

! Calculate Hessians of displaced geometries, to be used for finite-difference
! 3rd/4th derivatives. MPI-parallelized routine, the overall job list is partitioned
! along all processes. This routine is meant to be combined with derivatives_from_hessian_list.

subroutine calculate_displaced_hessians_mpi(hess_routine,nvar,nvareff,nhess,coord0, &
               & normal_modes,mv3,displacement_map,hessians_done,hessians_cart,deltaQdiff_in,&
               & silent_in)
use dlf_vpt2_utility, only: error_print, dlf_global_real_bcast_rank2, &
                          & dlf_global_int_bcast, dlf_global_int_scatter_rank0, matrix_output
use dlf_linalg_interface_mod
use dlf_allocate
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_HESS_DEBUG_OUT
  subroutine hess_routine(nvar,coords,hessian,status,calctag)
#else
  subroutine hess_routine(nvar,coords,hessian,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: hessian(nvar,nvar)
    integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine hess_routine
end interface
integer, intent(in) :: nvar,nvareff,nhess
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes
integer, dimension(-nvareff:+nvareff), intent(in) :: displacement_map
logical, dimension(nhess), intent(inout) :: hessians_done
real(rk), dimension(nvar,nvar,nhess), intent(inout) :: hessians_cart
real(rk), intent(in),dimension(:) :: deltaQdiff_in
logical, intent(in),optional :: silent_in

real(rk), dimension(nvar,nvar) :: hess, invsqrtmass
real(rk), dimension(nvar,nvareff) :: normal_modes_non_mw
real(rk), dimension(nvar) :: coordtmp
real(rk), dimension(nvareff) :: deltaQdiff
real(rk) :: dQ
integer, dimension(nhess) :: joblist
integer, dimension(:),allocatable :: joblist_todo
integer  :: istat,k,m,i,hesscount,nproc
integer  :: kmin, kmax, ires, k_loop
integer  :: ndone, ntodo, ntodo_this_process
character(len=500) :: fn_punch
logical :: punch_exists
#ifdef VPT2_HESS_DEBUG_OUT
character(len=4) :: ktag
#endif

integer, allocatable :: kmin_vec(:), kmax_vec(:), jmin_vec(:), jmax_vec(:), proc_assignment(:)
logical  :: silent

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(coord0), size(mv3), size(normal_modes, dim=1) , &
     &     size(hessians_cart,dim=1), size(hessians_cart,dim=2)  ], &
     &  'dim. mismatch, calculate_displaced_hessians_mpi')
     call checkeq( [nvareff, size(normal_modes, dim=2) ], &
     &  'dim. mismatch, calculate_displaced_hessians_mpi')
  call checkeq( [nhess, size(hessians_cart,dim=3), size(hessians_done)  ], &
     &  'dim. mismatch, calculate_displaced_hessians_mpi')
  call checkeq( [2*nvareff+1, size(displacement_map)  ], &
     &  'dim. mismatch, calculate_displaced_hessians_mpi')
  write(stdout,*) 'calculate_displaced_hessians_mpi: array shape check successful.'
#endif

fn_punch=''
write(fn_punch,'(A,I6.6,A)') 'dlf_vpt2_restart_proc_',glob%iam,'.dat'

inquire(file=trim(adjustl(fn_punch)), exist=punch_exists)
if (.not. punch_exists) then
  call error_print('calculate_displaced_hessians_mpi: punch file missing!')
endif

open(3499, file=trim(adjustl(fn_punch)), status="old", position="append", action="write")

nproc=glob%nprocs
silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif

! Use uniform or individual elongations dQ
if (size(deltaQdiff_in)==1) then
  deltaQdiff(:)=deltaQdiff_in(1)
elseif (size(deltaQdiff_in)==nvareff) then
  deltaQdiff(:)=deltaQdiff_in(:)
else
  call error_print('calculate_displaced_hessians_mpi: wrong dimension for deltaQdiff_in')
endif

! Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

! Find out what has been already done, and what's left to do and distribute job ranges along processes

ndone=count(hessians_done)
ntodo=nhess-ndone

if (glob%iam==0) then
  allocate(kmin_vec(0:nproc-1))
  allocate(kmax_vec(0:nproc-1))
  allocate(jmin_vec(0:nproc-1))
  allocate(jmax_vec(0:nproc-1))
  allocate(joblist_todo(ntodo))
  do k=1,nhess
    joblist(k)=-nvareff+k-1
  enddo
  joblist_todo = pack(joblist,.not.hessians_done)
  if (nproc==1) then
    kmin_vec(0)=-nvareff
    kmax_vec(0)= nvareff
  else
    jmax_vec(:)=int(ntodo/nproc)
    ires=mod(ntodo,nproc)
    if (ires>0) jmax_vec(0:ires-1)=jmax_vec(0:ires-1)+1
    jmin_vec(0)=1
    do i=1,nproc-1
      if (jmax_vec(i)==0) then
        jmin_vec(i)=-1
        jmax_vec(i)=-2
      else
        jmin_vec(i)=jmax_vec(i-1)+1
        jmax_vec(i)=jmin_vec(i)+jmax_vec(i)-1
      endif
    enddo
    do m=0,nproc-1
      if (jmin_vec(m)<0) then
        kmin_vec(m)=jmin_vec(m)
        kmax_vec(m)=jmax_vec(m)
      else
        kmin_vec(m)=joblist_todo(jmin_vec(m))
        kmax_vec(m)=joblist_todo(jmax_vec(m))
      endif
    enddo
  endif
else
  allocate(kmin_vec(0))
  allocate(kmax_vec(0))
endif

call dlf_global_int_scatter_rank0(kmin_vec,kmin,nproc,0)
call dlf_global_int_scatter_rank0(kmax_vec,kmax,nproc,0)

call allocate(proc_assignment,nhess)
if (glob%iam==0) then
  do i=1,nhess
    k=i-nvareff-1
    if (hessians_done(i)) then
      proc_assignment(i)=-1
    else
      proc_assignment(i)=0
      do m=0,nproc-1
        if (kmin_vec(m)<=k .and. k<=kmax_vec(m)) then
          exit
        endif
        proc_assignment(i)=proc_assignment(i)+1
      enddo
      if (proc_assignment(i)>nproc-1) &
      & call error_print('calculate_displaced_hessians_mpi: error in job assignment mapping!')
    endif
  enddo
endif

call dlf_global_int_bcast(proc_assignment,nhess,0)

if (glob%iam==0) then
  deallocate(kmin_vec)
  deallocate(kmax_vec)
  deallocate(jmin_vec)
  deallocate(jmax_vec)
  deallocate(joblist_todo)
else
  deallocate(kmin_vec)
  deallocate(kmax_vec)
endif

ntodo_this_process=count(.not.hessians_done(displacement_map(kmin):displacement_map(kmax)))
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~Numerical~Differentiation~of~Analytical~Hessians~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of Hessian calls:             ', nhess
if(.not.silent) write(stdout,'(A,I0)') 'Already done in previous run(s):           ', ndone
if(.not.silent) write(stdout,'(A,I0)') 'Total number of Hessian calls in this run: ', ntodo
if(.not.silent) write(stdout,'(A,I0)') 'Number of calls for this process:          ', ntodo_this_process
if(.not.silent) write(stdout,'(A)') ''

hesscount=0

! Iterate over normal modes
#ifdef TEST_MODIFIED_HESSIAN_ORDER
do k_loop=kmin,kmax
  k=modify_displ_order(k_loop,nvareff)
#else
do k=kmin,kmax
#endif
  if (hessians_done(displacement_map(k))) cycle
  if (k==0) then
    dQ=0._rk
    coordtmp=coord0
  else
    dQ=sign(deltaQdiff(abs(k)),real(k,kind=rk))
    coordtmp=coord0+dQ*normal_modes_non_mw(:,abs(k))
  endif
  hesscount=hesscount+1
  if(.not.silent) then 
    if (k>0) then 
      write(stdout,'(A,I0,A,I0,A,I0,A)') 'Computing Hessian with pos. elongation for normal mode ', &
                  & abs(k), ', progress for this proc.: (',hesscount,'/',ntodo_this_process,')'
    elseif (k<0) then
      write(stdout,'(A,I0,A,I0,A,I0,A)') 'Computing Hessian with neg. elongation for normal mode ', & 
                  & abs(k), ', progress for this proc.: (',hesscount,'/',ntodo_this_process,')'
    else
      write(stdout,'(A,I0,A,I0,A)') 'Computing central Hessian,                                 ' // &
                  & 'progress for this proc.: (',hesscount,'/',ntodo_this_process,')'
    endif
  endif
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') k
  call hess_routine(nvar,coordtmp,hess,istat,ktag)
#else
  call hess_routine(nvar,coordtmp,hess,istat)
#endif
  hessians_cart(:,:,displacement_map(k)) = hess
  ! write to punch file
  write(3499,'(A)') '$HESSIAN_CART'
  write(3499,'(I0,1X,I0,1X,ES15.6)') abs(k), sign(1,k), abs(dQ)
  call matrix_output(hessians_cart(:,:,displacement_map(k)),3499,'ES24.16','__BLANK__')
enddo

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

!! Gather and broadcast all intent(out) stuff
do i=1,nhess
  if (proc_assignment(i)<0) then
    cycle
  endif
  call dlf_global_real_bcast_rank2(hessians_cart(:,:,i),nvar,nvar,proc_assignment(i))
enddo

hessians_done(:)=.true.

close(3499)
call deallocate(proc_assignment)

return
end subroutine calculate_displaced_hessians_mpi

! ****************************************
! ****************************************
! ****************************************

! Do finite difference calculation of 3rd/4th derivatives, using a previously calculated
! table of Hessians at displaced geometries (calculated with calculate_displaced_hessians_mpi). 
! This allows a logical separation of the quantum-chemistry and derivative calculation parts, 
! which is useful for restart capabilities etc.

! normal_modes: normal mode vectors in mass-weighted Cartesians
! mv3: mass vector (m1, m1, m1, m2, m2, m2, ..., mnat, mnat, mnat) in atomic mass units (multiples of electron mass)
subroutine derivatives_from_hessian_list(nvar,nvareff,nhess,hessians_cart,normal_modes_non_mw, &
                         & displacement_map,cubic_fc,quartic_fc,deltaQdiff_in)
use dlf_vpt2_utility, only: error_print
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nvareff,nhess
real(rk), dimension(nvar,nvar,nhess), intent(in) :: hessians_cart
real(rk), dimension(nvar,nvareff), intent(in) :: normal_modes_non_mw
integer, dimension(-nvareff:+nvareff), intent(in) :: displacement_map
real(rk), intent(in),dimension(:) :: deltaQdiff_in
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: cubic_fc
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_fc

real(rk), parameter :: one_third=1._rk/3._rk, one_sixth=1._rk/6._rk
real(rk), dimension(nvar,nvar) :: hessplus,hessminus, hess0
real(rk), dimension(nvareff,nvareff) :: fd1,fd2
real(rk), dimension(nvareff) :: deltaQdiff
integer, dimension(3) :: ot
real(rk) :: inv2del,invdelsq,dQ
integer  :: k,i,j,inu,jnu,knu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, &
     &     size(hessians_cart,dim=1), size(hessians_cart,dim=2)  , &
     &     size(normal_modes_non_mw,dim=1) ] , &
     &  'dim. mismatch, derivatives_from_hessian_list')
  call checkeq( [nvareff, &
     &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
     &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
     &     size(normal_modes_non_mw,dim=2), &
     &     size(quartic_fc,dim=2), size(quartic_fc,dim=3)  ], &
     &  'dim. mismatch, derivatives_from_hessian_list')
  call checkeq( [nhess, size(hessians_cart,dim=3)  ], &
     &  'dim. mismatch, derivatives_from_hessian_list')
  call checkeq( [2*nvareff+1, size(displacement_map)  ], &
     &  'dim. mismatch, derivatives_from_hessian_list')
  write(stdout,*) 'derivatives_from_hessian_list: array shape check successful.'
#endif

cubic_fc=0._rk
quartic_fc=0._rk

! Use uniform or individual elongations dQ
if (size(deltaQdiff_in)==1) then
  deltaQdiff(:)=deltaQdiff_in(1)
elseif (size(deltaQdiff_in)==nvareff) then
  deltaQdiff(:)=deltaQdiff_in(:)
else
  call error_print('derivatives_from_hessian_list: wrong dimension for deltaQdiff_in')
endif

hess0=hessians_cart(:,:,displacement_map(0))

! Iterate over normal modes
do k=1,nvareff
  dQ=deltaQdiff(k)
  inv2del=0.5_rk/dQ
  invdelsq=1._rk/dQ/dQ
  hessplus(:,:) =hessians_cart(:,:,displacement_map(+k))
  hessminus(:,:)=hessians_cart(:,:,displacement_map(-k))
  fd1(:,:)=dlf_matrix_ortho_trans(normal_modes_non_mw, hessplus-hessminus, 0)
  fd2(:,:)=dlf_matrix_ortho_trans(normal_modes_non_mw, hessplus + hessminus - 2._rk*hess0, 0)
  do i=1,nvareff
    do j=1,nvareff
      ot=ot3(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+fd1(i,j)*inv2del
      ot=ot4(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+fd2(i,j)*invdelsq
    enddo
  enddo
enddo

do inu=1,nvareff
  do jnu=1,inu
    do knu=1,jnu
      if (knu.eq.jnu) then
        if (jnu.eq.inu) then
          continue
          ! three equal indices, do nothing
        else
          ! two equal indices, divide by 3
          cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
        endif
      elseif (jnu.eq.inu) then
        ! two equal indices, divide by 3
        cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
      else
        ! three different indices, divide by 6
        cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_sixth
      endif
      ! Quartic force constants iiii have exactly one source of info
      ! multiply by 2 here (see division by 2 below)
      if (inu.eq.jnu .and. jnu.eq.knu) then
        quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)*2._rk
      endif
    enddo
  enddo
enddo

quartic_fc=quartic_fc/2._rk

return
end subroutine derivatives_from_hessian_list

! ****************************************
! ****************************************
! ****************************************

! Combined calculation of displaced Hessians (single-normal-mode displacements)
! and conversion to cubic/quartic force constants using finite-difference formula.
! This is now deprecated, in favor of the split routines
! calculate_displaced_hessians_mpi and derivatives_from_hessian_list

! normal_modes: normal mode vectors in mass-weighted Cartesians
! mv3: mass vector (m1, m1, m1, m2, m2, m2, ..., mnat, mnat, mnat) in atomic mass units (multiples of electron mass)
subroutine differentiate_hessians(hess_routine,nvar,nzero,coord0,normal_modes,mv3, &
                                & cubic_fc,quartic_fc,deltaQdiff_in,silent_in)
use dlf_vpt2_utility, only: error_print
#ifdef TEST_FIXED_HESSIAN_ORDER
use dlf_vpt2_utility, only: matrix_output
#endif
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_HESS_DEBUG_OUT
  subroutine hess_routine(nvar,coords,hessian,status,calctag)
#else
  subroutine hess_routine(nvar,coords,hessian,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: hessian(nvar,nvar)
    integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine hess_routine
end interface
integer, intent(in) :: nvar,nzero
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), intent(in), dimension(nvar,nvar-nzero) :: normal_modes
real(rk), intent(in),dimension(:) :: deltaQdiff_in
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: cubic_fc
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: quartic_fc
logical, intent(in),optional :: silent_in

#ifdef VPT2_HESS_DEBUG_OUT
character(len=4) :: ktag
#endif
real(rk), parameter :: one_third=1._rk/3._rk, one_sixth=1._rk/6._rk
real(rk), dimension(nvar,nvar) :: hess0, hessplus, hessminus, invsqrtmass
real(rk), dimension(nvar-nzero,nvar-nzero) :: hessplusqn,hessminusqn, hess0qn
real(rk), dimension(nvar,nvar-nzero) :: normal_modes_non_mw
real(rk), dimension(nvar) :: coordtmp
real(rk), dimension(nvar-nzero) :: deltaQdiff
integer, dimension(3) :: ot
real(rk) :: inv2del,invdelsq,dQ
integer  :: nvareff
integer  :: istat,k,i,j,inu,jnu,knu,hesscount
logical  :: silent
#ifdef TEST_FIXED_HESSIAN_ORDER
real(rk), dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: hessplusqn_sav,hessminusqn_sav
#endif
#ifdef TEST_RANDOM_HESSIAN_ORDER
integer :: kp,nremaining
real(rk) :: ran_real_number
integer  :: random_integer
integer, dimension(nvar-nzero) :: randomized_index, remaining_index_list, index_list
logical, dimension(nvar-nzero) :: index_done
nvareff=nvar-nzero
write(stdout,*) 'Test in differentiate_hessians: Randomized order of normal mode displacements'
call random_seed()
index_done(:)=.false.
do kp=1,nvareff
  index_list(kp)=kp
enddo
do kp=1,nvareff
  nremaining=nvareff-kp+1
  call random_number(ran_real_number)
  ran_real_number=nremaining*ran_real_number+1._rk
  random_integer=int(ran_real_number)
  if (random_integer>nremaining) random_integer=nremaining
  remaining_index_list(1:nremaining)=pack(index_list,.not.index_done)
  randomized_index(kp)=remaining_index_list(random_integer)
  index_done(randomized_index(kp))=.true.
enddo
#endif
#ifdef TEST_FIXED_HESSIAN_ORDER
write(stdout,*) 'Test in differentiate_hessians: Fixed order of normal mode displacements, as in new, restartable routine'
#endif

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(coord0), size(mv3), size(normal_modes, dim=1) ], &
     &  'dim. mismatch, differentiate_hessians')
  call checkeq( [nvar-nzero, size(normal_modes, dim=2), &
     &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
     &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
     &     size(quartic_fc,dim=2), size(quartic_fc,dim=3)  ], &
     &  'dim. mismatch, differentiate_hessians')
  write(stdout,*) 'differentiate_hessians: array shape check successful.'
#endif

silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif

nvareff=nvar-nzero
cubic_fc=0._rk
quartic_fc=0._rk

! Use uniform or individual elongations dQ
if (size(deltaQdiff_in)==1) then
  deltaQdiff(:)=deltaQdiff_in(1)
elseif (size(deltaQdiff_in)==nvareff) then
  deltaQdiff(:)=deltaQdiff_in(:)
else
  call error_print('differentiate_hessians: wrong dimension for deltaQdiff_in')
endif

! Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~Numerical~Differentiation~of~Analytical~Hessians~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of Hessian calls: ', 2*nvareff+1
if(.not.silent) write(stdout,'(A)') ''
hesscount=1
if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', 2*nvareff+1, '...'

! Calculate and save central hessian

#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') 0
  call hess_routine(nvar,coord0,hess0,istat,ktag)
#else
  call hess_routine(nvar,coord0,hess0,istat)
#endif

hess0qn=dlf_matrix_ortho_trans(normal_modes_non_mw,hess0, 0)

! Iterate over normal modes
#ifdef TEST_FIXED_HESSIAN_ORDER
open(3499,file='dlf_vpt2_hessians.dat',action='write')
write(3499,'(A)') '$HESSIAN_CART'
write(3499,'(I0,1X,I0,1X,ES15.6)') 0, 0, 0._rk
call matrix_output(hess0qn(:,:),3499,'ES24.16','__BLANK__')

do k=-nvareff,nvareff
  if (k==0) cycle
  dQ=deltaQdiff(abs(k))
  coordtmp=coord0+sign(dQ,real(k))*normal_modes_non_mw(:,abs(k))
  hesscount=hesscount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', 2*nvareff+1, '...'
  if (k>0) then
#ifdef VPT2_HESS_DEBUG_OUT
    write(ktag,'(SP,I4.3)') k
    call hess_routine(nvar,coordtmp,hessplus,istat,ktag)
#else
    call hess_routine(nvar,coordtmp,hessplus,istat)
#endif
    hessplusqn_sav(:,:,abs(k)) =dlf_matrix_ortho_trans(normal_modes_non_mw,hessplus, 0)
  else
#ifdef VPT2_HESS_DEBUG_OUT
    write(ktag,'(SP,I4.3)') k
    call hess_routine(nvar,coordtmp,hessminus,istat,ktag)
#else
    call hess_routine(nvar,coordtmp,hessminus,istat)
#endif
    hessminusqn_sav(:,:,abs(k))=dlf_matrix_ortho_trans(normal_modes_non_mw,hessminus,0)
  endif
  write(3499,'(A)') '$HESSIAN_CART'
  write(3499,'(I0,1X,I0,1X,ES15.6)') abs(k), sign(1,k), abs(dQ)
  if (k>0) then
    call matrix_output(hessplusqn_sav(:,:,abs(k)),3499,'ES24.16','__BLANK__')
  else
    call matrix_output(hessminusqn_sav(:,:,abs(k)),3499,'ES24.16','__BLANK__')
  endif
enddo
close(3499)

do k=1,nvareff
  dQ=deltaQdiff(k)
  inv2del=0.5_rk/dQ
  invdelsq=1._rk/dQ/dQ
  do i=1,nvareff
    do j=1,nvareff
      ot=ot3(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+(hessplusqn_sav(i,j,k)-hessminusqn_sav(i,j,k))*inv2del
      ot=ot4(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+(hessplusqn_sav(i,j,k)+hessminusqn_sav(i,j,k)-2._rk*hess0qn(i,j))*invdelsq
    enddo
  enddo
enddo

#else

! Iterate over normal modes
#ifdef TEST_RANDOM_HESSIAN_ORDER
do kp=1,nvareff
  k=randomized_index(kp)
#else
do k=1,nvareff
#endif
  dQ=deltaQdiff(k)
  inv2del=0.5_rk/dQ
  invdelsq=1._rk/dQ/dQ
  coordtmp=coord0+dQ*normal_modes_non_mw(:,k)
  hesscount=hesscount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', 2*nvareff+1, '...'
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') k
  call hess_routine(nvar,coordtmp,hessplus,istat,ktag)
#else
  call hess_routine(nvar,coordtmp,hessplus,istat)
#endif
  coordtmp=coord0-dQ*normal_modes_non_mw(:,k)
  hesscount=hesscount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', 2*nvareff+1, '...'
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') -k
  call hess_routine(nvar,coordtmp,hessminus,istat,ktag)
#else
  call hess_routine(nvar,coordtmp,hessminus,istat)
#endif
  hessplusqn =dlf_matrix_ortho_trans(normal_modes_non_mw,hessplus, 0)
  hessminusqn=dlf_matrix_ortho_trans(normal_modes_non_mw,hessminus,0)
  do i=1,nvareff
    do j=1,nvareff
      ot=ot3(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+(hessplusqn(i,j)-hessminusqn(i,j))*inv2del
      ot=ot4(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+(hessplusqn(i,j)+hessminusqn(i,j)-2._rk*hess0qn(i,j))*invdelsq
    enddo
  enddo
enddo

#endif

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

do inu=1,nvareff
  do jnu=1,inu
    do knu=1,jnu
      if (knu.eq.jnu) then
        if (jnu.eq.inu) then
          continue
          ! three equal indices, do nothing
        else
          ! two equal indices, divide by 3
          cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
        endif
      elseif (jnu.eq.inu) then
        ! two equal indices, divide by 3
        cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
      else
        ! three different indices, divide by 6
        cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_sixth
      endif
      ! Quartic force constants iiii have exactly one source of info
      ! multiply by 2 here (see division by 2 below)
      if (inu.eq.jnu .and. jnu.eq.knu) then
        quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)*2._rk
      endif
    enddo
  enddo
enddo

quartic_fc=quartic_fc/2._rk

return
end subroutine differentiate_hessians

! ****************************************
! ****************************************
! ****************************************

! MPI-parallelized version of differentiate_hessians

subroutine differentiate_hessians_mpi(hess_routine,nvar,nzero,coord0,normal_modes,mv3, &
                                & cubic_fc,quartic_fc,deltaQdiff_in,silent_in)
use dlf_vpt2_utility, only: error_print, dlf_global_real_bcast_rank3, dlf_global_real_sum_rank3, &
                            & dlf_global_real_bcast_rank2, dlf_global_int_scatter_rank0
use dlf_linalg_interface_mod
use dlf_allocate
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
#ifdef VPT2_HESS_DEBUG_OUT
  subroutine hess_routine(nvar,coords,hessian,status,calctag)
#else
  subroutine hess_routine(nvar,coords,hessian,status)
#endif
    use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: hessian(nvar,nvar)
    integer   ,intent(out)   :: status
#ifdef VPT2_HESS_DEBUG_OUT
    character(len=*), intent(in),optional :: calctag
#endif
  end subroutine hess_routine
end interface
integer, intent(in) :: nvar,nzero
real(rk), intent(in), dimension(nvar) :: coord0, mv3
real(rk), intent(in), dimension(nvar,nvar-nzero) :: normal_modes
real(rk), intent(in),dimension(:) :: deltaQdiff_in
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: cubic_fc
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: quartic_fc
logical, intent(in),optional :: silent_in

real(rk), parameter :: one_third=1._rk/3._rk, one_sixth=1._rk/6._rk
real(rk), dimension(nvar,nvar) :: hess0, hessplus, hessminus, invsqrtmass
real(rk), dimension(nvar-nzero,nvar-nzero) :: hessplusqn,hessminusqn, hess0qn
real(rk), dimension(nvar,nvar-nzero) :: normal_modes_non_mw
real(rk), dimension(nvar) :: coordtmp
real(rk), dimension(nvar-nzero) :: deltaQdiff
integer, dimension(3) :: ot
real(rk) :: inv2del,invdelsq,dQ
integer  :: nvareff,istat,k,i,j,inu,jnu,knu,hesscount,nproc,nhesstot
integer  :: kmin, kmax, ires
#ifdef VPT2_HESS_DEBUG_OUT
character(len=4) :: ktag
#endif

integer, allocatable :: kmin_vec(:), kmax_vec(:)
logical  :: silent

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(coord0), size(mv3), size(normal_modes, dim=1) ], &
     &  'dim. mismatch, differentiate_hessians_mpi')
  call checkeq( [nvar-nzero, size(normal_modes, dim=2), &
     &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
     &     size(cubic_fc,dim=3), size(quartic_fc,dim=1), &
     &     size(quartic_fc,dim=2), size(quartic_fc,dim=3)  ], &
     &  'dim. mismatch, differentiate_hessians_mpi')
  write(stdout,*) 'differentiate_hessians_mpi: array shape check successful.'
#endif

nproc=glob%nprocs
silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif

nvareff=nvar-nzero
cubic_fc=0._rk
quartic_fc=0._rk

! Use uniform or individual elongations dQ
if (size(deltaQdiff_in)==1) then
  deltaQdiff(:)=deltaQdiff_in(1)
elseif (size(deltaQdiff_in)==nvareff) then
  deltaQdiff(:)=deltaQdiff_in(:)
else
  call error_print('differentiate_hessians_mpi: wrong dimension for deltaQdiff_in')
endif

! Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

nhesstot=2*nvareff+1

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~Numerical~Differentiation~of~Analytical~Hessians~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of Hessian calls: ', nhesstot
if(.not.silent) write(stdout,'(A)') ''

if (glob%iam==0) then
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', 1, ' of ', nhesstot, '...'
  ! Calculate and save central hessian
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') 0
  call hess_routine(nvar,coord0,hess0,istat,ktag)
#else
  call hess_routine(nvar,coord0,hess0,istat)
#endif
  hess0qn=dlf_matrix_ortho_trans(normal_modes_non_mw,hess0, 0)
endif

call dlf_global_real_bcast_rank2(hess0,nvar,nvar,0)
call dlf_global_real_bcast_rank2(hess0qn,nvar-nzero,nvar-nzero,0)

if (glob%iam==0) then
  allocate(kmin_vec(0:nproc-1))
  allocate(kmax_vec(0:nproc-1))
  if (nproc==1) then
    kmin_vec(0)=1
    kmax_vec(0)=nvareff
  else
    kmax_vec(:)=int(nvareff/nproc)
    ires=mod(nvareff,nproc)
    if (ires>0) kmax_vec(0:ires-1)=kmax_vec(0:ires-1)+1
    kmin_vec(0)=1
    do i=1,nproc-1
      if (kmax_vec(i)==0) then
        kmin_vec(i)=-1
        kmax_vec(i)=-2
      else
        kmin_vec(i)=kmax_vec(i-1)+1
        kmax_vec(i)=kmin_vec(i)+kmax_vec(i)-1
      endif
    enddo
  endif
else
  allocate(kmin_vec(0))
  allocate(kmax_vec(0))
endif

call dlf_global_int_scatter_rank0(kmin_vec,kmin,nproc,0)
call dlf_global_int_scatter_rank0(kmax_vec,kmax,nproc,0)

deallocate(kmin_vec)
deallocate(kmax_vec)

! Iterate over normal modes
do k=kmin,kmax
  dQ=deltaQdiff(k)
  inv2del=0.5_rk/dQ
  invdelsq=1._rk/dQ/dQ
  coordtmp=coord0+dQ*normal_modes_non_mw(:,k)
  hesscount=2*k
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', nhesstot, '...'
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') k
  call hess_routine(nvar,coordtmp,hessplus,istat,ktag)
#else
  call hess_routine(nvar,coordtmp,hessplus,istat)
#endif
  coordtmp=coord0-dQ*normal_modes_non_mw(:,k)
  hesscount=2*k+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing Hessian ', hesscount, ' of ', nhesstot, '...'
#ifdef VPT2_HESS_DEBUG_OUT
  write(ktag,'(SP,I4.3)') -k
  call hess_routine(nvar,coordtmp,hessminus,istat,ktag)
#else
  call hess_routine(nvar,coordtmp,hessminus,istat)
#endif
  hessplusqn =dlf_matrix_ortho_trans(normal_modes_non_mw,hessplus, 0)
  hessminusqn=dlf_matrix_ortho_trans(normal_modes_non_mw,hessminus,0)
  do i=1,nvareff
    do j=1,nvareff
      ot=ot3(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)+(hessplusqn(i,j)-hessminusqn(i,j))*inv2del
      ot=ot4(k,i,j)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)+(hessplusqn(i,j)+hessminusqn(i,j)-2._rk*hess0qn(i,j))*invdelsq
    enddo
  enddo
enddo

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

call dlf_global_real_sum_rank3(cubic_fc,nvar-nzero,nvar-nzero,nvar-nzero)
call dlf_global_real_sum_rank3(quartic_fc,nvar-nzero,nvar-nzero,nvar-nzero)

if (glob%iam==0) then
  do inu=1,nvareff
    do jnu=1,inu
      do knu=1,jnu
        if (knu.eq.jnu) then
          if (jnu.eq.inu) then
            continue
            ! three equal indices, do nothing
          else
            ! two equal indices, divide by 3
            cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
          endif
        elseif (jnu.eq.inu) then
          ! two equal indices, divide by 3
          cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_third
        else
          ! three different indices, divide by 6
          cubic_fc(inu,jnu,knu)=cubic_fc(inu,jnu,knu)*one_sixth
        endif
        ! Quartic force constants iiii have exactly one source of info
        ! multiply by 2 here (see division by 2 below)
        if (inu.eq.jnu .and. jnu.eq.knu) then
          quartic_fc(inu,jnu,knu)=quartic_fc(inu,jnu,knu)*2._rk
        endif
      enddo
    enddo
  enddo

quartic_fc=quartic_fc/2._rk
endif

call dlf_global_real_bcast_rank3(cubic_fc,  nvar-nzero,nvar-nzero,nvar-nzero,0)
call dlf_global_real_bcast_rank3(quartic_fc,nvar-nzero,nvar-nzero,nvar-nzero,0)

return
end subroutine differentiate_hessians_mpi

! ****************************************
! ****************************************
! ****************************************

! Assume that only array elements with i.ge.j are filled with 
! values at the beginning.
! Fill in all other elements, making use of the Hessian symmetry
subroutine symmetric_fill_hessian(neff,hessian_in,hessian_out)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), dimension(neff,neff), intent(in)  :: hessian_in
real(rk), dimension(neff,neff), intent(out) :: hessian_out

integer :: i, j, inu, jnu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(hessian_in,dim=1),  size(hessian_in,dim=2), &
     &     size(hessian_out,dim=1), size(hessian_out,dim=2)  ], &
     &  'dim. mismatch, symmetric_fill_hessian')
  write(stdout,*) 'symmetric_fill_hessian: array shape check successful.'
#endif

do i=1,neff
  do j=1,neff
    inu=max(i,j)
    jnu=min(i,j)
    hessian_out(i,j)=hessian_in(inu,jnu)
  enddo
enddo

return
end subroutine symmetric_fill_hessian

! ****************************************
! ****************************************
! ****************************************

! Assume that only array elements with i.ge.j.ge.k are stored at the beginning
! in the cubic force constant array.
! Fill in all other elements, exploiting symmetry
subroutine symmetric_fill_cubic(neff,cubic_in,cubic_out)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), dimension(neff,neff,neff), intent(in)  :: cubic_in
real(rk), dimension(neff,neff,neff), intent(out) :: cubic_out
integer, dimension(3) :: ot

integer :: i, j, k, inu, jnu, knu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(cubic_in,dim=1), size(cubic_in,dim=2), &
     &     size(cubic_in,dim=3), size(cubic_out,dim=1), &
     &     size(cubic_out,dim=2), size(cubic_out,dim=3)  ], &
     &  'dim. mismatch, symmetric_fill_cubic')
  write(stdout,*) 'symmetric_fill_cubic: array shape check successful.'
#endif

do i=1,neff
  do j=1,neff
    do k=1,neff
      ot=ot3(i,j,k)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      cubic_out(i,j,k)=cubic_in(inu,jnu,knu)
    enddo
  enddo
enddo

return
end subroutine symmetric_fill_cubic

! ****************************************
! ****************************************
! ****************************************

! Assume that only array elements with i.ge.j.ge.k are stored at the beginning
! in the quartic force constant array.
! Fill in all other elements, exploiting symmetry
subroutine symmetric_fill_quartic_semidiag(neff,quartic_in,quartic_out)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), dimension(neff,neff,neff), intent(in)  :: quartic_in
real(rk), dimension(neff,neff,neff), intent(out) :: quartic_out
integer, dimension(3) :: ot

integer :: i, j, k, inu, jnu, knu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(quartic_in,dim=1), size(quartic_in,dim=2), &
     &     size(quartic_in,dim=3), size(quartic_out,dim=1), &
     &     size(quartic_out,dim=2), size(quartic_out,dim=3)  ], &
     &  'dim. mismatch, symmetric_fill_quartic_semidiag')
  write(stdout,*) 'symmetric_fill_quartic_semidiag: array shape check successful.'
#endif

do i=1,neff
  do j=1,neff
    do k=1,neff
      ot=ot4(i,j,k)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      quartic_out(i,j,k)=quartic_in(inu,jnu,knu)
    enddo
  enddo
enddo

return
end subroutine symmetric_fill_quartic_semidiag

! ****************************************
! ****************************************
! ****************************************

! Like symmetric_fill_cubic, but doesn't assume that the non-zero elements
! are necessarily stored in elements ijk with i>=j>=k.
! Instead, find it out by actually inspecting the elements
subroutine symmetric_fill_cubic_generalized(neff,cubic_in,cubic_out)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), dimension(neff,neff,neff), intent(in)  :: cubic_in
real(rk), dimension(neff,neff,neff), intent(out) :: cubic_out

integer  :: i, j, k, mmax
real(rk),dimension(6) :: v
real(rk) :: valu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(cubic_in,dim=1), size(cubic_in,dim=2), &
     &     size(cubic_in,dim=3), size(cubic_out,dim=1), &
     &     size(cubic_out,dim=2), size(cubic_out,dim=3)  ], &
     &  'dim. mismatch, symmetric_fill_cubic_generalized')
  write(stdout,*) 'symmetric_fill_cubic_generalized: array shape check successful.'
#endif

cubic_out=cubic_in
do i=1,neff
  do j=1,neff
    do k=1,neff
      v(1)=cubic_in(i,j,k)
      v(2)=cubic_in(i,k,j)
      v(3)=cubic_in(j,i,k)
      v(4)=cubic_in(j,k,i)
      v(5)=cubic_in(k,i,j)
      v(6)=cubic_in(k,j,i)
      mmax=maxloc(abs(v),dim=1)
      valu=v(mmax)
      cubic_out(i,j,k)=valu
      cubic_out(i,k,j)=valu
      cubic_out(j,i,k)=valu
      cubic_out(j,k,i)=valu
      cubic_out(k,i,j)=valu
      cubic_out(k,j,i)=valu
    enddo
  enddo
enddo

return
end subroutine symmetric_fill_cubic_generalized

! ****************************************
! ****************************************
! ****************************************

! Like symmetric_fill_quartic, but doesn't assume that the non-zero elements
! are necessarily stored in elements ijk with i>=j>=k.
! Instead, find it out by actually inspecting the elements
subroutine symmetric_fill_quartic_generalized(neff,quartic_in,quartic_out)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), dimension(neff,neff,neff), intent(in)  :: quartic_in
real(rk), dimension(neff,neff,neff), intent(out) :: quartic_out

integer  :: i, j, k, mmax
real(rk),dimension(3) :: v
real(rk) :: valu

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(quartic_in,dim=1), size(quartic_in,dim=2), &
     &     size(quartic_in,dim=3), size(quartic_out,dim=1), &
     &     size(quartic_out,dim=2), size(quartic_out,dim=3)  ], &
     &  'dim. mismatch, symmetric_fill_quartic_generalized')
  write(stdout,*) 'symmetric_fill_quartic_generalized: array shape check successful.'
#endif

quartic_out=quartic_in
do i=1,neff
  do j=1,neff
    do k=1,neff
      v(1)=quartic_in(i,j,k)
      v(2)=quartic_in(i,k,j)
      if (j.eq.k) then
        v(3)=quartic_in(j,i,i)
      else
        v(3)=0._rk
      endif
      mmax=maxloc(abs(v),dim=1)
      valu=v(mmax)
      quartic_out(i,j,k)=valu
      quartic_out(i,k,j)=valu
      if (j.eq.k) quartic_out(j,i,i)=valu
    enddo
  enddo
enddo

return
end subroutine symmetric_fill_quartic_generalized

! ****************************************
! ****************************************
! ****************************************

! Ordered triple for cubic force constants.
! three unordered integer indices i,j,k, 
! an ordered triple of integers is output in the 
! rank-3 array ot3(inew,jnew,knew). The order is 
! such that inew >= jnew >= knew 
function ot3(i,j,k)
implicit none
integer,intent(in) :: i,j,k
integer,dimension(3) :: ot3

logical :: igej,igek,jgek

igej=i.ge.j
igek=i.ge.k
jgek=j.ge.k

if (igej) then
  if (igek) then
    if (jgek) then
      ot3=(/ i,j,k /)  ! i>j>k
    else
      ot3=(/ i,k,j /)  ! i>k>j
    endif
  else
    ot3=(/ k,i,j /)    ! k>i>j
  endif
else
  if (igek) then
    ot3=(/ j,i,k /)    ! j>i>k
  else
    if (jgek) then
      ot3=(/ j,k,i /)  ! j>k>i
    else
      ot3=(/ k,j,i /)  ! k>j>i
    endif
  endif
endif

return
end function ot3

! ****************************************
! ****************************************
! ****************************************

! Ordered triple for semidiagonal quartic constants
! Input: three integer i,j,k describing the semidiagonal
! quartic force constants, i.e. describing Phi_iijk
! An ordered triple is output in ot4 such that 
! 1. j>=k
! 2. If j=k, and j>i, then i and j are exchanged, 
!    i.e. Phi_iijj => Phi_jjii
! In summary, it is made sure that inew >= jnew >= knew
function ot4(i,j,k)
implicit none
integer,intent(in) :: i,j,k
integer,dimension(3) :: ot4

logical :: jgek, jeqk, jgti

jgek=j.ge.k
jeqk=j.eq.k
jgti=j.gt.i

if (jeqk) then
  if (jgti) then
    ot4=(/j,i,i/)
  else
    ot4=(/i,j,k/)
  endif
else
  if (jgek) then
    ot4=(/i,j,k/)
  else
    ot4=(/i,k,j/)
  endif
endif

return
end function ot4

! ****************************************
! ****************************************
! ****************************************

! Ordered triple for semidiagonal quartic constants, obtained from
! unordered quartet
! Similar to ot4, but with less assumptions about the indices (i.e. here
! i is not the special two-fold index as in ot4). Any unordered quartet can 
! be input. An ordered triple is output, where inew is the two-fold index
! and jnew, knew are the 3rd and 4th indices. The routine will drop an error
! if all four input indices are different.

function ot4gen(i,j,k,m)
use dlf_vpt2_utility, only: error_print
implicit none
integer,intent(in) :: i,j,k,m
integer,dimension(3) :: ot4gen

integer :: inu,jnu,knu
logical :: jgek, jeqk, jgti

if     (i.eq.j) then
  inu=i
  jnu=k
  knu=m
elseif (i.eq.k) then
  inu=i
  jnu=j
  knu=m
elseif (i.eq.m) then
  inu=i
  jnu=j
  knu=k
elseif (j.eq.k) then
  inu=j
  jnu=i
  knu=m
elseif (j.eq.m) then
  inu=j
  jnu=i
  knu=k
elseif (k.eq.m) then
  inu=k
  jnu=i
  knu=j
else
  call error_print('ot4gen: invalid usage: all four indices are different')
endif

jgek=jnu.ge.knu
jeqk=jnu.eq.knu
jgti=jnu.gt.inu

if (jeqk) then
  if (jgti) then
    ot4gen=(/jnu,inu,inu/)
  else
    ot4gen=(/inu,jnu,knu/)
  endif
else
  if (jgek) then
    ot4gen=(/inu,jnu,knu/)
  else
    ot4gen=(/inu,knu,jnu/)
  endif
endif

return
end function ot4gen

! ****************************************
! ****************************************
! ****************************************

! Get cubic and quartic force constants via a routine that provides those 
! derivatives in Cartesian coordinates, then transform these into normal 
! coordinates. Depends on analytic routine "cubic_quartic_routine"

subroutine get_cubic_quartic_nm_via_analytical_routine(cubic_quartic_routine,nvar,nvareff, &
                                    &  coords,mv3,normal_modes,cubic_fc_nm,quartic_fc_nm)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  use dlf_vpt2_utility, only: checkeq
#endif
  implicit none
  interface
    subroutine cubic_quartic_routine(nvar,coords,cubic,quartic,status)
      use dlf_parameter_module, only: rk
      implicit none
      integer   ,intent(in)    :: nvar
      real(rk)  ,intent(in)    :: coords(nvar)
      real(rk)  ,intent(out)   :: cubic(nvar,nvar,nvar)
      real(rk)  ,intent(out)   :: quartic(nvar,nvar,nvar,nvar)
      integer   ,intent(out)   :: status
    end subroutine cubic_quartic_routine
  end interface
  integer, intent(in) :: nvar,nvareff
  real(rk), intent(in), dimension(nvar) :: coords
  real(rk), intent(in), dimension(nvar) :: mv3
  real(rk), intent(in), dimension(nvar,nvareff) :: normal_modes
  real(rk), intent(out), dimension(nvareff,nvareff,nvareff) :: cubic_fc_nm
  real(rk), intent(out), dimension(nvareff,nvareff,nvareff) :: quartic_fc_nm

  real(rk), dimension(nvar,nvar,nvar) :: cubic_fc_cart
  real(rk), dimension(nvar,nvar,nvar,nvar) :: quartic_fc_cart
  real(rk), dimension(nvareff,nvareff,nvareff,nvareff) :: quartic_fc_nm_full
  integer :: istat,i
  
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, size(coords), &
     &     size(mv3), size(normal_modes,dim=1)  ], &
     &  'dim. mismatch, get_cubic_quartic_nm_via_analytical_routine')
  call checkeq( [nvareff, &
     &     size(normal_modes,dim=2), &
     &     size(cubic_fc_nm,dim=1), size(cubic_fc_nm,dim=2), &
     &     size(cubic_fc_nm,dim=3), &
     &     size(quartic_fc_nm,dim=1), size(quartic_fc_nm,dim=2), &
     &     size(quartic_fc_nm,dim=3)  ], &
     &  'dim. mismatch, get_cubic_quartic_nm_via_analytical_routine')
  write(stdout,*) 'get_cubic_quartic_nm_via_analytical_routine: array shape check successful.'
#endif
  
  call cubic_quartic_routine(nvar,coords,cubic_fc_cart,quartic_fc_cart,istat)
  call convert_cubic_cart_to_normal_coord(nvar,nvar-nvareff,normal_modes,mv3,cubic_fc_cart,cubic_fc_nm)
  call convert_quartic_cart_to_normal_coord(nvar,nvar-nvareff,normal_modes,mv3,quartic_fc_cart,quartic_fc_nm_full)
  do i=1,nvareff
    quartic_fc_nm(i,:,:)=quartic_fc_nm_full(i,i,:,:)
  enddo
  
  return
end subroutine get_cubic_quartic_nm_via_analytical_routine

! ****************************************
! ****************************************
! ****************************************

! Transform cubic force constants in Cartesian coordinates
! to cubic force constants in normal coordinates

subroutine convert_cubic_cart_to_normal_coord(nvar,nzero,normal_modes,mv3,cubic_cart,cubic_nm)
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nzero
real(rk), intent(in), dimension(nvar) :: mv3
real(rk), intent(in), dimension(nvar,nvar-nzero) :: normal_modes
real(rk), intent(in), dimension(nvar,nvar,nvar)  :: cubic_cart
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero) :: cubic_nm

real(rk), dimension(nvar,nvar) :: invsqrtmass
real(rk), dimension(nvar,nvar) :: phik
real(rk), dimension(nvar,nvar-nzero) :: normal_modes_non_mw

integer  :: j,k,nvareff

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, &
     &     size(mv3), size(normal_modes,dim=1), &
     &     size(cubic_cart,dim=1), size(cubic_cart,dim=2), &
     &     size(cubic_cart,dim=3)  ], &
     &  'dim. mismatch, convert_cubic_cart_to_normal_coord')
  call checkeq( [nvar-nzero, &
     &     size(normal_modes,dim=2), &
     &     size(cubic_nm,dim=1), size(cubic_nm,dim=2), &
     &     size(cubic_nm,dim=3)  ], &
     &  'dim. mismatch, convert_cubic_cart_to_normal_coord')
  write(stdout,*) 'convert_cubic_cart_to_normal_coord: array shape check successful.'
#endif

nvareff=nvar-nzero

! Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

! Convert force constants
do k=1,nvareff
  phik=0._rk
  do j=1,nvar
    phik=phik+normal_modes_non_mw(j,k)*cubic_cart(:,:,j)
  enddo
  cubic_nm(:,:,k)=dlf_matrix_ortho_trans(normal_modes_non_mw,phik,0)
enddo

return
end subroutine convert_cubic_cart_to_normal_coord

! ****************************************
! ****************************************
! ****************************************

! Rotate quartic force constant to new, improved set of normal coordinates

subroutine rotate_quartic_force_constants_semidiag(nvareff,eigvec,quartic,quartic_rot)
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvareff
real(rk), intent(in), dimension(nvareff,nvareff) :: eigvec
real(rk), intent(in), dimension(nvareff,nvareff,nvareff) :: quartic
real(rk), intent(out),dimension(nvareff,nvareff,nvareff) :: quartic_rot

integer, dimension(3) :: ot
integer  :: i,j,k,m,inu,jnu,knu
real(rk), dimension(nvareff,nvareff,nvareff,nvareff) :: quartic_4ind
real(rk), dimension(nvareff,nvareff,nvareff,nvareff) :: quartic_4ind_trans
real(rk) :: tmp

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvareff, &
     &     size(eigvec,dim=1), &
     &     size(eigvec,dim=2), &
     &     size(quartic,dim=1), &
     &     size(quartic,dim=2), &
     &     size(quartic,dim=3), &
     &     size(quartic_rot,dim=1), &
     &     size(quartic_rot,dim=2), &
     &     size(quartic_rot,dim=3)  ], &
     &  'dim. mismatch, rotate_quartic_force_constants_semidiag')
  write(stdout,*) 'rotate_quartic_force_constants_semidiag: array shape check successful.'
#endif

quartic_4ind(:,:,:,:)=0._rk
quartic_4ind_trans(:,:,:,:)=0._rk

do i=1,nvareff
  quartic_4ind(:,:,i,i)=quartic(:,:,i)
enddo
do i=1,nvareff
  do j=1,i-1
    do k=1,nvareff
      m=j
      ot=ot4gen(i,j,k,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      tmp=quartic(inu,jnu,knu)
      quartic_4ind(k,m,i,j)=tmp
      quartic_4ind(k,m,j,i)=tmp
      quartic_4ind(m,k,i,j)=tmp
      quartic_4ind(m,k,j,i)=tmp
      m=i
      ot=ot4gen(i,j,k,m)
      inu=ot(1)
      jnu=ot(2)
      knu=ot(3)
      tmp=quartic(inu,jnu,knu)
      quartic_4ind(k,m,i,j)=tmp
      quartic_4ind(k,m,j,i)=tmp
      quartic_4ind(m,k,i,j)=tmp
      quartic_4ind(m,k,j,i)=tmp
    enddo
  enddo
enddo

do i=1,nvareff
  do j=1,i
    quartic_4ind_trans(:,:,i,j)=dlf_matrix_ortho_trans(eigvec,quartic_4ind(:,:,i,j),0)
  enddo
enddo
do i=1,nvareff
  do j=i+1,nvareff
    quartic_4ind_trans(:,:,i,j)=quartic_4ind_trans(:,:,j,i)
  enddo
enddo

do i=1,nvareff
  quartic_rot(i,:,:)=0._rk
  do j=1,nvareff
    do k=1,nvareff
      quartic_rot(i,:,:)= quartic_rot(i,:,:)+eigvec(j,i)*eigvec(k,i)*quartic_4ind(:,:,j,k)
    enddo
  enddo
enddo

return
end subroutine rotate_quartic_force_constants_semidiag

! ****************************************
! ****************************************
! ****************************************

! Transform quartic force constants in Cartesian coordinates
! to quartic force constants in normal coordinates

subroutine convert_quartic_cart_to_normal_coord(nvar,nzero,normal_modes,mv3,quartic_cart,quartic_nm)
use dlf_linalg_interface_mod
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: nvar,nzero
real(rk), intent(in), dimension(nvar) :: mv3
real(rk), intent(in), dimension(nvar,nvar-nzero) :: normal_modes
real(rk), intent(in), dimension(nvar,nvar,nvar,nvar)  :: quartic_cart
real(rk), intent(out),dimension(nvar-nzero,nvar-nzero,nvar-nzero,nvar-nzero) :: quartic_nm

real(rk), dimension(nvar,nvar) :: invsqrtmass
real(rk), dimension(nvar,nvar-nzero) :: normal_modes_non_mw

integer  :: i,j,k,l,m,n,o,p,nvareff
real(rk) :: tmpm,tmpn,tmpo

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, &
     &     size(mv3), size(normal_modes,dim=1), &
     &     size(quartic_cart,dim=1), size(quartic_cart,dim=2), &
     &     size(quartic_cart,dim=3), size(quartic_cart,dim=4)  ], &
     &  'dim. mismatch, convert_quartic_cart_to_normal_coord')
  call checkeq( [nvar-nzero, &
     &     size(normal_modes,dim=2), &
     &     size(quartic_nm,dim=1), size(quartic_nm,dim=2), &
     &     size(quartic_nm,dim=3), size(quartic_nm,dim=4)  ], &
     &  'dim. mismatch, convert_quartic_cart_to_normal_coord')
  write(stdout,*) 'convert_quartic_cart_to_normal_coord: array shape check successful.'
#endif

nvareff=nvar-nzero

! Calculate non mass-weighted normal mode vectors (= normal mode Cartesian displacement vectors)
invsqrtmass=0._rk
do k=1,nvar
  invsqrtmass(k,k)=1._rk/sqrt(mv3(k))
enddo
normal_modes_non_mw=dlf_matmul_simp(invsqrtmass,normal_modes)

do i=1,nvareff
  do j=1,nvareff
    do k=1,nvareff
      do l=1,nvareff
        quartic_nm(i,j,k,l)=0._rk
        do m=1,nvar
          tmpm=0._rk
          do n=1,nvar
            tmpn=0._rk
            do o=1,nvar
              tmpo=0._rk
              do p=1,nvar
                tmpo=tmpo+quartic_cart(m,n,o,p)*normal_modes_non_mw(p,l)
              enddo
              tmpn=tmpn+tmpo*normal_modes_non_mw(o,k)
            enddo
            tmpm=tmpm+tmpn*normal_modes_non_mw(n,j)
          enddo
          quartic_nm(i,j,k,l)=quartic_nm(i,j,k,l)+tmpm*normal_modes_non_mw(m,i)
        enddo
      enddo
    enddo
  enddo
enddo

return
end subroutine convert_quartic_cart_to_normal_coord

! ****************************************
! ****************************************
! ****************************************

! Convert cubic force constants to reduced cubic force constants (divided by 
! product of vibrational frequencies of the involved modes, to the power of 1/4) 
!
! Expects normal mode cubic force constants in a.u. (Hartree/(bohr³*m_el^(3/2)))
! and eigenvalues of mass-weighted Hessian in a.u. ((atomic units of time)^(-2))
!
! Outputs reduced cubic force constants in cm^-1
! 
! CAUTION!
! Negative eigenvalues are treated as if they were positive!
! Special VPT2 formulas then have to be used for saddle points!
!

subroutine convert_cubic_to_reduced_cubic(neff,cubic_fc,eigenvalues,reduced_cubic_fc)
use dlf_constants
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff,neff,neff) :: cubic_fc
real(rk), intent(in), dimension(neff)           :: eigenvalues
real(rk), intent(out),dimension(neff,neff,neff) :: reduced_cubic_fc

integer :: i, j, k
real(rk) :: au2cmi

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(eigenvalues), &
     &     size(cubic_fc,dim=1), size(cubic_fc,dim=2), &
     &     size(cubic_fc,dim=3), &
     &     size(reduced_cubic_fc,dim=1), size(reduced_cubic_fc,dim=2), &
     &     size(reduced_cubic_fc,dim=3)  ], &
     &  'dim. mismatch, convert_cubic_to_reduced_cubic')
  write(stdout,*) 'convert_cubic_to_reduced_cubic: array shape check successful.'
#endif

call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)

do i=1,neff
  do j=1,neff
    do k=1,neff
      !reduced_cubic_fc(i,j,k)=cubic_fc(i,j,k)/sqrt(sqrt(eigenvalues(i)*eigenvalues(j)*eigenvalues(k)))
      reduced_cubic_fc(i,j,k)=cubic_fc(i,j,k)/sqrt(sqrt(abs(eigenvalues(i)*eigenvalues(j)*eigenvalues(k))))
    enddo
  enddo
enddo

! Convert from Hartree to cm^-1
reduced_cubic_fc=reduced_cubic_fc*au2cmi

return
end subroutine convert_cubic_to_reduced_cubic

! ****************************************
! ****************************************
! ****************************************
!
! Convert quartic force constants to reduced quartic force constants (divided by 
! product of vibrational frequencies of the involved modes, to the power of 1/4) 
!
! Expects normal mode semi-diagonal quartic force constants in a.u. (Hartree/(bohr^4*m_el^2))
! and eigenvalues of mass-weighted Hessian in a.u. ((atomic units of time)^(-2))
! 
! Outputs reduced quartic force constants in cm^-1
! 
! CAUTION!
! Negative eigenvalues are treated as if they were positive!
! Special VPT2 formulas then have to be used for saddle points!
!

subroutine convert_quartic_to_reduced_quartic(neff,quartic_fc,eigenvalues,reduced_quartic_fc)
use dlf_constants
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
integer, intent(in) :: neff
real(rk), intent(in), dimension(neff,neff,neff) :: quartic_fc
real(rk), intent(in), dimension(neff)           :: eigenvalues
real(rk), intent(out),dimension(neff,neff,neff) :: reduced_quartic_fc

integer :: i, j, k
real(rk) :: au2cmi

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [neff, &
     &     size(eigenvalues), &
     &     size(quartic_fc,dim=1), size(quartic_fc,dim=2), &
     &     size(quartic_fc,dim=3), &
     &     size(reduced_quartic_fc,dim=1), size(reduced_quartic_fc,dim=2), &
     &     size(reduced_quartic_fc,dim=3)  ], &
     &  'dim. mismatch, convert_quartic_to_reduced_quartic')
  write(stdout,*) 'convert_quartic_to_reduced_quartic: array shape check successful.'
#endif

call dlf_constants_get("CM_INV_FROM_HARTREE",au2cmi)

do i=1,neff
  do j=1,neff
    do k=1,neff
      !reduced_quartic_fc(i,j,k)=quartic_fc(i,j,k)/sqrt(eigenvalues(i)*sqrt(eigenvalues(j)*eigenvalues(k)))
      reduced_quartic_fc(i,j,k)=quartic_fc(i,j,k)/sqrt(sqrt(abs(eigenvalues(i)*eigenvalues(i)*eigenvalues(j)*eigenvalues(k))))
    enddo
  enddo
enddo

! Convert from Hartree to cm^-1
reduced_quartic_fc=reduced_quartic_fc*au2cmi

return
end subroutine convert_quartic_to_reduced_quartic

! ****************************************
! ****************************************
! ****************************************

!!! Expects normal mode cubic force constants in a.u. (Hartree/(bohr³*m_el^(3/2)))
!!! and eigenvalues of mass-weighted Hessian in a.u. ((atomic units of time)^(-2))
!!!
!!! Outputs reduced cubic force constants in cm^-1
!!
!!subroutine convert_cubic_to_reduced_cubic(neff,cubic_fc,eigenvalues,reduced_cubic_fc)
!!implicit none
!!integer, intent(in) :: neff
!!real(rk), intent(in), dimension(neff,neff,neff) :: cubic_fc
!!real(rk), intent(in), dimension(neff)           :: eigenvalues
!!real(rk), intent(out),dimension(neff,neff,neff) :: reduced_cubic_fc
!!
!!real(rk), dimension(neff) :: eigenvalues_tmp
!!real(rk), dimension(neff,neff,neff) :: cubic_fc_tmp
!!integer :: i, j, k
!!
!!! Convert cubic force constants from a.u. (Hartree/(bohr³*m_el^(3/2))) to SI units (J/(m³*kg(3/2)))
!!
!!cubic_fc_tmp=cubic_fc*hartree2joule
!!cubic_fc_tmp=cubic_fc_tmp/(bohr2meter**3)
!!cubic_fc_tmp=cubic_fc_tmp/(electron_mass_si**(1.5_rk))
!!
!!! Convert eigenvalues from a.u. to SI units (s**(-2))
!!eigenvalues_tmp=eigenvalues/(autime2seconds**2)
!!
!!do i=1,neff
!!  do j=1,neff
!!    do k=1,neff
!!      reduced_cubic_fc(i,j,k)=cubic_fc_tmp(i,j,k)/sqrt(sqrt(eigenvalues_tmp(i)*eigenvalues_tmp(j)*eigenvalues_tmp(k)))
!!    enddo
!!  enddo
!!enddo
!!
!!! Additional factor hbar**(3/2)
!!
!!reduced_cubic_fc=reduced_cubic_fc*hbar_si**1.5_rk
!!
!!! Convert from J to cm^-1
!!reduced_cubic_fc=reduced_cubic_fc/hplanck_si/speed_of_light_si/100._rk
!!
!!return
!!end subroutine convert_cubic_to_reduced_cubic

! ****************************************
! ****************************************
! ****************************************

! Finite difference Hessian (from gradients), done in 
! Cartesian coordinates (i.e., elongations in 3N directions)

subroutine differentiate_gradients_cartesian(grad_routine,nvar,coord0, &
                                & hessian,deltaQdiff,silent_in)
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
  use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
  end subroutine grad_routine
end interface

integer, intent(in) :: nvar
real(rk), intent(in), dimension(nvar) :: coord0
real(rk), intent(in) :: deltaQdiff
real(rk), intent(out),dimension(nvar,nvar) :: hessian
logical, intent(in),optional :: silent_in

real(rk), dimension(nvar) :: grad0, gradplus, gradminus
real(rk), dimension(nvar) :: coordtmp

real(rk) :: inv2del,energy,elem
integer  :: istat,gradcount,i,j,iimage,kiter,gradtotal,nat
logical  :: silent

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
  call checkeq( [nvar, &
     &     size(coord0), &
     &     size(hessian,dim=1), size(hessian,dim=2) ], &
     &  'dim. mismatch, differentiate_gradients_cartesian')
  write(stdout,*) 'differentiate_gradients_cartesian: array shape check successful.'
#endif

silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif

hessian=0._rk

nat=nvar/3
gradtotal=6*nat+1
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~Numerical~Differentiation~of~Analytical~Gradients~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of gradient calls: ', gradtotal
if(.not.silent) write(stdout,'(A)') ''
gradcount=1
if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'

! Calculate and save central gradient
call grad_routine(nvar,coord0,energy,grad0,iimage,kiter,istat)

! Iterate over Cartesian coordinates
do i=1,3*nat
  inv2del=0.5_rk/deltaQdiff
  coordtmp(:)=coord0(:)
  coordtmp(i)=coord0(i)+deltaQdiff
  gradcount=gradcount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'
  call grad_routine(nvar,coordtmp,energy,gradplus,iimage,kiter,istat)
  coordtmp(:)=coord0(:)
  coordtmp(i)=coord0(i)-deltaQdiff
  gradcount=gradcount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'
  call grad_routine(nvar,coordtmp,energy,gradminus,iimage,kiter,istat)
  do j=1,3*nat
    hessian(i,j)=hessian(i,j)+inv2del*(gradplus(j)-gradminus(j))
  enddo
enddo

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

! Symmetrize

do i=1,3*nat
  do j=1,i-1
    elem=0.5_rk*(hessian(i,j)+hessian(j,i))
    hessian(i,j)=elem
    hessian(j,i)=elem
  enddo
enddo

return
end subroutine differentiate_gradients_cartesian

! ****************************************
! ****************************************
! ****************************************

! Finite difference Hessian (from gradients), done in internal (Z matrix)
! coordinates

subroutine differentiate_gradients_internal(grad_routine,nat,nvar,&
                &  intn,nbonds,nbangles,ntorsions,def,bond_vals0, &
                &  angle_vals0,tors_vals0,cart0,mv,hessian,gradint0,dQ_bond,dQ_ang, &
                &  dQ_tors,silent_in)
use dlf_vpt2_utility, only: error_print
use dlf_vpt2_intcoord, only: int_to_cart, grad_c2i, generate_B_C_ridders, generate_inverse
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
  use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
  end subroutine grad_routine
end interface

integer, intent(in) :: nat,nvar,intn,nbonds,nbangles,ntorsions
integer, intent(in), dimension(intn,4) :: def
real(rk), intent(in), dimension(nbonds)    :: bond_vals0
real(rk), intent(in), dimension(nbangles)  :: angle_vals0
real(rk), intent(in), dimension(ntorsions) :: tors_vals0
real(rk), intent(in), dimension(nvar) :: cart0
real(rk), intent(in), dimension(nat) :: mv
real(rk), intent(in) :: dQ_bond, dQ_ang, dQ_tors
real(rk), intent(out),dimension(intn,intn) :: hessian
real(rk), intent(out), dimension(intn) :: gradint0
logical, intent(in),optional :: silent_in

real(rk), parameter :: cartconvtol=1.e-15_rk
real(rk), dimension(intn) :: gradintplus, gradintminus
real(rk), dimension(intn) :: intcoordtmp,intcoord0
real(rk), dimension(nvar) :: carttmp
real(rk), dimension(nvar) :: gradcart0,gradcartplus,gradcartminus
real(rk), dimension(nbonds)    :: bond_vals_tmp
real(rk), dimension(nbangles)  :: angle_vals_tmp
real(rk), dimension(ntorsions) :: tors_vals_tmp
character(len=1), dimension(intn) :: typemask
real(rk), dimension(3*nat,intn)        :: A
real(rk), dimension(intn,3*nat)        :: B
real(rk), dimension(3*nat,3*nat,intn)  :: C

real(rk) :: energy,elem,inv2del,dQ
integer  :: istat,gradcount,i,j,iimage,kiter,gradtotal
logical  :: silent

integer  :: bonds(nbonds,2)
integer  :: bangles(nbangles,3)
integer  :: torsions(ntorsions,4)

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nat, &
     &     size(mv) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [nvar, &
     &     size(cart0) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [intn, &
     &     size(def,dim=1),size(gradint0), &
     &     size(hessian,dim=1),size(hessian,dim=2) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [4, &
     &     size(def,dim=2) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [nbonds, &
     &     size(bond_vals0) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [nbangles, &
     &     size(angle_vals0) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
call checkeq( [ntorsions, &
     &     size(tors_vals0) ], &
     &  'dim. mismatch, differentiate_gradients_internal')
  write(stdout,*) 'differentiate_gradients_internal: array shape check successful.'
#endif

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)

silent=.false.
if (present(silent_in)) then
  silent=silent_in
endif

hessian=0._rk

do i=1,nbonds
  typemask(i)='r'
enddo
do i=nbonds+1,nbonds+nbangles
  typemask(i)='a'
enddo
do i=nbonds+nbangles+1,nbonds+nbangles+ntorsions
  typemask(i)='d'
enddo

gradtotal=2*intn+1
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~Numerical~Differentiation~of~Analytical~Gradients~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') ''
if(.not.silent) write(stdout,'(A,I0)') 'Total number of gradient calls: ', gradtotal
if(.not.silent) write(stdout,'(A)') ''
gradcount=1
if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'

! Calculate, transform and save central gradient

intcoord0(1:nbonds)                                    = bond_vals0(1:nbonds)
intcoord0(nbonds+1:nbonds+nbangles)                    = angle_vals0(1:nbangles)
intcoord0(nbonds+nbangles+1:nbonds+nbangles+ntorsions) = tors_vals0(1:ntorsions)

call grad_routine(nvar,cart0,energy,gradcart0,iimage,kiter,istat)
call generate_B_C_ridders(nat,intn,cart0,nbonds,nbangles, &
                           & ntorsions,bonds,bangles,torsions,B,C)
A = generate_inverse(nat,intn,B,mv)
call grad_c2i(nat,intn,gradcart0,gradint0,A)

! Iterate over internal coordinates
do i=1,intn
  if     (typemask(i)=='r') then
    dQ=dQ_bond
  elseif (typemask(i)=='a') then
    dQ=dQ_ang
  elseif (typemask(i)=='d') then
    dQ=dQ_tors
  else
    call error_print('differentiate_gradients_internal: inconsistent typemask')
  endif
  inv2del=1._rk/(2._rk*dQ)
  intcoordtmp(:)=intcoord0(:)
  intcoordtmp(i)=intcoord0(i)+dQ
  bond_vals_tmp(1:nbonds)     = intcoordtmp(1:nbonds)
  angle_vals_tmp(1:nbangles)  = intcoordtmp(nbonds+1:nbonds+nbangles)
  tors_vals_tmp(1:ntorsions)  = intcoordtmp(nbonds+nbangles+1:nbonds+nbangles+ntorsions)
  call int_to_cart(nat,intn,nbonds,nbangles,ntorsions,def,cart0,mv, &
                   &   bond_vals_tmp,angle_vals_tmp, &
                   &   tors_vals_tmp,carttmp,cartconvtol)
  gradcount=gradcount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'
  call grad_routine(nvar,carttmp,energy,gradcartplus,iimage,kiter,istat)
  call generate_B_C_ridders(nat,intn,carttmp,nbonds,nbangles, &
                           & ntorsions,bonds,bangles,torsions,B,C)
  A = generate_inverse(nat,intn,B,mv)
  call grad_c2i(nat,intn,gradcartplus,gradintplus,A)
  intcoordtmp(:)=intcoord0(:)
  intcoordtmp(i)=intcoord0(i)-dQ
  bond_vals_tmp(1:nbonds)     = intcoordtmp(1:nbonds)
  angle_vals_tmp(1:nbangles)  = intcoordtmp(nbonds+1:nbonds+nbangles)
  tors_vals_tmp(1:ntorsions)  = intcoordtmp(nbonds+nbangles+1:nbonds+nbangles+ntorsions)
  call int_to_cart(nat,intn,nbonds,nbangles,ntorsions,def,cart0,mv, &
                   &   bond_vals_tmp,angle_vals_tmp, &
                   &   tors_vals_tmp,carttmp,cartconvtol)
  gradcount=gradcount+1
  if(.not.silent) write(stdout,'(A,I0,A,I0,A)') 'Computing gradient ', gradcount, ' of ', gradtotal, '...'
  call grad_routine(nvar,carttmp,energy,gradcartminus,iimage,kiter,istat)
  call generate_B_C_ridders(nat,intn,carttmp,nbonds,nbangles, &
                           & ntorsions,bonds,bangles,torsions,B,C)
  A = generate_inverse(nat,intn,B,mv)
  call grad_c2i(nat,intn,gradcartminus,gradintminus,A)
  do j=1,intn
    hessian(i,j)=hessian(i,j)+inv2del*(gradintplus(j)-gradintminus(j))
  enddo
enddo

if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~     Numerical Differentiation Done!         ~~~~~~~~~~'
if(.not.silent) write(stdout,'(A)') '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

! Symmetrize

do i=1,intn
  do j=1,i-1
    elem=0.5_rk*(hessian(i,j)+hessian(j,i))
    hessian(i,j)=elem
    hessian(j,i)=elem
  enddo
enddo

return
end subroutine differentiate_gradients_internal

! ****************************************
! ****************************************
! ****************************************

! Finite-difference Hessian using gradients, via coordinate 
! transformation to curvilinear (Z matrix) coordinates
! i.e. the normal mode elongations are done in curvilinear coordinates
! and the resulting Hessian is transformed back to Cartesians

subroutine differentiate_gradients_cartesian_via_internal(grad_routine,nat,nvar,coord0, &
                                & mv,hess_cartesian,deltaQdiff,silent_in)
use dlf_vpt2_utility, only: matrix_output,vector_output
use dlf_vpt2_intcoord, only: read_int_coord_def, hess_i2c, cart_to_int, & 
                    & generate_B_C_ridders, generate_inverse, int_punch
use dlf_allocate, only: allocate, deallocate
#ifdef CHECK_ACTUAL_ARRAY_SHAPES
use dlf_vpt2_utility, only: checkeq
#endif
implicit none
interface
subroutine grad_routine(nvar,coords,energy,gradient,iimage,kiter,status)
  use dlf_parameter_module, only: rk
    implicit none
    integer   ,intent(in)    :: nvar
    real(rk)  ,intent(in)    :: coords(nvar)
    real(rk)  ,intent(out)   :: energy
    real(rk)  ,intent(out)   :: gradient(nvar)
    integer   ,intent(in)    :: iimage
    integer   ,intent(in)    :: kiter
    integer   ,intent(out)   :: status
  end subroutine grad_routine
end interface

integer, intent(in) :: nat,nvar
real(rk), intent(in), dimension(nvar) :: coord0
real(rk), intent(in), dimension(nat)  :: mv
real(rk), intent(in) :: deltaQdiff
real(rk), intent(out),dimension(nvar,nvar) :: hess_cartesian
logical, intent(in),optional :: silent_in

integer, parameter  :: nzero=6
real(rk), parameter :: scal_ang=1.0_rk, scal_tors=1.0_rk
integer :: intn,nbonds,nbangles,ntorsions
integer, allocatable, dimension(:,:)    :: def
real(rk), allocatable, dimension(:)     :: bond_vals0, angle_vals0, tors_vals0
real(rk), allocatable, dimension(:)     :: grad_internal
real(rk), allocatable, dimension(:,:)   :: hess_internal,A,B
real(rk), allocatable, dimension(:,:,:) :: C
integer,allocatable,dimension(:,:) :: bonds
integer,allocatable,dimension(:,:) :: bangles
integer,allocatable,dimension(:,:) :: torsions
real(rk) :: dQ_bond,dQ_ang,dQ_tors

#ifdef CHECK_ACTUAL_ARRAY_SHAPES
call checkeq( [nvar, &
     &     size(coord0), &
     &     size(hess_cartesian,dim=1), size(hess_cartesian,dim=2) ], &
     &  'dim. mismatch, differentiate_gradients_cartesian_via_internal')
call checkeq( [nat, &
     &     size(mv) ], &
     &  'dim. mismatch, differentiate_gradients_cartesian_via_internal')
  write(stdout,*) 'differentiate_gradients_cartesian_via_internal: array shape check successful.'
#endif

dQ_bond=deltaQdiff
dQ_ang =deltaQdiff*scal_ang
dQ_tors=deltaQdiff*scal_tors

intn=nvar-nzero
call allocate(def,intn,4)
call read_int_coord_def(intn,nbonds,nbangles,ntorsions,def)
call allocate(bond_vals0,nbonds)
call allocate(angle_vals0,nbangles)
call allocate(tors_vals0,ntorsions)
call allocate(grad_internal,intn)
call allocate(hess_internal,intn,intn)
call allocate(A,3*nat,intn)
call allocate(B,intn,3*nat)
call allocate(C,3*nat,3*nat,intn)

call cart_to_int(nat,intn,nbonds,nbangles,ntorsions,def,coord0,bond_vals0,angle_vals0,tors_vals0)
call int_punch(nat,intn,nbonds,nbangles,ntorsions,def,bond_vals0,angle_vals0,tors_vals0,6)

call differentiate_gradients_internal(grad_routine,nat,nvar,&
                &  intn,nbonds,nbangles,ntorsions,def,bond_vals0, &
                &  angle_vals0,tors_vals0,coord0,mv,hess_internal,&
                &  grad_internal,dQ_bond,dQ_ang,dQ_tors,silent_in)

call matrix_output(hess_internal,stdout,'ES20.12','Hessian (internal)')
call vector_output(grad_internal,stdout,'ES20.12','Gradient (internal)')

call allocate(bonds,nbonds,2)
call allocate(bangles,nbangles,3)
call allocate(torsions,ntorsions,4)

bonds(1:nbonds,1:2)       = def(1:nbonds,1:2)
bangles(1:nbangles,1:3)   = def(nbonds+1:nbonds+nbangles,1:3)
torsions(1:ntorsions,1:4) = def(nbonds+nbangles+1:nbonds+nbangles+ntorsions,1:4)
call generate_B_C_ridders(nat,intn,coord0,nbonds,nbangles,ntorsions,bonds,bangles,torsions,B,C)
A=generate_inverse(nat,intn,B,mv)
call hess_i2c(nat,intn,hess_internal,hess_cartesian,B,C,grad_internal)

call deallocate(torsions)
call deallocate(bangles)
call deallocate(bonds)
call deallocate(C)
call deallocate(B)
call deallocate(A)
call deallocate(hess_internal)
call deallocate(grad_internal)
call deallocate(tors_vals0)
call deallocate(angle_vals0)
call deallocate(bond_vals0)
call deallocate(def)
return
end subroutine differentiate_gradients_cartesian_via_internal

! ****************************************
! ****************************************
! ****************************************

end module dlf_vpt2_hess_deriv


