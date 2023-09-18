module dlf_vpt2_utility
use dlf_parameter_module
!use dlf_global, only: glob, stdout
implicit none

logical, save :: redo_energy_option=.true.

private :: C_rename

! Interface for C function rename (GNU subroutine is not portable)
! Not directly callable from Fortran code, use wrapper routine
! f_rename instead; same applies to C_remove/f_remove
interface
  function C_rename (oldname,newname) bind (C, name='rename')
    use, intrinsic :: iso_c_binding, only: C_INT, C_CHAR
    implicit none
    integer(C_INT) :: C_rename
    character(kind=C_CHAR), intent(in) :: oldname(*)
    character(kind=C_CHAR), intent(in) :: newname(*)
  end function C_rename
end interface
interface
  function C_remove (filename) bind (C, name='remove')
    use, intrinsic :: iso_c_binding, only: C_INT, C_CHAR
    implicit none
    integer(C_INT) :: C_remove
    character(kind=C_CHAR), intent(in) :: filename(*)
  end function C_remove
end interface

! Explicit MPI interfaces and wrapping routines
! This makes it easier to broadcast, scatter and gather 
! data via MPI. At the same time, due to the explicit interfaces,
! the compiler can catch bugs that would be caused by calling 
! the routines in a wrong way.

interface dlf_gl_bcast
  subroutine dlf_global_real_bcast(a,n,iproc)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: iproc, n
    real(rk), dimension(n) :: a
  end subroutine dlf_global_real_bcast
  subroutine dlf_global_int_bcast(a,n,iproc)
    implicit none
    integer :: iproc, n
    integer, dimension(n) :: a
  end subroutine dlf_global_int_bcast
  subroutine dlf_global_log_bcast(a,n,iproc)
    implicit none
    integer :: iproc, n
    logical, dimension(n) :: a
  end subroutine dlf_global_log_bcast
  subroutine dlf_global_real_bcast_rank0(a,iproc)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: iproc
    real(rk),target :: a
  end subroutine dlf_global_real_bcast_rank0
  subroutine dlf_global_int_bcast_rank0(a,iproc)
    implicit none
    integer :: iproc
    integer,target :: a
  end subroutine dlf_global_int_bcast_rank0
  subroutine dlf_global_log_bcast_rank0(a,iproc)
    implicit none
    integer :: iproc
    logical,target :: a
  end subroutine dlf_global_log_bcast_rank0
  subroutine dlf_global_char_bcast_rank0(charvar,iproc)
    implicit none
    character(len=*) :: charvar
    integer :: iproc
  end subroutine dlf_global_char_bcast_rank0
  subroutine dlf_global_real_bcast_rank2(a,n1,n2,iproc)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: iproc, n1, n2
    real(rk), dimension(n1,n2), target  :: a
  end subroutine dlf_global_real_bcast_rank2
  subroutine dlf_global_int_bcast_rank2(a,n1,n2,iproc)
    implicit none
    integer :: iproc, n1, n2
    integer, dimension(n1,n2), target  :: a
  end subroutine dlf_global_int_bcast_rank2
  subroutine dlf_global_log_bcast_rank2(a,n1,n2,iproc)
    implicit none
    integer :: iproc, n1, n2
    logical, dimension(n1,n2), target  :: a
  end subroutine dlf_global_log_bcast_rank2
  subroutine dlf_global_real_bcast_rank3(a,n1,n2,n3,iproc)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: iproc, n1, n2, n3
    real(rk), dimension(n1,n2,n3), target  :: a
  end subroutine dlf_global_real_bcast_rank3
  subroutine dlf_global_int_bcast_rank3(a,n1,n2,n3,iproc)
    implicit none
    integer :: iproc, n1, n2, n3
    integer, dimension(n1,n2,n3), target  :: a
  end subroutine dlf_global_int_bcast_rank3
  subroutine dlf_global_log_bcast_rank3(a,n1,n2,n3,iproc)
    implicit none
    integer :: iproc, n1, n2, n3
    logical, dimension(n1,n2,n3), target  :: a
  end subroutine dlf_global_log_bcast_rank3
end interface dlf_gl_bcast

interface dlf_gl_scatter
  subroutine dlf_global_real_scatter_flat(aflat,n,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    real(rk), dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
    real(rk), dimension(n) :: b
  end subroutine dlf_global_real_scatter_flat
  subroutine dlf_global_int_scatter_flat(aflat,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    integer, dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
    integer, dimension(n) :: b
  end subroutine dlf_global_int_scatter_flat
  subroutine dlf_global_log_scatter_flat(aflat,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    logical, dimension(merge(n*m,0,glob%iam==iproc)) :: aflat
    logical, dimension(n) :: b
  end subroutine dlf_global_log_scatter_flat
  subroutine dlf_global_real_scatter_rank0(a,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: iproc, m
    real(rk), dimension(merge(m,0,glob%iam==iproc)) :: a
    real(rk),target :: b
  end subroutine dlf_global_real_scatter_rank0
  subroutine dlf_global_int_scatter_rank0(a,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, m
    integer, dimension(merge(m,0,glob%iam==iproc)) :: a
    integer,target :: b
  end subroutine dlf_global_int_scatter_rank0
  subroutine dlf_global_log_scatter_rank0(a,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, m
    logical, dimension(merge(m,0,glob%iam==iproc)) :: a
    logical,target :: b
  end subroutine dlf_global_log_scatter_rank0
  subroutine dlf_global_real_scatter_rank1(a,n,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    real(rk), dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                      & target  :: a
    real(rk), dimension(n) :: b
  end subroutine dlf_global_real_scatter_rank1
  subroutine dlf_global_int_scatter_rank1(a,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    integer, dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                      & target  :: a
    integer, dimension(n) :: b
  end subroutine dlf_global_int_scatter_rank1
  subroutine dlf_global_log_scatter_rank1(a,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n, m
    logical, dimension(merge(n,0,glob%iam==iproc),merge(m,0,glob%iam==iproc)), &
                      & target  :: a
    logical, dimension(n) :: b
  end subroutine dlf_global_log_scatter_rank1
  subroutine dlf_global_real_scatter_rank2(a,n1,n2,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, m
    real(rk), dimension(merge(n1,0,glob%iam==iproc), &
                      & merge(n2,0,glob%iam==iproc), &
                      & merge(m,0,glob%iam==iproc)), &
                      & target  :: a
    real(rk), dimension(n1,n2) :: b
  end subroutine dlf_global_real_scatter_rank2
  subroutine dlf_global_int_scatter_rank2(a,n1,n2,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, m
    integer, dimension(merge(n1,0,glob%iam==iproc), &
                     & merge(n2,0,glob%iam==iproc), &
                     & merge(m,0,glob%iam==iproc)), &
                     & target  :: a
    integer, dimension(n1,n2) :: b
  end subroutine dlf_global_int_scatter_rank2
  subroutine dlf_global_log_scatter_rank2(a,n1,n2,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, m
    logical, dimension(merge(n1,0,glob%iam==iproc), &
                     & merge(n2,0,glob%iam==iproc), &
                     & merge(m,0,glob%iam==iproc)), &
                     & target  :: a
    logical, dimension(n1,n2) :: b
  end subroutine dlf_global_log_scatter_rank2
  subroutine dlf_global_real_scatter_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, n3, m
    real(rk), dimension(merge(n1,0,glob%iam==iproc), &
                      & merge(n2,0,glob%iam==iproc), &
                      & merge(n3,0,glob%iam==iproc), &
                      & merge(m,0,glob%iam==iproc)), &
                      & target  :: a
    real(rk), dimension(n1,n2,n3) :: b
  end subroutine dlf_global_real_scatter_rank3
  subroutine dlf_global_int_scatter_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, n3, m
    integer, dimension(merge(n1,0,glob%iam==iproc), &
                     & merge(n2,0,glob%iam==iproc), &
                     & merge(n3,0,glob%iam==iproc), &
                     & merge(m,0,glob%iam==iproc)), &
                     & target  :: a
    integer, dimension(n1,n2,n3) :: b
  end subroutine dlf_global_int_scatter_rank3
  subroutine dlf_global_log_scatter_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: iproc, n1, n2, n3, m
    logical, dimension(merge(n1,0,glob%iam==iproc), &
                     & merge(n2,0,glob%iam==iproc), &
                     & merge(n3,0,glob%iam==iproc), &
                     & merge(m,0,glob%iam==iproc)), &
                     & target  :: a
    logical, dimension(n1,n2,n3) :: b
  end subroutine dlf_global_log_scatter_rank3
end interface dlf_gl_scatter

interface dlf_gl_gather
  subroutine dlf_global_real_gather_flat(a,n,bflat,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    real(rk), dimension(n) :: a
    real(rk), dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
  end subroutine dlf_global_real_gather_flat
  subroutine dlf_global_int_gather_flat(a,n,bflat,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    integer, dimension(n) :: a
    integer, dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
  end subroutine dlf_global_int_gather_flat
  subroutine dlf_global_log_gather_flat(a,n,bflat,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    logical, dimension(n) :: a
    logical, dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
  end subroutine dlf_global_log_gather_flat
  subroutine dlf_global_char_gather_flat(a,n,bflat,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    character(len=*), dimension(n) :: a
    character(len=*), dimension(merge(n*m,0,glob%iam==iproc)) :: bflat
  end subroutine dlf_global_char_gather_flat
  subroutine dlf_global_real_gather_rank0(a,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: m, iproc
    real(rk), target :: a
    real(rk), dimension(merge(m,0,glob%iam==iproc)) :: b
  end subroutine dlf_global_real_gather_rank0
  subroutine dlf_global_int_gather_rank0(a,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: m, iproc
    integer, target :: a
    integer, dimension(merge(m,0,glob%iam==iproc)) :: b
  end subroutine dlf_global_int_gather_rank0
  subroutine dlf_global_log_gather_rank0(a,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: m, iproc
    logical, target :: a
    logical, dimension(merge(m,0,glob%iam==iproc)) :: b
  end subroutine dlf_global_log_gather_rank0
  subroutine dlf_global_char_gather_rank0(a,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: m, iproc
    character(len=*) :: a
    character(len=*), dimension(merge(m,0,glob%iam==iproc)) :: b
  end subroutine dlf_global_char_gather_rank0
  subroutine dlf_global_real_gather_rank1(a,n,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    real(rk), dimension(n) :: a
    real(rk), dimension(merge(n,0,glob%iam==iproc),  & 
                  &     merge(m,0,glob%iam==iproc)), &
                  &     target :: b
  end subroutine dlf_global_real_gather_rank1
  subroutine dlf_global_int_gather_rank1(a,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    integer, dimension(n) :: a
    integer, dimension(merge(n,0,glob%iam==iproc),  & 
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_int_gather_rank1
  subroutine dlf_global_log_gather_rank1(a,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    logical, dimension(n) :: a
    logical, dimension(merge(n,0,glob%iam==iproc),  & 
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_log_gather_rank1
  subroutine dlf_global_char_gather_rank1(a,n,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n, m, iproc
    character(len=*), dimension(n) :: a
    character(len=*), dimension(merge(n,0,glob%iam==iproc),  & 
                     &          merge(m,0,glob%iam==iproc)), &
                     &          target :: b
  end subroutine dlf_global_char_gather_rank1
  subroutine dlf_global_real_gather_rank2(a,n1,n2,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, m, iproc
    real(rk), dimension(n1,n2) :: a
    real(rk), dimension(merge(n1,0,glob%iam==iproc), & 
                  &     merge(n2,0,glob%iam==iproc), &
                  &     merge(m,0,glob%iam==iproc)), &
                  &     target :: b
  end subroutine dlf_global_real_gather_rank2
  subroutine dlf_global_int_gather_rank2(a,n1,n2,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, m, iproc
    integer, dimension(n1,n2) :: a
    integer, dimension(merge(n1,0,glob%iam==iproc), & 
                  &    merge(n2,0,glob%iam==iproc), &
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_int_gather_rank2
  subroutine dlf_global_log_gather_rank2(a,n1,n2,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, m, iproc
    logical, dimension(n1,n2) :: a
    logical, dimension(merge(n1,0,glob%iam==iproc), & 
                  &    merge(n2,0,glob%iam==iproc), &
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_log_gather_rank2
  subroutine dlf_global_real_gather_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_parameter_module, only: rk
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, n3, m, iproc
    real(rk), dimension(n1,n2,n3) :: a
    real(rk), dimension(merge(n1,0,glob%iam==iproc), & 
                  &     merge(n2,0,glob%iam==iproc), &
                  &     merge(n3,0,glob%iam==iproc), &
                  &     merge(m,0,glob%iam==iproc)), &
                  &     target :: b
  end subroutine dlf_global_real_gather_rank3
  subroutine dlf_global_int_gather_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, n3, m, iproc
    integer, dimension(n1,n2,n3) :: a
    integer, dimension(merge(n1,0,glob%iam==iproc), & 
                  &    merge(n2,0,glob%iam==iproc), &
                  &    merge(n3,0,glob%iam==iproc), &
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_int_gather_rank3
  subroutine dlf_global_log_gather_rank3(a,n1,n2,n3,b,m,iproc)
    use dlf_global, only: glob
    implicit none
    integer :: n1, n2, n3, m, iproc
    logical, dimension(n1,n2,n3) :: a
    logical, dimension(merge(n1,0,glob%iam==iproc), & 
                  &    merge(n2,0,glob%iam==iproc), &
                  &    merge(n3,0,glob%iam==iproc), &
                  &    merge(m,0,glob%iam==iproc)), &
                  &    target :: b
  end subroutine dlf_global_log_gather_rank3
end interface dlf_gl_gather

interface dlf_gl_sum
  subroutine dlf_global_real_sum(a,n)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: n
    real(rk), dimension(n) :: a
  end subroutine dlf_global_real_sum
  subroutine dlf_global_int_sum(a,n)
    implicit none
    integer :: n
    integer, dimension(n) :: a
  end subroutine dlf_global_int_sum
  subroutine dlf_global_real_sum_rank0(a)
    use dlf_parameter_module, only: rk
    implicit none
    real(rk),target :: a
  end subroutine dlf_global_real_sum_rank0
  subroutine dlf_global_int_sum_rank0(a)
    implicit none
    integer,target :: a
  end subroutine dlf_global_int_sum_rank0
  subroutine dlf_global_real_sum_rank2(a,n1,n2)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: n1, n2
    real(rk), dimension(n1,n2), target  :: a
  end subroutine dlf_global_real_sum_rank2
  subroutine dlf_global_int_sum_rank2(a,n1,n2)
    implicit none
    integer :: n1, n2
    integer, dimension(n1,n2), target  :: a
  end subroutine dlf_global_int_sum_rank2
  subroutine dlf_global_real_sum_rank3(a,n1,n2,n3)
    use dlf_parameter_module, only: rk
    implicit none
    integer :: n1, n2, n3
    real(rk), dimension(n1,n2,n3), target  :: a
  end subroutine dlf_global_real_sum_rank3
  subroutine dlf_global_int_sum_rank3(a,n1,n2,n3)
    implicit none
    integer :: n1, n2, n3
    integer, dimension(n1,n2,n3), target  :: a
  end subroutine dlf_global_int_sum_rank3
end interface dlf_gl_sum

interface dlf_mpi_send
  subroutine dlf_mpi_send_real_rank1(sendbuff,n,target_rank,tag)
    use dlf_parameter_module, only: rk
    implicit none
    integer, intent(in) :: n, target_rank, tag
    real(rk),dimension(n), intent(in) :: sendbuff
  end subroutine dlf_mpi_send_real_rank1
  subroutine dlf_mpi_send_int_rank1(sendbuff,n,target_rank,tag)
    implicit none
    integer, intent(in) :: n, target_rank, tag
    integer,dimension(n), intent(in) :: sendbuff
  end subroutine dlf_mpi_send_int_rank1
  subroutine dlf_mpi_send_log_rank1(sendbuff,n,target_rank,tag)
    implicit none
    integer, intent(in) :: n, target_rank, tag
    logical,dimension(n), intent(in) :: sendbuff
  end subroutine dlf_mpi_send_log_rank1
  subroutine dlf_mpi_send_char_string(sendbuff,target_rank,tag)
    implicit none
    integer, intent(in) :: target_rank, tag
    character(len=*), intent(in) :: sendbuff
  end subroutine dlf_mpi_send_char_string
  subroutine dlf_mpi_send_real_rank0(sendbuff,target_rank,tag)
    use dlf_parameter_module, only: rk
    implicit none
    integer, intent(in) :: target_rank, tag
    real(rk), intent(in) :: sendbuff
  end subroutine dlf_mpi_send_real_rank0
  subroutine dlf_mpi_send_int_rank0(sendbuff,target_rank,tag)
    implicit none
    integer, intent(in) :: target_rank, tag
    integer, intent(in) :: sendbuff
  end subroutine dlf_mpi_send_int_rank0
  subroutine dlf_mpi_send_log_rank0(sendbuff,target_rank,tag)
    implicit none
    integer, intent(in) :: target_rank, tag
    logical, intent(in) :: sendbuff
  end subroutine dlf_mpi_send_log_rank0
end interface dlf_mpi_send

interface dlf_mpi_recv
  subroutine dlf_mpi_recv_real_rank1(recvbuff,n,source_rank,tag,recv_status)
    use dlf_parameter_module, only: rk
    implicit none
    integer, intent(in) :: n
    integer, intent(inout) :: source_rank, tag
    real(rk),dimension(n), intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_real_rank1
  subroutine dlf_mpi_recv_int_rank1(recvbuff,n,source_rank,tag,recv_status)
    implicit none
    integer, intent(in) :: n
    integer, intent(inout) :: source_rank, tag
    integer,dimension(n), intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_int_rank1
  subroutine dlf_mpi_recv_log_rank1(recvbuff,n,source_rank,tag,recv_status)
    implicit none
    integer, intent(in) :: n
    integer, intent(inout) :: source_rank, tag
    logical,dimension(n), intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_log_rank1
  subroutine dlf_mpi_recv_char_string(recvbuff,source_rank,tag,recv_status)
    implicit none
    integer, intent(inout) :: source_rank, tag
    character(len=*), intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_char_string
  subroutine dlf_mpi_recv_real_rank0(recvbuff,source_rank,tag,recv_status)
    use dlf_parameter_module, only: rk
    implicit none
    integer, intent(inout) :: source_rank, tag
    real(rk),intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_real_rank0
  subroutine dlf_mpi_recv_int_rank0(recvbuff,source_rank,tag,recv_status)
    implicit none
    integer, intent(inout) :: source_rank, tag
    integer,intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_int_rank0
  subroutine dlf_mpi_recv_log_rank0(recvbuff,source_rank,tag,recv_status)
    implicit none
    integer, intent(inout) :: source_rank, tag
    logical,intent(out) :: recvbuff
    integer, dimension(:), intent(out),optional :: recv_status
  end subroutine dlf_mpi_recv_log_rank0
end interface dlf_mpi_recv

interface checkeq
  module procedure checkeq_2arg, checkeq_3arg, checkeq_4arg, checkeq_narg
end interface

contains

!***************************************************
!***************************************************

! Fortran wrapping routine for the native C rename function

subroutine f_rename(oldname,newname,k_out)
  use, intrinsic :: iso_c_binding, only: C_INT, C_NULL_CHAR
  implicit none
  character(len=*), intent(in) :: oldname, newname
  integer, intent(out), optional :: k_out
  integer(kind=C_INT) :: k
  k= C_rename(TRIM(oldname)//C_NULL_CHAR,TRIM(newname)//C_NULL_CHAR)
  if (present(k_out)) then
    k_out=int(k)
  endif
  return
end subroutine f_rename

!***************************************************
!***************************************************

! Fortran wrapping routine for the native C remove (i.e. file delete) function

subroutine f_remove(filename,k_out)
  use, intrinsic :: iso_c_binding, only: C_INT, C_NULL_CHAR
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(out), optional :: k_out
  integer(kind=C_INT) :: k
  k= C_remove(TRIM(filename)//C_NULL_CHAR)
  if (present(k_out)) then
    k_out=int(k)
  endif
  return
end subroutine f_remove

!***************************************************
!***************************************************

! Check if two integer arguments are identical, and
! exit with an error message if not.

subroutine checkeq_2arg(n1,n2,text)
implicit none
integer, intent(in) :: n1,n2
character(*), intent(in) :: text

if (n1.ne.n2) then
  call error_print(text)
endif

return
end subroutine checkeq_2arg

!***************************************************
!***************************************************

! Check if 3 integer arguments are identical, and
! exit with an error message if not.

subroutine checkeq_3arg(n1,n2,n3,text)
implicit none
integer, intent(in) :: n1,n2,n3
character(*), intent(in) :: text

if (n1.ne.n2 .or. n1.ne.n3 .or. n2.ne.n3) then
  call error_print(text)
endif

return
end subroutine checkeq_3arg

!***************************************************
!***************************************************

! Check if 4 integer arguments are identical, and
! exit with an error message if not.

subroutine checkeq_4arg(n1,n2,n3,n4,text)
implicit none
integer, intent(in) :: n1,n2,n3,n4
character(*), intent(in) :: text

if (.not. (n1.eq.n2 .and. n1.eq.n3 .and. n1.eq.n4)) then
  call error_print(text)
endif

return
end subroutine checkeq_4arg

!***************************************************
!***************************************************

! Check if all n integer arguments are identical (stored in an array), and
! exit with an error message if not.

subroutine checkeq_narg(n,text)
implicit none
integer, intent(in), dimension(:) :: n
character(*), intent(in) :: text

if (.not. all(n(:).eq.n(1))) then
  call error_print(text)
endif

return
end subroutine checkeq_narg

!***************************************************
!***************************************************

! Swap data in scalar variables x and y

subroutine swapxy(x,y)
implicit none
real(rk), intent(inout) :: x,y
real(rk) :: tmp

tmp=x
x=y
y=tmp

return
end subroutine swapxy

!***************************************************
!***************************************************

! Swap data in rank-1 array variables v and w

subroutine swapvw(v,w)
implicit none
real(rk), intent(inout),dimension(:) :: v,w
real(rk),dimension(size(v)) :: tmp

call checkeq(size(v),size(w),'swapvw, dimension mismatch')

tmp=v
v=w
w=tmp

return
end subroutine swapvw

!***************************************************
!***************************************************

! Convert integer index to character representing the type of 
! optimization step

function step_type(i)
implicit none
integer, intent(in) :: i
character(1) :: step_type

if (i.eq.0) then
  step_type='-'
elseif (i.eq.1) then
  step_type='L'
elseif (i.eq.2) then
  step_type='P'
elseif (i.eq.3) then
  step_type='S'
elseif (i.eq.4) then
  step_type='O'
elseif (i.eq.5) then
  step_type='G'
else
  step_type=' '
endif

return
end function step_type

!***************************************************
!***************************************************

! Read 1-D array data from funit and save them in vector vec
! Two header lines (ignored) per array are assumed

subroutine vector_input(vec,funit,formtype)
implicit none
real(rk), intent(out), dimension(:) :: vec
integer, intent(in) :: funit
character(*), intent(in) :: formtype
integer :: n1,i

n1=size(vec)

read(funit,*)
read(funit,*)

do i=1,n1
  read(unit=funit,fmt='('//formtype//')') vec(i)
enddo

return
end subroutine vector_input

!***************************************************
!***************************************************

! Read mixed real/integer data from funit, save them in 
! realvec and intgvec
! Two header lines (ignored) per array are assumed

subroutine vector_inp_multi(realvec,intgvec,funit,formtype)
implicit none
real(rk), intent(out), dimension(:,:) :: realvec
integer, intent(out), dimension(:,:) :: intgvec
integer, intent(in) :: funit
character(*), intent(in) :: formtype

character(150) :: formt, realformt, intgformt
integer :: n1,i,j
integer :: nreal,nintg

nreal=size(realvec,2)
nintg=size(intgvec,2)

n1=size(realvec,1)

write(intgformt,'(I0,A)') nintg,'(I10)'
write(realformt,'(I0,A)') nreal,'('//formtype//')'
write(formt,'(A)') '('//trim(intgformt)//','//trim(realformt)//')'

read(funit,*)
read(funit,*)

do i=1,n1
  read(unit=funit,fmt=formt) (intgvec(i,j), j=1,nintg), (realvec(i,j), j=1,nreal)
enddo

return
end subroutine vector_inp_multi

!***************************************************
!***************************************************

! Read a rank-2 array (matrix) from funit, save it
! in variable mat
! Two header lines (ignored) per array are assumed

subroutine matrix_input(mat,funit,formtype)
implicit none
real(rk), intent(out), dimension(:,:) :: mat
integer, intent(in) :: funit
character(*), intent(in) :: formtype
integer :: n1,n2,i,j
character(150) :: formt

n1=size(mat,1)
n2=size(mat,2)

read(funit,*)
read(funit,*)

write(formt,'(A,I0,A)') '(',n2,'('//trim(formtype)//'))'

do i=1,n1
  read(unit=funit,fmt=formt) (mat(i,j), j=1,n2)
enddo

return
end subroutine matrix_input

!***************************************************
!***************************************************

! Write 1-D array data to funit, data source: vec
! Two header lines per array are created. 
! Title can be chosen by user, as well as format 
! specifier for the real data.

subroutine vector_output(vec,funit,formtype,title)
implicit none
real(rk), intent(in), dimension(:) :: vec
integer, intent(in) :: funit
character(*), intent(in) :: formtype
character(*), intent(in), optional :: title
integer :: n1,i

n1=size(vec)

if (present(title)) then
  if (title/='__BLANK__') then
    write(funit,'(A)') title
    write(funit,*) ''
  endif
else
  write(funit,*) ''
  write(funit,*) ''
endif

do i=1,n1
  write(unit=funit,fmt='('//formtype//')') vec(i)
enddo
if (title/='__BLANK__') write(funit,*)''
return
end subroutine vector_output

!***************************************************
!***************************************************

! Write mixed real/integer array data to funit, data sources: 
! realvec and intvec
! Two header lines per array are created.
! Title can be chosen by user, as well as format 
! specifier for the real data.

subroutine vector_out_multi(realvec,intgvec,funit,formtype,title)
implicit none
integer :: nreal,nintg
real(rk), intent(in), dimension(:,:) :: realvec
integer, intent(in), dimension(:,:) :: intgvec
integer, intent(in) :: funit
character(*), intent(in) :: formtype
character(*), intent(in), optional :: title
character(150) :: formt, realformt, intgformt
integer :: n1,i,j

nreal=size(realvec,2)
nintg=size(intgvec,2)

n1=size(realvec,1)

write(intgformt,'(I0,A)') nintg,'(I10)'
write(realformt,'(I0,A)') nreal,'('//formtype//')'

write(formt,'(A)') '('//trim(intgformt)//','//trim(realformt)//')'

if (present(title)) then
  write(funit,'(A)') title
  write(funit,*) ''
else
  write(funit,*) ''
  write(funit,*) ''
endif

do i=1,n1
  write(unit=funit,fmt=formt) (intgvec(i,j), j=1,nintg), (realvec(i,j), j=1,nreal)
enddo
write(funit,*)''
return
end subroutine vector_out_multi

!***************************************************
!***************************************************

! Write real rank-2 array (matrix) data to funit, data source: 
! mat.
! Two header lines per array are created.
! Title can be chosen by user, as well as format 
! specifier for the real data.

subroutine matrix_output(mat,funit,formtype,title)
use dlf_global, only: printl
implicit none
real(rk), intent(in), dimension(:,:) :: mat
integer, intent(in) :: funit
character(*), intent(in) :: formtype
character(*), intent(in), optional :: title
integer :: n1,n2,i,j
character(150) :: formt

! do not print VPT2 matrices for printl<4
if(printl<4) return

n1=size(mat,1)
n2=size(mat,2)

if (present(title)) then
  if (title/='__BLANK__') then
    write(funit,'(A)') title
    write(funit,*) ''
  endif
else
  write(funit,*) ''
  write(funit,*) ''
endif

write(formt,'(A,I0,A)') '(',n2,'('//trim(formtype)//'))'
!write(stdout,*)formt

do i=1,n1
  write(unit=funit,fmt=formt) (mat(i,j), j=1,n2)
enddo
if (title/='__BLANK__') write(funit,*)''
return
end subroutine matrix_output

!***************************************************
!***************************************************

!subroutine myerr(text,code)
!implicit none
!character(*), intent(in) :: text
!integer, intent(in) :: code
!
!write (stdout,*) 'Error! '//text
!write (stdout,*) 'Code: ', code
!
!!read(*,*)
!!call exit(code)
!!stop abs(code)
!stop 1
!
!end subroutine myerr

!***************************************************
!***************************************************

! Print user-specified error message, and exit
! gracefully (hopefully).

subroutine error_print(text)
use dlf_global, only: stdout
implicit none
character(*), intent(in) :: text

write (stdout,'(A)') 'Error! '//text
call dlf_error()

end subroutine error_print

!***************************************************
!***************************************************

! Remove garbage when reading Molpro Hessian lines
! from .out file

subroutine remove_junk(string)
implicit none
character(*), intent(inout) :: string

character(len(string))  :: newstring
character(1) :: singlechar
logical :: lastspace
integer :: i,stringstart

newstring=''
lastspace=.false.

do i=1,len(string)
  singlechar=string(i:i)
  if (isnumberorminus(singlechar) .and. lastspace) then
    stringstart=i
    exit
  endif
  if (singlechar==' ') then
    lastspace=.true.
  else
    lastspace=.false.
  endif
enddo

if (stringstart.eq.len(string)) then
  string=''
  return
else
  newstring=string(stringstart:len(string))
  string=adjustl(newstring)
  return
endif
end subroutine remove_junk

!***************************************************
!***************************************************

! Check if input character is a dash/minus or digit

function isnumberorminus(chara)
implicit none
logical :: isnumberorminus
character, intent(in) :: chara
integer :: i
isnumberorminus=.false.
i=iachar(chara)

if (i.ge.48 .and. i.le.57) isnumberorminus=.true.
if (i.eq.45) isnumberorminus=.true.

return
end function isnumberorminus

!***************************************************
!***************************************************

! Convert element symbol to atomic mass in a.m.u.
! Choose between isotope-abundance-weighted masses (default)
! or masses of the most abundant isotopes (controlled
! by the single_iso_in argument)

elemental function symb2mass(el,single_iso_in)
implicit none

real(rk) :: symb2mass
character(2), intent(in)   :: el
character(2) :: eladj,elup
logical,optional,intent(in) :: single_iso_in

logical :: single_iso

single_iso=.false.

if (present(single_iso_in)) then
  single_iso=single_iso_in
endif

eladj=adjustl(el)
elup=upcase(eladj(1:1))//upcase(eladj(2:2))

if (single_iso) then
  select case (elup)
  case ('H ')   
    symb2mass = 1.007825_rk
  case ('D ')   
    symb2mass = 2.014102_rk
  case ('T ')   
    symb2mass = 3.016049_rk
  case ('HE')  
    symb2mass = 4.002603_rk
  case ('LI')  
    symb2mass = 7.016004_rk
  case ('BE')  
    symb2mass = 9.012182_rk
  case ('B ')   
    symb2mass = 11.009305_rk
  case ('C ')   
    symb2mass = 12.000000_rk
  case ('N ')   
    symb2mass = 14.003074_rk
  case ('O ')   
    symb2mass = 15.994915_rk
  case ('F ')   
    symb2mass = 18.998403_rk
  case ('NE')  
    symb2mass = 19.992440_rk
  case ('NA')  
    symb2mass = 22.989770_rk
  case ('MG')  
    symb2mass = 23.985042_rk
  case ('AL')  
    symb2mass = 26.981538_rk
  case ('SI')  
    symb2mass = 27.976927_rk
  case ('P ')   
    symb2mass = 30.973762_rk
  case ('S ')   
    symb2mass = 31.972071_rk
  case ('CL')  
    symb2mass = 34.968853_rk
  case ('AR')  
    symb2mass = 39.962383_rk
  case ('K ')   
    symb2mass = 38.963707_rk
  case ('CA')  
    symb2mass = 39.962591_rk
  case ('SC')  
    symb2mass = 44.955910_rk
  case ('TI')  
    symb2mass = 47.947947_rk
  case ('V ')   
    symb2mass = 50.943964_rk
  case ('CR')  
    symb2mass = 51.940512_rk
  case ('MN')  
    symb2mass = 54.938050_rk
  case ('FE')  
    symb2mass = 55.934942_rk
  case ('CO')  
    symb2mass = 58.933200_rk
  case ('NI')  
    symb2mass = 57.935348_rk
  case ('CU')  
    symb2mass = 62.929601_rk
  case ('ZN')  
    symb2mass = 63.929147_rk
  case ('GA')  
    symb2mass = 68.925581_rk
  case ('GE')  
    symb2mass = 73.921178_rk
  case ('AS')  
    symb2mass = 74.921596_rk
  case ('SE')  
    symb2mass = 79.916522_rk
  case ('BR')  
    symb2mass = 78.918338_rk
  case ('KR')  
    symb2mass = 83.911507_rk
  case ('RB')  
    symb2mass = 84.911789_rk
  case ('SR')  
    symb2mass = 87.905614_rk
  case ('Y ')   
    symb2mass = 88.905848_rk
  case ('ZR')  
    symb2mass = 89.904704_rk
  case ('NB')  
    symb2mass = 92.906378_rk
  case ('MO')  
    symb2mass = 97.905408_rk
  case ('TC')  
    symb2mass = 97.907216_rk
  case ('RU')  
    symb2mass = 101.904350_rk
  case ('RH')  
    symb2mass = 102.905504_rk
  case ('PD')  
    symb2mass = 105.903483_rk
  case ('AG')  
    symb2mass = 106.905093_rk
  case ('CD')  
    symb2mass = 113.903358_rk
  case ('IN')  
    symb2mass = 114.903878_rk
  case ('SN')  
    symb2mass = 119.902197_rk
  case ('SB')  
    symb2mass = 120.903818_rk
  case ('TE')  
    symb2mass = 129.906223_rk
  case ('I ')   
    symb2mass = 126.904468_rk
  case ('XE')  
    symb2mass = 131.904154_rk
  case ('CS')  
    symb2mass = 132.905447_rk
  case ('BA')  
    symb2mass = 137.905241_rk
  case ('LA')  
    symb2mass = 138.906348_rk
  case ('CE')  
    symb2mass = 139.905434_rk
  case ('PR')  
    symb2mass = 140.907648_rk
  case ('ND')  
    symb2mass = 141.907719_rk
  case ('PM')  
    symb2mass = 144.912744_rk
  case ('SM')  
    symb2mass = 151.919728_rk
  case ('EU')  
    symb2mass = 152.921226_rk
  case ('GD')  
    symb2mass = 157.924101_rk
  case ('TB')  
    symb2mass = 158.925343_rk
  case ('DY')  
    symb2mass = 163.929171_rk
  case ('HO')  
    symb2mass = 164.930319_rk
  case ('ER')  
    symb2mass = 165.930290_rk
  case ('TM')  
    symb2mass = 168.934211_rk
  case ('YB')  
    symb2mass = 173.938858_rk
  case ('LU')  
    symb2mass = 174.940768_rk
  case ('HF')  
    symb2mass = 179.946549_rk
  case ('TA')  
    symb2mass = 180.947996_rk
  case ('W ')   
    symb2mass = 183.950933_rk
  case ('RE')  
    symb2mass = 186.955751_rk
  case ('OS')  
    symb2mass = 191.961479_rk
  case ('IR')  
    symb2mass = 192.962924_rk
  case ('PT')  
    symb2mass = 194.964774_rk
  case ('AU')  
    symb2mass = 196.966552_rk
  case ('HG')  
    symb2mass = 201.970626_rk
  case ('TL')  
    symb2mass = 204.974412_rk
  case ('PB')  
    symb2mass = 207.976636_rk
  case ('BI')  
    symb2mass = 208.980383_rk
  case ('PO')  
    symb2mass = 208.982416_rk
  case ('AT')  
    symb2mass = 209.987131_rk
  case ('RN')  
    symb2mass = 222.017570_rk
  case ('FR')  
    symb2mass = 223.019731_rk
  case ('RA')  
    symb2mass = 226.025403_rk
  case ('AC')  
    symb2mass = 227.027747_rk
  case ('TH')  
    symb2mass = 232.038050_rk
  case ('PA')  
    symb2mass = 231.035879_rk
  case ('U ')   
    symb2mass = 238.050783_rk
  case default 
    symb2mass = -7777777._rk
  end select
else
  select case (elup)
  case ('H ')   
    symb2mass = 1.00794_rk
  case ('D ')   
    symb2mass = 2.0141017778_rk
  case ('T ')   
    symb2mass = 3.0160492777_rk
  case ('HE')  
    symb2mass = 4.002602_rk
  case ('LI')  
    symb2mass = 6.941_rk
  case ('BE')  
    symb2mass = 9.012182_rk
  case ('B ')   
    symb2mass = 10.811_rk
  case ('C ')   
    symb2mass = 12.0107_rk
  case ('N ')   
    symb2mass = 14.0067_rk
  case ('O ')   
    symb2mass = 15.9994_rk
  case ('F ')   
    symb2mass = 18.9984032_rk
  case ('NE')  
    symb2mass = 20.1797_rk
  case ('NA')  
    symb2mass = 22.98976928_rk
  case ('MG')  
    symb2mass = 24.3050_rk
  case ('AL')  
    symb2mass = 26.9815386_rk
  case ('SI')  
    symb2mass = 28.0855_rk
  case ('P ')   
    symb2mass = 30.973762_rk
  case ('S ')   
    symb2mass = 32.065_rk
  case ('CL')  
    symb2mass = 35.453_rk
  case ('AR')  
    symb2mass = 39.948_rk
  case ('K ')   
    symb2mass = 39.0983_rk
  case ('CA')  
    symb2mass = 40.078_rk
  case ('SC')  
    symb2mass = 44.955912_rk
  case ('TI')  
    symb2mass = 47.867_rk
  case ('V ')   
    symb2mass = 50.9415_rk
  case ('CR')  
    symb2mass = 51.9961_rk
  case ('MN')  
    symb2mass = 54.938045_rk
  case ('FE')  
    symb2mass = 55.845_rk
  case ('CO')  
    symb2mass = 58.933195_rk
  case ('NI')  
    symb2mass = 58.6934_rk
  case ('CU')  
    symb2mass = 63.546_rk
  case ('ZN')  
    symb2mass = 65.38_rk
  case ('GA')  
    symb2mass = 69.723_rk
  case ('GE')  
    symb2mass = 72.64_rk
  case ('AS')  
    symb2mass = 74.92160_rk
  case ('SE')  
    symb2mass = 78.96_rk
  case ('BR')  
    symb2mass = 79.904_rk
  case ('KR')  
    symb2mass = 83.798_rk
  case ('RB')  
    symb2mass = 85.4678_rk
  case ('SR')  
    symb2mass = 87.62_rk
  case ('Y ')   
    symb2mass = 88.90585_rk
  case ('ZR')  
    symb2mass = 91.224_rk
  case ('NB')  
    symb2mass = 92.90638_rk
  case ('MO')  
    symb2mass = 95.96_rk
  case ('TC')  
    symb2mass = 98._rk
  case ('RU')  
    symb2mass = 101.07_rk
  case ('RH')  
    symb2mass = 102.90550_rk
  case ('PD')  
    symb2mass = 106.42_rk
  case ('AG')  
    symb2mass = 107.8682_rk
  case ('CD')  
    symb2mass = 112.411_rk
  case ('IN')  
    symb2mass = 114.818_rk
  case ('SN')  
    symb2mass = 118.710_rk
  case ('SB')  
    symb2mass = 121.760_rk
  case ('TE')  
    symb2mass = 127.60_rk
  case ('I ')   
    symb2mass = 126.90447_rk
  case ('XE')  
    symb2mass = 131.293_rk
  case ('CS')  
    symb2mass = 132.9054519_rk
  case ('BA')  
    symb2mass = 137.327_rk
  case ('LA')  
    symb2mass = 138.90547_rk
  case ('CE')  
    symb2mass = 140.116_rk
  case ('PR')  
    symb2mass = 140.90765_rk
  case ('ND')  
    symb2mass = 144.242_rk
  case ('PM')  
    symb2mass = 145._rk
  case ('SM')  
    symb2mass = 150.36_rk
  case ('EU')  
    symb2mass = 151.964_rk
  case ('GD')  
    symb2mass = 157.25_rk
  case ('TB')  
    symb2mass = 158.92535_rk
  case ('DY')  
    symb2mass = 162.500_rk
  case ('HO')  
    symb2mass = 164.93032_rk
  case ('ER')  
    symb2mass = 167.259_rk
  case ('TM')  
    symb2mass = 168.93421_rk
  case ('YB')  
    symb2mass = 173.054_rk
  case ('LU')  
    symb2mass = 174.9668_rk
  case ('HF')  
    symb2mass = 178.49_rk
  case ('TA')  
    symb2mass = 180.94788_rk
  case ('W ')   
    symb2mass = 183.84_rk
  case ('RE')  
    symb2mass = 186.207_rk
  case ('OS')  
    symb2mass = 190.23_rk
  case ('IR')  
    symb2mass = 192.217_rk
  case ('PT')  
    symb2mass = 195.084_rk
  case ('AU')  
    symb2mass = 196.966569_rk
  case ('HG')  
    symb2mass = 200.59_rk
  case ('TL')  
    symb2mass = 204.3833_rk
  case ('PB')  
    symb2mass = 207.2_rk
  case ('BI')  
    symb2mass = 208.98040_rk
  case ('PO')  
    symb2mass = 209._rk
  case ('AT')  
    symb2mass = 210._rk
  case ('RN')  
    symb2mass = 222._rk
  case ('FR')  
    symb2mass = 223._rk
  case ('RA')  
    symb2mass = 226._rk
  case ('AC')  
    symb2mass = 227._rk
  case ('TH')  
    symb2mass = 232.03806_rk
  case ('PA')  
    symb2mass = 231.03588_rk
  case ('U ')   
    symb2mass = 238.02891_rk
  case default 
    symb2mass = -6666666._rk
  end select
endif

return
end function symb2mass

!***************************************************
!***************************************************

! Convert integer nuclear charge (in units of the elementary charge)
! into atom symbols

elemental function znuc2symb(znuc) result(c)
implicit none

integer, intent(in) :: znuc
character(len=2) :: c

select case (znuc)
  case (1)
    c = "H "
  case (2)
    c = "He"
  case (3)
    c = "Li"
  case (4)
    c = "Be"
  case (5)
    c = "B "
  case (6)
    c = "C "
  case (7)
    c = "N "
  case (8)
    c = "O "
  case (9)
    c = "F "
  case (10)
    c = "Ne"
  case (11)
    c = "Na"
  case (12)
    c = "Mg"
  case (13)
    c = "Al"
  case (14)
    c = "Si"
  case (15)
    c = "P "
  case (16)
    c = "S "
  case (17)
    c = "Cl"
  case (18)
    c = "Ar"
  case (19)
    c = "K "
  case (20)
    c = "Ca"
  case (21)
    c = "Sc"
  case (22)
    c = "Ti"
  case (23)
    c = "V "
  case (24)
    c = "Cr"
  case (25)
    c = "Mn"
  case (26)
    c = "Fe"
  case (27)
    c = "Co"
  case (28)
    c = "Ni"
  case (29)
    c = "Cu"
  case (30)
    c = "Zn"
  case (31)
    c = "Ga"
  case (32)
    c = "Ge"
  case (33)
    c = "As"
  case (34)
    c = "Se"
  case (35)
    c = "Br"
  case (36)
    c = "Kr"
  case (37)
    c = "Rb"
  case (38)
    c = "Sr"
  case (39)
    c = "Y "
  case (40)
    c = "Zr"
  case (41)
    c = "Nb"
  case (42)
    c = "Mo"
  case (43)
    c = "Tc"
  case (44)
    c = "Ru"
  case (45)
    c = "Rh"
  case (46)
    c = "Pd"
  case (47)
    c = "Ag"
  case (48)
    c = "Cd"
  case (49)
    c = "In"
  case (50)
    c = "Sn"
  case (51)
    c = "Sb"
  case (52)
    c = "Te"
  case (53)
    c = "I "
  case (54)
    c = "Xe"
  case (55)
    c = "Cs"
  case (56)
    c = "Ba"
  case (57)
    c = "La"
  case (58)
    c = "Ce"
  case (59)
    c = "Pr"
  case (60)
    c = "Nd"
  case (61)
    c = "Pm"
  case (62)
    c = "Sm"
  case (63)
    c = "Eu"
  case (64)
    c = "Gd"
  case (65)
    c = "Tb"
  case (66)
    c = "Dy"
  case (67)
    c = "Ho"
  case (68)
    c = "Er"
  case (69)
    c = "Tm"
  case (70)
    c = "Yb"
  case (71)
    c = "Lu"
  case (72)
    c = "Hf"
  case (73)
    c = "Ta"
  case (74)
    c = "W "
  case (75)
    c = "Re"
  case (76)
    c = "Os"
  case (77)
    c = "Ir"
  case (78)
    c = "Pt"
  case (79)
    c = "Au"
  case (80)
    c = "Hg"
  case (81)
    c = "Tl"
  case (82)
    c = "Pb"
  case (83)
    c = "Bi"
  case (84)
    c = "Po"
  case (85)
    c = "At"
  case (86)
    c = "Rn"
  case (87)
    c = "Fr"
  case (88)
    c = "Ra"
  case (89)
    c = "Ac"
  case (90)
    c = "Th"
  case (91)
    c = "Pa"
  case (92)
    c = "U "
  case (93)
    c = "Np"
  case (94)
    c = "Pu"
  case (95)
    c = "Am"
  case (96)
    c = "Cm"
  case (97)
    c = "Bk"
  case (98)
    c = "Cf"
  case (99)
    c = "Es"
  case (100)
    c = "Fm"
  case (101)
    c = "Md"
  case (102)
    c = "No"
  case (103)
    c = "Lr"
  case (104)
    c = "Rf"
  case (105)
    c = "Db"
  case (106)
    c = "Sg"
  case (107)
    c = "Bh"
  case (108)
    c = "Hs"
  case (109)
    c = "Mt"
  case (110)
    c = "Ds"
  case (111)
    c = "Rg"
  case (112)
    c = "Cn"
  case (113)
    c = "Nh"
  case (114)
    c = "Fl"
  case (115)
    c = "Mc"
  case (116)
    c = "Lv"
  case (117)
    c = "Ts"
  case (118)
    c = "Og"
  case default
    c = "xx"
end select

return
end function znuc2symb

!***************************************************
!***************************************************

! Convert a single lower-case ASCII character 
! to upper case. If not a lower-case character, the input
! is returned unchanged.

pure function upcase(chara)
implicit none
character(1) :: upcase
character(1), intent(in) :: chara
integer :: ascii_original, ascii_new

ascii_original=iachar(chara)

if (ascii_original.ge.97 .and. ascii_original.le.122) then
  ascii_new=ascii_original-32
  upcase=achar(ascii_new)
else
  upcase=chara
endif

return
end function upcase

!***************************************************
!***************************************************

! Convert a single upper-case ASCII character 
! to lower case. If not an upper-case character, the input
! is returned unchanged.

pure function locase(chara)
implicit none
character(1) :: locase
character(1), intent(in) :: chara
integer :: ascii_original, ascii_new

ascii_original=iachar(chara)

if (ascii_original.ge.65 .and. ascii_original.le.90) then
  ascii_new=ascii_original+32
  locase=achar(ascii_new)
else
  locase=chara
endif

return
end function locase

!***************************************************
!***************************************************

! Finds the first .true. element in a logical mask array. 
! Returns -9999 if no .true. element was found.

function truloc(mask)
implicit none
logical, dimension(:), intent(in) :: mask
integer :: truloc
integer :: i

do i=1,size(mask)
  if (mask(i)) then
    truloc=i
    return
  endif
enddo

truloc=-99999

return
end function truloc

!***************************************************
!***************************************************

! Calculate the Levi-Civita symbol, a.k.a. epsilon tensor
! or totally antisymmetric permutation tensor

function leci(i,j,k)
implicit none
integer :: leci
integer,intent(in) :: i,j,k
integer, dimension(3) :: arr

if (i.gt.3 .or. j.gt.3 .or. k.gt.3 .or. &
  & i.lt.1 .or. j.lt.1 .or. k.lt.1) then
  call error_print('Integers out of range in leci.')
endif

if (i.eq.j .or. j.eq.k .or. k.eq.i) then
  leci=0
  return
else
  arr(1)=i
  arr(2)=j
  arr(3)=k
  do while (abs(arr(2)-arr(1)).gt.1 .or. abs(arr(3)-arr(2)).gt.1)
    arr=cshift(arr,1)
  enddo
  if     (arr(1).eq.1 .and. arr(2).eq.2 .and. arr(3).eq.3) then
    leci=1
  elseif (arr(1).eq.3 .and. arr(2).eq.2 .and. arr(3).eq.1) then
    leci=-1
  else
    call error_print('Something is going seriously wrong in leci.')
  endif
  return
endif
return
end function leci

!***************************************************
!***************************************************

! Determinant of a 3x3 matrix

function det3x3(A)
implicit none
real(rk) :: det3x3
real(rk), dimension(3,3), intent(in) :: A

det3x3=A(1,1)*A(2,2)*A(3,3)+A(1,2)*A(2,3)*A(3,1)+A(2,1)*A(3,2)*A(1,3) &
    & -A(1,3)*A(2,2)*A(3,1)-A(1,2)*A(2,1)*A(3,3)-A(2,3)*A(3,2)*A(1,1)

return
end function det3x3

!***************************************************
!***************************************************

! Basic Loewdin orthogonalisation of a set of input vectors

subroutine loewdin_ortho(Ndim,nsub,inpvec,outvec)
use dlf_linalg_interface_mod
implicit none
integer, intent(in) :: Ndim, nsub
real(rk), dimension(:,:),intent(in)  :: inpvec
real(rk), dimension(:,:),intent(out) :: outvec

integer :: i,j
real(rk), dimension(nsub,nsub) :: S,S_evec,s_min12,Utrans
real(rk), dimension(nsub) :: S_eval

if (size(inpvec,1).ne.nsub .or. size(inpvec,2).ne.Ndim) call error_print('Size mismatch for inpvec in loewdin_ortho.')
if (size(outvec,1).ne.nsub .or. size(outvec,2).ne.Ndim) call error_print('Size mismatch for outvec in loewdin_ortho.')

! Calculate overlap matrix S
do i=1,nsub
  do j=1,i
    S(i,j)=dlf_dot_product(inpvec(i,1:Ndim),inpvec(j,1:Ndim))
  enddo
enddo

do i=1,nsub
  do j=i+1,nsub
    S(i,j)=S(j,i)
  enddo
enddo


call dlf_matrix_diagonalise(nsub,S,S_eval,S_evec)

s_min12=0._rk
do i=1,nsub
  s_min12(i,i)=1._rk/sqrt(S_eval(i))
enddo

Utrans=dlf_matrix_ortho_trans(S_evec,S_min12,1)
outvec=dlf_matmul_simp(Utrans,inpvec)

return
end subroutine loewdin_ortho

!***************************************************
!***************************************************

end module dlf_vpt2_utility

