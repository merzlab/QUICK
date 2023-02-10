!
!	gpu_info.f90
!	new_quick
!
!	Created by Yipu Miao on 4/20/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

subroutine gpu_write_info(io, ierr)
    implicit none
    ! io unit
    integer io
    
    integer gpu_dev_count
    integer gpu_dev_id
    integer gpu_num_proc
    character(len=256) :: gpu_dev_name, gpu_arch_name
    integer gpu_dev_mem
    double precision gpu_core_freq
    integer :: name_len, arch_len
    integer majorv 
    integer minorv
    integer, intent(inout) :: ierr

    name_len=0
    arch_len=0

    write(io,*)
    
    call gpu_get_device_info(gpu_dev_count,gpu_dev_id,gpu_dev_mem,gpu_num_proc,&
    gpu_core_freq,gpu_dev_name,name_len,gpu_arch_name,arch_len,majorv,minorv,ierr)
    write(io,'(a)')        '|------------ GPU INFORMATION ---------------'
    write(io,'(a,i8)')     '| AMD DEVICE COUNT            : ', gpu_dev_count
    write(io,'(a,i8)')     '| AMD DEVICE IN USE           : ', gpu_dev_id
    write(io,'(a,a)')      '| AMD DEVICE NAME             : ', gpu_dev_name(1:name_len)
    write(io,'(a,a)')      '| AMD DEVICE ARCHITECTURE     : ', gpu_arch_name(1:arch_len)
    write(io,'(a,i8)')     '| AMD DEVICE COMPUTE UNITS    : ', gpu_num_proc
    write(io,'(a,f8.2)')   '| AMD DEVICE CORE FREQ(GHZ)   : ', gpu_core_freq
    write(io,'(a,i8)')     '| AMD DEVICE MEMORY SIZE (MB) : ', gpu_dev_mem
    write(io,'(a,i6,a,i1)')'| SUPPORTING ROCm VERSION     : ', majorv,'.',minorv
    write(io,'(a)')        '|--------------------------------------------'
    
end subroutine gpu_write_info

#ifdef HIP_MPIV
subroutine mgpu_write_info(io, gpu_dev_count, mgpu_ids, ierr)
    implicit none
    ! io unit
    integer io

    integer rank
    integer gpu_dev_count
    integer mgpu_ids(gpu_dev_count)
    integer gpu_dev_id
    integer gpu_num_proc
    character(len=256) :: gpu_dev_name, gpu_arch_name
    integer gpu_dev_mem
    double precision gpu_core_freq
    integer :: name_len, arch_len
    integer majorv
    integer minorv
    integer, intent(inout) :: ierr

    name_len=0
    arch_len=0

    write(io,*)

    write(io,'(a)')         '|------------ GPU INFORMATION -------------------------------'
    write(io,'(a,i8)')      '| AMD DEVICE COUNT             : ', gpu_dev_count

    do rank=0, gpu_dev_count-1
    write(io,'(a)')         '|                                                            '
    write(io,'(a,i3,a)')    '|        --    MPI RANK ',rank,' --          '
    gpu_dev_id=mgpu_ids(rank+1)
    call mgpu_get_device_info(gpu_dev_id,gpu_dev_mem,gpu_num_proc,gpu_core_freq,gpu_dev_name,name_len,gpu_arch_name,arch_len,majorv,minorv,ierr) 

    write(io,'(a,i8)')      '|   AMD DEVICE IN USE          : ', gpu_dev_id
    write(io,'(a,a)')       '|   AMD DEVICE NAME            : ', gpu_dev_name(1:name_len)
    write(io,'(a,a)')       '|   AMD DEVICE ARCHITECTURE    : ', gpu_arch_name(1:arch_len)
    write(io,'(a,i8)')      '|   AMD DEVICE COMPUTE UNITS   : ', gpu_num_proc
    write(io,'(a,f8.2)')    '|   AMD DEVICE CORE FREQ(GHZ)  : ', gpu_core_freq
    write(io,'(a,i8)')      '|   AMD DEVICE MEMORY SIZE (MB): ', gpu_dev_mem
    write(io,'(a,i6,a,i1)') '|   SUPPORTING ROCm VERSION    : ', majorv,'.',minorv

    enddo

    write(io,'(a)')         '|------------------------------------------------------------'

end subroutine mgpu_write_info
#endif    
