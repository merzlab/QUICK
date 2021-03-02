!
!	gpu_info.f90
!	new_quick
!
!	Created by Yipu Miao on 4/20/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

subroutine gpu_write_info(io)
    implicit none
    ! io unit
    integer io
    
    integer gpu_dev_count
    integer gpu_dev_id
    integer gpu_num_proc
    character(len=20) gpu_dev_name
    integer gpu_dev_mem
    double precision gpu_core_freq
    integer name_len
    integer majorv
    integer minorv

!    write(io,*)
!    write(io,'(a)')   '|------------ CUDA INFORMATION ---------------'
!    write(io,'(a)')   '|  CUDA version of QUICK in use'
!    write(io,'(a)')   '|  Implemented by'
!    write(io,'(a)')   '|                      Yipu Miao (Florida)'
!    write(io,'(a)')   '|  CAUTION: CUDA QUICK is currently experimental'
!    write(io,'(a)')   '|           and you may at risk to use it. Be sure'
!    write(io,'(a)')   '|           to check result carefully. Advises or '
!    write(io,'(a)')   '|           bug reports are welcome.'
!    write(io,'(a)')   '|---------------------------------------------'
    write(io,*)
    
    call gpu_get_device_info(gpu_dev_count,gpu_dev_id,gpu_dev_mem,gpu_num_proc,gpu_core_freq,gpu_dev_name,name_len,majorv,minorv)
    write(io,'(a)')        '|------------ GPU INFORMATION ---------------'
    write(io,'(a,i8)')     '| CUDA ENABLED DEVICE         : ', gpu_dev_count
    write(io,'(a,i8)')     '| CUDA DEVICE IN USE          : ', gpu_dev_id
    write(io,'(a,a)')        '| CUDA DEVICE NAME            : ', gpu_dev_name(1:name_len)
    write(io,'(a,i8)')     '| CUDA DEVICE PM              : ', gpu_num_proc
    write(io,'(a,f8.2)')   '| CUDA DEVICE CORE FREQ(GHZ)  : ', gpu_core_freq
    write(io,'(a,i8)')     '| CUDA DEVICE MEMORY SIZE (MB): ', gpu_dev_mem
    write(io,'(a,i6,a,i1)')     '| SUPPORTING CUDA VERSION     : ', majorv,'.',minorv
    write(io,'(a)')        '|--------------------------------------------'
    if (majorv <=1 .and. minorv<=2) call prtWrn(io,"Selected GPU does not support double-precision computation.")
    
end subroutine gpu_write_info

#ifdef CUDA_MPIV
subroutine mgpu_write_info(io, gpu_dev_count, mgpu_ids)
    implicit none
    ! io unit
    integer io

    integer rank
    integer gpu_dev_count
    integer mgpu_ids(gpu_dev_count)
    integer gpu_dev_id
    integer gpu_num_proc
    character(len=20) gpu_dev_name
    integer gpu_dev_mem
    double precision gpu_core_freq
    integer name_len
    integer majorv
    integer minorv

!    write(io,*)
!    write(io,'(a)')   '|------------ CUDA INFORMATION ---------------'
!    write(io,'(a)')   '|  CUDA version of QUICK in use'
!    write(io,'(a)')   '|  Implemented by'
!    write(io,'(a)')   '|                      Yipu Miao (Florida)'
!    write(io,'(a)')   '|  CAUTION: CUDA QUICK is currently experimental'
!    write(io,'(a)')   '|           and you may at risk to use it. Be sure'
!    write(io,'(a)')   '|           to check result carefully. Advises or '
!    write(io,'(a)')   '|           bug reports are welcome.'
!    write(io,'(a)')   '|---------------------------------------------'

    write(io,*)

    write(io,'(a)')         '|------------ GPU INFORMATION -------------------------------'
    write(io,'(a,i8)')      '| CUDA ENABLED DEVICES          : ', gpu_dev_count

    do rank=0, gpu_dev_count-1
    write(io,'(a)')         '|                                                            '
    write(io,'(a,i3,a)')      '|        --    MPI RANK ',rank,' --          '
    gpu_dev_id=mgpu_ids(rank+1)
    call mgpu_get_device_info(gpu_dev_id,gpu_dev_mem,gpu_num_proc,gpu_core_freq,gpu_dev_name,name_len,majorv,minorv) 

    write(io,'(a,i8)')      '|   CUDA DEVICE IN USE          : ', gpu_dev_id
    write(io,'(a,a)')       '|   CUDA DEVICE NAME            : ', gpu_dev_name(1:name_len)
    write(io,'(a,i8)')      '|   CUDA DEVICE PM              : ', gpu_num_proc
    write(io,'(a,f8.2)')    '|   CUDA DEVICE CORE FREQ(GHZ)  : ', gpu_core_freq
    write(io,'(a,i8)')      '|   CUDA DEVICE MEMORY SIZE (MB): ', gpu_dev_mem
    write(io,'(a,i6,a,i1)') '|   SUPPORTING CUDA VERSION     : ', majorv,'.',minorv

    enddo

    write(io,'(a)')         '|------------------------------------------------------------'

end subroutine mgpu_write_info
#endif    
