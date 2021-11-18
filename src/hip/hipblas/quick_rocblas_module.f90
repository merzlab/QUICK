!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Madu Manathunga 11/16/2021: 
! This module was created following fortran_gemv example provided in rocBLAS repo.
! Some interfaces and a couple of subrouitnes were used as they were. 
    
module quick_rocblas_module

    implicit none

    private
    public :: rocDGEMM


    ! AMD - TODO: hip workaround until plugin is ready.
    interface
        function hipMalloc(ptr, size) &
                result(c_int) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc
    end interface

    interface
        function hipFree(ptr) &
                result(c_int) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
        end function hipFree
    end interface

    interface
        function hipMemcpy(dst, src, size, kind) &
                result(c_int) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    interface
        function hipMemset(dst, val, size) &
                result(c_int) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset
    end interface

    interface
        function hipDeviceSynchronize() &
                result(c_int) &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
        end function hipDeviceSynchronize
    end interface

    interface
        function hipDeviceReset() &
                result(c_int) &
                bind(c, name = 'hipDeviceReset')
            use iso_c_binding
            implicit none
        end function hipDeviceReset
    end interface
    ! AMD - TODO end

    interface rocDGEMM
        module procedure quick_rocblas_dgemm
    end interface rocDGEMM

contains

    ! AMD: Error handling subroutines
    subroutine HIP_CHECK(stat)
        use iso_c_binding
    
        implicit none
    
        integer(c_int) :: stat
    
        if(stat /= 0) then
            write(*,*) 'Error: hip error'
            stop
        end if
    end subroutine HIP_CHECK
    
    subroutine ROCBLAS_CHECK(stat)
        use iso_c_binding
    
        implicit none
    
        integer(c_int) :: stat
    
        if(stat /= 0) then
            write(*,*) 'Error: rocblas error'
            stop
        endif
    end subroutine ROCBLAS_CHECK
    ! AMD: End of error handling subroutines



    ! MM: Wrapper function for rocblas_dgemm. 
    subroutine quick_rocblas_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        use iso_c_binding
        use rocblas
        use rocblas_enums

        implicit none

        character, intent(in) :: transa, transb
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        double precision, intent(in) :: alpha, beta
        double precision, dimension(lda,*), intent(in) :: A
        double precision, dimension(ldb,*), intent(in) :: B
        double precision, dimension(ldc,n), intent(out) :: C

        !internal variables
        integer(c_int) :: rb_n, rb_m, rb_k, rb_lda, rb_ldb, rb_ldc, rb_sizea, rb_sizeb, rb_sizec
        real(c_double), target :: rb_alpha, rb_beta
        
        integer(kind(rocblas_operation_none)) :: rb_transa
        integer(kind(rocblas_operation_transpose))  :: rb_transb 

        real(8), dimension(:), allocatable, target :: hA
        real(8), dimension(:), allocatable, target :: hB
        real(8), dimension(:), allocatable, target :: hC

        type(c_ptr), target :: dA
        type(c_ptr), target :: dB
        type(c_ptr), target :: dC

        type(c_ptr), target :: handle

        integer :: i, j, lidx

        ! Initialize internal variables
        rb_transa = rocblas_operation_none
        rb_transb = rocblas_operation_none

        rb_m = m
        rb_n = n
        rb_k = k
        rb_lda = lda
        rb_ldb = ldb
        rb_ldc = ldc

        if(transa .eq. 'n') then
          rb_lda = rb_m
          rb_sizea = rb_k * rb_lda
        else
          rb_lda = rb_k
          rb_sizea = rb_m * rb_lda
          rb_transa = rocblas_operation_transpose
        endif


        if(transb .eq. 'n') then
          rb_ldb = rb_k
          rb_sizeb = rb_n * rb_ldb
        else
          rb_ldb = rb_n
          rb_sizeb = rb_k * rb_ldb
          rb_transb = rocblas_operation_transpose
        endif

        rb_ldc = rb_m
        rb_sizec = rb_n * rb_ldc

        rb_alpha = alpha
        rb_beta = beta

        ! Allocate host-side memory
        if(.not. allocated(hA)) allocate(hA(rb_sizea))
        if(.not. allocated(hB)) allocate(hB(rb_sizeb))
        if(.not. allocated(hC)) allocate(hC(rb_sizec))

        ! Allocate device-side memory
        call HIP_CHECK(hipMalloc(c_loc(dA), int(rb_sizea, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dB), int(rb_sizeb, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dC), int(rb_sizec, c_size_t) * 8))

        ! Initialize host memory
        hA = reshape(A(1:lda,1:rb_lda), (/rb_sizea/))
        hB = reshape(B(1:ldb,1:rb_ldb), (/rb_sizeb/))

        ! Copy memory from host to device
        call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(rb_sizea, c_size_t) * 8, 1))
        call HIP_CHECK(hipMemcpy(dB, c_loc(hB), int(rb_sizeb, c_size_t) * 8, 1))
        !call HIP_CHECK(hipMemcpy(dC, c_loc(hC), int(rb_sizec, c_size_t) * 8, 1))        
        call HIP_CHECK(hipMemset(dC, 0, int(rb_sizec, c_size_t) * 8))

        ! Create rocBLAS handle
        call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

        ! Set handle and call rocblas_dgemm
        call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))

        call ROCBLAS_CHECK(rocblas_dgemm(handle, rb_transa, rb_transb, rb_m, rb_n, rb_k, c_loc(rb_alpha), dA,&
                                         rb_lda, dB, rb_ldb, c_loc(rb_beta), dC, rb_ldc))

        call HIP_CHECK(hipDeviceSynchronize())

        ! Copy output from device to host
        call HIP_CHECK(hipMemcpy(c_loc(hC), dC, int(rb_sizec, c_size_t) * 8, 2))

        ! Transfer result
        C = reshape(hC, (/ldc, n/))

        ! Cleanup
        call HIP_CHECK(hipFree(dA))
        call HIP_CHECK(hipFree(dB))
        call HIP_CHECK(hipFree(dC))
        call ROCBLAS_CHECK(rocblas_destroy_handle(handle))

        if(allocated(hA)) deallocate(hA)
        if(allocated(hB)) deallocate(hB)
        if(allocated(hC)) deallocate(hC)     

    end subroutine quick_rocblas_dgemm

end module quick_rocblas_module

