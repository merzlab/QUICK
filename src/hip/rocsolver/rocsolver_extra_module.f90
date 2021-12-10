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

! Madu Manathunga 11/19/2021: 
! This module contains some extra subroutines and wrappers useful for rocSolver. 
! AMD guys should bundle rocSolver wrapper into a rocsolver_module.f90 or something. 
! Then this module can be totally eliminated

module rocsolver_extra

    use iso_c_binding

    implicit none

    interface
        !rocsolver_dsyevd computes the eigenvalues and optionally the eigenvectors of a real symmetric
        !matrix A.
        
        !The eigenvalues are returned in ascending order. The eigenvectors are computed using a
        !divide-and-conquer algorithm, depending on the value of evect. The computed eigenvectors
        !are orthonormal.
        !@param[in]
        !handle      rocblas_handle.
        !@param[in]
        !evect       #rocblas_evect.
        !            Specifies whether the eigenvectors are to be computed.
        !            If evect is rocblas_evect_original, then the eigenvectors are computed.
        !            rocblas_evect_tridiagonal is not supported.
        !@param[in]
        !uplo        rocblas_fill.
        !            Specifies whether the upper or lower part of the symmetric matrix A is stored.
        !            If uplo indicates lower (or upper), then the upper (or lower) part of A
        !            is not used.
        !@param[in]
        !n           rocblas_int. n >= 0.
        !            Number of rows and columns of matrix A.
        !@param[inout]
        !A           pointer to type. Array on the GPU of dimension lda*n.
        !            On entry, the matrix A. On exit, the eigenvectors of A if they were computed and
        !            the algorithm converged; otherwise the contents of A are destroyed.
        !@param[in]
        !lda         rocblas_int. lda >= n.
        !            Specifies the leading dimension of matrix A.
        !@param[out]
        !D           pointer to type. Array on the GPU of dimension n.
        !            The eigenvalues of A in increasing order.
        !@param[out]
        !E           pointer to type. Array on the GPU of dimension n.
        !            This array is used to work internally with the tridiagonal matrix T associated with A.
        !            On exit, if info > 0, it contains the unconverged off-diagonal elements of T
        !            (or properly speaking, a tridiagonal matrix equivalent to T). The diagonal elements
        !            of this matrix are in D; those that converged correspond to a subset of the
        !            eigenvalues of A (not necessarily ordered).
        !@param[out]
        !info        pointer to a rocblas_int on the GPU.
        !            If info = 0, successful exit.
        !            If info = i > 0 and evect is rocblas_evect_none, the algorithm did not converge.
        !            i elements of E did not converge to zero.
        !            If info = i > 0 and evect is rocblas_evect_original, the algorithm failed to
        !            compute an eigenvalue in the submatrix from [i/(n+1), i/(n+1)] to [i%(n+1), i%(n+1)].

        function rocsolver_dsyevd(handle, evect, uplo, N, A, lda, D, E, info) &
                result(c_int) &
                bind(c, name = 'rocsolver_dsyevd')
            use iso_c_binding
            use rocblas_enums
            use rocblas_enums_extra
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_evect_original)), value :: evect
            integer(kind(rocblas_fill_lower)), value :: uplo
            integer(c_int), value :: N
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: info
        end function rocsolver_dsyevd
    end interface

contains

    subroutine ROCSOLVER_CHECK(stat)
        use iso_c_binding
    
        implicit none
    
        integer(c_int) :: stat
    
        if(stat /= 0) then
            write(*,*) 'Error: rocsolver error', stat
            stop
        endif
    end subroutine ROCSOLVER_CHECK

end module rocsolver_extra
