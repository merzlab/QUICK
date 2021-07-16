!! COPYRIGHT
!!
!!  Copyright 2007 Johannes Kaestner (kaestner@theochem.uni-stuttgart.de),
!!  Tom Keal (thomas.keal@stfc.ac.uk), Alex Turner, Salomon Billeter,
!!  Stephan Thiel, Max-Planck Institut fuer Kohlenforshung, Muelheim, 
!!  Germany.
!!
!!  This file is part of DL-FIND.
!!
!!  DL-FIND is free software: you can redistribute it and/or modify
!!  it under the terms of the GNU Lesser General Public License as 
!!  published by the Free Software Foundation, either version 3 of the 
!!  License, or (at your option) any later version.
!!
!!  DL-FIND is distributed in the hope that it will be useful,
!!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!  GNU Lesser General Public License for more details.
!!
!!  You should have received a copy of the GNU Lesser General Public 
!!  License along with DL-FIND.  If not, see 
!!  <http://www.gnu.org/licenses/>.
!!
MODULE dlfhdlc_matrixlib
!  USE global
  use dlf_parameter_module, only: rk
  use dlf_global, only: printl,stdout
  use dlf_allocate, only: allocate, deallocate
  IMPLICIT NONE

!------------------------------------------------------------------------------
! Constituents of a matrix object:
!   nv:    number of elements
!   np(1): number of rows
!   np(2): number of columns
!   tag:   name
!   data(np(1),np(2)): data
!------------------------------------------------------------------------------

  TYPE matrix
    INTEGER nv
    INTEGER np(2)
    CHARACTER (len=20) :: tag
    REAL (rk), allocatable :: data(:,:)
  END TYPE matrix

  TYPE int_matrix
    INTEGER np(2)
    CHARACTER (len=20) :: tag
    INTEGER, allocatable :: data(:,:)
  END TYPE int_matrix

CONTAINS

!------------------------------------------------------------------------------
! Basic matrix algebra
!------------------------------------------------------------------------------

!******************************************************************************

  INTEGER FUNCTION matrix_add(c,a,b)
    IMPLICIT NONE

! args
    TYPE (matrix) :: a, b
    TYPE (matrix) :: c

! begin
    ! use this line if compiled with bounds check (JK)
    c%data=reshape(a%data,shape(c%data))+reshape(b%data,shape(c%data))
    ! otherwhise use this
!    c%data=a%data + b%data
    matrix_add = 0
  END FUNCTION matrix_add

!******************************************************************************

  INTEGER FUNCTION matrix_transpose(a)
    IMPLICIT NONE

! args
    TYPE (matrix) :: a

! local vars
    INTEGER i, idum, j, n
    REAL (rk) scr
    TYPE (matrix) :: b

! begin, row or column vector - comment out for bounds check!
    IF ((a%np(1)==1) .OR. (a%np(2)==1)) THEN
      idum = a%np(1)
      a%np(1) = a%np(2)
      a%np(2) = idum

! square matrix
    ELSE IF (a%np(1)==a%np(2)) THEN
      n = a%np(1)
      DO i = 1, n
        DO j = i + 1, n
          scr = a%data(j,i)
          a%data(j,i) = a%data(i,j)
          a%data(i,j) = scr
        END DO
      END DO

! generic case
    ELSE
      idum=matrix_copy(a,b)
      idum=matrix_destroy(a)
      a=matrix_create(b%np(2),b%np(1),b%tag)
      DO i = 1, a%np(1)
        DO j = 1, a%np(2)
          a%data(i,j) = b%data(j,i)
        END DO
      END DO
      idum = matrix_destroy(b)
    END IF
    matrix_transpose = 0
  END FUNCTION matrix_transpose

!******************************************************************************

  INTEGER FUNCTION matrix_multiply(alpha,a,b,beta,c)
    IMPLICIT NONE

! args
    TYPE (matrix) :: a, b
    TYPE (matrix) :: c
    REAL (rk) alpha, beta

! local vars
    INTEGER nlink

! begin
    nlink = a%np(2)

    IF (nlink/=b%np(1)) THEN
      WRITE (stdout,'(a,i4,i4)') 'Mismatch dimensions - nlink', a%np(2), &
        b%np(1)
      CALL hdlc_errflag('Matrix error','abort')
    END IF

    IF (a%np(1)/=c%np(1)) THEN
      WRITE (stdout,'(a,i4,i4)') 'Mismatch dimensions - nrow(a) nrow(c)', &
        a%np(1), c%np(1)
      CALL hdlc_errflag('Matrix error','abort')
    END IF

    IF (b%np(2)/=c%np(2)) THEN
      WRITE (stdout,'(a,i4,i4)') 'Mismatch dimensions - ncol(b) ncol(c)', &
        b%np(2), c%np(2)
      CALL hdlc_errflag('Matrix error','abort')
    END IF
    CALL dgemm('N','N',c%np(1),c%np(2),nlink,alpha,a%data,a%np(1),b%data, &
      b%np(1),beta,c%data,c%np(1))

    matrix_multiply = 0
  END FUNCTION matrix_multiply

!******************************************************************************

!//////////////////////////////////////////////////////////////////////////////
! Matrix diagonalisation
!
! Object oriented approach: the dimensions of the passed arrays determine the
! behaviour of the routine:
!
! n     = order of the problem (dimensions of a)
! nval  = number of eigenvalues required (leading dimension of evalues)
! nvect = number of eigenvectors required (trailing dimension of evect)
! 
! Condition: n >= nval >= nvect
! If nvect or nval are less than the order of a, the lowest or the highest
! eigenvalues are taken if increasing or decreasing, respectively
!//////////////////////////////////////////////////////////////////////////////

  INTEGER FUNCTION matrix_diagonalise(a,evect,evalues,increasing)
    IMPLICIT NONE

! args
    LOGICAL increasing
    TYPE (matrix) :: a, evect, evalues

! local vars
    INTEGER n, nval, nvect

! begin, derive sizes from the matrix objects and bomb if inconsistent
    n = a%np(1)
    IF (a%np(2)/=n) THEN
      WRITE (stdout,'(a)') 'Matrix not square in matrix_diagonalise'
      WRITE (stdout,'(a,a)') 'Name of the matrix: ', a%tag
      matrix_diagonalise = -1
      RETURN
    END IF
    nval = evalues%np(1)
    IF (nval>n) THEN
      WRITE (stdout,'(a)') &
        'More eigenvalues requested than the order of the matrix'
      WRITE (stdout,'(a,a,a)') 'Names of the matrices: ', a%tag, evalues%tag
      matrix_diagonalise = -1
      RETURN
    END IF
    nvect = evect%np(2)
    IF (nvect>nval) THEN
      WRITE (stdout,'(a)') 'More eigenvectors requested than eigenvalues'
      WRITE (stdout,'(a,a,a,a)') 'Names of the matrices: ', a%tag, &
        evalues%tag, evect%tag
      matrix_diagonalise = -1
      RETURN
    END IF

! the dirty work is done in array_diagonalise
    matrix_diagonalise = array_diagonalise(a%data,evect%data,evalues%data,n, &
      nval,nvect,increasing)
    RETURN
  END FUNCTION matrix_diagonalise

!******************************************************************************

! array_diagonalise provides a non-object-oriented entry

  INTEGER FUNCTION array_diagonalise(a,evect,evalues,n,nval,nvect,increasing)
    IMPLICIT NONE

! args
    LOGICAL increasing
    INTEGER n, nval, nvect
    REAL (rk), DIMENSION (n,n) :: a
    REAL (rk), DIMENSION (nval) :: evalues
    REAL (rk), DIMENSION (n,nvect) :: evect

! externals
    INTEGER, EXTERNAL :: ilaenv
    REAL (rk), EXTERNAL :: dlamch

! local vars
    CHARACTER jobz
    INTEGER i, ierr, il, iu, lwork, nfound
    INTEGER :: nb, nb1, nb2
    INTEGER, DIMENSION (:), ALLOCATABLE :: ifail, iwork
    REAL (rk) :: abstol, dummy, vwork(nval)
    REAL (rk), DIMENSION (:), ALLOCATABLE :: evalwork, work
    REAL (rk), DIMENSION (:,:), ALLOCATABLE :: awork, evecwork

! begin

! copy the matrix to a work array as it would be destroyed otherwise
    call allocate (awork,n,n)
    awork = a

! full diagonalisation is required
    IF (nval==n) THEN
!       lwork = 3*n ! this is rather a minimum: unblocked algorithm
      nb = ilaenv(1,'dsytrd','L',n,-1,-1,-1)
      IF (nb<0 .AND. printl>=2) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dsytrd failed: returned ', nb
      END IF
      lwork = n*(nb+2)
      call allocate (work,lwork)

      IF (nvect==0) THEN
        jobz = 'N'
      ELSE
        jobz = 'V'
      END IF

! diagonaliser DSYEV and error check
      CALL dsyev(jobz,'L',n,awork,n,evalues,work,lwork,ierr)
      IF (ierr/=0 .AND. printl>=2) THEN
        WRITE (stdout,'(A,I5)') 'Matrix diagonaliser DSYEV failed: returned ', &
          ierr
      END IF

! clean up
      call deallocate (work)
!        if (increasing) then
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = awork(j,i)
!              end do
!           end do
!        else
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = awork(j,n-i+1)
!              end do
!           end do
!           k = n
!           do i = 1,n/2
!              vwork = evalues(i)
!              evalues(i) = evalues(k)
!              evalues(k) = vwork
!              k = k - 1
!           end do
!        end if

      IF (increasing) THEN
        evect = awork(:,1:nvect)
      ELSE
        evect = awork(:,n:n-nvect+1:-1)
        vwork = evalues
        evalues = vwork(nval:1:-1)
      END IF


! partial diagonalisation is required
    ELSE
!       lwork = 8*n ! this is rather a minimum: unblocked algorithm
      nb1 = ilaenv(1,'dsytrd','L',n,-1,-1,-1)
      IF (nb1<0 .AND. printl>=2) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dsytrd failed: returned ', nb1
      END IF

      nb2 = ilaenv(1,'dormtr','LLN',n,n,-1,-1)
      IF (nb2<0 .AND. printl>=2) THEN
        WRITE (stdout,'(A,I5)') &
          'Matrix diagonaliser: ILAENV for dormtr failed: returned ', nb2
      END IF

      nb = max(nb1,nb2)
      lwork = n*(nb+3)

      abstol = 2.0D0*dlamch('S') ! this is for maximum accuracy
      call allocate (work,lwork)
      call allocate (iwork,5*n)
      call allocate (evalwork,n)
      call allocate (evecwork,n,nval) ! note that this may be larger than nvect
      call allocate (ifail,n)
      IF (nvect==0) THEN
        jobz = 'N'
      ELSE
        jobz = 'V'
      END IF
      IF (increasing) THEN
        il = 1
        iu = nval
      ELSE
        il = n - nval + 1
        iu = n
      END IF

! diagonaliser DSYEVX and error check
      CALL dsyevx(jobz,'I','L',n,awork,n,dummy,dummy,il,iu,abstol,nfound, &
        evalwork,evecwork,n,work,lwork,iwork,ifail,ierr)
      IF (ierr/=0 .AND. printl>=2) THEN
        WRITE (stdout,'(A,I5)') 'Matrix diagonaliser DSYEVX failed: returned ' &
          , ierr
        IF (printl>=5) THEN
          WRITE (stdout,'(A)') 'Detailed error message (IFAIL):'
          WRITE (stdout,'(16I5)') (ifail(i),i=1,n)
        END IF
      END IF

! clean up
      call deallocate (iwork)
      call deallocate (work)
!        if (increasing) then
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = evecwork(j,i)
!              end do
!           end do
!           do i = 1,nval
!              evalues(i) = evalwork(i)
!           end do
!        else
!           do i = 1,nvect
!              do j = 1,n
!                 evect(j,i) = evecwork(j,nval-i+1)
!              end do
!           end do
!           do i = 1,nval
!              evalues(i) = evalwork(nval-i+1)
!           end do
!        end if

      IF (increasing) THEN
        evect = evecwork(:,1:nvect)
        evalues = evalwork(1:nval)
      ELSE
        evect = evecwork(:,nval:nval-nvect+1:-1)
        evalues = evalwork(nval:1:-1)
      END IF

      call deallocate (ifail)
      call deallocate (evecwork)
      call deallocate (evalwork)
    END IF

! clear working space
    call deallocate (awork)
    array_diagonalise = ierr
    RETURN
  END FUNCTION array_diagonalise

!******************************************************************************

!//////////////////////////////////////////////////////////////////////////////
! Matrix inversion
!
! Object oriented entry: matrix_invert (a, det, lcdet)
! Array entry:           array_invert  (a, det, lcdet, n)
!
! Leading dimension assumed equal the order of the matrix to be inverted
! Calculate the determinant of lcdet is set
!//////////////////////////////////////////////////////////////////////////////

  INTEGER FUNCTION matrix_invert(a,det,lcdet)
    IMPLICIT NONE

! args
    LOGICAL lcdet
    REAL (rk) det
    TYPE (matrix) :: a

! local vars
    INTEGER n

! begin
    n = a%np(1)
    IF (a%np(2)/=n) THEN
      CALL hdlc_errflag('Matrix not square in invert','abort')
      matrix_invert = -1
      RETURN
    ELSE
      matrix_invert = array_invert(a%data,det,lcdet,n)
    END IF
  END FUNCTION matrix_invert

!******************************************************************************

  INTEGER FUNCTION array_invert(a,det,lcdet,n)
    IMPLICIT NONE

! args
    LOGICAL lcdet
    INTEGER n
    REAL (rk) det
    REAL (rk), DIMENSION (n,n) :: a

! local vars
    INTEGER info, job, r
    REAL (rk), DIMENSION (2) :: dd
    INTEGER, DIMENSION (:), ALLOCATABLE :: ipvt
    REAL (rk), DIMENSION (:), ALLOCATABLE :: work

! begin
    r = n
    IF (lcdet) THEN
      job = 11
    ELSE
      job = 01
    END IF

! allocate memory
    call allocate (ipvt,n)

! get factors, replaces LINPACK: call dgefa (a, r, n, ipvt, info)
    CALL dgetrf(n,n,a,r,ipvt,info)

! test for singularity
    IF (info/=0) THEN
      WRITE (stdout,'(a)') &
        'Warning: attempt to invert a (probably) singular matrix'
      IF (printl>=5) WRITE (stdout,'(A,I5)') 'Info from DGETRF is: ', &
        info
      WRITE (stdout,'(a)') &
        'Matrix is left unchanged and determinant is set to zero'
      IF (lcdet) THEN
        dd(1) = 0.0D0
        dd(2) = 0.0D0
        det = dd(1)*10.0D0**dd(2)
      END IF
      array_invert = info
      call deallocate (ipvt)
      RETURN
    END IF

! determinant, replaces LINPACK: call dgedi (a, r, n, ipvt, dd, work, job)
    IF (lcdet) CALL dgedet(a,r,n,ipvt,dd,job)

    call allocate (work,64*n)
    CALL dgetri(n,a,r,ipvt,work,64*n,info)

    IF (info/=0 .AND. printl>=5) THEN
      WRITE (stdout,'(A,I5)') 'Warning from matrix inverter: DGETRI returned ' &
        , info
    END IF

! deallocate work space
    call deallocate (work)
    call deallocate (ipvt)

! recompute determinant
    IF (lcdet) det = dd(1)*10.0D0**dd(2)
    array_invert = info
  END FUNCTION array_invert

!******************************************************************************

  INTEGER FUNCTION matrix_scale(a,fact)
    IMPLICIT NONE

! args
    TYPE (matrix) :: a
    REAL (rk) fact

! begin
    a%data = fact*a%data
    matrix_scale = 0

  END FUNCTION matrix_scale

!******************************************************************************

  FUNCTION matrix_absmax(a)
    IMPLICIT NONE
    REAL (rk) matrix_absmax

! args
    TYPE (matrix) :: a

! local vars
    INTEGER i, j
    REAL (rk) t

! begin
    t = 0.0D0
    DO j = 1, a%np(2)
      DO i = 1, a%np(1)
        t = max(t,abs(a%data(i,j)))
      END DO
    END DO
    matrix_absmax = t

  END FUNCTION matrix_absmax

!******************************************************************************

  FUNCTION matrix_length(a)
    IMPLICIT NONE
    REAL (rk) matrix_length

! args
    TYPE (matrix) :: a

! local vars
    INTEGER i, j
    REAL (rk) element, sum

! begin
    sum = 0.0D0
    DO j = 1, a%np(2)
      DO i = 1, a%np(1)
        element = a%data(i,j)
        sum = sum + element*element
      END DO
    END DO
!   fac = 1.0D0 / (a%np(1)*a%np(2))
    matrix_length = sqrt(sum)

  END FUNCTION matrix_length

!------------------------------------------------------------------------------
! Supplements to library functions
!------------------------------------------------------------------------------

!//////////////////////////////////////////////////////////////////////////////
! calculate the determinant of a matrix using the factors computed by DGETRF
!
! this subroutine is cut out from LINPACK: SUBROUTINE DGEDI
!
! on entry:
!   a       the output from DGETRF (LAPACK ) / DGEFA (LINPACK).
!   lda     the leading dimension of the matrix a.
!   n       the order of the matrix a.
!   ipvt    the pivot vector from DGETRF / DGEFA.
!
! on return:
!   a       unchanged.
!   ipvt    unchanged.
!   det     determinant of original matrix. determinant = det(1) * 10.0**det(2)
!           with  1.0 .le. abs(det(1)) .lt. 10.0  or  det(1) .eq. 0.0 .
!//////////////////////////////////////////////////////////////////////////////

  SUBROUTINE dgedet(a,lda,n,ipvt,det,job)

    IMPLICIT NONE

! args
    INTEGER lda, n, ipvt(*), job
    REAL (rk) a(lda,*), det(*)

! local vars
    INTEGER i
    REAL (rk) one, ten, zero

! begin
    IF (job<10) RETURN
    zero = 0.0D0
    one = 1.0D0
    ten = 10.0D0
    det(1) = one
    det(2) = zero

! loop over diagonal elements of a
    DO i = 1, n
      IF (ipvt(i)/=i) det(1) = -det(1)
      det(1) = a(i,i)*det(1)

! exit if product is zero
      IF (det(1)==zero) RETURN

! cast result into the form determinant = det(1) * ten**det(2)
      DO WHILE (abs(det(1))<one)
        IF (det(2)<-1000.0D0) THEN
          CALL hdlc_errflag('Problems getting determinant','warn')
          det(1) = zero
          det(2) = zero
          RETURN
        END IF
        det(1) = ten*det(1)
        det(2) = det(2) - one
      END DO
      DO WHILE (abs(det(1))>=ten)
        det(1) = det(1)/ten
        det(2) = det(2) + one
      END DO
    END DO
  END SUBROUTINE dgedet

!------------------------------------------------------------------------------
! Creation and destruction etc.
!------------------------------------------------------------------------------

!!$  FUNCTION int_matrix_create(nr,nc,name)
!!$    IMPLICIT NONE
!!$    TYPE (int_matrix) :: int_matrix_create
!!$    INTEGER, INTENT (IN) :: nr, nc
!!$    CHARACTER*(*) name
!!$    TYPE (int_matrix), POINTER :: temp
!!$
!!$    allocate (temp)
!!$    temp%np(1) = nr
!!$    temp%np(2) = nc
!!$    temp%tag = name
!!$    allocate (temp%data(temp%np(1),temp%np(2)))
!!$    int_matrix_create => temp
!!$  END FUNCTION int_matrix_create

  FUNCTION int_matrix_create(nr,nc,name)
    use dlf_parameter_module, only: ik
    IMPLICIT NONE
    TYPE (int_matrix)    :: int_matrix_create
    INTEGER, INTENT (IN) :: nr, nc
    CHARACTER(*)         ::  name

    int_matrix_create%np(1) = nr
    int_matrix_create%np(2) = nc
    call allocate(int_matrix_create%data,nr,nc)

  END FUNCTION int_matrix_create

!!$  FUNCTION int_matrix_destroy(a)
!!$    IMPLICIT NONE
!!$    INTEGER int_matrix_destroy
!!$    TYPE (int_matrix), POINTER :: a
!!$
!!$    deallocate (a%data)
!!$    deallocate (a)
!!$    int_matrix_destroy = 0
!!$  END FUNCTION int_matrix_destroy

  FUNCTION int_matrix_destroy(a)
    use dlf_parameter_module, only: ik
    IMPLICIT NONE
    INTEGER int_matrix_destroy
    TYPE (int_matrix) :: a

    if(allocated(a%data)) then
      call deallocate (a%data)
    end if
    int_matrix_destroy = 0
  END FUNCTION int_matrix_destroy

  FUNCTION int_matrix_dimension(a,i)
    IMPLICIT NONE
    TYPE (int_matrix) :: a
    INTEGER int_matrix_dimension, i

    int_matrix_dimension = a%np(i)
  END FUNCTION int_matrix_dimension

  FUNCTION matrix_dimension(a,i)
    IMPLICIT NONE
    TYPE (matrix) :: a
    INTEGER matrix_dimension, i

    matrix_dimension = a%np(i)
  END FUNCTION matrix_dimension

! will have to be handeled somehow ..
!!$  FUNCTION matrix_data_array(a)
!!$    IMPLICIT NONE
!!$    TYPE (matrix), POINTER :: a
!!$    REAL (rk), DIMENSION (:,:), POINTER :: matrix_data_array
!!$
!!$    matrix_data_array => a%data
!!$  END FUNCTION matrix_data_array

!!$  FUNCTION matrix_create(nr,nc,name)
!!$    IMPLICIT NONE
!!$    TYPE (matrix), POINTER :: matrix_create
!!$    INTEGER, INTENT (IN) :: nr, nc
!!$    CHARACTER*(*) name
!!$    TYPE (matrix), POINTER :: temp
!!$
!!$    allocate (temp)
!!$    temp%nv = nr*nc
!!$    temp%np(1) = nr
!!$    temp%np(2) = nc
!!$    temp%tag = name
!!$    allocate (temp%data(temp%np(1),temp%np(2)))
!!$    matrix_create => temp
!!$  END FUNCTION matrix_create

  FUNCTION matrix_create(nr,nc,name)
    use dlf_parameter_module, only: rk
    IMPLICIT NONE
    TYPE (matrix) :: matrix_create
    INTEGER, INTENT (IN) :: nr, nc
    CHARACTER(*) :: name
    integer :: idum

    !if(allocated(matrix_create%data)) idum= matrix_destroy(matrix_create)
    matrix_create%nv = nr*nc
    matrix_create%np(1) = nr
    matrix_create%np(2) = nc
    matrix_create%tag = name
    call allocate (matrix_create%data,matrix_create%np(1),matrix_create%np(2))
    !print*,"Allocating matrix  =",name,nr,nc
  END FUNCTION matrix_create

  INTEGER FUNCTION matrix_set_column(a,size,data,col)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in)       :: size
    REAL (rk),   intent(in)     :: data(size)
    INTEGER,   intent(in)       :: col
    INTEGER                     :: i

    matrix_set_column = 0
    if(col>a%np(2)) return
    if(col<1) return
    DO i = 1, a%np(1)
      if(i>size) cycle
      a%data(i,col) = data(i)
    END DO
  END FUNCTION matrix_set_column

  INTEGER FUNCTION matrix_set_row(a,size,data,row)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in)       :: size
    REAL (rk),   intent(in)     :: data(size)
    INTEGER,   intent(in)       :: row
    INTEGER                     :: i

    matrix_set_row = 0
    if(row>a%np(1)) return
    if(row<1) return
    DO i = 1, a%np(2)
      if(i>size) cycle
      a%data(row,i) = data(i)
    END DO
  END FUNCTION matrix_set_row

  INTEGER FUNCTION matrix_set(a,size,data)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in) :: size
    REAL (rk), intent(in) :: data(size)
    INTEGER i, j, k

    k = 0
    DO j = 1, a%np(2)
      DO i = 1, a%np(1)
        k = k + 1
        if(k>size) exit
        a%data(i,j) = data(k)
      END DO
    END DO
    matrix_set = 0
  END FUNCTION matrix_set

  INTEGER FUNCTION matrix_get(a,size,data)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in)  :: size
    REAL (rk), intent(out) :: data(size)
    INTEGER i, j, k

    k = 0
    DO j = 1, a%np(2)
      DO i = 1, a%np(1)
        k = k + 1
        if(k>size) exit
        data(k) = a%data(i,j)
      END DO
    END DO
    matrix_get = 0
  END FUNCTION matrix_get

  INTEGER FUNCTION matrix_get_column(a,size,data,col)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in)  :: size
    REAL (rk), intent(out) :: data(size)
    INTEGER col, i
    
    matrix_get_column = 0
    if(col>a%np(2)) return
    if(col<1) return
    DO i = 1, a%np(1)
      if(i>size) cycle
      data(i) = a%data(i,col)
    END DO
  END FUNCTION matrix_get_column

  INTEGER FUNCTION matrix_get_row(a,size,data,row)
    IMPLICIT NONE
    TYPE (matrix) :: a
    integer,   intent(in)  :: size
    REAL (rk), intent(out) :: data(size)
    INTEGER row, i

    matrix_get_row = 0
    if(row>a%np(1)) return
    if(row<1) return
    DO i = 1, a%np(2)
      if(i>size) cycle
      data(i) = a%data(row,i)
    END DO
  END FUNCTION matrix_get_row

  INTEGER FUNCTION int_matrix_set_column(a,size,data,col)
    IMPLICIT NONE
    TYPE (int_matrix) :: a
    integer,   intent(in)  :: size
    integer,   intent(in)  :: data(size)
    INTEGER col, i

    int_matrix_set_column = 0
    if(col>a%np(2)) return
    if(col<1) return
  
    DO i = 1, a%np(1)
      if(i>size) cycle
      a%data(i,col) = data(i)
    END DO
  END FUNCTION int_matrix_set_column

  INTEGER FUNCTION int_matrix_set_element(a,data,row,col)
    ! this fuction is not used at the moment
    IMPLICIT NONE
    TYPE (int_matrix) :: a
    INTEGER data, col, row

    a%data(row,col) = data
    int_matrix_set_element = 0
  END FUNCTION int_matrix_set_element

  INTEGER FUNCTION matrix_assign_unit(a)
    IMPLICIT NONE
    TYPE (matrix) :: a
    INTEGER i

    a%data = 0.0D0
    DO i = 1, a%np(1)
      a%data(i,i) = 1.0D0
    END DO
    matrix_assign_unit = 0
  END FUNCTION matrix_assign_unit

  INTEGER FUNCTION matrix_copy(x,y)
    IMPLICIT NONE
    TYPE (matrix) :: x, y
    INTEGER       :: length,idum

    length = x%np(1)*x%np(2)
    if(allocated(y%data)) then
      if(y%np(1)*y%np(2)/=length) idum=matrix_destroy(y)
    end if
    IF ( .NOT. allocated(y%data)) THEN
!      y = matrix_create(x%np(2),x%np(1),x%tag) ! orig: why this order?
      y = matrix_create(x%np(1),x%np(2),x%tag)
    END IF
    CALL dcopy(length,x%data,1,y%data,1)
    matrix_copy = 0
  END FUNCTION matrix_copy

  INTEGER FUNCTION matrix_print(a)
    IMPLICIT NONE
    TYPE (matrix) :: a
!
    LOGICAL ohi

    ! check if allocated
    if(.not.allocated(a%data)) then
      CALL hdlc_errflag('matrix not allocated in matrix_print','abort')
    end if
!
    WRITE (stdout,*) ' '
    WRITE (stdout,'(a,a)') 'Contents of matrix: ', a%tag
    WRITE (stdout,*) '---------------------------------------'
!
    ohi = .FALSE.
    CALL prmat(a%data,a%np(1),a%np(2),a%np(1),ohi)
!
    matrix_print = 0
!
  CONTAINS
    SUBROUTINE prmat(v,m,n,ndim,ohi)
! m = number of columns
! n = number of rows
! ndim = leading dimension
! ohi print high precision
      INTEGER n, m, ndim
      REAL (rk) v(ndim,*)
      LOGICAL ohi
      INTEGER imin, imax, max, i, j

      ohi = .FALSE.

      max = 12
      IF (ohi) max = 7
      imax = 0
!
100   imin = imax + 1
      imax = imax + max
      IF (imax>n) imax = n
      WRITE (stdout,9008)
      IF ( .NOT. ohi) WRITE (stdout,8028) (i,i=imin,imax)
      IF (ohi) WRITE (stdout,9028) (i,i=imin,imax)
      WRITE (stdout,9008)
      DO j = 1, m
        IF (ohi) WRITE (stdout,9048) j, (v(j,i),i=imin,imax)
        IF ( .NOT. ohi) WRITE (stdout,8048) j, (v(j,i),i=imin,imax)
      END DO
      IF (imax<n) GO TO 100
      RETURN
9008  FORMAT (1X)
9028  FORMAT (6X,7(6X,I3,6X))
9048  FORMAT (I5,1X,7F15.10)
8028  FORMAT (6X,12(3X,I3,3X))
8048  FORMAT (I5,1X,12F9.5)
    END SUBROUTINE prmat
  END FUNCTION matrix_print

  INTEGER FUNCTION matrix_destroy(a)
    use dlf_parameter_module, only: rk
    IMPLICIT NONE
    TYPE (matrix) :: a

    if(allocated(a%data)) then
      call deallocate (a%data)
      !print*,"Deallocating matrix=",a%tag,a%np(1),a%np(2)
    end if
    matrix_destroy = 0
  END FUNCTION matrix_destroy

!------------------------------------------------------------------------------
! Checkpointing: dump to / undump from unit iunit
!------------------------------------------------------------------------------

  SUBROUTINE hdlc_wr_matrix(iunit,a,lform,lerr)
    IMPLICIT NONE
    LOGICAL lform, lerr
    INTEGER iunit
    TYPE (matrix) :: a
    INTEGER i, j

    lerr = .FALSE.
    IF (lform) THEN
      WRITE (iunit,*,err=98) a%nv, a%np(1), a%np(2)
      WRITE (iunit,'(a)',err=98) a%tag
      WRITE (iunit,*,err=98) ((a%data(i,j),i=1,a%np(1)),j=1,a%np(2))
    ELSE
      WRITE (iunit,err=98) a%nv, a%np(1), a%np(2)
      WRITE (iunit,err=98) a%tag
      WRITE (iunit,err=98) ((a%data(i,j),i=1,a%np(1)),j=1,a%np(2))
    END IF
    RETURN
98  lerr = .TRUE.
  END SUBROUTINE hdlc_wr_matrix

  SUBROUTINE hdlc_rd_matrix(iunit,a,lform,lerr)
    IMPLICIT NONE
    LOGICAL lform, lerr
    INTEGER iunit
    TYPE (matrix) :: a
    INTEGER i, j

    lerr = .FALSE.
! JK included the allocate (pgf requirement)
!    allocate (a)
    IF (lform) THEN
      READ (iunit,*,err=98) a%nv, i, j
      READ (iunit,'(a)',err=98) a%tag
    ELSE
      READ (iunit,err=98) a%nv, i, j
      READ (iunit,err=98) a%tag
    END IF

    if(i/=a%np(1) .or. j/=a%np(2)) then
!!$      ! more secure version: terminate if dimensions do not match
!!$      write(*,*) "Error reading matrix from checkpoint file"
!!$      write(*,'("Expecting dimensions ",2i5)') a%np(1),a%np(2)
!!$      write(*,'("Read dimensions      ",2i5)') i,j
!!$      lerr=.true.
!!$      return

      ! alternative: reallocate matrix and read it
      a%nv = i*j
      a%np(1) = i
      a%np(2) = j
      call deallocate (a%data)
      call allocate (a%data, a%np(1), a%np(2))

    end if

    IF (lform) THEN
      READ (iunit,*,err=98) ((a%data(i,j),i=1,a%np(1)),j=1,a%np(2))
    ELSE
      READ (iunit,err=98) ((a%data(i,j),i=1,a%np(1)),j=1,a%np(2))
    END IF
    RETURN
98  lerr = .TRUE.
  END SUBROUTINE hdlc_rd_matrix

END MODULE dlfhdlc_matrixlib
