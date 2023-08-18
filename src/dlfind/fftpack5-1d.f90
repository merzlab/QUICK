
subroutine cosq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQ1B: real double precision backward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    COSQ1B computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the backward transform or Fourier synthesis, transforming
!    the sequence from spectral to physical space.
!
!    This transform is normalized since a call to COSQ1B followed
!    by a call to COSQ1F (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the number of elements to be transformed 
!    in the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQ1I before the first call to routine COSQ1F 
!    or COSQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to COSQ1F and COSQ1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              lenx
  integer              n
  real ( rk       ) ssqrt2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cosq1b', 6)
    return
  else if (lensav < &
    2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosq1b', 8)
    return
  else if (lenwrk < n) then
    ier = 3
    call xerfft ('cosq1b', 10)
    return
  end if

      if (n-2) 300,102,103

 102  ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )
      x1 = x(1,1)+x(1,2)
      x(1,2) = ssqrt2*(x(1,1)-x(1,2))
      x(1,1) = x1
      return

  103 call cosqb1 (n,inc,x,wsave,work,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('cosq1b',-5)
      end if

  300 continue

  return
end
subroutine cosq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQ1F: real double precision forward cosine quarter wave transform, 1D.
!
!  Discussion:
!
!    COSQ1F computes the one-dimensional Fourier transform of a sequence 
!    which is a cosine series with odd wave numbers.  This transform is 
!    referred to as the forward transform or Fourier analysis, transforming 
!    the sequence from physical to spectral space.
!
!    This transform is normalized since a call to COSQ1F followed
!    by a call to COSQ1B (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the number of elements to be transformed 
!    in the sequence.  The transform is most efficient when N is a 
!    product of small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR); on input, containing the sequence 
!    to be transformed, and on output, containing the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQ1I before the first call to routine COSQ1F 
!    or COSQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to COSQ1F and COSQ1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              n
  integer              lenx
  real ( rk       ) ssqrt2
  real ( rk       ) tsqx
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cosq1f', 6)
    return
  else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosq1f', 8)
    return
  else if (lenwrk < n) then
    ier = 3
    call xerfft ('cosq1f', 10)
    return
  end if

      if (n-2) 102,101,103
  101 ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )
      tsqx = ssqrt2*x(1,2)
      x(1,2) = 0.5D+00 *x(1,1)-tsqx
      x(1,1) = 0.5D+00 *x(1,1)+tsqx
  102 return
  103 call cosqf1 (n,inc,x,wsave,work,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cosq1f',-5)
  end if

  return
end
subroutine cosq1i ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQ1I: initialization for COSQ1B and COSQ1F.
!
!  Discussion:
!
!    COSQ1I initializes array WSAVE for use in its companion routines 
!    COSQ1F and COSQ1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product 
!    of small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors of N
!    and also containing certain trigonometric values which will be used 
!    in routines COSQ1B or COSQ1F.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  real ( rk       ) fk
  integer              ier
  integer              ier1
  integer              k
  integer              lnsv
  integer              n
  real ( rk       ) pih
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosq1i', 3)
    return
  end if

  pih = 2.0D+00 * atan ( 1.0D+00 )
  dt = pih / real  ( n, rk       )
  fk = 0.0D+00

  do k=1,n
    fk = fk + 1.0D+00
    wsave(k) = cos(fk*dt)
  end do

  lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4
  call rfft1i (n, wsave(n+1), lnsv, ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cosq1i',-5)
  end if

  return
end
subroutine cosqb1 ( n, inc, x, wsave, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQB1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              lnsv
  integer              lnwk
  integer              modn
  integer              n
  integer              np2
  integer              ns2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xim1

  ier = 0
  ns2 = (n+1)/2
  np2 = n+2

  do i=3,n,2
    xim1 = x(1,i-1)+x(1,i)
    x(1,i) = 0.5D+00 * (x(1,i-1)-x(1,i))
    x(1,i-1) = 0.5D+00 * xim1
  end do

  x(1,1) = 0.5D+00 * x(1,1)
  modn = mod(n,2)

  if (modn == 0 ) then
    x(1,n) = 0.5D+00 * x(1,n)
  end if

  lenx = inc*(n-1)  + 1
  lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) + 4
  lnwk = n

  call rfft1b(n,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cosqb1',-5)
    return
  end if

  do k=2,ns2
    kc = np2-k
    work(k) = wsave(k-1)*x(1,kc)+wsave(kc-1)*x(1,k)
    work(kc) = wsave(k-1)*x(1,k)-wsave(kc-1)*x(1,kc)
  end do

  if (modn == 0) then
    x(1,ns2+1) = wsave(ns2)*(x(1,ns2+1)+x(1,ns2+1))
  end if

  do k=2,ns2
    kc = np2-k
    x(1,k) = work(k)+work(kc)
    x(1,kc) = work(k)-work(kc)
  end do

  x(1,1) = x(1,1)+x(1,1)

  return
end
subroutine cosqf1 ( n, inc, x, wsave, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQF1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              lnsv
  integer              lnwk
  integer              modn
  integer              n
  integer              np2
  integer              ns2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xim1

  ier = 0
  ns2 = (n+1)/2
  np2 = n+2

  do k=2,ns2
    kc = np2-k
    work(k)  = x(1,k)+x(1,kc)
    work(kc) = x(1,k)-x(1,kc)
  end do

  modn = mod(n,2)

  if (modn == 0) then
    work(ns2+1) = x(1,ns2+1)+x(1,ns2+1)
  end if

  do k=2,ns2
    kc = np2-k
    x(1,k)  = wsave(k-1)*work(kc)+wsave(kc-1)*work(k)
    x(1,kc) = wsave(k-1)*work(k) -wsave(kc-1)*work(kc)
  end do

  if (modn == 0) then
    x(1,ns2+1) = wsave(ns2)*work(ns2+1)
  end if

  lenx = inc*(n-1)  + 1
  lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) + 4
  lnwk = n

  call rfft1f(n,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cosqf1',-5)
    return
  end if

  do i=3,n,2
    xim1 = 0.5D+00 * (x(1,i-1)+x(1,i))
    x(1,i) = 0.5D+00 * (x(1,i-1)-x(1,i))
    x(1,i-1) = xim1
  end do

  return
end
subroutine cosqmb ( lot, jump, n, inc, x, lenx, wsave, lensav, work, lenwrk, &
  ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQMB: real double precision backward cosine quarter wave, multiple vectors.
!
!  Discussion:
!
!    COSQMB computes the one-dimensional Fourier transform of multiple
!    sequences, each of which is a cosine series with odd wave numbers.
!    This transform is referred to as the backward transform or Fourier
!    synthesis, transforming the sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSQMB followed
!    by a call to COSQMF (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), array containing LOT sequences, 
!    each having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQMI before the first call to routine COSQMF 
!    or COSQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COSQMF and COSQMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              jump
  integer              lenx
  integer              lj
  integer              lot
  integer              m
  integer              n
  real ( rk       ) ssqrt2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1
  logical xercon

  ier = 0

  if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cosqmb', 6)
    return
  else if (lensav < &
    2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosqmb', 8)
    return
  else if (lenwrk < lot*n) then
    ier = 3
    call xerfft ('cosqmb', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('cosqmb', -1)
    return
  end if

      lj = (lot-1)*jump+1
      if (n-2) 101,102,103
 101  do m=1,lj,jump
        x(m,1) = x(m,1)
      end do
      return
 102  ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )
      do m=1,lj,jump
        x1 = x(m,1)+x(m,2)
        x(m,2) = ssqrt2*(x(m,1)-x(m,2))
        x(m,1) = x1
      end do
      return

  103 call mcsqb1 (lot,jump,n,inc,x,wsave,work,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('cosqmb',-5)
      end if

  return
end
subroutine cosqmf ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQMF: real double precision forward cosine quarter wave, multiple vectors.
!
!  Discussion:
!
!    COSQMF computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each of the sequences is a 
!    cosine series with odd wave numbers.  This transform is referred to 
!    as the forward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSQMF followed
!    by a call to COSQMB (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), array containing LOT sequences, 
!    each having length N.  R can have any number of dimensions, but the total
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSQMI before the first call to routine COSQMF 
!    or COSQMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSQMF and COSQMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              jump
  integer              lenx
  integer              lj
  integer              lot
  integer              m
  integer              n
  real ( rk       ) ssqrt2
  real ( rk       ) tsqx
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon

  ier = 0

  if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cosqmf', 6)
    return
  else if (lensav < &
    2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosqmf', 8)
    return
  else if (lenwrk < lot*n) then
    ier = 3
    call xerfft ('cosqmf', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('cosqmf', -1)
    return
  end if

  lj = (lot-1)*jump+1

  if (n-2) 102,101,103
  101 ssqrt2 = 1.0D+00 / sqrt ( 2.0D+00 )

      do m=1,lj,jump
        tsqx = ssqrt2*x(m,2)
        x(m,2) = 0.5D+00 * x(m,1)-tsqx
        x(m,1) = 0.5D+00 * x(m,1)+tsqx
      end do

  102 return

  103 call mcsqf1 (lot,jump,n,inc,x,wsave,work,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('cosqmf',-5)
      end if

  return
end
subroutine cosqmi ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSQMI: initialization for COSQMB and COSQMF.
!
!  Discussion:
!
!    COSQMI initializes array WSAVE for use in its companion routines 
!    COSQMF and COSQMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors of 
!    N and also containing certain trigonometric values which will be used 
!    in routines COSQMB or COSQMF.
!
!    Input, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  real ( rk       ) fk
  integer              ier
  integer              ier1
  integer              k
  integer              lnsv
  integer              n
  real ( rk       ) pih
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cosqmi', 3)
    return
  end if

  pih = 2.0D+00 * atan ( 1.0D+00 )
  dt = pih/real ( n, rk       )
  fk = 0.0D+00

  do k=1,n
    fk = fk + 1.0D+00
    wsave(k) = cos(fk*dt)
  end do

  lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4

  call rfftmi (n, wsave(n+1), lnsv, ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cosqmi',-5)
  end if

  return
end
subroutine cost1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COST1B: real double precision backward cosine transform, 1D.
!
!  Discussion:
!
!    COST1B computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the sequence 
!    from spectral to physical space.
!
!    This transform is normalized since a call to COST1B followed
!    by a call to COST1F (or vice-versa) reproduces the original array 
!    within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing the sequence to 
!     be transformed.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COST1I before the first call to routine COST1F 
!    or COST1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COST1F and COST1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              lenx
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cost1b', 6)
    return
  else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cost1b', 8)
    return
  else if (lenwrk < n-1) then
    ier = 3
    call xerfft ('cost1b', 10)
    return
  end if

  if (n == 1) then
    return
  end if

  call costb1 (n,inc,x,wsave,work,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cost1b',-5)
  end if

  return
end
subroutine cost1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COST1F: real double precision forward cosine transform, 1D.
!
!  Discussion:
!
!    COST1F computes the one-dimensional Fourier transform of an even 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence 
!    from  physical to spectral space.
!
!    This transform is normalized since a call to COST1F followed by a call 
!    to COST1B (or vice-versa) reproduces the original array within 
!    roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing the sequence to 
!    be transformed.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COST1I before the first call to routine COST1F
!    or COST1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to COST1F and COST1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N-1.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              lenx
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('cost1f', 6)
    return
  else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cost1f', 8)
    return
  else if (lenwrk < n-1) then
    ier = 3
    call xerfft ('cost1f', 10)
    return
  end if

  if (n == 1) then
    return
  end if

  call costf1(n,inc,x,wsave,work,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cost1f',-5)
  end if

  return
end
subroutine cost1i ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COST1I: initialization for COST1B and COST1F.
!
!  Discussion:
!
!    COST1I initializes array WSAVE for use in its companion routines 
!    COST1F and COST1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product 
!    of small primes.
!
!    Input, integer              LENSAV, dimension of WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors of
!    N and also containing certain trigonometric values which will be used in
!    routines COST1B or COST1F.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  real ( rk       ) fk
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lnsv
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) pi
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('cost1i', 3)
    return
  end if

  if ( n <= 3 ) then
    return
  end if

  nm1 = n-1
  np1 = n+1
  ns2 = n/2
  pi = 4.0D+00 * atan ( 1.0D+00 )
  dt = pi/ real ( nm1, rk       )
  fk = 0.0D+00

  do k=2,ns2
    kc = np1-k
    fk = fk + 1.0D+00
    wsave(k) = 2.0D+00 * sin(fk*dt)
    wsave(kc) = 2.0D+00 * cos(fk*dt)
  end do

  lnsv = nm1 + int(log( real ( nm1, rk       ) )/log( 2.0D+00 )) +4

  call rfft1i (nm1, wsave(n+1), lnsv, ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('cost1i',-5)
  end if

  return
end
subroutine costb1 ( n, inc, x, wsave, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSTB1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum
  real ( rk       ) fnm1s2
  real ( rk       ) fnm1s4
  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              lnsv
  integer              lnwk
  integer              modn
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1h
  real ( rk       ) x1p3
  real ( rk       ) x2
  real ( rk       ) xi

  ier = 0

      nm1 = n-1
      np1 = n+1
      ns2 = n/2

      if (n-2) 106,101,102
  101 x1h = x(1,1)+x(1,2)
      x(1,2) = x(1,1)-x(1,2)
      x(1,1) = x1h
      return
  102 if ( 3 < n ) go to 103
      x1p3 = x(1,1)+x(1,3)
      x2 = x(1,2)
      x(1,2) = x(1,1)-x(1,3)
      x(1,1) = x1p3+x2
      x(1,3) = x1p3-x2
      return
  103 x(1,1) = x(1,1)+x(1,1)
      x(1,n) = x(1,n)+x(1,n)
      dsum = x(1,1)-x(1,n)
      x(1,1) = x(1,1)+x(1,n)

      do k=2,ns2
         kc = np1-k
         t1 = x(1,k)+x(1,kc)
         t2 = x(1,k)-x(1,kc)
         dsum = dsum+wsave(kc)*t2
         t2 = wsave(k)*t2
         x(1,k) = t1-t2
         x(1,kc) = t1+t2
      end do

      modn = mod(n,2)
      if (modn == 0) go to 124
      x(1,ns2+1) = x(1,ns2+1)+x(1,ns2+1)
  124 lenx = inc*(nm1-1)  + 1
      lnsv = nm1 + int(log( real ( nm1, rk       ) )/log( 2.0D+00 )) + 4
      lnwk = nm1

      call rfft1f(nm1,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('costb1',-5)
        return
      end if

      fnm1s2 = real ( nm1, rk       ) / 2.0D+00 
      dsum = 0.5D+00 * dsum
      x(1,1) = fnm1s2*x(1,1)
      if(mod(nm1,2) /= 0) go to 30
      x(1,nm1) = x(1,nm1)+x(1,nm1)
   30 fnm1s4 = real ( nm1, rk       ) / 4.0D+00

      do i=3,n,2
         xi = fnm1s4*x(1,i)
         x(1,i) = fnm1s4*x(1,i-1)
         x(1,i-1) = dsum
         dsum = dsum+xi
      end do

      if (modn /= 0) return
         x(1,n) = dsum
  106 continue

  return
end
subroutine costf1 ( n, inc, x, wsave, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSTF1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum
  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              lnsv
  integer              lnwk
  integer              modn
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) snm1
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) tx2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1h
  real ( rk       ) x1p3
  real ( rk       ) xi

  ier = 0

  nm1 = n-1
  np1 = n+1
  ns2 = n/2

      if (n-2) 200,101,102
  101 x1h = x(1,1)+x(1,2)
      x(1,2) = 0.5D+00 * (x(1,1)-x(1,2))
      x(1,1) = 0.5D+00 * x1h
      go to 200
  102 if ( 3 < n ) go to 103
      x1p3 = x(1,1)+x(1,3)
      tx2 = x(1,2)+x(1,2)
      x(1,2) = 0.5D+00 * (x(1,1)-x(1,3))
      x(1,1) = 0.25D+00 *(x1p3+tx2)
      x(1,3) = 0.25D+00 *(x1p3-tx2)
      go to 200
  103 dsum = x(1,1)-x(1,n)
      x(1,1) = x(1,1)+x(1,n)
      do k=2,ns2
         kc = np1-k
         t1 = x(1,k)+x(1,kc)
         t2 = x(1,k)-x(1,kc)
         dsum = dsum+wsave(kc)*t2
         t2 = wsave(k)*t2
         x(1,k) = t1-t2
         x(1,kc) = t1+t2
      end do
      modn = mod(n,2)
      if (modn == 0) go to 124
      x(1,ns2+1) = x(1,ns2+1)+x(1,ns2+1)
  124 lenx = inc*(nm1-1)  + 1
      lnsv = nm1 + int(log( real ( nm1, rk       ) )/log( 2.0D+00 )) + 4
      lnwk = nm1

      call rfft1f(nm1,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('costf1',-5)
        go to 200
      end if

      snm1 = 1.0D+00 / real ( nm1, rk       )
      dsum = snm1*dsum
      if(mod(nm1,2) /= 0) go to 30
      x(1,nm1) = x(1,nm1)+x(1,nm1)
   30 do i=3,n,2
         xi = 0.5D+00 * x(1,i)
         x(1,i) = 0.5D+00 * x(1,i-1)
         x(1,i-1) = dsum
         dsum = dsum+xi
      end do
      if (modn /= 0) go to 117
      x(1,n) = dsum
  117 x(1,1) = 0.5D+00 * x(1,1)
      x(1,n) = 0.5D+00 * x(1,n)
  200 continue

  return
end
subroutine costmb ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSTMB: real double precision backward cosine transform, multiple vectors.
!
!  Discussion:
!
!    COSTMB computes the one-dimensional Fourier transform of multiple 
!    even sequences within a real array.  This transform is referred to 
!    as the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to COSTMB followed
!    by a call to COSTMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), array containing LOT sequences, 
!    each having length N.  On input, the data to be transformed; on output,
!    the transormed data.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSTMI before the first call to routine COSTMF 
!    or COSTMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSTMF and COSTMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(N+1).
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              iw1
  integer              jump
  integer              lenx
  integer              lot
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon

  ier = 0

  if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('costmb', 6)
    return
  else if (lensav < &
    2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('costmb', 8)
    return
  else if (lenwrk < lot*(n+1)) then
    ier = 3
    call xerfft ('costmb', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('costmb', -1)
    return
  end if

  iw1 = lot+lot+1
  call mcstb1(lot,jump,n,inc,x,wsave,work,work(iw1),ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('costmb',-5)
  end if

  return
end
subroutine costmf ( lot, jump, n, inc, x, lenx, wsave, lensav, work, &
  lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSTMF: real double precision forward cosine transform, multiple vectors.
!
!  Discussion:
!
!    COSTMF computes the one-dimensional Fourier transform of multiple 
!    even sequences within a real array.  This transform is referred to 
!    as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space.
!
!    This transform is normalized since a call to COSTMF followed
!    by a call to COSTMB (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences to 
!    be transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N-1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), array containing LOT sequences, 
!    each having length N.  On input, the data to be transformed; on output,
!    the transormed data.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.
!
!    Input, integer              LENR, the dimension of the  R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to COSTMI before the first call to routine COSTMF
!    or COSTMB for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to COSTMF and COSTMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(N+1).
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              iw1
  integer              jump
  integer              lenx
  integer              lot
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon

  ier = 0

  if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('costmf', 6)
    return
  else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('costmf', 8)
    return
  else if (lenwrk < lot*(n+1)) then
    ier = 3
    call xerfft ('costmf', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('costmf', -1)
    return
  end if

  iw1 = lot+lot+1

  call mcstf1(lot,jump,n,inc,x,wsave,work,work(iw1),ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('costmf',-5)
  end if

  return
end
subroutine costmi ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! COSTMI: initialization for COSTMB and COSTMF.
!
!  Discussion:
!
!    COSTMI initializes array WSAVE for use in its companion routines 
!    COSTMF and COSTMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors of N 
!    and also containing certain trigonometric values which will be used 
!    in routines COSTMB or COSTMF.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  real ( rk       ) fk
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lnsv
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) pi
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('costmi', 3)
    return
  end if

  if (n <= 3) then
    return
  end if

      nm1 = n-1
      np1 = n+1
      ns2 = n/2
      pi = 4.0D+00 * atan ( 1.0D+00 )
      dt = pi/ real ( nm1, rk       )
      fk = 0.0D+00

      do k=2,ns2
         kc = np1-k
         fk = fk + 1.0D+00
         wsave(k) = 2.0D+00 * sin(fk*dt)
         wsave(kc) = 2.0D+00 * cos(fk*dt)
      end do

      lnsv = nm1 + int(log( real ( nm1, rk       ) )/log( 2.0D+00 )) +4

      call rfftmi (nm1, wsave(n+1), lnsv, ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('costmi',-5)
      end if

  return
end
subroutine mcsqb1 (lot,jump,n,inc,x,wsave,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MCSQB1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc
  integer              lot

  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lj
  integer              lnsv
  integer              lnwk
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              np2
  integer              ns2
  real ( rk       ) work(lot,*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xim1

  ier = 0
  lj = (lot-1)*jump+1
  ns2 = (n+1)/2
  np2 = n+2

  do i=3,n,2
    do m=1,lj,jump
      xim1 = x(m,i-1)+x(m,i)
      x(m,i) = 0.5D+00 * (x(m,i-1)-x(m,i))
      x(m,i-1) = 0.5D+00 * xim1
    end do
  end do

  do m=1,lj,jump
    x(m,1) = 0.5D+00 * x(m,1)
  end do

  modn = mod(n,2)
  if (modn == 0) then
    do m=1,lj,jump
      x(m,n) = 0.5D+00 * x(m,n)
    end do
  end if

  lenx = (lot-1)*jump + inc*(n-1)  + 1
  lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) + 4
  lnwk = lot*n

  call rfftmb(lot,jump,n,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('mcsqb1',-5)
    return
  end if

  do k=2,ns2
    kc = np2-k
    m1 = 0
    do m=1,lj,jump
      m1 = m1 + 1
      work(m1,k) = wsave(k-1)*x(m,kc)+wsave(kc-1)*x(m,k)
      work(m1,kc) = wsave(k-1)*x(m,k)-wsave(kc-1)*x(m,kc)
    end do
  end do

  if (modn == 0) then
    do m=1,lj,jump
      x(m,ns2+1) = wsave(ns2)*(x(m,ns2+1)+x(m,ns2+1))
    end do
  end if

  do k=2,ns2
    kc = np2-k
    m1 = 0
    do m=1,lj,jump
      m1 = m1 + 1
      x(m,k) = work(m1,k)+work(m1,kc)
      x(m,kc) = work(m1,k)-work(m1,kc)
    end do
  end do

  do m=1,lj,jump
    x(m,1) = x(m,1)+x(m,1)
  end do

  return
end
subroutine mcsqf1 (lot,jump,n,inc,x,wsave,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MCSQF1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc
  integer              lot

  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lnsv
  integer              lnwk
  integer              lj
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              np2
  integer              ns2
  real ( rk       ) work(lot,*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xim1

  ier = 0

  lj = (lot-1)*jump+1
  ns2 = (n+1)/2
  np2 = n+2

  do k=2,ns2
    kc = np2-k
    m1 = 0
    do m=1,lj,jump
      m1 = m1 + 1
      work(m1,k)  = x(m,k)+x(m,kc)
      work(m1,kc) = x(m,k)-x(m,kc)
    end do
  end do

  modn = mod(n,2)

  if (modn == 0) then
    m1 = 0
    do m=1,lj,jump
      m1 = m1 + 1
      work(m1,ns2+1) = x(m,ns2+1)+x(m,ns2+1)
    end do
  end if

       do 102 k=2,ns2
         kc = np2-k
         m1 = 0
         do 302 m=1,lj,jump
         m1 = m1 + 1
         x(m,k)  = wsave(k-1)*work(m1,kc)+wsave(kc-1)*work(m1,k)
         x(m,kc) = wsave(k-1)*work(m1,k) -wsave(kc-1)*work(m1,kc)
 302     continue
  102 continue

      if (modn /= 0) go to 303
      m1 = 0
      do 304 m=1,lj,jump
         m1 = m1 + 1
         x(m,ns2+1) = wsave(ns2)*work(m1,ns2+1)
 304  continue
 303  continue

      lenx = (lot-1)*jump + inc*(n-1)  + 1
      lnsv = n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) + 4
      lnwk = lot*n

      call rfftmf(lot,jump,n,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('mcsqf1',-5)
        go to 400
      end if

      do 103 i=3,n,2
         do 203 m=1,lj,jump
            xim1 = 0.5D+00 * (x(m,i-1)+x(m,i))
            x(m,i) = 0.5D+00 * (x(m,i-1)-x(m,i))
            x(m,i-1) = xim1
 203     continue
  103 continue

  400 continue

  return
end
subroutine mcstb1(lot,jump,n,inc,x,wsave,dsum,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MCSTB1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum(*)
  real ( rk       ) fnm1s2
  real ( rk       ) fnm1s4
  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lj
  integer              lnsv
  integer              lnwk
  integer              lot
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1h
  real ( rk       ) x1p3
  real ( rk       ) x2
  real ( rk       ) xi

  ier = 0

      nm1 = n-1
      np1 = n+1
      ns2 = n/2
      lj = (lot-1)*jump+1
      if (n-2) 106,101,102

  101 do 111 m=1,lj,jump
         x1h = x(m,1)+x(m,2)
         x(m,2) = x(m,1)-x(m,2)
         x(m,1) = x1h
  111 continue

      return

  102 if ( 3 < n ) go to 103
      do 112 m=1,lj,jump
         x1p3 = x(m,1)+x(m,3)
         x2 = x(m,2)
         x(m,2) = x(m,1)-x(m,3)
         x(m,1) = x1p3+x2
         x(m,3) = x1p3-x2
  112 continue

      return

 103  do m=1,lj,jump
        x(m,1) = x(m,1)+x(m,1)
        x(m,n) = x(m,n)+x(m,n)
      end do

      m1 = 0

      do m=1,lj,jump
         m1 = m1+1
         dsum(m1) = x(m,1)-x(m,n)
         x(m,1) = x(m,1)+x(m,n)
      end do

      do 104 k=2,ns2
         m1 = 0
         do 114 m=1,lj,jump
           m1 = m1+1
           kc = np1-k
           t1 = x(m,k)+x(m,kc)
           t2 = x(m,k)-x(m,kc)
           dsum(m1) = dsum(m1)+wsave(kc)*t2
           t2 = wsave(k)*t2
           x(m,k) = t1-t2
           x(m,kc) = t1+t2
  114    continue
  104 continue

      modn = mod(n,2)
      if (modn == 0) go to 124
         do 123 m=1,lj,jump
         x(m,ns2+1) = x(m,ns2+1)+x(m,ns2+1)
  123    continue
 124  continue
      lenx = (lot-1)*jump + inc*(nm1-1)  + 1
      lnsv = nm1 + int(log( real ( nm1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = lot*nm1

      call rfftmf(lot,jump,nm1,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('mcstb1',-5)
        go to 106
      end if

      fnm1s2 = real ( nm1, rk       ) /  2.0D+00
      m1 = 0
      do 10 m=1,lj,jump
      m1 = m1+1
      dsum(m1) = 0.5D+00 * dsum(m1)
      x(m,1) = fnm1s2 * x(m,1)
   10 continue
      if(mod(nm1,2) /= 0) go to 30
      do 20 m=1,lj,jump
      x(m,nm1) = x(m,nm1)+x(m,nm1)
   20 continue
 30   fnm1s4 = real ( nm1, rk       ) / 4.0D+00
      do 105 i=3,n,2
         m1 = 0
         do 115 m=1,lj,jump
            m1 = m1+1
            xi = fnm1s4*x(m,i)
            x(m,i) = fnm1s4*x(m,i-1)
            x(m,i-1) = dsum(m1)
            dsum(m1) = dsum(m1)+xi
  115 continue
  105 continue
      if (modn /= 0) return
      m1 = 0
      do m=1,lj,jump
         m1 = m1+1
         x(m,n) = dsum(m1)
      end do
  106 continue

  return
end
subroutine mcstf1(lot,jump,n,inc,x,wsave,dsum,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MCSTF1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum(*)
  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lj
  integer              lnsv
  integer              lnwk
  integer              lot
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              nm1
  integer              np1
  integer              ns2
  real ( rk       ) snm1
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) tx2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) x1h
  real ( rk       ) x1p3
  real ( rk       ) xi

  ier = 0

  nm1 = n-1
  np1 = n+1
  ns2 = n/2
  lj = (lot-1)*jump+1

      if (n-2) 200,101,102
  101 do 111 m=1,lj,jump
         x1h = x(m,1)+x(m,2)
         x(m,2) = 0.5D+00 * (x(m,1)-x(m,2))
         x(m,1) = 0.5D+00 * x1h
  111 continue
      go to 200
  102 if ( 3 < n ) go to 103
      do 112 m=1,lj,jump
         x1p3 = x(m,1)+x(m,3)
         tx2 = x(m,2)+x(m,2)
         x(m,2) = 0.5D+00 * (x(m,1)-x(m,3))
         x(m,1) = 0.25D+00 * (x1p3+tx2)
         x(m,3) = 0.25D+00 * (x1p3-tx2)
  112 continue
      go to 200
  103 m1 = 0
      do 113 m=1,lj,jump
         m1 = m1+1
         dsum(m1) = x(m,1)-x(m,n)
         x(m,1) = x(m,1)+x(m,n)
  113 continue
      do 104 k=2,ns2
         m1 = 0
         do 114 m=1,lj,jump
         m1 = m1+1
         kc = np1-k
         t1 = x(m,k)+x(m,kc)
         t2 = x(m,k)-x(m,kc)
         dsum(m1) = dsum(m1)+wsave(kc)*t2
         t2 = wsave(k)*t2
         x(m,k) = t1-t2
         x(m,kc) = t1+t2
  114    continue
  104 continue
      modn = mod(n,2)
      if (modn == 0) go to 124
         do 123 m=1,lj,jump
         x(m,ns2+1) = x(m,ns2+1)+x(m,ns2+1)
  123    continue
 124  continue
      lenx = (lot-1)*jump + inc*(nm1-1)  + 1
      lnsv = nm1 + int(log( real ( nm1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = lot*nm1

      call rfftmf(lot,jump,nm1,inc,x,lenx,wsave(n+1),lnsv,work,lnwk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('mcstf1',-5)
        return
      end if

      snm1 = 1.0D+00 / real ( nm1, rk       )
      do 10 m=1,lot
      dsum(m) = snm1*dsum(m)
   10 continue
      if(mod(nm1,2) /= 0) go to 30
      do 20 m=1,lj,jump
      x(m,nm1) = x(m,nm1)+x(m,nm1)
   20 continue
 30   do 105 i=3,n,2
         m1 = 0
         do 115 m=1,lj,jump
            m1 = m1+1
            xi = 0.5D+00 * x(m,i)
            x(m,i) = 0.5D+00 * x(m,i-1)
            x(m,i-1) = dsum(m1)
            dsum(m1) = dsum(m1)+xi
  115 continue
  105 continue
      if (modn /= 0) go to 117

      m1 = 0
      do m=1,lj,jump
         m1 = m1+1
         x(m,n) = dsum(m1)
      end do

 117  continue

      do m=1,lj,jump
        x(m,1) = 0.5D+00 * x(m,1)
        x(m,n) = 0.5D+00 * x(m,n)
      end do

 200  continue

  return
end
subroutine mradb2 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADB2 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,2,l1)
  real ( rk       ) ch(in2,ido,l1,2)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) wa1(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2

  do k = 1, l1
    m2 = m2s
    do m1=1,m1d,im1
      m2 = m2+im2
      ch(m2,1,k,1) = cc(m1,1,1,k)+cc(m1,ido,2,k)
      ch(m2,1,k,2) = cc(m1,1,1,k)-cc(m1,ido,2,k)
    end do
  end do

      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
               m2 = m2s
               do 1002 m1=1,m1d,im1
               m2 = m2+im2
        ch(m2,i-1,k,1) = cc(m1,i-1,1,k)+cc(m1,ic-1,2,k)
        ch(m2,i,k,1) = cc(m1,i,1,k)-cc(m1,ic,2,k)
        ch(m2,i-1,k,2) = wa1(i-2)*(cc(m1,i-1,1,k)-cc(m1,ic-1,2,k)) &
        -wa1(i-1)*(cc(m1,i,1,k)+cc(m1,ic,2,k))

        ch(m2,i,k,2) = wa1(i-2)*(cc(m1,i,1,k)+cc(m1,ic,2,k))+wa1(i-1) &
        *(cc(m1,i-1,1,k)-cc(m1,ic-1,2,k))

 1002          continue
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 do 106 k = 1, l1
          m2 = m2s
          do 1003 m1=1,m1d,im1
          m2 = m2+im2
         ch(m2,ido,k,1) = cc(m1,ido,1,k)+cc(m1,ido,1,k)
         ch(m2,ido,k,2) = -(cc(m1,1,2,k)+cc(m1,1,2,k))
 1003     continue
  106 continue
  107 continue

  return
end
subroutine mradb3 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADB3 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,3,l1)
  real ( rk       ) ch(in2,ido,l1,3)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) taui
  real ( rk       ) taur
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2
  arg= 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 3.0D+00
  taur=cos(arg)
  taui=sin(arg)

  do k = 1, l1
    m2 = m2s
    do m1=1,m1d,im1
      m2 = m2+im2
      ch(m2,1,k,1) = cc(m1,1,1,k)+ 2.0D+00 *cc(m1,ido,2,k)
      ch(m2,1,k,2) = cc(m1,1,1,k)+( 2.0D+00 *taur)*cc(m1,ido,2,k) &
        -( 2.0D+00 *taui)*cc(m1,1,3,k)
      ch(m2,1,k,3) = cc(m1,1,1,k)+( 2.0D+00 *taur)*cc(m1,ido,2,k) &
        + 2.0D+00 *taui*cc(m1,1,3,k)
    end do
  end do

  if (ido == 1) then
    return
  end if

  idp2 = ido+2

      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
               m2 = m2s
               do 1002 m1=1,m1d,im1
               m2 = m2+im2
        ch(m2,i-1,k,1) = cc(m1,i-1,1,k)+(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))
        ch(m2,i,k,1) = cc(m1,i,1,k)+(cc(m1,i,3,k)-cc(m1,ic,2,k))

        ch(m2,i-1,k,2) = wa1(i-2)* &
       ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))- &
       (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k)))) &
                         -wa1(i-1)* &
       ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))+ &
       (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))))

            ch(m2,i,k,2) = wa1(i-2)* &
       ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))+ &
       (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))) &
                        +wa1(i-1)* &
       ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))- &
       (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k))))

              ch(m2,i-1,k,3) = wa2(i-2)* &
       ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))+ &
       (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k)))) &
         -wa2(i-1)* &
       ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))- &
       (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))))

            ch(m2,i,k,3) = wa2(i-2)* &
       ((cc(m1,i,1,k)+taur*(cc(m1,i,3,k)-cc(m1,ic,2,k)))- &
       (taui*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))) &
                       +wa2(i-1)* &
       ((cc(m1,i-1,1,k)+taur*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))+ &
       (taui*(cc(m1,i,3,k)+cc(m1,ic,2,k))))

 1002          continue
  102    continue
  103 continue

  return
end
subroutine mradb4 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2,wa3)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADB4 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,4,l1)
  real ( rk       ) ch(in2,ido,l1,4)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) sqrt2
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)

      m1d = (m-1)*im1+1
      m2s = 1-im2
      sqrt2=sqrt( 2.0D+00 )
      do 101 k = 1, l1
          m2 = m2s
          do m1=1,m1d,im1
          m2 = m2+im2
         ch(m2,1,k,3) = (cc(m1,1,1,k)+cc(m1,ido,4,k)) &
         -(cc(m1,ido,2,k)+cc(m1,ido,2,k))
         ch(m2,1,k,1) = (cc(m1,1,1,k)+cc(m1,ido,4,k)) &
         +(cc(m1,ido,2,k)+cc(m1,ido,2,k))
         ch(m2,1,k,4) = (cc(m1,1,1,k)-cc(m1,ido,4,k)) &
         +(cc(m1,1,3,k)+cc(m1,1,3,k))
         ch(m2,1,k,2) = (cc(m1,1,1,k)-cc(m1,ido,4,k)) &
         -(cc(m1,1,3,k)+cc(m1,1,3,k))
        end do
  101 continue
      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
               m2 = m2s
               do 1002 m1=1,m1d,im1
               m2 = m2+im2
        ch(m2,i-1,k,1) = (cc(m1,i-1,1,k)+cc(m1,ic-1,4,k)) &
        +(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))
        ch(m2,i,k,1) = (cc(m1,i,1,k)-cc(m1,ic,4,k)) &
        +(cc(m1,i,3,k)-cc(m1,ic,2,k))
        ch(m2,i-1,k,2)=wa1(i-2)*((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k)) &
        -(cc(m1,i,3,k)+cc(m1,ic,2,k)))-wa1(i-1) &
        *((cc(m1,i,1,k)+cc(m1,ic,4,k))+(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))
        ch(m2,i,k,2)=wa1(i-2)*((cc(m1,i,1,k)+cc(m1,ic,4,k)) &
        +(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))+wa1(i-1) &
        *((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k))-(cc(m1,i,3,k)+cc(m1,ic,2,k)))
        ch(m2,i-1,k,3)=wa2(i-2)*((cc(m1,i-1,1,k)+cc(m1,ic-1,4,k)) &
        -(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)))-wa2(i-1) &
        *((cc(m1,i,1,k)-cc(m1,ic,4,k))-(cc(m1,i,3,k)-cc(m1,ic,2,k)))
        ch(m2,i,k,3)=wa2(i-2)*((cc(m1,i,1,k)-cc(m1,ic,4,k)) &
        -(cc(m1,i,3,k)-cc(m1,ic,2,k)))+wa2(i-1) &
        *((cc(m1,i-1,1,k)+cc(m1,ic-1,4,k))-(cc(m1,i-1,3,k) &
        +cc(m1,ic-1,2,k)))
        ch(m2,i-1,k,4)=wa3(i-2)*((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k)) &
        +(cc(m1,i,3,k)+cc(m1,ic,2,k)))-wa3(i-1) &
       *((cc(m1,i,1,k)+cc(m1,ic,4,k))-(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))
        ch(m2,i,k,4)=wa3(i-2)*((cc(m1,i,1,k)+cc(m1,ic,4,k)) &
        -(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k)))+wa3(i-1) &
        *((cc(m1,i-1,1,k)-cc(m1,ic-1,4,k))+(cc(m1,i,3,k)+cc(m1,ic,2,k)))
 1002          continue
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 continue
      do 106 k = 1, l1
               m2 = m2s
               do 1003 m1=1,m1d,im1
               m2 = m2+im2
         ch(m2,ido,k,1) = (cc(m1,ido,1,k)+cc(m1,ido,3,k)) &
         +(cc(m1,ido,1,k)+cc(m1,ido,3,k))
         ch(m2,ido,k,2) = sqrt2*((cc(m1,ido,1,k)-cc(m1,ido,3,k)) &
         -(cc(m1,1,2,k)+cc(m1,1,4,k)))
         ch(m2,ido,k,3) = (cc(m1,1,4,k)-cc(m1,1,2,k)) &
         +(cc(m1,1,4,k)-cc(m1,1,2,k))
         ch(m2,ido,k,4) = -sqrt2*((cc(m1,ido,1,k)-cc(m1,ido,3,k)) &
         +(cc(m1,1,2,k)+cc(m1,1,4,k)))
 1003          continue
  106 continue
  107 continue

  return
end
subroutine mradb5 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2,wa3,wa4)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADB5 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,5,l1)
  real ( rk       ) ch(in2,ido,l1,5)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) ti11
  real ( rk       ) ti12
  real ( rk       ) tr11
  real ( rk       ) tr12
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)
  real ( rk       ) wa4(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2
  arg= 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 5.0D+00 
  tr11=cos(arg)
  ti11=sin(arg)
  tr12=cos( 2.0D+00 *arg)
  ti12=sin( 2.0D+00 *arg)

      do 101 k = 1, l1
      m2 = m2s
      do 1001 m1=1,m1d,im1
         m2 = m2+im2
         ch(m2,1,k,1) = cc(m1,1,1,k)+ 2.0D+00 *cc(m1,ido,2,k) &
           + 2.0D+00 *cc(m1,ido,4,k)
         ch(m2,1,k,2) = (cc(m1,1,1,k)+tr11* 2.0D+00 *cc(m1,ido,2,k) &
         +tr12* 2.0D+00 *cc(m1,ido,4,k))-(ti11* 2.0D+00 *cc(m1,1,3,k) &
         +ti12* 2.0D+00 *cc(m1,1,5,k))
         ch(m2,1,k,3) = (cc(m1,1,1,k)+tr12* 2.0D+00 *cc(m1,ido,2,k) &
         +tr11* 2.0D+00 *cc(m1,ido,4,k))-(ti12* 2.0D+00 *cc(m1,1,3,k) &
         -ti11* 2.0D+00 *cc(m1,1,5,k))
         ch(m2,1,k,4) = (cc(m1,1,1,k)+tr12* 2.0D+00 *cc(m1,ido,2,k) &
         +tr11* 2.0D+00 *cc(m1,ido,4,k))+(ti12* 2.0D+00 *cc(m1,1,3,k) &
         -ti11* 2.0D+00 *cc(m1,1,5,k))
         ch(m2,1,k,5) = (cc(m1,1,1,k)+tr11* 2.0D+00 *cc(m1,ido,2,k) &
         +tr12* 2.0D+00 *cc(m1,ido,4,k))+(ti11* 2.0D+00 *cc(m1,1,3,k) &
         +ti12* 2.0D+00 *cc(m1,1,5,k))
 1001          continue
  101 continue

      if (ido == 1) return
      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
            m2 = m2s
      do 1002 m1=1,m1d,im1
        m2 = m2+im2
        ch(m2,i-1,k,1) = cc(m1,i-1,1,k)+(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k))
        ch(m2,i,k,1) = cc(m1,i,1,k)+(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
        +(cc(m1,i,5,k)-cc(m1,ic,4,k))
        ch(m2,i-1,k,2) = wa1(i-2)*((cc(m1,i-1,1,k)+tr11* &
       (cc(m1,i-1,3,k)+cc(m1,ic-1,2,k))+tr12 &
        *(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti11*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
        -wa1(i-1)*((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
        +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))+(ti11*(cc(m1,i-1,3,k) &
        -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) 

        ch(m2,i,k,2) = wa1(i-2)*((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k) &
        -cc(m1,ic,2,k))+tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
        +(ti11*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))+ti12 &
        *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))+wa1(i-1) &
        *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k) &
        +cc(m1,ic-1,2,k))+tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k))) &
        -(ti11*(cc(m1,i,3,k)+cc(m1,ic,2,k))+ti12 &
        *(cc(m1,i,5,k)+cc(m1,ic,4,k))))
        ch(m2,i-1,k,3) = wa2(i-2) &
        *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti12*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
       -wa2(i-1) &
       *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
        cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
        +(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
        *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))

        ch(m2,i,k,3) = wa2(i-2) &
       *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
        cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
        +(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
        *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
        +wa2(i-1) &
        *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))-(ti12*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k))))

        ch(m2,i-1,k,4) = wa3(i-2) &
        *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti12*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
        -wa3(i-1) &
       *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
        cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
        -(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
        *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))

        ch(m2,i,k,4) = wa3(i-2) &
       *((cc(m1,i,1,k)+tr12*(cc(m1,i,3,k)- &
        cc(m1,ic,2,k))+tr11*(cc(m1,i,5,k)-cc(m1,ic,4,k))) &
        -(ti12*(cc(m1,i-1,3,k)-cc(m1,ic-1,2,k))-ti11 &
        *(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
        +wa3(i-1) &
        *((cc(m1,i-1,1,k)+tr12*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr11*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti12*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))-ti11*(cc(m1,i,5,k)+cc(m1,ic,4,k))))

        ch(m2,i-1,k,5) = wa4(i-2) &
        *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti11*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k)))) &
        -wa4(i-1) &
        *((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
        +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))-(ti11*(cc(m1,i-1,3,k) &
        -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k))))

        ch(m2,i,k,5) = wa4(i-2) &
        *((cc(m1,i,1,k)+tr11*(cc(m1,i,3,k)-cc(m1,ic,2,k)) &
        +tr12*(cc(m1,i,5,k)-cc(m1,ic,4,k)))-(ti11*(cc(m1,i-1,3,k) &
        -cc(m1,ic-1,2,k))+ti12*(cc(m1,i-1,5,k)-cc(m1,ic-1,4,k)))) &
        +wa4(i-1) &
        *((cc(m1,i-1,1,k)+tr11*(cc(m1,i-1,3,k)+cc(m1,ic-1,2,k)) &
        +tr12*(cc(m1,i-1,5,k)+cc(m1,ic-1,4,k)))+(ti11*(cc(m1,i,3,k) &
        +cc(m1,ic,2,k))+ti12*(cc(m1,i,5,k)+cc(m1,ic,4,k))))

 1002      continue
  102    continue
  103 continue

  return
end
subroutine mradbg (m,ido,ip,l1,idl1,cc,c1,c2,im1,in1,ch,ch2,im2,in2,wa)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADBG is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              idl1
  integer              ido
  integer              in1
  integer              in2
  integer              ip
  integer              l1

  real ( rk       ) ai1
  real ( rk       ) ai2
  real ( rk       ) ar1
  real ( rk       ) ar1h
  real ( rk       ) ar2
  real ( rk       ) ar2h
  real ( rk       ) arg
  real ( rk       ) c1(in1,ido,l1,ip)
  real ( rk       ) c2(in1,idl1,ip)
  real ( rk       ) cc(in1,ido,ip,l1)
  real ( rk       ) ch(in2,ido,l1,ip)
  real ( rk       ) ch2(in2,idl1,ip)
  real ( rk       ) dc2
  real ( rk       ) dcp
  real ( rk       ) ds2
  real ( rk       ) dsp
  integer              i
  integer              ic
  integer              idij
  integer              idp2
  integer              ik
  integer              im1
  integer              im2
  integer              ipp2
  integer              ipph
  integer              is
  integer              j
  integer              j2
  integer              jc
  integer              k
  integer              l
  integer              lc
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  integer              nbd
  real ( rk       ) tpi
  real ( rk       ) wa(ido)

  m1d = ( m - 1 ) * im1 + 1
  m2s = 1 - im2
  tpi = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 )
  arg = tpi / real ( ip, rk       )
  dcp = cos ( arg )
  dsp = sin ( arg )
  idp2 = ido + 2
  nbd = (ido-1)/2
  ipp2 = ip+2
  ipph = (ip+1)/2

      if (ido < l1) go to 103

      do k = 1, l1
         do i=1,ido
            m2 = m2s
            do m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,i,k,1) = cc(m1,i,1,k)
            end do
      end do
  end do

      go to 106

  103 do 105 i=1,ido
         do 104 k = 1, l1
            m2 = m2s
            do 1004 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,i,k,1) = cc(m1,i,1,k)
 1004       continue
  104    continue
  105 continue
  106 do 108 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 107 k = 1, l1
            m2 = m2s
            do 1007 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,1,k,j) = cc(m1,ido,j2-2,k)+cc(m1,ido,j2-2,k)
            ch(m2,1,k,jc) = cc(m1,1,j2-1,k)+cc(m1,1,j2-1,k)
 1007       continue
  107    continue
  108 continue
      if (ido == 1) go to 116
      if (nbd < l1) go to 112
      do 111 j=2,ipph
         jc = ipp2-j
         do 110 k = 1, l1
            do 109 i=3,ido,2
               ic = idp2-i
               m2 = m2s
               do 1009 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = cc(m1,i-1,2*j-1,k)+cc(m1,ic-1,2*j-2,k)
               ch(m2,i-1,k,jc) = cc(m1,i-1,2*j-1,k)-cc(m1,ic-1,2*j-2,k)
               ch(m2,i,k,j) = cc(m1,i,2*j-1,k)-cc(m1,ic,2*j-2,k)
               ch(m2,i,k,jc) = cc(m1,i,2*j-1,k)+cc(m1,ic,2*j-2,k)
 1009          continue
  109       continue
  110    continue
  111 continue
      go to 116
  112 do 115 j=2,ipph
         jc = ipp2-j
         do 114 i=3,ido,2
            ic = idp2-i
            do 113 k = 1, l1
               m2 = m2s
               do 1013 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = cc(m1,i-1,2*j-1,k)+cc(m1,ic-1,2*j-2,k)
               ch(m2,i-1,k,jc) = cc(m1,i-1,2*j-1,k)-cc(m1,ic-1,2*j-2,k)
               ch(m2,i,k,j) = cc(m1,i,2*j-1,k)-cc(m1,ic,2*j-2,k)
               ch(m2,i,k,jc) = cc(m1,i,2*j-1,k)+cc(m1,ic,2*j-2,k)
 1013          continue
  113       continue
  114    continue
  115 continue
  116 ar1 = 1.0D+00
      ai1 = 0.0D+00
      do 120 l=2,ipph
         lc = ipp2-l
         ar1h = dcp*ar1-dsp*ai1
         ai1 = dcp*ai1+dsp*ar1
         ar1 = ar1h
         do 117 ik=1,idl1
            m2 = m2s
            do 1017 m1=1,m1d,im1
            m2 = m2+im2
            c2(m1,ik,l) = ch2(m2,ik,1)+ar1*ch2(m2,ik,2)
            c2(m1,ik,lc) = ai1*ch2(m2,ik,ip)
 1017       continue
  117    continue
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do 119 j=3,ipph
            jc = ipp2-j
            ar2h = dc2*ar2-ds2*ai2
            ai2 = dc2*ai2+ds2*ar2
            ar2 = ar2h
            do 118 ik=1,idl1
               m2 = m2s
               do 1018 m1=1,m1d,im1
               m2 = m2+im2
               c2(m1,ik,l) = c2(m1,ik,l)+ar2*ch2(m2,ik,j)
               c2(m1,ik,lc) = c2(m1,ik,lc)+ai2*ch2(m2,ik,jc)
 1018          continue
  118       continue
  119    continue
  120 continue
      do 122 j=2,ipph
         do 121 ik=1,idl1
            m2 = m2s
            do 1021 m1=1,m1d,im1
            m2 = m2+im2
            ch2(m2,ik,1) = ch2(m2,ik,1)+ch2(m2,ik,j)
 1021       continue
  121    continue
  122 continue
      do 124 j=2,ipph
         jc = ipp2-j
         do 123 k = 1, l1
            m2 = m2s
            do 1023 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,1,k,j) = c1(m1,1,k,j)-c1(m1,1,k,jc)
            ch(m2,1,k,jc) = c1(m1,1,k,j)+c1(m1,1,k,jc)
 1023       continue
  123    continue
  124 continue
      if (ido == 1) go to 132
      if (nbd < l1) go to 128
      do 127 j=2,ipph
         jc = ipp2-j
         do 126 k = 1, l1
            do 125 i=3,ido,2
               m2 = m2s
               do 1025 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = c1(m1,i-1,k,j)-c1(m1,i,k,jc)
               ch(m2,i-1,k,jc) = c1(m1,i-1,k,j)+c1(m1,i,k,jc)
               ch(m2,i,k,j) = c1(m1,i,k,j)+c1(m1,i-1,k,jc)
               ch(m2,i,k,jc) = c1(m1,i,k,j)-c1(m1,i-1,k,jc)
 1025          continue
  125       continue
  126    continue
  127 continue
      go to 132
  128 do 131 j=2,ipph
         jc = ipp2-j
         do 130 i=3,ido,2
            do 129 k = 1, l1
               m2 = m2s
               do 1029 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = c1(m1,i-1,k,j)-c1(m1,i,k,jc)
               ch(m2,i-1,k,jc) = c1(m1,i-1,k,j)+c1(m1,i,k,jc)
               ch(m2,i,k,j) = c1(m1,i,k,j)+c1(m1,i-1,k,jc)
               ch(m2,i,k,jc) = c1(m1,i,k,j)-c1(m1,i-1,k,jc)
 1029          continue
  129       continue
  130    continue
  131 continue
  132 continue
      if (ido == 1) return
      do 133 ik=1,idl1
         m2 = m2s
         do 1033 m1=1,m1d,im1
         m2 = m2+im2
         c2(m1,ik,1) = ch2(m2,ik,1)
 1033    continue
  133 continue
      do 135 j=2,ip
         do 134 k = 1, l1
            m2 = m2s
            do 1034 m1=1,m1d,im1
            m2 = m2+im2
            c1(m1,1,k,j) = ch(m2,1,k,j)
 1034       continue
  134    continue
  135 continue
      if (l1 < nbd ) go to 139
      is = -ido
      do 138 j=2,ip
         is = is+ido
         idij = is
         do 137 i=3,ido,2
            idij = idij+2
            do 136 k = 1, l1
               m2 = m2s
               do 1036 m1=1,m1d,im1
               m2 = m2+im2
               c1(m1,i-1,k,j) = wa(idij-1)*ch(m2,i-1,k,j)-wa(idij)* ch(m2,i,k,j)
               c1(m1,i,k,j) = wa(idij-1)*ch(m2,i,k,j)+wa(idij)* ch(m2,i-1,k,j)
 1036          continue
  136       continue
  137    continue
  138 continue
      go to 143
  139 is = -ido
      do 142 j=2,ip
         is = is+ido
         do 141 k = 1, l1
            idij = is
            do 140 i=3,ido,2
               idij = idij+2
               m2 = m2s
               do 1040 m1=1,m1d,im1
               m2 = m2+im2
               c1(m1,i-1,k,j) = wa(idij-1)*ch(m2,i-1,k,j)-wa(idij)*ch(m2,i,k,j)
               c1(m1,i,k,j) = wa(idij-1)*ch(m2,i,k,j)+wa(idij)*ch(m2,i-1,k,j)
 1040          continue
  140       continue
  141    continue
  142 continue
  143 continue

  return
end
subroutine mradf2 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADF2 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,l1,2)
  real ( rk       ) ch(in2,ido,2,l1)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) wa1(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2

      do 101 k = 1, l1
         m2 = m2s
         do m1=1,m1d,im1
           m2 = m2+im2
           ch(m2,1,1,k) = cc(m1,1,k,1)+cc(m1,1,k,2)
           ch(m2,ido,2,k) = cc(m1,1,k,1)-cc(m1,1,k,2)
         end do
  101 continue

      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
            m2 = m2s
            do 1003 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,i,1,k) = cc(m1,i,k,1)+(wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))
            ch(m2,ic,2,k) = (wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
             cc(m1,i-1,k,2))-cc(m1,i,k,1)
            ch(m2,i-1,1,k) = cc(m1,i-1,k,1)+(wa1(i-2)*cc(m1,i-1,k,2)+ &
             wa1(i-1)*cc(m1,i,k,2))
            ch(m2,ic-1,2,k) = cc(m1,i-1,k,1)-(wa1(i-2)*cc(m1,i-1,k,2)+ &
             wa1(i-1)*cc(m1,i,k,2))
 1003       continue
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 do 106 k = 1, l1
         m2 = m2s
         do 1006 m1=1,m1d,im1
         m2 = m2+im2
         ch(m2,1,2,k) = -cc(m1,ido,k,2)
         ch(m2,ido,1,k) = cc(m1,ido,k,1)
 1006    continue
  106 continue
  107 continue

  return
end
subroutine mradf3 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADF3 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,l1,3)
  real ( rk       ) ch(in2,ido,3,l1)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) taui
  real ( rk       ) taur
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2
  arg= 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 3.0D+00
  taur=cos(arg)
  taui=sin(arg)

      do 101 k = 1, l1
         m2 = m2s
         do 1001 m1=1,m1d,im1
         m2 = m2+im2
         ch(m2,1,1,k) = cc(m1,1,k,1)+(cc(m1,1,k,2)+cc(m1,1,k,3))
         ch(m2,1,3,k) = taui*(cc(m1,1,k,3)-cc(m1,1,k,2))
         ch(m2,ido,2,k) = cc(m1,1,k,1)+taur*(cc(m1,1,k,2)+cc(m1,1,k,3))
 1001    continue
  101 continue

  if (ido == 1) then
    return
  end if

      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
            m2 = m2s
            do 1002 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,i-1,1,k) = cc(m1,i-1,k,1)+((wa1(i-2)*cc(m1,i-1,k,2)+ &
             wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3)))

            ch(m2,i,1,k) = cc(m1,i,k,1)+((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3)))

            ch(m2,i-1,3,k) = (cc(m1,i-1,k,1)+taur*((wa1(i-2)* &
             cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)* &
             cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))))+(taui*((wa1(i-2)* &
             cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa2(i-2)* &
             cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))))

            ch(m2,ic-1,2,k) = (cc(m1,i-1,k,1)+taur*((wa1(i-2)* &
             cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa2(i-2)* &
             cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))))-(taui*((wa1(i-2)* &
             cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa2(i-2)* &
             cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))))

            ch(m2,i,3,k) = (cc(m1,i,k,1)+taur*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))))+(taui*((wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2))))

            ch(m2,ic,2,k) = (taui*((wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2))))-(cc(m1,i,k,1)+taur*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))))
 1002       continue
  102    continue
  103 continue

  return
end
subroutine mradf4 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2,wa3)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADF4 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,l1,4)
  real ( rk       ) ch(in2,ido,4,l1)
  real ( rk       ) hsqt2
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)

  hsqt2=sqrt( 2.0D+00 ) / 2.0D+00 
  m1d = (m-1)*im1+1
  m2s = 1-im2

      do 101 k = 1, l1
         m2 = m2s
         do 1001 m1=1,m1d,im1
         m2 = m2+im2
         ch(m2,1,1,k) = (cc(m1,1,k,2)+cc(m1,1,k,4)) &
            +(cc(m1,1,k,1)+cc(m1,1,k,3))
         ch(m2,ido,4,k) = (cc(m1,1,k,1)+cc(m1,1,k,3)) &
            -(cc(m1,1,k,2)+cc(m1,1,k,4))
         ch(m2,ido,2,k) = cc(m1,1,k,1)-cc(m1,1,k,3)
         ch(m2,1,3,k) = cc(m1,1,k,4)-cc(m1,1,k,2)
 1001    continue
  101 continue

      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
            m2 = m2s
            do 1003 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,i-1,1,k) = ((wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2))+(wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4)))+(cc(m1,i-1,k,1)+(wa2(i-2)*cc(m1,i-1,k,3)+ &
             wa2(i-1)*cc(m1,i,k,3)))

            ch(m2,ic-1,4,k) = (cc(m1,i-1,k,1)+(wa2(i-2)*cc(m1,i-1,k,3)+ &
             wa2(i-1)*cc(m1,i,k,3)))-((wa1(i-2)*cc(m1,i-1,k,2)+ &
             wa1(i-1)*cc(m1,i,k,2))+(wa3(i-2)*cc(m1,i-1,k,4)+ &
             wa3(i-1)*cc(m1,i,k,4)))

            ch(m2,i,1,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
             cc(m1,i-1,k,2))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4)))+(cc(m1,i,k,1)+(wa2(i-2)*cc(m1,i,k,3)- &
             wa2(i-1)*cc(m1,i-1,k,3)))

            ch(m2,ic,4,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
             cc(m1,i-1,k,2))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4)))-(cc(m1,i,k,1)+(wa2(i-2)*cc(m1,i,k,3)- &
             wa2(i-1)*cc(m1,i-1,k,3)))

            ch(m2,i-1,3,k) = ((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
             cc(m1,i-1,k,2))-(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4)))+(cc(m1,i-1,k,1)-(wa2(i-2)*cc(m1,i-1,k,3)+ &
             wa2(i-1)*cc(m1,i,k,3)))

            ch(m2,ic-1,2,k) = (cc(m1,i-1,k,1)-(wa2(i-2)*cc(m1,i-1,k,3)+ &
             wa2(i-1)*cc(m1,i,k,3)))-((wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)* &
             cc(m1,i-1,k,2))-(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4)))

            ch(m2,i,3,k) = ((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))+(cc(m1,i,k,1)-(wa2(i-2)*cc(m1,i,k,3)- &
             wa2(i-1)*cc(m1,i-1,k,3)))

            ch(m2,ic,2,k) = ((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))-(cc(m1,i,k,1)-(wa2(i-2)*cc(m1,i,k,3)- &
             wa2(i-1)*cc(m1,i-1,k,3)))

 1003       continue
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 continue
      do 106 k = 1, l1
         m2 = m2s
         do 1006 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,ido,1,k) = (hsqt2*(cc(m1,ido,k,2)-cc(m1,ido,k,4)))+ &
             cc(m1,ido,k,1)
            ch(m2,ido,3,k) = cc(m1,ido,k,1)-(hsqt2*(cc(m1,ido,k,2)- &
             cc(m1,ido,k,4)))
            ch(m2,1,2,k) = (-hsqt2*(cc(m1,ido,k,2)+cc(m1,ido,k,4)))- &
             cc(m1,ido,k,3)
            ch(m2,1,4,k) = (-hsqt2*(cc(m1,ido,k,2)+cc(m1,ido,k,4)))+ &
             cc(m1,ido,k,3)
 1006    continue
  106 continue
  107 continue

  return
end
subroutine mradf5 (m,ido,l1,cc,im1,in1,ch,im2,in2,wa1,wa2,wa3,wa4)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADF5 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,l1,5)
  real ( rk       ) ch(in2,ido,5,l1)
  integer              i
  integer              ic
  integer              idp2
  integer              im1
  integer              im2
  integer              k
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  real ( rk       ) ti11
  real ( rk       ) ti12
  real ( rk       ) tr11
  real ( rk       ) tr12
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)
  real ( rk       ) wa4(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2
  arg= 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 5.0D+00
  tr11=cos(arg)
  ti11=sin(arg)
  tr12=cos( 2.0D+00 *arg)
  ti12=sin( 2.0D+00 *arg)

      do 101 k = 1, l1
         m2 = m2s
         do 1001 m1=1,m1d,im1
         m2 = m2+im2
         ch(m2,1,1,k) = cc(m1,1,k,1)+(cc(m1,1,k,5)+cc(m1,1,k,2))+ &
          (cc(m1,1,k,4)+cc(m1,1,k,3))
         ch(m2,ido,2,k) = cc(m1,1,k,1)+tr11*(cc(m1,1,k,5)+cc(m1,1,k,2))+ &
          tr12*(cc(m1,1,k,4)+cc(m1,1,k,3))
         ch(m2,1,3,k) = ti11*(cc(m1,1,k,5)-cc(m1,1,k,2))+ti12* &
          (cc(m1,1,k,4)-cc(m1,1,k,3))
         ch(m2,ido,4,k) = cc(m1,1,k,1)+tr12*(cc(m1,1,k,5)+cc(m1,1,k,2))+ &
          tr11*(cc(m1,1,k,4)+cc(m1,1,k,3))
         ch(m2,1,5,k) = ti12*(cc(m1,1,k,5)-cc(m1,1,k,2))-ti11* &
          (cc(m1,1,k,4)-cc(m1,1,k,3))
 1001    continue
  101 continue

      if (ido == 1) return
      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
            m2 = m2s
            do 1002 m1=1,m1d,im1
            m2 = m2+im2

            ch(m2,i-1,1,k) = cc(m1,i-1,k,1)+((wa1(i-2)*cc(m1,i-1,k,2)+ &
             wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
             cc(m1,i,k,5)))+((wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))+(wa3(i-2)*cc(m1,i-1,k,4)+ &
             wa3(i-1)*cc(m1,i,k,4)))

            ch(m2,i,1,k) = cc(m1,i,k,1)+((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
             cc(m1,i-1,k,5)))+((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4)))

            ch(m2,i-1,3,k) = cc(m1,i-1,k,1)+tr11* &
            ( wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2) &
             +wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5))+tr12* &
            ( wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3) &
             +wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))+ti11* &
            ( wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2) &
             -(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))+ti12* &
            ( wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3) &
             -(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4)))

            ch(m2,ic-1,2,k) = cc(m1,i-1,k,1)+tr11* &
            ( wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2) &
             +wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5))+tr12* &
           ( wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3) &
            +wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))-(ti11* &
            ( wa1(i-2)*cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2) &
             -(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))+ti12* &
            ( wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3) &
             -(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))

            ch(m2,i,3,k) = (cc(m1,i,k,1)+tr11*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
             cc(m1,i-1,k,5)))+tr12*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4))))+(ti11*((wa4(i-2)*cc(m1,i-1,k,5)+ &
             wa4(i-1)*cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))+ti12*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))))

            ch(m2,ic,2,k) = (ti11*((wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
             cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))+ti12*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))))-(cc(m1,i,k,1)+tr11*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
             cc(m1,i-1,k,5)))+tr12*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4))))

            ch(m2,i-1,5,k) = (cc(m1,i-1,k,1)+tr12*((wa1(i-2)* &
             cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)* &
             cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5)))+tr11*((wa2(i-2)* &
             cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))+(wa3(i-2)* &
             cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))))+(ti12*((wa1(i-2)* &
             cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa4(i-2)* &
             cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))-ti11*((wa2(i-2)* &
             cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))-(wa3(i-2)* &
             cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))

            ch(m2,ic-1,4,k) = (cc(m1,i-1,k,1)+tr12*((wa1(i-2)* &
             cc(m1,i-1,k,2)+wa1(i-1)*cc(m1,i,k,2))+(wa4(i-2)* &
             cc(m1,i-1,k,5)+wa4(i-1)*cc(m1,i,k,5)))+tr11*((wa2(i-2)* &
             cc(m1,i-1,k,3)+wa2(i-1)*cc(m1,i,k,3))+(wa3(i-2)* &
             cc(m1,i-1,k,4)+wa3(i-1)*cc(m1,i,k,4))))-(ti12*((wa1(i-2)* &
             cc(m1,i,k,2)-wa1(i-1)*cc(m1,i-1,k,2))-(wa4(i-2)* &
             cc(m1,i,k,5)-wa4(i-1)*cc(m1,i-1,k,5)))-ti11*((wa2(i-2)* &
             cc(m1,i,k,3)-wa2(i-1)*cc(m1,i-1,k,3))-(wa3(i-2)* &
             cc(m1,i,k,4)-wa3(i-1)*cc(m1,i-1,k,4))))

            ch(m2,i,5,k) = (cc(m1,i,k,1)+tr12*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
             cc(m1,i-1,k,5)))+tr11*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4))))+(ti12*((wa4(i-2)*cc(m1,i-1,k,5)+ &
             wa4(i-1)*cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))-ti11*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))))

            ch(m2,ic,4,k) = (ti12*((wa4(i-2)*cc(m1,i-1,k,5)+wa4(i-1)* &
             cc(m1,i,k,5))-(wa1(i-2)*cc(m1,i-1,k,2)+wa1(i-1)* &
             cc(m1,i,k,2)))-ti11*((wa3(i-2)*cc(m1,i-1,k,4)+wa3(i-1)* &
             cc(m1,i,k,4))-(wa2(i-2)*cc(m1,i-1,k,3)+wa2(i-1)* &
             cc(m1,i,k,3))))-(cc(m1,i,k,1)+tr12*((wa1(i-2)*cc(m1,i,k,2)- &
             wa1(i-1)*cc(m1,i-1,k,2))+(wa4(i-2)*cc(m1,i,k,5)-wa4(i-1)* &
             cc(m1,i-1,k,5)))+tr11*((wa2(i-2)*cc(m1,i,k,3)-wa2(i-1)* &
             cc(m1,i-1,k,3))+(wa3(i-2)*cc(m1,i,k,4)-wa3(i-1)* &
             cc(m1,i-1,k,4))))
 1002       continue
  102    continue
  103 continue

  return
end
subroutine mradfg (m,ido,ip,l1,idl1,cc,c1,c2,im1,in1,ch,ch2,im2,in2,wa)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRADFG is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              idl1
  integer              ido
  integer              in1
  integer              in2
  integer              ip
  integer              l1

  real ( rk       ) ai1
  real ( rk       ) ai2
  real ( rk       ) ar1
  real ( rk       ) ar1h
  real ( rk       ) ar2
  real ( rk       ) ar2h
  real ( rk       ) arg
  real ( rk       ) c1(in1,ido,l1,ip)
  real ( rk       ) c2(in1,idl1,ip)
  real ( rk       ) cc(in1,ido,ip,l1)
  real ( rk       ) ch(in2,ido,l1,ip)
  real ( rk       ) ch2(in2,idl1,ip)
  real ( rk       ) dc2
  real ( rk       ) dcp
  real ( rk       ) ds2
  real ( rk       ) dsp
  integer              i
  integer              ic
  integer              idij
  integer              idp2
  integer              ik
  integer              im1
  integer              im2
  integer              ipp2
  integer              ipph
  integer              is
  integer              j
  integer              j2
  integer              jc
  integer              k
  integer              l
  integer              lc
  integer              m
  integer              m1
  integer              m1d
  integer              m2
  integer              m2s
  integer              nbd
  real ( rk       ) tpi
  real ( rk       ) wa(ido)

  m1d = (m-1)*im1+1
  m2s = 1-im2
  tpi= 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 )
  arg = tpi / real ( ip, rk       )
  dcp = cos(arg)
  dsp = sin(arg)
  ipph = (ip+1)/2
  ipp2 = ip+2
  idp2 = ido+2
  nbd = (ido-1)/2

      if (ido == 1) go to 119

      do 101 ik=1,idl1
         m2 = m2s
         do 1001 m1=1,m1d,im1
         m2 = m2+im2
         ch2(m2,ik,1) = c2(m1,ik,1)
 1001    continue
  101 continue

      do 103 j=2,ip
         do 102 k = 1, l1
            m2 = m2s
            do 1002 m1=1,m1d,im1
            m2 = m2+im2
            ch(m2,1,k,j) = c1(m1,1,k,j)
 1002       continue
  102    continue
  103 continue
      if ( l1 < nbd ) go to 107
      is = -ido
      do 106 j=2,ip
         is = is+ido
         idij = is
         do 105 i=3,ido,2
            idij = idij+2
            do 104 k = 1, l1
               m2 = m2s
               do 1004 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = wa(idij-1)*c1(m1,i-1,k,j)+wa(idij)*c1(m1,i,k,j)
               ch(m2,i,k,j) = wa(idij-1)*c1(m1,i,k,j)-wa(idij)*c1(m1,i-1,k,j)
 1004          continue
  104       continue
  105    continue
  106 continue
      go to 111
  107 is = -ido
      do 110 j=2,ip
         is = is+ido
         do 109 k = 1, l1
            idij = is
            do 108 i=3,ido,2
               idij = idij+2
               m2 = m2s
               do 1008 m1=1,m1d,im1
               m2 = m2+im2
               ch(m2,i-1,k,j) = wa(idij-1)*c1(m1,i-1,k,j)+wa(idij)*c1(m1,i,k,j)
               ch(m2,i,k,j) = wa(idij-1)*c1(m1,i,k,j)-wa(idij)*c1(m1,i-1,k,j)
 1008          continue
  108       continue
  109    continue
  110 continue
  111 if (nbd < l1) go to 115
      do 114 j=2,ipph
         jc = ipp2-j
         do 113 k = 1, l1
            do 112 i=3,ido,2
               m2 = m2s
               do 1012 m1=1,m1d,im1
               m2 = m2+im2
               c1(m1,i-1,k,j) = ch(m2,i-1,k,j)+ch(m2,i-1,k,jc)
               c1(m1,i-1,k,jc) = ch(m2,i,k,j)-ch(m2,i,k,jc)
               c1(m1,i,k,j) = ch(m2,i,k,j)+ch(m2,i,k,jc)
               c1(m1,i,k,jc) = ch(m2,i-1,k,jc)-ch(m2,i-1,k,j)
 1012          continue
  112       continue
  113    continue
  114 continue
      go to 121
  115 do 118 j=2,ipph
         jc = ipp2-j
         do 117 i=3,ido,2
            do 116 k = 1, l1
               m2 = m2s
               do 1016 m1=1,m1d,im1
               m2 = m2+im2
               c1(m1,i-1,k,j) = ch(m2,i-1,k,j)+ch(m2,i-1,k,jc)
               c1(m1,i-1,k,jc) = ch(m2,i,k,j)-ch(m2,i,k,jc)
               c1(m1,i,k,j) = ch(m2,i,k,j)+ch(m2,i,k,jc)
               c1(m1,i,k,jc) = ch(m2,i-1,k,jc)-ch(m2,i-1,k,j)
 1016          continue
  116       continue
  117    continue
  118 continue
      go to 121
  119 do 120 ik=1,idl1
         m2 = m2s
         do 1020 m1=1,m1d,im1
         m2 = m2+im2
         c2(m1,ik,1) = ch2(m2,ik,1)
 1020    continue
  120 continue
  121 do 123 j=2,ipph
         jc = ipp2-j
         do 122 k = 1, l1
            m2 = m2s
            do 1022 m1=1,m1d,im1
            m2 = m2+im2
            c1(m1,1,k,j) = ch(m2,1,k,j)+ch(m2,1,k,jc)
            c1(m1,1,k,jc) = ch(m2,1,k,jc)-ch(m2,1,k,j)
 1022       continue
  122    continue
  123 continue

      ar1 = 1.0D+00
      ai1 = 0.0D+00
      do 127 l=2,ipph
         lc = ipp2-l
         ar1h = dcp*ar1-dsp*ai1
         ai1 = dcp*ai1+dsp*ar1
         ar1 = ar1h
         do 124 ik=1,idl1
            m2 = m2s
            do 1024 m1=1,m1d,im1
            m2 = m2+im2
            ch2(m2,ik,l) = c2(m1,ik,1)+ar1*c2(m1,ik,2)
            ch2(m2,ik,lc) = ai1*c2(m1,ik,ip)
 1024       continue
  124    continue
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do 126 j=3,ipph
            jc = ipp2-j
            ar2h = dc2*ar2-ds2*ai2
            ai2 = dc2*ai2+ds2*ar2
            ar2 = ar2h
            do 125 ik=1,idl1
               m2 = m2s
               do 1025 m1=1,m1d,im1
               m2 = m2+im2
               ch2(m2,ik,l) = ch2(m2,ik,l)+ar2*c2(m1,ik,j)
               ch2(m2,ik,lc) = ch2(m2,ik,lc)+ai2*c2(m1,ik,jc)
 1025          continue
  125       continue
  126    continue
  127 continue
      do 129 j=2,ipph
         do 128 ik=1,idl1
            m2 = m2s
            do 1028 m1=1,m1d,im1
            m2 = m2+im2
            ch2(m2,ik,1) = ch2(m2,ik,1)+c2(m1,ik,j)
 1028       continue
  128    continue
  129 continue

      if (ido < l1) go to 132
      do 131 k = 1, l1
         do 130 i=1,ido
            m2 = m2s
            do 1030 m1=1,m1d,im1
            m2 = m2+im2
            cc(m1,i,1,k) = ch(m2,i,k,1)
 1030       continue
  130    continue
  131 continue
      go to 135
  132 do 134 i=1,ido
         do 133 k = 1, l1
            m2 = m2s
            do 1033 m1=1,m1d,im1
            m2 = m2+im2
            cc(m1,i,1,k) = ch(m2,i,k,1)
 1033       continue
  133    continue
  134 continue
  135 do 137 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 136 k = 1, l1
            m2 = m2s
            do 1036 m1=1,m1d,im1
            m2 = m2+im2
            cc(m1,ido,j2-2,k) = ch(m2,1,k,j)
            cc(m1,1,j2-1,k) = ch(m2,1,k,jc)
 1036       continue
  136    continue
  137 continue
      if (ido == 1) return
      if (nbd < l1) go to 141
      do 140 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 139 k = 1, l1
            do 138 i=3,ido,2
               ic = idp2-i
               m2 = m2s
               do 1038 m1=1,m1d,im1
               m2 = m2+im2
               cc(m1,i-1,j2-1,k) = ch(m2,i-1,k,j)+ch(m2,i-1,k,jc)
               cc(m1,ic-1,j2-2,k) = ch(m2,i-1,k,j)-ch(m2,i-1,k,jc)
               cc(m1,i,j2-1,k) = ch(m2,i,k,j)+ch(m2,i,k,jc)
               cc(m1,ic,j2-2,k) = ch(m2,i,k,jc)-ch(m2,i,k,j)
 1038          continue
  138       continue
  139    continue
  140 continue
      return
  141 do 144 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 143 i=3,ido,2
            ic = idp2-i
            do 142 k = 1, l1
               m2 = m2s
               do 1042 m1=1,m1d,im1
               m2 = m2+im2
               cc(m1,i-1,j2-1,k) = ch(m2,i-1,k,j)+ch(m2,i-1,k,jc)
               cc(m1,ic-1,j2-2,k) = ch(m2,i-1,k,j)-ch(m2,i-1,k,jc)
               cc(m1,i,j2-1,k) = ch(m2,i,k,j)+ch(m2,i,k,jc)
               cc(m1,ic,j2-2,k) = ch(m2,i,k,jc)-ch(m2,i,k,j)
 1042          continue
  142       continue
  143    continue
  144 continue

  return
end
subroutine mrftb1 (m,im,n,in,c,ch,wa,fac)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRFTB1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              in
  integer              m
  integer              n

  real ( rk       ) c(in,*)
  real ( rk       ) ch(m,*)
  real ( rk       ) fac(15)
  real ( rk       ) half
  real ( rk       ) halfm
  integer              i
  integer              idl1
  integer              ido
  integer              im
  integer              ip
  integer              iw
  integer              ix2
  integer              ix3
  integer              ix4
  integer              j
  integer              k1
  integer              l1
  integer              l2
  integer              m2
  integer              modn
  integer              na
  integer              nf
  integer              nl
  real ( rk       ) wa(n)

  nf = INT(fac(2))
  na = 0

  do k1=1,nf
      ip = INT(fac(k1+2))
      na = 1-na      
      if(ip <= 5) go to 10
      if(k1 == nf) go to 10
      na = 1-na
   10 continue 
  end do

      half = 0.5D+00
      halfm = -0.5D+00
      modn = mod(n,2)
      nl = n-2
      if(modn /= 0) nl = n-1
      if (na == 0) go to 120
      m2 = 1-im
      do 117 i=1,m
      m2 = m2+im
      ch(i,1) = c(m2,1)
      ch(i,n) = c(m2,n)
  117 continue
      do 118 j=2,nl,2
      m2 = 1-im
      do 118 i=1,m
         m2 = m2+im
	 ch(i,j) = half*c(m2,j)
	 ch(i,j+1) = halfm*c(m2,j+1)
  118 continue
      go to 124
  120 continue
      do 122 j=2,nl,2
      m2 = 1-im
      do 122 i=1,m
         m2 = m2+im
	 c(m2,j) = half*c(m2,j)
	 c(m2,j+1) = halfm*c(m2,j+1)
  122 continue
  124 l1 = 1
      iw = 1
      do 116 k1=1,nf
         ip = INT(fac(k1+2))
         l2 = ip*l1
         ido = n/l2
         idl1 = ido*l1
         if (ip /= 4) go to 103
         ix2 = iw+ido
         ix3 = ix2+ido
         if (na /= 0) go to 101
         call mradb4 (m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2),wa(ix3))
         go to 102
  101    call mradb4 (m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2),wa(ix3))
  102    na = 1-na
         go to 115
  103    if (ip /= 2) go to 106
         if (na /= 0) go to 104
	 call mradb2 (m,ido,l1,c,im,in,ch,1,m,wa(iw))
         go to 105
  104    call mradb2 (m,ido,l1,ch,1,m,c,im,in,wa(iw))
  105    na = 1-na
         go to 115
  106    if (ip /= 3) go to 109
         ix2 = iw+ido
         if (na /= 0) go to 107
	 call mradb3 (m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2))
         go to 108
  107    call mradb3 (m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2))
  108    na = 1-na
         go to 115
  109    if (ip /= 5) go to 112
         ix2 = iw+ido
         ix3 = ix2+ido
         ix4 = ix3+ido
         if (na /= 0) go to 110
         call mradb5 (m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 111
  110    call mradb5 (m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2),wa(ix3),wa(ix4))
  111    na = 1-na
         go to 115
  112    if (na /= 0) go to 113
	 call mradbg (m,ido,ip,l1,idl1,c,c,c,im,in,ch,ch,1,m,wa(iw))
         go to 114
  113    call mradbg (m,ido,ip,l1,idl1,ch,ch,ch,1,m,c,c,im,in,wa(iw))
  114    if (ido == 1) na = 1-na
  115    l1 = l2
         iw = iw+(ip-1)*ido
  116 continue

  return
end
subroutine mrftf1 (m,im,n,in,c,ch,wa,fac)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRFTF1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              in
  integer              m
  integer              n

  real ( rk       ) c(in,*)
  real ( rk       ) ch(m,*)
  real ( rk       ) fac(15)
  integer              i
  integer              idl1
  integer              ido
  integer              im
  integer              ip
  integer              iw
  integer              ix2
  integer              ix3
  integer              ix4
  integer              j
  integer              k1
  integer              kh
  integer              l1
  integer              l2
  integer              m2
  integer              modn
  integer              na
  integer              nf
  integer              nl
  real ( rk       ) sn
  real ( rk       ) tsn
  real ( rk       ) tsnm
  real ( rk       ) wa(n)

  nf = INT(fac(2))
  na = 1
  l2 = n
  iw = n

      do 111 k1=1,nf
         kh = nf-k1
         ip = INT(fac(kh+3))
         l1 = l2/ip
         ido = n/l2
         idl1 = ido*l1
         iw = iw-(ip-1)*ido
         na = 1-na
         if (ip /= 4) go to 102
         ix2 = iw+ido
         ix3 = ix2+ido
         if (na /= 0) go to 101
	     call mradf4 (m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2),wa(ix3))
         go to 110
  101    call mradf4 (m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2),wa(ix3))
         go to 110
  102    if (ip /= 2) go to 104
         if (na /= 0) go to 103
	     call mradf2 (m,ido,l1,c,im,in,ch,1,m,wa(iw))
         go to 110
  103    call mradf2 (m,ido,l1,ch,1,m,c,im,in,wa(iw))
         go to 110
  104    if (ip /= 3) go to 106
         ix2 = iw+ido
         if (na /= 0) go to 105
	     call mradf3 (m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2))
         go to 110
  105    call mradf3 (m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2))
         go to 110
  106    if (ip /= 5) go to 108
         ix2 = iw+ido
         ix3 = ix2+ido
         ix4 = ix3+ido
         if (na /= 0) go to 107
         call mradf5(m,ido,l1,c,im,in,ch,1,m,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
  107    call mradf5(m,ido,l1,ch,1,m,c,im,in,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
  108    if (ido == 1) na = 1-na
         if (na /= 0) go to 109
         call mradfg (m,ido,ip,l1,idl1,c,c,c,im,in,ch,ch,1,m,wa(iw))
         na = 1
         go to 110
  109    call mradfg (m,ido,ip,l1,idl1,ch,ch,ch,1,m,c,c,im,in,wa(iw))
         na = 0
  110    l2 = l1
  111 continue

      sn = 1.0D+00 / real ( n, rk       )
      tsn =  2.0D+00 / real ( n, rk       )
      tsnm = -tsn
      modn = mod(n,2)
      nl = n-2
      if(modn /= 0) nl = n-1
      if (na /= 0) go to 120
      m2 = 1-im

      do i=1,m
        m2 = m2+im
        c(m2,1) = sn*ch(i,1)
      end do

      do j=2,nl,2
        m2 = 1-im
        do i=1,m
          m2 = m2+im
	      c(m2,j) = tsn*ch(i,j)
	      c(m2,j+1) = tsnm*ch(i,j+1)
        end do
      end do

      if(modn /= 0) return
      m2 = 1-im
      do 119 i=1,m
         m2 = m2+im
         c(m2,n) = sn*ch(i,n)
  119 continue
      return
  120 m2 = 1-im
      do 121 i=1,m
         m2 = m2+im
         c(m2,1) = sn*c(m2,1)
  121 continue
      do 122 j=2,nl,2
      m2 = 1-im
      do 122 i=1,m
         m2 = m2+im
	 c(m2,j) = tsn*c(m2,j)
	 c(m2,j+1) = tsnm*c(m2,j+1)
  122 continue
      if(modn /= 0) return
      m2 = 1-im
      do i=1,m
         m2 = m2+im
         c(m2,n) = sn*c(m2,n)
      end do

  return
end
subroutine mrfti1 (n,wa,fac)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MRFTI1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the number for which factorization and 
!    other information is needed.
!
!    Output, real ( rk       ) WA(N), trigonometric information.
!
!    Output, real ( rk       ) FAC(15), factorization information.  FAC(1) is 
!    N, FAC(2) is NF, the number of factors, and FAC(3:NF+2) are the factors.
!
  implicit none

  integer              n

  real ( rk       ) arg
  real ( rk       ) argh
  real ( rk       ) argld
  real ( rk       ) fac(15)
  real ( rk       ) fi
  integer              i
  integer              ib
  integer              ido
  integer              ii
  integer              ip
  integer              ipm
  integer              is
  integer              j
  integer              k1
  integer              l1
  integer              l2
  integer              ld
  integer              nf
  integer              nfm1
  integer              nl
  integer              nq
  integer              nr
  integer              ntry
  integer              ntryh(4)
  real ( rk       ) tpi
  real ( rk       ) wa(n)

  save ntryh

  data ntryh / 4, 2, 3, 5 /

  nl = n
  nf = 0
  j = 0

  101 j = j+1
      if (j-4) 102,102,103
  102 ntry = ntryh(j)
      go to 104
  103 ntry = ntry+2
  104 nq = nl/ntry
      nr = nl-ntry*nq
      if (nr) 101,105,101
  105 nf = nf+1
      fac(nf+2) = ntry
      nl = nq
      if (ntry /= 2) go to 107
      do i=2,nf
         ib = nf-i+2
         fac(ib+2) = fac(ib+1)
      end do
      fac(3) = 2
  107 if (nl /= 1) go to 104
      fac(1) = n
      fac(2) = nf
      tpi = 8.0D+00 * atan ( 1.0D+00 )
      argh = tpi / real ( n, rk       )
      is = 0
      nfm1 = nf-1
      l1 = 1

      do k1=1,nfm1
         ip = INT(fac(k1+2))
         ld = 0
         l2 = l1*ip
         ido = n/l2
         ipm = ip-1
         do j=1,ipm
            ld = ld+l1
            i = is
            argld = real ( ld, rk       ) * argh
            fi = 0.0D+00
            do ii=3,ido,2
              i = i+2
              fi = fi + 1.0D+00
              arg = fi*argld
	          wa(i-1) = cos ( arg )
	          wa(i) = sin ( arg )
            end do
            is = is+ido
         end do
         l1 = l2
      end do

  return
end
subroutine msntb1(lot,jump,n,inc,x,wsave,dsum,xh,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MSNTB1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc
  integer              lot

  real ( rk       ) dsum(*)
  real ( rk       ) fnp1s4
  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lj
  integer              lnsv
  integer              lnwk
  integer              lnxh
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) srt3s2
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xh(lot,*)
  real ( rk       ) xhold

  ier = 0
  lj = (lot-1)*jump+1

      if (n-2) 200,102,103
 102  srt3s2 = sqrt( 3.0D+00 )/ 2.0D+00 

      do m=1,lj,jump
         xhold = srt3s2*(x(m,1)+x(m,2))
         x(m,2) = srt3s2*(x(m,1)-x(m,2))
         x(m,1) = xhold
      end do

      go to 200
  103 np1 = n+1
      ns2 = n/2
      do 104 k=1,ns2
         kc = np1-k
         m1 = 0
         do 114 m=1,lj,jump
         m1 = m1+1
         t1 = x(m,k)-x(m,kc)
         t2 = wsave(k)*(x(m,k)+x(m,kc))
         xh(m1,k+1) = t1+t2
         xh(m1,kc+1) = t2-t1
  114    continue
  104 continue
      modn = mod(n,2)
      if (modn == 0) go to 124
      m1 = 0
      do 123 m=1,lj,jump
         m1 = m1+1
         xh(m1,ns2+2) =  4.0D+00 * x(m,ns2+1)
  123 continue
  124 do m=1,lot
         xh(m,1) = 0.0D+00
      end do

      lnxh = lot-1 + lot*(np1-1) + 1
      lnsv = np1 + int(log( real ( np1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = lot*np1

      call rfftmf(lot,1,np1,lot,xh,lnxh,wsave(ns2+1),lnsv,work,lnwk,ier1)     

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('msntb1',-5)
        go to 200
      end if

      if(mod(np1,2) /= 0) go to 30
      do m=1,lot
        xh(m,np1) = xh(m,np1)+xh(m,np1)
      end do
 30   fnp1s4 = real ( np1 , rk) / 4.0D+00
      m1 = 0
      do 125 m=1,lj,jump
         m1 = m1+1
         x(m,1) = fnp1s4*xh(m1,1)
         dsum(m1) = x(m,1)
  125 continue
      do 105 i=3,n,2
         m1 = 0
         do 115 m=1,lj,jump
            m1 = m1+1
            x(m,i-1) = fnp1s4*xh(m1,i)
            dsum(m1) = dsum(m1)+fnp1s4*xh(m1,i-1)
            x(m,i) = dsum(m1)
  115    continue
  105 continue
      if (modn /= 0) go to 200
      m1 = 0
      do 116 m=1,lj,jump
         m1 = m1+1
         x(m,n) = fnp1s4*xh(m1,n+1)
  116 continue

  200 continue

  return
end
subroutine msntf1(lot,jump,n,inc,x,wsave,dsum,xh,work,ier)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! MSNTF1 is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc
  integer              lot

  real ( rk       ) dsum(*)
  integer              i
  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lj
  integer              lnsv
  integer              lnwk
  integer              lnxh
  integer              m
  integer              m1
  integer              modn
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) sfnp1
  real ( rk       ) ssqrt3
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xh(lot,*)
  real ( rk       ) xhold

  ier = 0
  lj = (lot-1)*jump+1

      if (n-2) 101,102,103
 102  ssqrt3 = 1.0D+00 / sqrt ( 3.0D+00 )

      do m=1,lj,jump
         xhold = ssqrt3*(x(m,1)+x(m,2))
         x(m,2) = ssqrt3*(x(m,1)-x(m,2))
         x(m,1) = xhold
      end do

  101  go to 200
  103 np1 = n+1
      ns2 = n/2
      do 104 k=1,ns2
         kc = np1-k
         m1 = 0
         do 114 m=1,lj,jump
         m1 = m1 + 1
         t1 = x(m,k)-x(m,kc)
         t2 = wsave(k)*(x(m,k)+x(m,kc))
         xh(m1,k+1) = t1+t2
         xh(m1,kc+1) = t2-t1
  114    continue
  104 continue
      modn = mod(n,2)
      if (modn == 0) go to 124
      m1 = 0
      do 123 m=1,lj,jump
         m1 = m1 + 1
         xh(m1,ns2+2) =  4.0D+00  * x(m,ns2+1)
  123 continue
  124 do 127 m=1,lot
         xh(m,1) = 0.0D+00
  127 continue 
      lnxh = lot-1 + lot*(np1-1) + 1
      lnsv = np1 + int(log( real ( np1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = lot*np1

      call rfftmf(lot,1,np1,lot,xh,lnxh,wsave(ns2+1),lnsv,work,lnwk,ier1)     
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('msntf1',-5)
        go to 200
      end if

      if(mod(np1,2) /= 0) go to 30
      do 20 m=1,lot
      xh(m,np1) = xh(m,np1)+xh(m,np1)
   20 continue
   30 sfnp1 = 1.0D+00 / real ( np1, rk       )
      m1 = 0
      do 125 m=1,lj,jump
         m1 = m1+1
         x(m,1) = 0.5D+00 * xh(m1,1)
         dsum(m1) = x(m,1)
  125 continue
      do 105 i=3,n,2
         m1 = 0
         do 115 m=1,lj,jump
            m1 = m1+1
            x(m,i-1) = 0.5D+00 * xh(m1,i)
            dsum(m1) = dsum(m1)+ 0.5D+00 * xh(m1,i-1)
            x(m,i) = dsum(m1)
  115    continue
  105 continue
      if (modn /= 0) go to 200

      m1 = 0
      do m=1,lj,jump
         m1 = m1+1
         x(m,n) = 0.5D+00 * xh(m1,n+1)
      end do

  200 continue

  return
end
subroutine r1f2kb (ido,l1,cc,in1,ch,in2,wa1)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F2KB is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,2,l1)
  real ( rk       ) ch(in2,ido,l1,2)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) wa1(ido)

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k)+cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k)-cc(1,ido,2,k)
  end do

      if (ido-2) 107,105,102
 102  idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
            
            ch(1,i-1,k,1) = cc(1,i-1,1,k)+cc(1,ic-1,2,k)
            ch(1,i,k,1) = cc(1,i,1,k)-cc(1,ic,2,k)
            
            ch(1,i-1,k,2) = wa1(i-2)*(cc(1,i-1,1,k)-cc(1,ic-1,2,k)) &
                 -wa1(i-1)*(cc(1,i,1,k)+cc(1,ic,2,k))
            ch(1,i,k,2) = wa1(i-2)*(cc(1,i,1,k)+cc(1,ic,2,k))+wa1(i-1) &
                 *(cc(1,i-1,1,k)-cc(1,ic-1,2,k))

 103     continue
 104  continue
      if (mod(ido,2) == 1) return
 105  do 106 k = 1, l1
         ch(1,ido,k,1) = cc(1,ido,1,k)+cc(1,ido,1,k)
         ch(1,ido,k,2) = -(cc(1,1,2,k)+cc(1,1,2,k))
 106  continue
 107  continue

  return
end
subroutine r1f2kf (ido,l1,cc,in1,ch,in2,wa1)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F1KF is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) ch(in2,ido,2,l1)
  real ( rk       ) cc(in1,ido,l1,2)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) wa1(ido)

  do k = 1, l1
    ch(1,1,1,k) = cc(1,1,k,1)+cc(1,1,k,2)
    ch(1,ido,2,k) = cc(1,1,k,1)-cc(1,1,k,2)
  end do

      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do i=3,ido,2
            ic = idp2-i
            ch(1,i,1,k) = cc(1,i,k,1)+(wa1(i-2)*cc(1,i,k,2) &
              -wa1(i-1)*cc(1,i-1,k,2))
            ch(1,ic,2,k) = (wa1(i-2)*cc(1,i,k,2) &
              -wa1(i-1)*cc(1,i-1,k,2))-cc(1,i,k,1)
            ch(1,i-1,1,k) = cc(1,i-1,k,1)+(wa1(i-2)*cc(1,i-1,k,2) &
              +wa1(i-1)*cc(1,i,k,2))
            ch(1,ic-1,2,k) = cc(1,i-1,k,1)-(wa1(i-2)*cc(1,i-1,k,2) &
              +wa1(i-1)*cc(1,i,k,2))
          end do
  104 continue
      if (mod(ido,2) == 1) return
  105 do 106 k = 1, l1
         ch(1,1,2,k) = -cc(1,ido,k,2)
         ch(1,ido,1,k) = cc(1,ido,k,1)
  106 continue
  107 continue

  return
end
subroutine r1f3kb (ido,l1,cc,in1,ch,in2,wa1,wa2)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F3KB is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,3,l1)
  real ( rk       ) ch(in2,ido,l1,3)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) taui
  real ( rk       ) taur
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)

  arg = 2.0D+00 * 4.0D+00 * atan ( 1.0D+00 ) / 3.0D+00 
  taur = cos ( arg )
  taui = sin ( arg )

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k) + 2.0D+00 * cc(1,ido,2,k)
    ch(1,1,k,2) = cc(1,1,1,k) + ( 2.0D+00 * taur ) * cc(1,ido,2,k) &
      - ( 2.0D+00 *taui)*cc(1,1,3,k)
    ch(1,1,k,3) = cc(1,1,1,k) + ( 2.0D+00 *taur)*cc(1,ido,2,k) &
      + 2.0D+00 *taui*cc(1,1,3,k)
  end do

  if (ido == 1) then
    return
  end if

  idp2 = ido+2

      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
        ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
        ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k))

        ch(1,i-1,k,2) = wa1(i-2)* &
       ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
       (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) &
                         -wa1(i-1)* &
       ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
       (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) 

            ch(1,i,k,2) = wa1(i-2)* &
       ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))+ &
       (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) &
                        +wa1(i-1)* &
       ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))- &
       (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))

              ch(1,i-1,k,3) = wa2(i-2)* &
       ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
       (taui*(cc(1,i,3,k)+cc(1,ic,2,k)))) &
         -wa2(i-1)* &
       ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
       (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))))

            ch(1,i,k,3) = wa2(i-2)* &
       ((cc(1,i,1,k)+taur*(cc(1,i,3,k)-cc(1,ic,2,k)))- &
       (taui*(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))) &
                       +wa2(i-1)* &
       ((cc(1,i-1,1,k)+taur*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))+ &
       (taui*(cc(1,i,3,k)+cc(1,ic,2,k))))

  102    continue
  103 continue

  return
end
subroutine r1f3kf (ido,l1,cc,in1,ch,in2,wa1,wa2)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F3KF is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,l1,3)
  real ( rk       ) ch(in2,ido,3,l1)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) taui
  real ( rk       ) taur
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)

  arg= 2.0D+00 * 4.0D+00 * atan( 1.0D+00 )/ 3.0D+00 
  taur=cos(arg)
  taui=sin(arg)

  do k = 1, l1
    ch(1,1,1,k) = cc(1,1,k,1)+(cc(1,1,k,2)+cc(1,1,k,3))
    ch(1,1,3,k) = taui*(cc(1,1,k,3)-cc(1,1,k,2))
    ch(1,ido,2,k) = cc(1,1,k,1)+taur*(cc(1,1,k,2)+cc(1,1,k,3))
  end do

  if (ido == 1) then
    return
  end if

      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i

            ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
             wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3)))

            ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3)))

            ch(1,i-1,3,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
             cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
             cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))+(taui*((wa1(i-2)* &
             cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
             cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))

            ch(1,ic-1,2,k) = (cc(1,i-1,k,1)+taur*((wa1(i-2)* &
             cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa2(i-2)* &
             cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))))-(taui*((wa1(i-2)* &
             cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa2(i-2)* &
             cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))))

            ch(1,i,3,k) = (cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))))+(taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2))))

            ch(1,ic,2,k) = (taui*((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2))))-(cc(1,i,k,1)+taur*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))))
  102    continue
  103 continue

  return
end
subroutine r1f4kb (ido,l1,cc,in1,ch,in2,wa1,wa2,wa3)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F4KB is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,4,l1)
  real ( rk       ) ch(in2,ido,l1,4)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) sqrt2
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)

  sqrt2=sqrt( 2.0D+00 )

  do k = 1, l1
    ch(1,1,k,3) = (cc(1,1,1,k)+cc(1,ido,4,k)) &
      -(cc(1,ido,2,k)+cc(1,ido,2,k))
    ch(1,1,k,1) = (cc(1,1,1,k)+cc(1,ido,4,k)) &
      +(cc(1,ido,2,k)+cc(1,ido,2,k))
    ch(1,1,k,4) = (cc(1,1,1,k)-cc(1,ido,4,k)) &
      +(cc(1,1,3,k)+cc(1,1,3,k))
    ch(1,1,k,2) = (cc(1,1,1,k)-cc(1,ido,4,k)) &
      -(cc(1,1,3,k)+cc(1,1,3,k))
  end do

      if (ido-2) 107,105,102

  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
        ch(1,i-1,k,1) = (cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
        +(cc(1,i-1,3,k)+cc(1,ic-1,2,k))
        ch(1,i,k,1) = (cc(1,i,1,k)-cc(1,ic,4,k)) &
        +(cc(1,i,3,k)-cc(1,ic,2,k))
        ch(1,i-1,k,2)=wa1(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
        -(cc(1,i,3,k)+cc(1,ic,2,k)))-wa1(i-1) &
        *((cc(1,i,1,k)+cc(1,ic,4,k))+(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,2)=wa1(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
        +(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa1(i-1) &
        *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))-(cc(1,i,3,k)+cc(1,ic,2,k)))
        ch(1,i-1,k,3)=wa2(i-2)*((cc(1,i-1,1,k)+cc(1,ic-1,4,k)) &
        -(cc(1,i-1,3,k)+cc(1,ic-1,2,k)))-wa2(i-1) &
        *((cc(1,i,1,k)-cc(1,ic,4,k))-(cc(1,i,3,k)-cc(1,ic,2,k)))
        ch(1,i,k,3)=wa2(i-2)*((cc(1,i,1,k)-cc(1,ic,4,k)) &
        -(cc(1,i,3,k)-cc(1,ic,2,k)))+wa2(i-1) &
        *((cc(1,i-1,1,k)+cc(1,ic-1,4,k))-(cc(1,i-1,3,k) &
        +cc(1,ic-1,2,k)))
        ch(1,i-1,k,4)=wa3(i-2)*((cc(1,i-1,1,k)-cc(1,ic-1,4,k)) &
        +(cc(1,i,3,k)+cc(1,ic,2,k)))-wa3(i-1) &
       *((cc(1,i,1,k)+cc(1,ic,4,k))-(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))
        ch(1,i,k,4)=wa3(i-2)*((cc(1,i,1,k)+cc(1,ic,4,k)) &
        -(cc(1,i-1,3,k)-cc(1,ic-1,2,k)))+wa3(i-1) &
        *((cc(1,i-1,1,k)-cc(1,ic-1,4,k))+(cc(1,i,3,k)+cc(1,ic,2,k)))
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 continue
      do 106 k = 1, l1
         ch(1,ido,k,1) = (cc(1,ido,1,k)+cc(1,ido,3,k)) &
         +(cc(1,ido,1,k)+cc(1,ido,3,k))
         ch(1,ido,k,2) = sqrt2*((cc(1,ido,1,k)-cc(1,ido,3,k)) &
         -(cc(1,1,2,k)+cc(1,1,4,k)))
         ch(1,ido,k,3) = (cc(1,1,4,k)-cc(1,1,2,k)) &
         +(cc(1,1,4,k)-cc(1,1,2,k))
         ch(1,ido,k,4) = -sqrt2*((cc(1,ido,1,k)-cc(1,ido,3,k)) &
         +(cc(1,1,2,k)+cc(1,1,4,k)))
  106 continue
  107 continue

  return
end
subroutine r1f4kf (ido,l1,cc,in1,ch,in2,wa1,wa2,wa3)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F4KF is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) cc(in1,ido,l1,4)
  real ( rk       ) ch(in2,ido,4,l1)
  real ( rk       ) hsqt2
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)

  hsqt2=sqrt( 2.0D+00 )/ 2.0D+00 

  do k = 1, l1
    ch(1,1,1,k) = (cc(1,1,k,2)+cc(1,1,k,4))+(cc(1,1,k,1)+cc(1,1,k,3))
    ch(1,ido,4,k) = (cc(1,1,k,1)+cc(1,1,k,3))-(cc(1,1,k,2)+cc(1,1,k,4))
    ch(1,ido,2,k) = cc(1,1,k,1)-cc(1,1,k,3)
    ch(1,1,3,k) = cc(1,1,k,4)-cc(1,1,k,2)
  end do

      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k = 1, l1
         do 103 i=3,ido,2
            ic = idp2-i
            ch(1,i-1,1,k) = ((wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4)))+(cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
             wa2(i-1)*cc(1,i,k,3)))
            ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+(wa2(i-2)*cc(1,i-1,k,3)+ &
             wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i-1,k,2)+ &
             wa1(i-1)*cc(1,i,k,2))+(wa3(i-2)*cc(1,i-1,k,4)+ &
             wa3(i-1)*cc(1,i,k,4)))
            ch(1,i,1,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
             cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4)))+(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
             wa2(i-1)*cc(1,i-1,k,3)))
            ch(1,ic,4,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
             cc(1,i-1,k,2))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4)))-(cc(1,i,k,1)+(wa2(i-2)*cc(1,i,k,3)- &
             wa2(i-1)*cc(1,i-1,k,3)))
            ch(1,i-1,3,k) = ((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
             cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4)))+(cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
             wa2(i-1)*cc(1,i,k,3)))
            ch(1,ic-1,2,k) = (cc(1,i-1,k,1)-(wa2(i-2)*cc(1,i-1,k,3)+ &
             wa2(i-1)*cc(1,i,k,3)))-((wa1(i-2)*cc(1,i,k,2)-wa1(i-1)* &
             cc(1,i-1,k,2))-(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4)))
            ch(1,i,3,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))+(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
             wa2(i-1)*cc(1,i-1,k,3)))
            ch(1,ic,2,k) = ((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))-(cc(1,i,k,1)-(wa2(i-2)*cc(1,i,k,3)- &
             wa2(i-1)*cc(1,i-1,k,3)))
  103    continue
  104 continue
      if (mod(ido,2) == 1) return
  105 continue
      do 106 k = 1, l1
            ch(1,ido,1,k) = (hsqt2*(cc(1,ido,k,2)-cc(1,ido,k,4)))+cc(1,ido,k,1)
            ch(1,ido,3,k) = cc(1,ido,k,1)-(hsqt2*(cc(1,ido,k,2)-cc(1,ido,k,4)))
            ch(1,1,2,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))-cc(1,ido,k,3)
            ch(1,1,4,k) = (-hsqt2*(cc(1,ido,k,2)+cc(1,ido,k,4)))+cc(1,ido,k,3)
  106 continue
  107 continue

  return
end
subroutine r1f5kb (ido,l1,cc,in1,ch,in2,wa1,wa2,wa3,wa4)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F5KB is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,5,l1)
  real ( rk       ) ch(in2,ido,l1,5)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) ti11
  real ( rk       ) ti12
  real ( rk       ) tr11
  real ( rk       ) tr12
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)
  real ( rk       ) wa4(ido)

  arg= 2.0D+00 * 4.0D+00 * atan( 1.0D+00 ) / 5.0D+00
  tr11=cos(arg)
  ti11=sin(arg)
  tr12=cos( 2.0D+00 *arg )
  ti12=sin( 2.0D+00 *arg )

  do k = 1, l1
    ch(1,1,k,1) = cc(1,1,1,k)+ 2.0D+00 *cc(1,ido,2,k)+ 2.0D+00 *cc(1,ido,4,k)
    ch(1,1,k,2) = (cc(1,1,1,k)+tr11* 2.0D+00 *cc(1,ido,2,k) &
      +tr12* 2.0D+00 *cc(1,ido,4,k))-(ti11* 2.0D+00 *cc(1,1,3,k) &
      +ti12* 2.0D+00 *cc(1,1,5,k))
    ch(1,1,k,3) = (cc(1,1,1,k)+tr12* 2.0D+00 *cc(1,ido,2,k) &
      +tr11* 2.0D+00 *cc(1,ido,4,k))-(ti12* 2.0D+00 *cc(1,1,3,k) &
      -ti11* 2.0D+00 *cc(1,1,5,k))
    ch(1,1,k,4) = (cc(1,1,1,k)+tr12* 2.0D+00 *cc(1,ido,2,k) &
      +tr11* 2.0D+00 *cc(1,ido,4,k))+(ti12* 2.0D+00 *cc(1,1,3,k) &
      -ti11* 2.0D+00 *cc(1,1,5,k))
    ch(1,1,k,5) = (cc(1,1,1,k)+tr11* 2.0D+00 *cc(1,ido,2,k) &
      +tr12* 2.0D+00 *cc(1,ido,4,k))+(ti11* 2.0D+00 *cc(1,1,3,k) &
      +ti12* 2.0D+00 *cc(1,1,5,k))
  end do

  if (ido == 1) return

      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i
        ch(1,i-1,k,1) = cc(1,i-1,1,k)+(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +(cc(1,i-1,5,k)+cc(1,ic-1,4,k))
        ch(1,i,k,1) = cc(1,i,1,k)+(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +(cc(1,i,5,k)-cc(1,ic,4,k))
        ch(1,i-1,k,2) = wa1(i-2)*((cc(1,i-1,1,k)+tr11* &
        (cc(1,i-1,3,k)+cc(1,ic-1,2,k))+tr12 &
        *(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa1(i-1)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))+(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))

        ch(1,i,k,2) = wa1(i-2)*((cc(1,i,1,k)+tr11*(cc(1,i,3,k) &
        -cc(1,ic,2,k))+tr12*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti11*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))+ti12 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))+wa1(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k) &
        +cc(1,ic-1,2,k))+tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k))) &
        -(ti11*(cc(1,i,3,k)+cc(1,ic,2,k))+ti12 &
        *(cc(1,i,5,k)+cc(1,ic,4,k))))

        ch(1,i-1,k,3) = wa2(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
       -wa2(i-1) &
       *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))

        ch(1,i,k,3) = wa2(i-2) &
       *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        +(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa2(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))-(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))

        ch(1,i-1,k,4) = wa3(i-2) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa3(i-1) &
       *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))

        ch(1,i,k,4) = wa3(i-2) &
       *((cc(1,i,1,k)+tr12*(cc(1,i,3,k)- &
        cc(1,ic,2,k))+tr11*(cc(1,i,5,k)-cc(1,ic,4,k))) &
        -(ti12*(cc(1,i-1,3,k)-cc(1,ic-1,2,k))-ti11 &
        *(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa3(i-1) &
        *((cc(1,i-1,1,k)+tr12*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr11*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti12*(cc(1,i,3,k) &
        +cc(1,ic,2,k))-ti11*(cc(1,i,5,k)+cc(1,ic,4,k))))

        ch(1,i-1,k,5) = wa4(i-2) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k)))) &
        -wa4(i-1) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k))))

        ch(1,i,k,5) = wa4(i-2) &
        *((cc(1,i,1,k)+tr11*(cc(1,i,3,k)-cc(1,ic,2,k)) &
        +tr12*(cc(1,i,5,k)-cc(1,ic,4,k)))-(ti11*(cc(1,i-1,3,k) &
        -cc(1,ic-1,2,k))+ti12*(cc(1,i-1,5,k)-cc(1,ic-1,4,k)))) &
        +wa4(i-1) &
        *((cc(1,i-1,1,k)+tr11*(cc(1,i-1,3,k)+cc(1,ic-1,2,k)) &
        +tr12*(cc(1,i-1,5,k)+cc(1,ic-1,4,k)))+(ti11*(cc(1,i,3,k) &
        +cc(1,ic,2,k))+ti12*(cc(1,i,5,k)+cc(1,ic,4,k))))

  102    continue
  103 continue

  return
end
subroutine r1f5kf (ido,l1,cc,in1,ch,in2,wa1,wa2,wa3,wa4)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1F5KF is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              in1
  integer              in2
  integer              l1

  real ( rk       ) arg
  real ( rk       ) cc(in1,ido,l1,5)
  real ( rk       ) ch(in2,ido,5,l1)
  integer              i
  integer              ic
  integer              idp2
  integer              k
  real ( rk       ) ti11
  real ( rk       ) ti12
  real ( rk       ) tr11
  real ( rk       ) tr12
  real ( rk       ) wa1(ido)
  real ( rk       ) wa2(ido)
  real ( rk       ) wa3(ido)
  real ( rk       ) wa4(ido)

  arg= 2.0D+00 * 4.0D+00 * atan( 1.0D+00 ) / 5.0D+00
  tr11=cos(arg)
  ti11=sin(arg)
  tr12=cos( 2.0D+00 *arg)
  ti12=sin( 2.0D+00 *arg)

  do k = 1, l1
    ch(1,1,1,k) = cc(1,1,k,1)+(cc(1,1,k,5)+cc(1,1,k,2))+ &
      (cc(1,1,k,4)+cc(1,1,k,3))
    ch(1,ido,2,k) = cc(1,1,k,1)+tr11*(cc(1,1,k,5)+cc(1,1,k,2))+ &
      tr12*(cc(1,1,k,4)+cc(1,1,k,3))
    ch(1,1,3,k) = ti11*(cc(1,1,k,5)-cc(1,1,k,2))+ti12* &
      (cc(1,1,k,4)-cc(1,1,k,3))
    ch(1,ido,4,k) = cc(1,1,k,1)+tr12*(cc(1,1,k,5)+cc(1,1,k,2))+ &
      tr11*(cc(1,1,k,4)+cc(1,1,k,3))
    ch(1,1,5,k) = ti12*(cc(1,1,k,5)-cc(1,1,k,2))-ti11* &
      (cc(1,1,k,4)-cc(1,1,k,3))
  end do

  if (ido == 1) return

      idp2 = ido+2
      do 103 k = 1, l1
         do 102 i=3,ido,2
            ic = idp2-i

            ch(1,i-1,1,k) = cc(1,i-1,k,1)+((wa1(i-2)*cc(1,i-1,k,2)+ &
            wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
            cc(1,i,k,5)))+((wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
            cc(1,i,k,3))+(wa3(i-2)*cc(1,i-1,k,4)+ &
            wa3(i-1)*cc(1,i,k,4))) 

            ch(1,i,1,k) = cc(1,i,k,1)+((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
             cc(1,i-1,k,5)))+((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4)))

            ch(1,i-1,3,k) = cc(1,i-1,k,1)+tr11* &
            ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
             +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
            ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
             +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))+ti11* &
            ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
             -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
            ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
             -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4)))

            ch(1,ic-1,2,k) = cc(1,i-1,k,1)+tr11* &
            ( wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2) &
             +wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5))+tr12* &
           ( wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3) &
            +wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))-(ti11* &
            ( wa1(i-2)*cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2) &
             -(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))+ti12* &
            ( wa2(i-2)*cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3) &
             -(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))

            ch(1,i,3,k) = (cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
             cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4))))+(ti11*((wa4(i-2)*cc(1,i-1,k,5)+ &
             wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))))

            ch(1,ic,2,k) = (ti11*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
             cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))+ti12*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))))-(cc(1,i,k,1)+tr11*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
             cc(1,i-1,k,5)))+tr12*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4))))

            ch(1,i-1,5,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
             cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
             cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
             cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
             cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))+(ti12*((wa1(i-2)* &
             cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
             cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
             cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
             cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))

            ch(1,ic-1,4,k) = (cc(1,i-1,k,1)+tr12*((wa1(i-2)* &
             cc(1,i-1,k,2)+wa1(i-1)*cc(1,i,k,2))+(wa4(i-2)* &
             cc(1,i-1,k,5)+wa4(i-1)*cc(1,i,k,5)))+tr11*((wa2(i-2)* &
             cc(1,i-1,k,3)+wa2(i-1)*cc(1,i,k,3))+(wa3(i-2)* &
             cc(1,i-1,k,4)+wa3(i-1)*cc(1,i,k,4))))-(ti12*((wa1(i-2)* &
             cc(1,i,k,2)-wa1(i-1)*cc(1,i-1,k,2))-(wa4(i-2)* &
             cc(1,i,k,5)-wa4(i-1)*cc(1,i-1,k,5)))-ti11*((wa2(i-2)* &
             cc(1,i,k,3)-wa2(i-1)*cc(1,i-1,k,3))-(wa3(i-2)* &
             cc(1,i,k,4)-wa3(i-1)*cc(1,i-1,k,4))))

            ch(1,i,5,k) = (cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
             cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4))))+(ti12*((wa4(i-2)*cc(1,i-1,k,5)+ &
             wa4(i-1)*cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))))

            ch(1,ic,4,k) = (ti12*((wa4(i-2)*cc(1,i-1,k,5)+wa4(i-1)* &
             cc(1,i,k,5))-(wa1(i-2)*cc(1,i-1,k,2)+wa1(i-1)* &
             cc(1,i,k,2)))-ti11*((wa3(i-2)*cc(1,i-1,k,4)+wa3(i-1)* &
             cc(1,i,k,4))-(wa2(i-2)*cc(1,i-1,k,3)+wa2(i-1)* &
             cc(1,i,k,3))))-(cc(1,i,k,1)+tr12*((wa1(i-2)*cc(1,i,k,2)- &
             wa1(i-1)*cc(1,i-1,k,2))+(wa4(i-2)*cc(1,i,k,5)-wa4(i-1)* &
             cc(1,i-1,k,5)))+tr11*((wa2(i-2)*cc(1,i,k,3)-wa2(i-1)* &
             cc(1,i-1,k,3))+(wa3(i-2)*cc(1,i,k,4)-wa3(i-1)* &
             cc(1,i-1,k,4))))

  102    continue
  103 continue

  return
end
subroutine r1fgkb (ido,ip,l1,idl1,cc,c1,c2,in1,ch,ch2,in2,wa)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1FGKB is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              idl1
  integer              ido
  integer              in1
  integer              in2
  integer              ip
  integer              l1

  real ( rk       ) ai1
  real ( rk       ) ai2
  real ( rk       ) ar1
  real ( rk       ) ar1h
  real ( rk       ) ar2
  real ( rk       ) ar2h
  real ( rk       ) arg
  real ( rk       ) c1(in1,ido,l1,ip)
  real ( rk       ) c2(in1,idl1,ip)
  real ( rk       ) cc(in1,ido,ip,l1)
  real ( rk       ) ch(in2,ido,l1,ip)
  real ( rk       ) ch2(in2,idl1,ip)
  real ( rk       ) dc2
  real ( rk       ) dcp
  real ( rk       ) ds2
  real ( rk       ) dsp
  integer              i
  integer              ic
  integer              idij
  integer              idp2
  integer              ik
  integer              ipp2
  integer              ipph
  integer              is
  integer              j
  integer              j2
  integer              jc
  integer              k
  integer              l
  integer              lc
  integer              nbd
  real ( rk       ) tpi
  real ( rk       ) wa(ido)

  tpi= 2.0D+00 * 4.0D+00 * atan( 1.0D+00 )
  arg = tpi / real ( ip, rk       )
  dcp = cos(arg)
  dsp = sin(arg)
  idp2 = ido+2
  nbd = (ido-1)/2
  ipp2 = ip+2
  ipph = (ip+1)/2

  if (ido < l1) go to 103

      do k = 1, l1
         do i=1,ido
            ch(1,i,k,1) = cc(1,i,1,k)
         end do
      end do

      go to 106

  103 continue

  do i=1,ido
    do k = 1, l1
      ch(1,i,k,1) = cc(1,i,1,k)
    end do
  end do

  106 do 108 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 107 k = 1, l1
            ch(1,1,k,j) = cc(1,ido,j2-2,k)+cc(1,ido,j2-2,k)
            ch(1,1,k,jc) = cc(1,1,j2-1,k)+cc(1,1,j2-1,k)
 1007       continue
  107    continue
  108 continue
      if (ido == 1) go to 116
      if (nbd < l1) go to 112
      do 111 j=2,ipph
         jc = ipp2-j
         do 110 k = 1, l1
            do 109 i=3,ido,2
               ic = idp2-i
               ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
               ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
               ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
               ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
  109       continue
  110    continue
  111 continue
      go to 116
  112 do 115 j=2,ipph
         jc = ipp2-j
         do 114 i=3,ido,2
            ic = idp2-i
            do 113 k = 1, l1
               ch(1,i-1,k,j) = cc(1,i-1,2*j-1,k)+cc(1,ic-1,2*j-2,k)
               ch(1,i-1,k,jc) = cc(1,i-1,2*j-1,k)-cc(1,ic-1,2*j-2,k)
               ch(1,i,k,j) = cc(1,i,2*j-1,k)-cc(1,ic,2*j-2,k)
               ch(1,i,k,jc) = cc(1,i,2*j-1,k)+cc(1,ic,2*j-2,k)
  113       continue
  114    continue
  115 continue
  116 ar1 = 1.0D+00
      ai1 = 0.0D+00
      do 120 l=2,ipph
         lc = ipp2-l
         ar1h = dcp*ar1-dsp*ai1
         ai1 = dcp*ai1+dsp*ar1
         ar1 = ar1h
         do 117 ik=1,idl1
            c2(1,ik,l) = ch2(1,ik,1)+ar1*ch2(1,ik,2)
            c2(1,ik,lc) = ai1*ch2(1,ik,ip)
  117    continue
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do 119 j=3,ipph
            jc = ipp2-j
            ar2h = dc2*ar2-ds2*ai2
            ai2 = dc2*ai2+ds2*ar2
            ar2 = ar2h
            do 118 ik=1,idl1
               c2(1,ik,l) = c2(1,ik,l)+ar2*ch2(1,ik,j)
               c2(1,ik,lc) = c2(1,ik,lc)+ai2*ch2(1,ik,jc)
  118       continue
  119    continue
  120 continue
      do 122 j=2,ipph
         do 121 ik=1,idl1
            ch2(1,ik,1) = ch2(1,ik,1)+ch2(1,ik,j)
  121    continue
  122 continue
      do 124 j=2,ipph
         jc = ipp2-j
         do 123 k = 1, l1
            ch(1,1,k,j) = c1(1,1,k,j)-c1(1,1,k,jc)
            ch(1,1,k,jc) = c1(1,1,k,j)+c1(1,1,k,jc)
  123    continue
  124 continue
      if (ido == 1) go to 132
      if (nbd < l1) go to 128
      do 127 j=2,ipph
         jc = ipp2-j
         do 126 k = 1, l1
            do 125 i=3,ido,2
               ch(1,i-1,k,j) = c1(1,i-1,k,j)-c1(1,i,k,jc)
               ch(1,i-1,k,jc) = c1(1,i-1,k,j)+c1(1,i,k,jc)
               ch(1,i,k,j) = c1(1,i,k,j)+c1(1,i-1,k,jc)
               ch(1,i,k,jc) = c1(1,i,k,j)-c1(1,i-1,k,jc)
  125       continue
  126    continue
  127 continue
      go to 132
  128 do 131 j=2,ipph
         jc = ipp2-j
         do 130 i=3,ido,2
            do 129 k = 1, l1
               ch(1,i-1,k,j) = c1(1,i-1,k,j)-c1(1,i,k,jc)
               ch(1,i-1,k,jc) = c1(1,i-1,k,j)+c1(1,i,k,jc)
               ch(1,i,k,j) = c1(1,i,k,j)+c1(1,i-1,k,jc)
               ch(1,i,k,jc) = c1(1,i,k,j)-c1(1,i-1,k,jc)
  129       continue
  130    continue
  131 continue
  132 continue
      if (ido == 1) return
      do 133 ik=1,idl1
         c2(1,ik,1) = ch2(1,ik,1)
  133 continue
      do 135 j=2,ip
         do 134 k = 1, l1
            c1(1,1,k,j) = ch(1,1,k,j)
  134    continue
  135 continue
      if ( l1 < nbd ) go to 139
      is = -ido
      do 138 j=2,ip
         is = is+ido
         idij = is
         do 137 i=3,ido,2
            idij = idij+2
            do 136 k = 1, l1
               c1(1,i-1,k,j) = wa(idij-1)*ch(1,i-1,k,j)-wa(idij)*ch(1,i,k,j)
               c1(1,i,k,j) = wa(idij-1)*ch(1,i,k,j)+wa(idij)*ch(1,i-1,k,j)
  136       continue
  137    continue
  138 continue
      go to 143
  139 is = -ido
      do 142 j=2,ip
         is = is+ido
         do 141 k = 1, l1
            idij = is
            do 140 i=3,ido,2
               idij = idij+2
               c1(1,i-1,k,j) = wa(idij-1)*ch(1,i-1,k,j)-wa(idij)*ch(1,i,k,j)
               c1(1,i,k,j) = wa(idij-1)*ch(1,i,k,j)+wa(idij)*ch(1,i-1,k,j)
  140       continue
  141    continue
  142 continue
  143 continue

  return
end
subroutine r1fgkf (ido,ip,l1,idl1,cc,c1,c2,in1,ch,ch2,in2,wa)

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R1FGKF is an FFTPACK5.1 auxilliary function.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              idl1
  integer              ido
  integer              in1
  integer              in2
  integer              ip
  integer              l1

  real ( rk       ) ai1
  real ( rk       ) ai2
  real ( rk       ) ar1
  real ( rk       ) ar1h
  real ( rk       ) ar2
  real ( rk       ) ar2h
  real ( rk       ) arg
  real ( rk       ) c1(in1,ido,l1,ip)
  real ( rk       ) c2(in1,idl1,ip)
  real ( rk       ) cc(in1,ido,ip,l1)
  real ( rk       ) ch(in2,ido,l1,ip)
  real ( rk       ) ch2(in2,idl1,ip)
  real ( rk       ) dc2
  real ( rk       ) dcp
  real ( rk       ) ds2
  real ( rk       ) dsp
  integer              i
  integer              ic
  integer              idij
  integer              idp2
  integer              ik
  integer              ipp2
  integer              ipph
  integer              is
  integer              j
  integer              j2
  integer              jc
  integer              k
  integer              l
  integer              lc
  integer              nbd
  real ( rk       ) tpi
  real ( rk       ) wa(ido)

  tpi= 2.0D+00 * 4.0D+00 * atan( 1.0D+00 )
  arg = tpi/real ( ip, rk       )
  dcp = cos(arg)
  dsp = sin(arg)
  ipph = (ip+1)/2
  ipp2 = ip+2
  idp2 = ido+2
  nbd = (ido-1)/2

      if (ido == 1) go to 119

      do ik=1,idl1
         ch2(1,ik,1) = c2(1,ik,1)
      end do

      do j=2,ip
         do k = 1, l1
            ch(1,1,k,j) = c1(1,1,k,j)
         end do
      end do

      if ( l1 < nbd ) go to 107
      is = -ido
      do 106 j=2,ip
         is = is+ido
         idij = is
         do 105 i=3,ido,2
            idij = idij+2
            do 104 k = 1, l1
               ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij)*c1(1,i,k,j)
               ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij)*c1(1,i-1,k,j)
  104       continue
  105    continue
  106 continue
      go to 111
  107 is = -ido
      do 110 j=2,ip
         is = is+ido
         do 109 k = 1, l1
            idij = is
            do 108 i=3,ido,2
               idij = idij+2
               ch(1,i-1,k,j) = wa(idij-1)*c1(1,i-1,k,j)+wa(idij)*c1(1,i,k,j)
               ch(1,i,k,j) = wa(idij-1)*c1(1,i,k,j)-wa(idij)*c1(1,i-1,k,j)
  108       continue
  109    continue
  110 continue
  111 if (nbd < l1) go to 115
      do 114 j=2,ipph
         jc = ipp2-j
         do 113 k = 1, l1
            do 112 i=3,ido,2
               c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
               c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
               c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
               c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
  112       continue
  113    continue
  114 continue
      go to 121
  115 do 118 j=2,ipph
         jc = ipp2-j
         do 117 i=3,ido,2
            do 116 k = 1, l1
               c1(1,i-1,k,j) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
               c1(1,i-1,k,jc) = ch(1,i,k,j)-ch(1,i,k,jc)
               c1(1,i,k,j) = ch(1,i,k,j)+ch(1,i,k,jc)
               c1(1,i,k,jc) = ch(1,i-1,k,jc)-ch(1,i-1,k,j)
  116       continue
  117    continue
  118 continue
      go to 121
  119 do 120 ik=1,idl1
         c2(1,ik,1) = ch2(1,ik,1)
  120 continue
  121 do 123 j=2,ipph
         jc = ipp2-j
         do 122 k = 1, l1
            c1(1,1,k,j) = ch(1,1,k,j)+ch(1,1,k,jc)
            c1(1,1,k,jc) = ch(1,1,k,jc)-ch(1,1,k,j)
  122    continue
  123 continue

      ar1 = 1.0D+00
      ai1 = 0.0D+00
      do 127 l=2,ipph
         lc = ipp2-l
         ar1h = dcp*ar1-dsp*ai1
         ai1 = dcp*ai1+dsp*ar1
         ar1 = ar1h
         do 124 ik=1,idl1
            ch2(1,ik,l) = c2(1,ik,1)+ar1*c2(1,ik,2)
            ch2(1,ik,lc) = ai1*c2(1,ik,ip)
  124    continue
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do 126 j=3,ipph
            jc = ipp2-j
            ar2h = dc2*ar2-ds2*ai2
            ai2 = dc2*ai2+ds2*ar2
            ar2 = ar2h
            do 125 ik=1,idl1
               ch2(1,ik,l) = ch2(1,ik,l)+ar2*c2(1,ik,j)
               ch2(1,ik,lc) = ch2(1,ik,lc)+ai2*c2(1,ik,jc)
  125       continue
  126    continue
  127 continue
      do 129 j=2,ipph
         do 128 ik=1,idl1
            ch2(1,ik,1) = ch2(1,ik,1)+c2(1,ik,j)
  128    continue
  129 continue

      if (ido < l1) go to 132
      do 131 k = 1, l1
         do 130 i=1,ido
            cc(1,i,1,k) = ch(1,i,k,1)
  130    continue
  131 continue
      go to 135
  132 do 134 i=1,ido
         do 133 k = 1, l1
            cc(1,i,1,k) = ch(1,i,k,1)
  133    continue
  134 continue
  135 do 137 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 136 k = 1, l1
            cc(1,ido,j2-2,k) = ch(1,1,k,j)
            cc(1,1,j2-1,k) = ch(1,1,k,jc)
  136    continue
  137 continue
      if (ido == 1) return
      if (nbd < l1) go to 141
      do 140 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 139 k = 1, l1
            do 138 i=3,ido,2
               ic = idp2-i
               cc(1,i-1,j2-1,k) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
               cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j)-ch(1,i-1,k,jc)
               cc(1,i,j2-1,k) = ch(1,i,k,j)+ch(1,i,k,jc)
               cc(1,ic,j2-2,k) = ch(1,i,k,jc)-ch(1,i,k,j)
  138       continue
  139    continue
  140 continue
      return
  141 do 144 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 143 i=3,ido,2
            ic = idp2-i
            do 142 k = 1, l1
               cc(1,i-1,j2-1,k) = ch(1,i-1,k,j)+ch(1,i-1,k,jc)
               cc(1,ic-1,j2-2,k) = ch(1,i-1,k,j)-ch(1,i-1,k,jc)
               cc(1,i,j2-1,k) = ch(1,i,k,j)+ch(1,i,k,jc)
               cc(1,ic,j2-2,k) = ch(1,i,k,jc)-ch(1,i,k,j)
  142       continue
  143    continue
  144 continue

  return
end
subroutine r2w ( ldr, ldw, l, m, r, w )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R2W copies a 2D array, allowing for different leading dimensions.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    13 May 2013
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LDR, the leading dimension of R.
!
!    Input, integer              LDW, the leading dimension of W.
!
!    Input, integer              L, M, the number of rows and columns
!    of information to be copied.
!
!    Input, real ( rk       ) R(LDR,M), the copied information.
!
!    Output, real (  rk       ) W(LDW,M), the original information.
!
  implicit none

  integer              ldr
  integer              ldw
  integer              m

  integer              l
  real ( rk       ) r(ldr,m)
  real ( rk       ) w(ldw,m)

  w(1:l,1:m) = r(1:l,1:m)

  return
end
subroutine r8_factor ( n, nf, fac )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R8_FACTOR factors of an integer for real double precision computations.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the number for which factorization and 
!    other information is needed.
!
!    Output, integer              NF, the number of factors.
!
!    Output, real ( rk       ) FAC(*), a list of factors of N.
!
  implicit none

  real ( rk       ) fac(*)
  integer              j
  integer              n
  integer              nf
  integer              nl
  integer              nq
  integer              nr
  integer              ntry

  nl = n
  nf = 0
  j = 0

  do while ( 1 < nl )

    j = j + 1

    if ( j == 1 ) then
      ntry = 4
    else if ( j == 2 ) then
      ntry = 2
    else if ( j == 3 ) then
      ntry = 3
    else if ( j == 4 ) then
      ntry = 5
    else
      ntry = ntry + 2
    end if

    do

      nq = nl / ntry
      nr = nl - ntry * nq

      if ( nr /= 0 ) then
        exit
      end if

      nf = nf + 1
      fac(nf) = real ( ntry, rk       )
      nl = nq

    end do

  end do

  return
end
subroutine r8_mcfti1 ( n, wa, fnf, fac )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R8_MCFTI1 sets up factors and tables, real double precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  real ( rk       ) fac(*)
  real ( rk       ) fnf
  integer              ido
  integer              ip
  integer              iw
  integer              k1
  integer              l1
  integer              l2
  integer              n
  integer              nf
  real ( rk       ) wa(*)
!
!  Get the factorization of N.
!
  call r8_factor ( n, nf, fac )
  fnf = real ( nf, rk       )
  iw = 1
  l1 = 1
!
!  Set up the trigonometric tables.
!
  do k1 = 1, nf
    ip = int ( fac(k1) )
    l2 = l1 * ip
    ido = n / l2
    call r8_tables ( ido, ip, wa(iw) )
    iw = iw + ( ip - 1 ) * ( ido + ido )
    l1 = l2
  end do

  return
end
subroutine r8_tables ( ido, ip, wa )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! R8_TABLES computes trigonometric tables, real double precision arithmetic.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              ido
  integer              ip

  real ( rk       ) arg1
  real ( rk       ) arg2
  real ( rk       ) arg3
  real ( rk       ) arg4
  real ( rk       ) argz
  integer              i
  integer              j
  real ( rk       ) tpi
  real ( rk       ) wa(ido,ip-1,2)

  tpi = 8.0D+00 * atan ( 1.0D+00 )
  argz = tpi / real ( ip, rk       )
  arg1 = tpi / real ( ido * ip, rk       )

  do j = 2, ip

    arg2 = real ( j - 1, rk       ) * arg1

    do i = 1, ido
      arg3 = real ( i - 1, rk       ) * arg2
      wa(i,j-1,1) = cos ( arg3 )
      wa(i,j-1,2) = sin ( arg3 )
    end do

    if ( 5 < ip ) then
      arg4 = real ( j - 1, rk       ) * argz
      wa(1,j-1,1) = cos ( arg4 )
      wa(1,j-1,2) = sin ( arg4 )
    end if

  end do

  return
end
subroutine rfft1b ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFT1B: real double precision backward fast Fourier transform, 1D.
!
!  Discussion:
!
!    RFFT1B computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the backward
!    transform or Fourier synthesis, transforming the sequence from 
!    spectral to physical space.  This transform is normalized since a 
!    call to RFFT1B followed by a call to RFFT1F (or vice-versa) reproduces
!    the original array within roundoff error. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( rk       ) R(LENR), on input, the data to be 
!    transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1. 
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT1I before the first call to routine
!    RFFT1F or RFFT1B for a given transform length N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4. 
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N. 
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer              lenr
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              inc
  integer              n
  real ( rk       ) r(lenr)
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lenr < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('rfft1b ', 6)
  else if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfft1b ', 8)
  else if (lenwrk < n) then
    ier = 3
    call xerfft ('rfft1b ', 10)
  end if

  if (n == 1) then
    return
  end if

  call rfftb1 (n,inc,r,work,wsave,wsave(n+1))

  return
end
subroutine rfft1f ( n, inc, r, lenr, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFT1F: real double precision forward fast Fourier transform, 1D.
!
!  Discussion:
!
!    RFFT1F computes the one-dimensional Fourier transform of a periodic
!    sequence within a real array.  This is referred to as the forward 
!    transform or Fourier analysis, transforming the sequence from physical
!    to spectral space.  This transform is normalized since a call to 
!    RFFT1F followed by a call to RFFT1B (or vice-versa) reproduces the 
!    original array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence. 
!
!    Input/output, real ( rk       ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1) + 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFT1I before the first call to routine RFFT1F
!    or RFFT1B for a given transform length N. 
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK). 
!
!    Input, integer              LENWRK, the dimension of the WORK array. 
!    LENWRK must be at least N. 
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough:
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough.
!
  implicit none

  integer              lenr
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              inc
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) r(lenr)

  ier = 0

  if (lenr < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('rfft1f ', 6)
  else if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfft1f ', 8)
  else if (lenwrk < n) then
    ier = 3
    call xerfft ('rfft1f ', 10)
  end if

  if (n == 1) then
    return
  end if

  call rfftf1 (n,inc,r,work,wsave,wsave(n+1))

  return
end
subroutine rfft1i ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFT1I: initialization for RFFT1B and RFFT1F.
!
!  Discussion:
!
!    RFFT1I initializes array WSAVE for use in its companion routines 
!    RFFT1B and RFFT1F.  The prime factorization of N together with a
!    tabulation of the trigonometric functions are computed and stored
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N. 
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors of
!    N and also containing certain trigonometric values which will be used in
!    routines RFFT1B or RFFT1F.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer              lensav

  integer              ier
  integer              n
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfft1i ', 3)
  end if

  if (n == 1) then
    return
  end if

  call rffti1 (n,wsave(1),wsave(n+1))

  return
end

subroutine rfftb1 ( n, in, c, ch, wa, fac )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTB1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              in
  integer              n

  real ( rk       ) c(in,*)
  real ( rk       ) ch(*)
  real ( rk       ) fac(15)
  real ( rk       ) half
  real ( rk       ) halfm
  integer              idl1
  integer              ido
  integer              ip
  integer              iw
  integer              ix2
  integer              ix3
  integer              ix4
  integer              j
  integer              k1
  integer              l1
  integer              l2
  integer              modn
  integer              na
  integer              nf
  integer              nl
  real ( rk       ) wa(n)

      nf = INT(fac(2))
      na = 0
      do 10 k1=1,nf
      ip = INT(fac(k1+2))
      na = 1-na      
      if(ip <= 5) go to 10
      if(k1 == nf) go to 10
      na = 1-na
   10 continue 
      half = 0.5D+00
      halfm = -0.5D+00
      modn = mod(n,2)
      nl = n-2
      if(modn /= 0) nl = n-1
      if (na == 0) go to 120

      ch(1) = c(1,1)
      ch(n) = c(1,n)
      do j=2,nl,2
	    ch(j) = half*c(1,j)
	    ch(j+1) = halfm*c(1,j+1)
      end do

      go to 124
  120 do 122 j=2,nl,2
	 c(1,j) = half*c(1,j)
	 c(1,j+1) = halfm*c(1,j+1)
  122 continue
  124 l1 = 1
      iw = 1
      do 116 k1=1,nf
         ip = INT(fac(k1+2))
         l2 = ip*l1
         ido = n/l2
         idl1 = ido*l1
         if (ip /= 4) go to 103
         ix2 = iw+ido
         ix3 = ix2+ido
         if (na /= 0) go to 101
         call r1f4kb (ido,l1,c,in,ch,1,wa(iw),wa(ix2),wa(ix3))
         go to 102
  101    call r1f4kb (ido,l1,ch,1,c,in,wa(iw),wa(ix2),wa(ix3))
  102    na = 1-na
         go to 115
  103    if (ip /= 2) go to 106
         if (na /= 0) go to 104
	 call r1f2kb (ido,l1,c,in,ch,1,wa(iw))
         go to 105
  104    call r1f2kb (ido,l1,ch,1,c,in,wa(iw))
  105    na = 1-na
         go to 115
  106    if (ip /= 3) go to 109
         ix2 = iw+ido
         if (na /= 0) go to 107
	 call r1f3kb (ido,l1,c,in,ch,1,wa(iw),wa(ix2))
         go to 108
  107    call r1f3kb (ido,l1,ch,1,c,in,wa(iw),wa(ix2))
  108    na = 1-na
         go to 115
  109    if (ip /= 5) go to 112
         ix2 = iw+ido
         ix3 = ix2+ido
         ix4 = ix3+ido
         if (na /= 0) go to 110
         call r1f5kb (ido,l1,c,in,ch,1,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 111
  110    call r1f5kb (ido,l1,ch,1,c,in,wa(iw),wa(ix2),wa(ix3),wa(ix4))
  111    na = 1-na
         go to 115
  112    if (na /= 0) go to 113
	 call r1fgkb (ido,ip,l1,idl1,c,c,c,in,ch,ch,1,wa(iw))
         go to 114
  113    call r1fgkb (ido,ip,l1,idl1,ch,ch,ch,1,c,c,in,wa(iw))
  114    if (ido == 1) na = 1-na
  115    l1 = l2
         iw = iw+(ip-1)*ido
  116 continue

  return
end
subroutine rfftf1 ( n, in, c, ch, wa, fac )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTF1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              in
  integer              n

  real ( rk       ) c(in,*)
  real ( rk       ) ch(*)
  real ( rk       ) fac(15)
  integer              idl1
  integer              ido
  integer              ip
  integer              iw
  integer              ix2
  integer              ix3
  integer              ix4
  integer              j
  integer              k1
  integer              kh
  integer              l1
  integer              l2
  integer              modn
  integer              na
  integer              nf
  integer              nl
  real ( rk       ) sn
  real ( rk       ) tsn
  real ( rk       ) tsnm
  real ( rk       ) wa(n)

      nf = int ( fac(2) )
      na = 1
      l2 = n
      iw = n

      do 111 k1=1,nf
         kh = nf-k1
         ip = INT(fac(kh+3))
         l1 = l2/ip
         ido = n/l2
         idl1 = ido*l1
         iw = iw-(ip-1)*ido
         na = 1-na
         if (ip /= 4) go to 102
         ix2 = iw+ido
         ix3 = ix2+ido
         if (na /= 0) go to 101
	     call r1f4kf (ido,l1,c,in,ch,1,wa(iw),wa(ix2),wa(ix3))
         go to 110
  101    call r1f4kf (ido,l1,ch,1,c,in,wa(iw),wa(ix2),wa(ix3))
         go to 110
  102    if (ip /= 2) go to 104
         if (na /= 0) go to 103
	 call r1f2kf (ido,l1,c,in,ch,1,wa(iw))
         go to 110
  103    call r1f2kf (ido,l1,ch,1,c,in,wa(iw))
         go to 110
  104    if (ip /= 3) go to 106
         ix2 = iw+ido
         if (na /= 0) go to 105
	 call r1f3kf (ido,l1,c,in,ch,1,wa(iw),wa(ix2))
         go to 110
  105    call r1f3kf (ido,l1,ch,1,c,in,wa(iw),wa(ix2))
         go to 110
  106    if (ip /= 5) go to 108
         ix2 = iw+ido
         ix3 = ix2+ido
         ix4 = ix3+ido
         if (na /= 0) go to 107
         call r1f5kf (ido,l1,c,in,ch,1,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
  107    call r1f5kf (ido,l1,ch,1,c,in,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
  108    if (ido == 1) na = 1-na
         if (na /= 0) go to 109
	 call r1fgkf (ido,ip,l1,idl1,c,c,c,in,ch,ch,1,wa(iw))
         na = 1
         go to 110
  109    call r1fgkf (ido,ip,l1,idl1,ch,ch,ch,1,c,c,in,wa(iw))
         na = 0
  110    l2 = l1
  111 continue

      sn = 1.0D+00 / real ( n, rk       )
      tsn =  2.0D+00  / real ( n, rk       )
      tsnm = -tsn
      modn = mod(n,2)
      nl = n-2
      if(modn /= 0) nl = n-1
      if (na /= 0) go to 120
      c(1,1) = sn*ch(1)
      do 118 j=2,nl,2
	 c(1,j) = tsn*ch(j)
	 c(1,j+1) = tsnm*ch(j+1)
  118 continue
      if(modn /= 0) return
      c(1,n) = sn*ch(n)
      return
  120 c(1,1) = sn*c(1,1)
      do 122 j=2,nl,2
	 c(1,j) = tsn*c(1,j)
	 c(1,j+1) = tsnm*c(1,j+1)
  122 continue
      if(modn /= 0) return
      c(1,n) = sn*c(1,n)

  return
end
subroutine rffti1 ( n, wa, fac )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTI1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the number for which factorization 
!    and other information is needed.
!
!    Output, real ( rk       ) WA(N), trigonometric information.
!
!    Output, real ( rk       ) FAC(15), factorization information.  
!    FAC(1) is N, FAC(2) is NF, the number of factors, and FAC(3:NF+2) are the 
!    factors.
!
  implicit none

  integer              n

  real ( rk       ) arg
  real ( rk       ) argh
  real ( rk       ) argld
  real ( rk       ) fac(15)
  real ( rk       ) fi
  integer              i
  integer              ib
  integer              ido
  integer              ii
  integer              ip
  integer              ipm
  integer              is
  integer              j
  integer              k1
  integer              l1
  integer              l2
  integer              ld
  integer              nf
  integer              nfm1
  integer              nl
  integer              nq
  integer              nr
  integer              ntry
  integer              ntryh(4)
  real ( rk       ) tpi
  real ( rk       ) wa(n)

  save ntryh

  data ntryh / 4, 2, 3, 5 /

      nl = n
      nf = 0
      j = 0
  101 j = j+1
      if (j-4) 102,102,103
  102 ntry = ntryh(j)
      go to 104
  103 ntry = ntry+2
  104 nq = nl/ntry
      nr = nl-ntry*nq
      if (nr) 101,105,101
  105 nf = nf+1
      fac(nf+2) = ntry
      nl = nq
      if (ntry /= 2) go to 107
      if (nf == 1) go to 107
      do 106 i=2,nf
         ib = nf-i+2
         fac(ib+2) = fac(ib+1)
  106 continue
      fac(3) = 2
  107 if (nl /= 1) go to 104
      fac(1) = n
      fac(2) = nf
      tpi = 8.0D+00 * atan ( 1.0D+00 )
      argh = tpi / real ( n, rk       )
      is = 0
      nfm1 = nf-1
      l1 = 1
      if (nfm1 == 0) return
      do 110 k1=1,nfm1
         ip = INT(fac(k1+2))
         ld = 0
         l2 = l1*ip
         ido = n/l2
         ipm = ip-1
         do 109 j=1,ipm
            ld = ld+l1
            i = is
            argld = real ( ld, rk       ) * argh
            fi = 0.0D+00
            do 108 ii=3,ido,2
               i = i+2
               fi = fi + 1.0D+00
               arg = fi*argld
	       wa(i-1) = cos ( arg )
	       wa(i) = sin ( arg )
  108       continue
            is = is+ido
  109    continue
         l1 = l2
  110 continue

  return
end
subroutine rfftmb ( lot, jump, n, inc, r, lenr, wsave, lensav, work, lenwrk, &
  ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTMB: real double precision backward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    RFFTMB computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a real array.  This transform is referred 
!    to as the backward transform or Fourier synthesis, transforming the
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to RFFTMB followed
!    by a call to RFFTMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, in
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), real array containing LOT 
!    sequences, each having length N.  R can have any number of dimensions, 
!    but the total number of locations must be at least LENR.  On input, the
!    spectral data to be transformed, on output the physical data.
!
!    Input, integer              LENR, the dimension of the R array. 
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1) + 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFTMI before the first call to routine RFFTMF 
!    or RFFTMB for a given transform length N.  
!
!    Input, integer              LENSAV, the dimension of the WSAVE array. 
!    LENSAV must  be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer              lenr
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              inc
  integer              jump
  integer              lot
  integer              n
  real ( rk       ) r(lenr)
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  logical xercon

  ier = 0

  if (lenr < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('rfftmb ', 6)
    return
  else if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfftmb ', 8)
    return
  else if (lenwrk < lot*n) then
    ier = 3
    call xerfft ('rfftmb ', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('rfftmb ', -1)
    return
  end if

  if (n == 1) then
    return
  end if

  call mrftb1 (lot,jump,n,inc,r,work,wsave,wsave(n+1))

  return
end
subroutine rfftmf ( lot, jump, n, inc, r, lenr, wsave, lensav, &
  work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTMF: real double precision forward FFT, 1D, multiple vectors.
!
!  Discussion:
!
!    RFFTMF computes the one-dimensional Fourier transform of multiple 
!    periodic sequences within a real array.  This transform is referred 
!    to as the forward transform or Fourier analysis, transforming the 
!    sequences from physical to spectral space.
!
!    This transform is normalized since a call to RFFTMF followed
!    by a call to RFFTMB (or vice-versa) reproduces the original array
!    within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed 
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), real array containing LOT 
!    sequences, each having length N.  R can have any number of dimensions, but 
!    the total number of locations must be at least LENR.  On input, the
!    physical data to be transformed, on output the spectral data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1) + 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to RFFTMI before the first call to routine RFFTMF 
!    or RFFTMB for a given transform length N.  
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC, JUMP, N, LOT are not consistent.
!
  implicit none

  integer              lenr
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              inc
  integer              jump
  integer              lot
  integer              n
  real ( rk       ) r(lenr)
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  logical xercon

  ier = 0

  if (lenr < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('rfftmf ', 6)
    return
  else if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfftmf ', 8)
    return
  else if (lenwrk < lot*n) then
    ier = 3
    call xerfft ('rfftmf ', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('rfftmf ', -1)
    return
  end if

  if (n == 1) then
    return
  end if

  call mrftf1 (lot,jump,n,inc,r,work,wsave,wsave(n+1))

  return
end
subroutine rfftmi ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! RFFTMI: initialization for RFFTMB and RFFTMF.
!
!  Discussion:
!
!    RFFTMI initializes array WSAVE for use in its companion routines 
!    RFFTMB and RFFTMF.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), work array containing the prime 
!    factors of N and also containing certain trigonometric 
!    values which will be used in routines RFFTMB or RFFTMF.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough.
!
  implicit none

  integer              lensav

  integer              ier
  integer              n
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('rfftmi ', 3)
    return
  end if

  if (n == 1) then
    return
  end if

  call mrfti1 (n,wsave(1),wsave(n+1))

  return
end
subroutine sinq1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQ1B: real double precision backward sine quarter wave transform, 1D.
!
!  Discussion:
!
!    SINQ1B computes the one-dimensional Fourier transform of a sequence 
!    which is a sine series with odd wave numbers.  This transform is 
!    referred to as the backward transform or Fourier synthesis, 
!    transforming the sequence from spectral to physical space.
!
!    This transform is normalized since a call to SINQ1B followed
!    by a call to SINQ1F (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), on input, the sequence to be 
!    transformed.  On output, the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQ1I before the first call to routine SINQ1F 
!    or SINQ1B for a given transform length N.  WSAVE's contents may be 
!    re-used for subsequent calls to SINQ1F and SINQ1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer              IER, the error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              n
  integer              ns2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  real ( rk       ) xhold

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('sinq1b', 6)
  else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sinq1b', 8)
  else if (lenwrk < n) then
    ier = 3
    call xerfft ('sinq1b', 10)
  end if

  if ( 1 < n ) go to 101
!
!     x(1,1) = 4.*x(1,1) line disabled by dick valent 08/26/2010
!
      return
  101 ns2 = n/2
      do 102 k=2,n,2
         x(1,k) = -x(1,k)
  102 continue
      call cosq1b (n,inc,x,lenx,wsave,lensav,work,lenwrk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sinq1b',-5)
        return
      end if

      do 103 k=1,ns2
         kc = n-k
         xhold = x(1,k)
         x(1,k) = x(1,kc+1)
         x(1,kc+1) = xhold
  103 continue

  return
end
subroutine sinq1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQ1F: real double precision forward sine quarter wave transform, 1D.
!
!  Discussion:
! 
!    SINQ1F computes the one-dimensional Fourier transform of a sequence 
!    which is a sine series of odd wave numbers.  This transform is 
!    referred to as the forward transform or Fourier analysis, transforming 
!    the sequence from physical to spectral space.
!
!    This transform is normalized since a call to SINQ1F followed
!    by a call to SINQ1B (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    13 May 2013
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), on input, the sequence to be 
!    transformed.  On output, the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQ1I before the first call to routine SINQ1F 
!    or SINQ1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQ1F and SINQ1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least N.
!
!    Output, integer              IER, the error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lenx
  integer              n
  integer              ns2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  real ( rk       ) xhold

  ier = 0

  if ( lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('sinq1f', 6)
    return
  end if

  if ( lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sinq1f', 8)
    return
  end if

  if (lenwrk < n) then
    ier = 3
    call xerfft ('sinq1f', 10)
    return
  end if

  if (n == 1) then
    return
  end if

  ns2 = n/2
  do k=1,ns2
    kc = n-k
    xhold = x(1,k)
    x(1,k) = x(1,kc+1)
    x(1,kc+1) = xhold
  end do

  call cosq1f (n,inc,x,lenx,wsave,lensav,work,lenwrk,ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sinq1f',-5)
    return
  end if

  do k=2,n,2
    x(1,k) = -x(1,k)
  end do

  return
end
subroutine sinq1i ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQ1I: initialization for SINQ1B and SINQ1F.
!
!  Discussion:
!
!    SINQ1I initializes array WSAVE for use in its companion routines 
!    SINQ1F and SINQ1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!   in routines SINQ1B or SINQ1F.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  integer              ier
  integer              ier1
  integer              n
  real ( rk       ) wsave(lensav)

      ier = 0

      if (lensav < 2*n + int(log( real ( n, rk       ) ) &
        /log( 2.0D+00 )) +4) then
        ier = 2
        call xerfft ('sinq1i', 3)
        go to 300
      end if

      call cosq1i (n, wsave, lensav, ier1)
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sinq1i',-5)
      end if
  300 continue

  return
end
subroutine sinqmb ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQMB: real double precision backward sine quarter wave, multiple vectors.
!
!  Discussion:
!
!    SINQMB computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each of the sequences is a 
!    sine series with odd wave numbers.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to SINQMB followed
!    by a call to SINQMF (or vice-versa) reproduces the original
!    array  within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations, in
!    array R, of the first elements of two consecutive sequences to be 
!    transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQMI before the first call to routine SINQMF
!    or SINQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQMF and SINQMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lj
  integer              lot
  integer              m
  integer              n
  integer              ns2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon
  real ( rk       ) xhold

      ier = 0

      if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
        ier = 1
        call xerfft ('sinqmb', 6)
      else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
        /log( 2.0D+00 )) +4) then
        ier = 2
        call xerfft ('sinqmb', 8)
      else if (lenwrk < lot*n) then
        ier = 3
        call xerfft ('sinqmb', 10)
      else if (.not. xercon(inc,jump,n,lot)) then
        ier = 4
        call xerfft ('sinqmb', -1)
      end if

      lj = (lot-1)*jump+1
      if (1 < n ) go to 101

      do m=1,lj,jump
         x(m,1) =  4.0D+00 * x(m,1)
      end do

      return
  101 ns2 = n/2

    do k=2,n,2
       do m=1,lj,jump
         x(m,k) = -x(m,k)
      end do
    end do

      call cosqmb (lot,jump,n,inc,x,lenx,wsave,lensav,work,lenwrk,ier1)
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sinqmb',-5)
        go to 300
      end if
      do 103 k=1,ns2
         kc = n-k
         do 203 m=1,lj,jump
         xhold = x(m,k)
         x(m,k) = x(m,kc+1)
         x(m,kc+1) = xhold
 203     continue
  103 continue
  300 continue

  return
end
subroutine sinqmf ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQMF: real double precision forward sine quarter wave, multiple vectors.
!
!  Discussion:
!
!    SINQMF computes the one-dimensional Fourier transform of multiple 
!    sequences within a real array, where each sequence is a sine series 
!    with odd wave numbers.  This transform is referred to as the forward
!    transform or Fourier synthesis, transforming the sequences from 
!    spectral to physical space.
!
!    This transform is normalized since a call to SINQMF followed
!    by a call to SINQMB (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed
!    within array R.
!
!    Input, integer              JUMP, the increment between the locations,
!    in array R, of the first elements of two consecutive sequences to 
!    be transformed.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINQMI before the first call to routine SINQMF 
!    or SINQMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINQMF and SINQMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*N.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              jump
  integer              k
  integer              kc
  integer              lenx
  integer              lj
  integer              lot
  integer              m
  integer              n
  integer              ns2
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon
  real ( rk       ) xhold

      ier = 0

      if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
        ier = 1
        call xerfft ('sinqmf', 6)
        return
      else if (lensav < 2*n + int(log( real ( n, rk       ) ) &
        /log( 2.0D+00 )) +4) then
        ier = 2
        call xerfft ('sinqmf', 8)
        return
      else if (lenwrk < lot*n) then
        ier = 3
        call xerfft ('sinqmf', 10)
        return
      else if (.not. xercon(inc,jump,n,lot)) then
        ier = 4
        call xerfft ('sinqmf', -1)
        return
      end if

      if (n == 1) then
        return
      end if

      ns2 = n/2
      lj = (lot-1)*jump+1
      do 101 k=1,ns2
         kc = n-k
         do m=1,lj,jump
           xhold = x(m,k)
           x(m,k) = x(m,kc+1)
           x(m,kc+1) = xhold
         end do
  101 continue
      call cosqmf (lot,jump,n,inc,x,lenx,wsave,lensav,work,lenwrk,ier1)

      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sinqmf',-5)
        return
      end if

  do k=2,n,2
    do m=1,lj,jump
      x(m,k) = -x(m,k)
    end do
  end do

  300 continue

  return
end
subroutine sinqmi ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINQMI: initialization for SINQMB and SINQMF.
!
!  Discussion:
!
!    SINQMI initializes array WSAVE for use in its companion routines 
!    SINQMF and SINQMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least 2*N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!    in routines SINQMB or SINQMF.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  integer              ier
  integer              ier1
  integer              n
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < 2*n + int(log( real ( n, rk       ) )/log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sinqmi', 3)
    return
  end if

  call cosqmi (n, wsave, lensav, ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sinqmi',-5)
  end if

  return
end
subroutine sint1b ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINT1B: real double precision backward sine transform, 1D.
!
!  Discussion:
!
!    SINT1B computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequence from spectral to physical space.
!
!    This transform is normalized since a call to SINT1B followed
!    by a call to SINT1F (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINT1I before the first call to routine SINT1F 
!    or SINT1B for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINT1F and SINT1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              lenx
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('sint1b', 6)
    return
  else if ( lensav < n / 2 + n + int ( log ( real ( n, rk       ) ) &
    / log ( 2.0D+00 ) ) + 4 ) then
    ier = 2
    call xerfft ('sint1b', 8)
    return
  else if (lenwrk < (2*n+2)) then
    ier = 3
    call xerfft ('sint1b', 10)
    return
  end if

  call sintb1(n,inc,x,wsave,work,work(n+2),ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sint1b',-5)
  end if

  return
end
subroutine sint1f ( n, inc, x, lenx, wsave, lensav, work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINT1F: real double precision forward sine transform, 1D.
!
!  Discussion:
!
!    SINT1F computes the one-dimensional Fourier transform of an odd 
!    sequence within a real array.  This transform is referred to as the 
!    forward transform or Fourier analysis, transforming the sequence
!    from physical to spectral space.
!
!    This transform is normalized since a call to SINT1F followed
!    by a call to SINT1B (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, 
!    in array R, of two consecutive elements within the sequence.
!
!    Input/output, real ( rk       ) R(LENR), on input, contains the sequence 
!    to be transformed, and on output, the transformed sequence.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINT1I before the first call to routine SINT1F 
!    or SINT1B for a given transform length N.  WSAVE's contents may be re-used
!    for subsequent calls to SINT1F and SINT1B with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least 2*N+2.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              lenx
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)

  ier = 0

  if (lenx < inc*(n-1) + 1) then
    ier = 1
    call xerfft ('sint1f', 6)
    return
  else if (lensav < n/2 + n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sint1f', 8)
    return
  else if (lenwrk < (2*n+2)) then
    ier = 3
    call xerfft ('sint1f', 10)
    return
  end if

  call sintf1(n,inc,x,wsave,work,work(n+2),ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sint1f',-5)
  end if

  return
end
subroutine sint1i ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINT1I: initialization for SINT1B and SINT1F.
!
!  Discussion:
!
!    SINT1I initializes array WSAVE for use in its companion routines 
!    SINT1F and SINT1B.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of the sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product 
!    of small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors 
!    of N and also containing certain trigonometric values which will be used 
!    in routines SINT1B or SINT1F.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  integer              ier
  integer              ier1
  integer              k
  integer              lnsv
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) pi
  real ( rk       ) wsave(lensav)

  ier = 0

  if (lensav < n/2 + n + int(log( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sint1i', 3)
    return
  end if

  pi =  4.0D+00 * atan ( 1.0D+00 )

  if (n <= 1) then
    return
  end if

  ns2 = n/2
  np1 = n+1
  dt = pi / real ( np1, rk       )

  do k=1,ns2
    wsave(k) =  2.0D+00 *sin(k*dt)
  end do

  lnsv = np1 + int(log( real ( np1, rk       ))/log( 2.0D+00 )) +4

  call rfft1i (np1, wsave(ns2+1), lnsv, ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sint1i',-5)
  end if

  return
end
subroutine sintb1 ( n, inc, x, wsave, xh, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINTB1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum
  real ( rk       ) fnp1s4
  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lnsv
  integer              lnwk
  integer              lnxh
  integer              modn
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) srt3s2
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xh(*)
  real ( rk       ) xhold

      ier = 0
      if (n-2) 200,102,103
  102 srt3s2 = sqrt( 3.0D+00 ) / 2.0D+00
      xhold = srt3s2*(x(1,1)+x(1,2))
      x(1,2) = srt3s2*(x(1,1)-x(1,2))
      x(1,1) = xhold
      return

  103 np1 = n+1
      ns2 = n/2
      do 104 k=1,ns2
         kc = np1-k
         t1 = x(1,k)-x(1,kc)
         t2 = wsave(k)*(x(1,k)+x(1,kc))
         xh(k+1) = t1+t2
         xh(kc+1) = t2-t1
  104 continue
      modn = mod(n,2)
      if (modn == 0) go to 124
      xh(ns2+2) =  4.0D+00 * x(1,ns2+1)
  124 xh(1) = 0.0D+00
      lnxh = np1
      lnsv = np1 + int(log( real ( np1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = np1

      call rfft1f(np1,1,xh,lnxh,wsave(ns2+1),lnsv,work,lnwk,ier1)    
 
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sintb1',-5)
        return
      end if

      if(mod(np1,2) /= 0) go to 30
      xh(np1) = xh(np1)+xh(np1)
 30   fnp1s4 = real ( np1, rk       ) / 4.0D+00
         x(1,1) = fnp1s4*xh(1)
         dsum = x(1,1)
      do i=3,n,2
            x(1,i-1) = fnp1s4*xh(i)
            dsum = dsum+fnp1s4*xh(i-1)
            x(1,i) = dsum
      end do

      if ( modn == 0 ) then
         x(1,n) = fnp1s4*xh(n+1)
      end if

  200 continue

  return
end
subroutine sintf1 ( n, inc, x, wsave, xh, work, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINTF1 is an FFTPACK5.1 auxiliary routine.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
  implicit none

  integer              inc

  real ( rk       ) dsum
  integer              i
  integer              ier
  integer              ier1
  integer              k
  integer              kc
  integer              lnsv
  integer              lnwk
  integer              lnxh
  integer              modn
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) sfnp1
  real ( rk       ) ssqrt3
  real ( rk       ) t1
  real ( rk       ) t2
  real ( rk       ) work(*)
  real ( rk       ) wsave(*)
  real ( rk       ) x(inc,*)
  real ( rk       ) xh(*)
  real ( rk       ) xhold

      ier = 0
      if (n-2) 200,102,103
  102 ssqrt3 = 1.0D+00 / sqrt ( 3.0D+00 )
      xhold = ssqrt3*(x(1,1)+x(1,2))
      x(1,2) = ssqrt3*(x(1,1)-x(1,2))
      x(1,1) = xhold
      go to 200
  103 np1 = n+1
      ns2 = n/2
      do k=1,ns2
         kc = np1-k
         t1 = x(1,k)-x(1,kc)
         t2 = wsave(k)*(x(1,k)+x(1,kc))
         xh(k+1) = t1+t2
         xh(kc+1) = t2-t1
      end do

      modn = mod(n,2)
      if (modn == 0) go to 124
      xh(ns2+2) =  4.0D+00 * x(1,ns2+1)
  124 xh(1) = 0.0D+00
      lnxh = np1
      lnsv = np1 + int(log( real ( np1, rk       ))/log( 2.0D+00 )) + 4
      lnwk = np1

      call rfft1f(np1,1,xh,lnxh,wsave(ns2+1),lnsv,work, lnwk,ier1)     
      if (ier1 /= 0) then
        ier = 20
        call xerfft ('sintf1',-5)
        go to 200
      end if

      if(mod(np1,2) /= 0) go to 30
      xh(np1) = xh(np1)+xh(np1)
   30 sfnp1 = 1.0D+00 / real ( np1, rk       )
         x(1,1) = 0.5D+00 * xh(1)
         dsum = x(1,1)

      do i=3,n,2
            x(1,i-1) = 0.5D+00 * xh(i)
            dsum = dsum + 0.5D+00 * xh(i-1)
            x(1,i) = dsum
      end do

      if (modn /= 0) go to 200
      x(1,n) = 0.5D+00 * xh(n+1)
  200 continue

  return
end
subroutine sintmb ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINTMB: real double precision backward sine transform, multiple vectors.
!
!  Discussion:
!
!    SINTMB computes the one-dimensional Fourier transform of multiple 
!    odd sequences within a real array.  This transform is referred to as 
!    the backward transform or Fourier synthesis, transforming the 
!    sequences from spectral to physical space.
!
!    This transform is normalized since a call to SINTMB followed
!    by a call to SINTMF (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be transformed
!    within the array R.
!
!    Input, integer              JUMP, the increment between the locations, in 
!    array R, of the first elements of two consecutive sequences.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINTMI before the first call to routine SINTMF 
!    or SINTMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINTMF and SINTMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(2*N+4).
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              iw1
  integer              iw2
  integer              jump
  integer              lenx
  integer              lot
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon

  ier = 0

  if (lenx < (lot-1)*jump + inc*(n-1) + 1) then
    ier = 1
    call xerfft ('sintmb', 6)
    return
  else if ( lensav < n / 2 + n + int ( log ( real ( n, rk       ) ) &
    /log( 2.0D+00 )) +4) then
    ier = 2
    call xerfft ('sintmb', 8)
    return
  else if (lenwrk < lot*(2*n+4)) then
    ier = 3
    call xerfft ('sintmb', 10)
    return
  else if (.not. xercon(inc,jump,n,lot)) then
    ier = 4
    call xerfft ('sintmb', -1)
    return
  end if

  iw1 = lot+lot+1
  iw2 = iw1+lot*(n+1)

  call msntb1(lot,jump,n,inc,x,wsave,work,work(iw1),work(iw2),ier1)

  if (ier1 /= 0) then
    ier = 20
    call xerfft ('sintmb',-5)
    return
  end if

  return
end
subroutine sintmf ( lot, jump, n, inc, x, lenx, wsave, lensav, &
  work, lenwrk, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINTMF: real double precision forward sine transform, multiple vectors.
!
!  Discussion:
!
!    SINTMF computes the one-dimensional Fourier transform of multiple 
!    odd sequences within a real array.  This transform is referred to as 
!    the forward transform or Fourier analysis, transforming the sequences 
!    from physical to spectral space.
!
!    This transform is normalized since a call to SINTMF followed
!    by a call to SINTMB (or vice-versa) reproduces the original
!    array within roundoff error.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LOT, the number of sequences to be 
!    transformed within.
!
!    Input, integer              JUMP, the increment between the locations, 
!    in array R, of the first elements of two consecutive sequences.
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N+1 is a product of 
!    small primes.
!
!    Input, integer              INC, the increment between the locations, in 
!    array R, of two consecutive elements within the same sequence.
!
!    Input/output, real ( rk       ) R(LENR), containing LOT sequences, each 
!    having length N.  R can have any number of dimensions, but the total 
!    number of locations must be at least LENR.  On input, R contains the data
!    to be transformed, and on output, the transformed data.
!
!    Input, integer              LENR, the dimension of the R array.  
!    LENR must be at least (LOT-1)*JUMP + INC*(N-1)+ 1.
!
!    Input, real ( rk       ) WSAVE(LENSAV).  WSAVE's contents must be 
!    initialized with a call to SINTMI before the first call to routine SINTMF
!    or SINTMB for a given transform length N.  WSAVE's contents may be re-used 
!    for subsequent calls to SINTMF and SINTMB with the same N.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Workspace, real ( rk       ) WORK(LENWRK).
!
!    Input, integer              LENWRK, the dimension of the WORK array.  
!    LENWRK must be at least LOT*(2*N+4).
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    1, input parameter LENR   not big enough;
!    2, input parameter LENSAV not big enough;
!    3, input parameter LENWRK not big enough;
!    4, input parameters INC,JUMP,N,LOT are not consistent;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              inc
  integer              lensav
  integer              lenwrk

  integer              ier
  integer              ier1
  integer              iw1
  integer              iw2
  integer              jump
  integer              lenx
  integer              lot
  integer              n
  real ( rk       ) work(lenwrk)
  real ( rk       ) wsave(lensav)
  real ( rk       ) x(inc,*)
  logical xercon

  ier = 0

  if ( lenx < ( lot - 1) * jump + inc * ( n - 1 ) + 1 ) then
    ier = 1
    call xerfft ( 'sintmf', 6 )
    return
  else if ( lensav < n / 2 + n + int ( log ( real ( n, rk       ) ) &
    / log ( 2.0D+00 ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sintmf', 8 )
    return
  else if ( lenwrk < lot * ( 2 * n + 4 ) ) then
    ier = 3
    call xerfft ( 'sintmf', 10 )
    return
  else if ( .not. xercon ( inc, jump, n, lot ) ) then
    ier = 4
    call xerfft ( 'sintmf', -1 )
    return
  end if

  iw1 = lot + lot + 1
  iw2 = iw1 + lot * ( n + 1 )
  call msntf1 ( lot, jump, n, inc, x, wsave, work, work(iw1), work(iw2), ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintmf', -5 )
  end if

  return
end
subroutine sintmi ( n, wsave, lensav, ier )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! SINTMI: initialization for SINTMB and SINTMF.
!
!  Discussion:
!
!    SINTMI initializes array WSAVE for use in its companion routines 
!    SINTMF and SINTMB.  The prime factorization of N together with a 
!    tabulation of the trigonometric functions are computed and stored 
!    in array WSAVE.  Separate WSAVE arrays are required for different 
!    values of N.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              N, the length of each sequence to be 
!    transformed.  The transform is most efficient when N is a product of 
!    small primes.
!
!    Input, integer              LENSAV, the dimension of the WSAVE array.  
!    LENSAV must be at least N/2 + N + INT(LOG(REAL(N))) + 4.
!
!    Output, real ( rk       ) WSAVE(LENSAV), containing the prime factors
!    of N and also containing certain trigonometric values which will be used
!    in routines SINTMB or SINTMF.
!
!    Output, integer              IER, error flag.
!    0, successful exit;
!    2, input parameter LENSAV not big enough;
!    20, input error returned by lower level routine.
!
  implicit none

  integer              lensav

  real ( rk       ) dt
  integer              ier
  integer              ier1
  integer              k
  integer              lnsv
  integer              n
  integer              np1
  integer              ns2
  real ( rk       ) pi
  real ( rk       ) wsave(lensav)

  ier = 0

  if ( lensav < n / 2 + n + int ( log ( real ( n, rk       ) ) &
    / log ( 2.0D+00 ) ) + 4 ) then
    ier = 2
    call xerfft ( 'sintmi', 3 )
    return
  end if

  pi = 4.0D+00 * atan ( 1.0D+00 )

  if ( n <= 1 ) then
    return
  end if

  ns2 = n / 2
  np1 = n + 1
  dt = pi / real ( np1, rk       )

  do k = 1, ns2
    wsave(k) = 2.0D+00 * sin ( k * dt )
  end do

  lnsv = np1 + int ( log ( real ( np1, rk       ) ) / log ( 2.0D+00 ) ) + 4
  call rfftmi ( np1, wsave(ns2+1), lnsv, ier1 )

  if ( ier1 /= 0 ) then
    ier = 20
    call xerfft ( 'sintmi', -5 )
  end if

  return
end
subroutine w2r ( ldr, ldw, l, m, r, w )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! W2R copies a 2D array, allowing for different leading dimensions.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    13 May 2013
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              LDR, the leading dimension of R.
!
!    Input, integer              LDW, the leading dimension of W.
!
!    Input, integer              L, M, the number of rows and columns
!    of information to be copied.
!
!    Output, real ( rk       ) R(LDR,M), the copied information.
!
!    Input, real (  rk       ) W(LDW,M), the original information.
!
  implicit none

  integer              ldr
  integer              ldw
  integer              m

  integer              i
  integer              j
  integer              l
  real ( rk       ) r(ldr,m)
  real ( rk       ) w(ldw,m)

  do j = 1, m
    do i = 1, l
      r(i,j) = w(i,j)
    end do
  end do

  return
end
function xercon ( inc, jump, n, lot )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! XERCON checks INC, JUMP, N and LOT for consistency.
!
!  Discussion:
!
!    Positive integers INC, JUMP, N and LOT are "consistent" if,
!    for any values I1 and I2 < N, and J1 and J2 < LOT,
!
!      I1 * INC + J1 * JUMP = I2 * INC + J2 * JUMP
!
!    can only occur if I1 = I2 and J1 = J2.
!
!    For multiple FFT's to execute correctly, INC, JUMP, N and LOT must
!    be consistent, or else at least one array element will be
!    transformed more than once.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, integer              INC, JUMP, N, LOT, the parameters to check.
!
!    Output, logical XERCON, is TRUE if the parameters are consistent.
!
  implicit none

  integer              i
  integer              inc
  integer              j
  integer              jnew
  integer              jump
  integer              lcm
  integer              lot
  integer              n
  logical xercon

  i = inc
  j = jump

  do while ( j /= 0 )
    jnew = mod ( i, j )
    i = j
    j = jnew
  end do
!
!  LCM = least common multiple of INC and JUMP.
!
  lcm = ( inc * jump ) / i

  if ( lcm <= ( n - 1 ) * inc .and. lcm <= ( lot - 1 ) * jump ) then
    xercon = .false.
  else
    xercon = .true.
  end if

  return
end
subroutine xerfft ( srname, info )

  use dlf_parameter_module, only: rk !*****************************************************************************80
!
!! XERFFT is an error handler for the FFTPACK routines.
!
!  Discussion:
!
!    XERFFT is an error handler for FFTPACK version 5.1 routines.
!    It is called by an FFTPACK 5.1 routine if an input parameter has an
!    invalid value.  A message is printed and execution stops.
!
!    Installers may consider modifying the stop statement in order to
!    call system-specific exception-handling facilities.
!
!  License:
!
!    Licensed under the GNU General Public License (GPL).
!    Copyright (C) 1995-2004, Scientific Computing Division,
!    University Corporation for Atmospheric Research
!
!  Modified:
!
!    15 November 2011
!
!  Author:
!
!    Original FORTRAN77 version by Paul Swarztrauber, Richard Valent.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Paul Swarztrauber,
!    Vectorizing the Fast Fourier Transforms,
!    in Parallel Computations,
!    edited by G. Rodrigue,
!    Academic Press, 1982.
!
!    Paul Swarztrauber,
!    Fast Fourier Transform Algorithms for Vector Computers,
!    Parallel Computing, pages 45-63, 1984.
!
!  Parameters:
!
!    Input, character ( len = * ) SRNAME, the name of the calling routine.
!
!    Input, integer              INFO, an error code.  When a single invalid 
!    parameter in the parameter list of the calling routine has been detected, 
!    INFO is the position of that parameter.  In the case when an illegal 
!    combination of LOT, JUMP, N, and INC has been detected, the calling 
!    subprogram calls XERFFT with INFO = -1.
!
  implicit none

  integer              info
  character ( len = * ) srname

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'XERFFT - Fatal error!'

  if ( 1 <= info ) then
    write ( *, '(a,a,a,i3,a)' ) '  On entry to ', trim ( srname ), &
      ' parameter number ', info, ' had an illegal value.'
  else if ( info == -1 ) then
    write ( *, '(a,a,a,a)' ) '  On entry to ', trim ( srname ), &
      ' parameters LOT, JUMP, N and INC are inconsistent.'
  else if ( info == -2 ) then
    write ( *, '(a,a,a,a)' ) '  On entry to ', trim ( srname ), &
      ' parameter L is greater than LDIM.'
  else if ( info == -3 ) then
    write ( *, '(a,a,a,a)' ) '  On entry to ', trim ( srname ), &
      ' parameter M is greater than MDIM.'
  else if ( info == -5 ) then
    write ( *, '(a,a,a,a)' ) '  Within ', trim ( srname ), &
      ' input error returned by lower level routine.'
  else if ( info == -6 ) then
    write ( *, '(a,a,a,a)' ) '  On entry to ', trim ( srname ), &
      ' parameter LDIM is less than 2*(L/2+1).'
  end if

  stop
end
