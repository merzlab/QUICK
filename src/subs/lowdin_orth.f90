!
!	lowdin_orth.f90
!

subroutine lowdin_orth(n, S, P, sqrtS, wantsqrtSinv, sqrtSinv, out)

   ! Writes S^{1/2}PS^{1/2} to `out`

   integer, intent(in) :: n
   double precision, intent(in) :: S(n, n), P(n, n)
   logical, intent(in) :: wantsqrtSinv
   double precision, intent(out) :: sqrtS(n, n), sqrtSinv(n, n), out(n, n)

   integer :: j
   double precision :: eval(n), evec(n, n)
   double precision :: hold1(n, n), hold2(n, n), hold3(n, n)

   ! assumes ascending order eigenvalues
   call MAT_DIAG(S, n, n, eval, evec)

   hold1 = 0.0d0
   hold2 = 0.0d0
   hold3 = 0.0d0

   ! S = UAU^T = evec*(hold1)^2*evec^T
   ! ==> hold1 = A^{1/2}
   !     hold3 = A^{-1/2}
   do j=n, 1, -1
      if (eval(j) <= 0.0d0) exit
      hold1(j,j) = sqrt(eval(j))
      hold3(j,j) = 1.0d0/hold1(j,j)
   enddo

   ! hold2 = U*A^{1/2} = evec*hold1
   call MAT_DGEMM('n', 'n', n, n, n, 1.0d0, evec, n, hold1, n, 0.0d0, hold2, n)
   ! sqrtS = S^{1/2} = (U*A^{1/2})*U^T = hold2*evec
   call MAT_DGEMM('n', 't', n, n, n, 1.0d0, hold2, n, evec, n, 0.0d0, sqrtS, n)
   ! hold1 = S^{1/2}P = sqrtS*P
   call MAT_DGEMM('n', 'n', n, n, n, 1.0d0, sqrtS, n, P, n, 0.0d0, hold1, n)
   ! out = (S^{1/2}P)S^{1/2} = hold1*sqrtS
   call MAT_DGEMM('n', 'n', n, n, n, 1.0d0, hold1, n, sqrtS, n, 0.0d0, out, n)
   
   if (wantsqrtSinv) then
      ! hold2 = U*A^{-1/2} = evec*hold3
      call MAT_DGEMM('n', 'n', n, n, n, 1.0d0, evec, n, hold3, n, 0.0d0, hold2, n)
      ! sqrtSinv = S^{-1/2} = (U*A^{-1/2})*U^T = hold2*evec
      call MAT_DGEMM('n', 't', n, n, n, 1.0d0, hold2, n, evec, n, 0.0d0, sqrtSinv, n)
   endif

end subroutine lowdin_orth
