subroutine mat_uut_eig_r(n, k, A, U)

   ! For symmetric matrix A,
   ! this subroutine finds real orthogonal
   ! matrix U such that
   !
   !     A is approximately UU^T.
   !
   ! where A is (n x n) and U is (n x k).
   ! This is done by taking the eigenvectors
   ! that correspond to the k largest eigenvalues
   ! from the eigendecomposition of A.

   implicit none
   integer, intent(in) :: n, k
   double precision, intent(in) :: A(n, n)
   double precision, intent(out) :: U(n, min(n, k))
   double precision :: eval(n), evec(n, n)
   integer :: j, jj

   call MAT_DIAG(A, n, n, eval, evec)

   ! TODO: enforce order. Right now assumes that cuSolver/LAPACK is used ==> ascending order
   do j = 1, min(n, k)
      jj = n - j + 1
      if (eval(jj) <= 0.0d0) exit
      U(:, j) = evec(:, jj)*sqrt(eval(jj))
   end do

end subroutine mat_uut_eig_r
