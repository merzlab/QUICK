module quick_cuest_module
   !
   ! This module contains the fortran bindings for C functions that call cuEST
   !
   use, intrinsic::iso_c_binding, only: c_int64_t, c_ptr
   implicit none

   ! unless commented otherwise, C will not modify the memory a pointer points to

   interface
      subroutine cuest_init(natom, nshell, nbasis, nauxshell, maxcontract, maxcontract_aux, xyz, chg, nextatom, extxyz, extchg) &
         bind(c, name="cuest_init")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_ptr
         implicit none
         integer(c_int64_t), intent(in), value :: natom
         integer(c_int64_t), intent(in), value :: nshell
         integer(c_int64_t), intent(in), value :: nbasis
         integer(c_int64_t), intent(in), value :: nauxshell
         integer(c_int64_t), intent(in), value :: maxcontract
         integer(c_int64_t), intent(in), value :: maxcontract_aux
         type(c_ptr), intent(in), value :: xyz ! double
         real(c_double), intent(in) :: chg(*)
         integer(c_int64_t), intent(in), value :: nextatom
         real(c_double), intent(in) :: extxyz(*)
         real(c_double), intent(in) :: extchg(*)
      end subroutine cuest_init
   end interface

   interface
      subroutine cuest_deinit() bind(c, name="cuest_deinit")
      end subroutine
   end interface

   interface
      subroutine cuest_init_basis(ncenter, katom, ktype, kprim, gcexpo, gccoeff, aux) &
         bind(c, name="cuest_init_basis")
         use, intrinsic::iso_c_binding, only: c_int64_t, c_double, c_bool
         implicit none
         integer(c_int64_t), intent(in) :: ncenter(*)
         integer(c_int64_t), intent(in) :: katom(*)
         integer(c_int64_t), intent(in) :: ktype(*)
         integer(c_int64_t), intent(in) :: kprim(*)
         real(c_double), intent(in) :: gcexpo(*)
         real(c_double), intent(in) :: gccoeff(*)
         logical(c_bool), intent(in), value :: aux
      end subroutine cuest_init_basis
   end interface

   interface
      subroutine cuest_init_oei_plan(cutoff) bind(c, name="cuest_init_oei_plan")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(in), value :: cutoff
      end subroutine cuest_init_oei_plan
   end interface

   interface
      subroutine cuest_init_dfint_plan() bind(c, name="cuest_init_dfint_plan")
      end subroutine cuest_init_dfint_plan
   end interface

   interface
      subroutine cuest_get_oei_S(o) bind(c, name="cuest_get_oei_S")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_S
   end interface

   interface
      subroutine cuest_get_oei_T(o) bind(c, name="cuest_get_oei_T")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_T
   end interface

   interface
      subroutine cuest_get_oei_V(o) bind(c, name="cuest_get_oei_V")
         use, intrinsic::iso_c_binding, only: c_ptr
         type(c_ptr), intent(in), value :: o ! double; modified
      end subroutine cuest_get_oei_V
   end interface

   interface
      subroutine cuest_get_eri_J(o, dense) bind(c, name="cuest_get_eri_J")
         use, intrinsic::iso_c_binding, only: c_ptr, c_double
         type(c_ptr), intent(in), value :: o ! double; modified
         real(c_double), intent(in) :: dense(*)
      end subroutine cuest_get_eri_J
   end interface

   interface
      subroutine cuest_get_eri_K(o, C, nocc) bind(c, name="cuest_get_eri_K")
         use, intrinsic::iso_c_binding, only: c_ptr, c_double, c_int64_t
         type(c_ptr), intent(in), value :: o ! double; modified
         real(c_double), intent(in) :: C(*)
         integer(c_int64_t), intent(in), value :: nocc
      end subroutine cuest_get_eri_K
   end interface

contains

   subroutine unify_cart_norm(coeff)
      !
      ! Undoes extra normalization for cartesian d and f orbitals
      !     d_xy  type has extra 1/sqrt(3)
      !     f_xxy type has extra 1/sqrt(5)
      !     f_xyz type has extra 1/sqrt(15)
      !
      use quick_basis_module, only: itype, ncontract, nbasis
      implicit none

      double precision, intent(inout) :: coeff(:, :)
      ! counters
      integer :: Ibas, Icon
      ! itype stuff
      integer :: l1, l2, l3, lsum, lmax
      double precision :: k

      do Ibas = 1, nbasis
         l1 = itype(1, Ibas)
         l2 = itype(2, Ibas)
         l3 = itype(3, Ibas)
         lsum = l1 + l2 + l3

         k = 1.0d0 ! factor

         if (lsum == 2 .and. max(l1, max(l2, l3)) == 1) then ! D and off diagonal
            k = dsqrt(3.0d0)
         else if (lsum == 3) then ! F
            lmax = max(l1, max(l2, l3))
            if (lmax == 1) then ! xyz
               k = dsqrt(15.0d0)
            else if (lmax == 2) then ! xxy type
               k = dsqrt(5.0d0)
            end if
         end if

         if (k /= 1.0d0) then
            do Icon = 1, ncontract(Ibas)
               coeff(Icon, Ibas) = k*coeff(Icon, Ibas)
            end do
         end if
      end do
   end subroutine unify_cart_norm
end module quick_cuest_module
