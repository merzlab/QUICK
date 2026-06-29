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
      subroutine cuest_deinit_oei_plan() bind(c, name="cuest_deinit_oei_plan")
      end subroutine
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
      subroutine cuest_init_oei_plan() bind(c, name="cuest_init_oei_plan")
      end subroutine cuest_init_oei_plan
   end interface

   interface
      subroutine cuest_init_pair_list(cutoff) bind(c, name="cuest_init_pair_list")
         use, intrinsic::iso_c_binding, only: c_double
         implicit none
         real(c_double), intent(in), value :: cutoff
      end subroutine cuest_init_pair_list
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

   subroutine correct_sym_o(m)
      !
      ! Corrects matrix m by fixing the order of d and f orbitals and their normalization
      !     d_xy  type has extra 1/sqrt(3)
      !     f_xxy type has extra 1/sqrt(5)
      !     f_xyz type has extra 1/sqrt(15)
      !
      ! QUICK
      !        1   2   3   4   5   6
      !     d: xx  xy  yy  xz  yz  zz
      !     f: xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz
      !
      ! cuEST
      !        1   2   3   4   5   6   7   8   9   10
      !     d: xx  xy  xz  yy  yz  zz
      !     f: xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
      !
      use quick_basis_module, only: itype, ncontract, nbasis
      implicit none

      double precision, intent(inout) :: m(:, :)
      ! counters
      integer :: Ibas, Jbas, Icon, Itmp
      ! itype stuff
      integer :: l1, l2, l3, lsum, lmax
      double precision :: sqrt3 = dsqrt(3.0d0)
      double precision :: sqrt5 = dsqrt(5.0d0)
      double precision :: sqrt15 = dsqrt(15.0d0)

      integer :: firstdf(nbasis)
      integer :: ifdf = 0

      double precision :: swaptmp
      double precision :: swaptmp_arr(nbasis)

#define SLICE_SWAP(i, j)               \
      swaptmp_arr = m(:, Ibas + i); \
      m(:, Ibas + i) = m(:, Ibas + j); \
      m(:, Ibas + j) = swaptmp_arr

      ! first swap columns
      Ibas = 1
      do while (Ibas <= nbasis)
         l1 = itype(1, Ibas)
         l2 = itype(2, Ibas)
         l3 = itype(3, Ibas)
         lsum = l1 + l2 + l3

         if (lsum == 2) then ! d orbital
            ! swaptmp_arr = m(:, Ibas + 2)
            ! m(:, Ibas + 2) = m(:, Ibas + 3)
            ! m(:, Ibas + 3) = swaptmp_arr
            SLICE_SWAP(2, 3)
            m(:, Ibas + 1) = m(:, Ibas + 1)*sqrt3
            m(:, Ibas + 3) = m(:, Ibas + 3)*sqrt3
            m(:, Ibas + 4) = m(:, Ibas + 4)*sqrt3
            Ibas = Ibas + 6
         else if (lsum == 3) then
            SLICE_SWAP(2, 4)
            SLICE_SWAP(3, 4)
            SLICE_SWAP(4, 5)
            SLICE_SWAP(5, 7)
            SLICE_SWAP(6, 7)
            m(:, Ibas + 1) = m(:, Ibas + 1)*sqrt5
            m(:, Ibas + 2) = m(:, Ibas + 2)*sqrt5
            m(:, Ibas + 4) = m(:, Ibas + 4)*sqrt5
            m(:, Ibas + 5) = m(:, Ibas + 5)*sqrt15
            m(:, Ibas + 6) = m(:, Ibas + 6)*sqrt5
            m(:, Ibas + 7) = m(:, Ibas + 7)*sqrt5
            m(:, Ibas + 8) = m(:, Ibas + 8)*sqrt5
            Ibas = Ibas + 10
         else
            Ibas = Ibas + 1
         end if
      end do

#define SCALAR_SWAP(i, j) \
      swaptmp = m(Jbas + i, Ibas); \
      m(Jbas + i, Ibas) = m(Jbas + j, Ibas); \
      m(Jbas + j, Ibas) = swaptmp

      do Ibas = 1, nbasis
         Jbas = 1
         do while (Jbas <= nbasis)
            l1 = itype(1, Ibas)
            l2 = itype(2, Ibas)
            l3 = itype(3, Ibas)
            lsum = l1 + l2 + l3

            if (lsum == 2) then
               SCALAR_SWAP(2, 3)
               m(Jbas + 1, Ibas) = m(Jbas + 1, Ibas)*sqrt3
               m(Jbas + 3, Ibas) = m(Jbas + 3, Ibas)*sqrt3
               m(Jbas + 4, Ibas) = m(Jbas + 4, Ibas)*sqrt3
               Jbas = Jbas + 6
            else if (lsum == 3) then
               SCALAR_SWAP(2, 4)
               SCALAR_SWAP(3, 4)
               SCALAR_SWAP(4, 5)
               SCALAR_SWAP(5, 7)
               SCALAR_SWAP(6, 7)
               m(Jbas + 1, Ibas) = m(Jbas + 1, Ibas)*sqrt5
               m(Jbas + 2, Ibas) = m(Jbas + 2, Ibas)*sqrt5
               m(Jbas + 4, Ibas) = m(Jbas + 4, Ibas)*sqrt5
               m(Jbas + 5, Ibas) = m(Jbas + 5, Ibas)*sqrt15
               m(Jbas + 6, Ibas) = m(Jbas + 6, Ibas)*sqrt5
               m(Jbas + 7, Ibas) = m(Jbas + 7, Ibas)*sqrt5
               m(Jbas + 8, Ibas) = m(Jbas + 8, Ibas)*sqrt5
               Jbas = Jbas + 10
            else
               Jbas = Jbas + 1
            end if
         end do
      end do
   end subroutine correct_sym_o
#undef SCALAR_SWAP
#undef SLICE_SWAP
end module quick_cuest_module
