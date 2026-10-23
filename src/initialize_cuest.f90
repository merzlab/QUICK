#ifdef CUEST
#include "util.fh"
!
!   initialize_cuest.f90
!
subroutine initialize_cuest
   !
   ! Initialize cuEST
   ! The following things should be already populated
   !   - quick_method
   !   - quick_basis and quick_fitbasis
   !   - quick_molspec, natom, xyz
   !
   use quick_cuest_module
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: quick_molspec, natom, xyz
   use quick_basis_module, only: nbasis
   use quick_calculated_module, only: quick_qm_struct
   use quick_basis_module, only: quick_basis, aexp, dcoeff, nshell, maxcontract
   use quick_files_module, only: ioutfile
   use quick_timer_module, only: timer_end, timer_cumer, timer_begin
   use quick_fitbasis_module, only: quick_fitbasis

   use, intrinsic::iso_c_binding, only: c_int64_t, c_int8_t, c_double, c_loc, c_bool

   ! init cuEST
   RECORD_TIME(timer_begin%TIniCuest)

   ! init cuest
   if (quick_method%UNRST) then
      call cuest_init( &
         int(natom, c_int64_t), &
         int(nshell, c_int64_t), &
         int(nbasis, c_int64_t), &
         int(quick_molspec%nelec, c_int64_t), &
         int(quick_molspec%nelecb, c_int64_t), &
         int(quick_fitbasis%nshell, c_int64_t), &
         int(maxcontract, c_int64_t), &
         int(quick_fitbasis%maxcontract, c_int64_t), &
         int(quick_molspec%iattype, c_int8_t), &
         c_loc(xyz), &
         quick_molspec%chg, &
         int(quick_molspec%nextatom, c_int64_t), &
         quick_molspec%extxyz, &
         quick_molspec%extchg &
      )
   else
      call cuest_init( &
         int(natom, c_int64_t), &
         int(nshell, c_int64_t), &
         int(nbasis, c_int64_t), &
         int(quick_molspec%nelec/2, c_int64_t), &
         int(0, c_int64_t), &
         int(quick_fitbasis%nshell, c_int64_t), &
         int(maxcontract, c_int64_t), &
         int(quick_fitbasis%maxcontract, c_int64_t), &
         int(quick_molspec%iattype, c_int8_t), &
         c_loc(xyz), &
         quick_molspec%chg, &
         int(quick_molspec%nextatom, c_int64_t), &
         quick_molspec%extxyz, &
         quick_molspec%extchg &
      )
   endif
   ! for testing; the following cuest functions should not be called here
   ! init primary (cartesian) basis
   call cuest_init_basis( &
      int(quick_basis%ncenter, c_int64_t), &
      int(quick_basis%katom, c_int64_t), &
      int(quick_basis%ktype, c_int64_t), &
      int(quick_basis%kprim, c_int64_t), &
      aexp, &
      dcoeff, &
      logical(.false., c_bool) &
   )

   ! init density fitting basis
   call cuest_init_basis( &
      int(quick_basis%ncenter, c_int64_t), &
      int(quick_fitbasis%katom, c_int64_t), &
      int(quick_fitbasis%ktype, c_int64_t), &
      int(quick_fitbasis%kprim, c_int64_t), &
      quick_fitbasis%gcexpo, &
      quick_fitbasis%gccoeff, &
      logical(.true., c_bool) &
   )

   call cuest_init_correct

   ! init pair list
   ! TODO: is this the right cutoff?
   call cuest_init_pair_list(quick_method%coreIntegralCutoff)

   call cuest_init_oei_plan

   ! init 2 electron integral plan
   call cuest_init_df(quick_method%x_hybrid_coeff, quick_method%cuest_fnl_code)

   call cuest_init_eri_J
   if (quick_method%x_hybrid_coeff /= 0.0d0) call cuest_init_eri_K(int(2d9, c_int64_t))

   RECORD_TIME(timer_end%TIniCuest)
   timer_cumer%TIniCuest = timer_cumer%TIniCuest + timer_end%TIniCuest - timer_begin%TIniCuest &
                           - (timer_end%T2elb - timer_begin%T2elb)
end subroutine initialize_cuest

subroutine deinitialize_cuest_compute
   use quick_cuest_module
   use quick_method_module, only: quick_method

   call cuest_deinit_oei_plan
   call cuest_deinit_eri_J
   if (quick_method%x_hybrid_coeff /= 0.0d0) call cuest_deinit_eri_K
   call cuest_deinit_df

   if (quick_method%DFT) then
      if (quick_method%UNRST) then
         call cuest_deinit_oshell_xc
      else
         call cuest_deinit_xc_dense
         call cuest_deinit_cshell_xc
      endif
   endif
end subroutine deinitialize_cuest_compute

subroutine finalize_cuest
   use quick_cuest_module
   call cuest_deinit_correct
   call cuest_deinit
end subroutine finalize_cuest

#endif ! #ifdef CUEST
