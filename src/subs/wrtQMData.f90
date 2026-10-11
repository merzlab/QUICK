#include "util.fh"
!
!	wrtQMData.f90
!	new_quick
!
!-----------------------------------------------------------
! wrtQMData
!-----------------------------------------------------------
! Writes the converged SCF data needed to reconstruct the calculation
! in an external package (e.g. PySCF): geometry/charge/spin, AO basis
! identity, MO coefficients/energies/occupations, overlap, core
! Hamiltonian, and the converged density. Controlled by the
! QMDATA_WRITE keyword, independent of CHK_WRITE (which remains the
! original density/geometry restart mechanism) and written to its own
! file via qmdata_init/qmdata_write/qmdata_close. Intended to be called
! once, after SCF has converged (not per iteration).
!
! Two cases, distinguished by the exported 'has_global_mo' flag:
!
! (a) Regular (non-DnC) SCF: has_global_mo=1. Global canonical MOs
!     (co/E) exist and are exported; F is NOT exported.
!     quick_qm_struct%o is QUICK's internal, DIIS-history-extrapolated
!     operator matrix, used for convergence acceleration. On the final
!     (post-convergence) iteration it does not exactly satisfy the
!     eigenvalue equation with the exported co/E (confirmed:
!     max|o @ co - s @ co @ diag(E)| ~ 1.9, vs ~1e-14 after
!     reconstructing F = S @ co @ diag(E) @ co^T @ S from the exported
!     s/co/e instead). Any consumer that needs F should reconstruct it
!     that way -- exactly how orca_msgpack_to_pyscf_mf.py already
!     treats ORCA's own Fock -- rather than trust a raw export.
!
! (b) Divide-and-conquer (DIVCON/DCMP2ONLY): has_global_mo=0. QUICK
!     never forms global canonical MOs in this mode -- co/E/occ are
!     exported as zero-filled placeholders, not meaningful, DO NOT USE.
!     F has no S/co/E to be reconstructed from, so it IS exported
!     directly here instead: unlike case (a), DnC's quick_qm_struct%o
!     (built fresh from the final density each cycle, no "extra final
!     iteration" lag -- confirmed: Tr(D@H) + 0.5*Tr(D@(o-H)) agrees with
!     QUICK's own reported electronic energy to ~5e-5 Ha, RHF/STO-3G
!     hexaglycine divcon/atombasis, denserms=1e-6, i.e. convergence-
!     tolerance noise, not a DIIS-lag-sized error) is self-consistent
!     with the exported density and safe to use as-is.
subroutine wrtQMData
   use allmod
   use quick_io_module, only: qmdata_init, qmdata_write, qmdata_close
   implicit none

   integer :: i
   logical :: is_dnc
   double precision, allocatable :: occ(:), occb(:)
   double precision, allocatable :: veff(:,:), veffb(:,:)
   double precision :: etot_arr(1)
   integer :: scalar_arr(1)
   integer :: basisname_codes(80)
   integer :: neleca, nelecb_local
   double precision :: occval

   is_dnc = quick_method%divcon .or. quick_method%dcmp2only

   call qmdata_init(natom, nbasis)

   ! Geometry, atom identity, charge and spin multiplicity
   call qmdata_write('xyz', 3, natom, quick_molspec%xyz)
   call qmdata_write('iattype', natom, quick_molspec%iattype)
   scalar_arr(1) = quick_molspec%molchg
   call qmdata_write('molchg', 1, scalar_arr)
   scalar_arr(1) = quick_molspec%imult
   call qmdata_write('imult', 1, scalar_arr)

   ! Basis set name, encoded as character codes (fixed width 80)
   do i = 1, 80
      basisname_codes(i) = ichar(basisSetName(i:i))
   enddo
   call qmdata_write('basisname_codes', 80, basisname_codes)

   ! Total converged energy
   etot_arr(1) = quick_qm_struct%Etot
   call qmdata_write('etot', 1, etot_arr)

   ! Number of basis functions actually used (<= nbasis when near-linear
   ! dependencies were removed): the true second dimension of co/cob and
   ! length of e/eb/occ/occb below.
   scalar_arr(1) = NBSuse
   call qmdata_write('nbsuse', 1, scalar_arr)

   ! Whether global canonical MOs exist at all -- see the module-level
   ! comment above for cases (a)/(b). Any consumer should check this
   ! before trusting co/E/occ (zero-filled placeholders in case (b)).
   scalar_arr(1) = 0
   if (.not. is_dnc) scalar_arr(1) = 1
   call qmdata_write('has_global_mo', 1, scalar_arr)

   ! Converged SCF density (alpha, or RHF/RKS)
   call qmdata_write('dense', nbasis, nbasis, quick_qm_struct%dense)

   ! MO coefficients (alpha, or RHF/RKS): nbasis x NBSuse, NOT nbasis x nbasis.
   ! Zero-filled placeholders in DnC mode (case (b) above) -- not meaningful.
   call qmdata_write('co', nbasis, NBSuse, quick_qm_struct%co)

   ! Overlap and core (one-electron) Hamiltonian, shared between alpha/beta
   call qmdata_write('s', nbasis, nbasis, quick_qm_struct%s)
   call qmdata_write('h', nbasis, nbasis, quick_qm_struct%oneElecO)

   ! Fock matrix and v_eff = F - H: only in DnC mode (case (b) above),
   ! where there is no co/E to reconstruct F from. See the module-level
   ! comment for why DnC's quick_qm_struct%o is trustworthy here while
   ! regular SCF's is not.
   if (is_dnc) then
      allocate(veff(nbasis,nbasis))
      veff = quick_qm_struct%o - quick_qm_struct%oneElecO
      call qmdata_write('f', nbasis, nbasis, quick_qm_struct%o)
      call qmdata_write('veff', nbasis, nbasis, veff)
      deallocate(veff)
   endif

   ! Orbital energies (alpha): length NBSuse, NOT nbasis. Zero-filled
   ! placeholders in DnC mode -- not meaningful.
   call qmdata_write('e', NBSuse, quick_qm_struct%E)

   ! Occupation numbers (alpha), length NBSuse
   allocate(occ(NBSuse))
   occ = 0.0d0
   if (.not. quick_method%unrst) then
      neleca = quick_molspec%nelec/2
      occval = 2.0d0
   else
      neleca = quick_molspec%nelec
      occval = 1.0d0
   endif
   do i = 1, NBSuse
      if (neleca .gt. 0) then
         occ(i) = occval
         neleca = neleca - 1
      endif
   enddo
   call qmdata_write('occ', NBSuse, occ)
   deallocate(occ)

   if (quick_method%unrst) then
      call qmdata_write('denseb', nbasis, nbasis, quick_qm_struct%denseb)
      call qmdata_write('cob', nbasis, NBSuse, quick_qm_struct%cob)

      if (is_dnc) then
         allocate(veffb(nbasis,nbasis))
         veffb = quick_qm_struct%ob - quick_qm_struct%oneElecO
         call qmdata_write('fb', nbasis, nbasis, quick_qm_struct%ob)
         call qmdata_write('veffb', nbasis, nbasis, veffb)
         deallocate(veffb)
      endif

      call qmdata_write('eb', NBSuse, quick_qm_struct%Eb)

      allocate(occb(NBSuse))
      occb = 0.0d0
      nelecb_local = quick_molspec%nelecb
      do i = 1, NBSuse
         if (nelecb_local .gt. 0) then
            occb(i) = 1.0d0
            nelecb_local = nelecb_local - 1
         endif
      enddo
      call qmdata_write('occb', NBSuse, occb)
      deallocate(occb)
   endif

   ! AO basis identity: atom center and Cartesian angular-momentum
   ! components (lx,ly,lz) per AO, e.g. (1,0,0) = px, (0,0,0) = s.
   ! Together with the primitive data below, this is enough to
   ! group AOs into shells and match QUICK's AO order to another
   ! package's convention without relying on a basis-set name lookup.
   call qmdata_write('ao_atom', nbasis, quick_basis%ncenter)
   call qmdata_write('ao_lx', nbasis, itype(1,1:nbasis))
   call qmdata_write('ao_ly', nbasis, itype(2,1:nbasis))
   call qmdata_write('ao_lz', nbasis, itype(3,1:nbasis))

   ! Per-AO contracted Gaussian primitives (exponents, coefficients) and
   ! normalization constant, padded to maxcontract primitives per AO
   ! ('ao_ncontract' gives the number of primitives actually used).
   scalar_arr(1) = maxcontract
   call qmdata_write('maxcontract', 1, scalar_arr)
   call qmdata_write('ao_ncontract', nbasis, ncontract(1:nbasis))
   call qmdata_write('ao_cons', nbasis, quick_basis%cons)
   call qmdata_write('ao_prim_exp', maxcontract, nbasis, aexp)
   call qmdata_write('ao_prim_coeff', maxcontract, nbasis, dcoeff)

   call qmdata_close()

   return
end subroutine wrtQMData
