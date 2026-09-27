#include "util.fh"
!
! mfcc_fragment_scf.f90
!
! Computes the per-fragment densities that the MFCC initial guess is
! assembled from. mfcc() (mfcc_start.f90) has already produced the fragment,
! cap and connection geometries; this routine runs a small closed-shell HF
! calculation on each of them and stores the converged density blocks in
! mfccdens*, which MFCC_initial_guess then sums into the global density.
!
! The pattern follows getmolsad (quick_sad_guess_module.f90): save the global
! molecule and method, substitute the sub-molecule, build its basis, converge
! it, harvest the density, then restore. Like the SAD guess this runs before
! getMol, so it must not depend on anything getMol populates.
!
! readbasis returns the fragment-local basis range for a given atom range,
! which is exactly mfccbases/mfccbasef; no separate bookkeeping is needed.
!

subroutine mfcc_fragment_scf(ierr)
   use allmod
   use quick_exception_module
   use quick_mpi_module, only: bMPI, master
   implicit none

   integer, intent(inout) :: ierr

   integer :: k, i, nat, maxbas, ncon, nb_frag
   integer :: natomsaved, nbs, nbf, nloc
   logical :: MPIsaved
   double precision, allocatable :: xyzsaved(:,:)
   type(quick_method_type) :: quick_method_save
   type(quick_molspec_type) :: quick_molspec_save

   if (npmfcc .le. 0) return

   ! ---------------------------------------------------------------
   ! Save the global state. getmolsad relies on getMol rebuilding the
   ! molecule afterwards; we are in the same position in the run, so the
   ! same assumption holds.
   ! ---------------------------------------------------------------
   quick_method_save = quick_method
   quick_molspec_save = quick_molspec
   natomsaved = natom
   MPIsaved = bMPI
   allocate(xyzsaved(3,natom))
   xyzsaved = xyz(1:3,1:natom)

   ! Fragments are ordinary closed-shell HF molecules. Turn off everything
   ! that does not apply to them, in particular divide and conquer, or the
   ! fragment SCF would recurse into the method we are producing a guess for.
   bMPI = .false.
   quick_method%HF = .true.
   quick_method%DFT = .false.
   quick_method%UNRST = .false.
   quick_method%divcon = .false.
   quick_method%dcmp2only = .false.
   quick_method%MP2 = .false.
   quick_method%opt = .false.
   quick_method%grad = .false.
   quick_method%ZMAT = .false.
   quick_method%nodirect = .false.
   quick_molspec%imult = 1

   if (master) call PrtAct(ioutfile,"Begin MFCC fragment densities")

   ! ---------------------------------------------------------------
   ! Pass 1: build each fragment basis to learn how large the density
   ! blocks must be. readbasis is cheap next to the SCF, so paying for it
   ! twice is preferable to guessing a bound or over-allocating.
   ! ---------------------------------------------------------------
   maxbas = 0
   do k = 1, npmfcc
      call deallocate_calculated
      call mfcc_set_submol(mfccatom(k),mfcccord(1,1,k),mfccatomxiao(1,k),ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstart(k),mfccfinal(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)
   enddo

   do k = 1, npmfcc-1
      call deallocate_calculated
      call mfcc_set_submol(mfccatomcap(k),mfcccordcap(1,1,k),mfccatomxiaocap(1,k),ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstartcap(k),mfccfinalcap(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)
   enddo

   ncon = max(kxiaoconnect,1)
   call allocate_MFCC(npmfcc,ncon,maxbas)

   if (master) write(ioutfile,'(" MFCC fragments =",i4,"  caps =",i4, &
         &"  connections =",i4,"  max basis =",i5)') npmfcc,npmfcc-1,kxiaoconnect,maxbas

   ! ---------------------------------------------------------------
   ! Pass 2: converge each sub-molecule and keep its density.
   ! ---------------------------------------------------------------
   do k = 1, npmfcc
      call mfcc_run_submol(mfccatom(k),mfcccord(1,1,k),mfccatomxiao(1,k), &
            mfccstart(k),mfccfinal(k),mfccbases(k),mfccbasef(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      ! MFCC_initial_guess reads mfccdens(k,i-mfccbases+1,...) with i starting
      ! at mfccbases, so block index 1 must be the fragment's first *real*
      ! basis function, not its first basis function. Storing from 1 would
      ! include the leading cap hydrogen and shift everything by one.
      nloc = mfccbasef(k)-mfccbases(k)+1
      mfccdens(k,1:nloc,1:nloc) = &
            quick_qm_struct%dense(mfccbases(k):mfccbasef(k),mfccbases(k):mfccbasef(k))
      if (master) write(ioutfile,'("   fragment ",i4," basis ",i5," local range ",i5," -",i5)') &
            k,nb_frag,mfccbases(k),mfccbasef(k)
   enddo

   do k = 1, npmfcc-1
      call mfcc_run_submol(mfccatomcap(k),mfcccordcap(1,1,k),mfccatomxiaocap(1,k), &
            mfccstartcap(k),mfccfinalcap(k),mfccbasescap(k),mfccbasefcap(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      nloc = mfccbasefcap(k)-mfccbasescap(k)+1
      mfccdenscap(k,1:nloc,1:nloc) = &
            quick_qm_struct%dense(mfccbasescap(k):mfccbasefcap(k),mfccbasescap(k):mfccbasefcap(k))
      if (master) write(ioutfile,'("   cap      ",i4," basis ",i5," local range ",i5," -",i5)') &
            k,nb_frag,mfccbasescap(k),mfccbasefcap(k)
   enddo

   if (master) call PrtAct(ioutfile,"Finish MFCC fragment densities")

900 continue

   ! ---------------------------------------------------------------
   ! Restore the global state.
   !
   ! The basis arrays are left sized for the last fragment, so they must be
   ! released here as well: getMol runs readbasis again for the whole
   ! molecule, and reusing a fragment-sized allocation overflows it.
   ! ---------------------------------------------------------------
   call deallocate_calculated
   call dealloc(quick_qm_struct)

   ! Order matters and follows getmolsad: natom and xyz first, then the derived
   ! types. quick_molspec%natom is a pointer to the natom target, so the
   ! derived-type assignment must not be the thing that defines it.
   natom = natomsaved
   xyz(1:3,1:natom) = xyzsaved
   quick_method = quick_method_save
   quick_molspec = quick_molspec_save
   bMPI = MPIsaved
   deallocate(xyzsaved)

end subroutine mfcc_fragment_scf


!-------------------------------------------------------
! mfcc_set_submol
!-------------------------------------------------------
! Install a fragment as the current molecule: atom count, coordinates and
! atom types. mfcc_start stores coordinates in Angstrom, QUICK works in bohr.
!-------------------------------------------------------

subroutine mfcc_set_submol(nat,cord,sym,ierr)
   use allmod
   implicit none

   integer, intent(in) :: nat
   double precision, intent(in) :: cord(3,*)
   character(len=2), intent(in) :: sym(*)
   integer, intent(inout) :: ierr

   integer :: i, iz

   if (nat .le. 0) then
      call PrtErr(iOutFile,'MFCC fragment has no atoms.')
      ierr = 44
      return
   endif

   natom = nat
   quick_molspec%natom = nat
   quick_molspec%nelec = 0

   do i = 1, nat
      call mfcc_symbol_to_z(sym(i),iz,ierr)
      if (ierr /= 0) return
      quick_molspec%iattype(i) = iz
      quick_molspec%chg(i) = dble(iz)
      quick_molspec%nelec = quick_molspec%nelec + iz
      xyz(1,i) = cord(1,i)/BOHR
      xyz(2,i) = cord(2,i)/BOHR
      xyz(3,i) = cord(3,i)/BOHR
   enddo

   ! Fragments and caps are built neutral and closed shell.
   if (mod(quick_molspec%nelec,2) .ne. 0) then
      call PrtErr(iOutFile,'MFCC fragment has an odd number of electrons; &
            &only closed shell fragments are supported.')
      ierr = 44
      return
   endif
   quick_molspec%nelecb = quick_molspec%nelec/2
   quick_molspec%imult = 1

end subroutine mfcc_set_submol


!-------------------------------------------------------
! mfcc_run_submol
!-------------------------------------------------------
! Build the basis for one sub-molecule, converge it, and report its basis
! size and the local basis range spanned by its real (non-cap) atoms.
!-------------------------------------------------------

subroutine mfcc_run_submol(nat,cord,sym,iatstart,iatfinal,ibasstart,ibasfinal,nb,ierr)
   use allmod
   use quick_exception_module
   implicit none

   integer, intent(in) :: nat, iatstart, iatfinal
   double precision, intent(in) :: cord(3,*)
   character(len=2), intent(in) :: sym(*)
   integer, intent(out) :: ibasstart, ibasfinal, nb
   integer, intent(inout) :: ierr

   integer :: i
   double precision :: diagelement

   ! Fragments differ in size, so the previous fragment's basis arrays must be
   ! released before readbasis rebuilds them. getmolsad gets away without this
   ! because every one of its sub-molecules is a single atom.
   call deallocate_calculated
   call dealloc(quick_qm_struct)

   call mfcc_set_submol(nat,cord,sym,ierr)
   if (ierr /= 0) return

   ! readbasis returns the basis range spanned by atoms iatstart..iatfinal,
   ! which is what MFCC_initial_guess needs as mfccbases/mfccbasef.
   call readbasis(natom,iatstart,iatfinal,ibasstart,ibasfinal,ierr)
   if (ierr /= 0) return

   nb = nbasis
   if (nbasis .lt. 1) then
      call PrtErr(iOutFile,'No basis functions found for an MFCC fragment.')
      ierr = 44
      return
   endif

   quick_qm_struct%nbasis => nbasis
   call alloc(quick_qm_struct)
   call init(quick_qm_struct)
   call normalize_basis()

   ! Crude diagonal starting density, as the SAD guess does for atoms.
   diagelement = dble(quick_molspec%nelec)/dble(nbasis)
   do i = 1, nbasis
      quick_qm_struct%dense(i,i) = diagelement
   enddo

   ! getEnergy with isGuess=.true. builds X and the nuclear repulsion and
   ! runs the SCF, while skipping the DFT grid and the verbose banners.
   call getEnergy(.true.,ierr)

end subroutine mfcc_run_submol


!-------------------------------------------------------
! mfcc_symbol_to_z
!-------------------------------------------------------
! Map an element symbol onto its atomic number using the table in
! quick_constants_module.
!-------------------------------------------------------

subroutine mfcc_symbol_to_z(sym,iz,ierr)
   use allmod
   implicit none

   character(len=2), intent(in) :: sym
   integer, intent(out) :: iz
   integer, intent(inout) :: ierr

   integer :: i
   character(len=2) :: want

   want = adjustl(sym)
   iz = 0
   do i = 1, SYMBOL_MAX
      if (symbol(i) .eq. want) then
         iz = i
         return
      endif
   enddo

   call PrtErr(iOutFile,'Unrecognised element symbol in an MFCC fragment.')
   ierr = 44

end subroutine mfcc_symbol_to_z
