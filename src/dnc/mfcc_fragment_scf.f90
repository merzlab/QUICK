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
#ifdef MPIV
   use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_comm_size, quick_mpi_error
   use mpi
#endif
   implicit none

   integer, intent(inout) :: ierr

   integer :: k, i, nat, maxbas, ncon, nb_frag
   integer :: natomsaved, nbs, nbf, nloc
   integer :: nbi, nbj, nati, natj, icbi, icbf, jcbi, jcbf
   logical :: mfcc_con_ok
   ! A combined connection block holds an I part and a J part, so it is bounded
   ! by twice whatever mfcc_start sized the per block arrays to. That is no
   ! longer a fixed 100, so this cannot be a fixed 200 either.
   integer :: MFCC_MAXAT
   double precision, allocatable :: concord(:,:)
   character(len=2), allocatable :: consym(:)
   logical :: MPIsaved, mastersaved, real_master
   integer :: myrank, nranks
   integer :: commsaved, ranksaved, sizesaved
   integer :: nd1, nd2, nd3, nc1, nc2, nc3
   double precision, allocatable :: xyzsaved(:,:)
   type(quick_method_type) :: quick_method_save
   type(quick_molspec_type) :: quick_molspec_save

   if (npmfcc .le. 0) return

   MFCC_MAXAT = 2*size(mfcccord,2)
   allocate(concord(3,MFCC_MAXAT), consym(MFCC_MAXAT), stat=ierr)
   if (ierr /= 0) then
      call PrtErr(iOutFile,'Could not allocate the MFCC connection scratch arrays.')
      ierr = 34
      return
   endif
   ierr = 0

   ! ---------------------------------------------------------------
   ! Save the global state. getmolsad relies on getMol rebuilding the
   ! molecule afterwards; we are in the same position in the run, so the
   ! same assumption holds.
   ! ---------------------------------------------------------------
   quick_method_save = quick_method
   quick_molspec_save = quick_molspec
   natomsaved = natom
   MPIsaved = bMPI
   mastersaved = master
   real_master = master
   myrank = 0
   nranks = 1
   allocate(xyzsaved(3,natom))
   xyzsaved = xyz(1:3,1:natom)

   ! Fragments are ordinary closed-shell HF molecules. Turn off everything
   ! that does not apply to them, in particular divide and conquer, or the
   ! fragment SCF would recurse into the method we are producing a guess for.
   bMPI = .false.
#ifdef MPIV
   ! Each rank solves whole sub-molecules by itself, which takes more than
   ! clearing bMPI.
   !
   ! First, electdiis sets diisdone only inside if(master), and with bMPI off
   ! the broadcast that would carry it to the others never happens, so a
   ! non-master rank spins in the SCF loop for ever. Forcing master true makes
   ! each rank self-contained.
   !
   ! Second, and less obviously, the MPI reductions in scf_operator sit inside
   ! a bare #ifdef MPIV with no bMPI test, so every Fock build unconditionally
   ! sums quick_qm_struct%o across quick_comm. With each rank holding a
   ! complete and different sub-molecule that sum is meaningless: the first
   ! fragment started at -729 instead of -235 and never converged. Pointing
   ! quick_comm at MPI_COMM_SELF turns those reductions into local no-ops
   ! without touching the shared operator, which every other calculation uses.
   if (MPIsaved) then
      myrank = quick_comm_rank
      nranks = quick_comm_size
      commsaved = quick_comm
      ranksaved = quick_comm_rank
      sizesaved = quick_comm_size
      quick_comm = MPI_COMM_SELF
      quick_comm_rank = 0
      quick_comm_size = 1
      master = .true.
   endif
#endif
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

   if (real_master) call PrtAct(ioutfile,"Begin MFCC fragment densities")

   ! ---------------------------------------------------------------
   ! Pass 1: build each fragment basis to learn how large the density
   ! blocks must be. readbasis is cheap next to the SCF, so paying for it
   ! twice is preferable to guessing a bound or over-allocating.
   ! ---------------------------------------------------------------
   maxbas = 0
   do k = 1, npmfcc+nmfccextra
      call deallocate_calculated
      call mfcc_set_submol(mfccatom(k),mfcccord(1,1,k),mfccatomxiao(1,k),mfcccharge(k),ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstart(k),mfccfinal(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)
   enddo

   do k = 1, npmfcc-1
      call deallocate_calculated
      call mfcc_set_submol(mfccatomcap(k),mfcccordcap(1,1,k),mfccatomxiaocap(1,k),mfccchargecap(k),ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstartcap(k),mfccfinalcap(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)
   enddo

   ! Connection blocks are sized here too: they can be larger than any fragment
   ! or cap, and the combined I+J block needs room for both.
   !
   ! First validate the geometry mfcc_start produced. The i==2 branch of the
   ! connection construction hardcodes mm=9, an atom index from some other
   ! system, so for a contact involving residue 2 it yields a negative atom
   ! count. That code has never run before now. If any block is malformed the
   ! whole connection layer is disabled rather than partially applied, since
   ! MFCC_initial_guess consumes blocks 1..nconuse contiguously.
   mfcc_con_ok = (kxiaoconnect .gt. 0)
   do k = 1, kxiaoconnect
      if (mfccatomconi(k) .le. 0 .or. mfccatomconj(k) .le. 0 .or. &
          mfccatomcon(k)  .le. 0 .or. mfccatomcon2(k) .le. 0) then
         mfcc_con_ok = .false.
         if (real_master) write(ioutfile,'(" MFCC connection block ",i4," has an invalid atom count (", &
               &4(i6))') k,mfccatomconi(k),mfccatomconj(k),mfccatomcon(k),mfccatomcon2(k)
      endif
   enddo

   if (kxiaoconnect .gt. 0 .and. .not.mfcc_con_ok) then
      call PrtWrn(iOutFile,'MFCC connection geometry is malformed; connection terms are disabled.')
      write(ioutfile,'("|          The connection construction in mfcc_start.f90 produced an")')
      write(ioutfile,'("|          invalid atom count for at least one block. Its i==2 branch")')
      write(ioutfile,'("|          hardcodes an atom index (mm=9) and cannot be correct in")')
      write(ioutfile,'("|          general. The guess falls back to the two term formula.")')
      write(ioutfile,'(a)')
      call flush(ioutfile)
      kxiaoconnect = 0
   endif

   do k = 1, kxiaoconnect
      call deallocate_calculated
      call mfcc_set_submol(mfccatomconi(k),mfcccordconi(1,1,k),mfccatomxiaoconi(1,k),0,ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstartconi(k),mfccfinalconi(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)

      call deallocate_calculated
      call mfcc_set_submol(mfccatomconj(k),mfcccordconj(1,1,k),mfccatomxiaoconj(1,k),0,ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,mfccstartconj(k),mfccfinalconj(k),nbs,nbf,ierr)
      if (ierr /= 0) goto 900
      maxbas = max(maxbas,nbasis)
   enddo

   ncon = max(kxiaoconnect,1)
   ! The fragment slots run 1..npmfcc+nmfccextra: the peptide fragments first,
   ! then one standalone fragment per solvent molecule, ion or ligand. The cap
   ! slots only run 1..npmfcc-1, but both density arrays are allocated with the
   ! same first dimension, so size them for the larger of the two. Sized at
   ! npmfcc, every standalone fragment wrote past the end of mfccdens: silent
   ! corruption with a hundred waters, a segfault with four hundred.
   call allocate_MFCC(npmfcc+nmfccextra,ncon,maxbas)

   if (real_master) write(ioutfile,'(" MFCC fragments =",i4,"  standalone =",i6,"  caps =",i4, &
         &"  connections =",i4,"  max basis =",i5)') npmfcc,nmfccextra,npmfcc-1,kxiaoconnect,maxbas

   ! ---------------------------------------------------------------
   ! Pass 2: converge each sub-molecule and keep its density.
   ! ---------------------------------------------------------------
   ! Sub-molecules are independent, so each rank solves only its own share.
   ! Dealing them out round robin rather than in contiguous blocks keeps the
   ! load even without sorting: consecutive fragments are similar in size, so
   ! strided assignment spreads the large ones across ranks.
   !
   ! Everything was allocated zeroed, and each rank writes only its own blocks,
   ! so a single sum over ranks at the end reconstructs the full set. That is
   ! why no packing or variable length gather is needed.
   do k = 1, npmfcc+nmfccextra
      if (mod(k-1,nranks) .ne. myrank) cycle
      call mfcc_run_submol(mfccatom(k),mfcccord(1,1,k),mfccatomxiao(1,k),mfcccharge(k), &
            mfccstart(k),mfccfinal(k),mfccbases(k),mfccbasef(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      ! MFCC_initial_guess reads mfccdens(k,i-mfccbases+1,...) with i starting
      ! at mfccbases, so block index 1 must be the fragment's first *real*
      ! basis function, not its first basis function. Storing from 1 would
      ! include the leading cap hydrogen and shift everything by one.
      nloc = mfccbasef(k)-mfccbases(k)+1
      mfccdens(k,1:nloc,1:nloc) = &
            quick_qm_struct%dense(mfccbases(k):mfccbasef(k),mfccbases(k):mfccbasef(k))
      if (real_master) write(ioutfile,'("   fragment ",i4," basis ",i5," local range ",i5," -",i5)') &
            k,nb_frag,mfccbases(k),mfccbasef(k)
   enddo

   do k = 1, npmfcc-1
      if (mod(k-1,nranks) .ne. myrank) cycle
      call mfcc_run_submol(mfccatomcap(k),mfcccordcap(1,1,k),mfccatomxiaocap(1,k),mfccchargecap(k), &
            mfccstartcap(k),mfccfinalcap(k),mfccbasescap(k),mfccbasefcap(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      nloc = mfccbasefcap(k)-mfccbasescap(k)+1
      mfccdenscap(k,1:nloc,1:nloc) = &
            quick_qm_struct%dense(mfccbasescap(k):mfccbasefcap(k),mfccbasescap(k):mfccbasefcap(k))
      if (real_master) write(ioutfile,'("   cap      ",i4," basis ",i5," local range ",i5," -",i5)') &
            k,nb_frag,mfccbasescap(k),mfccbasefcap(k)
   enddo

   ! ---------------------------------------------------------------
   ! Pass 3: connection blocks. For each non-sequential contact the guess needs
   ! a two-body correction, which MFCC_initial_guess assembles as
   !     D += D(I union J)|I + D(I union J)|J - D(I) - D(J)
   ! so three SCFs per connection: each fragment alone, then the two together.
   ! Only the diagonal sub-blocks of the combined density are used.
   ! ---------------------------------------------------------------
   do k = 1, kxiaoconnect
      if (mod(k-1,nranks) .ne. myrank) cycle

      ! the I fragment on its own
      call mfcc_run_submol(mfccatomconi(k),mfcccordconi(1,1,k),mfccatomxiaoconi(1,k),0, &
            mfccstartconi(k),mfccfinalconi(k),mfccbasesconi(k),mfccbasefconi(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      nbi = mfccbasefconi(k)-mfccbasesconi(k)+1
      mfccdensconi(k,1:nbi,1:nbi) = &
            quick_qm_struct%dense(mfccbasesconi(k):mfccbasefconi(k),mfccbasesconi(k):mfccbasefconi(k))

      ! the J fragment on its own
      call mfcc_run_submol(mfccatomconj(k),mfcccordconj(1,1,k),mfccatomxiaoconj(1,k),0, &
            mfccstartconj(k),mfccfinalconj(k),mfccbasesconj(k),mfccbasefconj(k),nb_frag,ierr)
      if (ierr /= 0) goto 900
      nbj = mfccbasefconj(k)-mfccbasesconj(k)+1
      mfccdensconj(k,1:nbj,1:nbj) = &
            quick_qm_struct%dense(mfccbasesconj(k):mfccbasefconj(k),mfccbasesconj(k):mfccbasefconj(k))

      ! the two together, as one molecule: con holds the I atoms, con2 the J atoms
      nati = mfccatomcon(k)
      natj = mfccatomcon2(k)
      if (nati+natj .gt. MFCC_MAXAT) then
         call PrtErr(iOutFile,'MFCC combined connection block exceeds the per-fragment atom limit.')
         ierr = 44
         goto 900
      endif
      do i = 1, nati
         concord(1:3,i) = mfcccordcon(1:3,i,k)
         consym(i) = mfccatomxiaocon(i,k)
      enddo
      do i = 1, natj
         concord(1:3,nati+i) = mfcccordcon2(1:3,i,k)
         consym(nati+i) = mfccatomxiaocon2(i,k)
      enddo

      ! Two readbasis passes on the same combined molecule, because it returns
      ! one atom-to-basis range per call and both parts' ranges are needed.
      call mfcc_run_submol(nati+natj,concord,consym,0, &
            mfccstartcon(k),mfccfinalcon(k),icbi,icbf,nb_frag,ierr)
      if (ierr /= 0) goto 900
      call readbasis(natom,nati+mfccstartconj(k)-1,nati+mfccfinalconj(k)-1,jcbi,jcbf,ierr)
      if (ierr /= 0) goto 900

      ! Pack the two diagonal sub-blocks adjacently: MFCC_initial_guess reads the
      ! J part offset by the I block size.
      nbi = icbf-icbi+1
      nbj = jcbf-jcbi+1
      mfccdenscon(k,1:nbi,1:nbi) = quick_qm_struct%dense(icbi:icbf,icbi:icbf)
      mfccdenscon(k,nbi+1:nbi+nbj,nbi+1:nbi+nbj) = quick_qm_struct%dense(jcbi:jcbf,jcbi:jcbf)

      if (real_master) write(ioutfile,'("   connection ",i3," I basis ",i5," J basis ",i5)') k,nbi,nbj
   enddo

   if (real_master) call PrtAct(ioutfile,"Finish MFCC fragment densities")

900 continue

#ifdef MPIV
   ! Collect the blocks. quick_comm is still MPI_COMM_SELF at this point, so
   ! restore the real communicator before reducing. Density blocks sum because
   ! every array was zeroed on allocation and each rank filled only its own
   ! slices; the basis range indices take a max for the same reason, unset
   ! entries being zero and real ones positive.
   if (MPIsaved) then
      quick_comm = commsaved
      quick_comm_rank = ranksaved
      quick_comm_size = sizesaved

      nd1 = size(mfccdens,1); nd2 = size(mfccdens,2); nd3 = size(mfccdens,3)
      call mfcc_reduce_dens(mfccdens,    nd1,nd2,nd3)
      call mfcc_reduce_dens(mfccdenscap, nd1,nd2,nd3)
      call mfcc_reduce_idx(mfccbases,    npmfcc+nmfccextra)
      call mfcc_reduce_idx(mfccbasef,    npmfcc+nmfccextra)
      call mfcc_reduce_idx(mfccbasescap, npmfcc)
      call mfcc_reduce_idx(mfccbasefcap, npmfcc)
      if (kxiaoconnect .gt. 0) then
         nc1 = size(mfccdenscon,1); nc2 = size(mfccdenscon,2); nc3 = size(mfccdenscon,3)
         call mfcc_reduce_dens(mfccdensconi,nc1,nc2,nc3)
         call mfcc_reduce_dens(mfccdensconj,nc1,nc2,nc3)
         call mfcc_reduce_dens(mfccdenscon, nc1,nc2,nc3)
         call mfcc_reduce_idx(mfccbasesconi,kxiaoconnect)
         call mfcc_reduce_idx(mfccbasefconi,kxiaoconnect)
         call mfcc_reduce_idx(mfccbasesconj,kxiaoconnect)
         call mfcc_reduce_idx(mfccbasefconj,kxiaoconnect)
      endif
   endif
#endif

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
#ifdef MPIV
   if (MPIsaved) then
      quick_comm = commsaved
      quick_comm_rank = ranksaved
      quick_comm_size = sizesaved
   endif
#endif
   master = mastersaved
   deallocate(xyzsaved)
   if (allocated(concord)) deallocate(concord)
   if (allocated(consym))  deallocate(consym)

end subroutine mfcc_fragment_scf


!-------------------------------------------------------
! mfcc_set_submol
!-------------------------------------------------------
! Install a fragment as the current molecule: atom count, coordinates and
! atom types. mfcc_start stores coordinates in Angstrom, QUICK works in bohr.
!-------------------------------------------------------

subroutine mfcc_set_submol(nat,cord,sym,icharge,ierr)
   use allmod
   implicit none

   integer, intent(in) :: nat
   double precision, intent(in) :: cord(3,*)
   character(len=2), intent(in) :: sym(*)
   integer, intent(in) :: icharge          ! formal charge of this sub-molecule
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

   ! Apply the formal charge before testing parity. A fragment holding one
   ! charged terminus of a zwitterion is an ion, so the neutral electron count
   ! is odd and only becomes even once the charge is accounted for.
   quick_molspec%nelec = quick_molspec%nelec - icharge
   quick_molspec%molchg = icharge

   if (mod(quick_molspec%nelec,2) .ne. 0) then
      call PrtErr(iOutFile,'MFCC sub-molecule has an odd number of electrons even after &
            &applying its formal charge; only closed shell fragments are supported.')
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

subroutine mfcc_run_submol(nat,cord,sym,icharge,iatstart,iatfinal,ibasstart,ibasfinal,nb,ierr)
   use allmod
   use quick_exception_module
   use quick_cutoff_module, only: schwarzoff
   use quick_eri_cshell_module, only: getEriPrecomputables
   implicit none

   integer, intent(in) :: nat, iatstart, iatfinal, icharge
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

   call mfcc_set_submol(nat,cord,sym,icharge,ierr)
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

   ! deallocate_calculated above released quick_basis, and with it the
   ! primitive-pair arrays the ERI engine works through: Apri, Kpri, Ppri,
   ! cutprim and Xcoeff, all dimensioned by jbasis. readbasis sets jbasis but
   ! does not reallocate them, so without this the two-electron contribution
   ! never reaches the Fock matrix and each fragment converges its bare
   ! one-electron Hamiltonian. This mirrors getMol, which allocates and zeroes
   ! them in exactly this order.
   call alloc(quick_basis)
   call alloc(quick_qm_struct)
   cutprim = 0.0d0
   quick_basis%Xcoeff = 0.0d0
   call init(quick_qm_struct)
   call normalize_basis()

   ! main calls these two once, after getMol, and mfcc_fragment_scf runs well
   ! before that. Without them Ycutoff is allocated but never filled, so the
   ! Schwarz test rejects every shell quartet and the fragment converges its
   ! bare one-electron Hamiltonian: the electrons collapse onto the most
   ! attractive nuclei and the fragment density is meaningless. They have to be
   ! redone per fragment anyway, since both are sized and valued by the basis.
   call getEriPrecomputables
   call schwarzoff

   ! Crude diagonal starting density, as the SAD guess does for atoms.
   diagelement = dble(quick_molspec%nelec)/dble(nbasis)
   do i = 1, nbasis
      quick_qm_struct%dense(i,i) = diagelement
   enddo

   ! getEnergy with isGuess=.true. builds X and the nuclear repulsion and
   ! runs the SCF, while skipping the DFT grid and the verbose banners.
   quick_method%scf_conv = .false.
   call getEnergy(.true.,ierr)
   if (ierr /= 0) return

   ! A sub-molecule that ran out of cycles leaves an unconverged density behind,
   ! and nothing downstream can tell: the guess is assembled from it, the SCF
   ! starts from garbage and reports a total energy that is simply wrong. Seen
   ! with MAXDIIS=2, where the charged C-terminal fragment hit the cycle limit
   ! and the run finished quietly at -761.33 instead of -1299.50. Fail here
   ! instead, since the fragment SCFs run before any of that is visible.
   if (.not. quick_method%scf_conv) then
      call PrtErr(iOutFile,'An MFCC sub-molecule SCF did not converge; its density &
            &cannot be used to build the initial guess.')
      ierr = 44
      return
   endif

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


#ifdef MPIV
!-------------------------------------------------------
! mfcc_reduce_dens / mfcc_reduce_idx
!-------------------------------------------------------
! Sum the per-rank density blocks, and max the per-rank basis range indices,
! over the real communicator. Separate routines only because the arrays differ
! in type and rank; both rely on the unwritten entries being exactly zero.
!-------------------------------------------------------

subroutine mfcc_reduce_dens(a,n1,n2,n3)
   use quick_mpi_module, only: quick_comm, quick_mpi_error
   use mpi
   implicit none
   integer, intent(in) :: n1,n2,n3
   double precision, intent(inout) :: a(n1,n2,n3)
   double precision, allocatable :: tmp(:,:,:)

   allocate(tmp(n1,n2,n3))
   tmp = a
   call MPI_ALLREDUCE(tmp,a,n1*n2*n3,mpi_double_precision,MPI_SUM,quick_comm,quick_mpi_error)
   deallocate(tmp)
end subroutine mfcc_reduce_dens


subroutine mfcc_reduce_idx(a,n)
   use quick_mpi_module, only: quick_comm, quick_mpi_error
   use mpi
   implicit none
   integer, intent(in) :: n
   integer, intent(inout) :: a(n)
   integer, allocatable :: tmp(:)

   if (n .le. 0) return
   allocate(tmp(n))
   tmp = a
   call MPI_ALLREDUCE(tmp,a,n,mpi_integer,MPI_MAX,quick_comm,quick_mpi_error)
   deallocate(tmp)
end subroutine mfcc_reduce_idx
#endif
