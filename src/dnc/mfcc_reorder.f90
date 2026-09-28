#include "util.fh"
!
! mfcc_reorder.f90
!
! MFCC defines every fragment, cap and connection block as a contiguous range
! of global atom index, and deliberately cuts those ranges mid-residue to
! capture the peptide linkage. That is only chemically closed if each hydrogen
! sits next to the heavy atom it is bonded to.
!
! Files written by Open Babel and many other tools instead group a residue's
! hydrogens after its heavy atoms. With that ordering a mid-residue cut both
! strands hydrogens whose heavy atom was excluded and drops hydrogens belonging
! to heavy atoms that were included. On the glycine hexamer that is 14 stranded
! and 30 dropped hydrogens across the fragments and caps, so every sub-molecule
! is a different species from the one the method assumes.
!
! The fix has to be the atom order itself: MFCC_initial_guess maps a fragment's
! local density block onto the global density with matombases + (i-mfccbases),
! which requires each fragment's atoms to stay a contiguous global range. Atoms
! cannot simply be added to or removed from a span.
!
! This routine detects the problem and reorders, writing both the pdb it will
! actually use and an xyz copy for inspection.
!

subroutine mfcc_check_atom_order(ierr)
   use allmod
   use quick_mpi_module, only: master
   implicit none

   integer, intent(inout) :: ierr

   ! Heap, not stack. These were automatic arrays with a fixed MAXAT, which put
   ! roughly 114 bytes per atom on the stack: a 120000 atom ceiling would have
   ! needed 13.7 MB against a typical 8 MB stack limit and simply crashed.
   ! Allocated from the atom count instead, so there is no built in ceiling.
   character(len=80), allocatable :: line(:)
   character(len=2),  allocatable :: elem(:)
   double precision,  allocatable :: crd(:,:)
   integer,           allocatable :: owner(:), perm(:)
   character(len=80) :: pdbline
   integer :: nat, i, j, k, ios, nmoved, nh, npass
   double precision :: d, dbest
   character(len=120) :: fname_pdb, fname_xyz
   logical :: identity

   ! H to heavy covalent cutoff, Angstrom. Generous: the longest X-H here is
   ! about 1.09 (C-H), and the shortest non-bond contact is well above 1.5.
   double precision, parameter :: HBOND_MAX = 1.35d0

   if (.not.master) return

   ! ------------------------------------------------------------------
   ! Read the pdb as records so it can be rewritten verbatim apart from order.
   ! ------------------------------------------------------------------
   ! Pass 1 counts the coordinate records, pass 2 stores them.
   do npass = 1, 2
      nat = 0
      open(REORDERFILEHANDLE,file=PDBFileName,status='OLD',iostat=ios)
      if (ios /= 0) then
         call PrtErr(iOutFile,'Could not open the pdb file to check MFCC atom ordering.')
         ierr = 45
         return
      endif
      do
         read(REORDERFILEHANDLE,'(a80)',iostat=ios) pdbline
         if (ios /= 0) exit
         if (pdbline(1:4).ne.'ATOM'.and.pdbline(1:6).ne.'HETATM') cycle
         nat = nat + 1
         if (npass .eq. 2) then
            line(nat) = pdbline
            elem(nat) = adjustl(pdbline(13:14))
            read(pdbline(31:54),'(3f8.3)') crd(1,nat),crd(2,nat),crd(3,nat)
         endif
      enddo
      close(REORDERFILEHANDLE)
      if (npass .eq. 1) then
         if (nat .lt. 2) return
         allocate(line(nat),elem(nat),crd(3,nat),owner(nat),perm(nat))
      endif
   enddo

   ! ------------------------------------------------------------------
   ! For each hydrogen find the heavy atom it is bonded to.
   ! This is an O(N_H * N_heavy) search: fine to tens of thousands of atoms,
   ! but it would need a cell list well before a hundred thousand.
   ! ------------------------------------------------------------------
   do i = 1, nat
      owner(i) = 0
      if (elem(i).ne.'H ') cycle
      dbest = 1.0d30
      do j = 1, nat
         if (elem(j).eq.'H ') cycle
         d = dsqrt((crd(1,i)-crd(1,j))**2+(crd(2,i)-crd(2,j))**2+(crd(3,i)-crd(3,j))**2)
         if (d .lt. dbest) then
            dbest = d
            owner(i) = j
         endif
      enddo
      if (dbest .gt. HBOND_MAX) then
         write(iOutFile,'(" MFCC: pdb atom ",i6," is a hydrogen with no heavy atom within ", &
               &f5.2," A")') i,HBOND_MAX
         call PrtErr(iOutFile,'Cannot determine hydrogen connectivity for the MFCC atom order check.')
         ierr = 45
         return
      endif
   enddo

   ! ------------------------------------------------------------------
   ! Required order: every heavy atom in file order, each followed by its own
   ! hydrogens. If that is already the file order there is nothing to do.
   ! ------------------------------------------------------------------
   k = 0
   do i = 1, nat
      if (elem(i).eq.'H ') cycle
      k = k + 1
      perm(k) = i
      do j = 1, nat
         if (elem(j).eq.'H ' .and. owner(j).eq.i) then
            k = k + 1
            perm(k) = j
         endif
      enddo
   enddo

   if (k .ne. nat) then
      call PrtErr(iOutFile,'MFCC atom order check produced an incomplete permutation.')
      ierr = 45
      return
   endif

   identity = .true.
   nmoved = 0
   do i = 1, nat
      if (perm(i).ne.i) then
         identity = .false.
         nmoved = nmoved + 1
      endif
   enddo

   if (identity) then
      write(iOutFile,'(" MFCC: atom order already has every hydrogen next to its heavy atom.")')
      deallocate(line,elem,crd,owner,perm)
      return
   endif

   ! ------------------------------------------------------------------
   ! Reorder. Write the pdb MFCC will read, plus an xyz for inspection.
   ! ------------------------------------------------------------------
   if (allocated(mfcc_perm)) deallocate(mfcc_perm)
   allocate(mfcc_perm(nat))
   do i = 1, nat
      mfcc_perm(i) = perm(i)
   enddo
   mfcc_reordered = .true.

   fname_pdb = trim(adjustl(baseinFileName)) // '_reordered.pdb'
   fname_xyz = trim(adjustl(baseinFileName)) // '_reordered.xyz'

   open(REORDERFILEHANDLE,file=fname_pdb,status='REPLACE')
   do i = 1, nat
      j = perm(i)
      write(REORDERFILEHANDLE,'(a6,i5,a)') line(j)(1:6), i, trim(line(j)(12:))
   enddo
   write(REORDERFILEHANDLE,'("TER")')
   write(REORDERFILEHANDLE,'("END")')
   close(REORDERFILEHANDLE)

   open(REORDERFILEHANDLE,file=fname_xyz,status='REPLACE')
   write(REORDERFILEHANDLE,'(i8)') nat
   write(REORDERFILEHANDLE,'("MFCC reordered atom order: each heavy atom followed by its hydrogens")')
   do i = 1, nat
      j = perm(i)
      write(REORDERFILEHANDLE,'(a2,3(2x,f14.8))') elem(j),crd(1,j),crd(2,j),crd(3,j)
   enddo
   close(REORDERFILEHANDLE)

   PDBFileName = fname_pdb

   nh = 0
   do i = 1, nat
      if (elem(i).eq.'H ') nh = nh + 1
   enddo

   call PrtWrn(iOutFile,'Atom order was NOT MFCC compatible and has been reordered.')
   write(iOutFile,'("|          ",i6," of ",i6," atoms changed position (",i6," hydrogens).")') &
         nmoved,nat,nh
   write(iOutFile,'("|")')
   write(iOutFile,'("|          MFCC cuts fragments and caps mid residue using contiguous atom")')
   write(iOutFile,'("|          index ranges, so each hydrogen must sit next to the heavy atom")')
   write(iOutFile,'("|          it is bonded to. The input grouped them differently, which would")')
   write(iOutFile,'("|          have left hydrogens stranded in fragments whose heavy atom was")')
   write(iOutFile,'("|          excluded, and dropped hydrogens from atoms that were included.")')
   write(iOutFile,'("|")')
   write(iOutFile,'("|          Reordered files written for inspection:")')
   write(iOutFile,'("|            ",a)') trim(fname_pdb)
   write(iOutFile,'("|            ",a)') trim(fname_xyz)
   write(iOutFile,'("|")')
   write(iOutFile,'("|          NOTE: all per atom output below (geometry, charges, gradients)")')
   write(iOutFile,'("|          is in the REORDERED order, not the order of your input file.")')
   write(iOutFile,'("|          The total energy is unaffected by atom ordering.")')
   write(iOutFile,'(a)')
   call flush(iOutFile)

   deallocate(line,elem,crd,owner,perm)

end subroutine mfcc_check_atom_order


!-------------------------------------------------------
! mfcc_apply_reorder
!-------------------------------------------------------
! Apply the permutation found by mfcc_check_atom_order to the global geometry.
! Called from getMol once the input coordinates have been read, so that the
! global basis, matombases and the DnC fragmentation all see the same order the
! MFCC fragmentation used.
!-------------------------------------------------------

subroutine mfcc_apply_reorder()
   use allmod
   implicit none

   integer :: i, j
   double precision, allocatable :: xyztmp(:,:)
   integer, allocatable :: ittmp(:)
   double precision, allocatable :: chgtmp(:)

   if (.not.mfcc_reordered) return
   if (.not.allocated(mfcc_perm)) return
   if (size(mfcc_perm) .ne. natom) return

   allocate(xyztmp(3,natom),ittmp(natom),chgtmp(natom))
   do i = 1, natom
      j = mfcc_perm(i)
      xyztmp(1:3,i) = xyz(1:3,j)
      ittmp(i) = quick_molspec%iattype(j)
      chgtmp(i) = quick_molspec%chg(j)
   enddo
   do i = 1, natom
      xyz(1:3,i) = xyztmp(1:3,i)
      quick_molspec%iattype(i) = ittmp(i)
      quick_molspec%chg(i) = chgtmp(i)
   enddo
   if (associated(quick_molspec%xyz)) quick_molspec%xyz(1:3,1:natom) = xyz(1:3,1:natom)
   deallocate(xyztmp,ittmp,chgtmp)

end subroutine mfcc_apply_reorder
