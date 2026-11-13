#include "util.fh"
!
!       quick_mfcc_species_module.f90
!
! Residue classification for the MFCC fragmentation.
!
! MFCC cuts a peptide chain between residues and caps the cuts, which only has
! meaning for amino acids. Everything else in a prepared structure - solvent,
! ions, ligands - has to be told apart from the chain and handled whole.
!
! Classification is by residue NAME, not by looking for backbone atom names,
! and that is deliberate. Testing "does this residue have N, CA and C" would be
! circular, because the backbone scan it feeds is the thing being protected:
! benzamidine in the Trypsin structure carries an atom whose name field is
! exactly ' C  ', the backbone carbonyl name, and an unfiltered scan counts it
! as a backbone carbon and shifts every fragment boundary after it.
!

module quick_mfcc_species_module
   implicit none
   private
   public :: mfcc_is_amino_acid, mfcc_species_charge, mfcc_species_kind
   public :: mfcc_chg_read, mfcc_chg_lookup, mfcc_chg_free
   public :: mfcc_chg_n, mfcc_chg_name, mfcc_chg_seq, mfcc_chg_val, mfcc_chg_used

   !--------------------------------------------------------------------!
   ! User-supplied charge overrides, read from <stem>.charges when the   !
   ! MFCCCHG keyword is given.                                           !
   !                                                                     !
   ! The charge of a standard residue can be worked out from its name    !
   ! and the hydrogens bonded to its charged groups, which is what       !
   ! mfcc_species_charge and the detection pass in mfcc_start do. A       !
   ! ligand cannot: BEN is benzamidine, whose amidinium group carries a   !
   ! +1 that nothing in the residue name reveals, and getting it wrong    !
   ! is not a small error -- neutral C7H9N2 has 65 electrons, so the      !
   ! fragment is not even closed shell. Hence this table, which lets the  !
   ! user state the charge of any residue by name and number.            !
   !--------------------------------------------------------------------!
   integer :: mfcc_chg_n = 0
   character(len=3), allocatable :: mfcc_chg_name(:)
   integer, allocatable :: mfcc_chg_seq(:)
   integer, allocatable :: mfcc_chg_val(:)
   ! Set when an entry has been matched to a residue, so mfcc_start can
   ! complain about one that matched nothing rather than ignore it.
   logical, allocatable :: mfcc_chg_used(:)

   ! The twenty standard residues plus the protonation and linkage variants
   ! AMBER writes, which are what actually turn up in prepared files: LYN
   ! neutral lysine, HID/HIE/HIP the histidine states, CYX disulfide-bonded
   ! cysteine, CYM deprotonated, ASH/GLH protonated acids, HYP hydroxyproline,
   ! ACE/NME/NHE the terminal caps.
   integer, parameter :: NAA = 37
   character(len=3), parameter :: AA_NAMES(NAA) = [ character(len=3) :: &
      'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE', &
      'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL', &
      'LYN','HID','HIE','HIP','CYX','CYM','ASH','GLH','HYP','ARN', &
      'ACE','NME','NHE','TYM','SEC','PYL','MSE' ]

contains

   !--------------------------------------------------------------------!
   ! Is this residue part of a peptide chain?                           !
   !--------------------------------------------------------------------!
   logical function mfcc_is_amino_acid(resnm) result(isaa)
      implicit none
      character(len=*), intent(in) :: resnm
      character(len=3) :: r
      integer :: i

      r = adjustl(resnm)
      call mfcc_upcase3(r)
      isaa = .false.
      do i = 1, NAA
         if (r .eq. AA_NAMES(i)) then
            isaa = .true.
            return
         endif
      enddo
   end function mfcc_is_amino_acid


   !--------------------------------------------------------------------!
   ! Kind of non-peptide residue, for reporting only.                   !
   !   1 solvent   2 monatomic   3 polyatomic (ligand)                  !
   !--------------------------------------------------------------------!
   integer function mfcc_species_kind(resnm, nat) result(kind)
      implicit none
      character(len=*), intent(in) :: resnm
      integer, intent(in) :: nat
      character(len=3) :: r

      r = adjustl(resnm)
      call mfcc_upcase3(r)

      if (r.eq.'WAT' .or. r.eq.'HOH' .or. r.eq.'SOL' .or. r.eq.'T3P' .or. &
          r.eq.'TIP') then
         kind = 1
      else if (nat .eq. 1) then
         kind = 2
      else
         kind = 3
      endif
   end function mfcc_species_kind


   !--------------------------------------------------------------------!
   ! Formal charge of a non-peptide residue.                            !
   !                                                                    !
   ! Solvent is neutral and the common monatomic ions are tabulated. An  !
   ! unrecognised residue returns zero with unknown set, because the     !
   ! charge of an arbitrary ligand cannot be had from its name. The      !
   ! caller warns, and the closed-shell parity test in mfcc_set_submol   !
   ! catches it when neutral turns out to be the wrong guess.            !
   !--------------------------------------------------------------------!
   integer function mfcc_species_charge(resnm, nat, unknown) result(chg)
      implicit none
      character(len=*), intent(in) :: resnm
      integer, intent(in) :: nat
      logical, intent(out) :: unknown
      character(len=3) :: r

      r = adjustl(resnm)
      call mfcc_upcase3(r)
      unknown = .false.
      chg = 0

      select case (r)
      case ('WAT','HOH','SOL','T3P','TIP')
         chg = 0
      case ('NA','LI','K','RB','CS','AG')
         chg = 1
      case ('MG','CA','ZN','MN','NI','CO','SR','BA','CU')
         chg = 2
      case ('AL','CR')
         chg = 3
      case ('CL','BR','F','I','OH')
         chg = -1
      case default
         unknown = .true.
         chg = 0
      end select
   end function mfcc_species_charge


   !--------------------------------------------------------------------!
   ! mfcc_chg_read                                                      !
   !                                                                    !
   ! Read the charge override table. One residue per line:              !
   !                                                                    !
   !     # residue  resSeq  charge                                      !
   !     BEN 225 +1                                                     !
   !     CA  224 +2                                                     !
   !                                                                    !
   ! Blank lines and lines beginning # or ! are ignored, the three       !
   ! fields are read list-directed so any amount of whitespace between   !
   ! them is fine, and the charge may be written +1, 1 or -1.           !
   !                                                                    !
   ! Nothing is validated against the structure here, because this runs  !
   ! before the pdb is known; mfcc_start matches each entry to a residue !
   ! and is what rejects a name or number that does not exist.          !
   !--------------------------------------------------------------------!
   subroutine mfcc_chg_read(funit, fname, io, ierr)
      implicit none
      integer, intent(in) :: funit        ! unit to use
      character(len=*), intent(in) :: fname
      integer, intent(in) :: io           ! output unit for messages
      integer, intent(inout) :: ierr

      character(len=200) :: line
      character(len=3) :: nm
      integer :: ios, nline, k, iseq, ival, ia, j
      logical :: exists

      call mfcc_chg_free()

      inquire(file=fname, exist=exists)
      if (.not.exists) then
         write(io,'(" MFCCCHG was requested but the charge file was not found: ",a)') trim(fname)
         ierr = 49
         return
      endif

      ! Two passes: count the data lines, then read them. Simpler than growing
      ! the arrays, and the file is a handful of lines.
      do k = 1, 2
         if (k .eq. 2) then
            allocate(mfcc_chg_name(max(nline,1)), mfcc_chg_seq(max(nline,1)), &
                     mfcc_chg_val(max(nline,1)), mfcc_chg_used(max(nline,1)), stat=ia)
            if (ia /= 0) then
               ierr = 34
               return
            endif
            mfcc_chg_name = '   '
            mfcc_chg_seq = 0
            mfcc_chg_val = 0
            mfcc_chg_used = .false.
         endif

         open(funit, file=fname, status='OLD', form='FORMATTED', action='READ', iostat=ios)
         if (ios /= 0) then
            write(io,'(" Could not open the MFCC charge file: ",a)') trim(fname)
            ierr = 49
            return
         endif

         nline = 0
         do
            read(funit,'(a200)',iostat=ios) line
            if (ios /= 0) exit
            line = adjustl(line)
            if (len_trim(line) .eq. 0) cycle
            if (line(1:1) .eq. '#' .or. line(1:1) .eq. '!') cycle
            nline = nline + 1
            if (k .eq. 1) cycle

            read(line,*,iostat=ios) nm, iseq, ival
            if (ios /= 0) then
               write(io,'(" Could not parse line ",i0," of ",a,":")') nline, trim(fname)
               write(io,'("   ",a)') trim(line)
               write(io,'(" Expected: residue name, residue number, charge (e.g. BEN 225 +1)")')
               close(funit)
               ierr = 49
               return
            endif
            call mfcc_upcase3(nm)
            mfcc_chg_name(nline) = nm
            mfcc_chg_seq(nline) = iseq
            mfcc_chg_val(nline) = ival
         enddo
         close(funit)
      enddo

      mfcc_chg_n = nline

      ! Two entries for one residue number would make the result depend on the
      ! order they happen to be scanned in.
      do k = 1, mfcc_chg_n
         do j = k+1, mfcc_chg_n
            if (mfcc_chg_seq(j) .eq. mfcc_chg_seq(k)) then
               write(io,'(" Residue ",i0," is listed twice in ",a)') mfcc_chg_seq(k), trim(fname)
               ierr = 49
               return
            endif
         enddo
      enddo

      if (mfcc_chg_n .eq. 0) then
         write(io,'(" MFCCCHG was requested but ",a," holds no entries.")') trim(fname)
         ierr = 49
         return
      endif

   end subroutine mfcc_chg_read


   !--------------------------------------------------------------------!
   ! mfcc_chg_lookup                                                    !
   !                                                                    !
   ! Is there an override for this residue? Matching is by residue       !
   ! number; the name is then compared as a check on the user, and a     !
   ! mismatch is reported through nameref so the caller can refuse to    !
   ! run rather than silently charge the wrong residue.                  !
   !--------------------------------------------------------------------!
   subroutine mfcc_chg_lookup(iseq, resnm, chg, found, mismatch, nameref)
      implicit none
      integer, intent(in) :: iseq
      character(len=*), intent(in) :: resnm
      integer, intent(out) :: chg
      logical, intent(out) :: found, mismatch
      character(len=3), intent(out) :: nameref
      integer :: k
      character(len=3) :: r

      chg = 0
      found = .false.
      mismatch = .false.
      nameref = '   '
      if (mfcc_chg_n .le. 0) return

      r = adjustl(resnm)
      call mfcc_upcase3(r)

      do k = 1, mfcc_chg_n
         if (mfcc_chg_seq(k) .ne. iseq) cycle
         found = .true.
         nameref = mfcc_chg_name(k)
         chg = mfcc_chg_val(k)
         mfcc_chg_used(k) = .true.
         if (mfcc_chg_name(k) .ne. r) mismatch = .true.
         return
      enddo

   end subroutine mfcc_chg_lookup


   subroutine mfcc_chg_free()
      implicit none
      if (allocated(mfcc_chg_name)) deallocate(mfcc_chg_name)
      if (allocated(mfcc_chg_seq))  deallocate(mfcc_chg_seq)
      if (allocated(mfcc_chg_val))  deallocate(mfcc_chg_val)
      if (allocated(mfcc_chg_used)) deallocate(mfcc_chg_used)
      mfcc_chg_n = 0
   end subroutine mfcc_chg_free


   subroutine mfcc_upcase3(r)
      implicit none
      character(len=3), intent(inout) :: r
      integer :: i, ic
      do i = 1, 3
         ic = ichar(r(i:i))
         if (ic .ge. 97 .and. ic .le. 122) r(i:i) = char(ic-32)
      enddo
   end subroutine mfcc_upcase3

end module quick_mfcc_species_module
