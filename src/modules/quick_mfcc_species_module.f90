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
