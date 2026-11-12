!
!	quick_mfcc_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

! MFCC Module
module quick_mfcc_module
    implicit none

! Danil ! was already commented out
!    integer, allocatable, dimension(:) :: mfccatom,mfcccharge

! This one I commented out
!    integer :: mfccatom(50),mfcccharge(50),IMFCC,kxiaoconnect
    integer :: IMFCC,kxiaoconnect

    ! Fragment count and per fragment/cap atom counts. These were local to
    ! mfcc() and therefore thrown away when it returned, which left the
    ! fragment densities impossible to compute afterwards.
    integer :: npmfcc
    ! Standalone fragments appended after the peptide chain: one per solvent
    ! molecule, ion or ligand. They carry no caps, because nothing was cut to
    ! make them, so the cap arrays stay at npmfcc-1 while the fragment arrays
    ! run to npmfcc+nmfccextra.
    integer :: nmfccextra = 0
    ! How many concaps there actually are. This was spelled npmfcc-1 at every
    ! site that touched a cap, which was true only while the backbone cuts were
    ! the only cuts. The disulfide correction below appends two more concaps per
    ! bridge, so the count is now carried explicitly.
    integer :: nmfcccap = 0

    ! ------------------------------------------------------------------!
    ! Disulfide correction blocks.                                       !
    !                                                                    !
    ! MFCC never cuts the bond it means to preserve: it moves the cut to  !
    ! an innocuous single bond, lets the preserved bond sit whole inside  !
    ! the fragment, and subtracts whatever that double counts as a        !
    ! concap. A disulfide is a bond through the chain between two         !
    ! sequence distant residues, and capping it with hydrogen (which is   !
    ! still done, because it is what keeps those fragments closed shell)  !
    ! leaves both halves described as thiols. Two thiols are not a        !
    ! disulfide: the sulfur is in the wrong oxidation state in both.      !
    !                                                                    !
    ! So the bridge is restored as a correction, in exactly the MFCC      !
    ! form. Per bridge:                                                   !
    !                                                                    !
    !   D += D(CH3-S-S-CH3) - D(CH3-S-H)_i - D(CH3-S-H)_j                !
    !                                                                    !
    ! The CH3-S-S-CH3 block carries the intact S-S at the real geometry,  !
    ! cut at the two CA-CB bonds and capped there with hydrogen, which is !
    ! the direct analogue of cutting the backbone at CA-C. The two        !
    ! thiols are what the capped fragments already contributed, so they   !
    ! are removed; they are ordinary concaps and live in the cap arrays.  !
    !                                                                    !
    ! The S-S block is the only piece of MFCC that does not map onto one  !
    ! contiguous run of global basis functions: its two halves sit in     !
    ! different residues. Its atoms are laid out locally as              !
    !                                                                    !
    !   [Hcap] [CB_i .. SG_i] [SG_j .. CB_j] [Hcap]                       !
    !                                                                    !
    ! so the real atoms are contiguous LOCALLY and split into two global  !
    ! runs. Only the split point and the two global starts are needed,    !
    ! and the off diagonal block between the halves is what carries the   !
    ! S-S bond density itself.                                           !
    !____________________________________________________________________!
    integer :: nmfccss = 0
    integer, allocatable :: mfccatomss(:), mfccchargess(:)
    double precision, allocatable :: mfcccordss(:,:,:)
    character(len=2), allocatable :: mfccatomxiaoss(:,:)
    ! Local atom/basis range of the real (non cap) atoms, as mfcc_run_submol
    ! reports it for any sub-molecule.
    integer, allocatable :: mfccstartss(:), mfccfinalss(:)
    integer, allocatable :: mfccbasesss(:), mfccbasefss(:)
    ! Global atom ranges of the two halves, and the global basis start of each.
    integer, allocatable :: matomstartss1(:), matomfinalss1(:)
    integer, allocatable :: matomstartss2(:), matomfinalss2(:)
    integer, allocatable :: matombasesss1(:), matombasefss1(:)
    integer, allocatable :: matombasesss2(:), matombasefss2(:)

    integer, allocatable :: mfccatom(:), mfccatomcap(:)
    ! Per fragment/cap formal charge. Set from the terminus composition in
    ! mfcc_start; zero for everything else.
    integer, allocatable :: mfcccharge(:), mfccchargecap(:)

    ! Atom reordering applied so that every hydrogen sits next to the heavy
    ! atom it is bonded to. MFCC defines fragments as contiguous ranges of
    ! global atom index and cuts them mid-residue, so a hydrogen separated
    ! from its heavy atom is either stranded in a fragment or dropped from
    ! one. mfcc_perm(new) = old index.
    logical :: mfcc_reordered = .false.
    integer, allocatable :: mfcc_perm(:)
    double precision, allocatable :: mfcccord(:,:,:)
    integer ::Ftmp(300)
    character(len=100)::linetmp
    character(len=2), allocatable :: mfccatomxiao(:,:)
    integer, allocatable :: mfccstart(:), mfccfinal(:), mfccbases(:), mfccbasef(:)
    integer, allocatable :: matomstart(:), matomfinal(:), matombases(:), matombasef(:)

! Danil ! was already commented out
!    integer, dimension(:), allocatable :: matomstart,matomfinal,matombases &
!    ,matombasef

! This one I commented out
!    integer :: mfccatomcap(50),mfccchargecap(50)
    double precision, allocatable :: mfcccordcap(:,:,:)
    character(len=2), allocatable :: mfccatomxiaocap(:,:)
    integer, allocatable :: mfccstartcap(:), mfccfinalcap(:), mfccbasescap(:), mfccbasefcap(:)
    integer, allocatable :: matomstartcap(:), matomfinalcap(:), matombasescap(:), matombasefcap(:)

! Danil ! was already commented out
!    integer, dimension(:), allocatable :: matomstartcap,matomfinalcap,matombasescap &
!    ,matombasefcap

    integer, allocatable :: mfccatomcon(:), mfccchargecon(:)
    double precision, allocatable :: mfcccordcon(:,:,:)
    character(len=2), allocatable :: mfccatomxiaocon(:,:)
    integer, allocatable :: mfccstartcon(:), mfccfinalcon(:), mfccbasescon(:), mfccbasefcon(:)
    integer, allocatable :: matomstartcon(:), matomfinalcon(:), matombasescon(:), matombasefcon(:)

! Danil ! was already commented out
!    integer, dimension(:), allocatable :: matomstartcap,matomfinalcap,matombasescap &
!    ,matombasefcap

    integer, allocatable :: mfccatomcon2(:), mfccchargecon2(:)
    double precision, allocatable :: mfcccordcon2(:,:,:)
    character(len=2), allocatable :: mfccatomxiaocon2(:,:)
    integer, allocatable :: mfccstartcon2(:), mfccfinalcon2(:), mfccbasescon2(:), mfccbasefcon2(:)
    integer, allocatable :: matomstartcon2(:), matomfinalcon2(:), matombasescon2(:), matombasefcon2(:)

    integer, allocatable :: mfccatomconi(:), mfccchargeconi(:)
    double precision, allocatable :: mfcccordconi(:,:,:)
    character(len=2), allocatable :: mfccatomxiaoconi(:,:)
    integer, allocatable :: mfccstartconi(:), mfccfinalconi(:), mfccbasesconi(:), mfccbasefconi(:)
    integer, allocatable :: matomstartconi(:), matomfinalconi(:), matombasesconi(:), matombasefconi(:)

    integer, allocatable :: mfccatomconj(:), mfccchargeconj(:)
    double precision, allocatable :: mfcccordconj(:,:,:)
    character(len=2), allocatable :: mfccatomxiaoconj(:,:)
    integer, allocatable :: mfccstartconj(:), mfccfinalconj(:), mfccbasesconj(:), mfccbasefconj(:)
    integer, allocatable :: matomstartconj(:), matomfinalconj(:), matombasesconj(:), matombasefconj(:)

    double precision, allocatable, dimension(:,:,:) :: mfccdens,mfccdenscap,mfccdenscon &
                                            ,mfccdenscon2,mfccdensconi,mfccdensconj
    ! Density of each intact CH3-S-S-CH3 block. Unlike every other block this
    ! one spans two separate runs of global basis functions, so its off diagonal
    ! part -- the S-S bond density itself -- is scattered as well.
    double precision, allocatable, dimension(:,:,:) :: mfccdensss

contains

   !----------------------------------------------------------------------!
   ! mfcc_alloc_frag                                                      !
   !                                                                      !
   ! Size the fragment and cap arrays from the system instead of the      !
   ! fixed 50 fragments by 100 atoms these used to carry. npmfcc comes    !
   ! straight from the residue number in the pdb and nothing checked it,  !
   ! so a 51 residue protein wrote past the end of every one of these     !
   ! arrays with no diagnostic at all.                                    !
   !                                                                      !
   ! nfrag is the fragment count and natmax the largest number of atoms   !
   ! any single fragment, cap or connection block can hold, both known    !
   ! once the backbone has been located.                                  !
   !______________________________________________________________________!
   subroutine mfcc_alloc_frag(nfrag, natmax, ierr)
      implicit none
      integer, intent(in)    :: nfrag, natmax
      integer, intent(inout) :: ierr
      integer :: ia

      call mfcc_dealloc_frag()

      allocate(mfccatom(nfrag), mfccatomcap(nfrag), &
               mfcccharge(nfrag), mfccchargecap(nfrag), &
               mfccstart(nfrag), mfccfinal(nfrag), mfccbases(nfrag), mfccbasef(nfrag), &
               matomstart(nfrag), matomfinal(nfrag), matombases(nfrag), matombasef(nfrag), &
               mfccstartcap(nfrag), mfccfinalcap(nfrag), mfccbasescap(nfrag), mfccbasefcap(nfrag), &
               matomstartcap(nfrag), matomfinalcap(nfrag), matombasescap(nfrag), &
               matombasefcap(nfrag), stat=ia)
      if (ia /= 0) goto 900

      allocate(mfcccord(3,natmax,nfrag), mfcccordcap(3,natmax,nfrag), &
               mfccatomxiao(natmax,nfrag), mfccatomxiaocap(natmax,nfrag), stat=ia)
      if (ia /= 0) goto 900

      mfccatom = 0; mfccatomcap = 0
      mfcccharge = 0; mfccchargecap = 0
      mfccstart = 0; mfccfinal = 0; mfccbases = 0; mfccbasef = 0
      matomstart = 0; matomfinal = 0; matombases = 0; matombasef = 0
      mfccstartcap = 0; mfccfinalcap = 0; mfccbasescap = 0; mfccbasefcap = 0
      matomstartcap = 0; matomfinalcap = 0; matombasescap = 0; matombasefcap = 0
      mfcccord = 0.0d0; mfcccordcap = 0.0d0
      mfccatomxiao = '  '; mfccatomxiaocap = '  '
      return

900   ierr = 34
      return
   end subroutine mfcc_alloc_frag


   !----------------------------------------------------------------------!
   ! mfcc_alloc_con                                                       !
   !                                                                      !
   ! The connection blocks are sized separately: how many there are is    !
   ! only known once the contact search has run, and on a folded protein  !
   ! there can be far more of them than there are fragments. Trp-cage     !
   ! alone finds 24 contacts within 3 A.                                  !
   !______________________________________________________________________!
   subroutine mfcc_alloc_con(ncon, natmax, ierr)
      implicit none
      integer, intent(in)    :: ncon, natmax
      integer, intent(inout) :: ierr
      integer :: ia, n

      call mfcc_dealloc_con()
      n = max(ncon,1)

      allocate(mfccatomcon(n), mfccchargecon(n), mfccatomcon2(n), mfccchargecon2(n), &
               mfccatomconi(n), mfccchargeconi(n), mfccatomconj(n), mfccchargeconj(n), &
               mfccstartcon(n), mfccfinalcon(n), mfccbasescon(n), mfccbasefcon(n), &
               matomstartcon(n), matomfinalcon(n), matombasescon(n), matombasefcon(n), &
               mfccstartcon2(n), mfccfinalcon2(n), mfccbasescon2(n), mfccbasefcon2(n), &
               matomstartcon2(n), matomfinalcon2(n), matombasescon2(n), matombasefcon2(n), &
               mfccstartconi(n), mfccfinalconi(n), mfccbasesconi(n), mfccbasefconi(n), &
               matomstartconi(n), matomfinalconi(n), matombasesconi(n), matombasefconi(n), &
               mfccstartconj(n), mfccfinalconj(n), mfccbasesconj(n), mfccbasefconj(n), &
               matomstartconj(n), matomfinalconj(n), matombasesconj(n), matombasefconj(n), &
               stat=ia)
      if (ia /= 0) goto 900

      allocate(mfcccordcon(3,natmax,n), mfcccordcon2(3,natmax,n), &
               mfcccordconi(3,natmax,n), mfcccordconj(3,natmax,n), &
               mfccatomxiaocon(natmax,n), mfccatomxiaocon2(natmax,n), &
               mfccatomxiaoconi(natmax,n), mfccatomxiaoconj(natmax,n), stat=ia)
      if (ia /= 0) goto 900

      mfccatomcon = 0; mfccchargecon = 0; mfccatomcon2 = 0; mfccchargecon2 = 0
      mfccatomconi = 0; mfccchargeconi = 0; mfccatomconj = 0; mfccchargeconj = 0
      mfccstartcon = 0; mfccfinalcon = 0; mfccbasescon = 0; mfccbasefcon = 0
      matomstartcon = 0; matomfinalcon = 0; matombasescon = 0; matombasefcon = 0
      mfccstartcon2 = 0; mfccfinalcon2 = 0; mfccbasescon2 = 0; mfccbasefcon2 = 0
      matomstartcon2 = 0; matomfinalcon2 = 0; matombasescon2 = 0; matombasefcon2 = 0
      mfccstartconi = 0; mfccfinalconi = 0; mfccbasesconi = 0; mfccbasefconi = 0
      matomstartconi = 0; matomfinalconi = 0; matombasesconi = 0; matombasefconi = 0
      mfccstartconj = 0; mfccfinalconj = 0; mfccbasesconj = 0; mfccbasefconj = 0
      matomstartconj = 0; matomfinalconj = 0; matombasesconj = 0; matombasefconj = 0
      mfcccordcon = 0.0d0; mfcccordcon2 = 0.0d0
      mfcccordconi = 0.0d0; mfcccordconj = 0.0d0
      mfccatomxiaocon = '  '; mfccatomxiaocon2 = '  '
      mfccatomxiaoconi = '  '; mfccatomxiaoconj = '  '
      return

900   ierr = 34
      return
   end subroutine mfcc_alloc_con


   !----------------------------------------------------------------------!
   ! mfcc_alloc_ss / mfcc_dealloc_ss                                      !
   !                                                                      !
   ! The disulfide correction blocks, one per bridge. Sized separately    !
   ! because how many bridges there are is only known once the S..S       !
   ! search has run.                                                      !
   !______________________________________________________________________!
   subroutine mfcc_alloc_ss(nss, natmax, ierr)
      implicit none
      integer, intent(in)    :: nss, natmax
      integer, intent(inout) :: ierr
      integer :: ia, n

      call mfcc_dealloc_ss()
      n = max(nss,1)

      allocate(mfccatomss(n), mfccchargess(n), &
               mfccstartss(n), mfccfinalss(n), mfccbasesss(n), mfccbasefss(n), &
               matomstartss1(n), matomfinalss1(n), matomstartss2(n), matomfinalss2(n), &
               matombasesss1(n), matombasefss1(n), matombasesss2(n), matombasefss2(n), &
               stat=ia)
      if (ia /= 0) goto 900

      allocate(mfcccordss(3,natmax,n), mfccatomxiaoss(natmax,n), stat=ia)
      if (ia /= 0) goto 900

      mfccatomss = 0; mfccchargess = 0
      mfccstartss = 0; mfccfinalss = 0; mfccbasesss = 0; mfccbasefss = 0
      matomstartss1 = 0; matomfinalss1 = 0; matomstartss2 = 0; matomfinalss2 = 0
      matombasesss1 = 0; matombasefss1 = 0; matombasesss2 = 0; matombasefss2 = 0
      mfcccordss = 0.0d0
      mfccatomxiaoss = '  '
      return

900   ierr = 34
      return
   end subroutine mfcc_alloc_ss


   subroutine mfcc_dealloc_ss()
      implicit none
      if (allocated(mfccatomss))     deallocate(mfccatomss)
      if (allocated(mfccchargess))   deallocate(mfccchargess)
      if (allocated(mfccstartss))    deallocate(mfccstartss)
      if (allocated(mfccfinalss))    deallocate(mfccfinalss)
      if (allocated(mfccbasesss))    deallocate(mfccbasesss)
      if (allocated(mfccbasefss))    deallocate(mfccbasefss)
      if (allocated(matomstartss1))  deallocate(matomstartss1)
      if (allocated(matomfinalss1))  deallocate(matomfinalss1)
      if (allocated(matomstartss2))  deallocate(matomstartss2)
      if (allocated(matomfinalss2))  deallocate(matomfinalss2)
      if (allocated(matombasesss1))  deallocate(matombasesss1)
      if (allocated(matombasefss1))  deallocate(matombasefss1)
      if (allocated(matombasesss2))  deallocate(matombasesss2)
      if (allocated(matombasefss2))  deallocate(matombasefss2)
      if (allocated(mfcccordss))     deallocate(mfcccordss)
      if (allocated(mfccatomxiaoss)) deallocate(mfccatomxiaoss)
   end subroutine mfcc_dealloc_ss


   subroutine mfcc_dealloc_frag()
      implicit none
      if (allocated(mfccatom))        deallocate(mfccatom)
      if (allocated(mfccatomcap))     deallocate(mfccatomcap)
      if (allocated(mfcccharge))      deallocate(mfcccharge)
      if (allocated(mfccchargecap))   deallocate(mfccchargecap)
      if (allocated(mfccstart))       deallocate(mfccstart)
      if (allocated(mfccfinal))       deallocate(mfccfinal)
      if (allocated(mfccbases))       deallocate(mfccbases)
      if (allocated(mfccbasef))       deallocate(mfccbasef)
      if (allocated(matomstart))      deallocate(matomstart)
      if (allocated(matomfinal))      deallocate(matomfinal)
      if (allocated(matombases))      deallocate(matombases)
      if (allocated(matombasef))      deallocate(matombasef)
      if (allocated(mfccstartcap))    deallocate(mfccstartcap)
      if (allocated(mfccfinalcap))    deallocate(mfccfinalcap)
      if (allocated(mfccbasescap))    deallocate(mfccbasescap)
      if (allocated(mfccbasefcap))    deallocate(mfccbasefcap)
      if (allocated(matomstartcap))   deallocate(matomstartcap)
      if (allocated(matomfinalcap))   deallocate(matomfinalcap)
      if (allocated(matombasescap))   deallocate(matombasescap)
      if (allocated(matombasefcap))   deallocate(matombasefcap)
      if (allocated(mfcccord))        deallocate(mfcccord)
      if (allocated(mfcccordcap))     deallocate(mfcccordcap)
      if (allocated(mfccatomxiao))    deallocate(mfccatomxiao)
      if (allocated(mfccatomxiaocap)) deallocate(mfccatomxiaocap)
   end subroutine mfcc_dealloc_frag


   subroutine mfcc_dealloc_con()
      implicit none
      if (allocated(mfccatomcon))      deallocate(mfccatomcon)
      if (allocated(mfccchargecon))    deallocate(mfccchargecon)
      if (allocated(mfccatomcon2))     deallocate(mfccatomcon2)
      if (allocated(mfccchargecon2))   deallocate(mfccchargecon2)
      if (allocated(mfccatomconi))     deallocate(mfccatomconi)
      if (allocated(mfccchargeconi))   deallocate(mfccchargeconi)
      if (allocated(mfccatomconj))     deallocate(mfccatomconj)
      if (allocated(mfccchargeconj))   deallocate(mfccchargeconj)
      if (allocated(mfccstartcon))     deallocate(mfccstartcon)
      if (allocated(mfccfinalcon))     deallocate(mfccfinalcon)
      if (allocated(mfccbasescon))     deallocate(mfccbasescon)
      if (allocated(mfccbasefcon))     deallocate(mfccbasefcon)
      if (allocated(matomstartcon))    deallocate(matomstartcon)
      if (allocated(matomfinalcon))    deallocate(matomfinalcon)
      if (allocated(matombasescon))    deallocate(matombasescon)
      if (allocated(matombasefcon))    deallocate(matombasefcon)
      if (allocated(mfccstartcon2))    deallocate(mfccstartcon2)
      if (allocated(mfccfinalcon2))    deallocate(mfccfinalcon2)
      if (allocated(mfccbasescon2))    deallocate(mfccbasescon2)
      if (allocated(mfccbasefcon2))    deallocate(mfccbasefcon2)
      if (allocated(matomstartcon2))   deallocate(matomstartcon2)
      if (allocated(matomfinalcon2))   deallocate(matomfinalcon2)
      if (allocated(matombasescon2))   deallocate(matombasescon2)
      if (allocated(matombasefcon2))   deallocate(matombasefcon2)
      if (allocated(mfccstartconi))    deallocate(mfccstartconi)
      if (allocated(mfccfinalconi))    deallocate(mfccfinalconi)
      if (allocated(mfccbasesconi))    deallocate(mfccbasesconi)
      if (allocated(mfccbasefconi))    deallocate(mfccbasefconi)
      if (allocated(matomstartconi))   deallocate(matomstartconi)
      if (allocated(matomfinalconi))   deallocate(matomfinalconi)
      if (allocated(matombasesconi))   deallocate(matombasesconi)
      if (allocated(matombasefconi))   deallocate(matombasefconi)
      if (allocated(mfccstartconj))    deallocate(mfccstartconj)
      if (allocated(mfccfinalconj))    deallocate(mfccfinalconj)
      if (allocated(mfccbasesconj))    deallocate(mfccbasesconj)
      if (allocated(mfccbasefconj))    deallocate(mfccbasefconj)
      if (allocated(matomstartconj))   deallocate(matomstartconj)
      if (allocated(matomfinalconj))   deallocate(matomfinalconj)
      if (allocated(matombasesconj))   deallocate(matombasesconj)
      if (allocated(matombasefconj))   deallocate(matombasefconj)
      if (allocated(mfcccordcon))      deallocate(mfcccordcon)
      if (allocated(mfcccordcon2))     deallocate(mfcccordcon2)
      if (allocated(mfcccordconi))     deallocate(mfcccordconi)
      if (allocated(mfcccordconj))     deallocate(mfcccordconj)
      if (allocated(mfccatomxiaocon))  deallocate(mfccatomxiaocon)
      if (allocated(mfccatomxiaocon2)) deallocate(mfccatomxiaocon2)
      if (allocated(mfccatomxiaoconi)) deallocate(mfccatomxiaoconi)
      if (allocated(mfccatomxiaoconj)) deallocate(mfccatomxiaoconj)
   end subroutine mfcc_dealloc_con

end module quick_mfcc_module
