! Subroutine initiating the MFCC

subroutine mfcc(natomsaved)
   use quick_divcon_module, only: np
   use quick_files_module, only: iOutFile, iPDBFile, PDBFileName, iMfccXyzFile, mfccXyzFileName, &
                                 iMfccRptXyzFile, mfccRptXyzFileName, &
                                 iMfccChgFile, mfccChgFileName
   use quick_method_module, only: quick_method
   use quick_molspec_module, only: quick_molspec
   use quick_mfcc_species_module, only: mfcc_is_amino_acid, mfcc_species_charge, mfcc_species_kind, &
                                 mfcc_chg_read, mfcc_chg_lookup, mfcc_chg_n, mfcc_chg_name, &
                                 mfcc_chg_seq, mfcc_chg_val, mfcc_chg_used
   use quick_mfcc_module
   use quick_mfcc_repeat_module, only: mfcc_rpt_assign, mfcc_rpt_dealloc, mfcc_rpt_done, &
                                       mfcc_rpt_nrep, mfcc_rpt_repof
   use quick_mpi_module, only: master
!   use quick_method_module

   implicit none
   ! Sized from the residue count, not fixed at 100. The contact loops index it
   ! up to npmfcc+3, so any chain longer than 97 residues wrote past the end:
   ! Trypsin, at 223 residues, segfaulted here.
   integer, allocatable :: xiaoconnect(:,:)
   integer :: i,j,j1,j2,j3,number,mm,nn,kk
   integer :: mmm,nnn,nnnn,k,ii,jj
   integer :: ixiao,jxiao,kxiao
   double precision :: xiaodis   ! contact distance, was an integer and truncated
   character*6,allocatable:: sn(:)             ! series no.
   double precision,allocatable::coord(:,:)    ! cooridnates
   integer,allocatable::class(:),ttnumber(:)   ! class and residue number
   character*4,allocatable::atomname(:)        ! atom name
   character*3,allocatable::residue(:)         ! residue name
   integer natomsaved
   integer,allocatable::mselectC(:),mselectN(:),mselectCA(:)
   character*80 :: pdbline                     ! raw PDB record buffer
   integer :: ipdbstat                         ! iostat for PDB record reads
   character(len=2), external :: mfcc_element  ! element symbol from a pdb atom name
   integer :: mfccnatmax, mfccnat, mfccierr, mfccncon  ! MFCC array sizing
   integer :: nconskip                                ! contacts with a malformed span
   integer :: ierrxyz                          ! iostat for the fragment xyz dump
   integer :: nterm_h                          ! hydrogens on the N terminal nitrogen
   ! Residue classification: peptide chain versus everything else.
   integer, allocatable :: pepidx(:)           ! resSeq -> peptide fragment index, 0 if not peptide
   integer, allocatable :: xresof(:)           ! resSeq -> standalone fragment index, 0 if peptide
   integer, allocatable :: xfirst(:), xlast(:) ! atom range of each standalone fragment
   integer, allocatable :: xseq(:)             ! pdb resSeq of each standalone fragment
   character(len=3), allocatable :: xname(:)   ! residue name of each standalone fragment
   ! MFCCCHG charge overrides
   integer :: chgovr, ichg, ip, ifirst, nchgapp, chgtot, nhit
   ! Disulfide bridges. sspart(i) is the atom index of the sulfur that atom i is
   ! bonded to across an S-S bridge, or 0 when it has none.
   integer, allocatable :: sspart(:)
   integer :: nss, nsscap, iss, jss
   double precision :: ssd
   ! A disulfide is 2.03-2.08 Angstrom in a protein. 2.5 leaves room for a
   ! distorted or modestly refined structure while staying far below any
   ! non-bonded S..S contact, which does not come within 3 Angstrom.
   double precision, parameter :: SS_MAX = 2.5d0
   logical :: ovrfound, ovrbad
   character(len=3) :: ovrname
   integer :: maxres, ires2, npep, nxtra, pep_first, pep_last
   integer, allocatable :: natres(:)
   integer :: nsolv, nion, nlig, nunk, kx, nat_x
   logical :: chgunknown
   logical :: cterm_oxt                        ! C terminus carries OXT
   real(8)::xx,yy,zz,ym,zm
   integer :: mspin(50)

! integer :: kxiaoconnect
!   double precision :: mfcccord(:,:,:)
!   integer :: selectC(:), selectCA(:)
!   character*4,allocatable::mfccatomxiao(:,:)

   number=natomsaved ! avoid modification of important variable natomsaved

! Allocate arrays

   write(ioutfile,*) '=========== MFCC FRAGMENTATION OUTPUT ============'
   write(ioutfile,*) "MFCC started fragmentation"

   allocate(sn(number))
   allocate(coord(3,number))
   allocate(class(number))
   allocate(ttnumber(number))
   allocate(atomname(number))
   allocate(residue(number))

   allocate(mselectC(number))
   allocate(mselectCA(number))
   allocate(mselectN(number))

! Assign values of xiaoconnect to one
   ! xiaoconnect is allocated and initialised after npmfcc is known.

! Temporal files for fragmentation tests
!   open(20,file='initial.gjf')
!   open(40,file='number'//char(48+npmfcc/10) &
!   //char(48+npmfcc-npmfcc/10*10)//'.gjf')

! Read-in the PDB file
! Only ATOM/HETATM records are coordinate records. Skip everything else
! (COMPND, AUTHOR, REMARK, TER, CONECT, ...) so that PDB files written by
! common tools can be read as-is. Record i must still correspond to atom i
! of the input file.
   open(iPDBFile,file=PDBFileName)

   i=0
   do while (i.lt.number)
     read(iPDBFile,'(a80)',iostat=ipdbstat) pdbline
     if (ipdbstat.ne.0) exit
     if (pdbline(1:4).ne.'ATOM'.and.pdbline(1:6).ne.'HETATM') cycle
     i=i+1
     read(pdbline,100)sn(i),ttnumber(i),atomname(i),residue(i),class(i),(coord(j,i),j=1,3)
! Residue sequence number is columns 23-26 of a pdb ATOM record. Reading it as
! 3x,I3 skips column 23 and takes only 24-26, so anything past residue 999 comes
! back silently truncated to its last three digits: T4-Lysozyme's last water,
! residue 9382, was read as 382. Every residue above 999 then collapses onto a
! wrong, already-occupied residue index. 2x,I4 skips the blank and the chain id
! and reads the whole field. Files with 999 residues or fewer parse identically
! either way, since column 23 is blank for them.
100  format(a6,1x,I4,1x,a4,1x,a3,2x,I4,4x,3f8.3)
   enddo
   close(iPDBFile)

   if (i.ne.number) then
     call PrtErr(iOutFile,'PDB file does not contain one ATOM/HETATM record per atom of the input file.')
     call quick_exit(iOutFile,1)
   endif

   write(ioutfile,*) "MFCC processed PDB file"

! ---------------------------------------------------------------------
! Find the disulfide bridges.
!
! MFCC cuts the peptide backbone and caps each cut with a hydrogen, which is
! what makes a fragment a closed shell molecule. A disulfide is a second kind of
! bond through the chain, and nothing here knew about it: a CYX sulfur carries no
! hydrogen of its own because it is bonded to its partner, so a fragment holding
! one half of a bridge had a sulfur with an unsatisfied valence. On trypsin that
! left 60 of 458 sub-molecules with an odd electron count and the run stopped in
! the closed-shell guard. Pairs are found here and capped the same way the
! backbone cuts are, once the fragments exist.
! ---------------------------------------------------------------------
   allocate(sspart(number))
   sspart = 0
   nss = 0
   do i = 1, number
     if (mfcc_element(atomname(i)) .ne. 'S ') cycle
     if (sspart(i) .ne. 0) cycle
     do j = i+1, number
       if (mfcc_element(atomname(j)) .ne. 'S ') cycle
       if (sspart(j) .ne. 0) cycle
       ssd = dsqrt((coord(1,i)-coord(1,j))**2 + (coord(2,i)-coord(2,j))**2 &
                 + (coord(3,i)-coord(3,j))**2)
       if (ssd .le. SS_MAX) then
         sspart(i) = j
         sspart(j) = i
         nss = nss + 1
         write(ioutfile,'(" MFCC disulfide: atoms ",i7," and ",i7,"  (",a3,i5," - ",a3,i5,")  ", &
               &f6.3," A")') i,j,residue(i),class(i),residue(j),class(j),ssd
         exit
       endif
     enddo
   enddo
   if (nss .gt. 0) write(ioutfile,'(" MFCC found ",i0," disulfide bridge(s)")') nss

! Confirm reading of residue
! do i=1,number
!  write(*,*) residue(i)
! enddo

! Confirm reading of atomname
! do i=1,number
!  write(*,*) atomname(i)
! enddo

! Confirm reading of class
! do i=1,number
!  write(*,*) class(i)
! enddo

! ---------------------------------------------------------------------
! Classify residues, then renumber.
!
! npmfcc used to be class(number), the residue number of the last atom. In a
! prepared structure that is the last water molecule: 9382 for T4-Lysozyme,
! whose protein is 164 residues. The peptide chain has to be separated from the
! solvent, ions and ligands first, and the chain renumbered 1..npep so every
! loop below, which compares class against 1..npmfcc, keeps working untouched.
!
! Non-peptide residues become standalone fragments appended after the chain:
! one per water, one per ion, one per ligand, each with no caps because there
! is no bond to cut.
! ---------------------------------------------------------------------
  maxres = 0
  do i = 1, number
    maxres = max(maxres, class(i))
  enddo
  if (maxres .lt. 1) then
    call PrtErr(iOutFile,'No residue numbers were read from the pdb file.')
    call quick_exit(iOutFile,1)
  endif

  ! Charge overrides, if MFCCCHG was given. Read here because maxres is now
  ! known, so a residue number outside the structure can be rejected rather
  ! than used as an array index.
  if (quick_method%mfccchg) then
    mfccierr = 0
    call mfcc_chg_read(iMfccChgFile, mfccChgFileName, iOutFile, mfccierr)
    if (mfccierr /= 0) then
      call PrtErr(iOutFile,'Could not read the MFCC charge override file.')
      call quick_exit(iOutFile,1)
    endif
    write(ioutfile,'(" MFCC read ",i0," charge override(s) from ",a)') &
          mfcc_chg_n, trim(mfccChgFileName)
    do ichg = 1, mfcc_chg_n
      if (mfcc_chg_seq(ichg) .lt. 1 .or. mfcc_chg_seq(ichg) .gt. maxres) then
        call PrtErr(iOutFile,'A charge override names a residue number that is not in the pdb.')
        write(ioutfile,'("   ",a3,1x,i0,1x,i0,"   (pdb residue numbers run 1 ..",i0,")")') &
              mfcc_chg_name(ichg), mfcc_chg_seq(ichg), mfcc_chg_val(ichg), maxres
        call quick_exit(iOutFile,1)
      endif
    enddo
  endif

  allocate(pepidx(maxres), xresof(maxres))
  pepidx = 0
  xresof = 0

  ! Mark which residue numbers are present and whether each is a peptide
  ! residue. Classification is by residue name; see quick_mfcc_species_module
  ! for why it cannot be done by looking for backbone atom names.
  npep = 0
  nxtra = 0
  do ires2 = 1, maxres
    j = 0
    do i = 1, number
      if (class(i).eq.ires2) then
        j = i
        exit
      endif
    enddo
    if (j .eq. 0) cycle                      ! residue number not used
    if (mfcc_is_amino_acid(residue(j))) then
      npep = npep + 1
      pepidx(ires2) = npep
    else
      nxtra = nxtra + 1
      xresof(ires2) = nxtra
    endif
  enddo

  if (npep .lt. 2) then
    call PrtErr(iOutFile,'MFCC found fewer than two amino acid residues; there is no &
          &peptide chain to fragment.')
    call quick_exit(iOutFile,1)
  endif

  ! Standalone fragment atom ranges, and the peptide atom span.
  allocate(xfirst(max(nxtra,1)), xlast(max(nxtra,1)), xname(max(nxtra,1)), &
           xseq(max(nxtra,1)))
  xfirst = 0
  xlast = 0
  xname = '   '
  xseq = 0
  pep_first = 0
  pep_last = 0
  do i = 1, number
    if (pepidx(class(i)) .gt. 0) then
      if (pep_first .eq. 0) pep_first = i
      pep_last = i
    else
      kx = xresof(class(i))
      if (xfirst(kx) .eq. 0) xfirst(kx) = i
      xlast(kx) = i
      xname(kx) = residue(i)
      xseq(kx) = class(i)        ! still the pdb resSeq; renumbered below
    endif
  enddo

  ! The chain is cut as contiguous index ranges, so the peptide atoms have to
  ! form one unbroken block. In the files this was written for they do: the
  ! ligand and the solvent sit before and after it. A structure that interleaves
  ! them cannot be fragmented this way, and saying so beats producing fragments
  ! that quietly contain somebody else's atoms.
  do i = pep_first, pep_last
    if (pepidx(class(i)) .le. 0) then
      call PrtErr(iOutFile,'Peptide and non-peptide atoms are interleaved in this pdb. &
            &MFCC cuts fragments as contiguous atom ranges, so the chain must be one &
            &unbroken block of atoms.')
      write(ioutfile,'(" First offending atom: ",i8,"  residue ",a3)') i, residue(i)
      call quick_exit(iOutFile,1)
    endif
  enddo

  ! Renumber: peptide residues become 1..npep, everything else 0 so the chain
  ! loops skip it.
  do i = 1, number
    class(i) = pepidx(class(i))
  enddo

  npmfcc = npep
  nmfccextra = nxtra

  ! 0 marks a contact, so the array starts at 1 everywhere.
  allocate(xiaoconnect(npmfcc+3,npmfcc+3))
  xiaoconnect = 1

  nsolv = 0
  nion = 0
  nlig = 0
  do kx = 1, nxtra
    nat_x = xlast(kx)-xfirst(kx)+1
    select case (mfcc_species_kind(xname(kx), nat_x))
    case (1)
      nsolv = nsolv + 1
    case (2)
      nion = nion + 1
    case default
      nlig = nlig + 1
    end select
  enddo

  write(ioutfile,'(" MFCC residue classification")')
  write(ioutfile,'("   peptide residues   :",i7,"   (atoms ",i7," ..",i7,")")') &
        npep, pep_first, pep_last
  write(ioutfile,'("   solvent molecules  :",i7)') nsolv
  write(ioutfile,'("   monatomic ions     :",i7)') nion
  write(ioutfile,'("   other molecules    :",i7)') nlig

 write(ioutfile,*) 'Number of MFCC fragments', ' is ', npmfcc

! ---------------------------------------------------------------------
! Size the MFCC arrays from this system. They used to be fixed at 50
! fragments by 100 atoms, and npmfcc is taken straight from the residue
! number in the pdb with nothing checking it, so a 51 residue protein
! wrote past the end of every one of them and corrupted memory silently.
!
! A fragment never reaches past the residue on either side of its own, so
! the widest any of them can be is the largest run of three consecutive
! residues, plus the two capping hydrogens. Counting that from class needs
! nothing but the pdb and so can be done here, before anything is stored.
! ---------------------------------------------------------------------
  ! Atoms per peptide residue first, so the widest three-residue window is a
  ! sum rather than a rescan. The old form was a loop over residues times a loop
  ! over atoms, which on a solvated structure with the residue count taken from
  ! the last water was a hundred million iterations per pass.
  allocate(natres(npmfcc))
  natres = 0
  do i = 1, number
    if (class(i) .gt. 0) natres(class(i)) = natres(class(i)) + 1
  enddo

  mfccnatmax = 0
  do i = 1, npmfcc
    mfccnat = natres(i)
    if (i .gt. 1)      mfccnat = mfccnat + natres(i-1)
    if (i .lt. npmfcc) mfccnat = mfccnat + natres(i+1)
    mfccnatmax = max(mfccnatmax, mfccnat)
  enddo
  ! +2 for the backbone caps, and room for the disulfide caps: a fragment spans
  ! three residues and a cap two, so a handful covers any realistic case. The
  ! capping pass below checks the bound rather than trusting this.
  mfccnatmax = mfccnatmax + 2
  if (nss .gt. 0) mfccnatmax = mfccnatmax + 6
  deallocate(natres)

  ! A standalone fragment has to fit in the same arrays.
  do kx = 1, nxtra
    mfccnatmax = max(mfccnatmax, xlast(kx)-xfirst(kx)+1)
  enddo

  mfccierr = 0
  ! Any RPT grouping from a previous fragmentation describes the arrays that are
  ! about to be replaced, so throw it away here. It matters when QUICK is driven
  ! through the API and runs several molecules in one process: the module state
  ! is saved, and a stale mfcc_rpt_done would make the next run reuse the
  ! previous molecule's representatives and index them against a different
  ! nmfccextra.
  call mfcc_rpt_dealloc()
  ! The cap arrays share this dimension with the fragment arrays. Backbone
  ! concaps take npmfcc-1 slots, and the disulfide correction appends two thiol
  ! concaps per bridge, so the room for those has to be asked for here.
  call mfcc_alloc_frag(npmfcc+nxtra+2*nss, mfccnatmax, mfccierr)
  if (mfccierr /= 0) then
    call PrtErr(iOutFile,'Could not allocate the MFCC fragment arrays.')
    call quick_exit(iOutFile,1)
  endif

  write(ioutfile,'(" MFCC arrays sized for ",i6," chain fragments +",i7, &
        &" standalone, ",i6," atoms per fragment")') npmfcc, nxtra, mfccnatmax

!  write(*,*) "Assigned number of fragments"

! Assign zero values for initialization of MFCC
! Make multiplicity equal to one for all fragments

  do 999 i=1,number
999    continue

   do i=1,npmfcc
    mfcccharge(i)=0
    mfccatom(i)=0
    mfccchargecap(i)=0
    mfccatomcap(i)=0
    mspin(i)=1
  enddo

! write(*,*) "Initialiazed MFCC arguments"

! Identify C, N, and CA atoms
   j1=1
   j2=1
   j3=1
   ! Peptide atoms only. Benzamidine in the Trypsin structure has an atom whose
   ! name field is exactly ' C  ', so an unfiltered scan picks it up as a
   ! backbone carbonyl and every fragment boundary after it is wrong. class is
   ! zero for everything that is not part of the chain.
   do i=1,number
   if(class(i).le.0) cycle
   if(atomname(i).eq.' C  ')then
     mselectC(j1)=i
!     write(*,*) mselectC(j1), "C"
     j1=j1+1
   endif
   if(atomname(i).eq.' N  ')then
     mselectN(j2)=i
!     write(*,*) mselectN(j2), "N"
     j2=j2+1
   endif
   if(atomname(i).eq.' CA ')then
     mselectCA(j3)=i
!     write(*,*) mselectCA(j3), "CA"
     j3=j3+1
   endif
  enddo

! write(*,*) "Identified C, N, CA"

! Start assigning coordinates to MFCC fragments

 write(ioutfile,*) 'Second C and CA atoms are:', mselectC(2), &
   ' and ', mselectCA(2) 

   mm=mselectC(2)
   nn=mselectCA(2)

! write(*,*) mm,nn, "mm and nn values"

 write(ioutfile,*) '======================================'
 write(ioutfile,*) 'MFCC fragment #1'
 write(ioutfile,*) '  '

  ! From pep_first, not from atom 1, and stored at kk-pep_first+1 so the local
  ! slots line up with matomstart(1) below. Copying from atom 1 pulled whatever
  ! was numbered ahead of the protein into fragment 1: in the T4-Lysozyme
  ! structure that is the 24-atom ligand, which ended up sharing a fragment with
  ! the first two residues. pep_first is 1 for a bare peptide, so this is the
  ! same arithmetic as before for files with nothing in front of the chain.
  do kk=pep_first,mm-1
 write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
      mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
      mfccatomxiao(kk-pep_first+1,1)=mfcc_element(atomname(kk))
      do j=1,3
        mfcccord(j,kk-pep_first+1,1)=coord(j,kk)
      enddo
  enddo

! Initiate xyzchange

 call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
   coord(1,nn),coord(2,nn),coord(3,nn),xx,yy,zz)     

! write(*,*) "first call xyzchange output" 
 write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,yy,zz
! write(ioutfile,*) '  '
! write(ioutfile,*) 'End of MFCC fragment #1'
! write(ioutfile,*) '======================================'
! write(ioutfile,*) '  '

 mfccatomxiao(mm-pep_first+1,1)='H '

 mfcccord(1,mm-pep_first+1,1)=xx
 mfcccord(2,mm-pep_first+1,1)=yy
 mfcccord(3,mm-pep_first+1,1)=zz

 mfccatom(1)=mm-pep_first+1

 mfccstart(1)=1
 mfccfinal(1)=mm-pep_first

! The chain starts at pep_first, not at atom 1: a ligand numbered ahead of the
! protein puts its atoms first, as LIG does in the T4-Lysozyme structure.
 matomstart(1)=pep_first
 matomfinal(1)=mm-1

! write(ioutfile,*) '  '
! write(ioutfile,*) '======================================'
! write(ioutfile,*) 'START FIRST MFCC LOOP'
! write(ioutfile,*) '======================================' 
! write(ioutfile,*) '  '

  do k=2,npmfcc-1

 write(ioutfile,*) '======================================'
 write(ioutfile,*) 'MFCC fragment #', k
 write(ioutfile,*) '  '

! Another temporal file to debug MFCC fragmentation
!  open(30,file='number'//char(48+k/10) &
!  //char(48+k-k/10*10)//'.gjf')
 
   mm=mselectN(k-1)
   nn=mselectC(k+1)
   mmm=mselectCA(k-1)
   nnn=mselectCA(k+1)
   ! nnnn is only needed by the proline branch below. The loop starts at k=2,
   ! so mselectC(k-2) reads element 0 on the first iteration; guard the read
   ! rather than run off the start of the array.
   nnnn=0
   if (k.ge.3) nnnn=mselectC(k-2)
   if(residue(mselectN(k-1)).ne.'PRO')then
    call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
    coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)    
!    write(*,*) 'second call xyzchange output'
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

    mfccatomxiao(1,k)='H '

    mfcccord(1,1,k)=xx
    mfcccord(2,1,k)=ym
    mfcccord(3,1,k)=zm

    mfccatom(k)=nn-mmm+1+1

    mfccstart(k)=2
    mfccfinal(k)=nn-mmm+1

    matomstart(k)=mmm
    matomfinal(k)=nn-1

    do kk=mmm,nn-1
      write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
      mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
      mfccatomxiao(kk-mmm+2,k)=mfcc_element(atomname(kk))
      do j=1,3
        mfcccord(j,kk-mmm+2,k)=coord(j,kk)
      enddo
    enddo

   call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
!   write(*,*) 'third call xyzchange output'
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

   mfccatomxiao(nn-mmm+2,k)='H '

   mfcccord(1,nn-mmm+2,k)=xx
   mfcccord(2,nn-mmm+2,k)=ym
   mfcccord(3,nn-mmm+2,k)=zm

!  write(ioutfile,*) '  '
!  write(ioutfile,*) 'End of MFCC fragment #', k
!  write(ioutfile,*) '======================================'
!  write(ioutfile,*) '  '      
    else

   ! A proline this early in the chain would need the carbonyl carbon of a
   ! residue that does not exist. Say so instead of using a bogus index.
   if (nnnn.le.0) then
     call PrtErr(iOutFile,'MFCC cannot cap a proline at this chain position: it requires the &
           &carbonyl carbon of a preceding residue that does not exist.')
     call quick_exit(iOutFile,1)
   endif

   call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

   mfccatomxiao(1,k)='H '

   mfcccord(1,1,k)=xx
   mfcccord(2,1,k)=ym
   mfcccord(3,1,k)=zm

   mfccatom(k)=nn-mm+1+1

   mfccstart(k)=2
   mfccfinal(k)=nn-mm+1

   matomstart(k)=mm
   matomfinal(k)=nn-1

   do kk=mm,nn-1
      write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
      mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)

      mfccatomxiao(kk-mm+2,k)=mfcc_element(atomname(kk))
      do j=1,3
        mfcccord(j,kk-mm+2,k)=coord(j,kk)
      enddo

   enddo

   call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
   write(*,*) 'PROline call xyzchange output'
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

   mfccatomxiao(nn-mm+2,k)='H '

   mfcccord(1,nn-mm+2,k)=xx
   mfcccord(2,nn-mm+2,k)=ym
   mfcccord(3,nn-mm+2,k)=zm

!  write(ioutfile,*) '  '
!  write(ioutfile,*) 'End of MFCC fragment #', k
!  write(ioutfile,*) '======================================'
!  write(ioutfile,*) '  '

  endif
  enddo

! Start the second fragmentation cycle over the coordinates
! write(ioutfile,*) '  '
! write(ioutfile,*) '======================================'
! write(ioutfile,*) 'MFCC PRINT FOR LAST FRAGMENT'
! write(ioutfile,*) '======================================'
! write(ioutfile,*) '  '

 write(ioutfile,*) '======================================'
 write(ioutfile,*) 'MFCC fragment #', npmfcc
 write(ioutfile,*) '  '

  mm=mselectN(npmfcc-1)
  mmm=mselectCA(npmfcc-1)
  nnnn=mselectC(npmfcc-2)
  if(residue(mselectN(npmfcc-1)).ne.'PRO')then
  call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
  coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)    
!  write(*,*) "first call xyzchange in 2nd loop"
  write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

  mfccatomxiao(1,npmfcc)='H '
  
  mfcccord(1,1,npmfcc)=xx
  mfcccord(2,1,npmfcc)=ym
  mfcccord(3,1,npmfcc)=zm
  
  mfccatom(npmfcc)=pep_last-mmm+1+1
  
  mfccstart(npmfcc)=2
  mfccfinal(npmfcc)=pep_last-mmm+2
  
  matomstart(npmfcc)=mmm
  ! Ends at the last peptide atom. 'number' would swallow every solvent
  ! molecule after the chain into the final fragment.
  matomfinal(npmfcc)=pep_last

  do kk=mmm,pep_last
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
     mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)

     mfccatomxiao(kk-mmm+2,npmfcc)=mfcc_element(atomname(kk))
     do j=1,3
       mfcccord(j,kk-mmm+2,npmfcc)=coord(j,kk)
     enddo

   enddo
  else
  call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

    mfccatomxiao(1,npmfcc)='H '

    mfcccord(1,1,npmfcc)=xx
    mfcccord(2,1,npmfcc)=ym
    mfcccord(3,1,npmfcc)=zm

    mfccatom(npmfcc)=pep_last-mm+1+1

   mfccstart(npmfcc)=2
   ! This branch spans mm (the nitrogen), not mmm (the alpha carbon): the atoms
   ! stored below run kk=mm..number at local index kk-mm+2, and matomstart is mm.
   ! Reading number-mmm+2 here dropped every atom between N and CA of the
   ! preceding residue from the fragment's basis range, so those basis functions
   ! received no density at all. A proline ring puts CD, CG and CB in exactly
   ! that gap, which on Trp-cage left 34 of 919 basis functions empty and the
   ! guess 47 electrons short. Only the proline branch is affected, and only for
   ! the final fragment, so a system without a proline near the C terminus never
   ! shows it.
   mfccfinal(npmfcc)=pep_last-mm+2

   matomstart(npmfcc)=mm
   ! Ends at the last peptide atom. 'number' would swallow every solvent
   ! molecule after the chain into the final fragment.
   matomfinal(npmfcc)=pep_last

   do kk=mm,pep_last
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
     mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)

    mfccatomxiao(kk-mm+2,npmfcc)=mfcc_element(atomname(kk))
    do j=1,3
      mfcccord(j,kk-mm+2,npmfcc)=coord(j,kk)
    enddo

  enddo
  endif     

 write(ioutfile,*) '  '
 write(ioutfile,*) '======================================'
 write(ioutfile,*) 'MFCC PRINT FOR CAPS'
 write(ioutfile,*) '======================================'
 write(ioutfile,*) '  '

! Start loop over caps

   do k=1,npmfcc-1

! Temporal file for debug of caps
!   open(60,file='cap'//char(48+k/10) &
!   //char(48+k-k/10*10)//'.gjf')

 write(ioutfile,*) '======================================'
 write(ioutfile,*) 'MFCC cap #', k
 write(ioutfile,*) '  '

   mm=mselectN(k)
   nn=mselectC(k+1)
   mmm=mselectCA(k)
   nnn=mselectCA(k+1)
   ! Same guard as in the fragment loop: this cap loop starts at k=1, so
   ! mselectC(k-1) reads element 0 on the first iteration.
   nnnn=0
   if (k.ge.2) nnnn=mselectC(k-1)
   if(residue(mselectN(k)).ne.'PRO')then
    call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
    coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)       
!   write(*,*) '1st call for xyzchange in caps loop'
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

   mfccatomxiaocap(1,k)='H '

   mfcccordcap(1,1,k)=xx
   mfcccordcap(2,1,k)=ym
   mfcccordcap(3,1,k)=zm

   mfccatomcap(k)=nn-mmm+1+1

   mfccstartcap(k)=2
   mfccfinalcap(k)=nn-mmm+1

   matomstartcap(k)=mmm
   matomfinalcap(k)=nn-1

   do kk=mmm,nn-1
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
    mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
    mfccatomxiaocap(kk-mmm+2,k)=mfcc_element(atomname(kk))
    do j=1,3
      mfcccordcap(j,kk-mmm+2,k)=coord(j,kk)
    enddo

 enddo

 call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
  coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
!  write(*,*) '2nd xyzchange call for caps'
  write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

        mfccatomxiaocap(nn-mmm+2,k)='H '

        mfcccordcap(1,nn-mmm+2,k)=xx
        mfcccordcap(2,nn-mmm+2,k)=ym
        mfcccordcap(3,nn-mmm+2,k)=zm

    else

   if (nnnn.le.0) then
     call PrtErr(iOutFile,'MFCC cannot cap a proline at this chain position: it requires the &
           &carbonyl carbon of a preceding residue that does not exist.')
     call quick_exit(iOutFile,1)
   endif

  call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(*,*) 'nxyzchange call for caps if PROline present'
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

       mfccatomxiaocap(1,k)='H '

       mfcccordcap(1,1,k)=xx
       mfcccordcap(2,1,k)=ym
       mfcccordcap(3,1,k)=zm

       mfccatomcap(k)=nn-mm+1+1

      mfccstartcap(k)=2
      mfccfinalcap(k)=nn-mm+1

      matomstartcap(k)=mm
      matomfinalcap(k)=nn-1

     do kk=mm,nn-1
       write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
       mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
       mfccatomxiaocap(kk-mm+2,k)=mfcc_element(atomname(kk))
       do j=1,3
         mfcccordcap(j,kk-mm+2,k)=coord(j,kk)
       enddo

   enddo

  call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
    write(*,*) 'PROline xyzchange call for caps'
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm
       mfccatomxiaocap(nn-mm+2,k)='H '

       mfcccordcap(1,nn-mm+2,k)=xx
       mfcccordcap(2,nn-mm+2,k)=ym
       mfcccordcap(3,nn-mm+2,k)=zm

   endif   

 enddo

! ---------------------------------------------------------------------
! Standalone fragments: one per solvent molecule, ion or ligand.
!
! Nothing was cut to make these, so they carry no capping hydrogen and no cap
! partner: the local range is the whole residue, starting at 1 rather than 2.
! They occupy slots npmfcc+1 .. npmfcc+nmfccextra, after the chain, which is
! why the cap arrays stay at npmfcc-1 while the fragment loops run further.
! ---------------------------------------------------------------------
  nunk = 0
  do kx = 1, nxtra
    k = npmfcc + kx
    nat_x = xlast(kx)-xfirst(kx)+1

    mfccatom(k)    = nat_x
    mfccstart(k)   = 1
    mfccfinal(k)   = nat_x
    matomstart(k)  = xfirst(kx)
    matomfinal(k)  = xlast(kx)
    mfcccharge(k)  = mfcc_species_charge(xname(kx), nat_x, chgunknown)
    ! A user-supplied charge replaces whatever the name-based table said, and
    ! settles the unknown case outright: that is the whole point of the file.
    if (quick_method%mfccchg) then
      call mfcc_chg_lookup(xseq(kx), xname(kx), chgovr, ovrfound, ovrbad, ovrname)
      if (ovrfound) then
        if (ovrbad) then
          call PrtErr(iOutFile,'A charge override names the wrong residue for that number.')
          write(ioutfile,'("   the file says residue ",i0," is ",a3,", the pdb says it is ",a3)') &
                xseq(kx), ovrname, xname(kx)
          call quick_exit(iOutFile,1)
        endif
        mfcccharge(k) = chgovr
        chgunknown = .false.
      endif
    endif
    if (chgunknown) nunk = nunk + 1

    do kk = xfirst(kx), xlast(kx)
      mfccatomxiao(kk-xfirst(kx)+1,k) = mfcc_element(atomname(kk))
      do j = 1, 3
        mfcccord(j,kk-xfirst(kx)+1,k) = coord(j,kk)
      enddo
    enddo
  enddo

! ---------------------------------------------------------------------
! Cap the cut disulfides.
!
! Every fragment, cap and standalone sub-molecule now exists, and each one knows
! the global atom range it was cut from. A sulfur inside that range whose bridge
! partner lies outside it has lost the bond that satisfied its valence, so it
! gets a hydrogen placed along the direction of the partner it can no longer
! see. That is exactly what the backbone cuts already do, which is why the
! capping hydrogen is appended past mfccfinal: like a backbone cap it belongs to
! the sub-molecule's chemistry but not to the density block harvested from it.
!
! Both halves of a bridge get their own hydrogen when the bridge is cut, so the
! two fragments are a pair of thiols rather than a pair of radicals. That keeps
! every sub-molecule closed shell, which is what the cap is for, but on its own
! it would leave the guess describing a disulfide as two thiols -- the sulfur is
! in the wrong oxidation state in both. mfcc_ss_blocks below puts the bridge
! back as an MFCC correction, so these caps are the local valence fix only.
! ---------------------------------------------------------------------
  nmfcccap = max(npmfcc-1,0)
  nmfccss  = 0

  if (nss .gt. 0) then
    nsscap = 0
    call mfcc_ss_cap(npmfcc+nxtra, mfccnatmax, mfccatom, mfcccord, mfccatomxiao, &
          matomstart, matomfinal, number, coord, sspart, nsscap, mfccierr)
    if (mfccierr .eq. 0) &
      call mfcc_ss_cap(max(nmfcccap,1), mfccnatmax, mfccatomcap, mfcccordcap, &
            mfccatomxiaocap, matomstartcap, matomfinalcap, number, coord, sspart, &
            nsscap, mfccierr)
    if (mfccierr /= 0) then
      call PrtErr(iOutFile,'A sub-molecule ran out of room for its disulfide capping hydrogens.')
      call quick_exit(iOutFile,1)
    endif
    write(ioutfile,'(" MFCC capped ",i0," cut disulfide bond(s) with hydrogen")') nsscap

! ---------------------------------------------------------------------
! Restore the disulfide bonds.
!
! The capping above satisfies the valence but not the chemistry. MFCC's own
! answer to a bond that must not be broken is not to cap it: it moves the cut to
! a neighbouring single bond, keeps the bond whole inside the sub-molecule, and
! subtracts what that double counts. For the backbone the cut moves from the
! peptide C-N to the CA-C bond and the overlap comes back as a concap.
!
! The same construction for a disulfide moves the cut from S-S to the two CA-CB
! bonds. Per bridge:
!
!   D += D(CH3-S-S-CH3) - D(CH3-S-H)_i - D(CH3-S-H)_j
!
! The first term holds the intact S-S at the real geometry, with each partner
! truncated to a methyl; the two thiols are exactly what the capped fragments
! already put there, so they are removed. Both are closed shell: 50 electrons
! for the disulfide, 26 for each thiol.
!
! The thiols are ordinary concaps and go in the cap arrays, which is the whole
! reason the cap count is now a variable. The disulfide block is the one piece
! of MFCC whose atoms do not form a single run of global basis functions, so it
! is kept apart and carries the global span of each half.
! ---------------------------------------------------------------------
    call mfcc_alloc_ss(nss, mfccnatmax, mfccierr)
    if (mfccierr /= 0) then
      call PrtErr(iOutFile,'Could not allocate the MFCC disulfide correction arrays.')
      call quick_exit(iOutFile,1)
    endif

    call mfcc_ss_blocks(number, coord, atomname, class, sspart, mfccnatmax, &
          npmfcc+nxtra+2*nss, nmfcccap, nmfccss, mfccierr)
    if (mfccierr /= 0) then
      call PrtErr(iOutFile,'Could not build the MFCC disulfide correction blocks.')
      call quick_exit(iOutFile,1)
    endif

    if (nmfccss .gt. 0) then
      write(ioutfile,'(" MFCC disulfide correction: ",i0," intact S-S block(s) added, ", &
            &i0," thiol concap(s) subtracted")') nmfccss, 2*nmfccss
    endif
    if (nmfccss .lt. nss) then
      call PrtWrn(iOutFile,'Some disulfide bridges were left as plain hydrogen caps.')
      write(ioutfile,'("|          ",i0," of ",i0," bridge(s) could not be turned into an")') &
            nss-nmfccss, nss
      write(ioutfile,'("|          intact S-S correction block, because CB or CA was missing")')
      write(ioutfile,'("|          from the cysteine, or the atoms between CB and SG are not")')
      write(ioutfile,'("|          that residue''s own hydrogens. Those bridges keep the older")')
      write(ioutfile,'("|          treatment: both halves capped with hydrogen as thiols.")')
      write(ioutfile,'(a)')
    endif
  endif

  if (nunk .gt. 0) then
    call PrtWrn(iOutFile,'Some non-peptide residues have an unrecognised charge.')
    write(ioutfile,'("|          ",i7," residue(s) were given charge 0 because their name is")') nunk
    write(ioutfile,'("|          not in the table in quick_mfcc_species_module. If one of them")')
    write(ioutfile,'("|          is actually charged, its fragment is wrong; a charged species")')
    write(ioutfile,'("|          with an odd electron count will be caught as an open shell")')
    write(ioutfile,'("|          later, but an even one will pass silently. Check these:")')
    do kx = 1, nxtra
      nat_x = xlast(kx)-xfirst(kx)+1
      if (mfcc_species_charge(xname(kx), nat_x, chgunknown) .eq. 0 .and. chgunknown) then
        ! Silent for anything the user gave a charge for; mfcccharge already
        ! holds it and there is nothing left to warn about.
        if (quick_method%mfccchg) then
          call mfcc_chg_lookup(xseq(kx), xname(kx), chgovr, ovrfound, ovrbad, ovrname)
          if (ovrfound) cycle
        endif
        write(ioutfile,'("|            ",a3,"  (",i6," atoms)")') xname(kx), nat_x
      endif
    enddo
    write(ioutfile,'(a)')
    call flush(ioutfile)
  endif

! Start the final loop, which concerns neutral terminus.

  ! One pass over pairs of peptide atoms, rather than rescanning every atom
  ! pair for every pair of residues. The original form was four nested loops,
  ! residues squared times atoms squared, which on a solvated protein is around
  ! 10^12 iterations; this is the peptide atom count squared, and solvent is
  ! skipped outright since it is not part of the chain.
  !
  ! The residue separation test is unchanged: the first residue is at least 2
  ! and the second at least three further along, so only non-sequential
  ! contacts count.
  do ii=1,number
    if(class(ii).lt.2) cycle
    do jj=ii+1,number
      if(class(jj).lt.class(ii)+3) cycle
      xiaodis=dsqrt((coord(1,ii)-coord(1,jj))**2.0d0+ &
                    (coord(2,ii)-coord(2,jj))**2.0d0+ &
                    (coord(3,ii)-coord(3,jj))**2.0d0)
      ! This used to read 'xiaodis .le. -1.0d0'. A Euclidean distance is
      ! never negative, so the connection terms could never be generated;
      ! the whole con/coni/conj layer below was unreachable. The cutoff is
      ! now a real contact distance, adjustable with the MFCCCUT keyword.
      if(xiaodis.le.quick_method%MFCCCUT)then
        xiaoconnect(class(ii),class(jj))=0
        if(quick_method%debug) print*,class(ii),class(jj),ii,jj, 'ixiao,jxiao,ii,jj'
      endif
    enddo
  enddo

  kxiao=1
  nconskip=0

! ---------------------------------------------------------------------
! Size the connection arrays now that the contact search has run. How many
! there are is not known any earlier, and on a folded protein it can far
! exceed the fragment count: Trp-cage alone finds 24 contacts inside 3 A
! across 20 residues. These were fixed at 50 as well.
! ---------------------------------------------------------------------
  mfccncon = 0
  do i = 2, npmfcc
    do jj = i+3, npmfcc+3
      if (xiaoconnect(i,jj).eq.0) mfccncon = mfccncon + 1
    enddo
  enddo

  mfccierr = 0
  call mfcc_alloc_con(mfccncon, mfccnatmax, mfccierr)
  if (mfccierr /= 0) then
    call PrtErr(iOutFile,'Could not allocate the MFCC connection arrays.')
    call quick_exit(iOutFile,1)
  endif
  write(ioutfile,'(" MFCC connection arrays sized for ",i6," contacts")') mfccncon

  write(ioutfile,*) '======================================'
  write(ioutfile,*) 'MFCC checked for neutral terminus'

!-----------------------------------------------------------------------
! Assign fragment charges from the terminus composition.
!
! A peptide read from a pdb is normally a zwitterion: the N terminal
! nitrogen carries three hydrogens (NH3+) and the C terminal carboxylate
! carries OXT (COO-). The molecule is neutral overall, but a fragment that
! contains only one charged terminus is an ion. Treating it as neutral gives
! an odd electron count and the fragment SCF cannot converge as closed shell.
!
! Only these two cases are recognised. Any other charged group (Lys, Arg,
! Asp, Glu, a bound ion, a non standard terminus) is NOT detected and its
! fragment will still be treated as neutral.
!-----------------------------------------------------------------------
  ! Formal charges are localised on one representative atom per charged group,
  ! then summed over the atom range each fragment and cap actually spans. That
  ! matters because the spans overlap: an internal side chain sits in fragment
  ! k, in fragment k+1 and in cap k, and the MFCC sum is fragments minus caps,
  ! so charging all three leaves the right total (-1 -1 +1 = -1). Assigning per
  ! residue instead would count it twice.
  !
  ! Protonation is read from the geometry, by counting hydrogens bonded to the
  ! nitrogen or oxygen of the group, so it does not depend on how the file
  ! names its hydrogens. Only the residue name is trusted, which every pdb
  ! writer gets right.
  block
    integer, allocatable :: atomchg(:)
    integer :: ires, nh, irep, nsc, ia
    character(len=3) :: rnm
    character(len=2), external :: mfcc_element
    double precision :: dd
    ! Generous X-H covalent cutoff: the longest is about 1.09 A (C-H) and the
    ! shortest non bonded contact is well above 1.5 A.
    double precision, parameter :: HB = 1.35d0

    allocate(atomchg(number))
    atomchg = 0

    do ires = 1, npmfcc
      rnm = residue(minloc(class, dim=1, mask=(class.eq.ires)))

      ! --- side chain amines: Lys NZ, Arg guanidinium, His ring ---
      nh = 0
      nsc = 0
      irep = 0
      do i = 1, number
        if (class(i).ne.ires) cycle
        if (mfcc_element(atomname(i)).ne.'N ') cycle
        if (atomname(i).eq.' N  ') cycle          ! backbone amide
        nsc = nsc + 1
        if (irep.eq.0) irep = i
        do j = 1, number
          if (mfcc_element(atomname(j)).ne.'H ') cycle
          dd = dsqrt((coord(1,i)-coord(1,j))**2 + (coord(2,i)-coord(2,j))**2 &
                   + (coord(3,i)-coord(3,j))**2)
          if (dd.le.HB) nh = nh + 1
        enddo
      enddo
      if (irep.gt.0) then
        ! Neutral reference counts: Lys NH2 2, Arg guanidine 4, His ring 1.
        ! AMBER writes the protonation state into the residue name, so the
        ! charged and neutral forms have different names and both have to be
        ! listed. LYN is neutral lysine, HID and HIE the two neutral histidine
        ! tautomers, HIP the protonated one. The hydrogen count still decides,
        ! so a name that disagrees with the geometry loses to the geometry.
        if ((rnm.eq.'LYS'.or.rnm.eq.'LYN') .and. nh.ge.3) atomchg(irep) = 1
        if ((rnm.eq.'ARG'.or.rnm.eq.'ARN') .and. nh.ge.5) atomchg(irep) = 1
        if ((rnm.eq.'HIS'.or.rnm.eq.'HIP'.or.rnm.eq.'HID'.or.rnm.eq.'HIE') &
             .and. nh.ge.2) atomchg(irep) = 1
      endif

      ! --- side chain carboxylates: Asp, Glu ---
      ! ASH and GLH are the protonated (neutral) acids; they are listed so the
      ! hydrogen count is actually examined rather than the residue skipped.
      ! CYM is deprotonated cysteine, whose SG carries the charge.
      if (rnm.eq.'ASP' .or. rnm.eq.'GLU' .or. rnm.eq.'ASH' .or. rnm.eq.'GLH') then
        nh = 0
        irep = 0
        do i = 1, number
          if (class(i).ne.ires) cycle
          if (mfcc_element(atomname(i)).ne.'O ') cycle
          if (atomname(i).eq.' O  ' .or. trim(adjustl(atomname(i))).eq.'OXT') cycle
          if (irep.eq.0) irep = i
          do j = 1, number
            if (mfcc_element(atomname(j)).ne.'H ') cycle
            dd = dsqrt((coord(1,i)-coord(1,j))**2 + (coord(2,i)-coord(2,j))**2 &
                     + (coord(3,i)-coord(3,j))**2)
            if (dd.le.HB) nh = nh + 1
          enddo
        enddo
        if (irep.gt.0 .and. nh.eq.0) atomchg(irep) = -1
      endif
    enddo

    ! --- termini ---
    ! Count only the hydrogens ON the terminal nitrogen, not every hydrogen in
    ! residue 1: counting all of them also picks up the HA hydrogens on CA,
    ! which would make a neutral NH2 terminus look protonated.
    nterm_h = 0
    irep = 0
    do i = 1, number
      if (class(i).ne.1 .or. atomname(i).ne.' N  ') cycle
      irep = i
      do j = 1, number
        if (mfcc_element(atomname(j)).ne.'H ') cycle
        dd = dsqrt((coord(1,i)-coord(1,j))**2 + (coord(2,i)-coord(2,j))**2 &
                 + (coord(3,i)-coord(3,j))**2)
        if (dd.le.HB) nterm_h = nterm_h + 1
      enddo
    enddo
    if (irep.gt.0 .and. nterm_h.ge.3) atomchg(irep) = 1

    cterm_oxt = .false.
    do i = 1, number
      if (class(i).ne.npmfcc) cycle
      if (trim(adjustl(atomname(i))).ne.'OXT') cycle
      nh = 0
      do j = 1, number
        if (mfcc_element(atomname(j)).ne.'H ') cycle
        dd = dsqrt((coord(1,i)-coord(1,j))**2 + (coord(2,i)-coord(2,j))**2 &
                 + (coord(3,i)-coord(3,j))**2)
        if (dd.le.HB) nh = nh + 1
      enddo
      if (nh.eq.0) then
        cterm_oxt = .true.
        atomchg(i) = -1
      endif
    enddo

    ! --- user-supplied overrides for peptide residues ---
    !
    ! A peptide residue's atoms are split between a fragment and its caps, so
    ! the charge is placed on one atom and the span sums below carry it into
    ! whichever sub-molecule owns that atom. This is how the automatic
    ! detections already work (the C-terminal -1 sits on OXT), so an override
    ! needs no separate machinery.
    !
    ! Any detected charge inside the residue is cleared first. The keyword is
    ! an override, not an addition: 'GLU 46 0' has to be able to neutralise a
    ! carboxylate the detection pass found, and without the clear it would
    ! leave the -1 in place and silently do nothing.
    nchgapp = 0
    if (quick_method%mfccchg) then
      do ichg = 1, mfcc_chg_n
        ip = pepidx(mfcc_chg_seq(ichg))
        if (ip .le. 0) cycle                     ! not a peptide residue
        ifirst = 0
        do i = 1, number
          if (class(i) .ne. ip) cycle
          if (ifirst .eq. 0) ifirst = i
          atomchg(i) = 0
        enddo
        if (ifirst .eq. 0) cycle
        ! The name is checked against the structure, not trusted, so a typo in
        ! the number cannot quietly charge a different residue.
        if (mfcc_chg_name(ichg) .ne. adjustl(residue(ifirst))) then
          call PrtErr(iOutFile,'A charge override names the wrong residue for that number.')
          write(ioutfile,'("   the file says residue ",i0," is ",a3,", the pdb says it is ",a3)') &
                mfcc_chg_seq(ichg), mfcc_chg_name(ichg), adjustl(residue(ifirst))
          call quick_exit(iOutFile,1)
        endif
        atomchg(ifirst) = mfcc_chg_val(ichg)
        mfcc_chg_used(ichg) = .true.
        nchgapp = nchgapp + 1

        ! Say which sub-molecule ends up carrying it. The atom chosen above is
        ! the residue's first, and which fragment or cap owns that atom follows
        ! from where the chain was cut, not from anything the user controls, so
        ! printing it is the difference between a placement they can check and
        ! one they have to reverse-engineer.
        ! A residue in the overlap between two fragments shows up in both of
        ! them and in the cap that corrects for the overlap, so the charge is
        ! listed three times here. That is not triple counting: the MFCC sum is
        ! fragments minus caps, so +1 +1 -1 leaves the charge once. Each
        ! sub-molecule really does contain the charged group and really is an
        ! ion, which is why each must be told about it.
        nhit = 0
        do ires = 1, npmfcc
          if (ifirst .ge. matomstart(ires) .and. ifirst .le. matomfinal(ires)) then
            nhit = nhit + 1
            write(ioutfile,'("   residue ",i0," (",a3,") charge ",i0," -> fragment ",i0)') &
                  mfcc_chg_seq(ichg), adjustl(residue(ifirst)), mfcc_chg_val(ichg), ires
          endif
        enddo
        do ires = 1, nmfcccap
          if (ifirst .ge. matomstartcap(ires) .and. ifirst .le. matomfinalcap(ires)) then
            nhit = nhit + 1
            write(ioutfile,'("   residue ",i0," (",a3,") charge ",i0," -> cap ",i0)') &
                  mfcc_chg_seq(ichg), adjustl(residue(ifirst)), mfcc_chg_val(ichg), ires
          endif
        enddo
        if (nhit .gt. 1) write(ioutfile,'("     (fragments overlap and the caps subtract the &
              &overlap, so this charge is counted once overall)")')
      enddo
    endif

    ! --- sum over the span each sub-molecule really covers ---
    do ires = 1, npmfcc
      mfcccharge(ires) = 0
      do ia = matomstart(ires), matomfinal(ires)
        mfcccharge(ires) = mfcccharge(ires) + atomchg(ia)
      enddo
    enddo
    do ires = 1, nmfcccap
      mfccchargecap(ires) = 0
      do ia = matomstartcap(ires), matomfinalcap(ires)
        mfccchargecap(ires) = mfccchargecap(ires) + atomchg(ia)
      enddo
    enddo

    ! The peptide total alone used to be reported as "the" total, which left
    ! out every standalone species: on trypsin it printed -3 while the model
    ! was really -1, because the Ca(2+) sits in mfcccharge and never in
    ! atomchg. Report both parts and the sum.
    chgtot = sum(atomchg) + sum(mfcccharge(npmfcc+1:npmfcc+nmfccextra))
    write(ioutfile,'(" MFCC formal charge: peptide ",i0,"  + standalone species ",i0, &
          &"  = ",i0)') sum(atomchg), sum(mfcccharge(npmfcc+1:npmfcc+nmfccextra)), chgtot

    if (quick_method%mfccchg .and. nchgapp .gt. 0) &
      write(ioutfile,'(" MFCC applied ",i0," charge override(s) to peptide residues")') nchgapp

    ! Every entry must have matched something. An override that silently
    ! applied to nothing is the failure mode this whole file exists to avoid.
    if (quick_method%mfccchg) then
      do ichg = 1, mfcc_chg_n
        if (.not.mfcc_chg_used(ichg)) then
          call PrtErr(iOutFile,'A charge override matched no residue in the structure.')
          write(ioutfile,'("   ",a3,1x,i0,1x,i0)') mfcc_chg_name(ichg), &
                mfcc_chg_seq(ichg), mfcc_chg_val(ichg)
          call quick_exit(iOutFile,1)
        endif
      enddo
    endif

    ! The fragments add up to chgtot, so if the molecule was given a different
    ! charge the guess and the SCF are describing different species. Worth
    ! saying out loud: it is the kind of mismatch that produces a converged
    ! number that simply answers another question.
    if (chgtot .ne. quick_molspec%molchg) then
      call PrtWrn(iOutFile,'MFCC fragment charges do not sum to the molecular charge.')
      write(ioutfile,'("|          fragments sum to ",i0,", but the molecule is set to ",i0,".")') &
            chgtot, quick_molspec%molchg
      write(ioutfile,'("|          Set CHARGE=",i0," to agree, or correct the residue charges.")') chgtot
      write(ioutfile,'(a)')
      call flush(ioutfile)
    endif
    deallocate(atomchg)
  end block

  if (any(mfcccharge(1:npmfcc).ne.0) .or. any(mfccchargecap(1:nmfcccap).ne.0)) then
    call PrtWrn(iOutFile,'MFCC assigned formal charges to some fragments and caps.')
    do i = 1, npmfcc
      if (mfcccharge(i).ne.0) &
        write(ioutfile,'("|          fragment ",i4," charge ",i3)') i,mfcccharge(i)
    enddo
    do i = 1, nmfcccap
      if (mfccchargecap(i).ne.0) &
        write(ioutfile,'("|          cap      ",i4," charge ",i3)') i,mfccchargecap(i)
    enddo
    write(ioutfile,'("|")')
    write(ioutfile,'("|          Detected: NH3+ and COO- termini, Lys, Arg, protonated His,")')
    write(ioutfile,'("|          Asp and Glu, from the residue name and the hydrogens bonded")')
    write(ioutfile,'("|          to each group. Bound ions, non standard residues, modified")')
    write(ioutfile,'("|          termini and anything whose residue name is not recognised are")')
    write(ioutfile,'("|          NOT detected and are treated as neutral. Check these charges")')
    write(ioutfile,'("|          against the chemistry of your system before trusting the guess.")')
    write(ioutfile,'(a)')
    call flush(ioutfile)
  endif
  write(ioutfile,*) '  '

! The whole block of code (below) up until the end
! of the subroutine is only executed when
! xiaoconnect(i,jj).eq.0

  do i=2,npmfcc
    do jj=i+3,npmfcc+3

    if(xiaoconnect(i,jj).eq.0)then
      print*,'xiaoconnect(i,jj) is zero'

  write(ioutfile,*) '======================================'
  write(ioutfile,*) 'Neutral terminus fragment'
  write(ioutfile,*) '  '

! Temporal files for debug of 'connect'

!  open(600,file='connect'//char(48+i/10) &
!  //char(48+i-i/10*10)//char(48+jj/10) &
!  //char(48+jj-jj/10*10)//'.gjf')

  if(i.eq.2)then
    mm=9
    nn=mselectN(1)
    nnn=mselectCA(1)

  else
    mm=mselectCA(i-2)
    nn=mselectN(i-1)
    nnn=mselectCA(i-1)
  endif

  ! The i==2 branch above hardcodes mm=9, an atom index from whatever system
  ! this was written against. For any real molecule it is unrelated to the
  ! chain start, and the span mm..nnn-1 it implies comes out empty or reversed:
  ! on Trp-cage nnn-mm+2 is -2, and every store below indexed the connection
  ! arrays at -2. Those arrays used to be fixed size, so the writes landed
  ! quietly in whatever preceded them in static memory; now that they are
  ! allocated it is a clean out-of-bounds access instead. Either way the block
  ! is meaningless, so skip the contact rather than build it.
  !
  ! The whole connection layer is disabled downstream when a block turns out
  ! malformed, which is the behaviour this preserves. It becomes live again
  ! once the chain-start span is defined properly.
  ! A positive span is not enough: the span has to lie inside the peptide
  ! chain. T4-Lysozyme numbers its ligand ahead of the protein, so the
  ! hardcoded mm=9 lands in the ligand, nnn-mm+2 comes out positive, and the
  ! old test passed a block that starts inside a hydrocarbon and runs into the
  ! protein. class is zero for every atom that is not part of the chain, which
  ! is the check that actually means something here.
  if (nnn-mm+2 .lt. 2 .or. mm-2 .lt. 1 .or. &
      mm-2 .lt. pep_first .or. nnn .gt. pep_last .or. &
      class(mm) .le. 0 .or. class(nnn) .le. 0) then
    nconskip = nconskip + 1
    write(ioutfile,'(" MFCC skipping contact between residues ",i4," and ",i4, &
          &": the connection span is empty (start atom ",i6,", end atom ",i6,")")') &
          i,jj,mm,nnn
    cycle
  endif

  call xyzchange(coord(1,mm-2),coord(2,mm-2),coord(3,mm-2), &
  coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)
!     write(*,*) '1st call for xyzchange in final loop'
     write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

  mfccatomxiaocon(1,kxiao)='H '
  mfccatomxiaoconi(1,kxiao)='H '

  mfcccordcon(1,1,kxiao)=xx
  mfcccordcon(2,1,kxiao)=ym
  mfcccordcon(3,1,kxiao)=zm
  mfcccordconi(1,1,kxiao)=xx
  mfcccordconi(2,1,kxiao)=ym
  mfcccordconi(3,1,kxiao)=zm

  mfccatomcon(kxiao)=nnn-mm+2
  mfccatomconi(kxiao)=nnn-mm+2

  mfccstartcon(kxiao)=2
  mfccfinalcon(kxiao)=nnn-mm+1
  mfccstartconi(kxiao)=2
  mfccfinalconi(kxiao)=nnn-mm+1

  matomstartcon(kxiao)=mm
  matomfinalcon(kxiao)=nnn-1
  matomstartconi(kxiao)=mm
  matomfinalconi(kxiao)=nnn-1

  do kk=mm,nnn-1
     write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
     mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
     mfccatomxiaocon(kk-mm+2,kxiao)=mfcc_element(atomname(kk))
     mfccatomxiaoconi(kk-mm+2,kxiao)=mfcc_element(atomname(kk))

     do j=1,3
       mfcccordcon(j,kk-mm+2,kxiao)=coord(j,kk)
       mfcccordconi(j,kk-mm+2,kxiao)=coord(j,kk)
     enddo
  enddo

  call Nxyzchange(coord(1,nnn),coord(2,nnn),coord(3,nnn), &
  coord(1,nn),coord(2,nn),coord(3,nn),xx,ym,zm)
     write(*,*) 'call for Nxyzchange in final loop'
     write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

        mfccatomxiaocon(nnn-mm+2,kxiao)='H '
        mfccatomxiaoconi(nnn-mm+2,kxiao)='H '

        mfcccordcon(1,nnn-mm+2,kxiao)=xx
        mfcccordcon(2,nnn-mm+2,kxiao)=ym
        mfcccordcon(3,nnn-mm+2,kxiao)=zm
        mfcccordconi(1,nnn-mm+2,kxiao)=xx
        mfcccordconi(2,nnn-mm+2,kxiao)=ym
        mfcccordconi(3,nnn-mm+2,kxiao)=zm

         if(jj.eq.np+3)then
           mm=mselectCA(np)
           mmm=mselectC(np)
           nn=mselectC(np)+4
           nnn=number-7

         else
           mm=mselectCA(jj-3)
           mmm=mselectC(jj-3)
           nn=mselectCA(jj-2)
           nnn=mselectC(jj-2)
         endif

   call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
   coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)
      write(*,*) '2nd call for xyzchange in final loop'
      write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

        mfccatomxiaocon2(1,kxiao)='H '
        mfccatomxiaoconj(1,kxiao)='H '

        mfcccordcon2(1,1,kxiao)=xx
        mfcccordcon2(2,1,kxiao)=ym
        mfcccordcon2(3,1,kxiao)=zm
        mfcccordconj(1,1,kxiao)=xx
        mfcccordconj(2,1,kxiao)=ym
        mfcccordconj(3,1,kxiao)=zm

        mfccatomcon2(kxiao)=nnn-mmm+2
        mfccatomconj(kxiao)=nnn-mmm+2

        mfccstartcon2(kxiao)=2
        mfccfinalcon2(kxiao)=nnn-mmm+1
        mfccstartconj(kxiao)=2
        mfccfinalconj(kxiao)=nnn-mmm+1

        matomstartcon2(kxiao)=mmm
        matomfinalcon2(kxiao)=nnn-1
        matomstartconj(kxiao)=mmm
        matomfinalconj(kxiao)=nnn-1

    do kk=mmm,nnn-1
      write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
      mfcc_element(atomname(kk)),(coord(j,kk),j=1,3)
      mfccatomxiaocon2(kk-mmm+2,kxiao)=mfcc_element(atomname(kk))
      mfccatomxiaoconj(kk-mmm+2,kxiao)=mfcc_element(atomname(kk))

      do j=1,3
        mfcccordcon2(j,kk-mmm+2,kxiao)=coord(j,kk)
        mfcccordconj(j,kk-mmm+2,kxiao)=coord(j,kk)
      enddo
  enddo

  call xyzchange(coord(1,nnn),coord(2,nnn),coord(3,nnn), &
  coord(1,nn),coord(2,nn),coord(3,nn),xx,ym,zm)
   write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)')'H ',xx,ym,zm

        mfccatomxiaocon2(nnn-mmm+2,kxiao)='H '
        mfccatomxiaoconj(nnn-mmm+2,kxiao)='H '

        mfcccordcon2(1,nnn-mmm+2,kxiao)=xx
        mfcccordcon2(2,nnn-mmm+2,kxiao)=ym
        mfcccordcon2(3,nnn-mmm+2,kxiao)=zm
        mfcccordconj(1,nnn-mmm+2,kxiao)=xx
        mfcccordconj(2,nnn-mmm+2,kxiao)=ym
        mfcccordconj(3,nnn-mmm+2,kxiao)=zm

    kxiao=kxiao+1

    endif
   enddo
  enddo

  kxiaoconnect=kxiao-1

  ! Skipping the malformed contacts individually would leave the rest of the
  ! layer live, and the two-body path has never run: with the chain-start
  ! blocks dropped and the other twenty kept, Trp-cage reaches pass 3 and
  ! crashes there. Disable the layer as a whole instead, which is what used to
  ! happen anyway once mfcc_fragment_scf saw an invalid atom count. The
  ! difference is that nothing has been written out of bounds getting here.
  if (nconskip .gt. 0) then
    call PrtWrn(iOutFile,'MFCC connection terms are disabled.')
    write(ioutfile,'("|          ",i4," of ",i4," contacts have a malformed connection span,")') &
          nconskip,kxiaoconnect+nconskip
    write(ioutfile,'("|          from the hardcoded chain-start atom index in mfcc_start.f90.")')
    write(ioutfile,'("|          The guess falls back to fragments minus caps, which is the")')
    write(ioutfile,'("|          published MFCC form and what every run so far has used.")')
    write(ioutfile,'(a)')
    call flush(ioutfile)
    kxiaoconnect=0
  endif

! Dump fragments and caps as one multi-frame xyz file. Fragment and cap frames
! are tagged in the comment line so they stay distinguishable in a single
! trajectory. Coordinates are already in Angstrom.
!
! With RPT a second file is written beside it, holding only the sub-molecules
! that are actually solved: every peptide fragment and cap, but one frame per
! repeated species instead of one per copy. On a solvated protein that is the
! difference between 3314 frames and a few hundred, and it is the honest picture
! of what the guess computes -- the other 2985 waters are not solved at all,
! their densities are the first one's, rotated. It also makes the ligand and the
! ions findable, which they are not in a file that buries them in solvent.
!
! The grouping has to be decided here rather than in mfcc_fragment_scf, because
! MFCCFRAG exits before that routine ever runs, and the fragment-only runtype is
! exactly where someone wants to look at this. It is geometry only and costs
! milliseconds. mfcc_fragment_scf then reuses the same assignment, so the file
! and the calculation cannot disagree.
!
! Only the master writes. mfcc() runs on every rank, every rank reaches the same
! fragmentation, and they all used to open the same path and write it at once --
! harmless in that the content matched, but still several processes writing one
! file. main.f90 points non-masters' iOutFile at /dev/null for the MFCC phase
! for the same reason; the xyz units were simply missed. Guarding it here also
! keeps the second file from doubling the problem.
!
! The grouping itself is done on every rank, not just the writer: it is a pure
! function of the fragment geometries, so all ranks reach the same answer, and
! doing it here means mfcc_fragment_scf's call is a no-op everywhere rather than
! on the master alone.
  if (quick_method%mfccrpt .and. nmfccextra .gt. 0 .and. quick_method%fragxyz) then
     ierrxyz = 0
     call mfcc_rpt_assign(ierrxyz)
     if (ierrxyz /= 0) then
        call PrtWrn(iOutFile,'Could not group the repeated MFCC fragments; &
              &writing the full xyz only.')
        ierrxyz = 0
     endif
  endif

  if (quick_method%fragxyz .and. master) then
     ierrxyz = 0
     call quick_open(iMfccXyzFile,mfccXyzFileName,'R','F','R',.true.,ierrxyz)
     if (ierrxyz /= 0) then
        call PrtWrn(iOutFile,'Could not open MFCC xyz file, skipping the dump.')
     else
        call mfcc_write_frag_xyz(iMfccXyzFile,.false.,xname)
        close(iMfccXyzFile)
        write(ioutfile,*) "MFCC wrote fragment and cap geometries to ", trim(mfccXyzFileName)
     endif

     if (quick_method%mfccrpt .and. nmfccextra .gt. 0 .and. mfcc_rpt_done) then
        ierrxyz = 0
        call quick_open(iMfccRptXyzFile,mfccRptXyzFileName,'R','F','R',.true.,ierrxyz)
        if (ierrxyz /= 0) then
           call PrtWrn(iOutFile,'Could not open the RPT MFCC xyz file, skipping it.')
        else
           call mfcc_write_frag_xyz(iMfccRptXyzFile,.true.,xname)
           close(iMfccRptXyzFile)
           write(ioutfile,'(" MFCC RPT: ",i6," standalone fragments collapse to ",i0, &
                 &" solved sub-molecule(s)")') nmfccextra,mfcc_rpt_nrep+count(mfcc_rpt_repof .eq. 0)
           write(ioutfile,*) "MFCC wrote the collapsed geometries to ", trim(mfccRptXyzFileName)
        endif
     endif
  endif

end subroutine mfcc


!-----------------------------------------------------------------------!
! mfcc_write_frag_xyz                                                   !
!                                                                       !
! Write the fragment, standalone and cap geometries to an already open  !
! unit as a multi-frame xyz.                                            !
!                                                                       !
! collapse = .false. writes every fragment, which is the original dump. !
! collapse = .true. writes only the sub-molecules that get their own    !
! SCF: all the peptide fragments and caps, one frame per repeated        !
! species, and any standalone fragment that matched no representative    !
! and is therefore solved individually. A representative's frame says    !
! how many copies it stands for, so the file explains itself.            !
!_______________________________________________________________________!
subroutine mfcc_write_frag_xyz(iunit,collapse,xname)
   use quick_mfcc_module
   use quick_mfcc_repeat_module, only: mfcc_rpt_repof, mfcc_rpt_ncopies, &
                                       mfcc_rpt_nrep, mfcc_rpt_rmsdrep, mfcc_rpt_done
   implicit none

   integer, intent(in) :: iunit
   logical, intent(in) :: collapse
   ! The residue name of each standalone fragment. A local of mfcc(), which is
   ! the only caller, so it is passed rather than promoted to module state.
   character(len=3), intent(in) :: xname(*)

   integer :: k, i, kx, ir, nkept

   ! Peptide fragments: every one is distinct, so both forms carry all of them.
   do k = 1, npmfcc
      write(iunit,'(i8)') mfccatom(k)
      write(iunit,'("mfcc fragment ",i0,"/",i0," | atoms ",i0)') k,npmfcc,mfccatom(k)
      do i = 1, mfccatom(k)
         write(iunit,'(a2,3(2x,f14.8))') mfccatomxiao(i,k), &
               mfcccord(1,i,k),mfcccord(2,i,k),mfcccord(3,i,k)
      enddo
   enddo

   ! Standalone fragments: solvent, ions and ligands.
   nkept = 0
   do kx = 1, nmfccextra
      k = npmfcc + kx
      if (collapse) then
         ! Keep the representatives (repof < 0) and the ones nothing represents
         ! (repof == 0); drop the copies (repof > 0).
         if (.not.mfcc_rpt_done) cycle
         if (mfcc_rpt_repof(kx) .gt. 0) cycle
      endif
      nkept = nkept + 1

      write(iunit,'(i8)') mfccatom(k)
      if (.not.collapse) then
         write(iunit,'("mfcc standalone ",i0,"/",i0," | ",a3," | atoms ",i0, &
               &" | charge ",i0)') kx,nmfccextra,xname(kx),mfccatom(k),mfcccharge(k)
      else if (mfcc_rpt_repof(kx) .lt. 0) then
         ir = -mfcc_rpt_repof(kx)
         ! ncopies counts the representative itself, so 1 means nothing was
         ! folded into it and there is no superposition rmsd to report.
         if (mfcc_rpt_ncopies(ir) .gt. 1) then
            write(iunit,'("mfcc standalone rep ",i0,"/",i0," | ",a3," | atoms ",i0, &
                  &" | charge ",i0," | solved once for ",i0," copies | worst rmsd ",es9.2," A")') &
                  ir,mfcc_rpt_nrep,xname(kx),mfccatom(k),mfcccharge(k), &
                  mfcc_rpt_ncopies(ir),mfcc_rpt_rmsdrep(ir)
         else
            write(iunit,'("mfcc standalone rep ",i0,"/",i0," | ",a3," | atoms ",i0, &
                  &" | charge ",i0," | only copy, nothing reused from it")') &
                  ir,mfcc_rpt_nrep,xname(kx),mfccatom(k),mfcccharge(k)
         endif
      else
         write(iunit,'("mfcc standalone ",i0,"/",i0," | ",a3," | atoms ",i0, &
               &" | charge ",i0," | matched no representative, solved on its own")') &
               kx,nmfccextra,xname(kx),mfccatom(k),mfcccharge(k)
      endif
      do i = 1, mfccatom(k)
         write(iunit,'(a2,3(2x,f14.8))') mfccatomxiao(i,k), &
               mfcccord(1,i,k),mfcccord(2,i,k),mfcccord(3,i,k)
      enddo
   enddo

   ! Caps: one per peptide junction, plus two thiols per disulfide bridge.
   do k = 1, nmfcccap
      write(iunit,'(i8)') mfccatomcap(k)
      if (k .le. max(npmfcc-1,0)) then
         write(iunit,'("mfcc cap ",i0,"/",i0," | atoms ",i0," | subtracted")') &
               k,nmfcccap,mfccatomcap(k)
      else
         write(iunit,'("mfcc cap ",i0,"/",i0," | atoms ",i0, &
               &" | thiol superseded by an intact S-S | subtracted")') &
               k,nmfcccap,mfccatomcap(k)
      endif
      do i = 1, mfccatomcap(k)
         write(iunit,'(a2,3(2x,f14.8))') mfccatomxiaocap(i,k), &
               mfcccordcap(1,i,k),mfcccordcap(2,i,k),mfcccordcap(3,i,k)
      enddo
   enddo

   ! The intact disulfides, one per bridge. These are the only sub-molecules
   ! whose atoms come from two different residues, and the only ones added
   ! rather than subtracted alongside the fragments.
   do k = 1, nmfccss
      write(iunit,'(i8)') mfccatomss(k)
      write(iunit,'("mfcc S-S ",i0,"/",i0," | atoms ",i0, &
            &" | intact disulfide, cut at both CA-CB | added")') &
            k,nmfccss,mfccatomss(k)
      do i = 1, mfccatomss(k)
         write(iunit,'(a2,3(2x,f14.8))') mfccatomxiaoss(i,k), &
               mfcccordss(1,i,k),mfcccordss(2,i,k),mfcccordss(3,i,k)
      enddo
   enddo

end subroutine mfcc_write_frag_xyz

!-----------------------------------------------------------------------!
! mfcc_ss_cap                                                           !
!                                                                       !
! Append a capping hydrogen to every sub-molecule that holds one end of  !
! a cut disulfide bond.                                                  !
!                                                                       !
! mnat/mcord/msym are a sub-molecule array set (fragments, or caps), and  !
! gs/gf the global atom range each sub-molecule was cut from. A sulfur in !
! that range whose partner is outside it has a dangling valence, so a     !
! hydrogen goes on it, pointing where the partner was.                    !
!                                                                       !
! The hydrogen is appended after the existing atoms, which places it past !
! mfccfinal and therefore outside the basis range the density block is    !
! harvested from -- the same treatment the backbone caps get.            !
!-----------------------------------------------------------------------!
subroutine mfcc_ss_cap(nsub, natmax, mnat, mcord, msym, gs, gf, &
                       natom_g, coord_g, sspart, nadded, ierr)
  implicit none
  integer, intent(in) :: nsub, natmax, natom_g
  integer, intent(inout) :: mnat(nsub)
  double precision, intent(inout) :: mcord(3,natmax,nsub)
  character(len=2), intent(inout) :: msym(natmax,nsub)
  integer, intent(in) :: gs(nsub), gf(nsub)
  double precision, intent(in) :: coord_g(3,natom_g)
  integer, intent(in) :: sspart(natom_g)
  integer, intent(inout) :: nadded, ierr

  integer :: k, ia, ip, nnew
  double precision :: xh, yh, zh

  do k = 1, nsub
    ! Unused slots are left at zero; the cap arrays run one short of the
    ! fragment arrays and both are allocated to the larger size.
    if (gs(k) .lt. 1 .or. gf(k) .lt. gs(k)) cycle
    if (mnat(k) .lt. 1) cycle

    do ia = gs(k), gf(k)
      ip = sspart(ia)
      if (ip .le. 0) cycle                  ! not part of a bridge
      if (ip .ge. gs(k) .and. ip .le. gf(k)) cycle   ! partner is here too, bond intact

      nnew = mnat(k) + 1
      if (nnew .gt. natmax) then
        ierr = 44
        return
      endif
      call Sxyzchange(coord_g(1,ip),coord_g(2,ip),coord_g(3,ip), &
                      coord_g(1,ia),coord_g(2,ia),coord_g(3,ia), xh,yh,zh)
      msym(nnew,k) = 'H '
      mcord(1,nnew,k) = xh
      mcord(2,nnew,k) = yh
      mcord(3,nnew,k) = zh
      mnat(k) = nnew
      nadded = nadded + 1
    enddo
  enddo

end subroutine mfcc_ss_cap


!-----------------------------------------------------------------------!
! mfcc_ss_half                                                          !
!                                                                       !
! Locate the CB and CA of the cysteine that owns the bridging sulfur    !
! isg, and confirm that CB..SG is one unbroken run of that residue's    !
! own atoms with nothing but its hydrogens in between.                  !
!                                                                       !
! That run is what becomes one half of the disulfide block, and the     !
! scatter in MFCC_initial_guess maps it onto global basis functions by  !
! a single offset, so it has to be contiguous. MFCC already reorders    !
! every hydrogen next to its heavy atom, which makes CB,HB,HB,SG the    !
! normal case; ok comes back false when a structure does not comply,    !
! and the caller then leaves that bridge hydrogen capped.               !
!-----------------------------------------------------------------------!
subroutine mfcc_ss_half(natom_g, atomname_g, class_g, isg, icb, ica, ok)
  implicit none
  integer, intent(in) :: natom_g, isg
  character(len=4), intent(in) :: atomname_g(natom_g)
  integer, intent(in) :: class_g(natom_g)
  integer, intent(out) :: icb, ica
  logical, intent(out) :: ok

  character(len=2), external :: mfcc_element
  integer :: i, ires

  ok  = .false.
  icb = 0
  ica = 0

  ires = class_g(isg)
  if (ires .le. 0) return

  do i = 1, natom_g
    if (class_g(i) .ne. ires) cycle
    if (atomname_g(i) .eq. ' CB ') icb = i
    if (atomname_g(i) .eq. ' CA ') ica = i
  enddo

  if (icb .le. 0 .or. ica .le. 0) return
  if (icb .ge. isg) return

  do i = icb+1, isg-1
    if (class_g(i) .ne. ires) return
    if (mfcc_element(atomname_g(i)) .ne. 'H ') return
  enddo

  ok = .true.

end subroutine mfcc_ss_half


!-----------------------------------------------------------------------!
! mfcc_ss_blocks                                                        !
!                                                                       !
! Build the disulfide correction for every bridge: one intact           !
! CH3-S-S-CH3 block to add, and the two CH3-S-H concaps it supersedes   !
! to subtract.                                                          !
!                                                                       !
! The S-S block's atoms are laid out as                                  !
!                                                                       !
!   [Hcap] [CB_i .. SG_i] [CB_j .. SG_j] [Hcap]                          !
!                                                                       !
! so its real atoms are one contiguous run LOCALLY -- which is what     !
! readbasis needs to report a single local basis range -- while mapping !
! onto two separate runs globally. Only the global start of each half   !
! is required to put the density back, and the block between the halves !
! is the S-S bond density that the hydrogen caps could never provide.   !
!                                                                       !
! The capping hydrogens replace CA on each CB, at the C-H distance, the !
! same substitution the backbone caps make.                             !
!-----------------------------------------------------------------------!
subroutine mfcc_ss_blocks(natom_g, coord_g, atomname_g, class_g, sspart, &
                          natmax, ncapmax, ncap, nssb, ierr)
  use quick_mfcc_module, only: mfccatomss, mfccchargess, mfcccordss, mfccatomxiaoss, &
                               mfccstartss, mfccfinalss, &
                               matomstartss1, matomfinalss1, matomstartss2, matomfinalss2, &
                               mfccatomcap, mfccchargecap, mfcccordcap, mfccatomxiaocap, &
                               mfccstartcap, mfccfinalcap, matomstartcap, matomfinalcap
  implicit none
  integer, intent(in) :: natom_g, natmax, ncapmax
  double precision, intent(in) :: coord_g(3,natom_g)
  character(len=4), intent(in) :: atomname_g(natom_g)
  integer, intent(in) :: class_g(natom_g), sspart(natom_g)
  integer, intent(inout) :: ncap, nssb, ierr

  character(len=2), external :: mfcc_element
  integer :: ia, ja, icb, jcb, ica, jca, k, n1, n2, nloc, kk, j
  double precision :: xh, yh, zh
  logical :: oki, okj

  do ia = 1, natom_g
    ja = sspart(ia)
    if (ja .le. ia) cycle            ! visit each bridge once, with ia < ja

    call mfcc_ss_half(natom_g, atomname_g, class_g, ia, icb, ica, oki)
    call mfcc_ss_half(natom_g, atomname_g, class_g, ja, jcb, jca, okj)
    if (.not. oki .or. .not. okj) cycle   ! caller reports the shortfall

    n1   = ia - icb + 1
    n2   = ja - jcb + 1
    nloc = n1 + n2 + 2
    if (nloc .gt. natmax) then
      ierr = 44
      return
    endif
    if (ncap + 2 .gt. ncapmax) then
      ierr = 44
      return
    endif

    ! ---- the intact disulfide, added ----
    nssb = nssb + 1
    k = nssb

    call xyzchange(coord_g(1,ica),coord_g(2,ica),coord_g(3,ica), &
                   coord_g(1,icb),coord_g(2,icb),coord_g(3,icb), xh,yh,zh)
    mfccatomxiaoss(1,k) = 'H '
    mfcccordss(1,1,k) = xh
    mfcccordss(2,1,k) = yh
    mfcccordss(3,1,k) = zh

    do kk = icb, ia
      mfccatomxiaoss(kk-icb+2,k) = mfcc_element(atomname_g(kk))
      do j = 1, 3
        mfcccordss(j,kk-icb+2,k) = coord_g(j,kk)
      enddo
    enddo

    do kk = jcb, ja
      mfccatomxiaoss(kk-jcb+2+n1,k) = mfcc_element(atomname_g(kk))
      do j = 1, 3
        mfcccordss(j,kk-jcb+2+n1,k) = coord_g(j,kk)
      enddo
    enddo

    call xyzchange(coord_g(1,jca),coord_g(2,jca),coord_g(3,jca), &
                   coord_g(1,jcb),coord_g(2,jcb),coord_g(3,jcb), xh,yh,zh)
    mfccatomxiaoss(nloc,k) = 'H '
    mfcccordss(1,nloc,k) = xh
    mfcccordss(2,nloc,k) = yh
    mfcccordss(3,nloc,k) = zh

    mfccatomss(k)   = nloc
    mfccchargess(k) = 0
    mfccstartss(k)  = 2
    mfccfinalss(k)  = 1 + n1 + n2
    matomstartss1(k) = icb
    matomfinalss1(k) = ia
    matomstartss2(k) = jcb
    matomfinalss2(k) = ja

    ! ---- the two thiols it replaces, subtracted as concaps ----
    call mfcc_ss_thiol(natom_g, coord_g, atomname_g, natmax, icb, ia, ica, ja, ncap)
    call mfcc_ss_thiol(natom_g, coord_g, atomname_g, natmax, jcb, ja, jca, ia, ncap)
  enddo

end subroutine mfcc_ss_blocks


!-----------------------------------------------------------------------!
! mfcc_ss_thiol                                                         !
!                                                                       !
! Append one CH3-S-H concap, covering global atoms icb..isg: a capping  !
! hydrogen on CB where CA was, the run itself, and a capping hydrogen   !
! on SG where the bridge partner was. This is what a hydrogen capped    !
! fragment already contributes around that sulfur, so subtracting it    !
! leaves the intact S-S block as the only description of the bridge.    !
!-----------------------------------------------------------------------!
subroutine mfcc_ss_thiol(natom_g, coord_g, atomname_g, natmax, icb, isg, ica, ipart, ncap)
  use quick_mfcc_module, only: mfccatomcap, mfccchargecap, mfcccordcap, mfccatomxiaocap, &
                               mfccstartcap, mfccfinalcap, matomstartcap, matomfinalcap
  implicit none
  integer, intent(in) :: natom_g, natmax, icb, isg, ica, ipart
  double precision, intent(in) :: coord_g(3,natom_g)
  character(len=4), intent(in) :: atomname_g(natom_g)
  integer, intent(inout) :: ncap

  character(len=2), external :: mfcc_element
  integer :: k, n, kk, j
  double precision :: xh, yh, zh

  n = isg - icb + 1
  ncap = ncap + 1
  k = ncap

  call xyzchange(coord_g(1,ica),coord_g(2,ica),coord_g(3,ica), &
                 coord_g(1,icb),coord_g(2,icb),coord_g(3,icb), xh,yh,zh)
  mfccatomxiaocap(1,k) = 'H '
  mfcccordcap(1,1,k) = xh
  mfcccordcap(2,1,k) = yh
  mfcccordcap(3,1,k) = zh

  do kk = icb, isg
    mfccatomxiaocap(kk-icb+2,k) = mfcc_element(atomname_g(kk))
    do j = 1, 3
      mfcccordcap(j,kk-icb+2,k) = coord_g(j,kk)
    enddo
  enddo

  call Sxyzchange(coord_g(1,ipart),coord_g(2,ipart),coord_g(3,ipart), &
                  coord_g(1,isg),coord_g(2,isg),coord_g(3,isg), xh,yh,zh)
  mfccatomxiaocap(n+2,k) = 'H '
  mfcccordcap(1,n+2,k) = xh
  mfcccordcap(2,n+2,k) = yh
  mfcccordcap(3,n+2,k) = zh

  mfccatomcap(k)    = n + 2
  mfccchargecap(k)  = 0
  mfccstartcap(k)   = 2
  mfccfinalcap(k)   = n + 1
  matomstartcap(k)  = icb
  matomfinalcap(k)  = isg

end subroutine mfcc_ss_thiol


subroutine xyzchange(xold,yold,zold,xzero,yzero,zzero, &
  xnew,ynew,znew)

  implicit none
  real(8)::grad,xold,yold,zold,xzero,yzero,zzero
  real(8)::xnew,ynew,znew

  grad=dsqrt(1.09d0**2/((xold-xzero)**2+(yold-yzero)**2 &
  +(zold-zzero)**2))
  xnew=xzero+grad*(xold-xzero)
  ynew=yzero+grad*(yold-yzero)
  znew=zzero+grad*(zold-zzero)
end

! Place a hydrogen 1.34 Angstrom from the sulfur at (xzero,yzero,zzero),
! along the direction of the atom it was bonded to. Same construction as
! xyzchange (C-H, 1.09) and Nxyzchange (N-H, 1.01), with the S-H bond length.
subroutine Sxyzchange(xold,yold,zold,xzero,yzero,zzero, &
  xnew,ynew,znew)

  implicit none
  real(8)::grad,xold,yold,zold,xzero,yzero,zzero
  real(8)::xnew,ynew,znew

  grad=dsqrt(1.34d0**2/((xold-xzero)**2+(yold-yzero)**2 &
  +(zold-zzero)**2))
  xnew=xzero+grad*(xold-xzero)
  ynew=yzero+grad*(yold-yzero)
  znew=zzero+grad*(zold-zzero)
end subroutine Sxyzchange


subroutine Nxyzchange(xold,yold,zold,xzero,yzero,zzero, &
  xnew,ynew,znew)

  implicit none
  real(8)::grad,xold,yold,zold,xzero,yzero,zzero
  real(8)::xnew,ynew,znew

  grad=dsqrt(1.01d0**2/((xold-xzero)**2+(yold-yzero)**2 &
  +(zold-zzero)**2))
  xnew=xzero+grad*(xold-xzero)
  ynew=yzero+grad*(yold-yzero)
  znew=zzero+grad*(zold-zzero)
end


!-----------------------------------------------------------------------!
! mfcc_element                                                          !
!                                                                       !
! Element symbol for a pdb atom name field (columns 13-16 of the        !
! record, so characters 1-4 here).                                      !
!                                                                       !
! A pdb right justifies the element symbol in columns 13-14, which is    !
! why characters 1-2 of this field usually give it directly. But a name  !
! needing all four characters starts in column 13 instead, and in a      !
! protein those are always hydrogens: HD11, HG12, HH21, HB13 and so on.  !
! Taking characters 1-2 there returns 'HD', 'HG' or 'HH', which are not  !
! elements, and the fragment then carries a bogus atom type. A glycine   !
! only system never exposes this, since glycine has no branched side     !
! chain and so no four character hydrogen name.                          !
!                                                                       !
! The older style that puts the branch digit first (1HB, 2HG1) is        !
! handled too: a leading digit only ever appears on a hydrogen.          !
!_______________________________________________________________________!

function mfcc_element(nm) result(el)
  implicit none
  character(len=4), intent(in) :: nm
  character(len=2) :: el
  character(len=4) :: nmt

  nmt = adjustl(nm)
  if (nmt(1:1).ge.'0' .and. nmt(1:1).le.'9') then
     el = 'H '
  else if (len_trim(nmt).eq.4 .and. nmt(1:1).eq.'H') then
     el = 'H '
  else
     el = adjustl(nm(1:2))
  endif
end function mfcc_element
