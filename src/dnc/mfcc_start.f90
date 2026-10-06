! Subroutine initiating the MFCC

subroutine mfcc(natomsaved)
   use allmod
   use quick_mfcc_species_module, only: mfcc_is_amino_acid, mfcc_species_charge, mfcc_species_kind
   use quick_mfcc_module
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
   character(len=3), allocatable :: xname(:)   ! residue name of each standalone fragment
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
  allocate(xfirst(max(nxtra,1)), xlast(max(nxtra,1)), xname(max(nxtra,1)))
  xfirst = 0
  xlast = 0
  xname = '   '
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
  mfccnatmax = mfccnatmax + 2
  deallocate(natres)

  ! A standalone fragment has to fit in the same arrays.
  do kx = 1, nxtra
    mfccnatmax = max(mfccnatmax, xlast(kx)-xfirst(kx)+1)
  enddo

  mfccierr = 0
  call mfcc_alloc_frag(npmfcc+nxtra, mfccnatmax, mfccierr)
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
    if (chgunknown) nunk = nunk + 1

    do kk = xfirst(kx), xlast(kx)
      mfccatomxiao(kk-xfirst(kx)+1,k) = mfcc_element(atomname(kk))
      do j = 1, 3
        mfcccord(j,kk-xfirst(kx)+1,k) = coord(j,kk)
      enddo
    enddo
  enddo

  if (nunk .gt. 0) then
    call PrtWrn(iOutFile,'Some non-peptide residues have an unrecognised charge.')
    write(ioutfile,'("|          ",i7," residue(s) were given charge 0 because their name is")') nunk
    write(ioutfile,'("|          not in the table in quick_mfcc_species_module. If one of them")')
    write(ioutfile,'("|          is actually charged, its fragment is wrong; a charged species")')
    write(ioutfile,'("|          with an odd electron count will be caught as an open shell")')
    write(ioutfile,'("|          later, but an even one will pass silently. Check these:")')
    do kx = 1, nxtra
      nat_x = xlast(kx)-xfirst(kx)+1
      if (mfcc_species_charge(xname(kx), nat_x, chgunknown) .eq. 0 .and. chgunknown) &
        write(ioutfile,'("|            ",a3,"  (",i6," atoms)")') xname(kx), nat_x
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

    ! --- sum over the span each sub-molecule really covers ---
    do ires = 1, npmfcc
      mfcccharge(ires) = 0
      do ia = matomstart(ires), matomfinal(ires)
        mfcccharge(ires) = mfcccharge(ires) + atomchg(ia)
      enddo
    enddo
    do ires = 1, npmfcc-1
      mfccchargecap(ires) = 0
      do ia = matomstartcap(ires), matomfinalcap(ires)
        mfccchargecap(ires) = mfccchargecap(ires) + atomchg(ia)
      enddo
    enddo

    write(ioutfile,'(" MFCC total formal charge from detected groups: ",i4)') sum(atomchg)
    deallocate(atomchg)
  end block

  if (any(mfcccharge(1:npmfcc).ne.0) .or. any(mfccchargecap(1:npmfcc-1).ne.0)) then
    call PrtWrn(iOutFile,'MFCC assigned formal charges to some fragments and caps.')
    do i = 1, npmfcc
      if (mfcccharge(i).ne.0) &
        write(ioutfile,'("|          fragment ",i4," charge ",i3)') i,mfcccharge(i)
    enddo
    do i = 1, npmfcc-1
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
  if (quick_method%fragxyz) then
     ierrxyz = 0
     call quick_open(iMfccXyzFile,mfccXyzFileName,'R','F','R',.true.,ierrxyz)
     if (ierrxyz /= 0) then
        call PrtWrn(iOutFile,'Could not open MFCC xyz file, skipping the dump.')
     else
        do k=1,npmfcc+nmfccextra
           write(iMfccXyzFile,'(i8)') mfccatom(k)
           if (k .le. npmfcc) then
              write(iMfccXyzFile,'("mfcc fragment ",i0,"/",i0," | atoms ",i0)') k,npmfcc,mfccatom(k)
           else
              write(iMfccXyzFile,'("mfcc standalone ",i0,"/",i0," | ",a3," | atoms ",i0, &
                    &" | charge ",i0)') k-npmfcc,nmfccextra,xname(k-npmfcc),mfccatom(k),mfcccharge(k)
           endif
           do i=1,mfccatom(k)
              write(iMfccXyzFile,'(a2,3(2x,f14.8))') mfccatomxiao(i,k), &
                    mfcccord(1,i,k),mfcccord(2,i,k),mfcccord(3,i,k)
           enddo
        enddo
        do k=1,npmfcc-1
           write(iMfccXyzFile,'(i8)') mfccatomcap(k)
           write(iMfccXyzFile,'("mfcc cap ",i0,"/",i0," | atoms ",i0)') k,npmfcc-1,mfccatomcap(k)
           do i=1,mfccatomcap(k)
              write(iMfccXyzFile,'(a2,3(2x,f14.8))') mfccatomxiaocap(i,k), &
                    mfcccordcap(1,i,k),mfcccordcap(2,i,k),mfcccordcap(3,i,k)
           enddo
        enddo
        close(iMfccXyzFile)
        write(ioutfile,*) "MFCC wrote fragment and cap geometries to ", trim(mfccXyzFileName)
     endif
  endif

end

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
