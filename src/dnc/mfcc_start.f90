! Subroutine initiating the MFCC

subroutine mfcc(natomsaved)
   use allmod
   use quick_mfcc_module
!   use quick_method_module

   implicit none
   integer xiaoconnect(100,100)
   integer :: i,j,j1,j2,j3,number,mm,nn,kk
   integer :: mmm,nnn,nnnn,k,ii,jj
   integer :: ixiao,jxiao,xiaodis,kxiao
   character*6,allocatable:: sn(:)             ! series no.
   double precision,allocatable::coord(:,:)    ! cooridnates
   integer,allocatable::class(:),ttnumber(:)   ! class and residue number
   character*4,allocatable::atomname(:)        ! atom name
   character*3,allocatable::residue(:)         ! residue name
   integer natomsaved
   integer,allocatable::mselectC(:),mselectN(:),mselectCA(:)
   character*80 :: pdbline                     ! raw PDB record buffer
   integer :: ipdbstat                         ! iostat for PDB record reads
   integer :: ierrxyz                          ! iostat for the fragment xyz dump
   real(8)::xx,yy,zz,ym,zm
   integer :: mfcccharge(50)
   integer :: mfccchargecap(50)
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
   do i=1,100
     do j=1,100
       xiaoconnect(i,j)=1
     enddo
   enddo 

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
100  format(a6,1x,I4,1x,a4,1x,a3,3x,I3,4x,3f8.3)
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

! Number of fragments
 npmfcc=class(number)

 write(ioutfile,*) 'Number of MFCC fragments', ' is ', npmfcc

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
   do i=1,number
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

  do kk=1,mm-1
!      write(ioutfile,*)adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
 write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
      adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
      mfccatomxiao(kk,1)=adjustl(atomname(kk)(1:2))
      do j=1,3
        mfcccord(j,kk,1)=coord(j,kk)
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

 mfccatomxiao(mm,1)='H '

 mfcccord(1,mm,1)=xx
 mfcccord(2,mm,1)=yy
 mfcccord(3,mm,1)=zz

 mfccatom(1)=mm

 mfccstart(1)=1
 mfccfinal(1)=mm-1

 matomstart(1)=1
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
      adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
      mfccatomxiao(kk-mmm+2,k)=adjustl(atomname(kk)(1:2))
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
      adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)

      mfccatomxiao(kk-mm+2,k)=adjustl(atomname(kk)(1:2))
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
  
  mfccatom(npmfcc)=number-mmm+1+1
  
  mfccstart(npmfcc)=2
  mfccfinal(npmfcc)=number-mmm+2
  
  matomstart(npmfcc)=mmm
  matomfinal(npmfcc)=number  

  do kk=mmm,number
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
     adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)

     mfccatomxiao(kk-mmm+2,npmfcc)=adjustl(atomname(kk)(1:2))
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

    mfccatom(npmfcc)=number-mm+1+1

   mfccstart(npmfcc)=2
   mfccfinal(npmfcc)=number-mmm+2

   matomstart(npmfcc)=mm
   matomfinal(npmfcc)=number

   do kk=mm,number
    write(ioutfile,'(4x,A2,6x,F10.4,3x,F10.4,3x,F10.4)') &
     adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)

    mfccatomxiao(kk-mm+2,npmfcc)=adjustl(atomname(kk)(1:2))
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
    adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
    mfccatomxiaocap(kk-mmm+2,k)=adjustl(atomname(kk)(1:2))
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
       adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
       mfccatomxiaocap(kk-mm+2,k)=adjustl(atomname(kk)(1:2))
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

! Start the final loop, which concerns neutral terminus.

  do ixiao=2,npmfcc
    do jxiao=ixiao+3,npmfcc+3
      do ii=1,number
        do jj=1,number
          if((class(ii).eq.ixiao).and.(class(jj).eq.jxiao))then
            xiaodis=dsqrt((coord(1,ii)-coord(1,jj))**2.0d0+ &
                          (coord(2,ii)-coord(2,jj))**2.0d0+ &
                          (coord(3,ii)-coord(3,jj))**2.0d0)
            if(xiaodis.le.-1.0d0)then
              xiaoconnect(ixiao,jxiao)=0
              print*,ixiao,jxiao,ii,jj, 'ixiao,jxiao,ii,jj'
            endif
          endif
         enddo
       enddo
     enddo
   enddo

  kxiao=1

  write(ioutfile,*) '======================================'
  write(ioutfile,*) 'MFCC checked for neutral terminus'
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
     adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
     mfccatomxiaocon(kk-mm+2,kxiao)=adjustl(atomname(kk)(1:2))
     mfccatomxiaoconi(kk-mm+2,kxiao)=adjustl(atomname(kk)(1:2))

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
      adjustl(atomname(kk)(1:2)),(coord(j,kk),j=1,3)
      mfccatomxiaocon2(kk-mmm+2,kxiao)=adjustl(atomname(kk)(1:2))
      mfccatomxiaoconj(kk-mmm+2,kxiao)=adjustl(atomname(kk)(1:2))

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

! Dump fragments and caps as one multi-frame xyz file. Fragment and cap frames
! are tagged in the comment line so they stay distinguishable in a single
! trajectory. Coordinates are already in Angstrom.
  if (quick_method%fragxyz) then
     ierrxyz = 0
     call quick_open(iMfccXyzFile,mfccXyzFileName,'R','F','R',.true.,ierrxyz)
     if (ierrxyz /= 0) then
        call PrtWrn(iOutFile,'Could not open MFCC xyz file, skipping the dump.')
     else
        do k=1,npmfcc
           write(iMfccXyzFile,'(i8)') mfccatom(k)
           write(iMfccXyzFile,'("mfcc fragment ",i0,"/",i0," | atoms ",i0)') k,npmfcc,mfccatom(k)
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
