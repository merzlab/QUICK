! Subroutine initiating the MFCC

subroutine mfcc(natomsaved)
   use allmod
   use quick_mfcc_module
!   use quick_method_module

   implicit none
   integer xiaoconnect(100,100)
   integer :: i,j,j1,j2,j3,number,mm,nn,kk
   character*6,allocatable:: sn(:)             ! series no.
   double precision,allocatable::coord(:,:)    ! cooridnates
   integer,allocatable::class(:),ttnumber(:)   ! class and residue number
   character*4,allocatable::atomname(:)        ! atom name
   character*3,allocatable::residue(:)         ! residue name
   integer natomsaved
   integer,allocatable::mselectC(:),mselectN(:),mselectCA(:) 

!   double precision :: mfcccord(:,:,:)
!   integer :: selectC(:), selectCA(:)
!   character*4,allocatable::mfccatomxiao(:,:)

   number=natomsaved ! avoid modification of important variable natomsaved

! Allocate arrays

   write(*,*) "Started MFCC fragmentation"

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

! Temporal file for tests
!   open(20,file='number01.gjf')

! Read-in the PDB file
   open(iPDBFile,file=PDBFileName)

   do 99 i=1,number
     read(iPDBFile,100)sn(i),ttnumber(i),atomname(i),residue(i),class(i),(coord(j,i),j=1,3)
100  format(a6,1x,I4,1x,a4,1x,a3,3x,I3,4x,3f8.3)
99   enddo
   close(iPDBFile)

   write(*,*) "Processed PDB file"

! Confirm reading of PDB
! do i=1,number
!  write(*,*) atomname(i)
! enddo

! Confirm reading of class
! do i=1,number
!  write(*,*) class(i)
! enddo

! Number of fragments
 npmfcc=class(number)

  write(*,*) "Assigned number of fragments"

! Assign zero values for initialization of MFCC
! Make multiplicity equal to one for all fragments

   do i=1,np
    mfcccharge(i)=0
    mfccatom(i)=0
    mfccchargecap(i)=0
    mfccatomcap(i)=0
    spin(i)=1
  enddo

 write(*,*) "Initialiazed MFCC arguments"

! Identify C, N, and CA atoms
   j1=1
   j2=1
   j3=1
   do i=1,number
   if(atomname(i).eq.' C  ')then
     mselectC(j1)=i
     write(*,*) mselectC(j1), "C"
     j1=j1+1
   endif
   if(atomname(i).eq.' N  ')then
     mselectN(j2)=i
     write(*,*) mselectN(j2), "N"
     j2=j2+1
   endif
   if(atomname(i).eq.' CA ')then
     mselectCA(j3)=i
     write(*,*) mselectCA(j3), "CA"
     j3=j3+1
   endif
  enddo

 write(*,*) "Identified C, N, CA"

! Start assigning coordinates to MFCC fragments

 write(*,*) mselectC(2), mselectCA(2), "mselectC and mselectCA"

   mm=mselectC(2)
   nn=mselectCA(2)

 write(*,*) mm,nn, "mm and nn values"

  do kk=1,mm-1
      write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
      mfccatomxiao(kk,1)=atomname(kk)(2:2)//' '
      do j=1,3
        mfcccord(j,kk,1)=coord(j,kk)
      enddo
  enddo

end
