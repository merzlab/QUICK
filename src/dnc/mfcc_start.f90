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
   integer natomsaved,npmfcc
   integer,allocatable::mselectC(:),mselectN(:),mselectCA(:) 
   real(8)::xx,yy,zz,ym,zm
   integer :: mfccatom(50),mfcccharge(50)
   integer :: mfccatomcap(50),mfccchargecap(50)
   integer :: mspin(50)

! integer :: kxiaoconnect
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

  write(*,*) "npmfcc=", npmfcc
  write(*,*) "Assigned number of fragments"

! Assign zero values for initialization of MFCC
! Make multiplicity equal to one for all fragments

   do i=1,npmfcc
    mfcccharge(i)=0
    mfccatom(i)=0
    mfccchargecap(i)=0
    mfccatomcap(i)=0
    mspin(i)=1
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

! Initiate xyzchange

 call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
   coord(1,nn),coord(2,nn),coord(3,nn),xx,yy,zz)     

 write(*,*) "first call xyzchange output" 
 write(*,*)'H ',xx,yy,zz

 mfccatomxiao(mm,1)='H '

 mfcccord(1,mm,1)=xx
 mfcccord(2,mm,1)=yy
 mfcccord(3,mm,1)=zz

 mfccatom(1)=mm

 mfccstart(1)=1
 mfccfinal(1)=mm-1

 matomstart(1)=1
 matomfinal(1)=mm-1

  do k=2,npmfcc-1
 
   mm=mselectN(k-1)
   nn=mselectC(k+1)
   mmm=mselectCA(k-1)
   nnn=mselectCA(k+1)
   nnnn=mselectC(k-2)
   !write(*,*) residue(mselectN(k-1)), 'residue(mselectN(k-1))'
   if(residue(mselectN(k-1)).ne.'PRO')then
    call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
    coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)    
    write(*,*) 'second call xyzchange output'
    write(*,*)'H ',xx,ym,zm

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
      write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
      mfccatomxiao(kk-mmm+2,k)=atomname(kk)(2:2)//' '
      do j=1,3
        mfcccord(j,kk-mmm+2,k)=coord(j,kk)
      enddo
    enddo

   call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
   write(*,*) 'third call xyzchange output'
   write(*,*)'H ',xx,ym,zm

   mfccatomxiao(nn-mmm+2,k)='H '

   mfcccord(1,nn-mmm+2,k)=xx
   mfcccord(2,nn-mmm+2,k)=ym
   mfcccord(3,nn-mmm+2,k)=zm
      
    else

   call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(*,*)'H ',xx,ym,zm

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
      write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)

      mfccatomxiao(kk-mm+2,k)=atomname(kk)(2:2)//' '
      do j=1,3
        mfcccord(j,kk-mm+2,k)=coord(j,kk)
      enddo

   enddo

   call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
   write(*,*) 'PROline call xyzchange output'
   write(*,*)'H ',xx,ym,zm

   mfccatomxiao(nn-mm+2,k)='H '

   mfcccord(1,nn-mm+2,k)=xx
   mfcccord(2,nn-mm+2,k)=ym
   mfcccord(3,nn-mm+2,k)=zm

  endif
  enddo

! Start the second fragmentation cycle over the coordinates

  mm=mselectN(npmfcc-1)
  mmm=mselectCA(npmfcc-1)
  nnnn=mselectC(npmfcc-2)
  if(residue(mselectN(npmfcc-1)).ne.'PRO')then
  call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
  coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)    
  write(*,*) "first call xyzchange in 2nd loop"
  write(*,*)'H ',xx,ym,zm

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
    write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)

     mfccatomxiao(kk-mmm+2,npmfcc)=atomname(kk)(2:2)//' '
     do j=1,3
       mfcccord(j,kk-mmm+2,npmfcc)=coord(j,kk)
     enddo

   enddo
  else
  call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(*,*)'H ',xx,ym,zm

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
    write(40,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)

    mfccatomxiao(kk-mm+2,npmfcc)=atomname(kk)(2:2)//' '
    do j=1,3
      mfcccord(j,kk-mm+2,npmfcc)=coord(j,kk)
    enddo

  enddo
  endif     

! Start loop over caps
   do k=1,npmfcc-1
   mm=mselectN(k)
   nn=mselectC(k+1)
   mmm=mselectCA(k)
   nnn=mselectCA(k+1)
   nnnn=mselectC(k-1)
   if(residue(mselectN(k)).ne.'PRO')then
    call xyzchange(coord(1,mm),coord(2,mm),coord(3,mm), &
    coord(1,mmm),coord(2,mmm),coord(3,mmm),xx,ym,zm)       
   write(*,*) '1st call for xyzchange in caps loop'
   write(*,*)'H ',xx,ym,zm

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
    write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
    mfccatomxiaocap(kk-mmm+2,k)=atomname(kk)(2:2)//' '
    do j=1,3
      mfcccordcap(j,kk-mmm+2,k)=coord(j,kk)
    enddo

 enddo

 call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
  coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
  write(*,*) '2nd xyzchange call for caps'
  write(*,*)'H ',xx,ym,zm

        mfccatomxiaocap(nn-mmm+2,k)='H '

        mfcccordcap(1,nn-mmm+2,k)=xx
        mfcccordcap(2,nn-mmm+2,k)=ym
        mfcccordcap(3,nn-mmm+2,k)=zm

    else

  call Nxyzchange(coord(1,nnnn),coord(2,nnnn),coord(3,nnnn), &
   coord(1,mm),coord(2,mm),coord(3,mm),xx,ym,zm)       
   write(*,*) 'nxyzchange call for caps if PROline present'
   write(*,*)'H ',xx,ym,zm

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
       write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
       mfccatomxiaocap(kk-mm+2,k)=atomname(kk)(2:2)//' '
       do j=1,3
         mfcccordcap(j,kk-mm+2,k)=coord(j,kk)
       enddo

   enddo

  call xyzchange(coord(1,nn),coord(2,nn),coord(3,nn), &
   coord(1,nnn),coord(2,nnn),coord(3,nnn),xx,ym,zm)
    write(*,*) 'PROline xyzchange call for caps'
    write(*,*)'H ',xx,ym,zm
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

  write(*,*) 'MFCC checked for neutral terminus'

! The whole block of code (below) up until the end
! of the subroutine is only executed when
! xiaoconnect(i,jj).eq.0

  do i=2,npmfcc
    do jj=i+3,npmfcc+3

    if(xiaoconnect(i,jj).eq.0)then
      print*,'xiaoconnect(i,jj) is zero'

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
     write(*,*) '1st call for xyzchange in final loop'
     write(*,*)'H ',xx,ym,zm

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
     write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
     mfccatomxiaocon(kk-mm+2,kxiao)=atomname(kk)(2:2)//' '
     mfccatomxiaoconi(kk-mm+2,kxiao)=atomname(kk)(2:2)//' '

     do j=1,3
       mfcccordcon(j,kk-mm+2,kxiao)=coord(j,kk)
       mfcccordconi(j,kk-mm+2,kxiao)=coord(j,kk)
     enddo
  enddo

  call Nxyzchange(coord(1,nnn),coord(2,nnn),coord(3,nnn), &
  coord(1,nn),coord(2,nn),coord(3,nn),xx,ym,zm)
     write(*,*) 'call for Nxyzchange in final loop'
     write(*,*)'H ',xx,ym,zm

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
      write(*,*)'H ',xx,ym,zm

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
      write(*,*)atomname(kk)(2:2)//' ',(coord(j,kk),j=1,3)
      mfccatomxiaocon2(kk-mmm+2,kxiao)=atomname(kk)(2:2)//' '
      mfccatomxiaoconj(kk-mmm+2,kxiao)=atomname(kk)(2:2)//' '

      do j=1,3
        mfcccordcon2(j,kk-mmm+2,kxiao)=coord(j,kk)
        mfcccordconj(j,kk-mmm+2,kxiao)=coord(j,kk)
      enddo
  enddo

  call xyzchange(coord(1,nnn),coord(2,nnn),coord(3,nnn), &
  coord(1,nn),coord(2,nn),coord(3,nn),xx,ym,zm)
     write(*,*)'H ',xx,ym,zm

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
