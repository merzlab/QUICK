! Ed Brothers. November 27, 2001
! Xiao HE. September 14,2008
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

    subroutine calmp2divcon
    use allmod
    use quick_gaussian_class_module
    implicit double precision(a-h,o-z)
    include 'divcon.h'

    logical locallog1,locallog2

    double precision Xiaotest,testtmp
 integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
 common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

 is = 0
 quick_basis%first_shell_basis_function(1) = 1
 do ii=1,natom-1
      is=is+quick_basis%kshell(quick_molspec%iattype(ii))
      quick_basis%last_shell_basis_function(ii)=is
      quick_basis%first_shell_basis_function(ii+1) = is+1
 enddo
 quick_basis%last_shell_basis_function(natom) = nshell

 do ii=1,natom
   print*,"nshell=",ii,quick_basis%first_shell_basis_function(ii),quick_basis%last_shell_basis_function(ii)
!   print*,"nbasis=",ii,first_basis_function(ii),last_basis_function(ii)
 enddo

 allocate(wtospoint(np,nbasis))
 call wtoscorr

 xiaocutoffmp2=1.0d-7

 quick_qm_struct%EMP2=0.0d0
 emp2temp=0.0d0


do itt=1,np

 do i=1,nbasisdc(itt)
  do j=1,nbasisdc(itt)
    quick_qm_struct%co(i,j)=COdcsub(i,j,itt)
  enddo
 enddo 

 do i=1,nbasisdc(itt)
  do j=1,nbasisdc(itt)
    quick_scratch%hold(i,j)=quick_qm_struct%co(j,i)
  enddo
 enddo

 ttt=0.0d0

 do iiat=1,dcsubn(itt)
  iiatom=dcsub(itt,iiat)
  do II=quick_basis%first_shell_basis_function(iiatom),quick_basis%last_shell_basis_function(iiatom)
    do jjat=iiat,dcsubn(itt)
     jjatom=dcsub(itt,jjat)
      do JJ=max(quick_basis%first_shell_basis_function(jjatom),II),quick_basis%last_shell_basis_function(jjatom)
!  do II=1,jshell
!   do JJ=II,jshell
     Testtmp=Ycutoff(II,JJ)
     ttt=max(ttt,Testtmp)
     enddo
    enddo
   enddo
 enddo


 if(mod(nelecmp2sub(itt),2).eq.1)then
  nelecmp2sub(itt)=nelecmp2sub(itt)+1
 endif

 if(mod(nelecmp2sub(itt),2).eq.1)then
   iocc=Nelecmp2sub(itt)/2+1
   ivir=nbasisdc(itt)-iocc+1
 else
   iocc=Nelecmp2sub(itt)/2
   ivir=nbasisdc(itt)-iocc
 endif

! ivir=nbasisdc(itt)-iocc

 allocate(mp2shell(nbasisdc(itt)))
 allocate(orbmp2(iocc,ivir,ivir))
 allocate(orbmp2dcsub(iocc,ivir,ivir))
 if(ffunxiao)then
  nbasistemp=6
  allocate(orbmp2i331(nbasisdc(itt),6,6,2))
  allocate(orbmp2j331(ivir,6,6,2))
 else
  nbasistemp=10
  allocate(orbmp2i331(nbasisdc(itt),10,10,2))
  allocate(orbmp2j331(ivir,10,10,2))
 endif

 allocate(orbmp2k331(iocc,ivir,nbasisdc(itt)))
 allocate(orbmp2k331dcsub(iocc,ivir,nbasisdc(itt)))

! print*,"iocc=",iocc,"ivir=",ivir

 Nxiao1=0
 Nxiao2=0

! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
! Reference: Strout DL and Scuseria JCP 102(1995),8448.

! print*,"before 2e"

! do jmax=1,nbasis-nelec/2
!tttmax=0.0d0
!do imax=1,nbasis
!  tttmax=max(tttmax,dabs(co(imax,jmax)))
! enddo
!print*,jmax,tttmax
!enddo
!stop

do i3=1,iocc

     do l1=1,ivir
    do k1=1,ivir
   do j1=1,iocc
       orbmp2(j1,k1,l1)=0.0d0
     enddo
    enddo
   enddo

  do j1=1,nbasisdc(itt)
   do k1=1,ivir
 do i1=1,iocc
    orbmp2k331(i1,k1,j1)=0.0d0
   enddo
  enddo
 enddo

     do l1=1,ivir
    do k1=1,ivir
   do j1=1,iocc
       orbmp2dcsub(j1,k1,l1)=0.0d0
     enddo
    enddo
   enddo

  do j1=1,nbasisdc(itt)
   do k1=1,ivir
 do i1=1,iocc
    orbmp2k331dcsub(i1,k1,j1)=0.0d0
   enddo
  enddo
 enddo

ntemp=0

! do II=1,jshell
!   do JJ=II,jshell
 do iiat=1,dcsubn(itt)
  iiatom=dcsub(itt,iiat)
  do II=quick_basis%first_shell_basis_function(iiatom),quick_basis%last_shell_basis_function(iiatom)
    do jjat=iiat,dcsubn(itt)
     jjatom=dcsub(itt,jjat)
      do JJ=max(quick_basis%first_shell_basis_function(jjatom),II),quick_basis%last_shell_basis_function(jjatom)

     Testtmp=Ycutoff(II,JJ)
!     ttt=max(ttt,Testtmp)
    if(Testtmp.gt.XiaoCUTOFFmp2/ttt)then

    do l1=1,2
  do j1=1,nbasistemp
 do i1=1,nbasistemp
   do k1=1,nbasisdc(itt)
    orbmp2i331(k1,i1,j1,l1)=0.0d0
   enddo
  enddo
 enddo
 enddo

    do l1=1,2
  do j1=1,nbasistemp
 do i1=1,nbasistemp
   do k1=1,ivir
    orbmp2j331(k1,i1,j1,l1)=0.0d0
   enddo
  enddo
 enddo
 enddo

!  do i1=1,nbasis
!    mp2shell(i1)=.false.
!  enddo

!     do KK=II,jshell
!       do LL=KK,jshell

 do kkat=1,dcsubn(itt)
  kkatom=dcsub(itt,kkat)
  do KK=quick_basis%first_shell_basis_function(kkatom),quick_basis%last_shell_basis_function(kkatom)
    do LLat=kkat,dcsubn(itt)
     LLatom=dcsub(itt,LLat)
      do LL=max(quick_basis%first_shell_basis_function(LLatom),KK),quick_basis%last_shell_basis_function(LLatom)

!     do KK=1,jshell
!       do LL=KK,jshell

            comax=0.d0
            XiaoTEST1 = TESTtmp*Ycutoff(KK,LL)
          if(XiaoTEST1.gt.XiaoCUTOFFmp2)then

 NKK1=quick_basis%Qstart(KK)
 NKK2=quick_basis%Qfinal(KK)
 NLL1=quick_basis%Qstart(LL)
 NLL2=quick_basis%Qfinal(LL)

   NBK1=quick_basis%Qsbasis(KK,NKK1)
   NBK2=quick_basis%Qfbasis(KK,NKK2)
   NBL1=quick_basis%Qsbasis(LL,NLL1)
   NBL2=quick_basis%Qfbasis(LL,NLL2)

   KK111=quick_basis%ksumtype(KK)+NBK1
   KK112=quick_basis%ksumtype(KK)+NBK2
   LL111=quick_basis%ksumtype(LL)+NBL1
   LL112=quick_basis%ksumtype(LL)+NBL2

       do KKK=KK111,KK112
          do LLL=max(KKK,LL111),LL112

!            print*,co(kkk,i3),co(lll,i3)
            comax=max(comax,dabs(quick_qm_struct%co(wtospoint(itt,kkk),i3)))
            comax=max(comax,dabs(quick_qm_struct%co(wtospoint(itt,lll),i3)))    
       
          enddo
       enddo        

            Xiaotest=xiaotest1*comax
!            DNmax=max(4.0d0*cutmatrix(II,JJ),4.0d0*cutmatrix(KK,LL), &
!                  cutmatrix(II,LL),cutmatrix(II,KK),cutmatrix(JJ,KK),cutmatrix(JJ,LL))
!            XiaoTest=Xiaotest1*DNmax
            if(XiaoTEST.gt.XiaoCUTOFFmp2)then
              ntemp=ntemp+1
              call shellmp2divcon(i3,itt)
            endif

           endif

       enddo
     enddo

    enddo
   enddo


 NII1=quick_basis%Qstart(II)
 NII2=quick_basis%Qfinal(II)
 NJJ1=quick_basis%Qstart(JJ)
 NJJ2=quick_basis%Qfinal(JJ)

   NBI1=quick_basis%Qsbasis(II,NII1)
   NBI2=quick_basis%Qfbasis(II,NII2)
   NBJ1=quick_basis%Qsbasis(JJ,NJJ1)
   NBJ2=quick_basis%Qfbasis(JJ,NJJ2)

   II111=quick_basis%ksumtype(II)+NBI1
   II112=quick_basis%ksumtype(II)+NBI2
   JJ111=quick_basis%ksumtype(JJ)+NBJ1
   JJ112=quick_basis%ksumtype(JJ)+NBJ2

       do III=II111,II112
          do JJJ=max(III,JJ111),JJ112

          IIInew=III-II111+1
          JJJnew=JJJ-JJ111+1

     do LLL=1,nbasisdc(itt)
!    if(mp2shell(LLL).eq..true.)then
   do j33=1,ivir
     j33new=j33+iocc
if(mod(nelecmp2sub(itt),2).eq.1)j33new=j33+iocc-1
       atemp=quick_qm_struct%co(LLL,j33new)
       orbmp2j331(j33,IIInew,JJJnew,1)=orbmp2j331(j33,IIInew,JJJnew,1) + &
                                     orbmp2i331(LLL,IIInew,JJJnew,1)*atemp
       if(III.ne.JJJ)then
        orbmp2j331(j33,JJJnew,IIInew,2)=orbmp2j331(j33,JJJnew,IIInew,2) + &
                                      orbmp2i331(LLL,JJJnew,IIInew,2)*atemp
       endif
     enddo
!    endif
   enddo

   do j33=1,ivir
     do k33=i3,iocc
       orbmp2k331(k33,j33,wtospoint(itt,JJJ))=orbmp2k331(k33,j33,wtospoint(itt,JJJ))+ &
                               orbmp2j331(j33,IIInew,JJJnew,1)*quick_scratch%hold(k33,wtospoint(itt,III))
       if(III.ne.JJJ)then
        orbmp2k331(k33,j33,wtospoint(itt,III))=orbmp2k331(k33,j33,wtospoint(itt,III))+ &
                                orbmp2j331(j33,JJJnew,IIInew,2)*quick_scratch%hold(k33,wtospoint(itt,JJJ))
       endif
!       print*,orbmp2j331(III,JJJ,j33),orbmp2k331(k33,JJJ,j33)
     enddo
   enddo

 locallog1=.false.
 locallog2=.false.

 do iiatdc=1,dccoren(itt)
  iiatomdc=dccore(itt,iiatdc)
  do IInbasisdc=quick_basis%first_basis_function(iiatomdc),quick_basis%last_basis_function(iiatomdc)
   if(III.eq.IInbasisdc)locallog1=.true.
   if(JJJ.eq.IInbasisdc)locallog2=.true.
  enddo
 enddo

 if(locallog1)then
   do j33=1,ivir
     do k33=i3,iocc
       orbmp2k331dcsub(k33,j33,wtospoint(itt,JJJ))=orbmp2k331dcsub(k33,j33,wtospoint(itt,JJJ))+ &
                               orbmp2j331(j33,IIInew,JJJnew,1)*quick_scratch%hold(k33,wtospoint(itt,III))
     enddo
   enddo
 endif

 if(locallog2.and.III.ne.JJJ)then
   do j33=1,ivir
     do k33=i3,iocc
        orbmp2k331dcsub(k33,j33,wtospoint(itt,III))=orbmp2k331dcsub(k33,j33,wtospoint(itt,III))+ &
                                orbmp2j331(j33,JJJnew,IIInew,2)*quick_scratch%hold(k33,wtospoint(itt,JJJ))
     enddo
   enddo
 endif

           enddo
       enddo

   endif

   enddo
 enddo

 enddo
enddo

  write (ioutfile,*)"ntemp=",ntemp

                   do LLL=1,nbasisdc(itt)
                  do J3=1,ivir
                do L3=1,ivir
                   L3new=L3+iocc
if(mod(nelecmp2sub(itt),2).eq.1)L3new=L3+iocc-1
                 do k3=i3,iocc
                    orbmp2(k3,l3,j3)=orbmp2(k3,l3,j3)+orbmp2k331(k3,j3,LLL)*quick_scratch%hold(L3new,LLL)
!                    print*,orbmp2(k3,l3,i3,j3),orbmp2k331(k3,JJJ,j33)
                    orbmp2dcsub(k3,l3,j3)=orbmp2dcsub(k3,l3,j3)+ &
                                          orbmp2k331dcsub(k3,j3,LLL)*quick_scratch%hold(L3new,LLL)
                   enddo
                  enddo
                 enddo
                enddo

if(mod(nelecmp2sub(itt),2).eq.0)then
              do l=1,ivir
            do k=i3,iocc
          do j=1,ivir
               if(k.gt.i3)then
                quick_qm_struct%EMP2=quick_qm_struct%EMP2+2.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                    -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
       *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(k,j,l)-orbmp2(k,l,j))
               endif
               if(k.eq.i3)then
                quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                    -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
       *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(k,j,l)-orbmp2(k,l,j))
               endif
                enddo
          enddo
      enddo
endif

if(mod(nelecmp2sub(itt),2).eq.1)then
              do l=1,ivir
            do k=i3,iocc
          do j=1,ivir
               if(k.gt.i3)then
                quick_qm_struct%EMP2=quick_qm_struct%EMP2+2.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                    -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
       *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(k,j,l)-orbmp2(k,l,j))
               endif
               if(k.eq.i3)then
                quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                    -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
       *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(k,j,l)-orbmp2(k,l,j))
               endif
                enddo
          enddo
      enddo
endif


 enddo

 write(ioutfile,*)quick_qm_struct%EMP2,quick_qm_struct%EMP2-emp2temp

 emp2temp=quick_qm_struct%EMP2

 deallocate(mp2shell)
 deallocate(orbmp2)
  deallocate(orbmp2i331)
  deallocate(orbmp2j331)
 deallocate(orbmp2k331)
 deallocate(orbmp2dcsub)
 deallocate(orbmp2k331dcsub)

enddo

! print*,'Nxiao1=',Nxiao1,'Nxiao2=',Nxiao2,XiaoCUTOFF

!     do i=1,nbasis
!       print*,i,E(i)
!     enddo

!    print*,"max=",ttt

    return
    end subroutine calmp2divcon



