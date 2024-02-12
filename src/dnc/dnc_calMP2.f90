#include "util.fh"
! Ed Brothers. November 27, 2001
! Xiao HE. September 14,2008
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

! Ed Brothers. November 27, 2001
! Xiao HE. September 14,2008
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine calmp2divcon
  use allmod
  use quick_gaussian_class_module
  implicit double precision(a-h,o-z)

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

  allocate(wtospoint(np,nbasis))
  call wtoscorr

  xiaocutoffmp2=1.0d-7

  quick_qm_struct%EMP2=0.0d0
  emp2temp=0.0d0
  nstep=1


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


        do k3=1,2
            do j3=1,ivir
            do l3=1,ivir
                L3new=j3+iocc
                do lll=1,nbasisdc(itt)
                 !write(10,*) k3,j3,LLL,L3new,orbmp2k331(1,k3,j3,LLL),quick_qm_struct%co(LLL,L3new)
                enddo
            enddo
            enddo
        enddo

     ttt=0.0d0

     do iiat=1,dcsubn(itt)
        iiatom=dcsub(itt,iiat)
        do II=quick_basis%first_shell_basis_function(iiatom),quick_basis%last_shell_basis_function(iiatom)
           do jjat=iiat,dcsubn(itt)
              jjatom=dcsub(itt,jjat)
              do JJ=max(quick_basis%first_shell_basis_function(jjatom),II),quick_basis%last_shell_basis_function(jjatom)
                 Testtmp=Ycutoff(II,JJ)
                 ttt=max(ttt,Testtmp)
              enddo
           enddo
        enddo
     enddo


    ! determine the electrons for subsystem
     if(mod(nelecmp2sub(itt),2).eq.1)then
        nelecmp2sub(itt)=nelecmp2sub(itt)+1
     endif

    ! determine the occupied and virtual orbitals
     if(mod(nelecmp2sub(itt),2).eq.1)then
        iocc=Nelecmp2sub(itt)/2+1
        ivir=nbasisdc(itt)-iocc+1
     else
        iocc=Nelecmp2sub(itt)/2
        ivir=nbasisdc(itt)-iocc
     endif
!write(*,*) iocc,ivir
     ! with f orbital
     if (quick_method%ffunxiao) then
        nbasistemp=6
     else
        nbasistemp=10
     endif

     ! allocate varibles
     allocate(orbmp2(100,100))
     allocate(orbmp2dcsub(iocc,ivir,ivir))
     allocate(orbmp2i331(nstep,nbasisdc(itt),nbasistemp,nbasistemp,2))
     allocate(orbmp2j331(nstep,ivir,nbasistemp,nbasistemp,2))
     allocate(orbmp2k331(nstep,iocc,ivir,nbasisdc(itt)))
     allocate(orbmp2k331dcsub(iocc,ivir,nbasisdc(itt)))

     ! Schwartz cutoff is implemented here. (ab|cd)**2<=(ab|ab)*(cd|cd)
     ! Reference: Strout DL and Scuseria JCP 102(1995),8448.

     do i3=1,iocc
        do l1=1,ivir
           do k1=1,ivir
                 orbmp2(j1,k1)=0.0d0
           enddo
        enddo

        call initialOrbmp2k331(orbmp2k331,nstep,nbasisdc(itt),ivir,iocc,nstep)

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

        do iiat=1,dcsubn(itt)
           iiatom=dcsub(itt,iiat)

           do jjat=iiat,dcsubn(itt)
              jjatom=dcsub(itt,jjat)

              ! iiatom and jjatom is the atom of the subsystem
              ! first, we need to figure out which shell should we calclulate
              ! which is IIstart1 and IIstart2 for I and JJ is from JJstart1 to JJstart2
              if(iiatom.le.jjatom)then
                 IIstart1=quick_basis%first_shell_basis_function(iiatom)
                 IIstart2=quick_basis%last_shell_basis_function(iiatom)
                 JJstart1=quick_basis%first_shell_basis_function(jjatom)
                 JJstart2=quick_basis%last_shell_basis_function(jjatom)
              endif
              if(iiatom.gt.jjatom)then
                 JJstart1=quick_basis%first_shell_basis_function(iiatom)
                 JJstart2=quick_basis%last_shell_basis_function(iiatom)
                 IIstart1=quick_basis%first_shell_basis_function(jjatom)
                 IIstart2=quick_basis%last_shell_basis_function(jjatom)
              endif

              do II=IIstart1,IIstart2
                 do JJ=max(JJstart1,II),JJstart2

                    Testtmp=Ycutoff(II,JJ)
                    if(Testtmp.gt.XiaoCUTOFFmp2/ttt)then

                        call initialOrbmp2ij(orbmp2i331,nstep,nstep,nbasisdc(itt),nbasistemp,nbasistemp)
                        call initialOrbmp2ij(orbmp2j331,nstep,nstep,ivir,nbasistemp,nbasistemp)

                       ! Now we will determine K shell and L shell, the last two indices
                       do kkat=1,dcsubn(itt)
                          kkatom=dcsub(itt,kkat)
                          do LLat=kkat,dcsubn(itt)
                             LLatom=dcsub(itt,LLat)
                             if(KKatom.le.LLatom)then
                                KKstart1=quick_basis%first_shell_basis_function(KKatom)
                                KKstart2=quick_basis%last_shell_basis_function(KKatom)
                                LLstart1=quick_basis%first_shell_basis_function(LLatom)
                                LLstart2=quick_basis%last_shell_basis_function(LLatom)
                             endif
                             if(KKatom.gt.LLatom)then
                                LLstart1=quick_basis%first_shell_basis_function(KKatom)
                                LLstart2=quick_basis%last_shell_basis_function(KKatom)
                                KKstart1=quick_basis%first_shell_basis_function(LLatom)
                                KKstart2=quick_basis%last_shell_basis_function(LLatom)
                             endif
                             do KK=KKstart1,KKstart2
                                do LL=max(LLstart1,KK),LLstart2
                                   comax=0.d0
                                   XiaoTEST1 = Ycutoff(II,JJ)*Ycutoff(KK,LL)
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

                                            comax=max(comax,dabs(quick_qm_struct%co(wtospoint(itt,kkk),i3)))
                                            comax=max(comax,dabs(quick_qm_struct%co(wtospoint(itt,lll),i3)))

                                         enddo
                                      enddo

                                      Xiaotest=xiaotest1*comax
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
                                do j33=1,ivir
                                   j33new=j33+iocc
                                   if(mod(nelecmp2sub(itt),2).eq.1)j33new=j33+iocc-1
                                   atemp=quick_qm_struct%co(LLL,j33new)
                                   orbmp2j331(nstep,j33,IIInew,JJJnew,1)=orbmp2j331(nstep,j33,IIInew,JJJnew,1) + &
                                        orbmp2i331(nstep,LLL,IIInew,JJJnew,1)*atemp
                                   if(III.ne.JJJ)then
                                      orbmp2j331(nstep,j33,JJJnew,IIInew,2)=orbmp2j331(nstep,j33,JJJnew,IIInew,2) + &
                                           orbmp2i331(nstep,LLL,JJJnew,IIInew,2)*atemp
                                   endif
                                enddo
                             enddo

                             do j33=1,ivir
                                do k33=1,iocc
                                   orbmp2k331(nstep,k33,j33,wtospoint(itt,JJJ))=orbmp2k331(nstep,k33,j33,wtospoint(itt,JJJ))+ &
                                        orbmp2j331(nstep,j33,IIInew,JJJnew,1)*quick_scratch%hold(k33,wtospoint(itt,III))
                                   if(III.ne.JJJ)then
                                      orbmp2k331(nstep,k33,j33,wtospoint(itt,III))=orbmp2k331(nstep,k33,j33,wtospoint(itt,III))+ &
                                           orbmp2j331(nstep,j33,JJJnew,IIInew,2)*quick_scratch%hold(k33,wtospoint(itt,JJJ))
                                   endif
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
                                   do k33=1,iocc
                                      orbmp2k331dcsub(k33,j33,wtospoint(itt,JJJ))=orbmp2k331dcsub(k33,j33,wtospoint(itt,JJJ))+ &
                                           orbmp2j331(nstep,j33,IIInew,JJJnew,1)*quick_scratch%hold(k33,wtospoint(itt,III))
                                   enddo
                                enddo
                             endif

                             if(locallog2.and.III.ne.JJJ)then
                                do j33=1,ivir
                                   do k33=1,iocc
                                      orbmp2k331dcsub(k33,j33,wtospoint(itt,III))=orbmp2k331dcsub(k33,j33,wtospoint(itt,III))+ &
                                           orbmp2j331(nstep,j33,JJJnew,IIInew,2)*quick_scratch%hold(k33,wtospoint(itt,JJJ))
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

        do k3=1,2
            do j3=1,ivir
            do l3=1,ivir
                L3new=j3+iocc
                do lll=1,nbasisdc(itt)
                 !write(10,*) k3,j3,LLL,L3new,orbmp2k331(1,k3,j3,LLL),quick_qm_struct%co(LLL,L3new)
                enddo
            enddo
            enddo
        enddo

        do LLL=1,nbasisdc(itt)
           do J3=1,ivir
              do L3=1,ivir
                 L3new=L3+iocc
                 orbmp2(L3,j3)=0.0d0
                 if(mod(nelecmp2sub(itt),2).eq.1)L3new=L3+iocc-1
                 do k3=1,iocc
                    orbmp2(l3,j3)=orbmp2(l3,j3)+orbmp2k331(nstep,k3,j3,LLL)*quick_scratch%hold(L3new,LLL)
                    orbmp2dcsub(k3,l3,j3)=orbmp2dcsub(k3,l3,j3)+orbmp2k331dcsub(k3,j3,LLL)*quick_scratch%hold(L3new,LLL)
                 enddo
              enddo
           enddo
        enddo

        if(mod(nelecmp2sub(itt),2).eq.0)then
           do l=1,ivir
              do k=1,iocc
                 do j=1,ivir
                    quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                         -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
                         *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(j,l)-orbmp2(l,j))
                 enddo
              enddo
           enddo
        endif

        if(mod(nelecmp2sub(itt),2).eq.1)then
           do l=1,ivir
              do k=1,iocc
                 do j=1,ivir
                    quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(Evaldcsub(itt,i3)+Evaldcsub(itt,k) &
                         -Evaldcsub(itt,j+nelecmp2sub(itt)/2)-Evaldcsub(itt,l+nelecmp2sub(itt)/2)) &
                         *orbmp2dcsub(k,j,l)*(2.0d0*orbmp2(j,l)-orbmp2(l,j))
                 enddo
              enddo
           enddo
        endif
     enddo
     write(ioutfile,*) itt,quick_qm_struct%EMP2,quick_qm_struct%EMP2-emp2temp

     emp2temp=quick_qm_struct%EMP2

!     deallocate(mp2shell)
     if (allocated(orbmp2)) deallocate(orbmp2)
     if (allocated(orbmp2i331)) deallocate(orbmp2i331)
     if (allocated(orbmp2j331)) deallocate(orbmp2j331)
     if (allocated(orbmp2k331)) deallocate(orbmp2k331)
     if (allocated(orbmp2dcsub)) deallocate(orbmp2dcsub)
     if (allocated(orbmp2k331dcsub)) deallocate(orbmp2k331dcsub)

  enddo

999 return
end subroutine calmp2divcon
