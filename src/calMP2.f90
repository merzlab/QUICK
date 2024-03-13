#include "util.fh"
! Ed Brothers. November 27, 2001
! Xiao HE. September 14,2008
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine calmp2
  use allmod
  use quick_gaussian_class_module
  use quick_cutoff_module, only: cshell_density_cutoff
  implicit double precision(a-h,o-z)

  double precision cutoffTest,testtmp,testCutoff
  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
    integer :: nelec,nelecb

    nelec = quick_molspec%nelec
    nelecb = quick_molspec%nelecb

  call PrtAct(ioutfile,"Begin MP2 Calculation")
  cutoffmp2=1.0d-8  ! cutoff criteria
  quick_method%primLimit=1.0d-8
  quick_qm_struct%EMP2=0.0d0

  ! occupied and virtual orbitals number
  iocc=Nelec/2
  ivir=Nbasis-Nelec/2

  ! calculate memory usage and determine steps
  ememorysum=real(iocc*ivir*nbasis*8.0d0/1024.0d0/1024.0d0/1024.0d0)

  ! actually nstep is step length
  nstep=min(int(1.5d0/ememorysum),Nelec/2)

  ! if with f orbital
  if(quick_method%hasF)then
     nbasistemp=6
  else
     nbasistemp=10
  endif

  ! Allocate some variables
  allocate(mp2shell(nbasis))
  allocate(orbmp2(ivir,ivir))
  allocate(orbmp2i331(nstep,nbasis,nbasistemp,nbasistemp,2))
  allocate(orbmp2j331(nstep,ivir,nbasistemp,nbasistemp,2))
  allocate(orbmp2k331(nstep,iocc,ivir,nbasis))

  ! with nstep(acutally, it represetns step lenght), we can
  ! have no. of steps for mp2 calculation
  nstepmp2=nelec/2/nstep
  nstepmp2=nstepmp2+1
  if(nstep*(nstepmp2-1).eq.nelec/2)then
     nstepmp2=nstepmp2-1
  endif

  ! Pre-step for density cutoff
  call cshell_density_cutoff

  ! first save coeffecient.
  do i=1,nbasis
     do j=1,nbasis
        quick_scratch%hold(i,j)=quick_qm_struct%co(j,i)
     enddo
  enddo

  ttt=MAXVAL(Ycutoff) ! Max Value of Ycutoff

  do i3new=1,nstepmp2               ! Step counter

     RECORD_TIME(timer_begin%TMP2)
     ntemp=0    ! integer counter
     nstepmp2s=(i3new-1)*nstep+1    ! Step start n
     nstepmp2f=i3new*nstep          ! Step end n

     if(i3new.eq.nstepmp2)nstepmp2f=nelec/2
     nsteplength=nstepmp2f-nstepmp2s+1  ! Step Lengh, from nstepmp2s to nstepmp2f

     ! Initial orbmp2k331
     call initialOrbmp2k331(orbmp2k331,nstep,nbasis,ivir,iocc,nsteplength)
     do II=1,jshell
        do JJ=II,jshell
           if(Ycutoff(II,JJ).gt.cutoffmp2/ttt)then
              call initialOrbmp2ij(orbmp2i331,nstep,nsteplength,nbasis,nbasistemp,nbasistemp)
              call initialOrbmp2ij(orbmp2j331,nstep,nsteplength,ivir,nbasistemp,nbasistemp)

              do KK=1,jshell
                 do LL=KK,jshell

                    ! Schwarts cutoff is implemented here
                    comax=0.d0
                    testCutoff = Ycutoff(II,JJ)*Ycutoff(KK,LL)
                    if(testCutoff.gt.cutoffmp2)then

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

                       do icycle=1,nsteplength
                            i3=nstepmp2s+icycle-1
                            do KKK=KK111,KK112
                                do LLL=max(KKK,LL111),LL112
                                    comax=max(comax,dabs(quick_qm_struct%co(kkk,i3)))
                                    comax=max(comax,dabs(quick_qm_struct%co(lll,i3)))
                                enddo
                            enddo
                       enddo

                       testCutoff=testCutoff*comax
                       if(testCutoff.gt.cutoffmp2)then
                          dnmax=comax
                          ntemp=ntemp+1
                          call shellmp2(nstepmp2s,nsteplength)
                       endif

                    endif

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

                    do LLL=1,nbasis
                       do j33=1,ivir
                          j33new=j33+nelec/2
                          atemp=quick_qm_struct%co(LLL,j33new)
                          do icycle=1,nsteplength
                             orbmp2j331(icycle,j33,IIInew,JJJnew,1)=orbmp2j331(icycle,j33,IIInew,JJJnew,1) + &
                                  orbmp2i331(icycle,LLL,IIInew,JJJnew,1)*atemp
                             if(III.ne.JJJ)then
                                orbmp2j331(icycle,j33,JJJnew,IIInew,2)=orbmp2j331(icycle,j33,JJJnew,IIInew,2) + &
                                     orbmp2i331(icycle,LLL,JJJnew,IIInew,2)*atemp
                             endif
                          enddo
                       enddo
                    enddo

                    do j33=1,ivir
                       do k33=1,nelec/2
                          atemp=quick_scratch%hold(k33,III)
                          atemp2=quick_scratch%hold(k33,JJJ)
                          do icycle=1,nsteplength
                             orbmp2k331(icycle,k33,j33,JJJ)=orbmp2k331(icycle,k33,j33,JJJ)+ &
                                  orbmp2j331(icycle,j33,IIInew,JJJnew,1)*atemp
                             if(III.ne.JJJ)then
                                orbmp2k331(icycle,k33,j33,III)=orbmp2k331(icycle,k33,j33,III)+ &
                                     orbmp2j331(icycle,j33,JJJnew,IIInew,2)*atemp2
                             endif
                          enddo
                       enddo
                    enddo
                 enddo
              enddo
           endif

        enddo
     enddo

     do icycle=1,nsteplength
        do k3=1,nelec/2
            do j3=1,nbasis-nelec/2
            do l3=1,nbasis-nelec/2
            L3new=l3+nelec/2
                do lll=1,nbasis
                 !write(10,*) k3,j3,LLL,L3new,orbmp2k331(icycle,k3,j3,LLL),quick_qm_struct%co(LLL,L3new)
                enddo
            enddo
            enddo
        enddo
     enddo

     do icycle=1,nsteplength
        i3=nstepmp2s+icycle-1
        do k3=i3,nelec/2

           do J3=1,nbasis-nelec/2
              do L3=1,nbasis-nelec/2
                 orbmp2(L3,J3)=0.0d0
                 L3new=L3+nelec/2
                 do LLL=1,nbasis
                    orbmp2(L3,J3)=orbmp2(L3,J3)+orbmp2k331(icycle,k3,j3,LLL)*quick_qm_struct%co(LLL,L3new)
                 enddo
              enddo
           enddo

           do J3=1,nbasis-nelec/2
              do L3=1,nbasis-nelec/2
                 if(k3.gt.i3)then
                    quick_qm_struct%EMP2=quick_qm_struct%EMP2+2.0d0/(quick_qm_struct%E(i3)+quick_qm_struct%E(k3) &
                        -quick_qm_struct%E(j3+nelec/2)-quick_qm_struct%E(l3+nelec/2)) &
                         *orbmp2(j3,l3)*(2.0d0*orbmp2(j3,l3)-orbmp2(l3,j3))
                 endif
                 if(k3.eq.i3)then
                    quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(quick_qm_struct%E(i3)+quick_qm_struct%E(k3) &
                        -quick_qm_struct%E(j3+nelec/2)-quick_qm_struct%E(l3+nelec/2)) &
                         *orbmp2(j3,l3)*(2.0d0*orbmp2(j3,l3)-orbmp2(l3,j3))
                 endif

              enddo
           enddo
        enddo
     enddo

     RECORD_TIME(timer_end%TMP2)
     timer_cumer%TMP2=timer_end%TMP2-timer_begin%TMP2+timer_cumer%TMP2

  enddo

  write (iOutFile,'("SECOND ORDER ENERGY =",F16.9)') quick_qm_struct%EMP2
  write (iOutFile,'("EMP2                =",F16.9)') quick_qm_struct%Etot+quick_qm_struct%EMP2
  call PrtAct(ioutfile,"End MP2 Calculation")
  return
end subroutine calmp2


#ifdef MPIV

! Ed Brothers. November 27, 2001
! Xiao HE. September 14,2008
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP
subroutine MPI_calmp2
  use allmod
  use quick_gaussian_class_module
  use quick_cutoff_module, only: cshell_density_cutoff
  use mpi
  implicit double precision(a-h,o-z)

  double precision cutoffTest,testtmp,testCutoff
  double precision, allocatable:: temp4d(:,:,:,:)
  integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,total_ntemp
  common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
    integer :: nelec,nelecb

    nelec = quick_molspec%nelec
    nelecb = quick_molspec%nelecb

  if (bMPI) then
!     call MPI_BCAST(DENSE,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!     call MPI_BCAST(CO,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
!     call MPI_BCAST(E,nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
  endif


  cutoffmp2=1.0d-8  ! cutoff criteria
  quick_method%primLimit=1.0d-8
  quick_qm_struct%EMP2=0.0d0

  ! occupied and virtual orbitals number
  iocc=Nelec/2
  ivir=Nbasis-Nelec/2


  ! calculate memory usage and determine steps
  ememorysum=real(iocc*ivir*nbasis*8.0d0/1024.0d0/1024.0d0/1024.0d0)
  if (master) then
     call PrtAct(ioutfile,"Begin MP2 Calculation")
     !write(ioutfile,'("| CURRENT MEMORY USAGE= ",E12.6," M")') ememorysum
  endif

  ! actually nstep is step length
  nstep=min(int(1.5d0/ememorysum),Nelec/2)

  ! if with f orbital
  if(quick_method%hasF)then
     nbasistemp=6
  else
     nbasistemp=10
  endif

  ! Allocate some variables
  allocate(mp2shell(nbasis))
  allocate(orbmp2(ivir,ivir))
  allocate(orbmp2i331(nstep,nbasis,nbasistemp,nbasistemp,2))
  allocate(orbmp2j331(nstep,ivir,nbasistemp,nbasistemp,2))
  allocate(orbmp2k331(nstep,iocc,ivir,nbasis))
  allocate(temp4d(nstep,iocc,ivir,nbasis))

  ! with nstep(acutally, it represetns step lenght), we can
  ! have no. of steps for mp2 calculation
  nstepmp2=nelec/2/nstep
  nstepmp2=nstepmp2+1
  if(nstep*(nstepmp2-1).eq.nelec/2)then
     nstepmp2=nstepmp2-1
  endif
  !write(ioutfile,'("TOTAL STEP          =",I6)') nstepmp2
  ! Pre-step for density cutoff
  call cshell_density_cutoff

  ! first save coeffecient.
  do i=1,nbasis
     do j=1,nbasis
        quick_scratch%hold(i,j)=quick_qm_struct%co(j,i)
     enddo
  enddo

  ttt=MAXVAL(Ycutoff) ! Max Value of Ycutoff

  do i3new=1,nstepmp2               ! Step counter
     RECORD_TIME(timer_begin%TMP2)
     ntemp=0    ! integer counter
     total_ntemp=0
     nstepmp2s=(i3new-1)*nstep+1    ! Step start n
     nstepmp2f=i3new*nstep          ! Step end n

     if(i3new.eq.nstepmp2)nstepmp2f=nelec/2
     nsteplength=nstepmp2f-nstepmp2s+1  ! Step Lengh, from nstepmp2s to nstepmp2f

     ! Initial orbmp2k331
     call initialOrbmp2k331(orbmp2k331,nstep,nbasis,ivir,iocc,nsteplength)

     !---------------- MPI/ ALL NODES -----------------------------
     do i=1,mpi_jshelln(mpirank)
        II=mpi_jshell(mpirank,i)
        !     do II=1,jshell
        do JJ=II,jshell

           ! First we do integral and sum them properly
           if(Ycutoff(II,JJ).gt.cutoffmp2/ttt)then
              call initialOrbmp2ij(orbmp2i331,nstep,nsteplength,nbasis,nbasistemp,nbasistemp)
              call initialOrbmp2ij(orbmp2j331,nstep,nsteplength,ivir,nbasistemp,nbasistemp)

              do KK=1,jshell
                 do LL=KK,jshell

                    ! Schwarts cutoff is implemented here
                    comax=0.d0
                    testCutoff = Ycutoff(II,JJ)*Ycutoff(KK,LL)
                    if(testCutoff.gt.cutoffmp2)then

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

                       ! find the co-max value
                       do icycle=1,nsteplength
                            i3=nstepmp2s+icycle-1
                            do KKK=KK111,KK112
                                do LLL=max(KKK,LL111),LL112
                                    comax=max(comax,dabs(quick_qm_struct%co(kkk,i3)))
                                    comax=max(comax,dabs(quick_qm_struct%co(lll,i3)))
                                enddo
                            enddo
                       enddo



                       testCutoff=testCutoff*comax
                       if(testCutoff.gt.cutoffmp2)then
                          dnmax=comax
                          ntemp=ntemp+1
                          call shellmp2(nstepmp2s,nsteplength)
                       endif

                    endif

                 enddo
              enddo


              ! Next step is to folding integers. Without folding, the scaling will
              ! be n^8, and after folding, scaling is reduced to 4n^5

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

                    ! Folding 4 indices integers into 3 indices
                    do LLL=1,nbasis
                       do j33=1,ivir
                          j33new=j33+nelec/2
                          atemp=quick_qm_struct%co(LLL,j33new)
                          do icycle=1,nsteplength
                             orbmp2j331(icycle,j33,IIInew,JJJnew,1)=orbmp2j331(icycle,j33,IIInew,JJJnew,1) + &
                                  orbmp2i331(icycle,LLL,IIInew,JJJnew,1)*atemp
                             if(III.ne.JJJ)then
                                orbmp2j331(icycle,j33,JJJnew,IIInew,2)=orbmp2j331(icycle,j33,JJJnew,IIInew,2) + &
                                     orbmp2i331(icycle,LLL,JJJnew,IIInew,2)*atemp
                             endif
                          enddo
                       enddo
                    enddo

                    do j33=1,ivir
                       do k33=1,nelec/2
                          atemp=quick_scratch%hold(k33,III)
                          atemp2=quick_scratch%hold(k33,JJJ)
                          do icycle=1,nsteplength
                             orbmp2k331(icycle,k33,j33,JJJ)=orbmp2k331(icycle,k33,j33,JJJ)+ &
                                  orbmp2j331(icycle,j33,IIInew,JJJnew,1)*atemp
                             if(III.ne.JJJ)then
                                orbmp2k331(icycle,k33,j33,III)=orbmp2k331(icycle,k33,j33,III)+ &
                                     orbmp2j331(icycle,j33,JJJnew,IIInew,2)*atemp2
                             endif
                          enddo
                       enddo
                    enddo
                 enddo
              enddo
           endif

        enddo
     enddo

     RECORD_TIME(timer_end%TMP2)
     timer_cumer%TMP2=timer_end%TMP2-timer_begin%TMP2+timer_cumer%TMP2

     ! Send the integral package after folding to master node
     ! and master node will receive them and do next two folding steps

     ! slave node will send infos
     if(.not.master) then
        tempE=quick_qm_struct%EMP2
        do j1=1,nbasis
           do k1=1,ivir
              do i1=1,iocc
                 do icycle=1,nsteplength
                    temp4d(icycle,i1,k1,j1)= orbmp2k331(icycle,i1,k1,j1)
                 enddo
              enddo
           enddo
        enddo
        ! send 3 indices integrals to master node
        call MPI_SEND(temp4d,nbasis*ivir*iocc*nsteplength,mpi_double_precision,0,mpirank,MPI_COMM_WORLD,IERROR)
        ! master node will receive infos from every nodes
     else
        do i=1,mpisize-1
           ! receive integrals from slave nodes
           call MPI_RECV(temp4d,nbasis*ivir*iocc*nsteplength,mpi_double_precision,i,i,MPI_COMM_WORLD,MPI_STATUS,IERROR)
           ! and sum them into operator
           do j1=1,nbasis
              do k1=1,ivir
                 do i1=1,iocc
                    do icycle=1,nsteplength
                       orbmp2k331(icycle,i1,k1,j1)= orbmp2k331(icycle,i1,k1,j1) +temp4d(icycle,i1,k1,j1)
                    enddo
                 enddo
              enddo
           enddo
        enddo
     endif

     !---------------- END MPI/ ALL NODES ------------------------

     !---------------- MPI/ MASTER -------------------------------

     call MPI_Reduce(ntemp, total_ntemp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD,IERROR);
     if (master) then
        !write (ioutfile,'("EFFECT INTEGRALS    =",i8)') total_ntemp

        do icycle=1,nsteplength
           i3=nstepmp2s+icycle-1
           do k3=i3,nelec/2
              ! fold 3 indices integral into 2 indices
              do J3=1,nbasis-nelec/2
                 do L3=1,nbasis-nelec/2
                    orbmp2(L3,J3)=0.0d0
                    L3new=L3+nelec/2
                    do LLL=1,nbasis
                       orbmp2(L3,J3)=orbmp2(L3,J3)+orbmp2k331(icycle,k3,j3,LLL)*quick_qm_struct%co(LLL,L3new)
                    enddo
                 enddo
              enddo

              ! Now we can get energy
              do J3=1,nbasis-nelec/2
                 do L3=1,nbasis-nelec/2
                    if(k3.gt.i3)then
                       quick_qm_struct%EMP2=quick_qm_struct%EMP2+2.0d0/(quick_qm_struct%E(i3)+ &
                          quick_qm_struct%E(k3)-quick_qm_struct%E(j3+nelec/2)-quick_qm_struct%E(l3+nelec/2)) &
                          *orbmp2(j3,l3)*(2.0d0*orbmp2(j3,l3)-orbmp2(l3,j3))
                    endif
                    if(k3.eq.i3)then
                       quick_qm_struct%EMP2=quick_qm_struct%EMP2+1.0d0/(quick_qm_struct%E(i3)+ &
                          quick_qm_struct%E(k3)-quick_qm_struct%E(j3+nelec/2)-quick_qm_struct%E(l3+nelec/2)) &
                          *orbmp2(j3,l3)*(2.0d0*orbmp2(j3,l3)-orbmp2(l3,j3))
                    endif

                 enddo
              enddo
           enddo
        enddo
     endif
     !---------------- ALL MPI/ MASTER ---------------------------

     ! sync all nodes
     call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
  enddo

  if (master) then
     write (iOutFile,'("SECOND ORDER ENERGY =",F16.9)') quick_qm_struct%EMP2
     write (iOutFile,'("EMP2                =",F16.9)') quick_qm_struct%Etot+quick_qm_struct%EMP2
     call PrtAct(ioutfile,"End MP2 Calculation")
  endif
  return
end subroutine mpi_calmp2
#endif

! Initial Orbmp2k331
subroutine  initialOrbmp2k331(orbmp2k331,nstep,nbasis,ivir,iocc,nsteplength)
  integer nbasis,ivir,iocc,nsteplength,i1,jk1,j1,icycle,nstep
  double precision orbmp2k331(nstep,iocc,ivir,nbasis)
  do j1=1,nbasis
     do k1=1,ivir
        do i1=1,iocc
           do icycle=1,nsteplength
              orbmp2k331(icycle,i1,k1,j1)=0.0d0
           enddo
        enddo
     enddo
  enddo
end subroutine initialOrbmp2k331

! Initial Orbmp2i331 and Orbmp2j331
subroutine initialOrbmp2ij(orbmp2i331,nstep,nsteplength,nbasis,nbasistemp,nbasistemp2)
  integer nstep,nsteplength,nbasis,nbasistemp,nbasistemp2
  integer l1,j1,i1,k1,icycle
  double precision orbmp2i331(nstep,nbasis,nbasistemp,nbasistemp2,2)

  do l1=1,2
     do j1=1,nbasistemp
        do i1=1,nbasistemp2
           do k1=1,nbasis
              do icycle=1,nsteplength
                 orbmp2i331(icycle,k1,i1,j1,l1)=0.0d0
              enddo
           enddo
        enddo
     enddo
  enddo
end subroutine initialOrbmp2ij

! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shellmp2(nstepmp2s,nsteplength)
   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD,AAtemp(3)
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

   COMMON /COM1/RA,RB,RC,RD

   do M=1,3
      RA(M)=xyz(M,quick_basis%katom(II))
      RB(M)=xyz(M,quick_basis%katom(JJ))
      RC(M)=xyz(M,quick_basis%katom(KK))
      RD(M)=xyz(M,quick_basis%katom(LL))
   enddo

   NII1=quick_basis%Qstart(II)
   NII2=quick_basis%Qfinal(II)
   NJJ1=quick_basis%Qstart(JJ)
   NJJ2=quick_basis%Qfinal(JJ)
   NKK1=quick_basis%Qstart(KK)
   NKK2=quick_basis%Qfinal(KK)
   NLL1=quick_basis%Qstart(LL)
   NLL2=quick_basis%Qfinal(LL)

   NNAB=(NII2+NJJ2)
   NNCD=(NKK2+NLL2)

   NABCDTYPE=NNAB*10+NNCD

   NNAB=sumindex(NNAB)
   NNCD=sumindex(NNCD)

   NNA=Sumindex(NII1-1)+1

   NNC=Sumindex(NKK1-1)+1

   NABCD=NII2+NJJ2+NKK2+NLL2
   ITT=0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         AB=Apri(Nprii,Nprij)
         ABtemp=0.5d0/AB
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do M=1,3
            P(M)=Ppri(M,Nprii,Nprij)
            AAtemp(M)=P(M)*AB
            Ptemp(M)=P(M)-RA(M)
         enddo
         !            KAB=Kpri(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               !                       print*,cutoffprim
               if(cutoffprim.gt.quick_method%primLimit)then
                  CD=Apri(Nprik,Npril)
                  ABCD=AB+CD
                  ROU=AB*CD/ABCD
                  RPQ=0.0d0
                  ABCDxiao=dsqrt(ABCD)

                  CDtemp=0.5d0/CD
                  ABcom=AB/ABCD
                  CDcom=CD/ABCD
                  ABCDtemp=0.5d0/ABCD

                  do M=1,3
                     Q(M)=Ppri(M,Nprik,Npril)
                     W(M)=(AAtemp(M)+Q(M)*CD)/ABCD
                     XXXtemp=P(M)-Q(M)
                     RPQ=RPQ+XXXtemp*XXXtemp
                     Qtemp(M)=Q(M)-RC(M)
                     WQtemp(M)=W(M)-Q(M)
                     WPtemp(M)=W(M)-P(M)
                  enddo
                  !                         KCD=Kpri(Nprik,Npril)

                  T=RPQ*ROU

                  !                         NABCD=0
                  !                         call FmT(0,T,FM)
                  !                         do iitemp=0,0
                  !                           Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                  !                         enddo
                  call FmT(NABCD,T,FM)
                  do iitemp=0,NABCD
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                  enddo
                  !                         if(II.eq.1.and.JJ.eq.4.and.KK.eq.10.and.LL.eq.16)then
                  !                          print*,III,JJJ,KKK,LLL,T,NABCD,FM(0:NABCD)
                  !                         endif
                  !                         print*,III,JJJ,KKK,LLL,FM
                  ITT=ITT+1

                  call vertical(NABCDTYPE)

                  do I2=NNC,NNCD
                     do I1=NNA,NNAB
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo
                  !                           else
                  !!                             print*,cutoffprim
                  !                             ITT=ITT+1
                  !                           do I2=NNC,NNCD
                  !                             do I1=NNA,NNAB
                  !                               Yxiao(ITT,I1,I2)=0.0d0
                  !                             enddo
                  !                           enddo
               endif
            enddo
         enddo
      enddo
   enddo


   do I=NII1,NII2
      NNA=Sumindex(I-1)+1
      do J=NJJ1,NJJ2
         NNAB=SumINDEX(I+J)
         do K=NKK1,NKK2
            NNC=Sumindex(k-1)+1
            do L=NLL1,NLL2
               NNCD=SumIndex(K+L)
               call classmp2(I,J,K,L,NNA,NNC,NNAB,NNCD,nstepmp2s,nsteplength)
               !                   call class
            enddo
         enddo
      enddo
   enddo

end subroutine shellmp2


! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classmp2(I,J,K,L,NNA,NNC,NNAB,NNCD,nstepmp2s,nsteplength)
   ! subroutine class
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
   double precision X44(129600)

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   ITT=0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%primLimit)then
                  ITT=ITT+1
                  X44(ITT)=X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
               endif
            enddo
         enddo
      enddo
   enddo

   do MM2=NNC,NNCD
      do MM1=NNA,NNAB
         Ytemp=0.0d0
         do itemp=1,ITT
            Ytemp=Ytemp+X44(itemp)*Yxiao(itemp,MM1,MM2)
            !                           Ytemp=Ytemp+Yxiao(itemp,MM1,MM2)
         enddo
         store(MM1,MM2)=Ytemp
      enddo
   enddo


   NBI1=quick_basis%Qsbasis(II,I)
   NBI2=quick_basis%Qfbasis(II,I)
   NBJ1=quick_basis%Qsbasis(JJ,J)
   NBJ2=quick_basis%Qfbasis(JJ,J)
   NBK1=quick_basis%Qsbasis(KK,K)
   NBK2=quick_basis%Qfbasis(KK,K)
   NBL1=quick_basis%Qsbasis(LL,L)
   NBL2=quick_basis%Qfbasis(LL,L)

   !       IJKLtype=1000*I+100*J+10*K+L
   IJtype=10*I+J
   KLtype=10*K+L
   IJKLtype=100*IJtype+KLtype

   !*****       if(max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0))IJKLtype=999
   if((max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0)).or.(max(I,J,K,L).ge.3))IJKLtype=999
   !       IJKLtype=999
   !      if(J.eq.0.and.L.eq.0)then

   III1=quick_basis%ksumtype(II)+NBI1
   III2=quick_basis%ksumtype(II)+NBI2
   JJJ1=quick_basis%ksumtype(JJ)+NBJ1
   JJJ2=quick_basis%ksumtype(JJ)+NBJ2
   KKK1=quick_basis%ksumtype(KK)+NBK1
   KKK2=quick_basis%ksumtype(KK)+NBK2
   LLL1=quick_basis%ksumtype(LL)+NBL1
   LLL2=quick_basis%ksumtype(LL)+NBL2


   NII1=quick_basis%Qstart(II)
   NJJ1=quick_basis%Qstart(JJ)

   NBI1=quick_basis%Qsbasis(II,NII1)
   NBJ1=quick_basis%Qsbasis(JJ,NJJ1)

   II111=quick_basis%ksumtype(II)+NBI1
   JJ111=quick_basis%ksumtype(JJ)+NBJ1

   if(II.lt.JJ.and.KK.lt.LL)then

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  call hrrwhole
                  if (dabs(Y).gt.quick_method%integralCutoff) then
                     do i3mp2=1,nsteplength
                        i3mp2new=nstepmp2s+i3mp2-1
                        atemp=quick_qm_struct%co(KKK,i3mp2new)*Y
                        btemp=quick_qm_struct%co(LLL,i3mp2new)*Y
                        IIInew=III-II111+1
                        JJJnew=JJJ-JJ111+1

                        orbmp2i331(i3mp2,LLL,IIInew,JJJnew,1)= &
                              orbmp2i331(i3mp2,LLL,IIInew,JJJnew,1)+atemp
                        orbmp2i331(i3mp2,LLL,JJJnew,IIInew,2)= &
                              orbmp2i331(i3mp2,LLL,JJJnew,IIInew,2)+atemp
                        orbmp2i331(i3mp2,KKK,IIInew,JJJnew,1)= &
                              orbmp2i331(i3mp2,KKK,IIInew,JJJnew,1)+btemp
                        orbmp2i331(i3mp2,KKK,JJJnew,IIInew,2)= &
                              orbmp2i331(i3mp2,KKK,JJJnew,IIInew,2)+btemp
                     enddo
                  endif
               enddo
            enddo
         enddo
      enddo

   else

      do III=III1,III2
         if(max(III,JJJ1).le.JJJ2)then
            do JJJ=max(III,JJJ1),JJJ2
               do KKK=KKK1,KKK2
                  if(max(KKK,LLL1).le.LLL2)then
                     do LLL=max(KKK,LLL1),LLL2

                        call hrrwhole
                        if (dabs(Y).gt.quick_method%integralCutoff) then
                           do i3mp2=1,nsteplength
                              i3mp2new=nstepmp2s+i3mp2-1
                              atemp=quick_qm_struct%co(KKK,i3mp2new)*Y
                              btemp=quick_qm_struct%co(LLL,i3mp2new)*Y

                              IIInew=III-II111+1
                              JJJnew=JJJ-JJ111+1

                              orbmp2i331(i3mp2,LLL,IIInew,JJJnew,1)= &
                                    orbmp2i331(i3mp2,LLL,IIInew,JJJnew,1)+atemp
                              if(JJJ.ne.III)then
                                 orbmp2i331(i3mp2,LLL,JJJnew,IIInew,2)= &
                                       orbmp2i331(i3mp2,LLL,JJJnew,IIInew,2)+atemp
                              endif
                              if(KKK.ne.LLL)then
                                 orbmp2i331(i3mp2,KKK,IIInew,JJJnew,1)= &
                                       orbmp2i331(i3mp2,KKK,IIInew,JJJnew,1)+btemp
                                 if(III.ne.JJJ)then
                                    orbmp2i331(i3mp2,KKK,JJJnew,IIInew,2)= &
                                          orbmp2i331(i3mp2,KKK,JJJnew,IIInew,2)+btemp
                                 endif
                              endif

                           enddo
                        endif
                     enddo
                  endif
               enddo
            enddo
         endif
      enddo

   endif

End subroutine classmp2


