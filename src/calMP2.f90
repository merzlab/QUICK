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
  if(quick_method%ffunxiao)then
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

     call cpu_time(timer_begin%TMP2)
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

     call cpu_time(timer_end%TMP2)
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
  implicit double precision(a-h,o-z)

  include "mpif.h"

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
  if(quick_method%ffunxiao)then
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
     call cpu_time(timer_begin%TMP2)
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

     call cpu_time(timer_end%TMP2)
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
