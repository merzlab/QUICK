
subroutine g2eshell
   !--------------------------------------------------------
   ! This subroutine is to Use the shell structure as initial guess
   ! to save the computational time
   ! this subroutine generates Apri, Ppri, Kpri and Xcoeff
   !--------------------------------------------------------

   use allmod
   implicit none

#ifdef MPIV
   include 'mpif.h'
#endif

   integer ics,ips,jcs,jps,itemp,itemp2,i
   integer NA,NB
   double precision AA,BB,XYZA(3),XYZB(3),DAB
   if (master) then
   ! ics cycle
   do ics=1,jshell                       ! ics is the shell no.
      do ips=1,quick_basis%kprim(ics)                ! ips is prim no. for certain shell

         ! jcs cycle
         do jcs=1,jshell
            do jps=1,quick_basis%kprim(jcs)

               ! We have ics,jcs, ips and jps, which is the prime for shell, so we can
               ! obtain its global prim. and its exponents and coeffecients
               NA=quick_basis%kstart(ics)+IPS-1         ! we can get globle prim no. for ips
               NB=quick_basis%kstart(jcs)+jps-1         ! and jps

               AA=quick_basis%gcexpo(ips,quick_basis%ksumtype(ics))    ! so we have the exponent part for ics shell ips prim
               BB=quick_basis%gcexpo(jps,quick_basis%ksumtype(jcs))    ! and jcs shell jps prim
               Apri(NA,NB)=AA+BB                     ! A'=expo(A)+expo(B)
               do i=1,3
                  XYZA(i)=xyz(i,quick_basis%katom(ics))     ! xyz(A)
                  XYZB(i)=xyz(i,quick_basis%katom(jcs))     ! xyz(B)
               enddo
               !DAB=quick_molspec%atomdistance(quick_basis%katom(ics),quick_basis%katom(jcs))
               DAB = dsqrt((xyz(1,quick_basis%katom(ics))-xyz(1,quick_basis%katom(jcs)))**2 + &
                           (xyz(2,quick_basis%katom(ics))-xyz(2,quick_basis%katom(jcs)))**2 + &
                           (xyz(3,quick_basis%katom(ics))-xyz(3,quick_basis%katom(jcs)))**2 )
                
               ! P' is the weighting center of NpriI and NpriJ
               !              expo(A)*xyz(A)+expo(B)*xyz(B)
               ! P'(A,B)  = ------------------------------
               !                 expo(A) + expo(B)
               do i=1,3
                  Ppri(i,NA,NB) = (XYZA(i)*AA + XYZB(i)*BB)/(AA+BB)
               enddo

               !                    expo(A)*expo(B)*(xyz(A)-xyz(B))^2              1
               ! K'(A,B) =  exp[ - ------------------------------------]* -------------------
               !                            expo(A)+expo(B)                  expo(A)+expo(B)
               Kpri(NA,NB) = dexp(-AA*BB/(AA+BB)*(DAB**2))/(AA+BB)

               do  itemp=quick_basis%Qstart(ics),quick_basis%Qfinal(ics)
                  do itemp2=quick_basis%Qstart(jcs),quick_basis%Qfinal(jcs)

                     ! Xcoeff(A,B,itmp1,itmp2)=K'(A,B)*a(itmp1)*a(itmp2)
                     quick_basis%Xcoeff(NA,NB,itemp,itemp2)=Kpri(NA,NB)* &

                           quick_basis%gccoeff(ips,quick_basis%ksumtype(ics)+Itemp )* &

                           quick_basis%gccoeff(jps,quick_basis%ksumtype(jcs)+Itemp2)
                  enddo
               enddo

            enddo
         enddo
      enddo
   enddo
   endif


#ifdef MPIV
      if (bMPI) then
         call MPI_BCAST(Ppri,3*jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Kpri,jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(Apri,jbasis*jbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BCAST(quick_basis%Xcoeff,jbasis*jbasis*4*4,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
         call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
      endif
#endif

end subroutine g2eshell

subroutine writeInt(iIntFile, intDim, a, b, int)
   Implicit none
   integer i,intDim, iIntFile
   integer a(intDim), b(intDim)
   double precision int(intDim)

   write(iIntFile) a, b, int

!do i = 1, intDim
 !  write(*,*) i, a(i),b(i),int(i)
!enddo
end subroutine writeInt


subroutine readInt(iIntFile, intDim, a, b, int)
   Implicit none
   integer, parameter :: llInt = selected_int_kind (16)
   integer iIntFile, i
   integer(kind=llInt) intDim
   integer a(intDim), b(intDim)
   double precision int(intDim)
   read(iIntFile) a, b, int
end subroutine readInt

!-------------------------
!  aoint
!  writen by Yipu Miao 07/16/12
!-------------------------
subroutine aoint
   !------------------------------
   !  This subroutine is used to store 2e-integral into files
   !------------------------------
   use allmod
   Implicit none
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,INTNUM, INTBEG, INTTOT, I, J
   double precision leastIntegralCutoff, t1, t2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   call PrtAct(ioutfile,"Begin Calculation 2E TO DISK")



   write(ioutfile, '("  2-ELECTRON INTEGRAL")')
   write(ioutfile, '("-----------------------------")')

   call cpu_time(timer_begin%T2eAll)  ! Terminate the timer for 2e-integrals

   call obtain_leastIntCutoff(quick_method)

#ifdef CUDA
   write(ioutfile, '("  GPU-BASED 2-ELECTRON INTEGRAL GENERATOR")')
   write(ioutfile, '("  WRITTEN BY YIPU MIAO(FLORIDA)")')
   write(ioutfile, '("  THIS PROGRAM IS UNDER TEST PHASE")')
   write(ioutfile, '("  CONTACT THE AUTHOR FOR SUPPORT")')

   call gpu_aoint(quick_method%leastIntegralCutoff, quick_method%maxIntegralCutoff, intindex, intFileName)
   inttot = intindex
#else

   if (quick_method%nodirect) then
      !call quick_open(iIntFile, intFileName, 'R', 'U', 'R',.true.)
      !open(unit=iIntFile, file=intFileName, form="unformatted", access="stream",convert='big_endian')
      open(unit=iIntFile, file=intFileName,  form="unformatted", access="stream") 
   endif


   intbeg = 0
   intindex = 0
   bufferInt = 0
   incoreIndex = 0

   do II = 1,jshell
      INTNUM = 0
      do JJ = II,jshell;
         do KK = II,jshell; do LL = KK,jshell
            if ( Ycutoff(II,JJ)*Ycutoff(KK,LL).gt. quick_method%leastIntegralCutoff) then
               dnmax = 1.0
               call shell
               intnum = intnum+1
            endif
         enddo; enddo;
      enddo

      write(ioutfile, '("  II = ",i4," INTEGRAL=",i8, "  BEGIN=", i15)') II, intnum, intbeg
      intbeg = intindex
   enddo

   if (incoreInt) then
      do i = 1, bufferInt
         aIncore(i+incoreIndex) = aBuffer(i)
         bIncore(i+incoreIndex) = bBuffer(i)
         intIncore(i+incoreIndex) = intBuffer(i)
      enddo
   else
      call writeInt(iIntFile, bufferInt, aBuffer, bBuffer, intBuffer)
   endif



   inttot = intbeg

   if (quick_method%nodirect) then
      close(iIntFile)
   endif
#endif


   call cpu_time(timer_end%T2eAll)  ! Terminate the timer for 2e-integrals
   timer_cumer%T2eAll=timer_cumer%T2eAll+timer_end%T2eAll-timer_begin%T2eAll ! add the time to cumer


   write(ioutfile, '("-----------------------------")')
   write(ioutfile, '("      TOTAL INTEGRAL     = ", i12)') inttot
   write(ioutfile, '("      INTEGRAL FILE SIZE = ", f12.2, " MB")')  &
         dble(dble(intindex) * (kind(0.0d0) + 2 * kind(I))/1024/1024)
   write(ioutfile, '("      INTEGRAL RECORD    = ", i12)') intindex / bufferSize + 1
   write(ioutfile, '("      USAGE TIME         = ", f12.2, " s")')  timer_cumer%T2eAll
   call PrtAct(ioutfile,"FINISH 2E Calculation")

end subroutine aoint


subroutine addInt
   use allmod
   Implicit none

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,INTNUM, INTBEG, INTTOT
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   double precision DENSEKI, DENSEKJ, DENSELJ, DENSELI, DENSELK, DENSEJI, DENSEII, DENSEJJ, DENSEKK
   integer  I,J,K,L
   integer*4 A, B
   integer III1, III2, JJJ1, JJJ2, KKK1, KKK2, LLL1, LLL2
   integer NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2
   logical intSkip

   integer bufferPackNum, remainingBufffer, totalInt
   integer(kind=longLongInt) :: thisBuffer

   if (quick_method%nodirect) then
      !call quick_open(iIntFile, intFileName, 'O', 'U', 'W',.true.)
      !open(unit=iIntFile, file=intFileName,  form="unformatted", access="direct", recl=kind(1.0d0)+2*kind(1), status="old")
      open(unit=iIntFile, file=intFileName,  form="unformatted", access="stream")
   endif

   bufferPackNum = intindex / bufferSize + 1
   remainingBufffer = intindex
   rewind(iIntFile)

   totalInt = 0
   incoreIndex = 0

   if (incoreInt) then
      bufferPackNum = 1
      thisBuffer = intindex
   endif

   do II = 1, bufferPackNum

      if (.not. incoreInt) then
         if (remainingBufffer .gt. bufferSize) then
            thisBuffer = bufferSize
            remainingBufffer = remainingBufffer - bufferSize
         else
            thisBuffer = remainingBufffer
         endif
         call readInt(iIntFile, thisBuffer, aBuffer, bBuffer, intBuffer)
      endif


      do i = 1, thisBuffer

         if (incoreInt) then
            A = aIncore(i)
            B = bIncore(i)
            Y = intIncore(i)
         else
            A = aBuffer(i)
            B = bBuffer(i)
            Y = intBuffer(i)
         endif
         III = int(A/nbasis) + 1
         JJJ = mod(A, nbasis) + 1
         KKK = int(B/nbasis) + 1
         LLL = mod(B, nbasis) + 1

!write(*,*) III,JJJ,KKK,LLL, Y
         if(III.lt.JJJ.and.III.lt.KKK.and.KKK.lt.LLL)then

            !write(*,*) IJKLTYPE,NABCDTYPE, Y, II,JJ,KK,LL,III,JJJ,KKK,LLL
            DENSEKI=quick_qm_struct%dense(KKK,III)
            DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
            DENSELJ=quick_qm_struct%dense(LLL,JJJ)
            DENSELI=quick_qm_struct%dense(LLL,III)
            DENSELK=quick_qm_struct%dense(LLL,KKK)
            DENSEJI=quick_qm_struct%dense(JJJ,III)

            ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
            ! can be equal.
            quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
            quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y
            quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.5d0*DENSELJ*Y
            quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.5d0*DENSEKJ*Y
            quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.5d0*DENSELI*Y
            quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-.5d0*DENSEKI*Y
            quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.5d0*DENSELI*Y
            quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-.5d0*DENSEKI*Y

         else
            if(III.LT.KKK)then
               if(III.lt.JJJ.and.KKK.lt.LLL)then

                  DENSEKI=quick_qm_struct%dense(KKK,III)
                  DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                  DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                  DENSELI=quick_qm_struct%dense(LLL,III)
                  DENSELK=quick_qm_struct%dense(LLL,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)

                  ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                  ! can be equal.

                  quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                  quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y
                  quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.5d0*DENSELJ*Y
                  quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.5d0*DENSEKJ*Y
                  quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.5d0*DENSELI*Y
                  quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-.5d0*DENSEKI*Y
                  quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.5d0*DENSELI*Y
                  quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-.5d0*DENSEKI*Y

               else if(III.eq.JJJ.and.KKK.eq.LLL)then

                  DENSEJI=quick_qm_struct%dense(KKK,III)
                  DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                  DENSEII=quick_qm_struct%dense(III,III)
                  ! Find  all the (ii|jj) integrals.
                  quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEJJ*Y
                  quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+DENSEII*Y
                  quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.5d0*DENSEJI*Y

               else if(JJJ.eq.KKK.and.JJJ.eq.LLL)then

                  DENSEJI=quick_qm_struct%dense(JJJ,III)
                  DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                  ! Find  all the (ij|jj) integrals.
                  quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+.5d0*DENSEJJ*Y
                  quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)+DENSEJI*Y
                  !        ! Find  all the (ii|ij) integrals.
                  !        ! Find all the (ij|ij) integrals


                  ! Find all the (ij|ik) integrals where j>i,k>j
               else if(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then

                  DENSEKI=quick_qm_struct%dense(KKK,III)
                  DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                  DENSEKK=quick_qm_struct%dense(KKK,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)

                  ! Find all the (ij|kk) integrals where j>i, k>j.
                  quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEKK*Y
                  quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+2.d0*DENSEJI*Y
                  quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.5d0*DENSEKJ*Y
                  quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.5d0*DENSEKI*Y
                  quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.5d0*DENSEKI*Y
                  !        ! Find all the (ik|jj) integrals where j>i, k>j.

               else if(III.eq.JJJ.and.KKK.lt.LLL)then

                  DENSEII=quick_qm_struct%dense(III,III)
                  DENSEJI=quick_qm_struct%dense(KKK,III)
                  DENSEKI=quick_qm_struct%dense(LLL,III)
                  DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                  ! Find all the (ii|jk) integrals where j>i, k>j.
                  quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+DENSEII*Y
                  quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.d0*DENSEKJ*Y
                  quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.5d0*DENSEKI*Y
                  quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.5d0*DENSEJI*Y

               endif
            else
               if(JJJ.LE.LLL)then
                  if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then

                     DENSEII=quick_qm_struct%dense(III,III)

                     ! do all the (ii|ii) integrals.
                     quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+.5d0*DENSEII*Y

                  else if(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then

                     DENSEJI=quick_qm_struct%dense(LLL,III)
                     DENSEII=quick_qm_struct%dense(III,III)

                     ! Find  all the (ii|ij) integrals.
                     quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+.5d0*DENSEII*Y
                     quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEJI*Y

                  else if(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then

                     DENSEJI=quick_qm_struct%dense(JJJ,III)
                     DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)
                     DENSEII=quick_qm_struct%dense(III,III)

                     ! Find all the (ij|ij) integrals
                     quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+1.50*DENSEJI*Y
                     quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)-.5d0*DENSEII*Y
                     quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-.5d0*DENSEJJ*Y

                  else if(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then

                     DENSEKI=quick_qm_struct%dense(LLL,III)
                     DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                     DENSEII=quick_qm_struct%dense(III,III)
                     DENSEJI=quick_qm_struct%dense(JJJ,III)

                     ! Find all the (ij|ik) integrals where j>i,k>j
                     quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+1.5d0*DENSEKI*Y
                     quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+1.5d0*DENSEJI*Y
                     quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-1.d0*DENSEKJ*Y
                     quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-.5d0*DENSEII*Y

                  endif
               endif
            endif


         endif
         1000     continue
      enddo
   enddo

   if (quick_method%nodirect) then
      close(iIntFile)
   endif

   goto 100

   ! this part is designed for integral that has accuracy that beyond 2e integral file
   quick_method%nodirect = .false.


   do II = 1,jshell
      do JJ = II,jshell
         do KK = II,jshell
            do LL = KK,jshell
               DNmax =  max(4.0d0*cutmatrix(II,JJ), &
                     4.0d0*cutmatrix(KK,LL), &
                     cutmatrix(II,LL), &
                     cutmatrix(II,KK), &
                     cutmatrix(JJ,KK), &
                     cutmatrix(JJ,LL))
               ! (IJ|KL)^2<=(II|JJ)*(KK|LL) if smaller than cutoff criteria, then
               ! ignore the calculation to save computation time
               if ( (Ycutoff(II,JJ)*Ycutoff(KK,LL)        .gt. quick_method%integralCutoff).and. &
                     (Ycutoff(II,JJ)*Ycutoff(KK,LL)*DNmax  .gt. quick_method%integralCutoff) .and. &
                     (Ycutoff(II,JJ)*Ycutoff(KK,LL)  .lt. quick_method%leastIntegralCutoff))  &
                     call shell

            enddo
         enddo

      enddo
   enddo
   quick_method%nodirect = .true.
   100 continue

end subroutine addInt


! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shell

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

  ! Get angular momenta quantum number for each of the 4 shells
  ! s=0~0, p=1~1, sp=0~1, d=2~2, f=3~3

   NII1=quick_basis%Qstart(II)
   NII2=quick_basis%Qfinal(II)
   NJJ1=quick_basis%Qstart(JJ)
   NJJ2=quick_basis%Qfinal(JJ)
   NKK1=quick_basis%Qstart(KK)
   NKK2=quick_basis%Qfinal(KK)
   NLL1=quick_basis%Qstart(LL)
   NLL2=quick_basis%Qfinal(LL)

!--------------------Madu---------------------------
!  write(*,'(A40,1x,I2,1x,I2,1x,I2,1x,I2,1x,I2,1x,I2,1x,I2,1x,I2)') &
!  "Madu: I1, I2, J1, J2, K1, K2, L1, L2", &
!  NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2
 ! stop
!--------------------Madu---------------------------

   NNAB=(NII2+NJJ2)
   NNCD=(NKK2+NLL2)

   NABCDTYPE=NNAB*10+NNCD


   NNAB=sumindex(NNAB)
   NNCD=sumindex(NNCD)
   NNA=sumindex(NII1-1)+1
   NNC=sumindex(NKK1-1)+1

   !The summation of the highest angular momentum number 
   !of each shell. 
   NABCD=NII2+NJJ2+NKK2+NLL2
   ITT=0

!--------------------Madu---------------------------
!  write(*,'(A30,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: NABCDTYPE, NNAB, NNCD, NNA, NNC, NABCD", &
!  NABCDTYPE, NNAB, NNCD, NNA, NNC, NABCD
  !stop
!--------------------Madu--------------------------

   !  the first cycle is for j prim
   !  JJJ and NpriJ are the tracking indices
   do JJJ=1,quick_basis%kprim(JJ)
     
      Nprij=quick_basis%kstart(JJ)+JJJ-1

      ! the second cycle is for i prim
      ! II and NpriI are the tracking indices
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         !For NpriI and NpriJ primitives, we calculate the following quantities
         AB=Apri(Nprii,Nprij)    ! AB = Apri = expo(NpriI)+expo(NpriJ). Eqn 8 of HGP.
         ABtemp=0.5d0/AB         ! ABtemp = 1/(2Apri) = 1/2(expo(NpriI)+expo(NpriJ))
         ! This is term is required for Eqn 6 of HGP. 
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)

         do M=1,3
            !Eqn 9 of HGP
            ! P' is the weighting center of NpriI and NpriJ
            !                           --->           --->
            ! ->  ------>       expo(I)*xyz(I)+expo(J)*xyz(J)
            ! P = P'(I,J)  = ------------------------------
            !                       expo(I) + expo(J)
            P(M)=Ppri(M,Nprii,Nprij)
           
            !Multiplication of Eqns 9  by Eqn 8 of HGP.. 
            !                        -->            -->
            ! ----->         expo(I)*xyz(I)+expo(J)*xyz(J)                                 -->            -->
            ! AAtemp = ----------------------------------- * (expo(I) + expo(J)) = expo(I)*xyz(I)+expo(J)*xyz(J)
            !                  expo(I) + expo(J)
            AAtemp(M)=P(M)*AB

            !Requires for HGP Eqn 6. 
            ! ----->   ->  ->
            ! Ptemp  = P - A
            Ptemp(M)=P(M)-RA(M)
         enddo

         ! the third cycle is for l prim
         ! LLL and npriL are the tracking indices
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1

            ! the forth cycle is for k prim
            ! the KKK and nprik are the tracking indices
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1

               ! prim cutoff: cutoffprim(I,J,K,L) = dnmax * cutprim(I,J) * cutprim(K,L)
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%primLimit)then

                  !Nita quantity of HGP Eqn 10. This is same as
                  !zita (AB) above. 
                  CD=Apri(Nprik,Npril)  ! CD = Apri = expo(NpriK) + expo(NpriL)

                  !First term of HGP Eqn 12 without sqrt. 
                  ABCD=AB+CD            ! ABCD = expo(NpriI)+expo(NpriJ)+expo(NpriK)+expo(NpriL)

                  !First term of HGP Eqn 13.
                  !         AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                  ! Rou = ----------- = ------------------------------------
                  !         AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                  ROU=AB*CD/ABCD

                  RPQ=0.0d0
                  
                  !First term of HGP Eqn 12 with sqrt. 
                  !              _______________________________
                  ! ABCDxiao = \/expo(I)+expo(J)+expo(K)+expo(L)
                  ABCDxiao=dsqrt(ABCD)

                  !Not sure why we calculate the following. 
                  CDtemp=0.5d0/CD       ! CDtemp =  1/2(expo(NpriK)+expo(NpriL))

                  !These terms are required for HGP Eqn 6.
                  !                expo(I)+expo(J)                        expo(K)+expo(L)
                  ! ABcom = --------------------------------  CDcom = --------------------------------
                  !          expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                  ABcom=AB/ABCD
                  CDcom=CD/ABCD

                  ! ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                  ABCDtemp=0.5d0/ABCD

                  do M=1,3

                     !Calculate Q of HGP 10, which is same as P above. 
                     ! Q' is the weighting center of NpriK and NpriL
                     !                           --->           --->
                     ! ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                     ! Q = P'(K,L)  = ------------------------------
                     !                       expo(K) + expo(L)
                     Q(M)=Ppri(M,Nprik,Npril)

                     !HGP Eqn 10. 
                     ! W' is the weight center for NpriI,NpriJ,NpriK and NpriL
                     !                --->             --->             --->            --->
                     ! ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                     ! W = -------------------------------------------------------------------
                     !                    expo(I) + expo(J) + expo(K) + expo(L)
                     W(M)=(AAtemp(M)+Q(M)*CD)/ABCD

                     !Required for HGP Eqn 13.
                     !        ->  ->  2
                     ! RPQ =| P - Q |
                     XXXtemp=P(M)-Q(M)
                     RPQ=RPQ+XXXtemp*XXXtemp
                        
                     !Not sure why we need the next two terms. 
                     ! ---->   ->  ->
                     ! Qtemp = Q - K
                     Qtemp(M)=Q(M)-RC(M)

                     ! ----->   ->  ->
                     ! WQtemp = W - Q
                     ! ----->   ->  ->
                     ! WPtemp = W - P
                     WQtemp(M)=W(M)-Q(M)

                     !Required for HGP Eqns 6 and 16.
                     WPtemp(M)=W(M)-P(M)
                  enddo

                  !HGP Eqn 13. 
                  !             ->  -> 2
                  ! T = ROU * | P - Q|
                  T=RPQ*ROU
                  !                         2m        2
                  ! Fm(T) = integral(1,0) {t   exp(-Tt )dt}
                  ! NABCD is the m value, and FM returns the FmT value
                  !call FmT(NABCD,T,FM)
                  call mirp_fmt(NABCD,T,FM)

                  !Go through all m values, obtain Fm values from FM array we
                  !just computed and calculate quantities required for HGP Eqn
                  !12. 
                  do iitemp=0,NABCD
                     ! Yxiaotemp(1,1,iitemp) is the starting point of recurrsion
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                     !              _______________________________
                     ! ABCDxiao = \/expo(I)+expo(J)+expo(K)+expo(L)
                  enddo

                  ITT=ITT+1
                  ! now we will do vrr and and the double-electron integral
                  call vertical(NABCDTYPE)
                  do I2=NNC,NNCD
                     do I1=NNA,NNAB
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo
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
!--------------------Madu---------------------------
!  write(*,'(A50,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: I, J, K, L, NNA, NNC, NNAB, NNCD", &
!  I, J, K, L, NNA, NNC, NNAB, NNCD 
!  stop
!--------------------Madu--------------------------
               call iclass(I,J,K,L,NNA,NNC,NNAB,NNCD)
            enddo
         enddo
      enddo
   enddo
   201 return
end subroutine shell

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine iclass(I,J,K,L,NNA,NNC,NNAB,NNCD)
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
   double precision AAtemp(3)
   double precision INTTMP(10000)
   integer INTNUM, IINT
   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp

   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

   double precision X44(129600)

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM
   integer*4 A, B
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   do MM2 = NNC, NNCD
      do MM1 = NNA, NNAB
         store(MM1,MM2) = 0
      enddo
   enddo
!Madu: Following loop may be removed. 
!Its not used anywhere within the subroutine

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


   NABCDTYPE=(NII2+NJJ2)*10+(NKK2+NLL2)

   NABCD=NII2+NJJ2+NKK2+NLL2
   itt = 0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1

      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1

         !X0 = 2.0d0*(PI)**(2.5d0), constants for HGP 15 
         ! multiplied twice for KAB and KCD

         X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)

         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1

            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)

               if(cutoffprim.gt.quick_method%primLimit)then

                  itt = itt+1
                  !This is the KAB x KCD value reqired for HGP 12.
                  !itt is the m value.
                  X44(ITT) = X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
               endif
            enddo
         enddo
      enddo
   enddo

   !Here we complete HGP 12. 
   do MM2=NNC,NNCD
      do MM1=NNA,NNAB
         Ytemp=0.0d0
         do itemp=1,ITT
            Ytemp=Ytemp+X44(itemp)*Yxiao(itemp,MM1,MM2)
         enddo
         store(MM1,MM2)=Ytemp
!write(*,*) mpirank, Ytemp
      enddo
   enddo

!Get the start and end basis numbers for each angular momentum. 
!For eg. Qsbasis and Qfbasis are 1 and 3 for P basis. 
   NBI1=quick_basis%Qsbasis(II,I)
   NBI2=quick_basis%Qfbasis(II,I)
   NBJ1=quick_basis%Qsbasis(JJ,J)
   NBJ2=quick_basis%Qfbasis(JJ,J)
   NBK1=quick_basis%Qsbasis(KK,K)
   NBK2=quick_basis%Qfbasis(KK,K)
   NBL1=quick_basis%Qsbasis(LL,L)
   NBL2=quick_basis%Qfbasis(LL,L)

!--------------------Madu---------------------------
!  write(*,'(A50,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: I1, I2, J1, J2, K1, K2, L1, L2", &
!  NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2 
!  stop
!--------------------Madu--------------------------

   IJtype=10*I+J
   KLtype=10*K+L
   IJKLtype=100*IJtype+KLtype


   if((max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0)).or.(max(I,J,K,L).ge.3))IJKLtype=999

!quick_basis%ksumtype array has a cumulative sum of number of components of all
!shells 
   III1=quick_basis%ksumtype(II)+NBI1
   III2=quick_basis%ksumtype(II)+NBI2
   JJJ1=quick_basis%ksumtype(JJ)+NBJ1
   JJJ2=quick_basis%ksumtype(JJ)+NBJ2
   KKK1=quick_basis%ksumtype(KK)+NBK1
   KKK2=quick_basis%ksumtype(KK)+NBK2
   LLL1=quick_basis%ksumtype(LL)+NBL1
   LLL2=quick_basis%ksumtype(LL)+NBL2

!write(*,'(A20,2x,I3,2x,I3,2x,I3,2x,I3)') "Madu: II, JJ, KK,LL", &
!quick_basis%ksumtype(II),quick_basis%ksumtype(JJ), quick_basis%ksumtype(KK),&
!quick_basis%ksumtype(LL)

!do imadu=1,size(quick_basis%ksumtype)
!   write(*,*) imadu, quick_basis%ksumtype(imadu)
!enddo
!stop
!--------------------Madu---------------------------
!  write(*,'(A50,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: I1, I2, J1, J2, K1, K2, L1, L2", &
!  III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------

   if (quick_method%nodirect) then
      INTNUM = 0
      do III=III1,III2
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=max(III,KKK1),KKK2
               do LLL=max(KKK,LLL1),LLL2
                  if (III.lt.JJJ .and. III.lt. KKK .and. KKK.lt. LLL) then
                     call hrrwhole
                     if (abs(Y).gt.quick_method%maxIntegralCutoff) then
                        A = (III-1)*nbasis+JJJ-1
                        B = (KKK-1)*nbasis+LLL-1
                        INTNUM=INTNUM+1

                        if (incoreInt) then
                           incoreIndex = incoreIndex + 1
                           aIncore(incoreIndex) = A
                           bIncore(incoreIndex) = B
                           intIncore(incoreIndex) = Y
                        else
                           bufferInt = bufferInt + 1
                           aBuffer(bufferInt) = A
                           bBuffer(bufferInt) = B 
                           intBuffer(bufferInt) = Y
                        endif
                        if (bufferInt .eq. bufferSize) then
                           if (incoreInt) then
                           else
                              call writeInt(iIntFile, bufferSize, aBuffer, bBuffer, intBuffer)
                           endif

                           bufferInt = 0
                        endif
                     endif
                  else if((III.LT.KKK).OR.(JJJ.LE.LLL))then
                     call hrrwhole
                     if (abs(Y).gt.quick_method%maxintegralCutoff) then

                        A = (III-1)*nbasis+JJJ-1
                        B = (KKK-1)*nbasis+LLL-1

                        INTNUM=INTNUM+1

                        if (incoreInt) then
                           incoreIndex = incoreIndex + 1
                           aIncore(incoreIndex) = A
                           bIncore(incoreIndex) = B
                           intIncore(incoreIndex) = Y
                        else
                           bufferInt = bufferInt + 1
                           aBuffer(bufferInt) = A
                           bBuffer(bufferInt) = B
                           intBuffer(bufferInt) = Y
                        endif

                        if (bufferInt .eq. bufferSize) then
                           if (incoreInt) then

                           else
                              call writeInt(iIntFile, bufferSize, aBuffer, bBuffer, intBuffer)
                           endif
                           bufferInt = 0
                        endif
                     endif
                  endif
               enddo
            enddo
         enddo
      enddo

      intindex = intindex + INTNUM
   else


      if(II.lt.JJ.and.II.lt.KK.and.KK.lt.LL)then
         do III=III1,III2
            do JJJ=JJJ1,JJJ2
               do KKK=KKK1,KKK2
                  do LLL=LLL1,LLL2
                     call hrrwhole
                     !write(*,*) Y,III,JJJ,KKK,LLL
                     DENSEKI=quick_qm_struct%dense(KKK,III)
                     DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                     DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                     DENSELI=quick_qm_struct%dense(LLL,III)
                     DENSELK=quick_qm_struct%dense(LLL,KKK)
                     DENSEJI=quick_qm_struct%dense(JJJ,III)

                     ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                     ! can be equal.
                     quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                     quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y
                     quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-quick_method%x_hybrid_coeff*.5d0*DENSELJ*Y
                     quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-quick_method%x_hybrid_coeff*.5d0*DENSEKJ*Y
                     quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-quick_method%x_hybrid_coeff*.5d0*DENSELI*Y
                     quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y
                     quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSELI*Y
                     quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y

!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = true"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL
  !stop
!--------------------Madu--------------------------

                  enddo
               enddo
            enddo
         enddo
      else
         do III=III1,III2
            do JJJ=max(III,JJJ1),JJJ2
               do KKK=max(III,KKK1),KKK2
                  do LLL=max(KKK,LLL1),LLL2
                     if(III.LT.KKK)then
                        call hrrwhole
                          !write(*,*) Y,III,JJJ,KKK,LLL
                        if(III.lt.JJJ.and.KKK.lt.LLL)then
                           DENSEKI=quick_qm_struct%dense(KKK,III)
                           DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                           DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                           DENSELI=quick_qm_struct%dense(LLL,III)
                           DENSELK=quick_qm_struct%dense(LLL,KKK)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                           ! can be equal.

                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                           quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y
                           quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-quick_method%x_hybrid_coeff*.5d0*DENSELJ*Y
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-quick_method%x_hybrid_coeff*.5d0*DENSEKJ*Y
                           quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-quick_method%x_hybrid_coeff*.5d0*DENSELI*Y
                           quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y
                           quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSELI*Y
                           quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y

!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = true"
!write(*,*) "III<JJJ and KKK<LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL
  !stop
!--------------------Madu--------------------------

                           elseif(III.eq.JJJ.and.KKK.eq.LLL)then

                           DENSEJI=quick_qm_struct%dense(KKK,III)
                           DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                           DENSEII=quick_qm_struct%dense(III,III)
                           ! Find  all the (ii|jj) integrals.
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEJJ*Y
                           quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+DENSEII*Y
                           quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III) &
                                -quick_method%x_hybrid_coeff*.5d0*DENSEJI*Y

!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = true"
!write(*,*) "III==JJJ and KKK==LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------

                           elseif(JJJ.eq.KKK.and.JJJ.eq.LLL)then

                           DENSEJI=quick_qm_struct%dense(JJJ,III)
                           DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                           ! Find  all the (ij|jj) integrals.
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEJJ*Y &
                                -quick_method%x_hybrid_coeff*.5d0*DENSEJJ*Y
                           quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)+2.0d0*DENSEJI*Y &
                                -quick_method%x_hybrid_coeff*DENSEJI*Y 
                           !        ! Find  all the (ii|ij) integrals.
                           !        ! Find all the (ij|ij) integrals

                           ! Find all the (ij|ik) integrals where j>i,k>j
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = true"
!write(*,*) "JJJ==KKK and JJJ==LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL
  !stop
!--------------------Madu--------------------------
                           elseif(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then
                           DENSEKI=quick_qm_struct%dense(KKK,III)
                           DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                           DENSEKK=quick_qm_struct%dense(KKK,KKK)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           ! Find all the (ij|kk) integrals where j>i, k>j.
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEKK*Y
                           quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+2.d0*DENSEJI*Y
                           quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-quick_method%x_hybrid_coeff*.5d0*DENSEKJ*Y
                           quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y
                           quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = true"
!write(*,*) "KKK==LLL and III<JJJ and JJJ != KKK"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------

                           !        ! Find all the (ik|jj) integrals where j>i, k>j.
                           elseif(III.eq.JJJ.and.KKK.lt.LLL)then
                           DENSEII=quick_qm_struct%dense(III,III)
                           DENSEJI=quick_qm_struct%dense(KKK,III)
                           DENSEKI=quick_qm_struct%dense(LLL,III)
                           DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                           ! Find all the (ii|jk) integrals where j>i, k>j.
                           quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+DENSEII*Y
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.d0*DENSEKJ*Y
                           quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-quick_method%x_hybrid_coeff*.5d0*DENSEKI*Y
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-quick_method%x_hybrid_coeff*.5d0*DENSEJI*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = true"
!write(*,*) "III<JJJ and KKK<LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------
                        endif

                     else
                        if(JJJ.LE.LLL)then
                           call hrrwhole
                           !   write(*,*) Y, III,JJJ,KKK,LLL
                           if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
                              DENSEII=quick_qm_struct%dense(III,III)

                              ! do all the (ii|ii) integrals.
                              quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEII*Y &
                                -quick_method%x_hybrid_coeff*.5d0*DENSEII*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = false, JJJ<LLL=true"
!write(*,*) "III==JJJ and III==KKK and III==LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------
                              elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
                              DENSEJI=quick_qm_struct%dense(LLL,III)
                              DENSEII=quick_qm_struct%dense(III,III)

                              ! Find  all the (ii|ij) integrals.
                              quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+DENSEII*Y &
                                -quick_method%x_hybrid_coeff*.5d0*DENSEII*Y
                              quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.0d0*DENSEJI*Y &
                                -quick_method%x_hybrid_coeff*DENSEJI*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = false, JJJ<LLL=true"
!write(*,*) "III==JJJ and III==KKK and III<LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------

                              elseif(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then
                              DENSEJI=quick_qm_struct%dense(JJJ,III)
                              DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)
                              DENSEII=quick_qm_struct%dense(III,III)

                              ! Find all the (ij|ij) integrals
                              quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.0d0*DENSEJI*Y &
                                -quick_method%x_hybrid_coeff*0.5d0*DENSEJI*Y
                              quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSEII*Y
                              quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-quick_method%x_hybrid_coeff*.5d0*DENSEJJ*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = false, JJJ<LLL=true"
!write(*,*) "III==KKK and JJJ==LLL and III<JJJ"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------
                              elseif(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then
                              DENSEKI=quick_qm_struct%dense(LLL,III)
                              DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                              DENSEII=quick_qm_struct%dense(III,III)
                              DENSEJI=quick_qm_struct%dense(JJJ,III)

                              ! Find all the (ij|ik) integrals where j>i,k>j
                              quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.0d0*DENSEKI*Y &
                                -quick_method%x_hybrid_coeff*0.5d0*DENSEKI*Y
                              quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+2.0d0*DENSEJI*Y &
                                - quick_method%x_hybrid_coeff*0.5d0*DENSEJI*Y
                              quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-quick_method%x_hybrid_coeff*1.d0*DENSEKJ*Y
                              quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-quick_method%x_hybrid_coeff*.5d0*DENSEII*Y
!--------------------Madu---------------------------
!write(*,*) "II<JJ and II < KK and KK<LL = false, III<KKK = false, JJJ<LLL=true"
!write(*,*) "III==KKK and III<JJJ and JJJ<LLL"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)')&
!  "Madu: II, JJ, KK, LL, I1, I2, J1, J2, K1, K2, L1, L2", &
!  II, JJ, KK, LL, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2
  !stop
!--------------------Madu--------------------------

                           endif
                        endif
                     endif
                  enddo
               enddo
            enddo
         enddo
      endif
   endif

   201 return
End subroutine iclass

! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

!    subroutine attrashell(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az, &

      !    Bx,By,Bz,Cx,Cy,Cz,Z)
subroutine attrashellenergy(IIsh,JJsh)
   use allmod
   !    use xiaoconstants
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   double precision RA(3),RB(3),RP(3)

   ! Variables needed later:
   !    pi=3.1415926535897932385

   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))

   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))

   !   Cx=sumx
   !   Cy=sumy
   !   Cz=sumz

   ! The purpose of this subroutine is to calculate the nuclear attraction
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
   ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
   ! arises from the recusion relationship.

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! The first step is generating all the necessary auxillary integrals.
   ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
   ! The values of m range from 0 to i+j+k+ii+jj+kk.

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2


   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))

         g = a+b
         Px = (a*Ax + b*Bx)/g
         Py = (a*Ay + b*By)/g
         Pz = (a*Az + b*Bz)/g
         g_table = g**(-1.5)

         constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &

               * 2.d0 * sqrt(g/Pi)

         do iatom=1,natom+quick_molspec%nextatom
            if(iatom<=natom)then
               Cx=xyz(1,iatom)
               Cy=xyz(2,iatom)
               Cz=xyz(3,iatom)
               Z=-1.0d0*quick_molspec%chg(iatom)
            else
               Cx=quick_molspec%extxyz(1,iatom-natom)
               Cy=quick_molspec%extxyz(2,iatom-natom)
               Cz=quick_molspec%extxyz(3,iatom-natom)
               Z=-quick_molspec%extchg(iatom-natom)
            endif

            PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2

            U = g* PCsquare
            !    Maxm = i+j+k+ii+jj+kk
            call FmT(Maxm,U,aux)
            do L = 0,maxm
               aux(L) = aux(L)*constant*Z
               attraxiao(1,1,L)=aux(L)
            enddo

            ! At this point all the auxillary integrals have been calculated.
            ! It is now time to decompase the attraction integral to it's
            ! auxillary integrals through the recursion scheme.  To do this we use
            ! a recursive function.

            !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &

                  !    Cx,Cy,Cz,Px,Py,Pz,g)
            NIJ1=10*NII2+NJJ2

            call nuclearattraenergy(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &

                  Cx,Cy,Cz,Px,Py,Pz,iatom)

         enddo

      enddo
   enddo

   ! Xiao HE remember to multiply Z   01/12/2008
   !    attraction = attraction*(-1.d0)* Z
   return
end subroutine attrashellenergy



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


! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shellmp2divcon(i33,ittsub)
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
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
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
                  T=RPQ*ROU

                  call FmT(NABCD,T,FM)
                  do iitemp=0,NABCD
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                  enddo
                  ITT=ITT+1

                  call vertical(NABCDTYPE)

                  do I2=NNC,NNCD
                     do I1=NNA,NNAB
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo
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
               call classmp2divcon(I,J,K,L,NNA,NNC,NNAB,NNCD,i33,ittsub)
            enddo
         enddo
      enddo
   enddo

end subroutine shellmp2divcon

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classmp2divcon(I,J,K,L,NNA,NNC,NNAB,NNCD,i33,ittsub)
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

                  KKKsub=wtospoint(ittsub,KKK)
                  LLLsub=wtospoint(ittsub,LLL)

                  atemp=quick_qm_struct%co(KKKsub,i33)*Y
                  btemp=quick_qm_struct%co(LLLsub,i33)*Y

                  IIInew=III-II111+1
                  JJJnew=JJJ-JJ111+1

                  orbmp2i331(1,LLLsub,IIInew,JJJnew,1)=orbmp2i331(1,LLLsub,IIInew,JJJnew,1)+atemp
                  orbmp2i331(1,LLLsub,JJJnew,IIInew,2)=orbmp2i331(1,LLLsub,JJJnew,IIInew,2)+atemp
                  orbmp2i331(1,KKKsub,IIInew,JJJnew,1)=orbmp2i331(1,KKKsub,IIInew,JJJnew,1)+btemp
                  orbmp2i331(1,KKKsub,JJJnew,IIInew,2)=orbmp2i331(1,KKKsub,JJJnew,IIInew,2)+btemp

               enddo
            enddo
         enddo
      enddo

   else

      do III=III1,III2
         !         if(max(III,JJJ1).le.JJJ2)then
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=KKK1,KKK2
               !            if(max(KKK,LLL1).le.LLL2)then
               do LLL=max(KKK,LLL1),LLL2

                  call hrrwhole

                  KKKsub=wtospoint(ittsub,KKK)
                  LLLsub=wtospoint(ittsub,LLL)

                  atemp=quick_qm_struct%co(KKKsub,i33)*Y
                  btemp=quick_qm_struct%co(LLLsub,i33)*Y

                  IIInew=III-II111+1
                  JJJnew=JJJ-JJ111+1

                  !                 mp2shell(KKK)=.true.
                  !                 mp2shell(LLL)=.true.

                  orbmp2i331(1,LLLsub,IIInew,JJJnew,1)=orbmp2i331(1,LLLsub,IIInew,JJJnew,1)+atemp
                  if(JJJ.ne.III)then
                     orbmp2i331(1,LLLsub,JJJnew,IIInew,2)=orbmp2i331(1,LLLsub,JJJnew,IIInew,2)+atemp
                  endif
                  if(KKK.ne.LLL)then
                     orbmp2i331(1,KKKsub,IIInew,JJJnew,1)=orbmp2i331(1,KKKsub,IIInew,JJJnew,1)+btemp
                     if(III.ne.JJJ)then
                        orbmp2i331(1,KKKsub,JJJnew,IIInew,2)=orbmp2i331(1,KKKsub,JJJnew,IIInew,2)+btemp
                     endif
                  endif

               enddo
               !            endif
            enddo
         enddo
         !        endif
      enddo

   endif

End subroutine classmp2divcon


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

! Ed Brothers. October 23, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

!    subroutine attrashell(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az, &

      !    Bx,By,Bz,Cx,Cy,Cz,Z)
subroutine attrashellopt(IIsh,JJsh)
   use allmod
   !    use xiaoconstants
   implicit double precision(a-h,o-z)
   dimension aux(0:20)
   double precision AA(3),BB(3),CC(3),PP(3)
   common /xiaoattra/attra,aux,AA,BB,CC,PP,g

   double precision RA(3),RB(3),RP(3), valopf
#ifdef MPIV
   include "mpif.h"
#endif

   Ax=xyz(1,quick_basis%katom(IIsh))
   Ay=xyz(2,quick_basis%katom(IIsh))
   Az=xyz(3,quick_basis%katom(IIsh))

   Bx=xyz(1,quick_basis%katom(JJsh))
   By=xyz(2,quick_basis%katom(JJsh))
   Bz=xyz(3,quick_basis%katom(JJsh))

   ! The purpose of this subroutine is to calculate the nuclear attraction
   ! of an electron  distributed between gtfs with orbital exponents a
   ! and b on A and B with angular momentums defined by i,j,k (a's x, y
   ! and z exponents, respectively) and ii,jj,k and kk on B with the core at
   ! (Cx,Cy,Cz) with charge Z. m is the "order" of the integral which
   ! arises from the recusion relationship.

   ! The this is taken from the recursive relation found in Obara and Saika,
   ! J. Chem. Phys. 84 (7) 1986, 3963.

   ! The first step is generating all the necessary auxillary integrals.
   ! These are (0|1/rc|0)^(m) = 2 Sqrt (g/Pi) (0||0) Fm(g(Rpc)^2)
   ! The values of m range from 0 to i+j+k+ii+jj+kk.

   NII2=quick_basis%Qfinal(IIsh)
   NJJ2=quick_basis%Qfinal(JJsh)
   Maxm=NII2+NJJ2+1+1

   do ips=1,quick_basis%kprim(IIsh)
      a=quick_basis%gcexpo(ips,quick_basis%ksumtype(IIsh))
      do jps=1,quick_basis%kprim(JJsh)
         b=quick_basis%gcexpo(jps,quick_basis%ksumtype(JJsh))

         valopf = opf(a, b, quick_basis%gccoeff(ips,quick_basis%ksumtype(IIsh)),&
         quick_basis%gccoeff(jps,quick_basis%ksumtype(JJsh)), Ax, Ay, Az, Bx, By, Bz)

         if(abs(valopf) .gt. quick_method%coreIntegralCutoff) then

           g = a+b
           Px = (a*Ax + b*Bx)/g
           Py = (a*Ay + b*By)/g
           Pz = (a*Az + b*Bz)/g
           g_table = g**(-1.5)
          
           constant = overlap(a,b,0,0,0,0,0,0,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table) &
                 * 2.d0 * sqrt(g/Pi)
          
           do iatom=1,natom+quick_molspec%nextatom
              if(quick_basis%katom(IIsh).eq.iatom.and.quick_basis%katom(JJsh).eq.iatom)then
                  continue
               else
                 if(iatom<=natom)then
                  Cx=xyz(1,iatom)
                  Cy=xyz(2,iatom)
                  Cz=xyz(3,iatom)
                  Z=-1.0d0*quick_molspec%chg(iatom)
                 else
                  Cx=quick_molspec%extxyz(1,iatom-natom)
                  Cy=quick_molspec%extxyz(2,iatom-natom)
                  Cz=quick_molspec%extxyz(3,iatom-natom)
                  Z=-1.0d0*quick_molspec%extchg(iatom-natom)
                 endif
          
                 PCsquare = (Px-Cx)**2 + (Py -Cy)**2 + (Pz -Cz)**2
          
                 U = g* PCsquare
                 !    Maxm = i+j+k+ii+jj+kk
                 call FmT(Maxm,U,aux)
                 do L = 0,maxm
                    aux(L) = aux(L)*constant*Z
                    attraxiao(1,1,L)=aux(L)
                 enddo
                 
                 do L = 0,maxm-1
                    attraxiaoopt(1,1,1,L)=2.0d0*g*(Px-Cx)*aux(L+1)
                    attraxiaoopt(2,1,1,L)=2.0d0*g*(Py-Cy)*aux(L+1)
                    attraxiaoopt(3,1,1,L)=2.0d0*g*(Pz-Cz)*aux(L+1)
                 enddo
          
                 ! At this point all the auxillary integrals have been calculated.
                 ! It is now time to decompase the attraction integral to it's
                 ! auxillary integrals through the recursion scheme.  To do this we use
                 ! a recursive function.
          
                 !    attraction = attrecurse(i,j,k,ii,jj,kk,0,aux,Ax,Ay,Az,Bx,By,Bz, &
          
                       !    Cx,Cy,Cz,Px,Py,Pz,g)
                 NIJ1=10*NII2+NJJ2
          
                 call nuclearattraopt(ips,jps,IIsh,JJsh,NIJ1,Ax,Ay,Az,Bx,By,Bz, &
          
                       Cx,Cy,Cz,Px,Py,Pz,iatom)
          
              endif
          
           enddo
         endif
      enddo
   enddo
   
   ! Xiao HE remember to multiply Z   01/12/2008
   !    attraction = attraction*(-1.d0)* Z
   return
end subroutine



!!!!!!!!BE careful of IJKLtype and Fmt.f!!!!!!!That's the difference of 6d and 10f
! vertical and hrr for gradient 10/01/2007

!Be careful of store(1,1),IASTART,size of FM, STORE, coefangxiaoL(4),coefangxiaoR(4)!!!!
! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shellopt
   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=14)
   double precision FM(0:14)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,NNABfirst,NNCDfirst
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

   COMMON /COM1/RA,RB,RC,RD

   !    logical same
   !    same = .false.

   ! print*,II,JJ,KK,LL

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

   NNABfirst=sumindex(NII2+NJJ2+1)
   NNCDfirst=sumindex(NKK2+NLL2+1)

   NNA=Sumindex(NII1-2)+1

   NNC=Sumindex(NKK1-2)+1

   NABCD=NII2+NJJ2+NKK2+NLL2

   ! For first derivative of nuclui motion, the total angular momentum is raised by 1
   NABCD=NABCD+1+1


   !print*,'NABCD=',NABCD

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
            Ptemp(M)=P(M)-RA(M)
         enddo
         !            KAB=Kpri(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%gradCutoff)then
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
                     W(M)=(P(M)*AB+Q(M)*CD)/ABCD
                     XXXtemp=P(M)-Q(M)
                     RPQ=RPQ+XXXtemp*XXXtemp
                     Qtemp(M)=Q(M)-RC(M)
                     WQtemp(M)=W(M)-Q(M)
                     WPtemp(M)=W(M)-P(M)
                  enddo
                  !                         KCD=Kpri(Nprik,Npril)

                  T=RPQ*ROU

                  call FmT(NABCD,T,FM)
                  do iitemp=0,NABCD
                     Yxiaotemp(1,1,iitemp)=FM(iitemp)/ABCDxiao
                  enddo

                  ITT=ITT+1

                  call vertical(NABCDTYPE+11)

                  !                           if(NABCDTYPE.eq.44)print*,'xiao',NABCD,FM

                  do I2=NNC,NNCDfirst
                     do I1=NNA,NNABfirst
                        Yxiao(ITT,I1,I2)=Yxiaotemp(I1,I2,0)
                     enddo
                  enddo

               endif
            enddo
         enddo
      enddo
   enddo

   ! NNA=1
   ! NNC=1

   ! allocate scratch memory for X arrays
   call allocshellopt(quick_scratch,maxcontract)

   do I=NII1,NII2
      NNA=Sumindex(I-2)+1
      do J=NJJ1,NJJ2
         NNAB=SumINDEX(I+J)
         ! change for first derivative
         NNABfirst=SumINDEX(I+J+1)
         do K=NKK1,NKK2
            NNC=Sumindex(k-2)+1
            do L=NLL1,NLL2
               NNCD=SumIndex(K+L)
               NNCDfirst=SumIndex(K+L+1)
               call classopt(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
               !                   call class
            enddo
         enddo
      enddo
   enddo

   ! deallocate scratch memory for X arrays
   call deallocshellopt(quick_scratch)

end subroutine shellopt

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classopt(I,J,K,L,NNA,NNC,NNAB,NNCD,NNABfirst,NNCDfirst)
   ! subroutine class
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)

   double precision storeaa(120,120)
   double precision storebb(120,120)
   double precision storecc(120,120)
   double precision storedd(120,120)

   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=14)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
!   double precision X44(129600)

!   double precision X44aa(4096)
!   double precision X44bb(4096)
!   double precision X44cc(4096)
!   double precision X44dd(4096)
   double precision AA,BB,CC,DD

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /xiaostoreopt/storeaa,storebb,storecc,storedd
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   ITT=0
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1
      BB=quick_basis%gcexpo(JJJ,quick_basis%ksumtype(JJ))
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1
         AA=quick_basis%gcexpo(III,quick_basis%ksumtype(II))

         X2=X0*quick_basis%Xcoeff(Nprii,Nprij,I,J)
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            DD=quick_basis%gcexpo(LLL,quick_basis%ksumtype(LL))
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               CC=quick_basis%gcexpo(KKK,quick_basis%ksumtype(KK))
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               if(cutoffprim.gt.quick_method%gradCutoff)then
                  ITT=ITT+1
                  quick_scratch%X44(ITT)=X2*quick_basis%Xcoeff(Nprik,Npril,K,L)
                  quick_scratch%X44AA(ITT)=quick_scratch%X44(ITT)*AA*2.0d0
                  quick_scratch%X44BB(ITT)=quick_scratch%X44(ITT)*BB*2.0d0
                  quick_scratch%X44CC(ITT)=quick_scratch%X44(ITT)*CC*2.0d0
                  !                       X44DD(ITT)=X44(ITT)*DD*2.0d0
               endif
            enddo
         enddo
      enddo
   enddo

   !       do III=1,quick_basis%kprim(II)
   !         AA=gcexpo(III,quick_basis%ksumtype(II))
   do MM2=NNC,NNCD
      do MM1=NNA,NNAB
         Ytemp=0.0d0
         !                           YtempAA=0.0d0
         !                           YtempBB=0.0d0
         !                           YtempCC=0.0d0
         do itemp=1,ITT
            Ytemp=Ytemp+quick_scratch%X44(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempAA=YtempAA+X44AA(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempBB=YtempBB+X44BB(itemp)*Yxiao(itemp,MM1,MM2)
            !                           YtempCC=YtempCC+X44CC(itemp)*Yxiao(itemp,MM1,MM2)

         enddo
         store(MM1,MM2)=Ytemp
         !                         storeAA(MM1,MM2)=YtempAA
         !                         storeBB(MM1,MM2)=YtempBB
         !                         storeCC(MM1,MM2)=YtempCC

      enddo
   enddo


   do MM2=NNC,NNCDfirst
      do MM1=NNA,NNABfirst
         YtempAA=0.0d0
         YtempBB=0.0d0
         YtempCC=0.0d0
         do itemp=1,ITT
            YtempAA=YtempAA+quick_scratch%X44AA(itemp)*Yxiao(itemp,MM1,MM2)
            YtempBB=YtempBB+quick_scratch%X44BB(itemp)*Yxiao(itemp,MM1,MM2)
            YtempCC=YtempCC+quick_scratch%X44CC(itemp)*Yxiao(itemp,MM1,MM2)

            !                  if(II.eq.1.and.JJ.eq.1.and.KK.eq.13.and.LL.eq.13)then
            !                    print*,KKK-39,LLL-39,III+12,JJJ+12,itemp,X44AA(itemp),Yxiao(itemp,MM1,MM2)
            !                  endif

         enddo
         storeAA(MM1,MM2)=YtempAA
         storeBB(MM1,MM2)=YtempBB
         storeCC(MM1,MM2)=YtempCC

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

   !    iA = quick_basis%ncenter(II)
   !    iB = quick_basis%ncenter(JJ)
   !    iC = quick_basis%ncenter(KK)
   !    iD = quick_basis%ncenter(LL)

   !    same = iA.eq.iB .and. iB.eq.iC.and. iC.eq.iD

   !    print*,II,JJ,KK,LL

   !    if (same.eq..true.) return

   !    iAstart = (iA-1)*3
   !    iBstart = (iB-1)*3
   !    iCstart = (iC-1)*3
   !    iDstart = (iD-1)*3

   Agrad1=0.d0
   Bgrad1=0.d0
   Cgrad1=0.d0
   Dgrad1=0.d0
   Agrad2=0.d0
   Bgrad2=0.d0
   Cgrad2=0.d0
   Dgrad2=0.d0
   Agrad3=0.d0
   Bgrad3=0.d0
   Cgrad3=0.d0
   Dgrad3=0.d0

   !       IJKLtype=1000*I+100*J+10*K+L
   IJtype=10*I+J
   KLtype=10*K+L
   IJKLtype=100*IJtype+KLtype

   !       if(max(I,J,K,L).eq.2.and.(J.ne.0.or.L.ne.0))IJKLtype=999
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

   iA = quick_basis%ncenter(III2)
   iB = quick_basis%ncenter(JJJ2)
   iC = quick_basis%ncenter(KKK2)
   iD = quick_basis%ncenter(LLL2)

   iAstart = (iA-1)*3
   iBstart = (iB-1)*3
   iCstart = (iC-1)*3
   iDstart = (iD-1)*3

   if(II.lt.JJ.and.II.lt.KK.and.KK.lt.LL)then

      !       do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
      !         do JJJ=quick_basis%ksumtype(JJ)+NBJ1,quick_basis%ksumtype(JJ)+NBJ2
      !            do KKK=quick_basis%ksumtype(KK)+NBK1,quick_basis%ksumtype(KK)+NBK2
      !              do LLL=quick_basis%ksumtype(LL)+NBL1,quick_basis%ksumtype(LL)+NBL2

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  call hrrwholeopt

                  DENSEKI=quick_qm_struct%dense(KKK,III)
                  DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                  DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                  DENSELI=quick_qm_struct%dense(LLL,III)
                  DENSELK=quick_qm_struct%dense(LLL,KKK)

                  DENSEJI=quick_qm_struct%dense(JJJ,III)
                  ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                  ! can be equal.

                  !                    O(JJJ,III) = O(JJJ,III)+2.d0*DENSELK*Y
                  !                    O(LLL,KKK) = O(LLL,KKK)+2.d0*DENSEJI*Y
                  !                    O(KKK,III) = O(KKK,III)-.5d0*DENSELJ*Y
                  !                    O(LLL,III) = O(LLL,III)-.5d0*DENSEKJ*Y
                  !                        O(JJJ,KKK) = O(JJJ,KKK)-.5d0*DENSELI*Y
                  !                        O(JJJ,LLL) = O(JJJ,LLL)-.5d0*DENSEKI*Y
                  !                        O(KKK,JJJ) = O(KKK,JJJ)-.5d0*DENSELI*Y
                  !                        O(LLL,JJJ) = O(LLL,JJJ)-.5d0*DENSEKI*Y

                  constant = (4.d0*DENSEJI*DENSELK-quick_method%x_hybrid_coeff*DENSEKI*DENSELJ &
                        -quick_method%x_hybrid_coeff*DENSELI*DENSEKJ)

!--------------------Madu---------------------------
!  write(*,*) "II<JJ and II < KK and KK<LL = true"
!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL
  !stop

!--------------------Madu--------------------------

                  !                    print*,'here',constant

                  Agrad1=Agrad1+Yaa(1)*constant
                  Agrad2=Agrad2+Yaa(2)*constant
                  Agrad3=Agrad3+Yaa(3)*constant
                  Bgrad1=Bgrad1+Ybb(1)*constant
                  Bgrad2=Bgrad2+Ybb(2)*constant
                  Bgrad3=Bgrad3+Ybb(3)*constant
                  Cgrad1=Cgrad1+Ycc(1)*constant
                  Cgrad2=Cgrad2+Ycc(2)*constant
                  Cgrad3=Cgrad3+Ycc(3)*constant

               enddo
            enddo
         enddo
      enddo
 !  endif !

!if (.false.) then
   else

      do III=III1,III2
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=max(III,KKK1),KKK2
               do LLL=max(KKK,LLL1),LLL2

                  if(III.LT.KKK)then

                     call hrrwholeopt

                     if(III.lt.JJJ.and.KKK.lt.LLL)then
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)

                        DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                        DENSELI=quick_qm_struct%dense(LLL,III)
                        DENSELK=quick_qm_struct%dense(LLL,KKK)

                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                        ! can be equal.
                        constant = (4.d0*DENSEJI*DENSELK-quick_method%x_hybrid_coeff*DENSEKI*DENSELJ &

                              -quick_method%x_hybrid_coeff*DENSELI*DENSEKJ)

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        !                      print*,III,JJJ,KKK,LLL,Yaa(1)*constant,Ybb(1)*constant,Ycc(1)*constant, &

                              !                             Yaa(2)*constant,Ybb(2)*constant,Ycc(2)*constant, &

                              !                             Yaa(3)*constant,Ybb(3)*constant,Ycc(3)*constant

                        !    ! do all the (ii|ii) integrals.
                        !        ! Set some variables to reduce access time for some of the more
                        !        ! used quantities. (AGAIN)
                        ElseIf(III.eq.JJJ.and.KKK.eq.LLL)then
                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)

                        ! Find  all the (ii|jj) integrals.
                        !            O(III,III) = O(III,III)+DENSEJJ*Y
                        !            O(KKK,KKK) = O(KKK,KKK)+DENSEII*Y
                        !            O(KKK,III) = O(KKK,III)-.5d0*DENSEJI*Y

!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL

                        constant = (DENSEII*DENSEJJ-.5d0*quick_method%x_hybrid_coeff*DENSEJI*DENSEJI)

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        elseif(JJJ.eq.KKK.and.JJJ.eq.LLL)then
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                        ! Find  all the (ij|jj) integrals.
                        !            O(JJJ,III) = O(JJJ,III)+.5d0*DENSEJJ*Y
                        !            O(JJJ,JJJ) = O(JJJ,JJJ)+DENSEJI*Y

                        constant = 2.0d0*DENSEJJ*DENSEJI-quick_method%x_hybrid_coeff*DENSEJJ*DENSEJI

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        !        ! Find  all the (ii|ij) integrals.
                        !
                        !        ! Find all the (ij|ij) integrals
                        !
                        ! Find all the (ij|ik) integrals where j>i,k>j
                        elseif(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                        DENSEKK=quick_qm_struct%dense(KKK,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)

                        ! Find all the (ij|kk) integrals where j>i, k>j.
                        !                O(JJJ,III) = O(JJJ,III)+DENSEKK*Y
                        !                O(KKK,KKK) = O(KKK,KKK)+2.d0*DENSEJI*Y
                        !                O(KKK,III) = O(KKK,III)-.5d0*DENSEKJ*Y
                        !                O(KKK,JJJ) = O(KKK,JJJ)-.5d0*DENSEKI*Y
                        !                O(JJJ,KKK) = O(JJJ,KKK)-.5d0*DENSEKI*Y

                       constant=(2.d0*DENSEJI*DENSEKK-quick_method%x_hybrid_coeff*DENSEKI*DENSEKJ)

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                        !            ! Find all the (ik|jj) integrals where j>i, k>j.
                        elseif(III.eq.JJJ.and.KKK.lt.LLL)then
                        DENSEII=quick_qm_struct%dense(III,III)
                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEKI=quick_qm_struct%dense(LLL,III)
                        DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                        ! Find all the (ii|jk) integrals where j>i, k>j.
                        !                O(LLL,KKK) = O(LLL,KKK)+DENSEII*Y
                        !                O(III,III) = O(III,III)+2.d0*DENSEKJ*Y
                        !                O(KKK,III) = O(KKK,III)-.5d0*DENSEKI*Y
                        !                O(LLL,III) = O(LLL,III)-.5d0*DENSEJI*Y

!  write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
!  "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
!  II, JJ, KK, LL, III,JJJ,KKK,LLL

                        constant = (2.d0*DENSEKJ*DENSEII-quick_method%x_hybrid_coeff*DENSEJI*DENSEKI)

                        Agrad1=Agrad1+Yaa(1)*constant
                        Agrad2=Agrad2+Yaa(2)*constant
                        Agrad3=Agrad3+Yaa(3)*constant
                        Bgrad1=Bgrad1+Ybb(1)*constant
                        Bgrad2=Bgrad2+Ybb(2)*constant
                        Bgrad3=Bgrad3+Ybb(3)*constant
                        Cgrad1=Cgrad1+Ycc(1)*constant
                        Cgrad2=Cgrad2+Ycc(2)*constant
                        Cgrad3=Cgrad3+Ycc(3)*constant

                     endif

                  else
                     if(JJJ.LE.LLL)then

                        !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                        call hrrwholeopt

                        if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! do all the (ii|ii) integrals.
                           !        O(III,III) = O(III,III)+.5d0*DENSEII*Y

                           constant=0.0d0

                           elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
                           DENSEJI=quick_qm_struct%dense(LLL,III)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find  all the (ii|ij) integrals.
                           !            O(LLL,III) = O(LLL,III)+.5d0*DENSEII*Y
                           !            O(III,III) = O(III,III)+DENSEJI*Y

                           constant = 2.0d0*DENSEJI*DENSEII-quick_method%x_hybrid_coeff*DENSEJI*DENSEII

                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                           elseif(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then
                           DENSEJI=quick_qm_struct%dense(JJJ,III)
                           DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find all the (ij|ij) integrals
                           !            O(JJJ,III) = O(JJJ,III)+1.50*DENSEJI*Y
                           !            O(JJJ,JJJ) = O(JJJ,JJJ)-.5d0*DENSEII*Y
                           !            O(III,III) = O(III,III)-.5d0*DENSEJJ*Y

                           constant =(2.0d0*DENSEJI*DENSEJI-0.5d0*quick_method%x_hybrid_coeff*DENSEJI*DENSEJI &
                           -0.50d0*quick_method%x_hybrid_coeff*DENSEJJ*DENSEII)

                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                           elseif(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then
                           DENSEKI=quick_qm_struct%dense(LLL,III)
                           DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                           !                DENSEKK=quick_qm_struct%dense(LLL,LLL)
                           DENSEII=quick_qm_struct%dense(III,III)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           ! Find all the (ij|ik) integrals where j>i,k>j
                           !                O(JJJ,III) = O(JJJ,III)+1.5d0*DENSEKI*Y
                           !                O(LLL,III) = O(LLL,III)+1.5d0*DENSEJI*Y
                           !                O(III,III) = O(III,III)-1.d0*DENSEKJ*Y
                           !                O(LLL,JJJ) = O(LLL,JJJ)-.5d0*DENSEII*Y

  !write(*,'(A52,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3,1x,I3)') &
 ! "Madu: II, JJ, KK, LL, III, JJJ, KKK, LLL", &
 ! II, JJ, KK, LL, III,JJJ,KKK,LLL

                           constant = (4.0d0*DENSEJI*DENSEKI-quick_method%x_hybrid_coeff*DENSEJI*DENSEKI &
                           -quick_method%x_hybrid_coeff*DENSEKJ*DENSEII)

                           Agrad1=Agrad1+Yaa(1)*constant
                           Agrad2=Agrad2+Yaa(2)*constant
                           Agrad3=Agrad3+Yaa(3)*constant
                           Bgrad1=Bgrad1+Ybb(1)*constant
                           Bgrad2=Bgrad2+Ybb(2)*constant
                           Bgrad3=Bgrad3+Ybb(3)*constant
                           Cgrad1=Cgrad1+Ycc(1)*constant
                           Cgrad2=Cgrad2+Ycc(2)*constant
                           Cgrad3=Cgrad3+Ycc(3)*constant

                        endif

                     endif
                  endif

               enddo
            enddo
         enddo
      enddo
   endif
!if (abs(agrad1).gt.0.or.abs(agrad2).gt.0.or.abs(agrad3).gt.0.or. &
!    abs(bgrad1).gt.0.or.abs(bgrad2).gt.0.or.abs(bgrad3).gt.0.or. &
!    abs(cgrad1).gt.0.or.abs(cgrad2).gt.0.or.abs(cgrad3).gt.0 ) then
!write(*,*) II,JJ,KK,LL,I,J,K,L, Agrad1, Agrad2, Agrad3, Bgrad1, Bgrad2, &
!Bgrad3,Cgrad1,Cgrad2,Cgrad3
!endif

   quick_qm_struct%gradient(iASTART+1) = quick_qm_struct%gradient(iASTART+1)+ &

         AGrad1
   quick_qm_struct%gradient(iBSTART+1) = quick_qm_struct%gradient(iBSTART+1)+ &

         BGrad1
   quick_qm_struct%gradient(iCSTART+1) = quick_qm_struct%gradient(iCSTART+1)+ &

         CGrad1
   quick_qm_struct%gradient(iDSTART+1) = quick_qm_struct%gradient(iDSTART+1) &

         -AGrad1-BGrad1-CGrad1

   quick_qm_struct%gradient(iASTART+2) = quick_qm_struct%gradient(iASTART+2)+ &

         AGrad2
   quick_qm_struct%gradient(iBSTART+2) = quick_qm_struct%gradient(iBSTART+2)+ &

         BGrad2
   quick_qm_struct%gradient(iCSTART+2) = quick_qm_struct%gradient(iCSTART+2)+ &

         CGrad2
   quick_qm_struct%gradient(iDSTART+2) = quick_qm_struct%gradient(iDSTART+2) &

         -AGrad2-BGrad2-CGrad2

   quick_qm_struct%gradient(iASTART+3) = quick_qm_struct%gradient(iASTART+3)+ &

         AGrad3
   quick_qm_struct%gradient(iBSTART+3) = quick_qm_struct%gradient(iBSTART+3)+ &

         BGrad3
   quick_qm_struct%gradient(iCSTART+3) = quick_qm_struct%gradient(iCSTART+3)+ &

         CGrad3
   quick_qm_struct%gradient(iDSTART+3) = quick_qm_struct%gradient(iDSTART+3) &

         -AGrad3-BGrad3-CGrad3

!write(*,*) " II JJ KK LL", II, JJ, KK,LL
!write(*,*) " I J K L", I, J, K,L

!write(*,*) AGrad1, AGrad2, AGrad3
!write(*,*) BGrad1, BGrad2, BGrad3
!write(*,*) CGrad1, CGrad2, CGrad3
!write(*,*) iASTART, iBSTART, iCSTART, iDSTART

End subroutine classopt



! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shelldft(IItemp,JJtemp,KKtemp,LLtemp)

   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp
   COMMON /COM1/RA,RB,RC,RD

   II=IItemp
   JJ=JJtemp
   KK=KKtemp
   LL=LLtemp

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

!--------------------Madu---------------------------
!  write(*,'(A30,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2", &
!  NII1, NII2, NJJ1, NJJ2, NKK1, NKK2, NLL1, NLL2
!  stop
!--------------------Madu--------------------------

   NNAB=(NII2+NJJ2)
   NNCD=(NKK2+NLL2)

   NABCDTYPE=NNAB*10+NNCD


   NNAB=sumindex(NNAB)
   NNCD=sumindex(NNCD)
   NNA=Sumindex(NII1-1)+1
   NNC=Sumindex(NKK1-1)+1
   NABCD=NII2+NJJ2+NKK2+NLL2
   ITT=0

!--------------------Madu---------------------------
!  write(*,'(A30,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: NABCDTYPE, NNAB, NNCD, NNA, NNC, NABCD", &
!  NABCDTYPE, NNAB, NNCD, NNA, NNC, NABCD
  !stop
!--------------------Madu--------------------------

   !  the first cycle is for j prim
   !  JJJ and NpriJ are the tracking indices
   do JJJ=1,quick_basis%kprim(JJ)
      Nprij=quick_basis%kstart(JJ)+JJJ-1

      ! the second cycle is for i prim
      ! II and NpriI are the tracking indices
      do III=1,quick_basis%kprim(II)
         Nprii=quick_basis%kstart(II)+III-1

         AB=Apri(Nprii,Nprij) ! AB = Apri = expo(NpriI)+expo(NpriJ)
         ABtemp=0.5d0/AB ! ABtemp = 1/(2Apri) = 1/2(expo(NpriI)+expo(NpriJ))
         cutoffprim1=dnmax*cutprim(Nprii,Nprij)

         do M=1,3

            P(M)=Ppri(M,Nprii,Nprij)
            Ptemp(M)=P(M)-RA(M)
         enddo
         !            KAB=Kpri(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               !                       print*,cutoffprim,quick_method%primLimit
               !                       stop
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
                     W(M)=(P(M)*AB+Q(M)*CD)/ABCD
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
                     !                           print*,iitemp,FM(iitemp),ABCDxiao,Yxiaotemp(1,1,iitemp)
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
               call classdft(I,J,K,L,NNA,NNC,NNAB,NNCD)
               !                   call class
!--------------------Madu---------------------------
!  write(*,'(A50,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5,2x,I5)') &
!  "Madu: I, J, K, L, NNA, NNC, NNAB, NNCD", &
!  I, J, K, L, NNA, NNC, NNAB, NNCD
  !stop
!--------------------Madu--------------------------
               !call iclass(I,J,K,L,NNA,NNC,NNAB,NNCD) !Madu
            enddo
         enddo
      enddo
   enddo

end subroutine

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classdft(I,J,K,L,NNA,NNC,NNAB,NNCD)
   ! subroutine class
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
   double precision X44(1296)

   COMMON /COM1/RA,RB,RC,RD
   COMMON /COM2/AA,BB,CC,DD,AB,CD,ROU,ABCD
   COMMON /COM4/P,Q,W
   COMMON /COM5/FM

   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /xiaostore/store
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

!-----Madu--------------
   do MM2 = NNC, NNCD
      do MM1 = NNA, NNAB
         store(MM1,MM2) = 0
      enddo
   enddo

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
    

   NABCDTYPE=(NII2+NJJ2)*10+(NKK2+NLL2) 
    
   NABCD=NII2+NJJ2+NKK2+NLL2 

 !-----Madu--------------

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

   if(II.lt.JJ.and.II.lt.KK.and.KK.lt.LL)then

      !       do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
      !         do JJJ=quick_basis%ksumtype(JJ)+NBJ1,quick_basis%ksumtype(JJ)+NBJ2
      !            do KKK=quick_basis%ksumtype(KK)+NBK1,quick_basis%ksumtype(KK)+NBK2
      !              do LLL=quick_basis%ksumtype(LL)+NBL1,quick_basis%ksumtype(LL)+NBL2

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  !                 if(III.eq.1.and.JJJ.eq.2.and.KKK.eq.2.and.LLL.eq.30)then
                  !                   print*,'xiao',IJtype,KLtype,IJKLtype,KLMN(1:3,III),KLMN(1:3,JJJ),KLMN(1:3,KKK),KLMN(1:3,LLL), &

                        !                         store(1,18),store(1,8),store(1,2),store(1,1),RC(1)-RD(1)
                  !                 endif
                  !                 if(III.eq.1.and.JJJ.eq.4.and.KKK.eq.7.and.LLL.eq.19)then
                  !                   print*,'xiao',IJtype,KLtype,IJKLtype,KLMN(1:3,III),KLMN(1:3,JJJ),KLMN(1:3,KKK),KLMN(1:3,LLL), &

                        !                         store(3,12)*dsqrt(3.0d0),III,JJJ,KKK,LLL,'xiao11'
                  !                 endif

                  !                 if(III.eq.1.and.JJJ.eq.10.and.KKK.eq.41.and.LLL.eq.47)then
                  !                   print*,'xiao',IJtype,KLtype,IJKLtype,KLMN(1:3,III),KLMN(1:3,JJJ),KLMN(1:3,KKK),KLMN(1:3,LLL), &

                        !                         store(1,1),III,JJJ,KKK,LLL,'xiao00'
                  !                           do itemp=1,ITT
                  !                           print*,itemp,X44(itemp)*Yxiao(itemp,1,1)
                  !                           enddo
                  !                         stop
                  !                 endif

                  !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                  call hrrwhole

                  DENSELK=quick_qm_struct%dense(LLL,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)
                  ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                  ! can be equal.

                  quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                  quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y

                  !                      print*,III,JJJ,KKK,LLL,Y
               enddo
            enddo
         enddo
      enddo

   else

      !       do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
      !         do JJJ=max(III,quick_basis%ksumtype(JJ)+NBJ1),quick_basis%ksumtype(JJ)+NBJ2
      !            do KKK=max(III,quick_basis%ksumtype(KK)+NBK1),quick_basis%ksumtype(KK)+NBK2
      !              do LLL=max(KKK,quick_basis%ksumtype(LL)+NBL1),quick_basis%ksumtype(LL)+NBL2

      do III=III1,III2
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=max(III,KKK1),KKK2
               do LLL=max(KKK,LLL1),LLL2

                  if(III.LT.KKK)then

                     !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                     !                 if(III.eq.1.and.JJJ.eq.2.and.KKK.eq.2.and.LLL.eq.30)then
                     !                   print*,'xiao',store(1,18),store(1,8),store(1,2),store(1,1),RC(1)-RD(1)
                     !                 endif
                     call hrrwhole

                     if(III.lt.JJJ.and.KKK.lt.LLL)then
                        DENSELK=quick_qm_struct%dense(LLL,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                        ! can be equal.

                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                        quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y

                        !                      print*,III,JJJ,KKK,LLL,Y

                        !    ! do all the (ii|ii) integrals.
                        !        ! Set some variables to reduce access time for some of the more
                        !        ! used quantities. (AGAIN)
                        elseif(III.eq.JJJ.and.KKK.eq.LLL)then
                        DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)

                        ! Find  all the (ii|jj) integrals.
                        quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEJJ*Y
                        quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+DENSEII*Y

                        elseif(JJJ.eq.KKK.and.JJJ.eq.LLL)then
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                        ! Find  all the (ij|jj) integrals.
                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEJJ*Y
                        quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)+2.0d0*DENSEJI*Y

                        !        ! Find  all the (ii|ij) integrals.
                        !
                        !        ! Find all the (ij|ij) integrals
                        !
                        ! Find all the (ij|ik) integrals where j>i,k>j
                        elseif(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then
                        DENSEKK=quick_qm_struct%dense(KKK,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)

                        ! Find all the (ij|kk) integrals where j>i, k>j.
                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEKK*Y
                        quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+2.d0*DENSEJI*Y

                        !            ! Find all the (ik|jj) integrals where j>i, k>j.
                        elseif(III.eq.JJJ.and.KKK.lt.LLL)then
                        DENSEII=quick_qm_struct%dense(III,III)
                        DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                        ! Find all the (ii|jk) integrals where j>i, k>j.
                        quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+DENSEII*Y
                        quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.d0*DENSEKJ*Y

                     endif

                  else
                     if(JJJ.LE.LLL)then

                        !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                        call hrrwhole

                        if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! do all the (ii|ii) integrals.
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEII*Y

                           elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
                           DENSEJI=quick_qm_struct%dense(LLL,III)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find  all the (ii|ij) integrals.
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+DENSEII*Y
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.0d0*DENSEJI*Y

                           elseif(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           ! Find all the (ij|ij) integrals
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.0d0*DENSEJI*Y

                           elseif(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then
                           DENSEKI=quick_qm_struct%dense(LLL,III)
                           !                DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                           !                DENSEKK=quick_qm_struct%dense(LLL,LLL)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           ! Find all the (ij|ik) integrals where j>i,k>j
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.0d0*DENSEKI*Y
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+2.0d0*DENSEJI*Y

                        endif

                     endif
                  endif
               enddo
            enddo
         enddo
      enddo
   endif

End subroutine

! Vertical Recursion by Xiao HE 07/07/07 version
subroutine shelldftb3lyp(IItemp,JJtemp,KKtemp,LLtemp)
   use allmod

   Implicit double precision(a-h,o-z)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)

   double precision Qtemp(3),WQtemp(3),CDtemp,ABcom,Ptemp(3),WPtemp(3),ABtemp,CDcom,ABCDtemp
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   COMMON /VRRcom/Qtemp,WQtemp,CDtemp,ABcom,Ptemp,WPtemp,ABtemp,CDcom,ABCDtemp

   COMMON /COM1/RA,RB,RC,RD

   II=IItemp
   JJ=JJtemp
   KK=KKtemp
   LL=LLtemp

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
            Ptemp(M)=P(M)-RA(M)
         enddo
         !            KAB=Kpri(Nprii,Nprij)
         do LLL=1,quick_basis%kprim(LL)
            Npril=quick_basis%kstart(LL)+LLL-1
            do KKK=1,quick_basis%kprim(KK)
               Nprik=quick_basis%kstart(KK)+KKK-1
               cutoffprim=cutoffprim1*cutprim(Nprik,Npril)
               !                       print*,cutoffprim,quick_method%primLimit
               !                       stop
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
                     W(M)=(P(M)*AB+Q(M)*CD)/ABCD
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
                     !                           print*,iitemp,FM(iitemp),ABCDxiao,Yxiaotemp(1,1,iitemp)
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
               call classdftb3lyp(I,J,K,L,NNA,NNC,NNAB,NNCD)
               !                   call class
            enddo
         enddo
      enddo
   enddo

end subroutine

! Horrizontal recursion and Fock matrix builder by Xiao HE 07/07/07 version
subroutine classdftb3lyp(I,J,K,L,NNA,NNC,NNAB,NNCD)
   ! subroutine class
   use allmod

   Implicit double precision(A-H,O-Z)
   double precision store(120,120)
   INTEGER NA(3),NB(3),NC(3),ND(3)
   double precision P(3),Q(3),W(3),KAB,KCD
   Parameter(NN=13)
   double precision FM(0:13)
   double precision RA(3),RB(3),RC(3),RD(3)
   double precision X44(1296)

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

   if(II.lt.JJ.and.II.lt.KK.and.KK.lt.LL)then

      !       do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
      !         do JJJ=quick_basis%ksumtype(JJ)+NBJ1,quick_basis%ksumtype(JJ)+NBJ2
      !            do KKK=quick_basis%ksumtype(KK)+NBK1,quick_basis%ksumtype(KK)+NBK2
      !              do LLL=quick_basis%ksumtype(LL)+NBL1,quick_basis%ksumtype(LL)+NBL2

      do III=III1,III2
         do JJJ=JJJ1,JJJ2
            do KKK=KKK1,KKK2
               do LLL=LLL1,LLL2

                  !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                  call hrrwhole

                  DENSEKI=quick_qm_struct%dense(KKK,III)
                  DENSEKJ=quick_qm_struct%dense(KKK,JJJ)
                  DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                  DENSELI=quick_qm_struct%dense(LLL,III)

                  DENSELK=quick_qm_struct%dense(LLL,KKK)
                  DENSEJI=quick_qm_struct%dense(JJJ,III)
                  ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                  ! can be equal.

                  quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                  quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y

                  quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.1d0*DENSELJ*Y
                  quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.1d0*DENSEKJ*Y
                  quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.1d0*DENSELI*Y
                  quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-.1d0*DENSEKI*Y
                  quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.1d0*DENSELI*Y
                  quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-.1d0*DENSEKI*Y

                  !                      print*,III,JJJ,KKK,LLL,Y
               enddo
            enddo
         enddo
      enddo

   else

      !       do III=quick_basis%ksumtype(II)+NBI1,quick_basis%ksumtype(II)+NBI2
      !         do JJJ=max(III,quick_basis%ksumtype(JJ)+NBJ1),quick_basis%ksumtype(JJ)+NBJ2
      !            do KKK=max(III,quick_basis%ksumtype(KK)+NBK1),quick_basis%ksumtype(KK)+NBK2
      !              do LLL=max(KKK,quick_basis%ksumtype(LL)+NBL1),quick_basis%ksumtype(LL)+NBL2

      do III=III1,III2
         do JJJ=max(III,JJJ1),JJJ2
            do KKK=max(III,KKK1),KKK2
               do LLL=max(KKK,LLL1),LLL2

                  if(III.LT.KKK)then

                     !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                     !                 if(III.eq.1.and.JJJ.eq.2.and.KKK.eq.2.and.LLL.eq.30)then
                     !                   print*,'xiao',store(1,18),store(1,8),store(1,2),store(1,1),RC(1)-RD(1)
                     !                 endif
                     call hrrwhole

                     if(III.lt.JJJ.and.KKK.lt.LLL)then
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)

                        DENSELJ=quick_qm_struct%dense(LLL,JJJ)
                        DENSELI=quick_qm_struct%dense(LLL,III)

                        DENSELK=quick_qm_struct%dense(LLL,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        ! Find the (ij|kl) integrals where j>i,k>i,l>k. Note that k and j
                        ! can be equal.

                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+2.d0*DENSELK*Y
                        quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+2.d0*DENSEJI*Y

                        quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.1d0*DENSELJ*Y
                        quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.1d0*DENSEKJ*Y
                        quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.1d0*DENSELI*Y
                        quick_qm_struct%o(JJJ,LLL) = quick_qm_struct%o(JJJ,LLL)-.1d0*DENSEKI*Y
                        quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.1d0*DENSELI*Y
                        quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-.1d0*DENSEKI*Y

                        !                      print*,III,JJJ,KKK,LLL,Y

                        !    ! do all the (ii|ii) integrals.
                        !        ! Set some variables to reduce access time for some of the more
                        !        ! used quantities. (AGAIN)
                        elseif(III.eq.JJJ.and.KKK.eq.LLL)then
                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEJJ=quick_qm_struct%dense(KKK,KKK)
                        DENSEII=quick_qm_struct%dense(III,III)

                        ! Find  all the (ii|jj) integrals.
                        quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+DENSEJJ*Y
                        quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+DENSEII*Y
                        quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.1d0*DENSEJI*Y

                        elseif(JJJ.eq.KKK.and.JJJ.eq.LLL)then
                        DENSEJI=quick_qm_struct%dense(JJJ,III)
                        DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)

                        ! Find  all the (ij|jj) integrals.
                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+.9d0*DENSEJJ*Y
                        quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)+1.8d0*DENSEJI*Y

                        !        ! Find  all the (ii|ij) integrals.
                        !
                        !        ! Find all the (ij|ij) integrals
                        !
                        ! Find all the (ij|ik) integrals where j>i,k>j
                        elseif(KKK.eq.LLL.and.III.lt.JJJ.and.JJJ.ne.KKK)then
                        DENSEKI=quick_qm_struct%dense(KKK,III)
                        DENSEKJ=quick_qm_struct%dense(KKK,JJJ)

                        DENSEKK=quick_qm_struct%dense(KKK,KKK)
                        DENSEJI=quick_qm_struct%dense(JJJ,III)

                        ! Find all the (ij|kk) integrals where j>i, k>j.
                        quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+DENSEKK*Y
                        quick_qm_struct%o(KKK,KKK) = quick_qm_struct%o(KKK,KKK)+2.d0*DENSEJI*Y

                        quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.1d0*DENSEKJ*Y
                        quick_qm_struct%o(KKK,JJJ) = quick_qm_struct%o(KKK,JJJ)-.1d0*DENSEKI*Y
                        quick_qm_struct%o(JJJ,KKK) = quick_qm_struct%o(JJJ,KKK)-.1d0*DENSEKI*Y

                        !            ! Find all the (ik|jj) integrals where j>i, k>j.
                        elseif(III.eq.JJJ.and.KKK.lt.LLL)then
                        DENSEII=quick_qm_struct%dense(III,III)
                        DENSEKJ=quick_qm_struct%dense(LLL,KKK)

                        DENSEJI=quick_qm_struct%dense(KKK,III)
                        DENSEKI=quick_qm_struct%dense(LLL,III)

                        ! Find all the (ii|jk) integrals where j>i, k>j.
                        quick_qm_struct%o(LLL,KKK) = quick_qm_struct%o(LLL,KKK)+DENSEII*Y
                        quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+2.d0*DENSEKJ*Y

                        quick_qm_struct%o(KKK,III) = quick_qm_struct%o(KKK,III)-.1d0*DENSEKI*Y
                        quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)-.1d0*DENSEJI*Y

                     endif

                  else
                     if(JJJ.LE.LLL)then

                        !                call hrrwhole(IJKLtype,III,JJJ,KKK,LLL,Y)
                        call hrrwhole

                        if(III.eq.JJJ.and.III.eq.KKK.and.III.eq.LLL)then
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! do all the (ii|ii) integrals.
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+0.9d0*DENSEII*Y

                           elseif(III.eq.JJJ.and.III.eq.KKK.and.III.lt.LLL)then
                           DENSEJI=quick_qm_struct%dense(LLL,III)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find  all the (ii|ij) integrals.
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+0.9d0*DENSEII*Y
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)+1.8d0*DENSEJI*Y

                           elseif(III.eq.KKK.and.JJJ.eq.LLL.and.III.lt.JJJ)then
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           DENSEJJ=quick_qm_struct%dense(JJJ,JJJ)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find all the (ij|ij) integrals
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+1.9d0*DENSEJI*Y
                           quick_qm_struct%o(JJJ,JJJ) = quick_qm_struct%o(JJJ,JJJ)-.1d0*DENSEII*Y
                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-.1d0*DENSEJJ*Y

                           elseif(III.eq.KKK.and.III.lt.JJJ.and.JJJ.lt.LLL)then
                           DENSEKI=quick_qm_struct%dense(LLL,III)
                           !                DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                           !                DENSEKK=quick_qm_struct%dense(LLL,LLL)
                           DENSEJI=quick_qm_struct%dense(JJJ,III)

                           DENSEKJ=quick_qm_struct%dense(LLL,JJJ)
                           !                DENSEKK=quick_qm_struct%dense(LLL,LLL)
                           DENSEII=quick_qm_struct%dense(III,III)

                           ! Find all the (ij|ik) integrals where j>i,k>j
                           quick_qm_struct%o(JJJ,III) = quick_qm_struct%o(JJJ,III)+1.9d0*DENSEKI*Y
                           quick_qm_struct%o(LLL,III) = quick_qm_struct%o(LLL,III)+1.9d0*DENSEJI*Y

                           quick_qm_struct%o(III,III) = quick_qm_struct%o(III,III)-0.2d0*DENSEKJ*Y
                           quick_qm_struct%o(LLL,JJJ) = quick_qm_struct%o(LLL,JJJ)-0.1d0*DENSEII*Y

                        endif

                     endif
                  endif
               enddo
            enddo
         enddo
      enddo
   endif

End subroutine

