#include "util.fh"

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
subroutine aoint(ierr)
   !------------------------------
   !  This subroutine is used to store 2e-integral into files
   !------------------------------
   use allmod
   use quick_cshell_eri_module, only: cshell
   Implicit none
   integer II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2,INTNUM, INTBEG, INTTOT, I, J
   double precision leastIntegralCutoff, t1, t2
   integer, intent(inout) :: ierr
   common /hrrstore/II,JJ,KK,LL,NBI1,NBI2,NBJ1,NBJ2,NBK1,NBK2,NBL1,NBL2

   call PrtAct(ioutfile,"Begin Calculation 2E TO DISK")



   write(ioutfile, '("  2-ELECTRON INTEGRAL")')
   write(ioutfile, '("-----------------------------")')

   call cpu_time(timer_begin%T2eAll)  ! Terminate the timer for 2e-integrals

   call obtain_leastIntCutoff(quick_method,ierr)

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
               call cshell
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
   use quick_cshell_eri_module, only: cshell
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
                     call cshell

            enddo
         enddo

      enddo
   enddo
   quick_method%nodirect = .true.
   100 continue

end subroutine addInt


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

