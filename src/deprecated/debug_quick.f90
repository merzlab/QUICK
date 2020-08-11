!********************************************************
! debug.f90
!********************************************************
! This file contains debug subroutines
! --Yipu Miao 10/09/2010
!********************************************************
! Subroutine List:
! debug_SCF()                    : SCF debug
! debugElecdii()                : Elecdii debug
!

!*******************************************************
! debug_SCF()
!-------------------------------------------------------
! this subroutine is to output some infos in debug mode
!
subroutine debug_SCF()
   use allmod
   implicit double precision(a-h,o-z)

   !Densitry matrix
   write(ioutfile,'(//,"DENSITY MATRIX AT START OF", &
         &  " CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,dense,'f14.8')

   ! Operator matrix
   write(ioutfile,'(//,"OPERATOR MATRIX FOR CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,O,'f14.8')


   ! Molecular orbital energy
   write(ioutfile,'(//,"MOLECULAR ORBITAL ENERGIES", &
         &  " FOR CYCLE",I4)') jscf
   do I=1,nbasis
      write (ioutfile,'(I4,F18.10)') I,E(I)
   enddo

   ! C matrix
   write(ioutfile,'(//,"C MATRIX", &
         &  " FOR CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,CO,'f14.8')


   ! Densitry matrix at end of this cycle
   write(ioutfile,'(//,"DENSITY MATRIX AT end OF", &
         &  " CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,dense,'f14.8')

   total=0.d0
   do I=1,nbasis
      do J=1,nbasis
         do K=1,ncontract(I)
            do L=1,ncontract(J)
               total = total + DENSE(I,J)* &
                     dcoeff(K,I)*dcoeff(L,J) &
                     *overlap(aexp(L,J),aexp(K,I), &
                     itype(1,J),itype(2,J),itype(3,J), &
                     itype(1,I),itype(2,I),itype(3,I), &
                     xyz(1,ncenter(J)),xyz(2,ncenter(J)), &
                     xyz(3,ncenter(J)),xyz(1,ncenter(I)), &
                     xyz(2,ncenter(I)),xyz(3,ncenter(I)))
            enddo
         enddo
      enddo
   enddo

   write(ioutfile,'(//,"TOTAL NUMBER OF ELECTRONS=",F18.10)')total

end subroutine debug_SCF

!*******************************************************
! debugElecdii()
!-------------------------------------------------------
! this subroutine is to output some infos in debug mode
!
subroutine debugElecdii()
   use allmod
   implicit double precision(a-h,o-z)

   write(iOutFile,*) "O OPERATOR for cycle ",jscf
   call PriSym(iOutFile,nbasis,O,'f14.8')

   write(iOutFile,*) "DENSITY MATRIX for cycle ", jscf
   call PriSym(iOutFile,nbasis,dense,'f14.8')


end subroutine debugElecdii

!*******************************************************
! debugElecdii()
!-------------------------------------------------------
! this subroutine is to output normalization info for divcon
!
subroutine debugDivconNorm()
   use allmod

   implicit double precision(a-h,o-z)

   ! Xiaotemp inidicates normalization infos for system
   Xiaotemp=0.0d0
   do i=1,nbasis
      do j=1,nbasis
         Xiaotemp=Xiaotemp+DENSE(j,i)*Smatrix(j,i)
      enddo
   enddo

   ! Write Normalization for both full system and subsystem
   write(ioutfile,*)'ELECTION NORMALIZATION'
   write(ioutfile,*)'-------------------------------'
   write(ioutfile,*)'FULL=',Xiaotemp
   write(ioutfile,*)'SUBSYSTEM     NORMALIZATION'
   do itt=1,np
      Xiaotemp=0.0d0
      do i=1,nbasisdc(itt)
         do j=1,nbasisdc(itt)
            Xiaotemp=Xiaotemp+Pdcsub(itt,j,i)*smatrixdcsub(itt,j,i)
         enddo
      enddo
      write(ioutfile,*) itt,Xiaotemp
   enddo
   write(ioutfile,*)'-------------------------------'

end subroutine debugDivconNorm

subroutine debugBasis
   use allmod
   implicit double precision(a-h,o-z)
   do I=1,nbasis
      write(iOutFile,'(/"BASIS FUNCTON ",I4," ON ATOM ",I4)') &
            I,ncenter(I)
      write(iOutFile,'("THIS IS AN ",I1,I1,I1," FUNCTION")') &
            itype(1,I),itype(2,I),itype(3,I)
      write(iOutFile,'("THERE ARE ",I4," CONTRACTED GAUSSIANS")') &
            ncontract(I)
      do J=1,ncontract(I)
         write(iOutFile,'(F10.6,6x,F10.6)') aexp(J,I),dcoeff(J,I)
      enddo
   enddo
end subroutine debugBasis


subroutine debugInitialGuess
    use allmod
         write(iOutFile,*) "DENSITY MATRIX AFTER INITIAL GUESS"
         call PriSym(iOutFile,nbasis,dense,'f14.8')
end subroutine