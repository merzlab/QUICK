#include "util.fh"
!
!	quick_debug.f90
!	new_quick
!
!	Created by Yipu Miao on 4/12/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   This file contains debug subroutines
!   Subroutine inventory:
!       debug_SCF                  : SCF debug
!       debugElecdii               : Elecdii debug
!       debugDivconNorm            : divcon Normalization debug
!       debugBasis                 : basis debug
!       debugFullX                 : tranformation matrix debug
!       debugInitialGuess          : initial guess debug

! this subroutine is to output some infos in debug mode in SCF
subroutine debug_SCF(jscf)
   use allmod
   use quick_overlap_module, only: overlap, gpt
   implicit none
   double precision total, g_table(200),Ax,Ay,Az,Bx,By,Bz, Px,Py,Pz,a,b
   integer i,j,k,l,ii,jj,kk,jscf,g_count,ibas,jbas

   !Densitry matrix
   write(ioutfile,'("DENSITY MATRIX AT START OF CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,quick_qm_struct%dense,'f14.8')

   ! Operator matrix
   write(ioutfile,'("OPERATOR MATRIX FOR CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,quick_qm_struct%o,'f14.8')


   ! Molecular orbital energy
   write(ioutfile,'("MOLECULAR ORBITAL ENERGIES FOR CYCLE",I4)') jscf
   do I=1,nbasis
      write (ioutfile,'(I4,F18.10)') I,quick_qm_struct%E(I)
   enddo

   ! C matrix
   write(ioutfile,'("C MATRIX FOR CYCLE",I4)') jscf
   call PriSym(iOutFile,nbasis,quick_qm_struct%co,'f14.8')


   ! Densitry matrix at end of this cycle
   !write(ioutfile,'("DENSITY MATRIX AT END OF CYCLE",I4)') jscf
   !call PriSym(iOutFile,nbasis,quick_qm_struct%dense,'f14.8')


   ! n=sigma(PS)
   total=0.d0
   do Ibas=1,nbasis
      ii = itype(1,I)
      jj = itype(2,I)
      kk = itype(3,I)

      Bx = xyz(1,quick_basis%ncenter(Ibas))
      By = xyz(2,quick_basis%ncenter(Ibas))
      Bz = xyz(3,quick_basis%ncenter(Ibas))

      do Jbas=1,nbasis

         i = itype(1,Jbas)
         j = itype(2,Jbas)
         k = itype(3,Jbas)
         g_count = i+ii+j+jj+k+kk

         Ax = xyz(1,quick_basis%ncenter(Jbas))
         Ay = xyz(2,quick_basis%ncenter(Jbas))
         Az = xyz(3,quick_basis%ncenter(Jbas))

         do K=1,ncontract(Ibas)
            b = aexp(K,Ibas)
            do L=1,ncontract(Jbas)
               a = aexp(L,Jbas)
               call gpt(a,b,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_count,g_table)

               total = total + quick_qm_struct%dense(I,J)* &
                     dcoeff(K,I)*dcoeff(L,J) &
                     *overlap(a,b,i,j,k,ii,jj,kk,Ax,Ay,Az,Bx,By,Bz,Px,Py,Pz,g_table)
            enddo
         enddo
      enddo
   enddo

   write(ioutfile,'("TOTAL NUMBER OF ELECTRONS=",F18.10)')total

end subroutine debug_SCF

! debugElecdii()
! this subroutine is to output some infos in debug mode
subroutine debugElecdii(jscf)
   use allmod
   implicit none
   integer jscf

   write(iOutFile,*) "O OPERATOR for cycle ",jscf
   call PriSym(iOutFile,nbasis,quick_qm_struct%o,'f14.8')

   write(iOutFile,*) "DENSITY MATRIX for cycle ", jscf
   call PriSym(iOutFile,nbasis,quick_qm_struct%dense,'f14.8')
end subroutine debugElecdii


! debugDivconNorm
! this subroutine is to output normalization info for divcon
subroutine debugDivconNorm()
   use allmod

   implicit none
   double precision tmp
   integer i,j,itt
   ! Xiaotemp inidicates normalization infos for system
   tmp=0.0d0
   do i=1,nbasis
      do j=1,nbasis
         tmp=tmp+quick_qm_struct%dense(j,i)*quick_qm_struct%s(j,i)
      enddo
   enddo

   ! Write Normalization for both full system and subsystem
   write(ioutfile,*)'ELECTION NORMALIZATION'
   write(ioutfile,*)'-------------------------------'
   write(ioutfile,*)'FULL=',tmp
   write(ioutfile,*)'SUBSYSTEM     NORMALIZATION'
   do itt=1,np
      tmp=0.0d0
      do i=1,nbasisdc(itt)
         do j=1,nbasisdc(itt)
            tmp=tmp+Pdcsub(itt,j,i)*smatrixdcsub(itt,j,i)
         enddo
      enddo
      write(ioutfile,*) itt,tmp
   enddo
   write(ioutfile,*)'-------------------------------'

end subroutine debugDivconNorm

! debugBasis
subroutine debugBasis
   use allmod
   implicit none
   integer i,j
   do I=1,nbasis
      write(iOutFile,'("FOR BASIS SET  ",I4)') i
      write(iOutFile,'("BASIS FUNCTON ",I4," ON ATOM ",I4)') I,quick_basis%ncenter(I)
      write(iOutFile,'("THIS IS AN ",I1,I1,I1," FUNCTION")') itype(1,I),itype(2,I),itype(3,I)
      write(iOutFile,'("THERE ARE ",I4," CONTRACTED GAUSSIANS")') ncontract(I)
      do J=1,ncontract(I)
         write(iOutFile,'(F10.6,6x,F10.6)') aexp(J,I),dcoeff(J,I)
      enddo
      write(iOutFile,'("CONS = ",F10.6)') quick_basis%cons(i)
      write(iOutFile,'("KLMN = ",I4, 2x, I4, 2x, I4)') quick_basis%KLMN(1,i),quick_basis%KLMN(2,i),quick_basis%KLMN(3,i)
   enddo
   
   do i = 1, natom
    write(iOutFile,'(" FOR ATOM ",I4)') i
    write(iOutFile,'(" FIRST BASIS ",I4, " LAST BASIS",I4)') quick_basis%first_basis_function(i), &
                                                             quick_basis%last_basis_function(i)
                                                             
    write(iOutFile,'(" FIRST SHELL BASIS ",I4, " LAST SHELL BASIS",I4)')&
                                                            quick_basis%first_shell_basis_function(i), &
                                                            quick_basis%last_shell_basis_function(i)
    write(iOutFile,*)                                                         
   enddo
   
   do i = 1, nshell
    write(iOutFile,'(" FOR SHELL ",I4)') i
    write(iOutFile,'(" KSTART =  ",I4)') quick_basis%kstart(i)
    write(iOutFile,'(" KATOM =  ",I4)') quick_basis%katom(i)
    write(iOutFile,'(" KTYPE =  ",I4)') quick_basis%ktype(i)
    write(iOutFile,'(" KPRIM =  ",I4)') quick_basis%kprim(i)
    
    write(iOutFile,'(" QNUMBER =  ",I4)') quick_basis%QNUMBER(i)
    write(iOutFile,'(" QSTART =  ",I4)') quick_basis%QSTART(i)
    write(iOutFile,'(" QFINAL =  ",I4)') quick_basis%QFINAL(i)
    write(iOutFile,'(" KSUMTYPE =  ",I4)') quick_basis%KSUMTYPE(i)
    
    
    write(iOutFile,'(" Qsbasis =  ",I4,2x,I4,2x,I4,2x,I4)') &
                                          quick_basis%Qsbasis(i,0), quick_basis%Qsbasis(i,1), &
                                          quick_basis%Qsbasis(i,2), quick_basis%Qsbasis(i,3)  
    write(iOutFile,'(" Qfbasis =  ",I4,2x,I4,2x,I4,2x,I4)') &
                                          quick_basis%Qfbasis(i,0), quick_basis%Qfbasis(i,1), &
                                          quick_basis%Qfbasis(i,2), quick_basis%Qfbasis(i,3)
    write(iOutFile,*)    
   enddo
   
   
end subroutine debugBasis


! debugFullX
subroutine debugFullX
    use allmod
    write(ioutfile,'("THE OVERLAP MATRIX")')
    call PriSym(iOutFile,nbasis,quick_qm_struct%s,'F18.10')
    call flush(iOutFile)
    
    write(ioutfile,'("THE X MATRIX")')
    call PriSym(iOutFile,nbasis,quick_qm_struct%x,'F18.10')
    
end subroutine debugFullX

! debugInitialGuess
subroutine debugInitialGuess
    use allmod
         write(iOutFile,*) "DENSITY MATRIX AFTER INITIAL GUESS"
         call PriSym(iOutFile,nbasis,quick_qm_struct%dense,'f14.8')
end subroutine
