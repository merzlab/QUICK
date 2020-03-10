
!
!	denspt.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   written by Ed Brothers. January 17, 2002
!   3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

subroutine denspt(gridx, gridy, gridz, densitya, densityb, &
                  gax,   gay,   gaz, &
                  gbx,   gby,   gbz)
   use allmod
   implicit none
   ! Given a point in space, this function calculates the densities and
   ! gradient  at that point.  The gradients are stored in the common block
   ! three element arrays ga and gb for alpha and beta electron gradients. Thus
   ! the x component of the alpha density is stored in ga(1).
   
   
   ! INPUT PARAMETERS
   double precision :: gridx,gridy,gridz
   double precision :: densitya,densityb
   double precision :: gax,gay,gaz
   double precision :: gbx,gby,gbz

   ! INNER VARIBLES
   double precision :: dphidx,dphidy,dphidz
   double precision :: phi,phi2
   double precision :: densebij,denseij
   integer :: Ibas,Jbas

   densitya=0.d0

   gax=0.d0
   gay=0.d0
   gaz=0.d0

   do Ibas=1,nbasis
      DENSEIJ=quick_qm_struct%dense(Ibas,Ibas)
      if(DABS(quick_qm_struct%dense(Ibas,Ibas)) < quick_method%DMCutoff) then
         continue
      else

         DENSEBIJ=quick_qm_struct%dense(Ibas,Ibas)
         phi=phixiao(Ibas)
         dphidx=dphidxxiao(Ibas)
         dphidy=dphidyxiao(Ibas)
         dphidz=dphidzxiao(Ibas)


         if (DABS(dphidx+dphidy+dphidz+phi) < quick_method%DMCutoff ) then
            continue
         else

            densitya=densitya+DENSEIJ*phi*phi/2.0d0
            
              ! write(*,*) "a",densitya,DENSEIJ,phi,phi,quick_qm_struct%dense(Ibas,Ibas)
               
            gax=gax+DENSEIJ*phi*dphidx
            gay=gay+DENSEIJ*phi*dphidy
            gaz=gaz+DENSEIJ*phi*dphidz

            do Jbas=Ibas+1,nbasis
               DENSEIJ=quick_qm_struct%dense(Jbas,Ibas)
               phi2=phixiao(Jbas)
               
               densitya=densitya+DENSEIJ*phi*phi2
             !  write(*,*) densitya,DENSEIJ,phi,phi2
               gax=gax+DENSEIJ*(phi*dphidxxiao(Jbas)+phi2*dphidx)
               gay=gay+DENSEIJ*(phi*dphidyxiao(Jbas)+phi2*dphidy)
               gaz=gaz+DENSEIJ*(phi*dphidzxiao(Jbas)+phi2*dphidz)

            enddo
         endif
      endif
   enddo

   densityb=densitya
   gbx =gax
   gby =gay
   gbz =gaz
   
   !write(*,*) gax,gay,gaz,gbx,gby,gbz

end subroutine denspt

