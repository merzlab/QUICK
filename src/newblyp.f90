subroutine newblyp
   use allmod
   use quick_gaussian_class_module
   implicit double precision(a-h,o-z)

   double precision oneElecO(nbasis,nbasis)
   logical :: deltaO

      do Iatm=1,natom
         if(quick_method%ISG.eq.1)then
            Iradtemp=50
         else
            if(quick_molspec%iattype(iatm).le.10)then
               Iradtemp=23
            else
               Iradtemp=26
            endif
         endif

         do Irad=1,Iradtemp
            if(quick_method%ISG.eq.1)then
               call gridformnew(iatm,RGRID(Irad),iiangt)
               rad = radii(quick_molspec%iattype(iatm))
            else
               call gridformSG0(iatm,Iradtemp+1-Irad,iiangt,RGRID,RWT)
               rad = radii2(quick_molspec%iattype(iatm))
            endif
!                     write(*,*)
!                     "CPU-Pretest:",iatm,natom,irad,Iradtemp,iiangt

            rad3 = rad*rad*rad
            do Iang=1,iiangt
               gridx=xyz(1,Iatm)+rad*RGRID(Irad)*XANG(Iang)
               gridy=xyz(2,Iatm)+rad*RGRID(Irad)*YANG(Iang)
               gridz=xyz(3,Iatm)+rad*RGRID(Irad)*ZANG(Iang)

               ! Next, calculate the weight of the grid point in the SSW
               ! scheme.
               ! if
               ! the grid point has a zero weight, we can skip it.

               weight=SSW(gridx,gridy,gridz,Iatm) &
                     *WTANG(Iang)*RWT(Irad)*rad3

                     !write(*,*)"MPI-Pretest:",iatm,irad,iang,iiangt

               if (weight < quick_method%DMCutoff ) then
                  continue
               else

                  do Ibas=1,nbasis

                     !write(*,*) "Madu: Pteval test", gridx,gridy,gridz              

                     call pteval(gridx,gridy,gridz,phi,dphidx,dphidy, &
                           dphidz,Ibas)
                     phixiao(Ibas)=phi
                     dphidxxiao(Ibas)=dphidx
                     dphidyxiao(Ibas)=dphidy
                     dphidzxiao(Ibas)=dphidz

!                     write(*,*) "Madu: Pteval test",phi, dphidx,
!                     dphidy, dphidz
                  enddo


                  ! Next, evaluate the densities at the grid point and
                  ! the
                  ! gradient
                  ! at that grid point.

                  call denspt(gridx,gridy,gridz,density,densityb,gax,gay,gaz, &
                        gbx,gby,gbz)
!                     write(*,*) "CPU-Pre
!                     test:",iatm,irad,ibas,density,quick_method%DMCutoff

                  if (density < quick_method%DMCutoff ) then
                     continue
                  else
!                     write(*,*) "Pre
!                     becke:",density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex
                     call becke_E(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex)
                     !write(*,*) "CPU-Post
                     !becke:",density,densityb,gax,gay,gaz,gbx,gby,gbz,Ex
                     call lyp_e(density,densityb,gax,gay,gaz,gbx,gby,gbz,Ec)
                     !write(*,*) "CPU-Post becke:",iatm,
                     !irad,ibas,density,densityb,gax,gay,&
                     !gaz,gbx,gby,gbz,Ex,Ec

                     Eelxc = Eelxc + (param7*Ex+param8*Ec) &
                           *weight

   !                  write(*,*) "Eelxc:",Eelxc,param7,Ex, param8,weight
                       

                     quick_qm_struct%aelec = weight*density+quick_qm_struct%aelec
                     quick_qm_struct%belec = weight*densityb+quick_qm_struct%belec

                     ! This allows the calculation of the derivative of
                     ! the
                     ! functional
                     ! with regard to the density (dfdr), with regard to
                     ! the
                     ! alpha-alpha
                     ! density invariant (df/dgaa), and the alpha-beta
                     ! density
                     ! invariant.

                       !write(*,*) "quicktest before becke:",&
                       !density,gax,gay,gaz,gbx,gby,gbz,dfdr,dfdgaa,dfdgab

                     call becke(density,gax,gay,gaz,gbx,gby,gbz, &
                           dfdr,dfdgaa,dfdgab)

                      ! write(*,*) "quicktest after becke:",&
                      ! density,gax,gay,gaz,gbx,gby,gbz,dfdr,dfdgaa,dfdgab

                     call lyp(density,densityb,gax,gay,gaz,gbx,gby,gbz,&
                           dfdr2,dfdgaa2,dfdgab2)

                       !write(*,*) "quicktest after lyp:",&
                       !density,gax,gay,gaz,gbx,gby,gbz,dfdr2,dfdgaa2,dfdgab2


                     dfdr = dfdr+dfdr2
                     dfdgaa = dfdgaa + dfdgaa2
                     dfdgab = dfdgab + dfdgab2

!                       write(*,*) "quicktest after lyp:",&
!                       dfdr,dfdgaa,dfdgab

                     ! Calculate the first term in the dot product shown
                     ! above,i.e.:
                     ! (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT
                     ! Grad(Phimu Phinu))

                     xdot=2.d0*dfdgaa*gax+dfdgab*gbx
                     ydot=2.d0*dfdgaa*gay+dfdgab*gby
                     zdot=2.d0*dfdgaa*gaz+dfdgab*gbz

                     ! Now loop over basis functions and compute the
                     ! addition to
                     ! the matrix
                     ! element.

                     do Ibas=1,nbasis
                        phi=phixiao(Ibas)
                        dphidx=dphidxxiao(Ibas)
                        dphidy=dphidyxiao(Ibas)
                        dphidz=dphidzxiao(Ibas)
                        quicktest = DABS(dphidx+dphidy+dphidz+ &
                              phi)


                        if (quicktest < quick_method%DMCutoff ) then
                           continue
                        else
                           do Jbas=Ibas,nbasis
                              phi2=phixiao(Jbas)
                              dphi2dx=dphidxxiao(Jbas)
                              dphi2dy=dphidyxiao(Jbas)
                              dphi2dz=dphidzxiao(Jbas)
                              temp = phi*phi2
                              tempgx = phi*dphi2dx + phi2*dphidx
                              tempgy = phi*dphi2dy + phi2*dphidy
                              tempgz = phi*dphi2dz + phi2*dphidz

                              quick_qm_struct%o(Jbas,Ibas)=quick_qm_struct%o(Jbas,Ibas)+(temp*dfdr+&
                                    xdot*tempgx+ydot*tempgy+zdot*tempgz)*weight
                          enddo
                        endif
                     enddo
                  endif
               endif
            enddo
         enddo
      enddo
   quick_qm_struct%Eel=quick_qm_struct%Eel+Eelxc
end subroutine newblyp
