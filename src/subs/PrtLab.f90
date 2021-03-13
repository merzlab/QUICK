#include "util.fh"
!
!	PrtLab.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------
! PrtLab
!-----------------------------------------------------------
!Li Wei write Ktmp(kk) into line: eg. "1,3,5-8,2*0,9"
!-----------------------------------------------------------
subroutine PrtLab(line,kk,Ktmp)
  integer Ktmp(kk)
  character ch,ch2,line*(*),line1*100,line2*100
  parameter (ch=',',ch2='-')

  line=' '; ini=1; fc=1

  write(line1,*) Ktmp(1)
  call EffChar(line1,1,100,k1,k2)
  line(ini:ini+k2-k1)=line1(k1:k2)
  ini=ini+k2-k1+1
  nz=0
  if (Ktmp(1)==0.and.Ktmp(2)==0) then
     fc=0; nz=1; ini=1
  endif

  do 110 i=2,kk
     write(line1,*) Ktmp(i)
     call EffChar(line1,1,100,k1,k2)
     if (Ktmp(i)-Ktmp(i-1)==1) then
        if (i==kk.or.Ktmp(i+1)-Ktmp(i).ne.1) then
           line(ini:ini+k2-k1+1)=ch2//line1(k1:k2)
           ini=ini+k2-k1+2
        endif
     elseif (Ktmp(i)==0) then
        nz=nz+1
        if (i==kk) then
           if (nz==1) then
              line(ini:ini+1)=ch//'0'; ini=ini+2
           else
              write(line2,*) nz; call EffChar(line2,1,100,k3,k4)
              if (fc==1) then  ! 2005.01.09 add
                 line(ini:ini+k4-k3+3)=ch//line2(k3:k4)//'*0'
                 ini=ini+k4-k3+4; exit
              else
                 line(ini:ini+k4-k3+2)=line2(k3:k4)//'*0'
                 ini=ini+k4-k3+3; fc=1; exit
              endif
           endif
        else
           if (Ktmp(i+1).ne.0.and.nz==1) then
              line(ini:ini+1)=ch//'0'; ini=ini+2; nz=0
           elseif (Ktmp(i+1).ne.0.and.nz.ne.1) then
              write(line2,*) nz; call EffChar(line2,1,100,k3,k4)
              if (fc==1) then  ! 2005.01.09 add
                 line(ini:ini+k4-k3+3)=ch//line2(k3:k4)//'*0'
                 ini=ini+k4-k3+4; nz=0
              else
                 line(ini:ini+k4-k3+2)=line2(k3:k4)//'*0'
                 ini=ini+k4-k3+3; nz=0; fc=1
              endif
           endif
        endif
     else
        line(ini:ini+k2-k1+1)=ch//line1(k1:k2)
        ini=ini+k2-k1+2
     endif
110 enddo

end subroutine PrtLab
