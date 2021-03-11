#include "util.fh"
!
!	prtcol.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------
! PriCol
!-----------------------------------------------------------
! 2004.12.22 print column(c1:c2) of mat(n,n) to file(io)
!-----------------------------------------------------------
subroutine PriCol(io,n,mat,c1,c2,fm) ! format: f(x.y) x>7 sugg 12.5,12.7,14.9
  implicit none
  integer i,j,jj,n,io,c1,c2,n5,nf,nc,x,y,k
  double precision mat(n,n)
  character fm*(*),ch,fm2*10
  character*40 fmt1,fmt2,fmt3,fmt4

  nc=c2-c1+1; n5=nc/5; nf=mod(nc,5)
  fm2=fm; ch=fm2(1:1); k=index(fm2,'.')
  read(fm2(2:k-1),*) x; read(fm2(k+1:10),*) y

  write(fmt1,101) ch,x,y; write(fmt2,102) nf,ch,x,y
101 format('(i7,5',a1,i2,'.',i2,')')
102 format('(i7,',i2,a1,i2,'.',i2,')')
  write(fmt3,103) x-7; write(fmt4,104) nf,x-7
103 format('(3x,5(',i2,'x,i7))')
104 format('(3x,',i2,'(',i2,'x,i7))')

  do jj=1,n5
     write(io,fmt3) (j,j=c1+(jj-1)*5,c1+jj*5-1)
     write(io,fmt1) (i,(mat(i,j),j=c1+(jj-1)*5,c1+jj*5-1),i=1,n)
     !         if (jj.ne.n5.or.nf.ne.0) write(io,*)
  enddo

  if (nf.ne.0) then
     write(io,fmt4)(j,j=c1+n5*5,c2)
     write(io,fmt2) (i,(mat(i,j),j=c1+n5*5,c2),i=1,n)
  endif
  call flush(io)

end subroutine PriCol
