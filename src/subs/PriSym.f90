!
!	PriSym.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!   This subroutine is from Wei Li, Nanjing University

!-----------------------------------------------------------      
! PriSym
!-----------------------------------------------------------
! print symmetric mat(n,n) to file(io)
!-----------------------------------------------------------
subroutine PriSym(io,n,mat,fm) ! format: f(x.y) x>7 sugg 12.5,12.7,14.9
  implicit none
  integer j,jj,n,io,n5,nf,x,y,ini,ifi,k
  double precision mat(n,n)
  character fm*(*),ch,fm2*10
  character*40 fmt1,fmt2,fmt3,fmt4

  n5=n/5
  nf=mod(n,5)
  fm2=fm
  ch=fm2(1:1)
  k=index(fm2,'.')
  read(fm2(2:k-1),*) x
  read(fm2(k+1:10),*) y

  write(fmt1,101) ch,x,y
  write(fmt2,102) nf,ch,x,y
101 format('(i7,5',a1,i2,'.',i2,')')
102 format('(i7,',i2,a1,i2,'.',i2,')')
  write(fmt3,103) x-7
  write(fmt4,104) nf,x-7
103 format('(3x,5(',i2,'x,i7))')
104 format('(3x,',i2,'(',i2,'x,i7))')

  do jj=1,n5
     ini=1+(jj-1)*5
     write(io,fmt3) (j,j=ini,jj*5)
     do k=1+(jj-1)*5,n
        ifi=min(jj*5,k)
        write(io,fmt1) k,(mat(k,j),j=ini,ifi)
     enddo
     !         if (jj.ne.n5.or.nf.ne.0) write(io,*)
  enddo

  if (nf.ne.0) then
     ini=n-nf+1
     write(io,fmt4)(j,j=ini,n)
     do k=ini,n
        write(io,fmt2) k,(mat(k,j),j=ini,k)
     enddo
  endif
  call flush(io)

end subroutine PriSym
