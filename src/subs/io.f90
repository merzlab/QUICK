#include "util.fh"
!
!	io.f90
!	new_quick
!
!	Created by Yipu Miao on 7/12/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!    write and read key and value from unformmatted files


! write one int value to chk file
subroutine wchk_int(chk,key,nvalu,fail)
   implicit none
   integer chk,nvalu,i,j,k,l,fail
   character kline*40,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo
   100  rewind(chk)
   write(chk) '#'//kline(1:40)
   write(chk) 'I '
   write(chk) nvalu
   fail=1
   200  return

end


! read one int value from chk file
subroutine rchk_int(chk,key,nvalu,fail)
   implicit none
   integer chk,nvalu,i,j,k,l,num,fail
   character kline*40,ktype*2,line*41,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype(1:1).ne.'I') exit
      read(chk,end=100,err=100) nvalu
      fail=1
      exit
   120    continue
   enddo

   100  return

end


! write one real value to chk file
subroutine wchk_real(chk,key,rvalu,fail)
   implicit none
   integer chk,i,j,k,l,fail
   character kline*40,key*(*)
   real*4 rvalu

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo

   100  write(chk) '#'//kline(1:40)
   write(chk) 'r '
   write(chk) rvalu
   fail=1
   200  return

end


! read one real value from chk file
subroutine rchk_real(chk,key,rvalu,fail)
   implicit none
   integer chk,nvalu,i,j,k,l,num,fail
   character kline*40,ktype*2,line*41,key*(*)
   real*4 rvalu

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype(1:1).ne.'r') exit
      read(chk,end=100,err=100) rvalu
      fail=1
      exit
   120    continue
   enddo

   100  return

end


! write one double value to chk file
subroutine wchk_double(chk,key,dvalu,fail)
   implicit none
   integer chk,i,j,k,l,fail
   character kline*40,key*(*)
   real*8 dvalu

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo

   100  write(chk) '#'//kline(1:40)
   write(chk) 'R '
   write(chk) dvalu
   fail=1
   200  return

end


! read one double value from chk file
subroutine rchk_double(chk,key,dvalu,fail)
   implicit none
   integer chk,nvalu,i,j,k,l,num,fail
   character kline*40,ktype*2,line*41,key*(*)
   real*8 dvalu

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype(1:1).ne.'R') exit
      read(chk,end=100,err=100) dvalu
      fail=1
      exit
   120    continue
   enddo

   100  return

end


! write one int array to chk file
subroutine wchk_iarray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,fail
   integer dim(x,y,z)
   character kline*40,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo

   100  write(chk) '#'//kline(1:40)
   write(chk) 'II'
   write(chk) x*y*z
   write(chk) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
   fail=1
   200  return

end


! read one int array from chk file
subroutine rchk_iarray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,fail
   integer num
   integer dim(x,y,z),dim_t(2*x,2*y,2*z)
   character kline*40,ktype*2,line*41,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype.ne.'II') exit
      read(chk,end=100,err=100) num
      if (num.ne.x*y*z) exit
      read(chk,end=100,err=100) (((dim_t(i,j,k),i=1,2*x-1),j=1,2*y-1),k=1,2*z-1)
      !	 read(chk,end=100,err=100) (((dim(i,j,k),i=1,x),j=1,y-1),k=1,z)

      fail=1
      exit
   120    continue
   enddo

   do i=1,x
      do j=1,y
         do k=1,z
            dim(i,1,1)=dim_t(2*i-1,2*j-1,2*k-1)
         enddo
      enddo
   enddo

   100  return

end


! write one real array to chk file
subroutine wchk_rarray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,fail
   real*4 dim(x,y,z)
   character kline*40,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo

   100  write(chk) '#'//kline(1:40)
   write(chk) 'rr'
   write(chk) x*y*z
   write(chk) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
   fail=1
   200  return

end


! read one real array from chk file
subroutine rchk_rarray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,num,fail
   real*4 dim(x,y,z)
   character kline*40,ktype*2,line*41,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype.ne.'rr') exit
      read(chk,end=100,err=100) num
      if (num.ne.x*y*z) exit
      read(chk,end=100,err=100) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
      fail=1
      exit
   120    continue
   enddo

   100  return

end

! write one double array to chk file
subroutine wchk_darray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,fail
   real*8 dim(x,y,z)
   character kline*40,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   do
      read(chk,end=100,err=200)
   enddo

   100  rewind(chk)
   write(chk) '#'//kline(1:40)
   write(chk) 'RR'
   write(chk) x*y*z
   write(chk) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
   fail=1
   200  return

end


! read one double array from chk file
subroutine rchk_darray(chk,key,x,y,z,dim,fail)
   implicit none
   integer chk,x,y,z,i,j,k,l,num,fail
   real*8 dim(x,y,z)
   character kline*40,ktype*2,line*41,key*(*)

   l=len(key)
   if (l>=40) then
      kline=key(1:40)
   else
      kline(1:l)=key(1:l)
      do k=l+1,40
         kline(k:k)=' '
      enddo
   endif

   fail=0
   rewind(chk)
   do
      read(chk,end=100,err=120) line
      if (line(1:1).ne.'#') cycle
      if (index(line,kline)==0) cycle
      read(chk,end=100,err=100) ktype
      if (ktype.ne.'RR') exit
      read(chk,end=100,err=100) num
      if (num.ne.x*y*z) exit
      read(chk,end=100,err=100) (((dim(i,j,k),i=1,x),j=1,y),k=1,z)
      fail=1
      exit
   120  continue
   enddo

   100  return

end



