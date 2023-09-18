module nr_gamma
use dlf_parameter_module
implicit none
integer, parameter :: sp=selected_real_kind(P=6)
integer, parameter :: dp=selected_real_kind(P=14)
INTEGER, PARAMETER :: NPAR_ARTH=16,NPAR2_ARTH=8
INTEGER, PARAMETER :: NPAR_CUMPROD=8

	INTERFACE assert_eq
		MODULE PROCEDURE assert_eq2,assert_eq3,assert_eq4,assert_eqn
	END INTERFACE
	INTERFACE arth
		MODULE PROCEDURE arth_r, arth_d, arth_i
	END INTERFACE
	INTERFACE gammln
		module procedure gammln_s
		module procedure gammln_v
	END INTERFACE
	INTERFACE gcf
		module procedure gcf_s
		module procedure gcf_v
	END INTERFACE
	INTERFACE gammp
		module procedure gammp_s
		module procedure gammp_v
	END INTERFACE
	INTERFACE gammq
		module procedure gammq_s
		module procedure gammq_v
	END INTERFACE
	INTERFACE gser
		module procedure gser_s
		module procedure gser_v
	END INTERFACE
	INTERFACE assert
		MODULE PROCEDURE assert1,assert2,assert3,assert4,assert_v
	END INTERFACE
	INTERFACE factrl
		MODULE PROCEDURE factrl_s
		MODULE PROCEDURE factrl_v
	END INTERFACE
contains
	SUBROUTINE nrerror(string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	write (*,*) 'nrerror: ',string
	STOP 'program terminated by nrerror'
	END SUBROUTINE nrerror
      
	SUBROUTINE assert1(n1,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	LOGICAL, INTENT(IN) :: n1
	if (.not. n1) then
		write (*,*) 'nrerror: an assertion failed with this tag:', &
			string
		STOP 'program terminated by assert1'
	end if
	END SUBROUTINE assert1
!BL
	SUBROUTINE assert2(n1,n2,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	LOGICAL, INTENT(IN) :: n1,n2
	if (.not. (n1 .and. n2)) then
		write (*,*) 'nrerror: an assertion failed with this tag:', &
			string
		STOP 'program terminated by assert2'
	end if
	END SUBROUTINE assert2
!BL
	SUBROUTINE assert3(n1,n2,n3,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	LOGICAL, INTENT(IN) :: n1,n2,n3
	if (.not. (n1 .and. n2 .and. n3)) then
		write (*,*) 'nrerror: an assertion failed with this tag:', &
			string
		STOP 'program terminated by assert3'
	end if
	END SUBROUTINE assert3
!BL
	SUBROUTINE assert4(n1,n2,n3,n4,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	LOGICAL, INTENT(IN) :: n1,n2,n3,n4
	if (.not. (n1 .and. n2 .and. n3 .and. n4)) then
		write (*,*) 'nrerror: an assertion failed with this tag:', &
			string
		STOP 'program terminated by assert4'
	end if
	END SUBROUTINE assert4
!BL
	SUBROUTINE assert_v(n,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	LOGICAL, DIMENSION(:), INTENT(IN) :: n
	if (.not. all(n)) then
		write (*,*) 'nrerror: an assertion failed with this tag:', &
			string
		STOP 'program terminated by assert_v'
	end if
	END SUBROUTINE assert_v
      
	FUNCTION assert_eq2(n1,n2,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	INTEGER, INTENT(IN) :: n1,n2
	INTEGER :: assert_eq2
	if (n1 == n2) then
		assert_eq2=n1
	else
		write (*,*) 'nrerror: an assert_eq failed with this tag:', &
			string
		STOP 'program terminated by assert_eq2'
	end if
	END FUNCTION assert_eq2
!BL
	FUNCTION assert_eq3(n1,n2,n3,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	INTEGER, INTENT(IN) :: n1,n2,n3
	INTEGER :: assert_eq3
	if (n1 == n2 .and. n2 == n3) then
		assert_eq3=n1
	else
		write (*,*) 'nrerror: an assert_eq failed with this tag:', &
			string
		STOP 'program terminated by assert_eq3'
	end if
	END FUNCTION assert_eq3
!BL
	FUNCTION assert_eq4(n1,n2,n3,n4,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	INTEGER, INTENT(IN) :: n1,n2,n3,n4
	INTEGER :: assert_eq4
	if (n1 == n2 .and. n2 == n3 .and. n3 == n4) then
		assert_eq4=n1
	else
		write (*,*) 'nrerror: an assert_eq failed with this tag:', &
			string
		STOP 'program terminated by assert_eq4'
	end if
	END FUNCTION assert_eq4
!BL
	FUNCTION assert_eqn(nn,string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	INTEGER, DIMENSION(:), INTENT(IN) :: nn
	INTEGER :: assert_eqn
	if (all(nn(2:) == nn(1))) then
		assert_eqn=nn(1)
	else
		write (*,*) 'nrerror: an assert_eq failed with this tag:', &
			string
		STOP 'program terminated by assert_eqn'
	end if
	END FUNCTION assert_eqn
      
	FUNCTION gammln_s(xx)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: xx
	REAL(SP) :: gammln_s
	REAL(DP) :: tmp,x
	REAL(DP) :: stp = 2.5066282746310005_dp
	REAL(DP), DIMENSION(6) :: coef = (/76.18009172947146_dp,&
		-86.50532032941677_dp,24.01409824083091_dp,&
		-1.231739572450155_dp,0.1208650973866179e-2_dp,&
		-0.5395239384953e-5_dp/)
	call assert(xx > 0.0, 'gammln_s arg')
	x=xx
	tmp=x+5.5_dp
	tmp=(x+0.5_dp)*log(tmp)-tmp
	gammln_s=tmp+log(stp*(1.000000000190015_dp+&
		sum(coef(:)/arth(x+1.0_dp,1.0_dp,size(coef))))/x)
	END FUNCTION gammln_s

	FUNCTION arth_r(first,increment,n)
	REAL(SP), INTENT(IN) :: first,increment
	integer, INTENT(IN) :: n
	REAL(SP), DIMENSION(n) :: arth_r
	integer :: k,k2
	REAL(SP) :: temp
	if (n > 0) arth_r(1)=first
	if (n <= NPAR_ARTH) then
		do k=2,n
			arth_r(k)=arth_r(k-1)+increment
		end do
	else
		do k=2,NPAR2_ARTH
			arth_r(k)=arth_r(k-1)+increment
		end do
		temp=increment*NPAR2_ARTH
		k=NPAR2_ARTH
		do
			if (k >= n) exit
			k2=k+k
			arth_r(k+1:min(k2,n))=temp+arth_r(1:min(k,n-k))
			temp=temp+temp
			k=k2
		end do
	end if
	END FUNCTION arth_r
!BL
	FUNCTION arth_d(first,increment,n)
	REAL(DP), INTENT(IN) :: first,increment
	integer, INTENT(IN) :: n
	REAL(DP), DIMENSION(n) :: arth_d
	integer :: k,k2
	REAL(DP) :: temp
	if (n > 0) arth_d(1)=first
	if (n <= NPAR_ARTH) then
		do k=2,n
			arth_d(k)=arth_d(k-1)+increment
		end do
	else
		do k=2,NPAR2_ARTH
			arth_d(k)=arth_d(k-1)+increment
		end do
		temp=increment*NPAR2_ARTH
		k=NPAR2_ARTH
		do
			if (k >= n) exit
			k2=k+k
			arth_d(k+1:min(k2,n))=temp+arth_d(1:min(k,n-k))
			temp=temp+temp
			k=k2
		end do
	end if
	END FUNCTION arth_d
!BL
	FUNCTION arth_i(first,increment,n)
	integer, INTENT(IN) :: first,increment,n
	integer, DIMENSION(n) :: arth_i
	integer :: k,k2,temp
	if (n > 0) arth_i(1)=first
	if (n <= NPAR_ARTH) then
		do k=2,n
			arth_i(k)=arth_i(k-1)+increment
		end do
	else
		do k=2,NPAR2_ARTH
			arth_i(k)=arth_i(k-1)+increment
		end do
		temp=increment*NPAR2_ARTH
		k=NPAR2_ARTH
		do
			if (k >= n) exit
			k2=k+k
			arth_i(k+1:min(k2,n))=temp+arth_i(1:min(k,n-k))
			temp=temp+temp
			k=k2
		end do
	end if
	END FUNCTION arth_i

	FUNCTION gammln_v(xx)
	IMPLICIT NONE
	integer :: i
	REAL(SP), DIMENSION(:), INTENT(IN) :: xx
	REAL(SP), DIMENSION(size(xx)) :: gammln_v
	REAL(DP), DIMENSION(size(xx)) :: ser,tmp,x,y
	REAL(DP) :: stp = 2.5066282746310005_dp
	REAL(DP), DIMENSION(6) :: coef = (/76.18009172947146_dp,&
		-86.50532032941677_dp,24.01409824083091_dp,&
		-1.231739572450155_dp,0.1208650973866179e-2_dp,&
		-0.5395239384953e-5_dp/)
	if (size(xx) == 0) RETURN
	call assert(all(xx > 0.0), 'gammln_v arg')
	x=xx
	tmp=x+5.5_dp
	tmp=(x+0.5_dp)*log(tmp)-tmp
	ser=1.000000000190015_dp
	y=x
	do i=1,size(coef)
		y=y+1.0_dp
		ser=ser+coef(i)/y
	end do
	gammln_v=tmp+log(stp*ser/x)
	END FUNCTION gammln_v
      
      
	FUNCTION gcf_s(a,x,gln)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: a,x
	REAL(SP), OPTIONAL, INTENT(OUT) :: gln
	REAL(SP) :: gcf_s
	integer, PARAMETER :: ITMAX=100
	REAL(SP), PARAMETER :: EPS=epsilon(x),FPMIN=tiny(x)/EPS
	integer :: i
	REAL(SP) :: an,b,c,d,del,h
	if (x == 0.0) then
		gcf_s=1.0
		RETURN
	end if
	b=x+1.0_sp-a
	c=1.0_sp/FPMIN
	d=1.0_sp/b
	h=d
	do i=1,ITMAX
		an=-i*(i-a)
		b=b+2.0_sp
		d=an*d+b
		if (abs(d) < FPMIN) d=FPMIN
		c=b+an/c
		if (abs(c) < FPMIN) c=FPMIN
		d=1.0_sp/d
		del=d*c
		h=h*del
		if (abs(del-1.0_sp) <= EPS) exit
	end do
	if (i > ITMAX) call nrerror('a too large, ITMAX too small in gcf_s')
	if (present(gln)) then
		gln=gammln(a)
		gcf_s=exp(-x+a*log(x)-gln)*h
	else
		gcf_s=exp(-x+a*log(x)-gammln(a))*h
	end if
	END FUNCTION gcf_s


	FUNCTION gcf_v(a,x,gln)
	IMPLICIT NONE
	REAL(SP), DIMENSION(:), INTENT(IN) :: a,x
	REAL(SP), DIMENSION(:), OPTIONAL, INTENT(OUT) :: gln
	REAL(SP), DIMENSION(size(a)) :: gcf_v
	integer, PARAMETER :: ITMAX=100
	REAL(SP), PARAMETER :: EPS=epsilon(x),FPMIN=tiny(x)/EPS
	integer :: i
	REAL(SP), DIMENSION(size(a)) :: an,b,c,d,del,h
	logical, DIMENSION(size(a)) :: converged,zero
	i=assert_eq(size(a),size(x),'gcf_v')
	zero=(x == 0.0)
	where (zero)
		gcf_v=1.0
	elsewhere
		b=x+1.0_sp-a
		c=1.0_sp/FPMIN
		d=1.0_sp/b
		h=d
	end where
	converged=zero
	do i=1,ITMAX
		where (.not. converged)
			an=-i*(i-a)
			b=b+2.0_sp
			d=an*d+b
			d=merge(FPMIN,d, abs(d)<FPMIN )
			c=b+an/c
			c=merge(FPMIN,c, abs(c)<FPMIN )
			d=1.0_sp/d
			del=d*c
			h=h*del
			converged = (abs(del-1.0_sp)<=EPS)
		end where
		if (all(converged)) exit
	end do
	if (i > ITMAX) call nrerror('a too large, ITMAX too small in gcf_v')
	if (present(gln)) then
		if (size(gln) < size(a)) call &
			nrerror('gser: Not enough space for gln')
		gln=gammln(a)
		where (.not. zero) gcf_v=exp(-x+a*log(x)-gln)*h
	else
		where (.not. zero) gcf_v=exp(-x+a*log(x)-gammln(a))*h
	end if
	END FUNCTION gcf_v
      
	FUNCTION gammq_s(a,x)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: a,x
	REAL(SP) :: gammq_s
	call assert( x >= 0.0,  a > 0.0, 'gammq_s args')
	if (x<a+1.0_sp) then
		gammq_s=1.0_sp-gser(a,x)
	else
		gammq_s=gcf(a,x)
	end if
	END FUNCTION gammq_s


	FUNCTION gammq_v(a,x)
	IMPLICIT NONE
	REAL(SP), DIMENSION(:), INTENT(IN) :: a,x
	REAL(SP), DIMENSION(size(a)) :: gammq_v
	logical, DIMENSION(size(x)) :: mask
	integer :: ndum
	ndum=assert_eq(size(a),size(x),'gammq_v')
	call assert( all(x >= 0.0),  all(a > 0.0), 'gammq_v args')
	mask = (x<a+1.0_sp)
	gammq_v=merge(1.0_sp-gser(a,merge(x,0.0_sp,mask)), &
		gcf(a,merge(x,0.0_sp,.not. mask)),mask)
	END FUNCTION gammq_v
      
	FUNCTION gammp_s(a,x)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: a,x
	REAL(SP) :: gammp_s
	call assert( x >= 0.0,  a > 0.0, 'gammp_s args')
	if (x<a+1.0_sp) then
		gammp_s=gser(a,x)
	else
		gammp_s=1.0_sp-gcf(a,x)
	end if
	END FUNCTION gammp_s


	FUNCTION gammp_v(a,x)
	IMPLICIT NONE
	REAL(SP), DIMENSION(:), INTENT(IN) :: a,x
	REAL(SP), DIMENSION(size(x)) :: gammp_v
	logical, DIMENSION(size(x)) :: mask
	integer :: ndum
	ndum=assert_eq(size(a),size(x),'gammp_v')
	call assert( all(x >= 0.0),  all(a > 0.0), 'gammp_v args')
	mask = (x<a+1.0_sp)
	gammp_v=merge(gser(a,merge(x,0.0_sp,mask)), &
		1.0_sp-gcf(a,merge(x,0.0_sp,.not. mask)),mask)
	END FUNCTION gammp_v
      
	FUNCTION gser_s(a,x,gln)
	IMPLICIT NONE
	REAL(SP), INTENT(IN) :: a,x
	REAL(SP), OPTIONAL, INTENT(OUT) :: gln
	REAL(SP) :: gser_s
	integer, PARAMETER :: ITMAX=100
	REAL(SP), PARAMETER :: EPS=epsilon(x)
	integer :: n
	REAL(SP) :: ap,del,summ
	if (x == 0.0) then
		gser_s=0.0
		RETURN
	end if
	ap=a
	summ=1.0_sp/a
	del=summ
	do n=1,ITMAX
		ap=ap+1.0_sp
		del=del*x/ap
		summ=summ+del
		if (abs(del) < abs(summ)*EPS) exit
	end do
	if (n > ITMAX) call nrerror('a too large, ITMAX too small in gser_s')
	if (present(gln)) then
		gln=gammln(a)
		gser_s=summ*exp(-x+a*log(x)-gln)
	else
		gser_s=summ*exp(-x+a*log(x)-gammln(a))
	end if
	END FUNCTION gser_s


	FUNCTION gser_v(a,x,gln)
	IMPLICIT NONE
	REAL(SP), DIMENSION(:), INTENT(IN) :: a,x
	REAL(SP), DIMENSION(:), OPTIONAL, INTENT(OUT) :: gln
	REAL(SP), DIMENSION(size(a)) :: gser_v
	integer, PARAMETER :: ITMAX=100
	REAL(SP), PARAMETER :: EPS=epsilon(x)
	integer :: n
	REAL(SP), DIMENSION(size(a)) :: ap,del,summ
	logical, DIMENSION(size(a)) :: converged,zero
	n=assert_eq(size(a),size(x),'gser_v')
	zero=(x == 0.0)
	where (zero) gser_v=0.0
	ap=a
	summ=1.0_sp/a
	del=summ
	converged=zero
	do n=1,ITMAX
		where (.not. converged)
			ap=ap+1.0_sp
			del=del*x/ap
			summ=summ+del
			converged = (abs(del) < abs(summ)*EPS)
		end where
		if (all(converged)) exit
	end do
	if (n > ITMAX) call nrerror('a too large, ITMAX too small in gser_v')
	if (present(gln)) then
		if (size(gln) < size(a)) call &
			nrerror('gser: Not enough space for gln')
		gln=gammln(a)
		where (.not. zero) gser_v=summ*exp(-x+a*log(x)-gln)
	else
		where (.not. zero) gser_v=summ*exp(-x+a*log(x)-gammln(a))
	end if
	END FUNCTION gser_v
      
      
	FUNCTION factrl_s(n)
	IMPLICIT NONE
	integer, INTENT(IN) :: n
	REAL(SP) :: factrl_s
	integer, SAVE :: ntop=0
	integer, PARAMETER :: NMAX=32
	REAL(SP), DIMENSION(NMAX), SAVE :: a
	call assert(n >= 0, 'factrl_s arg')
	if (n < ntop) then
		factrl_s=a(n+1)
	else if (n < NMAX) then
		ntop=NMAX
		a(1)=1.0
		a(2:NMAX)=cumprod(arth(1.0_sp,1.0_sp,NMAX-1))
		factrl_s=a(n+1)
	else
		factrl_s=exp(gammln(n+1.0_sp))
	end if
	END FUNCTION factrl_s

	FUNCTION factrl_v(n)
	IMPLICIT NONE
	integer, DIMENSION(:), INTENT(IN) :: n
	REAL(SP), DIMENSION(size(n)) :: factrl_v
	logical, DIMENSION(size(n)) :: mask
	integer, SAVE :: ntop=0
	integer, PARAMETER :: NMAX=32
	REAL(SP), DIMENSION(NMAX), SAVE :: a
	call assert(all(n >= 0), 'factrl_v arg')
	if (ntop == 0) then
		ntop=NMAX
		a(1)=1.0
		a(2:NMAX)=cumprod(arth(1.0_sp,1.0_sp,NMAX-1))
	end if
	mask = (n >= NMAX)
	factrl_v=unpack(exp(gammln(pack(n,mask)+1.0_sp)),mask,0.0_sp)
	where (.not. mask) factrl_v=a(n+1)
	END FUNCTION factrl_v

	RECURSIVE FUNCTION cumprod(arr,seed) RESULT(ans)
	REAL(SP), DIMENSION(:), INTENT(IN) :: arr
	REAL(SP), OPTIONAL, INTENT(IN) :: seed
	REAL(SP), DIMENSION(size(arr)) :: ans
	integer :: n,j
	REAL(SP) :: sd
	n=size(arr)
	if (n == 0) RETURN
	sd=1.0_sp
	if (present(seed)) sd=seed
	ans(1)=arr(1)*sd
	if (n < NPAR_CUMPROD) then
		do j=2,n
			ans(j)=ans(j-1)*arr(j)
		end do
	else
		ans(2:n:2)=cumprod(arr(2:n:2)*arr(1:n-1:2),sd)
		ans(3:n:2)=ans(2:n-1:2)*arr(3:n:2)
	end if
	END FUNCTION cumprod
end module nr_gamma