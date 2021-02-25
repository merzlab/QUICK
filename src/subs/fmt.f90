!
!	fmt.f90
!	new_quick
!
!	Created by Yipu Miao on 4/20/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! Ed Brothers. October 29, 2001
! 3456789012345678901234567890123456789012345678901234567890123456789012<<STOP

Subroutine FmT(MaxM,X,vals)
  use quick_constants_module
  implicit double precision (a-h,o-z)
  double precision, dimension(0:20) :: vals

  PIE4 = PI/4.0d0
  XINV = 1.d0/X
  E = exp(-X)

  if (X > 5.d0) then
     if (X > 15.d0) then
        if (X > 33.d0) then
           WW1 = sqrt(PIE4*XINV)
        else
           WW1 = (( 1.9623264149430d-01*XINV-4.9695241464490d-01)*XINV - &
                6.0156581186481d-05)*E + sqrt(PIE4*XINV)
        endif
     else if (X > 10.d0) then
        WW1 = (((-1.8784686463512d-01*XINV+2.2991849164985d-01)*XINV &
             -4.9893752514047d-01)*XINV-2.1916512131607d-05)*E &
             + sqrt(PIE4*XINV)
     else
        WW1 = (((((( 4.6897511375022d-01*XINV-6.9955602298985d-01)*XINV + &
             5.3689283271887d-01)*XINV-3.2883030418398d-01)*XINV + &
             2.4645596956002d-01)*XINV-4.9984072848436d-01)*XINV - &
             3.1501078774085d-06)*E + sqrt(PIE4*XINV)
     endif
  else if (X > 1.d0)  then
     if (X > 3.d0)  then
        Y = X-4.d0
        F1 = ((((((((((-2.62453564772299d-11*Y+3.24031041623823d-10 )*Y- &
             3.614965656163d-09)*Y+3.760256799971d-08)*Y- &
             3.553558319675d-07)*Y+3.022556449731d-06)*Y- &
             2.290098979647d-05)*Y+1.526537461148d-04)*Y- &
             8.81947375894379d-04 )*Y+4.33207949514611d-03 )*Y- &
             1.75257821619926d-02 )*Y+5.28406320615584d-02
        WW1 = (X+X)*F1+E
     else
        Y = X-2.d0
        F1 = ((((((((((-1.61702782425558d-10*Y+1.96215250865776d-09 )*Y- &
             2.14234468198419d-08 )*Y+2.17216556336318d-07 )*Y- &
             1.98850171329371d-06 )*Y+1.62429321438911d-05 )*Y- &
             1.16740298039895d-04 )*Y+7.24888732052332d-04 )*Y- &
             3.79490003707156d-03 )*Y+1.61723488664661d-02 )*Y- &
             5.29428148329736d-02 )*Y+1.15702180856167d-01
        WW1 = (X+X)*F1+E
     endif
  else if (X > 1.0d-01 .or. (X>1.0d-04.and.maxm<4)) then
     F1 =((((((((-8.36313918003957d-08*X+1.21222603512827d-06 )*X- &
          1.15662609053481d-05 )*X+9.25197374512647d-05 )*X- &
          6.40994113129432d-04 )*X+3.78787044215009d-03 )*X- &
          1.85185172458485d-02 )*X+7.14285713298222d-02 )*X- &
          1.99999999997023d-01 )*X+3.33333333333318d-01

     WW1 = (X+X)*F1+E
  else
     WW1 = (1.d0-X)/dble(2*maxm+1)
  endif

  if (X > 1.0d-1 .or. (X>1.0d-04.and.maxm<4) ) then
     vals(0) = WW1
     do m=1,maxm
        vals(m) = (((2*m-1)*vals(m-1))- E)*0.5d0*XINV
     enddo
  else
     vals(maxm) = WW1
     twot = X*2.d0
     do m=maxm-1,0,-1
        vals(m) = (twot * vals(m+1) + E) / dble(m*2+1)
     enddo
  endif

end subroutine fmt
