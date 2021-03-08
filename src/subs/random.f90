#include "util.fh"
!
!	random.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! RANDOM
!------------------------------------------------------------

SUBROUTINE RANDOM(IX,X)

  ! GENERATES INTEGER*4 AND doUBLE PRECISION RANdoM NUMBERS ON
  ! THE INTERVALS:

  ! 0 < IX < (2**31)-1
  ! AND      0.0 < X < 1.0

  ! ON THE FIRST CALL IX SHOULD SATISFY THE TOP INEQUALITY.

  ! NUMBERS ARE GENERATED USING THE RELATION,

  ! IX = IX*IC (MODULO (2**31)-1),  WHERE IC = 7**5


  double precision :: X


  ! INITIALIZE:     I15 = 2**15
  ! I16 = 2**16
  ! I31_1 = 2**31 - 1
  ! IC = 7**5

  DATA I15 /32768/
  DATA I16 /65536/
  DATA I31_1 /2147483647/
  DATA IC /16807/

  SAVE I15,I16,I31_1,IC

  IX16 = IX/I16
  I16RMD = IX - IX16*I16

  ! NOTE THAT IX = IX16*I16 + I16RMD    (I16RMD = 16-BIT REMAINDER)

  IX16IC = IX16*IC
  IXIC31 = IX16IC/I15

  ! NOTE:   IX*IC = (IX16*I16 + I16RMD)*IC
  ! = (IX16*IC*I16) + (I16RMD*IC)
  ! = (IX16IC*I16 ) + (I16RMD*IC)
  ! = (  TERM 1   ) + ( TERM 2  )
  ! AND,

  ! IX16IC = ((IX16IC/I15)*I15)  +  (IX16IC - (IX16IC/I15)*I15))
  ! = (  IXIC31  )*I15)  +  (IX16IC - (  IXIC31  )*I15 )
  ! (     15-BIT REMAINDER     )

  ! THEREFORE,  TERM 1 = ((IXIC31*I15) + (IX16IC - IXIC31*I15))*I16

  ! then,

  ! (   TERM A     )   (        TERM B          )   ( TERM C  )
  ! IX*IC = ((IXIC31*I16*I15) + (IX16IC - IXIC31*I15)*I16) + (I16RMD*IC)
  ! = (                  TERM 1                    ) + ( TERM 2  )


  ! NOTE THAT TERM B AND TERM C ARE BOTH LESS THAN 2**31 - 1.  ONLY
  ! TERM A HAS THE POSSIBILITY OF EXCEEDING 2**31 - 1.  BUT SINCE
  ! I16*I15 = 2**31, THE FACTOR IXIC31 INDICATES EXACTLY HOW MANY TIMES
  ! TERM A "WRAPS" AROUND THE INTERVAL (0,2**31 - 1).  THUS, FOR THE
  ! MODULO OPERATION, TERM A MAY BE REPLACED BY IXIC31.  THE SUM OF
  ! TERM A AND TERM B MIGHT EXCEED 2**31 - 1, BUT WE CAN SUBSTRACT
  ! 2**31 - 1 FROM ONE OF THEM TO PREVENT THIS FROM HAPPENING.

  IX = IXIC31 + ((IX16IC-IXIC31*I15)*I16 - I31_1) + I16RMD*IC

  ! ADD I31_1 BACK IN if THE SUBTRACTION MADE IX NEGATIVE.

  if(IX < 0) IX = IX + I31_1

  ! MAKE X RANdoM ON (0.0,1.0) BY MULTIPLYING IX BY 1.0/I31_1

  X = dble(IX)*4.6566128752458D-10
  RETURN
end SUBROUTINE RANdoM

