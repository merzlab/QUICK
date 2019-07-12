!
!	quick_constants_module.f90
!	new_quick
!
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!  Contants module. Store inportant constant. Most of them are attached with
!  parameter property

! 01/23/2019 
! Madu changed following physical constants to be consistent with psi4.
! PI, BOHR, BOLTZMANN, AVOGADRO, BOHRS_TO_A, AU_TO_EV 
module quick_constants_module
    implicit none
    ! general floating point constants
    
    double precision,parameter :: TEN_TO_MINUS1 = 1.0d-1
    double precision,parameter :: TEN_TO_MINUS2 = 1.0d-2
    double precision,parameter :: TEN_TO_MINUS3 = 1.0d-3
    double precision,parameter :: TEN_TO_MINUS4 = 1.0d-4
    double precision,parameter :: TEN_TO_MINUS5 = 1.0d-5
    double precision,parameter :: TEN_TO_MINUS6 = 1.0d-6
    double precision,parameter :: TEN_TO_MINUS7 = 1.0d-7
    double precision,parameter :: TEN_TO_MINUS8 = 1.0d-8
    double precision,parameter :: TEN_TO_MINUS9 = 1.0d-9
    double precision,parameter :: TEN_TO_MINUS10= 1.0d-10
    double precision,parameter :: TEN_TO_MINUS11= 1.0d-11
    double precision,parameter :: TEN_TO_MINUS12= 1.0d-12
    double precision,parameter :: TEN_TO_MINUS13= 1.0d-13
    double precision,parameter :: TEN_TO_MINUS14= 1.0d-14
    double precision,parameter :: TEN_TO_MINUS15= 1.0d-15
    double precision,parameter :: TEN_TO_MINUS16= 1.0d-16
    double precision,parameter :: TEN_TO_MINUS17= 1.0d-17
    
    double precision,parameter :: LEASTCUTOFF   = TEN_TO_MINUS9
    
    double precision,parameter :: ZERO      =0.0d0
    double precision,parameter :: ONE       =1.0d0
    double precision,parameter :: TWO       =2.0d0
    double precision,parameter :: THREE     =3.0d0
    double precision,parameter :: FOUR      =4.0d0
    double precision,parameter :: FIVE      =5.0d0
    double precision,parameter :: SIX       =6.0d0
    double precision,parameter :: SEVEN     =7.0d0
    double precision,parameter :: EIGHT     =8.0d0
    double precision,parameter :: NING      =9.0d0
    double precision,parameter :: TEN       =10.0d0
    double precision,parameter :: SIXTEEN   =16.0d0
    double precision,parameter :: THRITYTWO =32.0d0
    double precision,parameter :: SIXTYFOUR =64.0d0
    
    ! some mathmatical constants
    double precision, parameter :: PI = 3.14159265358979323846264338327950288d0
    double precision, parameter :: PITO3HALF = PI**1.5
    double precision, parameter :: PITO2 = PI*PI
    double precision, parameter :: X0 = 2.0d0*(PI)**(2.5d0)
!    double precision, parameter :: X00 = 1.0d0

    ! some physical constants
    double precision, parameter :: BOHR = 0.52917720859d0        ! bohr constant
    double precision, parameter :: HBAR = 627.509d0 * 0.0241888d-3 * 20.455d0
                                                                !Planck's constant in internal units
    double precision, parameter :: BOLTZMANN = 1.3806504d-23     !Boltzmann's constant in J/K
    double precision, parameter :: AVOGADRO = 6.02214179d+23     !Avogadro's number
    double precision, parameter :: CHARGE_ON_ELEC = 1.60217733d-19
                                                                !Charge on an electron in Coulombs
    
    ! some unit convertor
    double precision, parameter :: J_PER_CAL    = 4.184d0
    double precision, parameter :: JPKC         = J_PER_CAL * 1000.0d0  !kilocalories per joule
    
    double precision, parameter :: BOHRS_TO_A   = 0.52917720859D0         ! Bohrs to A, same with bohr constant
    double precision, parameter :: A_TO_BOHRS   = 1.0d0 / BOHRS_TO_A
    
    double precision, parameter :: AU_TO_EV     = 27.21138d0               !Conversion from AU to EV 
    double precision, parameter :: EV_TO_AU     = 1.0d0/AU_TO_EV
    double precision, parameter :: EV_TO_KCAL   = 23.061d0              ! conversion from EV to KCAL
    double precision, parameter :: KCAL_TO_EV   = 1.0d0/EV_TO_KCAL
    double precision, parameter :: AU_TO_KCAL   = AU_TO_EV*EV_TO_KCAL
    double precision, parameter :: KCAL_TO_AU   = 1.0d0/AU_TO_KCAL
    

    integer, parameter :: SYMBOL_MAX = 92       ! Max element supported
    
    character(len=2), dimension(0:92) :: SYMBOL = &
   & (/'XX','H ','HE','LI','BE','B ','C ','N ','O ','F ','NE', &
   & 'NA','MG','AL','SI','P ','S ','CL','AR','K ','CA', &
   & 'SC','TI','V ','CR','MN','FE','CO','NI','CU','ZN', &
   & 'GA','GE','AS','SE','BR','KR','RB','SR','Y ','ZR', &
   & 'NB','MO','TC','RU','RH','PD','AG','CD','IN','SN', &
   & 'SB','TE','I ','XE','CS','BA','LA','CE','PR','ND', &
   & 'PM','SM','EU','GD','TB','DY','HO','ER','TM','YB', &
   & 'LU','HF','TA','W ','RE','OS','IR','PT','AU','HG', &
   & 'TL','PB','BI','PO','AT','RN','FR','RA','AC','TH', &
   & 'PA','U '/)

    double precision, dimension(0:83) :: EMASS

    data EMASS &
    /0.d0, 1.0079d0, 4.0026d0, 6.941d0, 9.01218d0, &
    10.81d0, 12.011d0,14.0067d0, 15.99994d0, 18.99840d0, &
    20.179d0, 22.9898d0, 24.305d0, 26.98154d0, 28.0855d0, &
    30.97376d0, 32.06d0, 35.453d0, 39.948d0,39.0983d0, &
    40.08d0, 44.9559d0, 47.90d0, 50.9415d0, 51.996d0, &
    54.938d0, 55.847d0, 58.9332d0, 58.71d0, 63.546d0, &
    65.38d0, 69.737d0, 72.59d0, 74.9216d0, 78.96d0, 79.904d0, &
    83.80d0, 85.4678d0, 87.62d0, 88.9059d0, 91.22d0, 92.9064d0, &
    95.94d0, 98.9062d0, 101.07d0, 102.9055d0, 106.4d0, &
    107.868d0, 112.41d0, 114.82d0, 118.69d0, 121.75d0, 127.60d0, &
    126.9045d0, 131.30d0, 132.9054d0, 137.33d0, 15*0.000d0, &
    178.49d0, 180.9479d0, 183.850d0, 186.207d0, 190.20d0, &
    192.220d0, 195.090d0, 196.9665d0, 200.590d0, 204.370d0, &
    207.200d0, 208.9804d0/


    double precision, dimension(-2:30) :: FACT = &
    (/   0.d0,0.d0,1.d0,1.d0,  2.000000000000000D0, &
    6.000000000000000D0,24.00000000000000D0   , &
    120.0000000000000D0,720.0000000000000D0   , &
    5040.000000000000D0,40320.00000000000D0   , &
    362880.0000000000D0,3628800.000000000D0   , &
    39916800.00000000D0,479001600.0000000D0   , &
    6227020800.000000D0,87178291200.00000D0   , &
    1307674368000.000D0,20922789888000.00D0   , &
    355687428096000.0D0,6402373705728000.D0   , &
    1.2164510040883200D+17,2.4329020081766400D+18, &
    5.1090942171709440D+19,1.1240007277776077D+21, &
    2.5852016738884978D+22,6.2044840173323941D+23, &
    1.5511210043330986D+25,4.0329146112660565D+26, &
    1.0888869450418352D+28,3.0488834461171384D+29, &
    8.8417619937397008D+30,2.6525285981219103D+32/)

    double precision, dimension(0:83) :: RADII
    double precision, dimension(0:83) :: RADII2

    data RADII &
    /0.d0,1.d0,0.5882d0,3.0769d0,2.0513d0,1.5385d0, &
    1.2308d0,1.0256d0,0.8791d0,0.7692d0,0.6838d0, &
    4.0909d0,3.1579d0,2.5714d0,2.1687d0,1.8750d0, &
    1.6514d0,1.4754d0,1.3333d0,65*2.25d0/

    ! Xiao HE 02/11/2007    
    data RADII2 &
    /0.d0,1.30d0,0.0d0,1.95d0,2.20d0,1.45d0, &
    1.20d0,1.10d0,1.10d0,1.20d0,0.0d0, &
    2.30d0,2.20d0,2.10d0,1.30d0,1.30d0, &
    1.10d0,1.45d0,0.0d0,65*2.25d0/
    
    

end module quick_constants_module
