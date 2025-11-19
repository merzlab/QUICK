!
!	quick_constants_module.f90
!	new_quick
!
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

#include "util.fh"

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
    double precision, parameter :: BOHRS_TO_A_AMBER = 0.529177249d0
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

    integer, dimension(1:92) :: SPINMULT = &
     (/2,                                                 1, &
       2, 1,                               2, 3, 4, 3, 2, 1, &
       2, 1,                               2, 3, 4, 3, 2, 1, &
       2, 1, 2, 3, 4, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, &
       2, 1, 2, 3, 6, 7, 6, 5, 4, 1, 2, 1, 2, 3, 4, 3, 2, 1, &
       2, 1, &
        2, 1, 4, 5, 6, 7, 8, 9, 6, 5, 4, 3, 2, 1, &
             2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, &
       2, 1, 2, 3, 4, 5/)

    double precision, dimension(0:92) :: EMASS

    data EMASS &
 /0.0d0,1.007825d0,  4.002603d0,  7.016005d0,  9.012182d0, 11.009305d0, &
       12.000000d0, 14.003074d0, 15.994915d0, 18.998403d0, 19.992440d0, &
       22.989769d0, 23.985042d0, 26.981539d0, 27.976926d0, 30.973762d0, &
       31.972071d0, 34.968853d0, 39.962383d0, 38.963707d0, 39.962591d0, &
       44.955912d0, 47.947946d0, 50.943960d0, 51.940508d0, 54.938045d0, &
       55.934937d0, 58.933195d0, 57.935343d0, 62.929598d0, 63.929142d0, &
       68.925574d0, 73.921178d0, 74.921597d0, 79.916521d0, 78.918337d0, &
       83.911507d0, 84.911790d0, 87.905612d0, 88.905848d0, 89.904704d0, &
       92.906378d0, 97.905408d0, 97.907216d0,101.904349d0,102.905504d0, &
      105.903486d0,106.905097d0,113.903359d0,114.903878d0,119.902195d0, &
      120.903816d0,129.906224d0,126.904473d0,131.904154d0,132.905452d0, &
      137.905247d0,138.906353d0,139.905439d0,140.907653d0,141.907723d0, &
      144.912749d0,151.919732d0,152.921230d0,157.924104d0,158.925347d0, &
      163.929175d0,164.930322d0,165.930293d0,168.934213d0,173.938862d0, &
      174.940772d0,179.946550d0,180.947996d0,183.950931d0,186.955753d0, &
      191.961481d0,192.962926d0,194.964791d0,196.966569d0,201.970643d0, &
      204.974428d0,207.976652d0,208.980399d0,208.982430d0,208.987148d0, &
      222.017578d0,223.019736d0,226.025410d0,227.027752d0,232.038055d0, &
      231.035884d0,238.050788d0/

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

    ! Tripathy V 11/14/2025
    ! Radius of maximum density of outermost shells of
    ! neutral atoms as given by Slater.
    !
    ! J.C. Slater, Phys. Rev. 36 (1930) 57
    !
    ! Theses RADII (in bohr) are used in SG1 grid.
    !
    ! R = (n*)^2/(Z-s)
    !
    ! where n* is the effective principal quantum number
    ! of the outermost shell (not equal n for n > 3) and
    ! s is the screening constant (value depends on the
    ! type of shell and number of electrons).
    !
    data RADII &
    /0.d0,1.d0,0.5882d0,3.0769d0,2.0513d0,1.5385d0, &
    1.2308d0,1.0256d0,0.8791d0,0.7692d0,0.6838d0, &
    4.0909d0,3.1579d0,2.5714d0,2.1687d0,1.8750d0, &
    1.6514d0,1.4754d0,1.3333d0,6.2227d0,4.8035d0, &
    4.5633d0,4.3460d0,4.1485d0,3.9681d0,3.8028d0, &
    3.6507d0,3.5103d0,3.3802d0,3.2595d0,3.1471d0, &
    2.7380d0,2.4230d0,2.1730d0,1.9698d0,1.8013d0, &
    1.6594d0,7.2727d0,5.6140d0,5.3333d0,5.0794d0, &
    4.8485d0,4.6377d0,4.4444d0,4.2667d0,4.1026d0, &
    3.9506d0,3.8095d0,3.6782d0,3.2000d0,2.8319d0, &
    2.5397d0,2.3022d0,2.1053d0,1.9394d0,8.0182d0, &
    6.1895d0,6.1895d0,6.1895d0,6.1895d0,6.1895d0, &
    6.1895d0,6.1895d0,6.1895d0,6.1895d0,6.1895d0, &
    6.1895d0,6.1895d0,6.1895d0,6.1895d0,6.1895d0, &
    5.8800d0,5.6000d0,5.3455d0,5.1130d0,4.9000d0, &
    4.7040d0,4.5231d0,4.3556d0,4.2000d0,4.0552d0, &
    3.5280d0,3.1221d0,2.8000d0/

    ! Xiao HE 02/11/2007
    ! RADII2 is used in SG0 Grid. This is old and
    ! not recommended for production calculation. 
    data RADII2 &
    /0.d0,1.30d0,0.0d0,1.95d0,2.20d0,1.45d0, &
    1.20d0,1.10d0,1.10d0,1.20d0,0.0d0, &
    2.30d0,2.20d0,2.10d0,1.30d0,1.30d0, &
    1.10d0,1.45d0,0.0d0,65*2.25d0/
    
    

end module quick_constants_module
