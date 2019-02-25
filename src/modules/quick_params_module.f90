!
!	quick_params_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!
! Parameter Module
!
module quick_params_module

    implicit none

    double precision :: EK1prm(0:2,0:2,0:2,0:83), &
    At1prm(0:2,0:2,0:2,0:83), &
    At2prm(0:2,0:2,0:2,0:83), &
    Bndprm(0:2,0:2,0:2,0:83)
    double precision :: param7 = 1.d0
    double precision :: param8 = 1.d0
    double precision :: param9 = 1.d0
    double precision :: param10 = 1.d0

! SEDFT parameters for H.

    DATA EK1prm(0,0,0,1) / 2.946272d0 /
    DATA At1prm(0,0,0,1) / 1.885943d0 /

! SEDFT parameters for C.

    DATA EK1prm(0,0,0,6) / 0.599653d0 /
    DATA EK1prm(1,0,0,6) / 1.258270d0 /
    DATA EK1prm(0,1,0,6) / 1.258270d0 /
    DATA EK1prm(0,0,1,6) / 1.258270d0 /

    DATA At1prm(0,0,0,6) / 0.781378d0 /
    DATA At1prm(1,0,0,6) / 1.123674d0 /
    DATA At1prm(0,1,0,6) / 1.123674d0 /
    DATA At1prm(0,0,1,6) / 1.123674d0 /

! SEDFT parameters for N.

    DATA EK1prm(0,0,0,7) / 1.123192d0 /
    DATA EK1prm(1,0,0,7) / 1.050190d0 /
    DATA EK1prm(0,1,0,7) / 1.050190d0 /
    DATA EK1prm(0,0,1,7) / 1.050190d0 /

    DATA At1prm(0,0,0,7) / 0.866199d0 /
    DATA At1prm(1,0,0,7) / 1.027186d0 /
    DATA At1prm(0,1,0,7) / 1.027186d0 /
    DATA At1prm(0,0,1,7) / 1.027186d0 /

! SEDFT parameters for O
! TEMPORARY (THESE REALLY SUCK doNKEYS.)

    DATA EK1prm(0,0,0,8) / 0.467239d0 /
    DATA EK1prm(1,0,0,8) / 1.055385d0 /
    DATA EK1prm(0,1,0,8) / 1.055385d0 /
    DATA EK1prm(0,0,1,8) / 1.055385d0 /

    DATA At1prm(0,0,0,8) / 0.723604d0 /
    DATA At1prm(1,0,0,8) / 1.062795d0 /
    DATA At1prm(0,1,0,8) / 1.062795d0 /
    DATA At1prm(0,0,1,8) / 1.062795d0 /

! SEDFT parameters for F.
! TEMPORARY

    DATA EK1prm(0,0,0,9) / 0.617189d0 /
    DATA EK1prm(1,0,0,9) / 1.175433d0 /
    DATA EK1prm(0,1,0,9) / 1.175433d0 /
    DATA EK1prm(0,0,1,9) / 1.175433d0 /

    DATA At1prm(0,0,0,9) / 0.711736d0 /
    DATA At1prm(1,0,0,9) / 1.075532d0 /
    DATA At1prm(0,1,0,9) / 1.075532d0 /
    DATA At1prm(0,0,1,9) / 1.075532d0 /

    Integer :: Sumindex(-2:7)
    DATA Sumindex /0,0,1,4,10,20,35,56,84,120 /

    Integer :: trans(0:7,0:7,0:7)

    DATA trans(0,0,0) / 1 /
    DATA trans(0,0,1) / 4 /
    DATA trans(0,0,2) / 10 /
    DATA trans(0,0,3) / 20 /
    DATA trans(0,0,4) / 35 /
    DATA trans(0,1,0) / 3 /
    DATA trans(0,1,1) / 6 /
    DATA trans(0,1,2) / 17 /
    DATA trans(0,1,3) / 32 /
    DATA trans(0,2,0) / 9 /
    DATA trans(0,2,1) / 16 /
    DATA trans(0,2,2) / 23 /
    DATA trans(0,3,0) / 19 /
    DATA trans(0,3,1) / 31 /
    DATA trans(0,4,0) / 34 /
    DATA trans(1,0,0) / 2 /
    DATA trans(1,0,1) / 7 /
    DATA trans(1,0,2) / 15 /
    DATA trans(1,0,3) / 28 /
    DATA trans(1,1,0) / 5 /
    DATA trans(1,1,1) / 11 /
    DATA trans(1,1,2) / 26 /
    DATA trans(1,2,0) / 13 /
    DATA trans(1,2,1) / 25 /
    DATA trans(1,3,0) / 30 /
    DATA trans(2,0,0) / 8 /
    DATA trans(2,0,1) / 14 /
    DATA trans(2,0,2) / 22 /
    DATA trans(2,1,0) / 12 /
    DATA trans(2,1,1) / 24 /
    DATA trans(2,2,0) / 21 /
    DATA trans(3,0,0) / 18 /
    DATA trans(3,0,1) / 27 /
    DATA trans(3,1,0) / 29 /
    DATA trans(4,0,0) / 33 /

    DATA trans(1,2,2) / 36 /
    DATA trans(2,1,2) / 37 /
    DATA trans(2,2,1) / 38 /
    DATA trans(3,1,1) / 39 /
    DATA trans(1,3,1) / 40 /
    DATA trans(1,1,3) / 41 /
    DATA trans(0,2,3) / 42 /
    DATA trans(0,3,2) / 43 /
    DATA trans(2,0,3) / 44 /
    DATA trans(3,0,2) / 45 /
    DATA trans(2,3,0) / 46 /
    DATA trans(3,2,0) / 47 /
    DATA trans(0,1,4) / 48 /
    DATA trans(0,4,1) / 49 /
    DATA trans(1,0,4) / 50 /
    DATA trans(4,0,1) / 51 /
    DATA trans(1,4,0) / 52 /
    DATA trans(4,1,0) / 53 /
    DATA trans(5,0,0) / 54 /
    DATA trans(0,5,0) / 55 /
    DATA trans(0,0,5) / 56 /

    DATA trans(4,1,1) / 57 /
    DATA trans(1,4,1) / 58 /
    DATA trans(1,1,4) / 59 /
    DATA trans(1,2,3) / 60 /
    DATA trans(1,3,2) / 61 /
    DATA trans(2,1,3) / 62 /
    DATA trans(3,1,2) / 63 /
    DATA trans(2,3,1) / 64 /
    DATA trans(3,2,1) / 65 /
    DATA trans(2,2,2) / 66 /
    DATA trans(0,1,5) / 67 /
    DATA trans(0,5,1) / 68 /
    DATA trans(1,0,5) / 69 /
    DATA trans(5,0,1) / 70 /
    DATA trans(1,5,0) / 71 /
    DATA trans(5,1,0) / 72 /
    DATA trans(0,2,4) / 73 /
    DATA trans(0,4,2) / 74 /
    DATA trans(2,0,4) / 75 /
    DATA trans(4,0,2) / 76 /
    DATA trans(2,4,0) / 77 /
    DATA trans(4,2,0) / 78 /
    DATA trans(0,3,3) / 79 /
    DATA trans(3,0,3) / 80 /
    DATA trans(3,3,0) / 81 /
    DATA trans(6,0,0) / 82 /
    DATA trans(0,6,0) / 83 /
    DATA trans(0,0,6) / 84 /

    DATA trans(5,1,1) / 85 /
    DATA trans(1,5,1) / 86 /
    DATA trans(1,1,5) / 87 /
    DATA trans(1,2,4) / 88 /
    DATA trans(1,4,2) / 89 /
    DATA trans(2,1,4) / 90 /
    DATA trans(4,1,2) / 91 /
    DATA trans(2,4,1) / 92 /
    DATA trans(4,2,1) / 93 /
    DATA trans(1,3,3) / 94 /
    DATA trans(3,1,3) / 95 /
    DATA trans(3,3,1) / 96 /
    DATA trans(3,2,2) / 97 /
    DATA trans(2,3,2) / 98 /
    DATA trans(2,2,3) / 99 /
    DATA trans(0,1,6) / 100 /
    DATA trans(0,6,1) / 101 /
    DATA trans(1,0,6) / 102 /
    DATA trans(6,0,1) / 103 /
    DATA trans(1,6,0) / 104 /
    DATA trans(6,1,0) / 105 /
    DATA trans(0,2,5) / 106 /
    DATA trans(0,5,2) / 107 /
    DATA trans(2,0,5) / 108 /
    DATA trans(5,0,2) / 109 /
    DATA trans(2,5,0) / 110 /
    DATA trans(5,2,0) / 111 /
    DATA trans(0,3,4) / 112 /
    DATA trans(0,4,3) / 113 /
    DATA trans(3,0,4) / 114 /
    DATA trans(4,0,3) / 115 /
    DATA trans(3,4,0) / 116 /
    DATA trans(4,3,0) / 117 /
    DATA trans(7,0,0) / 118 /
    DATA trans(0,7,0) / 119 /
    DATA trans(0,0,7) / 120 /

    integer :: Mcal(3,120)
    DATA Mcal(1:3,1) /0,0,0/
    DATA Mcal(1:3,2) /1,0,0/
    DATA Mcal(1:3,3) /0,1,0/
    DATA Mcal(1:3,4) /0,0,1/
    DATA Mcal(1:3,5) /1,1,0/
    DATA Mcal(1:3,6) /0,1,1/
    DATA Mcal(1:3,7) /1,0,1/
    DATA Mcal(1:3,8) /2,0,0/
    DATA Mcal(1:3,9) /0,2,0/
    DATA Mcal(1:3,10) /0,0,2/
    DATA Mcal(1:3,11) /1,1,1/
    DATA Mcal(1:3,12) /2,1,0/
    DATA Mcal(1:3,13) /1,2,0/
    DATA Mcal(1:3,14) /2,0,1/
    DATA Mcal(1:3,15) /1,0,2/
    DATA Mcal(1:3,16) /0,2,1/
    DATA Mcal(1:3,17) /0,1,2/
    DATA Mcal(1:3,18) /3,0,0/
    DATA Mcal(1:3,19) /0,3,0/
    DATA Mcal(1:3,20) /0,0,3/
    DATA Mcal(1:3,21) /2,2,0/
    DATA Mcal(1:3,22) /2,0,2/
    DATA Mcal(1:3,23) /0,2,2/
    DATA Mcal(1:3,24) /2,1,1/
    DATA Mcal(1:3,25) /1,2,1/
    DATA Mcal(1:3,26) /1,1,2/
    DATA Mcal(1:3,27) /3,0,1/
    DATA Mcal(1:3,28) /1,0,3/
    DATA Mcal(1:3,29) /3,1,0/
    DATA Mcal(1:3,30) /1,3,0/
    DATA Mcal(1:3,31) /0,3,1/
    DATA Mcal(1:3,32) /0,1,3/
    DATA Mcal(1:3,33) /4,0,0/
    DATA Mcal(1:3,34) /0,4,0/
    DATA Mcal(1:3,35) /0,0,4/

    DATA Mcal(1:3,36) /1,2,2/
    DATA Mcal(1:3,37) /2,1,2/
    DATA Mcal(1:3,38) /2,2,1/
    DATA Mcal(1:3,39) /3,1,1/
    DATA Mcal(1:3,40) /1,3,1/
    DATA Mcal(1:3,41) /1,1,3/
    DATA Mcal(1:3,42) /0,2,3/
    DATA Mcal(1:3,43) /0,3,2/
    DATA Mcal(1:3,44) /2,0,3/
    DATA Mcal(1:3,45) /3,0,2/
    DATA Mcal(1:3,46) /2,3,0/
    DATA Mcal(1:3,47) /3,2,0/
    DATA Mcal(1:3,48) /0,1,4/
    DATA Mcal(1:3,49) /0,4,1/
    DATA Mcal(1:3,50) /1,0,4/
    DATA Mcal(1:3,51) /4,0,1/
    DATA Mcal(1:3,52) /1,4,0/
    DATA Mcal(1:3,53) /4,1,0/
    DATA Mcal(1:3,54) /5,0,0/
    DATA Mcal(1:3,55) /0,5,0/
    DATA Mcal(1:3,56) /0,0,5/

    DATA Mcal(1:3,57) /4,1,1/
    DATA Mcal(1:3,58) /1,4,1/
    DATA Mcal(1:3,59) /1,1,4/
    DATA Mcal(1:3,60) /1,2,3/
    DATA Mcal(1:3,61) /1,3,2/
    DATA Mcal(1:3,62) /2,1,3/
    DATA Mcal(1:3,63) /3,1,2/
    DATA Mcal(1:3,64) /2,3,1/
    DATA Mcal(1:3,65) /3,2,1/
    DATA Mcal(1:3,66) /2,2,2/
    DATA Mcal(1:3,67) /0,1,5/
    DATA Mcal(1:3,68) /0,5,1/
    DATA Mcal(1:3,69) /1,0,5/
    DATA Mcal(1:3,70) /5,0,1/
    DATA Mcal(1:3,71) /1,5,0/
    DATA Mcal(1:3,72) /5,1,0/
    DATA Mcal(1:3,73) /0,2,4/
    DATA Mcal(1:3,74) /0,4,2/
    DATA Mcal(1:3,75) /2,0,4/
    DATA Mcal(1:3,76) /4,0,2/
    DATA Mcal(1:3,77) /2,4,0/
    DATA Mcal(1:3,78) /4,2,0/
    DATA Mcal(1:3,79) /0,3,3/
    DATA Mcal(1:3,80) /3,0,3/
    DATA Mcal(1:3,81) /3,3,0/
    DATA Mcal(1:3,82) /6,0,0/
    DATA Mcal(1:3,83) /0,6,0/
    DATA Mcal(1:3,84) /0,0,6/

    DATA Mcal(1:3,85) /5,1,1/
    DATA Mcal(1:3,86) /1,5,1/
    DATA Mcal(1:3,87) /1,1,5/
    DATA Mcal(1:3,88) /1,2,4/
    DATA Mcal(1:3,89) /1,4,2/
    DATA Mcal(1:3,90) /2,1,4/
    DATA Mcal(1:3,91) /4,1,2/
    DATA Mcal(1:3,92) /2,4,1/
    DATA Mcal(1:3,93) /4,2,1/
    DATA Mcal(1:3,94) /1,3,3/
    DATA Mcal(1:3,95) /3,1,3/
    DATA Mcal(1:3,96) /3,3,1/
    DATA Mcal(1:3,97) /3,2,2/
    DATA Mcal(1:3,98) /2,3,2/
    DATA Mcal(1:3,99) /2,2,3/
    DATA Mcal(1:3,100) /0,1,6/
    DATA Mcal(1:3,101) /0,6,1/
    DATA Mcal(1:3,102) /1,0,6/
    DATA Mcal(1:3,103) /6,0,1/
    DATA Mcal(1:3,104) /1,6,0/
    DATA Mcal(1:3,105) /6,1,0/
    DATA Mcal(1:3,106) /0,2,5/
    DATA Mcal(1:3,107) /0,5,2/
    DATA Mcal(1:3,108) /2,0,5/
    DATA Mcal(1:3,109) /5,0,2/
    DATA Mcal(1:3,110) /2,5,0/
    DATA Mcal(1:3,111) /5,2,0/
    DATA Mcal(1:3,112) /0,3,4/
    DATA Mcal(1:3,113) /0,4,3/
    DATA Mcal(1:3,114) /3,0,4/
    DATA Mcal(1:3,115) /4,0,3/
    DATA Mcal(1:3,116) /3,4,0/
    DATA Mcal(1:3,117) /4,3,0/
    DATA Mcal(1:3,118) /7,0,0/
    DATA Mcal(1:3,119) /0,7,0/
    DATA Mcal(1:3,120) /0,0,7/

end module quick_params_module