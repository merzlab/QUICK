!
!	quick_mfcc_module.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

! MFCC Module
module quick_mfcc_module
    implicit none

!    integer, allocatable, dimension(:) :: mfccatom,mfcccharge
    integer :: mfccatom(50),mfcccharge(50),npmfcc,IMFCC,kxiaoconnect
    double precision :: mfcccord(3,100,50)
    integer ::Ftmp(300)
    character(len=100)::linetmp
    character(len=2) :: mfccatomxiao(100,50)
    integer :: mfccstart(50),mfccfinal(50),mfccbases(50),mfccbasef(50)
    integer :: matomstart(50),matomfinal(50),matombases(50),matombasef(50)
!    integer, dimension(:), allocatable :: matomstart,matomfinal,matombases &
!    ,matombasef

    integer :: mfccatomcap(50),mfccchargecap(50)
    double precision :: mfcccordcap(3,100,50)
    character(len=2) :: mfccatomxiaocap(100,50)
    integer :: mfccstartcap(50),mfccfinalcap(50),mfccbasescap(50),mfccbasefcap(50)
    integer :: matomstartcap(50),matomfinalcap(50),matombasescap(50) &
    ,matombasefcap(50)
!    integer, dimension(:), allocatable :: matomstartcap,matomfinalcap,matombasescap &
!    ,matombasefcap

    integer :: mfccatomcon(50),mfccchargecon(50)
    double precision :: mfcccordcon(3,100,50)
    character(len=2) :: mfccatomxiaocon(100,50)
    integer :: mfccstartcon(50),mfccfinalcon(50),mfccbasescon(50),mfccbasefcon(50)
    integer :: matomstartcon(50),matomfinalcon(50),matombasescon(50) &
    ,matombasefcon(50)
!    integer, dimension(:), allocatable :: matomstartcap,matomfinalcap,matombasescap &
!    ,matombasefcap

    integer :: mfccatomcon2(50),mfccchargecon2(50)
    double precision :: mfcccordcon2(3,100,50)
    character(len=2) :: mfccatomxiaocon2(100,50)
    integer :: mfccstartcon2(50),mfccfinalcon2(50),mfccbasescon2(50),mfccbasefcon2(50)
    integer :: matomstartcon2(50),matomfinalcon2(50),matombasescon2(50) &
    ,matombasefcon2(50)

    integer :: mfccatomconi(50),mfccchargeconi(50)
    double precision :: mfcccordconi(3,100,50)
    character(len=2) :: mfccatomxiaoconi(100,50)
    integer :: mfccstartconi(50),mfccfinalconi(50),mfccbasesconi(50),mfccbasefconi(50)
    integer :: matomstartconi(50),matomfinalconi(50),matombasesconi(50) &
    ,matombasefconi(50)

    integer :: mfccatomconj(50),mfccchargeconj(50)
    double precision :: mfcccordconj(3,100,50)
    character(len=2) :: mfccatomxiaoconj(100,50)
    integer :: mfccstartconj(50),mfccfinalconj(50),mfccbasesconj(50),mfccbasefconj(50)
    integer :: matomstartconj(50),matomfinalconj(50),matombasesconj(50) &
    ,matombasefconj(50)

    double precision, allocatable, dimension(:,:,:) :: mfccdens,mfccdenscap,mfccdenscon &
                                            ,mfccdenscon2,mfccdensconi,mfccdensconj

end module quick_mfcc_module
