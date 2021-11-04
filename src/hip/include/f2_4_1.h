__device__ __inline__   void h2_4_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
                                   QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                                   QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                                   QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                                   QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                                   QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                                   QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom)
{
    // call for L =            1  B =            0
    f_1_0_t f_1_0_0 ( VY( 0, 0, 0 ),  VY( 0, 0, 1 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_1 ( VY( 0, 0, 1 ),  VY( 0, 0, 2 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_0 ( f_1_0_0,  f_1_0_1, VY( 0, 0, 0 ), VY( 0, 0, 1 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_2 ( VY( 0, 0, 2 ),  VY( 0, 0, 3 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_1 ( f_1_0_1,  f_1_0_2, VY( 0, 0, 1 ), VY( 0, 0, 2 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_0 ( f_2_0_0,  f_2_0_1, f_1_0_0, f_1_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_3 ( VY( 0, 0, 3 ),  VY( 0, 0, 4 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_2 ( f_1_0_2,  f_1_0_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_1 ( f_2_0_1,  f_2_0_2, f_1_0_1, f_1_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_0 ( f_3_0_0,  f_3_0_1, f_2_0_0, f_2_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_4 ( VY( 0, 0, 4 ),  VY( 0, 0, 5 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_3 ( f_1_0_3,  f_1_0_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_2 ( f_2_0_2,  f_2_0_3, f_1_0_2, f_1_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_1 ( f_3_0_1,  f_3_0_2, f_2_0_1, f_2_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_0 ( f_4_0_0,  f_4_0_1,  f_3_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            4  J=           1
    LOCSTORE(store, 20,  1, STOREDIM, STOREDIM) = f_4_1_0.x_20_1 ;
    LOCSTORE(store, 20,  2, STOREDIM, STOREDIM) = f_4_1_0.x_20_2 ;
    LOCSTORE(store, 20,  3, STOREDIM, STOREDIM) = f_4_1_0.x_20_3 ;
    LOCSTORE(store, 21,  1, STOREDIM, STOREDIM) = f_4_1_0.x_21_1 ;
    LOCSTORE(store, 21,  2, STOREDIM, STOREDIM) = f_4_1_0.x_21_2 ;
    LOCSTORE(store, 21,  3, STOREDIM, STOREDIM) = f_4_1_0.x_21_3 ;
    LOCSTORE(store, 22,  1, STOREDIM, STOREDIM) = f_4_1_0.x_22_1 ;
    LOCSTORE(store, 22,  2, STOREDIM, STOREDIM) = f_4_1_0.x_22_2 ;
    LOCSTORE(store, 22,  3, STOREDIM, STOREDIM) = f_4_1_0.x_22_3 ;
    LOCSTORE(store, 23,  1, STOREDIM, STOREDIM) = f_4_1_0.x_23_1 ;
    LOCSTORE(store, 23,  2, STOREDIM, STOREDIM) = f_4_1_0.x_23_2 ;
    LOCSTORE(store, 23,  3, STOREDIM, STOREDIM) = f_4_1_0.x_23_3 ;
    LOCSTORE(store, 24,  1, STOREDIM, STOREDIM) = f_4_1_0.x_24_1 ;
    LOCSTORE(store, 24,  2, STOREDIM, STOREDIM) = f_4_1_0.x_24_2 ;
    LOCSTORE(store, 24,  3, STOREDIM, STOREDIM) = f_4_1_0.x_24_3 ;
    LOCSTORE(store, 25,  1, STOREDIM, STOREDIM) = f_4_1_0.x_25_1 ;
    LOCSTORE(store, 25,  2, STOREDIM, STOREDIM) = f_4_1_0.x_25_2 ;
    LOCSTORE(store, 25,  3, STOREDIM, STOREDIM) = f_4_1_0.x_25_3 ;
    LOCSTORE(store, 26,  1, STOREDIM, STOREDIM) = f_4_1_0.x_26_1 ;
    LOCSTORE(store, 26,  2, STOREDIM, STOREDIM) = f_4_1_0.x_26_2 ;
    LOCSTORE(store, 26,  3, STOREDIM, STOREDIM) = f_4_1_0.x_26_3 ;
    LOCSTORE(store, 27,  1, STOREDIM, STOREDIM) = f_4_1_0.x_27_1 ;
    LOCSTORE(store, 27,  2, STOREDIM, STOREDIM) = f_4_1_0.x_27_2 ;
    LOCSTORE(store, 27,  3, STOREDIM, STOREDIM) = f_4_1_0.x_27_3 ;
    LOCSTORE(store, 28,  1, STOREDIM, STOREDIM) = f_4_1_0.x_28_1 ;
    LOCSTORE(store, 28,  2, STOREDIM, STOREDIM) = f_4_1_0.x_28_2 ;
    LOCSTORE(store, 28,  3, STOREDIM, STOREDIM) = f_4_1_0.x_28_3 ;
    LOCSTORE(store, 29,  1, STOREDIM, STOREDIM) = f_4_1_0.x_29_1 ;
    LOCSTORE(store, 29,  2, STOREDIM, STOREDIM) = f_4_1_0.x_29_2 ;
    LOCSTORE(store, 29,  3, STOREDIM, STOREDIM) = f_4_1_0.x_29_3 ;
    LOCSTORE(store, 30,  1, STOREDIM, STOREDIM) = f_4_1_0.x_30_1 ;
    LOCSTORE(store, 30,  2, STOREDIM, STOREDIM) = f_4_1_0.x_30_2 ;
    LOCSTORE(store, 30,  3, STOREDIM, STOREDIM) = f_4_1_0.x_30_3 ;
    LOCSTORE(store, 31,  1, STOREDIM, STOREDIM) = f_4_1_0.x_31_1 ;
    LOCSTORE(store, 31,  2, STOREDIM, STOREDIM) = f_4_1_0.x_31_2 ;
    LOCSTORE(store, 31,  3, STOREDIM, STOREDIM) = f_4_1_0.x_31_3 ;
    LOCSTORE(store, 32,  1, STOREDIM, STOREDIM) = f_4_1_0.x_32_1 ;
    LOCSTORE(store, 32,  2, STOREDIM, STOREDIM) = f_4_1_0.x_32_2 ;
    LOCSTORE(store, 32,  3, STOREDIM, STOREDIM) = f_4_1_0.x_32_3 ;
    LOCSTORE(store, 33,  1, STOREDIM, STOREDIM) = f_4_1_0.x_33_1 ;
    LOCSTORE(store, 33,  2, STOREDIM, STOREDIM) = f_4_1_0.x_33_2 ;
    LOCSTORE(store, 33,  3, STOREDIM, STOREDIM) = f_4_1_0.x_33_3 ;
    LOCSTORE(store, 34,  1, STOREDIM, STOREDIM) = f_4_1_0.x_34_1 ;
    LOCSTORE(store, 34,  2, STOREDIM, STOREDIM) = f_4_1_0.x_34_2 ;
    LOCSTORE(store, 34,  3, STOREDIM, STOREDIM) = f_4_1_0.x_34_3 ;
}
