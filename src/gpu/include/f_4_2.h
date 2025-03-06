__device__ __inline__   void h_4_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_5 ( VY( 0, 0, 5 ),  VY( 0, 0, 6 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_4 ( f_1_0_4,  f_1_0_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_3 ( f_2_0_3,  f_2_0_4, f_1_0_3, f_1_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_2 ( f_3_0_2,  f_3_0_3, f_2_0_2, f_2_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_1 ( f_4_0_1,  f_4_0_2,  f_3_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_1 ( f_3_0_1,  f_3_0_2,  f_2_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            2
    f_4_2_t f_4_2_0 ( f_4_1_0,  f_4_1_1, f_4_0_0, f_4_0_1, CDtemp, ABcom, f_3_1_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            4  J=           2
    LOCSTORE(store, 20,  4, STOREDIM, STOREDIM) += f_4_2_0.x_20_4 ;
    LOCSTORE(store, 20,  5, STOREDIM, STOREDIM) += f_4_2_0.x_20_5 ;
    LOCSTORE(store, 20,  6, STOREDIM, STOREDIM) += f_4_2_0.x_20_6 ;
    LOCSTORE(store, 20,  7, STOREDIM, STOREDIM) += f_4_2_0.x_20_7 ;
    LOCSTORE(store, 20,  8, STOREDIM, STOREDIM) += f_4_2_0.x_20_8 ;
    LOCSTORE(store, 20,  9, STOREDIM, STOREDIM) += f_4_2_0.x_20_9 ;
    LOCSTORE(store, 21,  4, STOREDIM, STOREDIM) += f_4_2_0.x_21_4 ;
    LOCSTORE(store, 21,  5, STOREDIM, STOREDIM) += f_4_2_0.x_21_5 ;
    LOCSTORE(store, 21,  6, STOREDIM, STOREDIM) += f_4_2_0.x_21_6 ;
    LOCSTORE(store, 21,  7, STOREDIM, STOREDIM) += f_4_2_0.x_21_7 ;
    LOCSTORE(store, 21,  8, STOREDIM, STOREDIM) += f_4_2_0.x_21_8 ;
    LOCSTORE(store, 21,  9, STOREDIM, STOREDIM) += f_4_2_0.x_21_9 ;
    LOCSTORE(store, 22,  4, STOREDIM, STOREDIM) += f_4_2_0.x_22_4 ;
    LOCSTORE(store, 22,  5, STOREDIM, STOREDIM) += f_4_2_0.x_22_5 ;
    LOCSTORE(store, 22,  6, STOREDIM, STOREDIM) += f_4_2_0.x_22_6 ;
    LOCSTORE(store, 22,  7, STOREDIM, STOREDIM) += f_4_2_0.x_22_7 ;
    LOCSTORE(store, 22,  8, STOREDIM, STOREDIM) += f_4_2_0.x_22_8 ;
    LOCSTORE(store, 22,  9, STOREDIM, STOREDIM) += f_4_2_0.x_22_9 ;
    LOCSTORE(store, 23,  4, STOREDIM, STOREDIM) += f_4_2_0.x_23_4 ;
    LOCSTORE(store, 23,  5, STOREDIM, STOREDIM) += f_4_2_0.x_23_5 ;
    LOCSTORE(store, 23,  6, STOREDIM, STOREDIM) += f_4_2_0.x_23_6 ;
    LOCSTORE(store, 23,  7, STOREDIM, STOREDIM) += f_4_2_0.x_23_7 ;
    LOCSTORE(store, 23,  8, STOREDIM, STOREDIM) += f_4_2_0.x_23_8 ;
    LOCSTORE(store, 23,  9, STOREDIM, STOREDIM) += f_4_2_0.x_23_9 ;
    LOCSTORE(store, 24,  4, STOREDIM, STOREDIM) += f_4_2_0.x_24_4 ;
    LOCSTORE(store, 24,  5, STOREDIM, STOREDIM) += f_4_2_0.x_24_5 ;
    LOCSTORE(store, 24,  6, STOREDIM, STOREDIM) += f_4_2_0.x_24_6 ;
    LOCSTORE(store, 24,  7, STOREDIM, STOREDIM) += f_4_2_0.x_24_7 ;
    LOCSTORE(store, 24,  8, STOREDIM, STOREDIM) += f_4_2_0.x_24_8 ;
    LOCSTORE(store, 24,  9, STOREDIM, STOREDIM) += f_4_2_0.x_24_9 ;
    LOCSTORE(store, 25,  4, STOREDIM, STOREDIM) += f_4_2_0.x_25_4 ;
    LOCSTORE(store, 25,  5, STOREDIM, STOREDIM) += f_4_2_0.x_25_5 ;
    LOCSTORE(store, 25,  6, STOREDIM, STOREDIM) += f_4_2_0.x_25_6 ;
    LOCSTORE(store, 25,  7, STOREDIM, STOREDIM) += f_4_2_0.x_25_7 ;
    LOCSTORE(store, 25,  8, STOREDIM, STOREDIM) += f_4_2_0.x_25_8 ;
    LOCSTORE(store, 25,  9, STOREDIM, STOREDIM) += f_4_2_0.x_25_9 ;
    LOCSTORE(store, 26,  4, STOREDIM, STOREDIM) += f_4_2_0.x_26_4 ;
    LOCSTORE(store, 26,  5, STOREDIM, STOREDIM) += f_4_2_0.x_26_5 ;
    LOCSTORE(store, 26,  6, STOREDIM, STOREDIM) += f_4_2_0.x_26_6 ;
    LOCSTORE(store, 26,  7, STOREDIM, STOREDIM) += f_4_2_0.x_26_7 ;
    LOCSTORE(store, 26,  8, STOREDIM, STOREDIM) += f_4_2_0.x_26_8 ;
    LOCSTORE(store, 26,  9, STOREDIM, STOREDIM) += f_4_2_0.x_26_9 ;
    LOCSTORE(store, 27,  4, STOREDIM, STOREDIM) += f_4_2_0.x_27_4 ;
    LOCSTORE(store, 27,  5, STOREDIM, STOREDIM) += f_4_2_0.x_27_5 ;
    LOCSTORE(store, 27,  6, STOREDIM, STOREDIM) += f_4_2_0.x_27_6 ;
    LOCSTORE(store, 27,  7, STOREDIM, STOREDIM) += f_4_2_0.x_27_7 ;
    LOCSTORE(store, 27,  8, STOREDIM, STOREDIM) += f_4_2_0.x_27_8 ;
    LOCSTORE(store, 27,  9, STOREDIM, STOREDIM) += f_4_2_0.x_27_9 ;
    LOCSTORE(store, 28,  4, STOREDIM, STOREDIM) += f_4_2_0.x_28_4 ;
    LOCSTORE(store, 28,  5, STOREDIM, STOREDIM) += f_4_2_0.x_28_5 ;
    LOCSTORE(store, 28,  6, STOREDIM, STOREDIM) += f_4_2_0.x_28_6 ;
    LOCSTORE(store, 28,  7, STOREDIM, STOREDIM) += f_4_2_0.x_28_7 ;
    LOCSTORE(store, 28,  8, STOREDIM, STOREDIM) += f_4_2_0.x_28_8 ;
    LOCSTORE(store, 28,  9, STOREDIM, STOREDIM) += f_4_2_0.x_28_9 ;
    LOCSTORE(store, 29,  4, STOREDIM, STOREDIM) += f_4_2_0.x_29_4 ;
    LOCSTORE(store, 29,  5, STOREDIM, STOREDIM) += f_4_2_0.x_29_5 ;
    LOCSTORE(store, 29,  6, STOREDIM, STOREDIM) += f_4_2_0.x_29_6 ;
    LOCSTORE(store, 29,  7, STOREDIM, STOREDIM) += f_4_2_0.x_29_7 ;
    LOCSTORE(store, 29,  8, STOREDIM, STOREDIM) += f_4_2_0.x_29_8 ;
    LOCSTORE(store, 29,  9, STOREDIM, STOREDIM) += f_4_2_0.x_29_9 ;
    LOCSTORE(store, 30,  4, STOREDIM, STOREDIM) += f_4_2_0.x_30_4 ;
    LOCSTORE(store, 30,  5, STOREDIM, STOREDIM) += f_4_2_0.x_30_5 ;
    LOCSTORE(store, 30,  6, STOREDIM, STOREDIM) += f_4_2_0.x_30_6 ;
    LOCSTORE(store, 30,  7, STOREDIM, STOREDIM) += f_4_2_0.x_30_7 ;
    LOCSTORE(store, 30,  8, STOREDIM, STOREDIM) += f_4_2_0.x_30_8 ;
    LOCSTORE(store, 30,  9, STOREDIM, STOREDIM) += f_4_2_0.x_30_9 ;
    LOCSTORE(store, 31,  4, STOREDIM, STOREDIM) += f_4_2_0.x_31_4 ;
    LOCSTORE(store, 31,  5, STOREDIM, STOREDIM) += f_4_2_0.x_31_5 ;
    LOCSTORE(store, 31,  6, STOREDIM, STOREDIM) += f_4_2_0.x_31_6 ;
    LOCSTORE(store, 31,  7, STOREDIM, STOREDIM) += f_4_2_0.x_31_7 ;
    LOCSTORE(store, 31,  8, STOREDIM, STOREDIM) += f_4_2_0.x_31_8 ;
    LOCSTORE(store, 31,  9, STOREDIM, STOREDIM) += f_4_2_0.x_31_9 ;
    LOCSTORE(store, 32,  4, STOREDIM, STOREDIM) += f_4_2_0.x_32_4 ;
    LOCSTORE(store, 32,  5, STOREDIM, STOREDIM) += f_4_2_0.x_32_5 ;
    LOCSTORE(store, 32,  6, STOREDIM, STOREDIM) += f_4_2_0.x_32_6 ;
    LOCSTORE(store, 32,  7, STOREDIM, STOREDIM) += f_4_2_0.x_32_7 ;
    LOCSTORE(store, 32,  8, STOREDIM, STOREDIM) += f_4_2_0.x_32_8 ;
    LOCSTORE(store, 32,  9, STOREDIM, STOREDIM) += f_4_2_0.x_32_9 ;
    LOCSTORE(store, 33,  4, STOREDIM, STOREDIM) += f_4_2_0.x_33_4 ;
    LOCSTORE(store, 33,  5, STOREDIM, STOREDIM) += f_4_2_0.x_33_5 ;
    LOCSTORE(store, 33,  6, STOREDIM, STOREDIM) += f_4_2_0.x_33_6 ;
    LOCSTORE(store, 33,  7, STOREDIM, STOREDIM) += f_4_2_0.x_33_7 ;
    LOCSTORE(store, 33,  8, STOREDIM, STOREDIM) += f_4_2_0.x_33_8 ;
    LOCSTORE(store, 33,  9, STOREDIM, STOREDIM) += f_4_2_0.x_33_9 ;
    LOCSTORE(store, 34,  4, STOREDIM, STOREDIM) += f_4_2_0.x_34_4 ;
    LOCSTORE(store, 34,  5, STOREDIM, STOREDIM) += f_4_2_0.x_34_5 ;
    LOCSTORE(store, 34,  6, STOREDIM, STOREDIM) += f_4_2_0.x_34_6 ;
    LOCSTORE(store, 34,  7, STOREDIM, STOREDIM) += f_4_2_0.x_34_7 ;
    LOCSTORE(store, 34,  8, STOREDIM, STOREDIM) += f_4_2_0.x_34_8 ;
    LOCSTORE(store, 34,  9, STOREDIM, STOREDIM) += f_4_2_0.x_34_9 ;
}
